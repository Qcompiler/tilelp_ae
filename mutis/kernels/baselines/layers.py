from typing import Optional, Union
from  tilelp.common.common import gen_quant4
import torch
from torch import nn
import triton
import triton.language as tl

import mutis
from hidet import bfloat16
from hidet.graph.frontend.torch.utils import dtype_to_torch
from hidet.ir.type import data_type
from hidet.ir.dtypes import float16, uint8, uint7b, uint6b, uint5b, uint4b, uint3b, uint2b, uint1b, float32
from hidet.ir.dtypes import int8, int7b, int6b, int5b, int4b, int3b, int2b, int1b, int32
from hidet.ir.dtypes import float8_e4m3, float6_e3m2, float6_e2m3, float5_e3m1, float5_e2m2, float4_e2m1, float3_e1m1
from hidet.ir.dtypes import float7_e3m3, float7_e4m2, float7_e2m4, float8_e5m2, float7_e5m1, float6_e4m1
from mutis.kernels.vm.matmul_mma import matmul_mma
from mutis.kernels.vm.matmul_mma_decode import matmul_mma_decode
from mutis.types import DataType
from mutis.utils import benchmark_func



from dsl_int4_gemv import (
    w4a16,
    w4a16_tilelp_tilelp_acc_in_rf,
    w4a16_tilelp_evict,

)

# ── INT8 quantisation helper ──────────────────────────────────────────────────
def gen_quant8_my(n, k, w, groupsize=-1):
    """Quantise weight matrix to uint8 (asymmetric, zero-point=128).
    Returns (q_weight [n, k] uint8, scales [n, 1] float16).
    """
    if groupsize == -1:
        groupsize = k
    maxq = 255  # uint8 range 0-255
    w_reshaped = w.reshape(n, -1, groupsize)  # (n, num_groups, groupsize)
    abs_max = torch.max(torch.abs(w_reshaped), dim=2, keepdim=True)[0]
    scales = abs_max / 127.0  # map [-abs_max, abs_max] -> [-127, 127], then shift by 128
    scales = scales.clamp(min=1e-8)
    q = torch.round(w_reshaped / scales).int() + 128  # shift to [1, 255] range
    q = torch.clamp(q, 0, 255).to(torch.uint8)
    q = q.reshape(n, k)
    scales = scales.reshape(n, -1).to(torch.float16)  # (n, num_groups)
    return q, scales


# ── INT2 quantisation helper ──────────────────────────────────────────────────
def gen_quant2_my(n, k, w, groupsize=-1, tile=1):
    """Quantise weight matrix to int2 (values in {-2,-1,0,1}), packed 16 per int32.
    Returns (q_weight [n, k//16] int32, scales [n, 1] float16).
    """
    if groupsize == -1:
        groupsize = k
    import numpy as np
    w_reshaped = w.reshape(n, -1, groupsize)  # (n, num_groups, groupsize)
    abs_max = torch.max(torch.abs(w_reshaped), dim=2, keepdim=True)[0]
    scales = abs_max / 1.0  # int2 range [-2,1], centre at -0.5; use abs_max as scale
    scales = scales.clamp(min=1e-8)
    # quantise to {-2,-1,0,1}: round(w/scale) clamped
    q = torch.round(w_reshaped / scales).int()
    q = torch.clamp(q, -2, 1)          # values in [-2, 1]
    q_unsigned = (q + 2).to(torch.int32)  # shift to [0,3] for 2-bit unsigned packing
    q_flat = q_unsigned.reshape(n, k)   # (n, k)
    # pad k to multiple of 16
    pad_k = (16 - (k % 16)) % 16
    if pad_k > 0:
        q_flat = torch.nn.functional.pad(q_flat, (0, pad_k), value=2)  # 2 -> 0 after -2 shift
    k_padded = k + pad_k
    # pack 16 int2 values into one int32 (bits 0-1=elem0, bits 2-3=elem1, ...)
    q_np = q_flat.cpu().numpy().astype(np.uint32)  # (n, k_padded)
    packed = np.zeros((n, k_padded // 16), dtype=np.uint32)
    for i in range(16):
        packed |= (q_np[:, i::16] & 3) << (2 * i)
    q_weight = torch.from_numpy(packed.astype(np.int32)).to(w.device)  # (n, k//16)
    scales = scales.reshape(n, -1).to(torch.float16)  # (n, num_groups)
    return q_weight, scales


# ── INT8 tilelp Triton kernel (test_dequant) ──────────────────────────────────
@triton.jit
def _tilelp_int8_dequanti(b):
    x1, x2, x3, x4 = tl.inline_asm_elementwise(
        asm="""
            {
            .reg .b32 	r<15>;
            .reg .b16 	rs<9>;
            .reg .f32 	f<13>;

            mov.u32 r7, $4;
            and.b32 r8, r7, 0xff;
            cvt.rn.f32.u32 f2, r8;
            sub.f32 f1, f2, 0f43000000;
            cvt.rn.f16.f32 rs1, f1;
            shr.u32 r9, r7, 8;
            and.b32 r10, r9, 0xff;
            cvt.rn.f32.u32 f5, r10;
            sub.f32 f4, f5, 0f43000000;
            cvt.rn.f16.f32 rs2, f4;
            shr.u32 r11, r7, 16;
            and.b32 r12, r11, 0xff;
            cvt.rn.f32.u32 f8, r12;
            sub.f32 f7, f8, 0f43000000;
            cvt.rn.f16.f32 rs3, f7;
            shr.u32 r13, r7, 24;
            and.b32 r14, r13, 0xff;
            cvt.rn.f32.u32 f11, r14;
            sub.f32 f10, f11, 0f43000000;
            cvt.rn.f16.f32 rs4, f10;
            mov.b32 r5, {rs1, rs2};
            mov.b32 r6, {rs3, rs4};
            mov.b32 $0, r5;
            mov.b32 $1, r6;
            mov.b32 $2, r5;
            mov.b32 $3, r6;
            }
        """,
        constraints=("=r,=r,=r,=r,r"),
        args=[b],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    return x1, x2, x3, x4


@triton.jit
def _tilelp_int8_sum_4_half(x1, x2, vec1, vec2):
    y = tl.inline_asm_elementwise(
        asm="""
            {
              .reg .b32 vec1, vec2, vec_sum;
              .reg .b16 h_low, h_high, h_final;
              .reg .b32 x1, x2;
              mov.b32 vec1, $1;
              mov.b32 vec2, $2;
              mov.b32 x1, $3;
              mov.b32 x2, $4;
              mul.f16x2 vec1, vec1, x1;
              mul.f16x2 vec2, vec2, x2;
              add.f16x2 vec_sum, vec1, vec2;
              mov.b32 {h_high, h_low}, vec_sum;
              add.f16 h_final, h_high, h_low;
              cvt.u16.u16 $0, h_final;
            }
        """,
        constraints=("=f,r,r,r,r"),
        args=[x1, x2, vec1, vec2],
        dtype=(tl.float16),
        is_pure=False,
        pack=1,
    )
    return y


@triton.jit
def _tilelp_int8_load_v4_b32(ptr):
    return tl.inline_asm_elementwise(
        asm="ld.global.nc.v4.u32 {$0,$1,$2,$3}, [$4];",
        constraints=("=r,=r,=r,=r,l"),
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )


def get_autotune_config():
    configs = []
    for evict in ['evict_last', 'evict_first', None]:
        for evict_scales in ['evict_last', 'evict_first', None]:
            configs.append(triton.Config({ 'evict' : evict, 'evict_scales' : evict_scales}))

    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key = ['m', 'k'],
    use_cuda_graph = False
)

@triton.jit
def _tilelp_int8_dequant_kernel(
    A_ptr, x_ptr, y_ptr,
    scales_ptr,
    m, k, int8_k,
    stride_am,
    BLOCK_SIZE: tl.constexpr = 256,
    evict: tl.constexpr = 'evict_first',
    evict_scales: tl.constexpr = None,
):
    row_id = tl.program_id(0)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    offs_k = tl.arange(0, BLOCK_SIZE)

    A_ptr = A_ptr + row_id * stride_am + offs_k
    x_ptr = x_ptr + offs_k

    for _ in range(0, tl.cdiv(int8_k, BLOCK_SIZE)):
        mask_even = (offs_k < int8_k) & ((offs_k % 2) == 0)
        x_ptr_even = x_ptr
        x1, x2, x3, x4 = _tilelp_int8_load_v4_b32(x_ptr_even)
        a_even = tl.where(mask_even, tl.load(A_ptr, eviction_policy=evict), 0)
        a_even_1, a_even_2, a_even_3, a_even_4 = _tilelp_int8_dequanti(a_even)
        acc = acc + tl.where(mask_even, _tilelp_int8_sum_4_half(a_even_1, a_even_2, x1, x2), 0.0)

        mask_odd = (offs_k < int8_k) & ((offs_k % 2) == 1)
        x_ptr_odd = x_ptr - 1
        x1_odd, x2_odd, x3_odd, x4_odd = _tilelp_int8_load_v4_b32(x_ptr_odd)
        a_odd = tl.where(mask_odd, tl.load(A_ptr, eviction_policy=evict), 0)
        a_odd_1, a_odd_2, a_odd_3, a_odd_4 = _tilelp_int8_dequanti(a_odd)
        acc = acc + tl.where(mask_odd, _tilelp_int8_sum_4_half(a_odd_1, a_odd_2, x3_odd, x4_odd), 0.0)

        offs_k += BLOCK_SIZE
        A_ptr += BLOCK_SIZE
        x_ptr += BLOCK_SIZE

    acc = tl.sum(acc, axis=0)
    scales = tl.load(scales_ptr + row_id, eviction_policy=evict_scales)
    acc = acc * scales
    tl.store(y_ptr + row_id, acc)


def _tilelp_int8_test_dequant(A: torch.Tensor, vector: torch.Tensor, output, scales):
    """Wrapper matching test_dequant from test_gemv_int8_triton_inline_ptx.py."""
    n, _ = A.shape
    k = vector.shape[1]  # number of fp16 elements
    device = A.device
    stride_ak, _ = A.stride()
    assert vector.shape[1] == A.shape[1] * 4, "Vector and weight shape mismatch for int8 tilelp"
    storage = vector.untyped_storage()
    uint64_tensor = torch.tensor([], dtype=torch.uint64, device=device).set_(storage, 0, (k // 4,))
    int8_k = int(A.shape[1])
    grid = lambda meta: (n, 1)
    _tilelp_int8_dequant_kernel[grid](A, uint64_tensor, output, scales, n, k, int8_k, stride_ak)


# ── INT2 tilelp Triton kernel (test_dequant_int2) ─────────────────────────────
@triton.jit
def _tilelp_int2_dequanti(b):
    """Unpack one int32 containing 16 int2 values into 8 uint32s (each holding 2 fp16)."""
    x1, x2, x3, x4, x5, x6, x7, x8 = tl.inline_asm_elementwise(
        asm="""
            {
            .reg .b32 r<74>;
            .reg .b64 rd<3>;
            
    
            mov.u32 r0, $8;
            
          
            shr.u32 r1, r0, 4;
            
    
            mov.u32 r2, 16448;      
            mov.u32 r3, 16705;     
            mov.u32 r4, 16962;    
            mov.u32 r5, 17219;     
            
   
            { prmt.b32 r6, r0, 0, r2; }   
            { prmt.b32 r7, r1, 0, r2; } 
            { prmt.b32 r8, r0, 0, r3; }   
            { prmt.b32 r9, r1, 0, r3; }  
            { prmt.b32 r10, r0, 0, r4; }   
            { prmt.b32 r11, r1, 0, r4; }  
            { prmt.b32 r12, r0, 0, r5; } 
            { prmt.b32 r13, r1, 0, r5; }  
            
        
            { lop3.b32 r6, r6, 786435, 1677747200, 106; }
            { lop3.b32 r7, r7, 786435, 1677747200, 106; }
            { lop3.b32 r8, r8, 786435, 1677747200, 106; }
            { lop3.b32 r9, r9, 786435, 1677747200, 106; }
            { lop3.b32 r10, r10, 786435, 1677747200, 106; }
            { lop3.b32 r11, r11, 786435, 1677747200, 106; }
            { lop3.b32 r12, r12, 786435, 1677747200, 106; }
            { lop3.b32 r13, r13, 786435, 1677747200, 106; }
            
           
            mov.u32 r14, 872430592;   
            mov.u32 r15, -603921408;   
            
      
            { fma.rn.f16x2 r16, r6, r14, r15; }  
            { fma.rn.f16x2 r17, r7, r14, r15; }  
            { fma.rn.f16x2 r18, r8, r14, r15; }  
            { fma.rn.f16x2 r19, r9, r14, r15; }  
            { fma.rn.f16x2 r20, r10, r14, r15; }  
            { fma.rn.f16x2 r21, r11, r14, r15; }     
            { fma.rn.f16x2 r22, r12, r14, r15; }  
            { fma.rn.f16x2 r23, r13, r14, r15; }  
            
      
            mov.u32 $0, r16;
            mov.u32 $1, r17;
            mov.u32 $2, r18;
            mov.u32 $3, r19;
            mov.u32 $4, r20;
            mov.u32 $5, r21;
            mov.u32 $6, r22;
            mov.u32 $7, r23;
            }
        """,
        constraints=("=r,=r,=r,=r,=r,=r,=r,=r,r"),
        args=[b],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32, 
               tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    return x1, x2, x3, x4, x5, x6, x7, x8

@triton.jit
def _tilelp_int2_sum_8_half(a1, a2, a3, a4, x1, x2, x3, x4):
    """Compute a1*x1 + a2*x2 + a3*x3 + a4*x4 where each is a half2 packed in int32."""
    y = tl.inline_asm_elementwise(
        asm="""
            {
              .reg .b32 v1, v2, v3, v4, v12, v34, v1234;
              .reg .b16 h_low, h_high, h_final;
              mov.b32 v1, $1;
              mov.b32 v2, $2;
              mov.b32 v3, $3;
              mov.b32 v4, $4;
              mul.f16x2 v1, v1, $5;
              mul.f16x2 v2, v2, $6;
              mul.f16x2 v3, v3, $7;
              mul.f16x2 v4, v4, $8;
              add.f16x2 v12, v1, v2;
              add.f16x2 v34, v3, v4;
              add.f16x2 v1234, v12, v34;
              mov.b32 {h_high, h_low}, v1234;
              add.f16 h_final, h_high, h_low;
              cvt.u16.u16 $0, h_final;
            }
        """,
        constraints=("=f,r,r,r,r,r,r,r,r"),
        args=[a1, a2, a3, a4, x1, x2, x3, x4],
        dtype=tl.float16,
        is_pure=False,
        pack=1,
    )
    return y


@triton.jit
def _tilelp_int2_load_v8_b32(ptr):
    """Load 8 uint32s (32 bytes = 16 fp16) from global memory."""
    return tl.inline_asm_elementwise(
        asm="""
            {
            .reg .u64 ptr64;
            mov.u64 ptr64, $8;
            ld.global.nc.v4.u32 {$0,$1,$2,$3}, [ptr64];
            add.u64 ptr64, ptr64, 16;
            ld.global.nc.v4.u32 {$4,$5,$6,$7}, [ptr64];
            }
        """,
        constraints=("=r,=r,=r,=r,=r,=r,=r,=r,l"),
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32, tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )


def _tilelp_int2_autotune_config():
    configs = []
    for evict in ['evict_last', 'evict_first', None]:
        for evict_scales in ['evict_last', 'evict_first', None]:
            for BLOCK_SIZE in [64, 128, 256, 512, 1024]:
                configs.append(triton.Config({'evict': evict,
                                               'evict_scales': evict_scales, 
                                               'BLOCK_SIZE': BLOCK_SIZE}))
    return configs


@triton.autotune(
    configs=_tilelp_int2_autotune_config(),
    key=['m', 'int2_k'],
    use_cuda_graph=False,
)
@triton.jit
def _tilelp_int2_dequant_kernel(
    A_ptr, x_ptr, y_ptr,
    scales_ptr,
    m, k, int2_k,
    stride_am,
    BLOCK_SIZE: tl.constexpr,
    evict: tl.constexpr,
    evict_scales: tl.constexpr ,
):
    row_id = tl.program_id(0)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    offs_k = tl.arange(0, BLOCK_SIZE)

    A_ptr = A_ptr + row_id * stride_am + offs_k
    x_ptr = x_ptr + offs_k * 2

    for _ in range(0, tl.cdiv(int2_k, BLOCK_SIZE)):
        scales = tl.load(scales_ptr + row_id, eviction_policy=evict_scales)
        x1, x2, x3, x4, x5, x6, x7, x8 = _tilelp_int2_load_v8_b32(x_ptr)
        a = tl.load(A_ptr, eviction_policy=evict)
        a1, a2, a3, a4, a5, a6, a7, a8 = _tilelp_int2_dequanti(a)
        acc = acc + _tilelp_int2_sum_8_half(a1, a2, a3, a4, x1, x2, x3, x4) * scales
        acc = acc + _tilelp_int2_sum_8_half(a5, a6, a7, a8, x5, x6, x7, x8) * scales

        offs_k += BLOCK_SIZE
        A_ptr += BLOCK_SIZE
        x_ptr += BLOCK_SIZE * 2

    acc = tl.sum(acc, axis=0)

    tl.store(y_ptr + row_id, acc)


def _tilelp_int2_test_dequant(A: torch.Tensor, vector: torch.Tensor, output, scales):
    """Wrapper matching test_dequant from test_gemv_int2_triton_inline_ptx.py.
    A: (n, k//16) int32  — each int32 packs 16 int2 values
    vector: (1, k) fp16  — k == A.shape[1] * 16
    """
    n, _ = A.shape
    k = vector.shape[1]  # number of fp16 elements
    device = A.device
    stride_ak, _ = A.stride()
    assert vector.shape[1] == A.shape[1] * 16, "Vector and weight shape mismatch for int2 tilelp"
    storage = vector.untyped_storage()
    uint64_tensor = torch.tensor([], dtype=torch.uint64, device=device).set_(storage, 0, (k // 4,))
    int2_k = int(A.shape[1])
    grid = lambda meta: (n, 1)
    _tilelp_int2_dequant_kernel[grid](A, uint64_tensor, output, scales, n, k, int2_k, stride_ak)
# ─────────────────────────────────────────────────────────────────────────────

# Import jitcu for cutlass
import sys
sys.path.insert(0, 'tilelp/jitcu')
from jitcu import load_cuda_ops
    
class NotSupportedError(Exception):
    pass


class MatmulLayer(nn.Module):
    matmul_id = 0

    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__()
        self.a_dtype: DataType = a_dtype
        self.b_dtype: DataType = b_dtype
        self.group_size: int = group_size
        self.m: int = m
        self.k: int = k
        self.n: int = n

        if (a_dtype, b_dtype) not in self.supported_pairs():
            raise NotSupportedError(a_dtype, b_dtype)

        self._a: Optional[torch.Tensor] = None

    @staticmethod
    def get_cls(runner_name):
        return {
            'torch-f16': TorchF16Layer,
            'triton': TritonLayer,
            'bitblas': BitblasLayerV1,
            'quant-llm': QuantLLMLayer,
            'marlin': MarlinLayer,
            'mutis': MutisLayer,
            'tilelp' : TilelpLayer,
            'tilelp_evict': TilelpEvictLayer,
            'tilelp_acc_in_register': TilelpAccInRegisterLayer,
            'tilelp_no_acc_in_register_no_opt_micro_kernel': TilelpNoAccInRegisterNoOptMicroKernelLayer,
            'cutlass': CutlassLayer,
            'gemlite': GemLiteLayer,
        }[runner_name]

    @staticmethod
    def create(runner_name: str, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        show_memory = True and False
        prev = torch.cuda.memory_allocated()
        layer = MatmulLayer.get_cls(runner_name)(a_dtype, b_dtype, group_size, m, k, n)
        after = torch.cuda.memory_allocated()
        if show_memory:
            print(
                f"[{MatmulLayer.matmul_id // 4}][{MatmulLayer.matmul_id % 4}] Allocated {(after - prev) / 1024 / 1024 / 1024:.2f} GiB layer {k}x{n} with g{group_size}"
            )
            print(f'    current used: {after / 1024 / 1024 / 1024:.2f} GiB')
        MatmulLayer.matmul_id += 1
        return layer

    @staticmethod
    def supports(runner_name, a_dtype: Union[DataType, str], b_dtype: Union[DataType, str]) -> bool:
        a_dtype = data_type(a_dtype)
        b_dtype = data_type(b_dtype)
        return (a_dtype, b_dtype) in MatmulLayer.get_cls(runner_name).supported_pairs()

    @property
    def a(self):
        if self._a is None:
            self._a = torch.empty(self.m, self.k, dtype=mutis.dtype_to_torch(self.a_dtype), device='cuda')
        return self._a

    @staticmethod
    def supported_pairs():
        raise NotImplementedError()

    def run(self, a: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    def bench(self, warmup=None, repeat=None) -> float:
        return benchmark_func(
            run_func=lambda: self.run(),
            warmup=10 if warmup is None else warmup,
            repeat=50 if repeat is None else repeat,
            maximum_repeat_time=None,
            clear_l2_cache=True,
        )


class TorchF16Layer(MatmulLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        self.b = torch.randn(k, n, dtype=torch.float16, device='cuda')
        # self.b = torch.randn(n, k, dtype=torch.float16, device='cuda')
    @staticmethod
    def supported_pairs():
        return [(float16, float16)]

    def run(self, a: Optional[torch.Tensor] = None):
        # return torch.mm(a if a is not None else self.a, self.b.T) # 这个比较快，但是mutis的作者使用的是慢的，我们睁一只眼闭一只眼
        return torch.matmul(self.a if a is None else a, self.b)


class TritonLayer(MatmulLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        if b_dtype.nbits == 4:
            self.b = torch.randint(0, 1, size=[k // 8, n], dtype=torch.int32, device='cuda')
        else:
            self.b = torch.randint(0, max(int(b_dtype.max_value) // 2, 1), size=[k, n], dtype=torch.int8, device='cuda')
        self.zeros = torch.randint(0, 1, [k // group_size, n // 8], dtype=torch.int32, device='cuda')
        self.scales = torch.rand([k // group_size, n], dtype=mutis.dtype_to_torch(a_dtype), device='cuda')

    @staticmethod
    def supported_pairs():
        return [(float16, uint8), (float16, uint4b)]

    def triton_quantized_gemm(self, a, w, scales, zeros, group_size, b_dtype: DataType):
        from mutis.kernels.triton_kernels import triton_matmul_w8a16, triton_gemm_w4a16

        if b_dtype.nbits == 4:
            return triton_gemm_w4a16(groupsize=group_size, a=a, qweight=w, scales=scales, qzeros=zeros)
        elif b_dtype.nbits == 8:
            return triton_matmul_w8a16(a=a, b=w, scale=scales)
        else:
            raise NotImplementedError()

    def run(self, a: Optional[torch.Tensor] = None):
        a = self.a if a is None else a
        if self.m != a.size(0):
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)
        else:
            return self.triton_quantized_gemm(
                a, self.b, self.scales, self.zeros, group_size=self.group_size, b_dtype=self.b_dtype
            )


class BitblasBaseLayer(MatmulLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        import bitblas

        bitblas.set_log_level("ERROR")

    @staticmethod
    def supported_pairs():
        return [
            (mutis.float16, mutis.uint8),
            (mutis.float16, mutis.uint4b),
            (mutis.float16, mutis.uint2b),
            (mutis.float16, mutis.uint1b),
            (mutis.float16, mutis.int8),
            (mutis.float16, mutis.int4b),
            (mutis.float16, mutis.int2b),
            (mutis.float16, mutis.int1b),
            # (mutis.float16, mutis.float4_e2m1)
        ]

    def dtype_to_bitblas(self, dtype: DataType) -> str:
        if dtype.is_integer_subbyte():
            return dtype.name.removesuffix('b')
        if dtype == mutis.float4_e2m1:
            return 'fp4_e2m1'
        else:
            return dtype.name


class BitblasLayerV1(BitblasBaseLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)

        import bitblas.cache

        # bitblas.set_log_level("Debug")

        acc_dtype = 'float32'
        self.layer = bitblas.Linear(
            in_features=k,
            out_features=n,
            bias=False,
            A_dtype=self.dtype_to_bitblas(a_dtype),
            W_dtype=self.dtype_to_bitblas(b_dtype),
            accum_dtype=acc_dtype,
            out_dtype=self.dtype_to_bitblas(a_dtype),
            group_size=group_size,
            with_scaling=True,
            with_zeros=True if b_dtype.is_unsigned_integer() else False,
            zeros_mode='original',
            opt_M=[m],
        ).cuda()

    def run(self, a: Optional[torch.Tensor] = None):
        a = self.a if a is None else a
        if self.m != a.size(0):
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)
        else:
            return self.layer(a)


class BitblasLayerV2(BitblasBaseLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        from bitblas import MatmulConfig, Matmul

        config = MatmulConfig(
            M=self.m,
            N=self.n,
            K=self.k,
            A_dtype=self.dtype_to_bitblas(a_dtype),
            W_dtype=self.dtype_to_bitblas(b_dtype),
            out_dtype=self.dtype_to_bitblas(a_dtype),
            group_size=group_size,
            accum_dtype='float32',
            with_scaling=True,
            with_zeros=True,
            zeros_mode='original',
            storage_dtype="uint32" if b_dtype == float4_e2m1 else "int8",
        )
        self.op = Matmul(config)
        self.b = self.op.transform_weight(torch.randint(-2, 2, [n, k], dtype=torch.int8, device='cuda'))
        self.scales = torch.rand([k // group_size, n], dtype=dtype_to_torch(a_dtype), device='cuda')
        self.zeros = torch.rand([k // group_size, n], dtype=dtype_to_torch(a_dtype), device='cuda')

    def run(self, a: Optional[torch.Tensor] = None):
        a = self.a if a is None else a
        if self.m != a.size(0):
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)
        else:
            return self.op(a, self.b, self.scales, self.zeros)


class QuantLLMLayer(MatmulLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        import fp6_llm

        self.fp6_packed_weight = torch.empty(n, k // 16 * 3, dtype=torch.int32, device='cuda')
        self.fp16_scale = torch.randn(n, dtype=torch.float16, device='cuda')
        Number_GPU_SMs = torch.cuda.get_device_properties(0).multi_processor_count
        self.splitK = fp6_llm.HeuristicFuntion_SplitK(m, n, Number_GPU_SMs)

        assert a_dtype.is_float() and a_dtype.nbits == 16 and b_dtype.is_float() and b_dtype.nbits == 6

    @staticmethod
    def supported_pairs():
        return [(float16, float6_e3m2)]

    def run(self, a: Optional[torch.Tensor] = None):
        import fp6_llm

        a = self.a if a is None else a
        return fp6_llm.linear_forward_cuda(a, self.fp6_packed_weight, self.fp16_scale, self.splitK)


class MutisLayer(MatmulLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        mutis.set_benchmark_mode(True)
        group_size = k if group_size == -1 else group_size
        self.b = mutis.randn([k, n], dtype=b_dtype).storage
        self.scales = mutis.randn([k // group_size, n], dtype=self.get_scale_dtype()).torch()
        self.zeros = mutis.randn([k // group_size, n], dtype=self.get_scale_dtype()).torch()

    def get_mma_operand_dtype(self):
        if self.a_dtype in [float16, bfloat16]:
            return self.a_dtype
        elif self.a_dtype == int8 and self.b_dtype.is_signed_integer():
            return int8
        else:
            return float16

    def get_accumulate_dtype(self):
        return float32
        # if self.a_dtype in [float16, bfloat16]:
        #     return self.a_dtype
        # elif self.a_dtype == int8 and self.b_dtype.is_signed_integer():
        #     return float32
        # else:
        #     return float16

    def get_scale_dtype(self):
        if self.a_dtype in [float16, bfloat16]:
            return self.a_dtype
        else:
            return float16

    def get_zeros_dtype(self):
        if self.a_dtype in [float16, bfloat16]:
            return self.a_dtype
        else:
            return float16

    def get_c_dtype(self):
        if self.a_dtype in [float16, bfloat16]:
            return self.a_dtype
        else:
            return float16

    @staticmethod
    def supported_pairs():
        pairs = []
        for a_dtype in [float16, bfloat16, int8, uint8]:
            for b_dtype in [
                uint8,
                uint7b,
                uint6b,
                uint5b,
                uint4b,
                uint3b,
                uint2b,
                uint1b,
                int8,
                int7b,
                int6b,
                int5b,
                int4b,
                int3b,
                int2b,
                int1b,
                float8_e5m2,
                float8_e4m3,
                float7_e5m1,
                float7_e4m2,
                float7_e3m3,
                float7_e2m4,
                float6_e4m1,
                float6_e3m2,
                float6_e2m3,
                float5_e3m1,
                float5_e2m2,
                float4_e2m1,
                float3_e1m1,
            ]:
                pairs.append((a_dtype, b_dtype))
        return pairs

    def run(self, a: Optional[torch.Tensor] = None):
        a = self.a if a is None else a
        if a.size(0) != self.m:
            # skip other batch sizes, which is used to calculate the memory consumption by vllm
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)
        else:
            if self.m > 256:
                b = matmul_mma_decode(k=self.k, n=self.n, dtype=self.b_dtype, output_dtype=self.a_dtype, x=self.b)
                return torch.matmul(a, b)
            else:
                return matmul_mma(
                    m=self.m,
                    n=self.n,
                    k=self.k,
                    group_size=self.group_size,
                    a=a,
                    b=self.b,
                    scales=self.scales,
                    zeros=self.zeros,
                    a_dtype=self.a_dtype,
                    b_dtype=self.b_dtype,
                    c_dtype=self.a_dtype,
                    use_dynamic_m=False,
                )

import marlin
class MarlinLayer(MatmulLayer):
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        import marlin

        self.linear = nn.Linear(in_features=k, out_features=n, bias=False).cuda().half()
        # self.marlin_layer = marlin.Layer(infeatures=k, outfeatures=n, groupsize=group_size).cuda()
        device = self.linear.weight.device
        dtype = self.linear.weight.dtype
        self.n = n
        self.workspace = torch.zeros(n // 128 * 16, device=device)
        
        
        _, self.B, self.s = gen_quant4(k, n, self.linear.weight.t().contiguous(),   groupsize=128)  
        assert a_dtype == mutis.float16
        # assert b_dtype in [mutis.int4b]

    @staticmethod
    def supported_pairs():
        return [(float16, int4b), (float16, uint4b)]

    def run(self, a: Optional[torch.Tensor] = None):
        vector = self.a if a is None else a
        thread_k = 64
        thread_n = 256
        m = vector.shape[0]

        C_i4mar = torch.zeros((m, self.n), dtype=vector.dtype, device=vector.device)
        marlin.mul(vector, self.B, C_i4mar, self.s, self.workspace, thread_k, thread_n, -1)
        return C_i4mar




import triton
import triton.language as tl



@triton.jit
def dequanti_tensorRT_llm(b):
  #x1 int32
  # 一共四个
  # 总共16字节
  # 8个half
    x1, x2, x3, x4 = tl.inline_asm_elementwise(
        asm="""
            {
            .reg .b32 	r<23>;
            .reg .f32 	f<5>;
   

            mov.u32  r2, $4;
            shr.u32  r8, r2, 8;
            lop3.b32 r1, r2, 983055, 1677747200, 234;
            lop3.b32 r3, r2, 15728880, 1677747200, 234;

            lop3.b32 r5, r8, 983055, 1677747200, 234;
            lop3.b32 r7, r8, 15728880, 1677747200, 234;

            mov.u32 	r18, 1678271496;
            sub.f16x2 r9, r1, r18;
            mov.u32 	r21, 738208768;
            mov.u32 	r22, -729754496;
            fma.rn.f16x2 r12, r3, r21, r22;
            sub.f16x2  r16,  r5,  r18;
            fma.rn.f16x2 r19, r7, r21, r22;

            mov.b32 	f1, r19;
            mov.b32 	f2, r12;
            mov.b32 	f3, r16;
            mov.b32 	f4, r9;

            mov.b32   $3, f1; 
            mov.b32   $2, f3; 
            mov.b32   $1, f2;   
            mov.b32   $0, f4;  
            }
        """,
        constraints=(
            "=r,=r,=r,=r,r"
        ),
        args=[b], #输入
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32), #输出
        is_pure=False,
        pack=1,
    )
    return x1, x2, x3, x4




@triton.jit
def sum_4_half(x1, x2, vec1, vec2):
    # x1 : int32 x2: int32
    # vec1 : int32 vec2: int32
    # x1 * vec1 + x2 * vec2
    y = tl.inline_asm_elementwise(
        asm="""
            {
              .reg .b32 vec1, vec2, vec_sum;
              .reg .b16 h_low, h_high, h_final;
              .reg .b32 x1, x2;
              mov.b32 vec1, $1;
              mov.b32 vec2, $2;
              mov.b32 x1, $3;
              mov.b32 x2, $4;
              mul.f16x2 vec1, vec1, x1;
              mul.f16x2 vec2, vec2, x2;
              add.f16x2 vec_sum, vec1, vec2;
              mov.b32 {h_high, h_low}, vec_sum; 
              add.f16 h_final, h_high, h_low;              
              cvt.u16.u16 $0, h_final;
               
            }
        """,
        constraints=(
            "=f,r,r,r,r"
        ),
        args=[x1, x2, vec1, vec2],  # 参数 1 2 3 4
        dtype=(tl.float16), #参数 0
        is_pure=False,
        pack=1,
    )

    return y


@triton.jit
def load_v4_b32(ptr):
    return tl.inline_asm_elementwise(
        asm="ld.global.nc.v4.u32 {$0,$1,$2,$3}, [$4];",
        constraints=("=r,=r,=r,=r,l"),
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1
    )



def get_autotune_config():
    configs = []
    for evict in ['evict_last', 'evict_first', None]:
        for evict_scales in ['evict_last', 'evict_first', None]:
            for BLOCK_SIZE in [64,128, 256, 512, 1024]:    
                configs.append(triton.Config({ 'evict' : evict, 'evict_scales' : evict_scales, 
                                              'BLOCK_SIZE' : BLOCK_SIZE }))

    return configs


@triton.autotune(
    configs=get_autotune_config(),
    key = ['m', 'int4_k'],
    use_cuda_graph = False
)

@triton.jit
def w4a16_micro_kernel(
    A_ptr, x_ptr, y_ptr,
    scales_ptr,
    m, int4_k,
    stride_am, 
    BLOCK_SIZE : tl.constexpr ,
    evict : tl.constexpr ,
    evict_scales : tl.constexpr 
):



    row_id = tl.program_id(0)
 
    acc =   tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    offs_k =  tl.arange(0, BLOCK_SIZE)
 

    A_ptr = A_ptr + row_id * stride_am + (offs_k * 2)
    x_ptr = x_ptr + (offs_k * 4)
    
    for _ in range(0, tl.cdiv(int4_k // 2, BLOCK_SIZE)):
       
        x1, x2, x3, x4 = load_v4_b32(x_ptr)
        x5, x6, x7, x8 = load_v4_b32(x_ptr + 2)
        
        a = tl.load(A_ptr,  eviction_policy = evict)
        a_ = tl.load(A_ptr + 1, eviction_policy = evict)
        scales = tl.load(scales_ptr + row_id, eviction_policy = evict_scales)
        a1, a2, a3, a4 = dequanti_tensorRT_llm(a) 
        a5, a6, a7, a8 = dequanti_tensorRT_llm(a_)
        
        acc += (sum_4_half(a1, a2, x1, x2) + sum_4_half(a3, a4, x3, x4) + \
         sum_4_half(a5, a6, x5, x6) + sum_4_half(a7, a8, x7, x8)) * scales
 
  

        A_ptr += (BLOCK_SIZE) * 2
        x_ptr += (BLOCK_SIZE * 2) * 2

    acc = tl.sum(acc, axis=0) 
    

    tl.store(y_ptr + row_id, acc)

# === 生成的 w4a16_micro ===
def w4a16_micro(A: torch.Tensor, vector: torch.Tensor, output, scales):
    row, int4_k = A.shape
    k = vector.shape[1] # [m, k ]
    device = A.device
    stride_ak, stride_an = A.stride()


    # assert k == A.shape[1] * 8, "Vector and input tensor shape mismatch"
    # assert A.device == device and vector.device == device and output.device == device, "Tensors must be on CUDA"
    # grid = lambda meta: (row, 1)
    
    storage = vector.untyped_storage()

    uint64_tensor = torch.tensor([], dtype=torch.uint64,device=device).set_(storage, 0, (k // 4,))
 
    
    kernel = w4a16_micro_kernel[(row, 1)](A, uint64_tensor, output, scales, row, int4_k,  
                            stride_ak)
    
    
    return kernel


from tilelpbatch import gemm_splitK_forward
class TilelpLayer(MatmulLayer):
    """W4A16 / W8A16 GEMV via tilelp DSL kernels."""

    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        weight = torch.randn(n, k, dtype=torch.float16, device='cuda')

        if b_dtype == uint8:
            # INT8 path: quantise to uint8, pack 4 bytes into int32 for the kernel
            self.q_weight_uint8, self.scales = gen_quant8_my(n, k, torch.clone(weight), groupsize=-1)
            scales_1d = self.scales[:, 0] if self.scales.shape[1] == 1 else self.scales[:, 0]
            self.scales_1d = scales_1d.to(torch.float16)
            # Pack uint8 -> int32 (4 bytes per int32); pad if k % 4 != 0
            if k % 4 != 0:
                pad_k = 4 - (k % 4)
                q_padded = torch.nn.functional.pad(self.q_weight_uint8, (0, pad_k), value=128)
            else:
                pad_k = 0
                q_padded = self.q_weight_uint8
            self.q_weight_int32 = q_padded.contiguous().view(torch.int32).reshape(n, (k + pad_k) // 4)
            self.pad_k = pad_k
        elif b_dtype == uint2b:
            # INT2 path: gen_quant2_my produces (n, k//16) int32 already packed
            self.q_weight_int2, self.scales = gen_quant2_my(n, k, torch.clone(weight), groupsize=-1)
            scales_1d = self.scales[:, 0] if self.scales.shape[1] == 1 else self.scales[:, 0]
            self.scales_1d = scales_1d.to(torch.float16)
            self.pad_k2 = (16 - (k % 16)) % 16
        else:
            # INT4 path (original)
            from tilelp.common.common import gen_quant4_my
            self.q_weight, self.scales = gen_quant4_my(n, k, torch.clone(weight),
                                                       groupsize=-1, tile=1)

            from tilelp.common.common import pack_gemlite
            from tilelp.common.common import generate_randint, gen_quant4, gen_quant4_my, gen_quant4_uint8
            from tilelp.common.common import pack_gemlite
            BLOCK_SIZE_K = 128
            q_weight, scales  = gen_quant4_uint8(n, k, torch.clone(weight),
                                          block_size = BLOCK_SIZE_K,
                                          groupsize = 128, tile = 1)



            self.W_q, self.W_scales, _ = pack_gemlite(
                q_weight, scales, None  ,
                W_nbits=4, group_size=group_size,
                in_features=k, out_features=n,
            )
    @staticmethod
    def supported_pairs():
        return [(float16, int4b), (float16, uint4b), (float16, uint8), (float16, uint2b)]

    def run(self, a: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = self.a if a is None else a
        if a.size(0) != self.m:
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)

        if self.b_dtype == uint8:
            # INT8 path: use test_dequant from test_gemv_int8_triton_inline_ptx.py
            out = torch.zeros(self.m, self.n, dtype=torch.float16, device=a.device)
            for i in range(self.m):
                if self.pad_k > 0:
                    vec = torch.nn.functional.pad(a[i:i+1], (0, self.pad_k), value=0.0)
                else:
                    vec = a[i:i+1]
                _tilelp_int8_test_dequant(self.q_weight_int32, vec, out[i], self.scales_1d)
            return out
        elif self.b_dtype == uint2b:
            # INT2 path: use test_dequant from test_gemv_int2_triton_inline_ptx.py
            out = torch.zeros(self.m, self.n, dtype=torch.float16, device=a.device)
            for i in range(self.m):
                if self.pad_k2 > 0:
                    vec = torch.nn.functional.pad(a[i:i+1], (0, self.pad_k2), value=0.0)
                else:
                    vec = a[i:i+1]
                _tilelp_int2_test_dequant(self.q_weight_int2, vec, out[i], self.scales_1d)
            return out
        else:
            # INT4 path
            
            if self.m == 1:
                out = torch.empty(self.m, self.n, dtype=torch.float16, device=a.device)
                w4a16(self.q_weight, a, out, self.scales)
            else:
                out, _ = gemm_splitK_forward(a, self.W_q, 
                                 self.W_scales, self.group_size, elements_per_sample=8) 

            return out


class TilelpEvictLayer(MatmulLayer):
    """W4A16 GEMV via tilelp DSL kernel without eviction hint."""

    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        self.q_weight = torch.randint(0, 16, (n, k // 8), dtype=torch.int32, device='cuda')
        self.scales = torch.randn(n, dtype=torch.float16, device='cuda')

    @staticmethod
    def supported_pairs():
        return [(float16, uint4b)]

    def run(self, a=None):
        a = self.a if a is None else a
        if a.size(0) != self.m:
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)
        out = torch.empty(self.m, self.n, dtype=torch.float16, device=a.device)
        for i in range(self.m):
            w4a16_tilelp_evict(self.q_weight, a[i : i + 1], out[i], self.scales)
        return out


class TilelpAccInRegisterLayer(MatmulLayer):
    """W4A16 GEMV via tilelp DSL kernel without accumulator in register."""

    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        self.q_weight = torch.randint(0, 16, (n, k // 8), dtype=torch.int32, device='cuda')
        self.scales = torch.randn(n, dtype=torch.float16, device='cuda')

    @staticmethod
    def supported_pairs():
        return [(float16, uint4b)]

    def run(self, a=None):
        a = self.a if a is None else a
        if a.size(0) != self.m:
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)
        out = torch.empty(self.m, self.n, dtype=torch.float16, device=a.device)
        for i in range(self.m):
            w4a16_tilelp_tilelp_acc_in_rf(self.q_weight, a[i : i + 1], out[i], self.scales)
        return out


class TilelpNoAccInRegisterNoOptMicroKernelLayer(MatmulLayer):
    """W4A16 GEMV via tilelp DSL kernel without acc in register and without optimized micro kernel."""

    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        self.q_weight = torch.randint(0, 16, (n, k // 8), dtype=torch.int32, device='cuda')

    @staticmethod
    def supported_pairs():
        return [(float16, uint4b)]

    def run(self, a=None):
        a = self.a if a is None else a
        if a.size(0) != self.m:
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)
        out = torch.empty(self.m, self.n, dtype=torch.float16, device=a.device)
        for i in range(self.m):
            pass
        return out



import triton
import triton.language as tl
def get_autotune_config():
    configs = []
    for evict in ['evict_last', 'evict_first', None]:
            for evicta in ['evict_last', 'evict_first', None]:
                for BLOCK_SIZE in [128, 256, 512, 1024]:
                    configs.append(triton.Config({ 'evict' : evict, 'evicta' : evicta, 
                    'BLOCK_SIZE' : BLOCK_SIZE,
         }))

    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key = ['m', 'n'],
    use_cuda_graph = False
)
@triton.jit
def gemv_kernel(
    A_ptr, x_ptr, y_ptr,
    m, n,
    stride_am, stride_an,
    BLOCK_SIZE : tl.constexpr ,
    evict : tl.constexpr ,
    evicta : tl.constexpr ,
):
    row_id = tl.program_id(0)
    acc =   tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    
    for off in range(0, n, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        A_offset = row_id * stride_am + cols * stride_an
        x_offset = cols

        a = tl.load(A_ptr + A_offset,  eviction_policy = evict)
        x = tl.load(x_ptr + x_offset,  eviction_policy = evicta)
        acc += a * x
    
    acc = tl.sum(acc, axis=0) 
    tl.store(y_ptr + row_id, acc)

  


def triton_gemv(A: torch.Tensor, vector: torch.Tensor, output, ptx = 0):
    n, k = A.shape
    device = A.device
    stride_ak, stride_an = A.stride()
    assert vector.shape[1] == A.shape[1], "Vector and input tensor shape mismatch"
    assert A.device == device and vector.device == device and output.device == device, "Tensors must be on CUDA"
    grid = lambda meta: (n, )
    
    k = gemv_kernel[grid](A, vector, output, n, k, stride_ak, 
    stride_an)
         
    return k

class GemLiteLayer(MatmulLayer):
    """GemLite W4A16 GEMV using GemLiteLinearTriton with GEMV_REVSPLITK."""

    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        from gemlite.helper import GemLiteLinearTriton, DType
        from hqq.core.quantize import HQQLinear, BaseQuantizeConfig

        in_features = k
        out_features = n
        gs = 128 if group_size == -1 else group_size

        linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float16).cuda()
        orig_shape = (out_features, in_features)
        quant_config = BaseQuantizeConfig(nbits=4, group_size=gs)
        hqq_layer = HQQLinear(linear, quant_config=quant_config,
                              compute_dtype=torch.float16, device='cuda',
                              del_orig=False)

        self.gemlite_linear = GemLiteLinearTriton(
            W_nbits=4,
            group_size=gs,
            in_features=in_features,
            out_features=out_features,
            input_dtype=DType.FP16,
            output_dtype=DType.FP16,
        )
        self.gemlite_linear.pack(
            hqq_layer.unpack(dtype=torch.uint8).view(orig_shape),
            hqq_layer.meta['scale'].clone(),
            hqq_layer.meta['zero'].clone(),
            bias=None,
        )
        self.matmul_type = 'GEMM_SPLITK'

    @staticmethod
    def supported_pairs():
        return [(float16, uint4b)]

    def run(self, a: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = self.a if a is None else a
        if a.size(0) != self.m:
            return torch.empty(a.size(0), self.n, dtype=a.dtype, device=a.device)
        return self.gemlite_linear.forward_manual(a, matmul_type=self.matmul_type)


class CutlassLayer(MatmulLayer):
    """CUTLASS-based GEMV using warp_specialized_gemv_host."""
    
    
    def __init__(self, a_dtype: DataType, b_dtype: DataType, group_size: int, m: int, k: int, n: int):
        super().__init__(a_dtype, b_dtype, group_size, m, k, n)
        

        # Prepare weight matrix (transposed for GEMV: n x k)
        self.weight = torch.randn(n, k, dtype=torch.float16, device='cuda')
        self.weightT = torch.randn(k, n, dtype=torch.float16, device='cuda')

    
    @staticmethod
    def supported_pairs():
        return [(float16, uint4b)]
    
    def run(self, a: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = self.a if a is None else a
        
        if self.m == 1:
            out = torch.empty(self.m, self.n, dtype=torch.float16, device='cuda')

            for i in range(self.m):
                # w4a16(self.q_weight, a[i : i + 1], out[i], self.scales)
                triton_gemv(self.weight, a[i : i + 1], out[i])
        else:
            return torch.matmul(a, self.weightT)
        # return 
