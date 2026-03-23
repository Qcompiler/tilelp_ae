
import tilelp
import torch

import tilelp.language as tl


def get_autotune_config():
    configs = []
    for evict in ['evict_last', 'evict_first', None]:
        for evict_scales in ['evict_last', 'evict_first', None]:
            for BLOCK_SIZE in [128, 256, 512, 1024]:    
                configs.append(tilelp.Config({ 'evict' : evict, 'evict_scales' : evict_scales, 
                                              'BLOCK_SIZE' : BLOCK_SIZE }))

    return configs

@tilelp.autotune(
    configs=get_autotune_config(),
    key = ['m', 'col'],
    use_cuda_graph = False
)
@tilelp.jit
def w4a16_micro_kernel(
    A_ptr : tl.int4, 
    x_ptr : tl.float16x8, 
    y_ptr : torch.float16,
    scales_ptr,
    m, col,
    stride_am, 
    BLOCK_SIZE : tl.constexpr ,
    evict : tl.constexpr,
    evict_scales : tl.constexpr 
):
    row_id = tl.program_id(0)   
    acc =   tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    offs_k =  tl.arange(0, BLOCK_SIZE)
 
    A_ptr = A_ptr + row_id * stride_am + (offs_k)
    x_ptr = x_ptr + (offs_k * 2 )
    for _ in tl.range(0, tl.cdiv(col, BLOCK_SIZE), warp_specialize=True):


        
        scales = tl.load(scales_ptr + row_id, eviction_policy = evict_scales)
        x = tl.load_packed8(x_ptr)
        
        weight_int4 = tl.load(A_ptr, eviction_policy = evict)

        y = tl.dequant(weight_int4, packed_factor = 8) 
 
        acc += tl.dot(x, y) * scales  

        offs_k +=  BLOCK_SIZE 
        A_ptr += (BLOCK_SIZE) 
        x_ptr += (BLOCK_SIZE * 2)

    acc = tl.sum(acc, axis=0) 
    
    
    tl.store(y_ptr + row_id, acc)


def w4a16(A: torch.Tensor, vector: torch.Tensor, output, scales):
    row, column = A.shape
    stride_ak, stride_an = A.stride()
 
    activation = tl.convert(vector, dtype = tl.float16x8)
    A = tl.convert(A, dtype = tl.int4)
    kernel = w4a16_micro_kernel[(row, 1)](A, activation, output, scales, row, column,  
                            stride_ak)
    return kernel



def get_autotune_config():
    configs = []
    for evict in ['evict_last', 'evict_first', None]:
        for evict_scales in ['evict_last', 'evict_first', None]:
            configs.append(tilelp.Config({ 'evict' : evict, 'evict_scales' : evict_scales}))

    return configs

@tilelp.autotune(
    configs=get_autotune_config(),
    key = ['m', 'k'],
    use_cuda_graph = False
)
@tilelp.jit
def w4a16_micro_kernel_tilelp_evict(
    A_ptr , 
    x_ptr , 
    y_ptr ,
    scales_ptr,
    m, k,
    stride_am, 
    BLOCK_SIZE : tl.constexpr = 256,
    evict : tl.constexpr = 'evict_first',
    evict_scales : tl.constexpr = None
):
    row_id = tl.program_id(0)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    elements_per_sample = 8
    W_nbits = 4
    unpack_mask = 15
    
    offs_k =  tl.arange(0, BLOCK_SIZE)
    A_offset = row_id * stride_am + (offs_k // 8) 
    
    for kk in range(0, tl.cdiv(k, BLOCK_SIZE)):
          
      mask = offs_k < k
      scales = tl.load(scales_ptr + row_id, eviction_policy = evict_scales)
      a = tl.load(A_ptr + A_offset, eviction_policy = evict)

      q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)
      a = ((a >> q_shift) & unpack_mask) - 8 
      a = a.to(tl.float16)
      x = tl.load(x_ptr + offs_k, mask = mask, other=0.0)
      acc += (a * x * scales) 

      offs_k +=  BLOCK_SIZE
      A_offset += (BLOCK_SIZE // 8) 

    acc = tl.sum(acc, axis=0)
    tl.store(y_ptr + row_id, acc)


def w4a16_tilelp_evict(A: torch.Tensor, vector: torch.Tensor, output, scales):
    row, column = A.shape
    stride_ak, stride_an = A.stride()
    
    k = vector.shape[1] # [m, k ]
    kernel = w4a16_micro_kernel_tilelp_evict[(row, 1)](A, vector, output, scales, row, k,  
                            stride_ak)
    return kernel



@tilelp.jit
def w4a16_tilelp_tilelp_acc_in_rf_kernel(
    A_ptr : tl.int4, 
    x_ptr : tl.float16x8, 
    y_ptr : torch.float16,
    scales_ptr,
    m, k,
    stride_am, 
    BLOCK_SIZE : tl.constexpr = 256,
    evict : tl.constexpr = 'evict_first',
    evict_scales : tl.constexpr = None
):

    row_id = tl.program_id(0)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float16)
    elements_per_sample = 8
    W_nbits = 4
    unpack_mask = 15
    
    offs_k =  tl.arange(0, BLOCK_SIZE)
    A_offset = row_id * stride_am + (offs_k // 8) 
    
    for kk in range(0, tl.cdiv(k, BLOCK_SIZE)):
          
      mask = offs_k < k
      scales = tl.load(scales_ptr + row_id, eviction_policy = evict_scales)
      a = tl.load(A_ptr + A_offset)

      q_shift = ((offs_k % elements_per_sample) * W_nbits).to(tl.int32)
      a = ((a >> q_shift) & unpack_mask) - 8 
      a = a.to(tl.float16)
      x = tl.load(x_ptr + offs_k, mask = mask, other=0.0)
      acc += (a * x * scales) 

      offs_k +=  BLOCK_SIZE
      A_offset += (BLOCK_SIZE // 8) 

    acc = tl.sum(acc, axis=0)
    tl.store(y_ptr + row_id, acc)


def w4a16_tilelp_tilelp_acc_in_rf(A: torch.Tensor, vector: torch.Tensor, output, scales):

    n, _ = A.shape
    k = vector.shape[1] # [m, k]
    device = A.device
    stride_ak, stride_an = A.stride()
    assert vector.shape[1] == A.shape[1] * 8, "Vector and input tensor shape mismatch"
    assert A.device == device and vector.device == device and output.device == device, "Tensors must be on CUDA"
    grid = lambda meta: (n, 1)

    kernel = w4a16_tilelp_tilelp_acc_in_rf_kernel[ grid ](A, vector, 
                                                           output, 
                                                           scales, n, 
                                                           k,  
                                                            stride_ak)
    return kernel





import triton 


@triton.jit
def dequanti(b):
    x1, x2, x3, x4 = tl.inline_asm_elementwise(
        asm="""
            {
            .reg .b32 	r<16>;
            .reg .b32  r_high<2>, r_low<2>;

	        .reg .b64 	rd<2>;
            mov.u32 r2, $4;
            mov.u32 	r3, 983055;
            mov.u32 	r8, 1677747200;
            lop3.b32 r1, r2, r3, r8, 234;
            mov.u32 	r7, 15728880;
            lop3.b32 r5, r2, r7, r8, 234;
            mov.u32 	r11, 1678271496;
            mov.u32 	r14, 738208768;
            mov.u32 	r15, -729754496;
            fma.rn.f16x2 r12,r5,r14,r15;
            sub.f16x2 r9,r1,r11;
            
            shr.s32   r_high1, r9, 16;
            cvt.u16.u32   $0, r_high1;
            and.b32       r_low1, r9, 0xFFFF;
            cvt.u16.u32   $1, r_low1;

            shr.s32   r_high1, r12, 16;
            cvt.u16.u32   $2, r_high1;
            and.b32       r_low1, r12, 0xFFFF;
            cvt.u16.u32   $3, r_low1;
            }
        """,
        constraints=(
            "=f,=f,=f,=f,r"
        ),
        args=[b], #输入 参数4
        dtype=(tl.float16, tl.float16, tl.float16, tl.float16), #参数0
        is_pure=False,
        pack=1,
    )

    
    return x1, x2, x3, x4





@triton.jit
def dequant_uint32_2_half4(a):
    x1, x2 = tl.inline_asm_elementwise(
        asm="""
            {
              .reg .u32 t0;
              .reg .u16 h0, h1;
              mov.u32 t0, $2;
              cvt.u16.u32 $0, t0;      
              shr.u32 t0, t0, 16;      
              cvt.u16.u32 $1, t0;      
            }
        """,
        constraints=(
            "=f,=f,r"
        ),
        args=[a],  
        dtype=(tl.float16, tl.float16), #输出
        is_pure=False,
        pack=1,
    )

    
    return x1, x2




def get_autotune_config():
    configs = []
    for evict in ['evict_last', 'evict_first', None]:
        for evict_scales in ['evict_last', 'evict_first', None]:
            configs.append(tilelp.Config({ 'evict' : evict, 'evict_scales' : evict_scales}))

    return configs

@tilelp.autotune(
    configs=get_autotune_config(),
    key = ['m', 'int4_k'],
    use_cuda_graph = False
)
@triton.jit
def w4a16_tilelp_tilelp_no_acc_in_register_no_opt_micro_kernel(
    A_ptr, x_ptr, y_ptr, scales_ptr, 
    m, k, int4_k,
    stride_am, stride_an,
    BLOCK_SIZE: tl.constexpr = 256, a_evict: tl.constexpr = 'evict_first',
    evict_scales: tl.constexpr = 'evict_first'
):


    row_id = tl.program_id(0)
    acc = 0.0
    
    offs_k =  tl.arange(0, BLOCK_SIZE)
    A_offset = row_id * stride_am + (offs_k) * stride_an
    
    
    for kk in range(0, tl.cdiv(int4_k, BLOCK_SIZE)):
          
      scales = tl.load(scales_ptr + row_id, eviction_policy = evict_scales)
      mask = offs_k < int4_k
      
      a = tl.load(A_ptr + A_offset, mask = mask)


      u32_data1 = tl.load(x_ptr + (offs_k * 4), mask=mask)
      u32_data2 = tl.load(x_ptr + (offs_k * 4 + 1), mask=mask)
      u32_data3 = tl.load(x_ptr + (offs_k * 4 + 2), mask=mask)
      u32_data4 = tl.load(x_ptr + (offs_k * 4 + 3), mask=mask)
      a1, a2, a3, a4 = dequanti(a)
      x1, x2 =  dequant_uint32_2_half4(u32_data1)
      x3, x4 =  dequant_uint32_2_half4(u32_data2)
      x5, x6 =  dequant_uint32_2_half4(u32_data3)
      x7, x8 =  dequant_uint32_2_half4(u32_data4)
      a = a >> 8
      a5, a6, a7, a8 = dequanti(a)


      all = a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6 + a7 * x7 + a8 * x8
      
      acc += tl.sum(all * scales, axis=0) 

      offs_k +=  BLOCK_SIZE
      A_offset += (BLOCK_SIZE) * stride_an

    tl.store(y_ptr + row_id, acc)


def w4a16_tilelp_tilelp_no_acc_in_register_no_opt_micro(A: torch.Tensor, vector: torch.Tensor, output, ptx = 0):
    n, _ = A.shape
    k = vector.shape[1] # [m, k ]
    device = A.device
    stride_ak, stride_an = A.stride()
    assert vector.shape[1] == A.shape[1] * 8, "Vector and input tensor shape mismatch"
    assert A.device == device and vector.device == device and output.device == device, "Tensors must be on CUDA"
    grid = lambda meta: (n, 1)
    
    
    int4_k = A.shape[1]
    scales = torch.empty((n,), dtype=torch.float16, device=device)
    # 获取原始存储
    storage = vector.untyped_storage()
    k = vector.numel()  # 元素数量

    # 正确的set_用法 - 第三个参数必须是tuple
    uint32_tensor = torch.tensor([], dtype=torch.uint32,device=device).set_(storage, 0, (k // 2,))
    w4a16_tilelp_tilelp_no_acc_in_register_no_opt_micro_kernel[grid](A, uint32_tensor, output, scales, n, k, int4_k,  
                            stride_ak, stride_an)
    return 