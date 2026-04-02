

from torch import Tensor
import triton
import triton.language as tl
import torch

import torch, math, random, copy
from config import AUTOTUNE
from config import *



KEYS        = ['M', 'N', 'K', 'group_size', 'elements_per_sample',] 


@triton.jit
def dequanti_tensorRT_llm(b, scales):
    # b: int32 input
    # Returns 8 individual fp16 values
    x1, x2, x3, x4, x5, x6, x7, x8 = tl.inline_asm_elementwise(
        asm="""
            {
            .reg .b32 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12;
            .reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;
            .reg .b16 s;
            mov.u32 r0, $8;
            shr.u32 r1, r0, 8;
            lop3.b32 r2, r0, 983055, 1677747200, 234;
            lop3.b32 r3, r0, 15728880, 1677747200, 234;
            lop3.b32 r4, r1, 983055, 1677747200, 234;
            lop3.b32 r5, r1, 15728880, 1677747200, 234;
            mov.u32 r6, 1678271496;
            mov.u32 r8, 738208768;
            mov.u32 r9, -729754496;
            sub.f16x2 r7, r2, r6;
            sub.f16x2 r11, r4, r6;
            fma.rn.f16x2 r10, r3, r8, r9;
            fma.rn.f16x2 r12, r5, r8, r9;
            mov.b32 {h0, h1}, r7;
            mov.b32 {h2, h3}, r10;
            mov.b32 {h4, h5}, r11;
            mov.b32 {h6, h7}, r12;
            mov.b16 s, $9;
            mul.f16 h0, h0, s;
            mul.f16 h1, h1, s;
            mul.f16 h2, h2, s;
            mul.f16 h3, h3, s;
            mul.f16 h4, h4, s;
            mul.f16 h5, h5, s;
            mul.f16 h6, h6, s;
            mul.f16 h7, h7, s;
            mov.b16 $0, h0;
            mov.b16 $1, h1;
            mov.b16 $2, h2;
            mov.b16 $3, h3;
            mov.b16 $4, h4;
            mov.b16 $5, h5;
            mov.b16 $6, h6;
            mov.b16 $7, h7;
            }
        """,
        constraints="=h,=h,=h,=h,=h,=h,=h,=h,r,h",
        args=[b, scales],
        dtype=(tl.float16, tl.float16, tl.float16, tl.float16,
               tl.float16, tl.float16, tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    return x1, x2, x3, x4, x5, x6, x7, x8


# best config selected: BLOCK_SIZE_M: 16, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 128, GROUP_SIZE_M: 8, SPLIT_K: 1, 
# A_load_order: 2, NUM_STAGES: 1, num_warps: 4, num_ctas: 1, num_stages: 1, maxnreg: None;
MATMUL_TYPE = "GEMM_SPLITK"


TRITON_CONFIG_CACHE  = {}

def kernel_config_pruner(configs, nargs, **kwargs):
    from core import TRITON_CONFIG_CACHE

    m = nargs['M'] 
    n = nargs['N'] 
    k = nargs['K'] 
    g = nargs['group_size']
    e = nargs['elements_per_sample']
    t = kwargs.get('type_id', nargs.get('type_id', 0))
    a_sizeof = kwargs.get('a_sizeof', nargs.get('a_sizeof', 1))
    b_sizeof = kwargs.get('b_sizeof', nargs.get('b_sizeof', 1))

    #Check cache
    if(MATMUL_TYPE in TRITON_CONFIG_CACHE):
        signature = str(tuple([get_closest_m(m), n, k, g, e, t]))
        if(signature in TRITON_CONFIG_CACHE[MATMUL_TYPE]):
            config     = copy.deepcopy(TRITON_CONFIG_CACHE[MATMUL_TYPE][signature])
            num_stages = config.pop('num_stages')
            num_warps  = config.pop('num_warps')
            num_ctas   = config.pop('num_ctas')

            config.pop('num_buffers_warp_spec', None)
            config.pop('num_consumer_groups', None)
            config.pop('reg_dec_producer', None)
            config.pop('reg_inc_consumer', None)
            config["NUM_STAGES"] = num_stages

            yield triton.Config(config,
                num_stages=num_stages,
                num_warps=num_warps,
                pre_hook=init_to_zero("c_ptr") if (config['SPLIT_K'] > 1) else None,
            )

            return

    gpu_shared_memory = get_gpu_shared_memory() 
    load_scales_as_block = kwargs['load_scales_as_block']
    used = set()
    for config in configs:
        group_size_m = config.kwargs['GROUP_SIZE_M']
        block_size_m = config.kwargs['BLOCK_SIZE_M']
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])
        split_k      = config.kwargs['SPLIT_K']

        A_load_order = config.kwargs['A_load_order']
        num_stages   = config.num_stages
        num_warps    = config.num_warps

        #Autotune prune the batch_size (1..64)
        if m <= 16:   block_size_m = 16
        elif m <= 32: block_size_m = min(max(block_size_m, 16), 32) #m: [16, 32]
        elif m <= 64: block_size_m = min(max(block_size_m, 32), 64) #m: [32, 64]
        elif m > 64 : block_size_m = 64

        #Only use higher split_k values for smaller m
        if(m >= 32): split_k = min(split_k, 8)

        #Constraint: BLOCK_SIZE_K >= group_size, only for load_as_block = False
        if(load_scales_as_block):
            num_stages = max(num_stages, 2) #for dot_scaled kernels with pipelined loads
            if(e > 1):
                block_size_k = max(block_size_k, 64) #m16n8k64
            else:
                block_size_k = max(block_size_k, 32) #m16n8k32
        else:
            block_size_k = min(block_size_k, g)

        block_size_k = next_power_of_2(block_size_k)
        block_size_n = next_power_of_2(block_size_n)

        #Constraint: K needs to be divisible by BLOCK_SIZE_K * SPLIT_K 
        while split_k > 1 and not is_divisible(k, block_size_k * split_k):
        #while split_k > 1 and k > block_size_k * split_k:
            split_k //= 2

        #Nvidia
        if not IS_HIP:
            if e > 1 and not load_scales_as_block:
                #Limit num stages when data is packed
                num_stages = min(num_stages, 4)
            if(e == 1 and num_stages == 1): 
                #skip num_stages=1 for non-packed weights
                continue

        #Avoid OOM
        while num_stages > 0: #TODO: revisit MXFP case
            shared_mem = (block_size_m * block_size_k * a_sizeof + block_size_k * block_size_n * b_sizeof)
            if(e > 1 and not load_scales_as_block): 
                shared_mem += block_size_k * block_size_n * a_sizeof
            shared_mem *= num_stages
            if int(shared_mem) <= gpu_shared_memory:
                break
            num_stages -= 1

        if(num_stages == 0): continue #config too large

        ###########################################
        if(load_scales_as_block):#tmp MXFP fix
            block_size_k = min(block_size_k, 256)
        ###########################################

        key = (block_size_m, block_size_n, block_size_k, group_size_m, split_k, A_load_order, num_stages, num_warps)
        
        new_config = {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "GROUP_SIZE_M": group_size_m,
            "SPLIT_K": split_k,
            "A_load_order": A_load_order,
            "NUM_STAGES": num_stages,
        }

        if IS_HIP:
            new_config['waves_per_eu'] = config.kwargs.get('waves_per_eu', 0)
            new_config['matrix_instr_nonkdim'] = config.kwargs.get('matrix_instr_nonkdim', 16) #MI300X
            key = key + (new_config['waves_per_eu'], new_config['matrix_instr_nonkdim'])

        if key in used:
            continue

        used.add(key)
        yield triton.Config(new_config,
            num_stages=num_stages,
            num_warps=num_warps,
            pre_hook=init_to_zero("c_ptr") if split_k > 1 else None, 
        )

########################################################################################################################################################################
#Nvidia

#These autotunes are optimized for batch-size 1 to 64 (!)
def get_max_autotune_config_nvidia():
    stages  = [1, 2, 4, 5] if gpu_has_more_shared_memory() else [1, 2, 4]
    configs = []
    for A in [0, 2]:
        for w in [4, 8]:
            for s in stages:
                for M in [16, 32, 64]:
                    for N in [32, 64, 128, 256, 512]:
                        for K in [32, 64, 128,]:
                            for split_k in [1, 2, 4, 8, 16]:
                                configs.append(
                                    triton.Config(
                                        {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K // 8, 
                                        "SPLIT_K": split_k, "GROUP_SIZE_M": 8, "A_load_order": A},
                                        num_warps=w, num_stages=s,
                                    )
                                )
    return configs

#Faster autotuner 
def get_fast_autotune_config_nvidia():
    configs = []
    for split_k in [1, 2, 4, 8, 16]:
        for stages in [1, 2, 3, 4]:
            configs.append(triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 
                                        'BLOCK_SIZE_K': 32, 'SPLIT_K':split_k, 'GROUP_SIZE_M':8,
                                            'A_load_order':2}, num_stages=stages,
                        num_warps=4)) 
            configs.append(triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 
                                        'BLOCK_SIZE_K': 32, 'SPLIT_K':split_k, 'GROUP_SIZE_M':8,
                                            'A_load_order':2}, num_stages=stages,
                        num_warps=4)) 
            configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 
                                        'BLOCK_SIZE_K': 32, 'SPLIT_K':split_k, 'GROUP_SIZE_M':8,
                                            'A_load_order':2}, num_stages=stages,
                        num_warps=4)) 
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':64,  'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':128, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':256, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':512, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':32,  'SPLIT_K':8, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64,  'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=5))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':512, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
        
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'SPLIT_K':8, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=4)) 
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=8, num_stages=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=8, num_stages=4)) 
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=5))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':128, 'SPLIT_K':2, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=2))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':256, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=8, num_stages=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':512, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0}, num_warps=4, num_stages=4))
    
    # configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':512, 'BLOCK_SIZE_K':32, 
    #                                'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_warps=4, num_stages=4))
    
    
    
    
    # configs.append(triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=3,
    #                 num_warps=8))
    # configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=4,
    #                 num_warps=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=4,
    #                 num_warps=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=4,
    #                 num_warps=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=4,
    #                 num_warps=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=4,
    #                 num_warps=4))   
    # configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=5,
    #                 num_warps=2))           
    # configs.append(triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=5,
    #                 num_warps=2))
    # # Good   config for fp8 inputs.
    # configs.append(triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=3,
    #                 num_warps=8))
    # configs.append(triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=3,
    #                 num_warps=8))
    # configs.append(triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=4,
    #                 num_warps=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=4,
    #                 num_warps=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=4,
    #                 num_warps=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=4,
    #                 num_warps=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=4,
    #                 num_warps=4))
    # configs.append(triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'SPLIT_K':4, 'GROUP_SIZE_M':8, 'A_load_order':2}, num_stages=4,
    #                 num_warps=4))
    
    return configs

def get_default_config_nvidia():
    return [triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0, 'NUM_STAGES':2}, num_warps=4, num_stages=2)]

########################################################################################################################################################################
#AMD - Instinct MI300X

#These autotunes are optimized for batch-size 1 to 64 (!)
def get_max_autotune_config_amd():
    configs = []
    for A in [0]:
        for w in [4, 8]:
            for s in [1, 2]:
                for v in [0, 2, 4]:
                    for M in [16, 32, 64]:
                        for N in [32, 64, 128, 256, 512]:
                            for K in [32, 64, 128, 256, 512]:
                                for split_k in [1, 2, 4, 8, 16]:
                                    configs.append(
                                        triton.Config(
                                            {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K, 
                                            "SPLIT_K": split_k, "GROUP_SIZE_M": 8, "A_load_order": A, 'waves_per_eu': v},
                                            num_warps=w, num_stages=s,
                                        )
                                    )
    return configs

#Faster autotuner 
def get_fast_autotune_config_amd():
    configs = [] #BLOCK_SIZE_M is automatically adapted in the config pruning.
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':0}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':0}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':4, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':512, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=2))

    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'SPLIT_K':2, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'SPLIT_K':2, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':4, 'A_load_order':0, 'waves_per_eu':2}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':8, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'SPLIT_K':8, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':2}, num_warps=4, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':512, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=4, num_stages=1))

    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,  'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=1))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'SPLIT_K':4 ,'A_load_order':0, 'waves_per_eu':2}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,  'GROUP_SIZE_M':8, 'SPLIT_K':8 ,'A_load_order':0, 'waves_per_eu':2}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':2, 'A_load_order':0, 'waves_per_eu':4}, num_warps=8, num_stages=2))
    configs.append(triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128, 'GROUP_SIZE_M':8, 'SPLIT_K':1, 'A_load_order':0, 'waves_per_eu':0}, num_warps=8, num_stages=1))

    return configs

def get_default_config_amd():
    return [triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32, 'SPLIT_K':1, 'GROUP_SIZE_M':8, 'A_load_order':0, 'NUM_STAGES':2}, num_warps=4, num_stages=2)]
########################################################################################################################################################################

IS_HIP = is_hip() 
if IS_HIP:
    get_max_autotune_config = get_max_autotune_config_amd
    get_fast_autotune_config = get_fast_autotune_config_amd
    get_default_config = get_default_config_amd
else:
    get_max_autotune_config = get_max_autotune_config_nvidia
    get_fast_autotune_config = get_fast_autotune_config_nvidia
    get_default_config = get_default_config_nvidia



AUTOTUNE_SETTING = AUTOTUNE.GEMM_SPLITK
if(AUTOTUNE_SETTING == 'max'):
    get_autotune_config = get_max_autotune_config
elif(AUTOTUNE_SETTING == 'fast'):
    get_autotune_config = get_fast_autotune_config
else:
    get_autotune_config = get_default_config


def divide_block_size_k_by_8(configs):
    for cfg in configs:
        cfg.kwargs['BLOCK_SIZE_K'] //= 8
    return configs
config = divide_block_size_k_by_8(get_fast_autotune_config_nvidia())
@triton.autotune(
    configs=config,
    key = KEYS,
    # prune_configs_by = {'early_config_prune': kernel_config_pruner},
    use_cuda_graph = AUTOTUNE.USE_CUDA_GRAPH,
)
@triton.jit
def gemm_splitK_INT_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, 
    M, N, K, 
    group_size: tl.constexpr, 
    ######### Strides #########
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_a_m, stride_meta_a_g,
    stride_meta_g, stride_meta_n,
    ##
    elements_per_sample,
    type_id,
    a_sizeof, 
    b_sizeof,
    ##
    ######### Dtypes #########
    load_scales_as_block, #False | IF FALSE, RESTRICT BLOCK_SIZE_K <= 32
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    ######### Meta-data mode #########
    ######### tuning params #########
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr, A_load_order: tl.constexpr,  
    #################################
    meta_evict_policy: tl.constexpr = '',
    atomic_mode: tl.constexpr = 'relaxed',
    a_evict: tl.constexpr = 'evict_last',
    b_evict: tl.constexpr = 'evict_first',
):
    """
    Based on https://github.com/foundation-model-stack/foundation-model-stack/blob/triton/triton/kernels/gptq/splitk_dequant_gemm.py
    GEMM for C = matmul(A, dequantize(B, scales, zeros))
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K//elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16

    BLOCK_SIZE_M >=16
    BLOCK_SIZE_K * SPLIT_K <= group_size for imp1
    BLOCK_SIZE_K == SPLIT_K for imp2 (similar to original)
    """

    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    #Swizzle?
    pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)

    num_pid_k = tl.cdiv(K // 8, BLOCK_SIZE_K * SPLIT_K)

    #Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # offs_k indexes into the K//8 dimension (each b element expands to 8 fp16 k-values)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    offs_bn = offs_n  

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    # offs_ak indexes into the full K dimension (8x expanded)
    offs_ak = pid_k * 8 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K * 8)
    offs_bk = offs_k

    # b_ptrs: b is stored as packed ints, each row in K//8 dimension
    b_ptrs  = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    #Inputs
    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)  
    a_mask  = ((offs_am[:, None] < M) & (offs_ak[None, :] < K)).to(tl.int1)

    #Meta data stuff
    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n


    # stride_mul: how many group rows are covered per BLOCK_SIZE_K steps in the K//8 dimension
    stride_mul: tl.constexpr     = (BLOCK_SIZE_K * 8) / group_size
    BLOCK_SIZE_K_U: tl.constexpr = BLOCK_SIZE_K * SPLIT_K * 8  # advance in full-K dimension
    BLOCK_SIZE_K_P: tl.constexpr = BLOCK_SIZE_K * SPLIT_K      # advance in K//8 dimension


    #############################################################################################################
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)


    for k in tl.range(num_pid_k):

    
        if(A_load_order == 0): #Late load 
             
            a_block = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)
        
        

        b = tl.load(b_ptrs, eviction_policy=b_evict)

        if(A_load_order == 1): #Late load 
             
            a_block = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)
        #Meta-data loading policy
        k_m = ((k * SPLIT_K + pid_k) * stride_mul).to(tl.int32) 

        scales = tl.load(scales_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy)

        if(A_load_order == 2): #Late load 
             
            a_block = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)
        # Unpack and dequantize: tl.dequant_new outputs 8 fp16 tensors [BLOCK_SIZE_K, BLOCK_SIZE_N]
        # b1, b2, b3, b4, b5, b6, b7, b8 = dequanti_tensorRT_llm(b, scales)
        b1, b2, b3, b4, b5, b6, b7, b8 = dequanti_tensorRT_llm(b, scales)
        bx1 = tl.cat(b1, b2, dim = 0)  
        bx2 = tl.cat(b3, b4, dim = 0) 
        bx3 = tl.cat(b5, b6, dim = 0)  
        bx4 = tl.cat(b7, b8, dim = 0) 
        bxx1 = tl.cat(bx1, bx2, dim = 0)
        bxx2 = tl.cat(bx3, bx4, dim = 0)
        ball = tl.cat(bxx1, bxx2, dim = 0)


        if(A_load_order == 3): #Late load 
             
            a_block = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy=a_evict)

        acc = tl.dot(a_block, ball, acc=acc, out_dtype=acc_dtype)



        #Advance
        a_ptrs += BLOCK_SIZE_K_U * stride_ak
        b_ptrs += BLOCK_SIZE_K_P * stride_bk


    #############################################################################################################
    #Output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask    = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if(SPLIT_K > 1):
        tl.atomic_add(c_ptrs, acc, mask=mask, sem=atomic_mode) 
    else:
        tl.store(c_ptrs, acc, mask=mask) 


def gemm_splitK_forward(x: Tensor, W_q: Tensor, scales: Tensor, 
                         group_size: int,   elements_per_sample: int
                        ) -> Tensor: 
        

    M, K, N = x.shape[0], W_q.shape[0] * elements_per_sample, W_q.shape[1]


    output = torch.empty((M, N), device=W_q.device, dtype=torch.float16)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), 
                         META['SPLIT_K'])


    stride_meta_a_m, stride_meta_a_g = None, None

 
    # print("elements_per_sample:", elements_per_sample)
    # print("W_group_mode:", W_group_mode)
    # print(is_mx_dtype(input_dtype))
    
    # print("channel_scale_mode:", channel_scale_mode)
    # print("zeros is scalar:", zeros.numel() == 1)
    # exit()
    zeros = None

    kernel = gemm_splitK_INT_kernel[grid](
        x, W_q, output, 
        scales,  
        M, N, K, group_size,
        ###############################################
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        stride_meta_a_m, stride_meta_a_g,
        scales.stride(0), scales.stride(1),
        ################################################
        elements_per_sample,
        type_id = 0, 
        a_sizeof = x.itemsize, 
        b_sizeof = W_q.itemsize,
        ##############
        load_scales_as_block = False,
        input_dtype  = tl.float16,
        output_dtype = tl.float16,
        acc_dtype    = tl.float32,
        meta_dtype   = tl.float16
    )

    # with open("gemm_splitK_INT_kernel.ptx", "w") as f:
    #     f.write(kernel.asm["ptx"])
    # exit()
    return output, gemm_splitK_INT_kernel.best_config