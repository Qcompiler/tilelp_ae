ranked_executors = ['torch-f16', 'cutlass', 'triton', 'quant-llm', 'bitblas', 'marlin', 'mutis', 'tilelp_acc_in_register',
                    'tilelp_evict',  'tilelp_no_acc_in_register_no_opt_micro_kernel', 'tilelp', 'gemlite']

colors = [
    ['#fbb4ae'],
    ['#b3cde3'],
    ['#ccebc5'],
    ['#decbe4'],
    ['#fed9a6'],
    ['#ffffcc'],
    ["#DB8787"],
    ['#e5c494'],
    ['#a6d854'],
    ['#ffd92f'],
    ['#e78ac3'],
    ['#66c2a5'],
]

executor2color = {
    'torch-f16': 3,
    'triton': 1,
    'bitblas': 4,
    'quant-llm': 3,
    'marlin': 0,
    'mutis': 2,
    'tilelp': 6,
    'cutlass': 7,
    'tilelp_evict': 8,
    'tilelp_acc_in_register': 9,
    'tilelp_no_acc_in_register_no_opt_micro_kernel': 10,
    'gemlite': 11,
}
executor2label = {
    'torch-f16': 'cuBLAS (FP16)',
    'triton': 'Triton',
    'quant-llm': 'QuantLLM',
    'bitblas': 'Ladder',
    'marlin': 'Marlin',
    'mutis': 'Tilus',
    
    'cutlass': 'CUTLASS (FP16)',
    'tilelp_acc_in_register': 'TileLP (Micro Kernel)',
    'tilelp_evict': 'TileLP (Tune L1 Cache)',
    'tilelp_no_acc_in_register_no_opt_micro_kernel': 'TileLP (Tune xx)',
    'tilelp': 'TileLP',
    'gemlite': 'GemLite',
}

