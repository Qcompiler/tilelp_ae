import argparse
import copy
import itertools
import pickle as pkl
import time
from typing import Callable, Iterable, List, Tuple

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
import numpy as np
import matplotlib.pyplot as plt
# from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.utils import FlexibleArgumentParser

# DEFAULT_MODELS = list(WEIGHT_SHAPES.keys())[1:]
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512]
DEFAULT_TP_SIZES = [1]

# helpers

def calculate_tflops(m: int, k: int, n: int, time_seconds: float) -> float:
    """
    计算GEMM操作的TFLOPS
    GEMM操作数 = 2 * M * K * N (乘加操作)
    """
    flops = 2.0 * m * k * n
    tflops = flops / (time_seconds * 1e12)
    return tflops

def extract_tflops_from_measurement(measurement: TMeasurement, m: int, k: int, n: int) -> float:
    """
    从测量结果中提取TFLOPS
    """
    # 获取中位数时间（秒）
    median_time = measurement.median / 1000  # 转换为秒
    return calculate_tflops(m, k, n, median_time)

def to_fp8(tensor: torch.tensor) -> torch.tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)

def to_int8(tensor: torch.tensor) -> torch.tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def make_rand_tensors(dtype: torch.dtype, m: int, n: int,
                      k: int) -> Tuple[torch.tensor, torch.tensor]:

    a = torch.randn((m, k), device='cuda') * 5
    b = torch.randn((n, k), device='cuda').t() * 5

    if dtype == torch.int8:
        return to_int8(a), to_int8(b)
    if dtype == torch.float8_e4m3fn:
        return to_fp8(a), to_fp8(b)

    raise ValueError("unsupported dtype")

# impl

def pytorch_mm_impl(a: torch.tensor, b: torch.tensor, scale_a: torch.tensor,
                    scale_b: torch.tensor,
                    out_dtype: torch.dtype) -> torch.tensor:
    return torch.mm(a, b)

def pytorch_fp8_impl(a: torch.tensor, b: torch.tensor, scale_a: torch.tensor,
                     scale_b: torch.tensor,
                     out_dtype: torch.dtype) -> torch.tensor:
    return torch._scaled_mm(a,
                            b,
                            scale_a=scale_a,
                            scale_b=scale_b,
                            out_dtype=out_dtype)

def pytorch_fp8_impl_fast_accum(a: torch.tensor, b: torch.tensor,
                                scale_a: torch.tensor, scale_b: torch.tensor,
                                out_dtype: torch.dtype) -> torch.tensor:
    return torch._scaled_mm(a,
                            b,
                            scale_a=scale_a,
                            scale_b=scale_b,
                            out_dtype=out_dtype,
                            use_fast_accum=True)

def cutlass_impl(a: torch.tensor, b: torch.tensor, scale_a: torch.tensor,
                 scale_b: torch.tensor,
                 out_dtype: torch.dtype) -> torch.tensor:
    return ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype=out_dtype)

# bench
def bench_fn(a: torch.tensor, b: torch.tensor, scale_a: torch.tensor,
             scale_b: torch.tensor, out_dtype: torch.dtype, label: str,
             sub_label: str, fn: Callable, description: str) -> TMeasurement:

    min_run_time = 1

    globals = {
        "a": a,
        "b": b,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "out_dtype": out_dtype,
        "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(a, b, scale_a, scale_b, out_dtype)",
        globals=globals,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)

def bench_int8(dtype: torch.dtype, m: int, k: int, n: int, label: str,
               sub_label: str) -> Iterable[TMeasurement]:
    assert dtype == torch.int8
    a, b = make_rand_tensors(torch.int8, m, n, k)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    timers = []
    # pytorch impl
    timers.append(
        bench_fn(a.to(dtype=torch.bfloat16, device="cuda"),
                 b.to(dtype=torch.bfloat16, device="cuda"), scale_a, scale_b,
                 torch.bfloat16, label, sub_label, pytorch_mm_impl,
                 "pytorch_bf16_bf16_bf16_matmul-no-scales"))

    # cutlass impl
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.bfloat16, label, sub_label,
                 cutlass_impl, "cutlass_i8_i8_bf16_scaled_mm"))

    return timers

def bench_fp8(dtype: torch.dtype, m: int, k: int, n: int, label: str,
              sub_label: str) -> Iterable[TMeasurement]:
    assert dtype == torch.float8_e4m3fn
    a, b = make_rand_tensors(torch.float8_e4m3fn, m, n, k)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    timers = []

    # pytorch impl w. bf16
    timers.append(
        bench_fn(a.to(dtype=torch.bfloat16, device="cuda"),
                 b.to(dtype=torch.bfloat16, device="cuda"), scale_a, scale_b,
                 torch.bfloat16, label, sub_label, pytorch_mm_impl,
                 "pytorch_bf16_bf16_bf16_matmul-no-scales"))

    # pytorch impl: bf16 output, without fp8 fast accum
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.bfloat16, label, sub_label,
                 pytorch_fp8_impl, "pytorch_fp8_fp8_bf16_scaled_mm"))

    # pytorch impl: bf16 output, with fp8 fast accum
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.bfloat16, label, sub_label,
                 pytorch_fp8_impl_fast_accum,
                 "pytorch_fp8_fp8_bf16_scaled_mm_fast_accum"))

    # pytorch impl: fp16 output, without fp8 fast accum
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.float16, label, sub_label,
                 pytorch_fp8_impl, "pytorch_fp8_fp8_fp16_scaled_mm"))

    # pytorch impl: fp16 output, with fp8 fast accum
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.float16, label, sub_label,
                 pytorch_fp8_impl_fast_accum,
                 "pytorch_fp8_fp8_fp16_scaled_mm_fast_accum"))

    # cutlass impl: bf16 output
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.bfloat16, label, sub_label,
                 cutlass_impl, "cutlass_fp8_fp8_bf16_scaled_mm"))
    # cutlass impl: fp16 output
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.float16, label, sub_label,
                 cutlass_impl, "cutlass_fp8_fp8_fp16_scaled_mm"))
    return timers

def bench(dtype: torch.dtype, m: int, k: int, n: int, label: str,
          sub_label: str) -> Iterable[TMeasurement]:
    if dtype == torch.int8:
        return bench_int8(dtype, m, k, n, label, sub_label)
    if dtype == torch.float8_e4m3fn:
        return bench_fp8(dtype, m, k, n, label, sub_label)
    raise ValueError("unsupported type")

# runner
def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()

def print_tflops_table(timers_with_info: List[Tuple[TMeasurement, int, int, int]]):
    """
    打印TFLOPS表格
    timers_with_info: [(measurement, m, k, n), ...]
    """
    print("\n" + "="*100)
    print("TFLOPS Results:")
    print("="*100)
    print(f"{'Implementation':<45} {'M':<8} {'K':<8} {'N':<8} {'Time (ms)':<12} {'TFLOPS':<12}")
    print("-"*100)
    
    for measurement, m, k, n in timers_with_info:
        median_time_ms = measurement.median
        tflops = extract_tflops_from_measurement(measurement, m, k, n)
        print(f"{measurement.description:<45} {m:<8} {k:<8} {n:<8} {median_time_ms:<12.3f} {tflops:<12.2f}")
    print("="*100 + "\n")

def plot_tflops_vs_batch(tflops_data: dict, save_path: str = None):
    """
    绘制TFLOPS vs Batch Size的图表
    
    tflops_data: {
        'batch_sizes': [1, 16, 32, ...],
        'implementations': {
            'impl_name1': [tflops_values1, ...],
            'impl_name2': [tflops_values2, ...],
            ...
        },
        'title': 'Title',
        'xlabel': 'Batch Size',
        'ylabel': 'TFLOPS'
    }
    """
    plt.figure(figsize=(12, 8))
    
    batch_sizes = tflops_data['batch_sizes']
    
    for impl_name, tflops_values in tflops_data['implementations'].items():
        plt.plot(batch_sizes, tflops_values, marker='o', label=impl_name, linewidth=2)
    
    plt.xlabel(tflops_data.get('xlabel', 'Batch Size'), fontsize=12)
    plt.ylabel(tflops_data.get('ylabel', 'TFLOPS'), fontsize=12)
    plt.title(tflops_data.get('title', 'TFLOPS vs Batch Size'), fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴为对数刻度（如果batch sizes跨度较大）
    if max(batch_sizes) / min(batch_sizes) > 10:
        plt.xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_multiple_shapes(tflops_data_dict: dict, save_path: str = None):
    """
    绘制多个shape的TFLOPS对比图
    
    tflops_data_dict: {
        'shape1': {
            'batch_sizes': [...],
            'implementations': {...}
        },
        'shape2': {...},
        ...
    }
    """
    num_shapes = len(tflops_data_dict)
    fig, axes = plt.subplots(1, num_shapes, figsize=(6*num_shapes, 6))
    
    if num_shapes == 1:
        axes = [axes]
    
    for idx, (shape_name, data) in enumerate(tflops_data_dict.items()):
        ax = axes[idx]
        batch_sizes = data['batch_sizes']
        
        for impl_name, tflops_values in data['implementations'].items():
            ax.plot(batch_sizes, tflops_values, marker='o', label=impl_name, linewidth=2)
        
        ax.set_xlabel(data.get('xlabel', 'Batch Size'), fontsize=11)
        ax.set_ylabel(data.get('ylabel', 'TFLOPS'), fontsize=11)
        ax.set_title(shape_name, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if max(batch_sizes) / min(batch_sizes) > 10:
            ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def run_and_collect_tflops(dtype: torch.dtype,
                           MKNs: Iterable[Tuple[int, int, int]],
                           batch_dim_index: int = 0) -> Tuple[List[TMeasurement], dict]:
    """
    运行benchmark并收集TFLOPS数据
    
    batch_dim_index: 哪个维度是batch size (0 for M, 1 for K, 2 for N)
    """
    results = []
    timers_with_info = []
    
    # 收集所有测量结果
    for m, k, n in MKNs:
        timers = bench(dtype, m, k, n, f"scaled-{dtype}-gemm",
                       f"MKN=({m}x{k}x{n})")
        print_timers(timers)
        results.extend(timers)
        
        # 为每个timer添加维度信息
        for timer in timers:
            timers_with_info.append((timer, m, k, n))
    
    # 打印TFLOPS表格
    print_tflops_table(timers_with_info)
    
    # 组织TFLOPS数据用于绘图
    # 假设batch size是变化的维度，其他维度固定
    tflops_by_batch = {}
    batch_sizes = sorted(set([info[1] if batch_dim_index == 0 else 
                              info[2] if batch_dim_index == 1 else 
                              info[3] for info in timers_with_info]))
    
    for measurement, m, k, n in timers_with_info:
        impl_name = measurement.description
        if impl_name not in tflops_by_batch:
            tflops_by_batch[impl_name] = {}
        
        batch_size = m if batch_dim_index == 0 else (k if batch_dim_index == 1 else n)
        tflops = extract_tflops_from_measurement(measurement, m, k, n)
        tflops_by_batch[impl_name][batch_size] = tflops
    
    # 转换为绘图格式
    plot_data = {
        'batch_sizes': batch_sizes,
        'implementations': {}
    }
    
    for impl_name, batch_tflops in tflops_by_batch.items():
        plot_data['implementations'][impl_name] = [batch_tflops[bs] for bs in batch_sizes]
    
    return results, plot_data

# output makers
def make_output(data: Iterable[TMeasurement],
                MKNs: Iterable[Tuple[int, int, int]],
                base_description: str,
                timestamp=None):

    print(f"== All Results {base_description} ====")
    print_timers(data)
    
    # 计算并打印TFLOPS
    timers_with_info = []
    for timer, (m, k, n) in zip(data, MKNs):
        timers_with_info.append((timer, m, k, n))
    print_tflops_table(timers_with_info)

    # pickle all the results
    timestamp = int(time.time()) if timestamp is None else timestamp
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)

# argparse runners

def run_square_bench(args):
    dim_sizes = list(
        range(args.dim_start, args.dim_end + 1, args.dim_increment))
    MKNs = list(zip(dim_sizes, dim_sizes, dim_sizes))
    
    results, plot_data = run_and_collect_tflops(args.dtype, MKNs, batch_dim_index=0)
    
    # 绘图
    plot_data['title'] = f'Square GEMM TFLOPS vs Dimension Size ({args.dtype})'
    plot_data['xlabel'] = 'Dimension Size (M=K=N)'
    plot_data['ylabel'] = 'TFLOPS'
    
    # 保存图表
    save_path = f"square_bench_{args.dtype}_{int(time.time())}.png"
    plot_tflops_vs_batch(plot_data, save_path)
    
    make_output(results, MKNs, f"square_bench-{args.dtype}")

def run_range_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end, args.dim_increment))
    n = len(dim_sizes)
    Ms = [args.m_constant] * n if args.m_constant is not None else dim_sizes
    Ks = [args.k_constant] * n if args.k_constant is not None else dim_sizes
    Ns = [args.n_constant] * n if args.n_constant is not None else dim_sizes
    MKNs = list(zip(Ms, Ks, Ns))
    
    results, plot_data = run_and_collect_tflops(args.dtype, MKNs, batch_dim_index=0)
    
    # 确定变化的维度
    varying_dim = None
    if args.m_constant is None:
        varying_dim = "M"
    elif args.k_constant is None:
        varying_dim = "K"
    elif args.n_constant is None:
        varying_dim = "N"
    else:
        varying_dim = "All dimensions vary"
    
    plot_data['title'] = f'Range GEMM TFLOPS vs {varying_dim} ({args.dtype})'
    plot_data['xlabel'] = varying_dim if varying_dim != "All dimensions vary" else "Dimension"
    plot_data['ylabel'] = 'TFLOPS'
    
    save_path = f"range_bench_{args.dtype}_{int(time.time())}.png"
    plot_tflops_vs_batch(plot_data, save_path)
    
    make_output(results, MKNs, f"range_bench-{args.dtype}")

def run_model_bench(args):

    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    def model_shapes(model_name: str, tp_size: int) -> List[Tuple[int, int]]:
        KNs = []
        for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model_name]):
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            KNs.append(KN)
        return KNs

    model_bench_data = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    
    # 收集所有模型的数据用于绘图
    all_plot_data = {}
    
    for model, tp_size in models_tps:
        Ms = args.batch_sizes
        KNs = model_shapes(model, tp_size)
        MKNs = []
        shape_names = []
        
        for m in Ms:
            for k, n in KNs:
                MKNs.append((m, k, n))
                shape_names.append(f"K={k},N={n}")
        
        print(f"\nBenchmarking {model} with TP={tp_size}")
        results, plot_data = run_and_collect_tflops(args.dtype, MKNs, batch_dim_index=0)
        
        # 存储结果
        model_key = f"{model}_TP{tp_size}"
        model_bench_data.append((model_key, results))
        
        # 为每个(K,N)形状创建单独的数据集
        # 注意：这里简化处理，假设所有(M,K,N)中K和N是常数
        unique_shapes = {}
        for (m, k, n), timer in zip(MKNs, results):
            shape_key = f"K={k},N={n}"
            if shape_key not in unique_shapes:
                unique_shapes[shape_key] = {
                    'batch_sizes': [],
                    'implementations': {}
                }
            
            if shape_key not in all_plot_data:
                all_plot_data[shape_key] = {
                    'batch_sizes': [],
                    'implementations': {}
                }
            
            # 收集每个实现的TFLOPS
            tflops = extract_tflops_from_measurement(timer, m, k, n)
            impl_name = timer.description
            
            if impl_name not in unique_shapes[shape_key]['implementations']:
                unique_shapes[shape_key]['implementations'][impl_name] = []
            unique_shapes[shape_key]['implementations'][impl_name].append(tflops)
            unique_shapes[shape_key]['batch_sizes'].append(m)
            
            # 添加到全局数据
            if impl_name not in all_plot_data[shape_key]['implementations']:
                all_plot_data[shape_key]['implementations'][impl_name] = []
            all_plot_data[shape_key]['implementations'][impl_name].append(tflops)
            all_plot_data[shape_key]['batch_sizes'].append(m)
        
        # 为每个形状绘图
        for shape_key, data in unique_shapes.items():
            # 按batch size排序
            sorted_indices = np.argsort(data['batch_sizes'])
            data['batch_sizes'] = [data['batch_sizes'][i] for i in sorted_indices]
            for impl_name in data['implementations']:
                data['implementations'][impl_name] = [data['implementations'][impl_name][i] for i in sorted_indices]
            
            data['title'] = f'{model} (TP={tp_size}) - {shape_key}'
            data['xlabel'] = 'Batch Size (M)'
            data['ylabel'] = 'TFLOPS'
            
            save_path = f"model_bench_{model}_TP{tp_size}_{shape_key}_{int(time.time())}.png"
            plot_tflops_vs_batch(data, save_path)
    
    # 如果只有一个形状，绘制组合图
    if len(all_plot_data) == 1:
        shape_key = list(all_plot_data.keys())[0]
        plot_data = all_plot_data[shape_key]
        plot_data['title'] = f'Model Benchmark TFLOPS vs Batch Size ({args.dtype})'
        plot_data['xlabel'] = 'Batch Size (M)'
        plot_data['ylabel'] = 'TFLOPS'
        save_path = f"model_bench_combined_{args.dtype}_{int(time.time())}.png"
        plot_tflops_vs_batch(plot_data, save_path)
    elif len(all_plot_data) > 1:
        # 多个形状，绘制子图
        for shape_key, data in all_plot_data.items():
            # 按batch size排序
            sorted_indices = np.argsort(data['batch_sizes'])
            data['batch_sizes'] = [data['batch_sizes'][i] for i in sorted_indices]
            for impl_name in data['implementations']:
                data['implementations'][impl_name] = [data['implementations'][impl_name][i] for i in sorted_indices]
        
        plot_multiple_shapes(all_plot_data, f"model_bench_multi_shape_{args.dtype}_{int(time.time())}.png")

    # Print all results
    for model_key, data in model_bench_data:
        print(f"== Results {args.dtype} {model_key} ====")
        print_timers(data)
        # 打印TFLOPS
        timers_with_info = []
        # 需要重新获取M,K,N信息
        # 这里简化处理

    timestamp = int(time.time())

    all_data = []
    for _, d in model_bench_data:
        all_data.extend(d)
    # pickle all data
    with open(f"model_bench-{args.dtype}-{timestamp}.pkl", "wb") as f:
        pkl.dump(all_data, f)


if __name__ == '__main__':

    def to_torch_dtype(dt):
        if dt == "int8":
            return torch.int8
        if dt == "fp8":
            return torch.float8_e4m3fn
        raise ValueError("unsupported dtype")

    parser = FlexibleArgumentParser(
        description="""
Benchmark Cutlass GEMM.

    To run square GEMMs:
        python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 square_bench --dim-start 128 --dim-end 512 --dim-increment 64
    
    To run constant N and K and sweep M:
        python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 range_bench --dim-start 128 --dim-end 512 --dim-increment 64 --n-constant 16384 --k-constant 16384
    
    To run dimensions from a model:
        python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 model_bench --models meta-llama/Llama-2-7b-hf --batch-sizes 16 --tp-sizes 1
    
    Output:
        - a .pkl file, that is a list of raw torch.benchmark.utils.Measurements for the pytorch and cutlass implementations for the various GEMMs.
        - PNG plots showing TFLOPS vs batch size
            """,  # noqa: E501
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--dtype",
                        type=to_torch_dtype,
                        required=True,
                        help="Available options are ['int8', 'fp8']")
    subparsers = parser.add_subparsers(dest="cmd")

    square_parser = subparsers.add_parser("square_bench")
    square_parser.add_argument("--dim-start", type=int, required=True)
    square_parser.add_argument("--dim-end", type=int, required=True)
    square_parser.add_argument("--dim-increment", type=int, required=True)
    square_parser.set_defaults(func=run_square_bench)

    range_parser = subparsers.add_parser("range_bench")
    range_parser.add_argument("--dim-start", type=int, required=True)
    range_parser.add_argument("--dim-end", type=int, required=True)
    range_parser.add_argument("--dim-increment", type=int, required=True)
    range_parser.add_argument("--m-constant", type=int, default=None)
    range_parser.add_argument("--n-constant", type=int, default=None)
    range_parser.add_argument("--k-constant", type=int, default=None)
    range_parser.set_defaults(func=run_range_bench)

    model_parser = subparsers.add_parser("model_bench")
    model_parser.add_argument("--models",
                              nargs="+",
                              type=str,
                              default=None)
    model_parser.add_argument("--tp-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_TP_SIZES)
    model_parser.add_argument("--batch-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_BATCH_SIZES)
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()
    args.func(args)