import argparse
import copy
import itertools
import pickle as pkl
import time
from typing import Callable, Iterable, List, Tuple
from collections import defaultdict

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
import matplotlib.pyplot as plt

from vllm import _custom_ops as ops
from vllm.utils import FlexibleArgumentParser

DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512]
DEFAULT_TP_SIZES = [1]


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


def pytorch_mm_impl(a: torch.tensor, b: torch.tensor, scale_a: torch.tensor,
                    scale_b: torch.tensor,
                    out_dtype: torch.dtype) -> torch.tensor:
    return torch.mm(a, b)


def pytorch_fp8_impl(a: torch.tensor, b: torch.tensor, scale_a: torch.tensor,
                     scale_b: torch.tensor,
                     out_dtype: torch.dtype) -> torch.tensor:
    return torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b,
                            out_dtype=out_dtype)


def pytorch_fp8_impl_fast_accum(a: torch.tensor, b: torch.tensor,
                                scale_a: torch.tensor, scale_b: torch.tensor,
                                out_dtype: torch.dtype) -> torch.tensor:
    return torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b,
                            out_dtype=out_dtype, use_fast_accum=True)


def cutlass_impl(a: torch.tensor, b: torch.tensor, scale_a: torch.tensor,
                 scale_b: torch.tensor,
                 out_dtype: torch.dtype) -> torch.tensor:
    return ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype=out_dtype)


def bench_fn(a: torch.tensor, b: torch.tensor, scale_a: torch.tensor,
             scale_b: torch.tensor, out_dtype: torch.dtype, label: str,
             sub_label: str, fn: Callable, description: str) -> TMeasurement:
    min_run_time = 1
    globals_dict = {
        "a": a, "b": b, "scale_a": scale_a, "scale_b": scale_b,
        "out_dtype": out_dtype, "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(a, b, scale_a, scale_b, out_dtype)",
        globals=globals_dict,
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
    timers.append(
        bench_fn(a.to(dtype=torch.bfloat16, device="cuda"),
                 b.to(dtype=torch.bfloat16, device="cuda"), scale_a, scale_b,
                 torch.bfloat16, label, sub_label, pytorch_mm_impl,
                 "pytorch_bf16_bf16_bf16_matmul-no-scales"))
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
    timers.append(
        bench_fn(a.to(dtype=torch.bfloat16, device="cuda"),
                 b.to(dtype=torch.bfloat16, device="cuda"), scale_a, scale_b,
                 torch.bfloat16, label, sub_label, pytorch_mm_impl,
                 "pytorch_bf16_bf16_bf16_matmul-no-scales"))
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.bfloat16, label, sub_label,
                 pytorch_fp8_impl, "pytorch_fp8_fp8_bf16_scaled_mm"))
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.bfloat16, label, sub_label,
                 pytorch_fp8_impl_fast_accum,
                 "pytorch_fp8_fp8_bf16_scaled_mm_fast_accum"))
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.float16, label, sub_label,
                 pytorch_fp8_impl, "pytorch_fp8_fp8_fp16_scaled_mm"))
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.float16, label, sub_label,
                 pytorch_fp8_impl_fast_accum,
                 "pytorch_fp8_fp8_fp16_scaled_mm_fast_accum"))
    timers.append(
        bench_fn(a, b, scale_a, scale_b, torch.bfloat16, label, sub_label,
                 cutlass_impl, "cutlass_fp8_fp8_bf16_scaled_mm"))
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


def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def run(dtype: torch.dtype,
        MKNs: Iterable[Tuple[int, int, int]]) -> Iterable[TMeasurement]:
    results = []
    for m, k, n in MKNs:
        timers = bench(dtype, m, k, n, f"scaled-{dtype}-gemm",
                       f"MKN=({m}x{k}x{n})")
        print_timers(timers)
        results.extend(timers)
    return results


def measurement_to_tflops(measurement: TMeasurement) -> float:
    """Convert a measurement to TFLOPS."""
    sub = measurement.task_spec.sub_label
    mkn_str = sub.split("MKN=(")[1].rstrip(")")
    m, k, n = [int(x) for x in mkn_str.split("x")]
    time_sec = measurement.median
    flops = 2.0 * m * k * n
    return flops / time_sec / 1e12


def collect_tflops_by_batch(measurements: Iterable[TMeasurement]):
    """Collect TFLOPS grouped by implementation and batch size."""
    by_impl = defaultdict(list)

    for meas in measurements:
        sub = meas.task_spec.sub_label
        desc = meas.task_spec.description
        mkn_str = sub.split("MKN=(")[1].rstrip(")")
        m, k, n = [int(x) for x in mkn_str.split("x")]
        tflops = measurement_to_tflops(meas)
        by_impl[desc].append((m, tflops, k, n))

    out = {}
    for desc, rows in by_impl.items():
        rows = sorted(rows, key=lambda x: x[0])
        dedup = {}
        for m, tflops, k, n in rows:
            dedup[m] = (tflops, k, n)
        out[desc] = [(m, dedup[m][0]) for m in sorted(dedup.keys())]
    return out


def plot_tflops_vs_batch(measurements: Iterable[TMeasurement],
                         out_png: str = "tflops_vs_batch.png"):
    """Plot TFLOPS vs batch size."""
    curves = collect_tflops_by_batch(measurements)

    plt.figure(figsize=(10, 6))
    for desc, points in curves.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=6, label=desc)

    plt.xlabel("Batch size (m)", fontsize=12)
    plt.ylabel("TFLOPS", fontsize=12)
    plt.title("TFLOPS vs Batch Size", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved plot to {out_png}")
    plt.close()


def make_output(data: Iterable[TMeasurement],
                MKNs: Iterable[Tuple[int, int, int]],
                base_description: str,
                timestamp=None):
    print(f"== All Results {base_description} ====")
    print_timers(data)

    timestamp = int(time.time()) if timestamp is None else timestamp
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)

    plot_tflops_vs_batch(data, out_png=f"{base_description}-{timestamp}-tflops-vs-batch.png")


def run_square_bench(args):
    dim_sizes = list(
        range(args.dim_start, args.dim_end + 1, args.dim_increment))
    MKNs = list(zip(dim_sizes, dim_sizes, dim_sizes))
    data = run(args.dtype, MKNs)
    make_output(data, MKNs, f"square_bench-{args.dtype}")


def run_range_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end, args.dim_increment))
    n = len(dim_sizes)
    Ms = [args.m_constant] * n if args.m_constant is not None else dim_sizes
    Ks = [args.k_constant] * n if args.k_constant is not None else dim_sizes
    Ns = [args.n_constant] * n if args.n_constant is not None else dim_sizes
    MKNs = list(zip(Ms, Ks, Ns))
    data = run(args.dtype, MKNs)
    make_output(data, MKNs, f"range_bench-{args.dtype}")


def run_model_bench(args):
    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    model_bench_data = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        Ms = args.batch_sizes
        MKNs = [(m, 8192, 57344) for m in Ms]
        data = run(args.dtype, MKNs)
        model_bench_data.append(data)

    for data, model_tp in zip(model_bench_data, models_tps):
        model, tp_size = model_tp
        print(f"== Results {args.dtype} {model}-TP{tp_size} ====")
        print_timers(data)

    timestamp = int(time.time())
    all_data = []
    for d in model_bench_data:
        all_data.extend(d)
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
        description="""Benchmark Cutlass GEMM with TFLOPS calculation and plotting.

Examples:
  python w8a8_benchmarks_with_tflops.py --dtype fp8 range_bench \\
    --dim-start 64 --dim-end 1025 --dim-increment 64 \\
    --n-constant 57344 --k-constant 8192
        """,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--dtype", type=to_torch_dtype, required=True,
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
    model_parser.add_argument("--models", nargs="+", type=str, default=None)
    model_parser.add_argument("--tp-sizes", nargs="+", type=int,
                              default=DEFAULT_TP_SIZES)
    model_parser.add_argument("--batch-sizes", nargs="+", type=int,
                              default=DEFAULT_BATCH_SIZES)
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()
    args.func(args)
