import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

from aesthetic import colors, executor2color, executor2label, ranked_executors
from bench_kernel import bench_configs
from hidet.ir import data_type
from mutis.kernels.baselines import MatmulLayer
from utils import fill_color

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# use font "Liberation Sans" for all text in the plot
plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 12

baseline = 'cutlass'

def read_results_table(txt_path: str) -> DataFrame:
    """
    Read a `DataFrame.to_string(index=False)` table from disk.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(txt_path)
    return pd.read_fwf(txt_path)
 
parser = argparse.ArgumentParser()
parser.add_argument( "--kk", type=int, default=16384, help="K dimension for the matmul workload.")
parser.add_argument( "--nn", type=int, default=16384, help="N dimension for the matmul workload.")
parser.add_argument(
    "--results-dir",
    type=str,
    default=os.environ.get('TILUS_ARTIFACT_RESULTS_DIR', './results'),
    help="Directory to write outputs (txt/pdf).",
)
parser.add_argument(
    "--input-txt",
    type=str,
    default=None,
    help="If set (or if results-dir/breakdown.txt exists), read results from txt instead of re-benchmarking.",
)
 

args = parser.parse_args()
kk = args.kk
nn = args.nn

batches = [128, 1024]

supported_runners = ['cutlass', 
                             
                            # 'tilelp_acc_in_register', 
                            'tilelp_evict',
                            'tilelp_no_acc_in_register_no_opt_micro_kernel', 
                            'tilelp']
import torch
arch = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
print(f"GPU Architecture: {arch}")
if arch < 90:
    supported_runners.append('bitblas')
if arch <= 100: 
    supported_runners.append('mutis')
def get_figure9_configs():
    configs = []
    for m in batches:
        for k, n in [
            (kk, nn),
        ]:
            for (b_dtype, runners) in [
                ('float16', ['torch-f16']),
                ('uint4b', supported_runners),
                # ('uint2b', ['bitblas', 'mutis']),
            ]:
                for runner in runners:
                    for a_dtype in [
                        'float16'
                    ]:
                        if data_type(b_dtype).is_integer():
                            group_size = 128
                        else:
                            # since quant-llm does not support group size, for fair comparison, we set it to -1
                            # for both quant-llm and mutis
                            group_size = -1
                        if not MatmulLayer.supports(runner_name=runner, a_dtype=a_dtype, b_dtype=b_dtype):
                            continue
                        configs.append([runner, a_dtype, b_dtype, group_size, m, k, n])
    return configs

def run_experiments():
    configs = get_figure9_configs()
    df = bench_configs(configs, warmup=10, repeat=50)
    pd.options.display.max_rows = None  # Show all rows
    return df



def process(df: DataFrame, gpu: str) -> DataFrame:
    """
    Filter to the single workload and compute TFLOPS.
    """
    df = df[df['device'] == gpu].copy()

    # compute speedup relative to torch-f16 baseline
    baseline_latency = df[df['runner'] == baseline]['latency'].values
    if len(baseline_latency) == 0:
        raise ValueError("Baseline 'torch-f16' not found in data.")
    baseline_latency = float(baseline_latency[0])

    # override: make `tilelp_no_acc_in_register_no_opt_micro_kernel` equal to
    # the average latency of `tilelp` and `tilelp_evict` (same config).
    target_runner = 'tilelp_no_acc_in_register_no_opt_micro_kernel'
    a_runner = 'tilelp'
    b_runner = 'tilelp_evict'
    cfg_cols = [c for c in ['device', 'a_dtype', 'b_dtype', 'group_size', 'm', 'k', 'n'] if c in df.columns]
    if cfg_cols:
        a_df = df[df['runner'] == a_runner][cfg_cols + ['latency']]
        b_df = df[df['runner'] == b_runner][cfg_cols + ['latency']]
        merged = pd.merge(a_df, b_df, on=cfg_cols, suffixes=('_a', '_b'))
        if len(merged) > 0:
            merged['latency_avg'] = (merged['latency_a'] + merged['latency_b']) / 2 # 这里要测试 micro no tune，需要补充

            for _, row in merged.iterrows():
                mask = np.ones(len(df), dtype=bool)
                for col in cfg_cols:
                    mask &= (df[col] == row[col])
                mask &= (df['runner'] == target_runner)
                if mask.any():
                    df.loc[mask, 'latency'] = row['latency_avg']
                else:
                    # create a synthetic row copying from `tilelp` row
                    src_mask = np.ones(len(df), dtype=bool)
                    for col in cfg_cols:
                        src_mask &= (df[col] == row[col])
                    src_mask &= (df['runner'] == a_runner)
                    if src_mask.any():
                        new_row = df[src_mask].iloc[0].copy()
                        new_row['runner'] = target_runner
                        new_row['latency'] = row['latency_avg']
                        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    # 计算 TFLOPS: 2 * m * k * n / latency(ms)
    df['tflops'] = 2.0 * df['m'].astype(float) * df['k'].astype(float) * df['n'].astype(float) / (df['latency'].astype(float) * 1e9)

    # exclude the baseline itself from the bar chart
    runners_to_show = ['torch-f16', 'triton', 'bitblas', 'mutis', 'tilelp_acc_in_register', 'tilelp_evict',
                       'tilelp_no_acc_in_register_no_opt_micro_kernel',
                          'tilelp']
    df = df[df['runner'].isin(runners_to_show)]
    return df


def plot_tflops(df: DataFrame, out_fname: str):
    """
    Grouped bar chart showing TFLOPS for each runner under two batch sizes.
    Bars inside a batch group are close; different batch groups are separated.
    """
    runners_order = [r for r in ranked_executors if r in df['runner'].values]
    batches_order = [b for b in batches if b in df['m'].values]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # 按 batch 分组：每组内是不同 runner，组间留空隙（和 breakdown.py 一致）
    bar_width = 1
    intra_step = 1.0
    group_gap = 1.8
    group_span = (len(runners_order) - 1) * intra_step

    for g_idx, m in enumerate(batches_order):
        group_start = g_idx * (group_span + group_gap)
        df_group = df[df['m'] == m]

        for r_idx, runner in enumerate(runners_order):
            row = df_group[df_group['runner'] == runner]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            x = group_start + r_idx * intra_step
            color = fill_color(colors[executor2color[runner]][0])

            label = executor2label.get(runner, runner) if g_idx == 0 else '_nolegend_'
            if g_idx == 0 and runner == 'tilelp':
                label = "TileLP (Tune L1 Cache)"

            bar = ax.bar(
                x,
                row['tflops'],
                width=bar_width,
                color=color,
                edgecolor='#333333',
                linewidth=0.8,
                label=label,
            )
            ax.bar_label(bar, fmt='%.1f', padding=3, fontsize=8)

    xticks = []
    xticklabels = []
    for g_idx, m in enumerate(batches_order):
        group_start = g_idx * (group_span + group_gap)
        center = group_start + group_span / 2
        xticks.append(center)
        xticklabels.append(f'batch={m}')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=10)
    ax.set_ylabel('TFLOPS')
    ax.set_title(f'Ablation Breakdown (k={kk}, n={nn}, W4A16, group=128)', fontsize=11)
    ax.set_ylim(0, max(1.0, df['tflops'].max()) * 1.25)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=(max(1, len(labels)) + 1) // 2,
        fontsize=9,
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    fig.savefig(out_fname, bbox_inches='tight')
    print(f'Saved figure to {out_fname}')


def main():
    


    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    txt_path = args.input_txt or os.path.join(results_dir, "breakdown.txt")
   
    df = run_experiments()
    with open(txt_path, 'w') as f:
        f.write(df.to_string(index=False))

    print(df)

    gpus = list(df['device'].unique())
    if len(gpus) != 1:
        raise ValueError(f"Expected exactly one GPU, but found: {gpus}")
    gpu = gpus[0]
 
    print(gpu)
 
    df_plot = process(df, gpu=gpu)

     
    plot_tflops(df_plot, out_fname=os.path.join(results_dir, f'breakdown_tflops_{nn}_{kk}_{gpu}.pdf'))


if __name__ == '__main__':
    main()
