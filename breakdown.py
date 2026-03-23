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
parser.add_argument( "--kk", type=int, default=8192, help="K dimension for the matmul workload.")
parser.add_argument( "--nn", type=int, default=57344, help="N dimension for the matmul workload.")
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
def get_figure9_configs():
    configs = []
    for m in [
        1
    ]:
        for k, n in [
            (kk, nn),
        ]:
            for (b_dtype, runners) in [
                ('float16', ['torch-f16']),
                ('uint4b', ['cutlass', 'triton', 'mutis',  'tilelp_acc_in_register',  'tilelp_evict',
                            'tilelp_no_acc_in_register_no_opt_micro_kernel', 'tilelp']),
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
    Filter to the single workload (m=1, k=8192, n=8192) and compute speedup
    relative to the torch-f16 baseline.
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

    df['speedup'] = baseline_latency / df['latency']

    # exclude the baseline itself from the bar chart
    runners_to_show = ['triton', 'mutis', 'tilelp_acc_in_register', 'tilelp_evict',
                        'tilelp_no_acc_in_register_no_opt_micro_kernel',  'tilelp']
    df = df[df['runner'].isin(runners_to_show)]
    return df


def plot_latency(df: DataFrame, baseline_latency: float, out_fname: str):
    """
    Bar chart showing latency (ms) for each runner, with a dashed horizontal
    line for the torch-f16 baseline.
    """
    from aesthetic import colors, executor2color, executor2label, ranked_executors
    from utils import fill_color

    runners_order = [r for r in ranked_executors if r in df['runner'].values]
    df = df.set_index('runner').loc[runners_order].reset_index()

    fig, ax = plt.subplots(figsize=(10, 4.5))

    bar_width = 0.6
    x_positions = np.arange(len(runners_order))

    for i, runner in enumerate(runners_order):
        row = df[df['runner'] == runner].iloc[0]
        color = fill_color(colors[executor2color[runner]][0])
        bar = ax.bar(
            x_positions[i],
            row['latency'],
            width=bar_width,
            color=color,
            edgecolor='#333333',
            linewidth=0.8,
            label=executor2label.get(runner, runner),
        )
        ax.bar_label(bar, fmt='%.3f', padding=3, fontsize=9)

    # dashed line for torch-f16 baseline
    ax.axhline(
        y=baseline_latency,
        color='#555555',
        linewidth=1.2,
        linestyle='--',
        label='Cutlass (FP16) baseline',
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [executor2label.get(r, r) for r in runners_order],
        rotation=20,
        ha='right',
        fontsize=10,
    )
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'Ablation Breakdown (m=1, k={kk}, n={nn}, W4A16, group=128)', fontsize=11)
    ax.set_ylim(0, max(baseline_latency, df['latency'].max()) * 1.25)
    ax.tick_params(axis='x', which='both', length=0)

    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9, frameon=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    fig.savefig(out_fname, bbox_inches='tight')
    print(f'Saved figure to {out_fname}')


def plot_speedup(df: DataFrame, out_fname: str):
    """
    Bar chart showing speedup (x) for each runner, relative to torch-f16.
    """
    runners_order = [r for r in ranked_executors if r in df['runner'].values]
    df = df.set_index('runner').loc[runners_order].reset_index()

    fig, ax = plt.subplots(figsize=(10, 4.5))

    bar_width = 0.6
    x_positions = np.arange(len(runners_order))

    for i, runner in enumerate(runners_order):
        row = df[df['runner'] == runner].iloc[0]
        color = fill_color(colors[executor2color[runner]][0])
        bar = ax.bar(
            x_positions[i],
            row['speedup'],
            width=bar_width,
            color=color,
            edgecolor='#333333',
            linewidth=0.8,
            label=executor2label.get(runner, runner),
        )
        ax.bar_label(bar, fmt='%.2fx', padding=3, fontsize=9)

    ax.axhline(
        y=1.0,
        color='#555555',
        linewidth=1.2,
        linestyle='--',
        label='Cutlass (FP16) baseline (1.0x)',
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [executor2label.get(r, r) for r in runners_order],
        rotation=20,
        ha='right',
        fontsize=10,
    )
    ax.set_ylabel('Speedup (x)')
    ax.set_title(f'Ablation Breakdown (m=1, k={kk}, n={nn}, W4A16, group=128)', fontsize=11)
    ax.set_ylim(0, max(1.0, df['speedup'].max()) * 1.25)
    ax.tick_params(axis='x', which='both', length=0)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9, frameon=True)

    fig.tight_layout()
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
 
    baseline_latency = df[(df['runner'] == baseline) & (df['device'] == gpu)]['latency'].values[0]
    df_plot = process(df, gpu=gpu)

    # plot_latency(df_plot, baseline_latency=baseline_latency,
    #              out_fname=os.path.join(results_dir, 'breakdown_latency.pdf'))
    plat = 'h100' if 'h100' in gpu.lower() else '4090'
    plot_speedup(df_plot, out_fname=os.path.join(results_dir, f'breakdown_speedup_{nn}_{kk}_{plat}.pdf'))


if __name__ == '__main__':
    main()
