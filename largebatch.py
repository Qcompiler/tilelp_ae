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

import argparse
parser = argparse.ArgumentParser(description='Run large batch experiments')
args = parser.parse_args()
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 12

baseline = 'cutlass'

# supported_runners = ['cutlass', 'marlin', 'tilelp', 'triton', 'mutis', 'gemlite']

# baseline = 'torch-f16'
supported_runners = [  'cutlass',    'triton', 'mutis', 'gemlite', 'tilelp']
import torch
arch = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
print(f"GPU Architecture: {arch}")
if arch < 90:
    supported_runners.append('bitblas')


M_VALUES = range(128, 2049, 128)
M_VALUES =  [64] + list(M_VALUES)

def get_figure9_configs():
    configs = []
    for m in M_VALUES:
        for k, n in [
            # (8192, 8192),
            # (12288, 8192),
            (16384, 16384),
            # (28672, 8192),
            # (8192, 16384),
            # (8192, 57344),
            # (4096, 4096),
        ]:
            for (b_dtype, runners) in [
                ('float16', ['torch-f16']),
                ('uint4b', supported_runners),
                # ('uint4b', ['cutlass',  'marlin', 'tilelp', 'triton',   'mutis']),
            ]:
                for runner in runners:
                    for a_dtype in ['float16']:
                        if data_type(b_dtype).is_integer():
                            group_size = 128
                        else:
                            group_size = -1
                        if not MatmulLayer.supports(runner_name=runner, a_dtype=a_dtype, b_dtype=b_dtype):
                            continue
                        
                        configs.append([runner, a_dtype, b_dtype, group_size, m, k, n])
    return configs


def run_experiments():
    configs = get_figure9_configs()
    df = bench_configs(configs, warmup=50, repeat=200)
    pd.options.display.max_rows = None
    return df


def load_df(results_dir: str) -> DataFrame:
    txt_path = os.path.join(results_dir, 'figure9.txt')
    df = pd.read_csv(txt_path, sep=r'\s+', engine='python')
    return df


def process(df: DataFrame, gpu: str) -> DataFrame:
    """Process dataframe for selected m values."""
    df = df[df['device'] == gpu].copy()
    df = df[df['m'].isin(M_VALUES)].copy()
    

    # Build shape label: K x N
    df['shape'] = df.apply(lambda r: f"{int(r['k'])}x{int(r['n'])}", axis=1)
    
    # Calculate baseline latency
    baseline_df = df[df['runner'] == baseline]
    baseline_lat = {}
    for _, row in baseline_df.iterrows():
        baseline_lat[(int(row['k']), int(row['n']))] = row['latency']
    
    df['baseline'] = df.apply(
        lambda r: baseline_lat.get((int(r['k']), int(r['n'])), np.nan), axis=1
    )
    df['speedup'] = df['baseline'] / df['latency']
    df['tflops'] = 2.0 * df['m'] * df['k'] * df['n'] / (df['latency'] * 1e-3) / 1e12
    
    # Filter runners - 确保名称与颜色映射一致
    runners = [ 'torch-f16', 'triton', 'bitblas', 'marlin', 'mutis', 'tilelp', 'gemlite']  # 移除 quant-llm
    df = df[df['runner'].isin(runners)]
    
    # 修正数据类型过滤
    # df = df[~((df['runner'] == 'bitblas') & (df['b_dtype'] == 'uint4b'))]  # 改为 uint4b
    
    return df

def plot_by_m(df: DataFrame, out_fname: str):
    """Plot TFLOPS vs m for each runner."""

    m_values = M_VALUES
    runners = sorted(
        [e for e in df['runner'].unique() if e in ranked_executors],
        key=lambda x: ranked_executors.index(x)
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for executor in runners:
        df_e = df[df['runner'] == executor]
        tflops = []
        for m in m_values:
            row = df_e[df_e['m'] == m]
            tflops.append(row['tflops'].values[0] if len(row) > 0 else np.nan)

        ax.plot(
            m_values,
            tflops,
            marker='o',
            linewidth=2,
            markersize=5,
            color=fill_color(colors[executor2color[executor]][0]),
            label=executor2label.get(executor, executor),
        )

    ax.set_xlabel('m')
    ax.set_ylabel('TFLOPS')
    xtick_step = 2
    sparse_xticks = list(m_values[::xtick_step])
    if sparse_xticks[-1] != m_values[-1]:
        sparse_xticks.append(m_values[-1])
    ax.set_xticks(sparse_xticks)
    ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.98), loc='center',
               ncol=max(1, len(runners)) // 2, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    fig.savefig(out_fname, bbox_inches='tight')
    print(f'Saved to {out_fname}')

def main():
    results_dir = os.environ.get('TILUS_ARTIFACT_RESULTS_DIR', './results')
    
    # Load or run experiments
    txt_path = os.path.join(results_dir, 'figure9.txt')
    if os.path.exists(txt_path):
        print(f"Loading existing results from {txt_path}")
        df_raw = load_df(results_dir)
    else:
        print("Running experiments...")
        df_raw = run_experiments()
        os.makedirs(results_dir, exist_ok=True)
        with open(txt_path, 'w') as f:
            f.write(df_raw.to_string(index=False))
    
    # De-duplicate
    df_raw = df_raw.drop_duplicates()
    
    # Get GPU
    gpus = list(df_raw['device'].unique())
    if len(gpus) != 1:
        raise ValueError(f'Expected exactly one GPU, got: {gpus}')
    gpu = gpus[0]
    
    df_processed = process(df_raw, gpu=gpu)
    out_path = os.path.join(results_dir, f'gpu_{gpu}_batch_by_m_tflops.pdf')
    plot_by_m(df_processed, out_fname=out_path)


if __name__ == '__main__':
    main()
