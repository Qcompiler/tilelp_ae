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

plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 12

baseline = 'cutlass'

supported_runners = ['cutlass', 'marlin', 'tilelp', 'triton', 'mutis', ]

import torch
arch = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
print(f"GPU Architecture: {arch}")
if arch < 90:
    supported_runners.append('bitblas')
def get_figure9_configs():
    configs = []
    for m in [1, 2]:
        for k, n in [
            (8192, 8192),
            (12288, 8192),
            (16384, 8192),
            (28672, 8192),
            (8192, 16384),
            (8192, 57344),
            (28672, 8192),
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


def process(df: DataFrame, gpu: str, m_value: int) -> DataFrame:
    """Process dataframe for a specific m value."""
    df = df[df['device'] == gpu].copy()
    df = df[df['m'] == m_value].copy()
    
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
    
    # Filter runners - 确保名称与颜色映射一致
    runners = [ 'torch-f16', 'triton', 'bitblas', 'marlin', 'mutis', 'tilelp', 'gemlite']  # 移除 quant-llm
    df = df[df['runner'].isin(runners)]
    
    # 修正数据类型过滤
    # df = df[~((df['runner'] == 'bitblas') & (df['b_dtype'] == 'uint4b'))]  # 改为 uint4b
    
    return df

def plot_by_m(df: DataFrame, out_fname: str):
    """Plot two subplots, one for each m value, with shapes on x-axis."""
    
    # Get unique m values
    m_values = sorted(df['m'].unique())
    
    fig, axes = plt.subplots(1, 1, figsize=(14, 4))
    axes = [axes]  
    for ax_idx, (ax, m_val) in enumerate(zip(axes, m_values)):
        df_m = df[df['m'] == m_val]
        # df_m = df_m[df_m['b_dtype'] == 'uint4b']
        # Get shapes and executors
        shapes = sorted(df_m['shape'].unique())
        # 确保所有executor都在ranked_executors中
        executors = sorted(
            [e for e in df_m['runner'].unique() if e in ranked_executors], 
            key=lambda x: ranked_executors.index(x)
        )
        
        n_shapes = len(shapes)
        n_exec = len(executors)
        
        # Bar layout
        bar_w = 0.12
        group_w = n_exec * bar_w
        x_positions = np.arange(n_shapes)
        
        # Plot bars for each executor
        for e_idx, executor in enumerate(executors):
            speedups = []
            bar_labels = []
            
            for shape in shapes:
                row = df_m[(df_m['shape'] == shape) & (df_m['runner'] == executor)]
                if len(row) == 0:
                    continue  # 跳过没有数据的形状
                
                spdup = row['speedup'].values[0]
                speedups.append(spdup)
                
                # 为所有runner添加速度up数值标签
                bar_labels.append(f'{spdup:.2f}x')
                
                # # # 为mutis额外添加对比标签
                # if executor == 'mutis':
                    
                #     others = df_m[(df_m['shape'] == shape) & (df_m['runner'] != executor)]
                #     if len(others) > 0:
                #         best_other = others['speedup'].max()
                #         # 可以选择在bar上添加额外标记
                #         ax.text(x_positions[shapes.index(shape)] + e_idx * bar_w - group_w/2 + bar_w/2, 
                #                spdup + 0.1, 
                #                f'vs {best_other/spdup:.2f}', 
                #                ha='center', va='bottom', fontsize=7, rotation=90)
            
            if speedups:  # 只有当有数据时才绘制
                x_pos = x_positions[:len(speedups)] + e_idx * bar_w - group_w/2 + bar_w/2
                bars = ax.bar(
                    x_pos,
                    speedups,
                    width=bar_w,
                    color=fill_color(colors[executor2color[executor]][0]),
                    edgecolor=colors[-1][0],
                    linewidth=0.8,
                    label=executor2label.get(executor, executor),
                )
                ax.bar_label(bars, labels=bar_labels, padding=2, fontsize=10, rotation=90)
        
        # Baseline line
        ax.axhline(y=1, color=colors[-1][-1], linewidth=1, linestyle='--',
                   label='cutlass (fp16)' if ax_idx == 0 else None)
        
        # Formatting
        ymax = df_m['speedup'].max() if len(df_m) > 0 else 2.0
        ax.set_ylim(0, ymax * 1.30)  # 增加一点空间给额外标签
        ax.set_xticks(x_positions)
        ax.set_xticklabels(shapes,  ha='center')
        ax.set_ylabel('Speedup')
        # ax.set_title(f'M = {m_val}', fontsize=13, pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.98), loc='lower center',
               ncol=len(executors) + 1, fontsize=10)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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
    
    # Process for each m value
    df_m1 = process(df_raw, gpu=gpu, m_value=1)
    # df_m2 = process(df_raw, gpu=gpu, m_value=2)
    
    # Combine and plot
    combined_df = pd.concat([df_m1, ])
    out_path = os.path.join(results_dir, f'gpu_{gpu}_figure9_by_m.pdf')
    plot_by_m(combined_df, out_fname=out_path)


if __name__ == '__main__':
    main()
