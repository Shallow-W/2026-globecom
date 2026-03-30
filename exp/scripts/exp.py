"""
实验运行器 + 结果输出模块
"""

import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cfg import Config
from data import load_architecture_tables
from topo import Topology
from sim import Simulator
from algo import ALGORITHM_MAP


def run_single_algo(config: Config, topo: Topology, tables: Dict,
                    algo_name: str, traffic_mode: str) -> dict:
    """运行单个算法"""
    deploy_cls, route_cls = ALGORITHM_MAP[algo_name]
    np.random.seed(config.seed)
    import random; random.seed(config.seed)
    sim = Simulator(config, topo, tables, deploy_cls, route_cls, traffic_mode)
    return sim.run()


def run_experiment(config: Config, topology_scale: str = 'small',
                  traffic_mode: str = 'steady',
                  algorithms=None) -> dict:
    """运行一组算法对比实验"""
    tables = load_architecture_tables(config.data_path)
    if algorithms is None:
        algorithms = list(ALGORITHM_MAP.keys())

    results = {}
    for algo_name in algorithms:
        print(f"  [{algo_name}] ...", end=' ', flush=True)
        t0 = time.time()
        topo = Topology(config, scale=topology_scale)
        r = run_single_algo(config, topo, tables, algo_name, traffic_mode)
        results[algo_name] = r
        print(f"lat={r['avg_latency_ms']:.1f}ms succ={r['success_rate']:.3f} "
              f"sla={r['sla_violation_rate']:.3f} ({time.time()-t0:.0f}s)")

    return results


def save_results(results: dict, output_dir: str, filename: str):
    """保存主指标到 CSV"""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for algo, r in results.items():
        row = {
            'algorithm': algo,
            'avg_latency_ms': round(r['avg_latency_ms'], 2),
            'success_rate': round(r['success_rate'], 4),
            'sla_violation_rate': round(r['sla_violation_rate'], 4),
            'throughput': round(r['throughput'], 2),
            'avg_pull_delay_ms': round(r.get('avg_pull_delay_ms', 0), 2),
            'pull_count': r['pull_count'],
            'node_utilization_avg': round(r['node_utilization_avg'], 4),
            'total_requests': r['total_requests'],
        }
        for task, perf in r['task_performance'].items():
            row[f'perf_{task}'] = round(perf, 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, filename), index=False)
    return df


def save_latencies(results: dict, output_dir: str, prefix: str):
    """保存时延详细数据（JSON，供后续分析用）"""
    os.makedirs(output_dir, exist_ok=True)
    for algo, r in results.items():
        lat_data = {
            'algorithm': algo,
            'avg': r['avg_latency_ms'],
            'p50': float(np.median(r['latencies'])) if r['latencies'] else 0,
            'p95': float(np.percentile(r['latencies'], 95)) if r['latencies'] else 0,
            'p99': float(np.percentile(r['latencies'], 99)) if r['latencies'] else 0,
        }
        with open(os.path.join(output_dir, f'{prefix}_{algo}_latency.json'), 'w') as f:
            json.dump(lat_data, f, indent=2)


def plot_comparison_bar(results: dict, output_dir: str, filename: str):
    """柱状图对比（对齐论文 A 风格）"""
    algos = list(results.keys())
    n = len(algos)
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    metrics = ['avg_latency_ms', 'success_rate', 'sla_violation_rate', 'throughput']
    titles = ['Avg E2E Latency (ms)', 'Success Rate', 'SLA Violation Rate', 'Throughput (req/s)']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        vals = [results[a][metric] for a in algos]
        bars = ax.bar(algos, vals, color=colors)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        for bar, v in zip(bars, vals):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * max(vals) if vals else 0.01,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


def plot_cdf(results: dict, output_dir: str, filename: str):
    """时延 CDF 分布（对齐论文 B 风格）"""
    fig, ax = plt.subplots(figsize=(8, 5))
    algos = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(algos)))

    for (algo, r), color in zip(results.items(), colors):
        x, y = r['latencies_cdf_x'], r['latencies_cdf_y']
        ax.plot(y, x, label=algo, color=color, linewidth=2)

    ax.set_xlabel('E2E Latency (ms)', fontsize=11)
    ax.set_ylabel('CDF', fontsize=11)
    ax.set_title('Latency CDF Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


def plot_task_performance_heatmap(results: dict, output_dir: str, filename: str):
    """分任务真实性能热力图"""
    tasks = ['class_scene', 'class_object', 'room_layout', 'jigsaw',
             'segmentsemantic', 'normal', 'autoencoder']
    algos = list(results.keys())

    matrix = []
    for algo in algos:
        row = [results[algo]['task_performance'].get(t, 0) for t in tasks]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, aspect='auto', cmap='YlGn')
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=9)
    ax.set_title('Per-Task Actual Performance (task_final_performance)', fontsize=12)

    for i in range(len(algos)):
        for j in range(len(tasks)):
            val = matrix[i, j]
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


def print_summary(results: dict, title: str = ""):
    """打印汇总表"""
    print(f"\n{'='*80}")
    if title:
        print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Algorithm':<20} {'Latency(ms)':<14} {'SuccessRate':<14} {'SLAViol':<12} {'Throughput':<12}")
    print(f"{'-'*80}")
    for algo in sorted(results.keys()):
        r = results[algo]
        print(f"{algo:<20} {r['avg_latency_ms']:<14.2f} {r['success_rate']:<14.3f} "
              f"{r['sla_violation_rate']:<12.3f} {r['throughput']:<12.1f}")


# ============================================================
# 主实验入口
# ============================================================

def main():
    cfg = Config()
    cfg.n_slots = 100
    os.makedirs(cfg.output_dir, exist_ok=True)

    tables = load_architecture_tables(cfg.data_path)
    print("=" * 70)
    print("云边协同部署与路由仿真实验")
    print("算法: OURS (globecom.tex) | 基线: HEURISTIC_A(GMDA) | GREEDY_B(Greedy)")
    print("=" * 70)

    experiments = [
        ('small_steady',  'small',  'steady'),
        ('small_tidal',   'small',  'tidal'),
        ('small_burst',   'small',  'burst'),
        ('large_tidal',   'large',  'tidal'),
    ]

    for exp_name, scale, mode in experiments:
        print(f"\n### 实验: {exp_name} (拓扑={scale}, 流量={mode})")
        results = run_experiment(cfg, scale, mode)
        save_results(results, cfg.output_dir, f'results_{exp_name}.csv')
        save_latencies(results, cfg.output_dir, exp_name)
        plot_comparison_bar(results, cfg.output_dir, f'fig_bar_{exp_name}.png')
        plot_cdf(results, cfg.output_dir, f'fig_cdf_{exp_name}.png')
        print_summary(results, exp_name)

    print("\n实验完成！结果保存在:", cfg.output_dir)


if __name__ == '__main__':
    main()
