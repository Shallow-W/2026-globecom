"""
实验运行器 + 结果输出模块

支持实验类型：
1. run_experiment: 基本动态负载实验
2. run_perturbation_experiment: 单变量扰动实验（到达率/请求长度/请求类型数）
3. run_equal_latency_experiment: 等时延资源效率对比
4. run_scale_experiment: 规模扩展实验（小/中/大）
"""

import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
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
# 单变量扰动实验
# ============================================================

def run_perturbation_experiment(config: Config, perturb: str = 'arrival_rate',
                                 values=None, topology_scale: str = 'small',
                                 traffic_mode: str = 'steady',
                                 algorithms=None) -> dict:
    """
    单变量扰动实验（对齐 GSTC 静态部署实验范式）

    参数:
        perturb: 扰动变量名 ('arrival_rate', 'request_length', 'n_task_types')
        values: 扰动值列表
        topology_scale: 'small', 'medium', 'large'
        traffic_mode: 'steady', 'tidal', 'burst'
        algorithms: 算法列表
    """
    if values is None:
        if perturb == 'arrival_rate':
            values = [1, 2, 3, 5, 8, 12]
        elif perturb == 'request_length':
            values = [2, 5, 10, 20, 50]
        else:  # n_task_types
            values = [1, 2, 3, 5, 7]

    if algorithms is None:
        algorithms = list(ALGORITHM_MAP.keys())

    tables = load_architecture_tables(config.data_path)
    all_results = {}

    print(f"\n{'='*70}")
    print(f"单变量扰动实验: {perturb}, 拓扑={topology_scale}, 流量={traffic_mode}")
    print(f"{'='*70}")

    for val in values:
        print(f"\n### {perturb} = {val}")
        perturb_results = {}

        for algo_name in algorithms:
            print(f"  [{algo_name}] ...", end=' ', flush=True)
            t0 = time.time()

            # 每个算法用独立的 topo（避免节点状态互相干扰）
            np.random.seed(config.seed)
            import random; random.seed(config.seed)
            topo = Topology(config, scale=topology_scale)
            sim = Simulator(config, topo, tables,
                           *ALGORITHM_MAP[algo_name], traffic_mode)

            if perturb == 'arrival_rate':
                sim.traffic_gen.set_perturbation(lambda_override=val)
            elif perturb == 'request_length':
                sim.traffic_gen.set_perturbation(request_length=val)
            else:  # n_task_types
                sim.traffic_gen.set_perturbation(n_tasks=val)

            r = sim.run()
            perturb_results[algo_name] = r
            print(f"lat={r['avg_latency_ms']:.1f}ms succ={r['success_rate']:.3f} ({time.time()-t0:.0f}s)")

        all_results[val] = perturb_results

        # 保存当前值的CSV
        save_results(perturb_results, config.output_dir,
                    f'perturb_{perturb}_val={val}.csv')

    # 保存汇总CSV
    _save_perturbation_summary(all_results, config.output_dir, perturb)

    # 绘制扰动曲线
    _plot_perturbation_curve(all_results, config.output_dir, perturb, values)

    return all_results


def _save_perturbation_summary(all_results: dict, output_dir: str, perturb: str):
    """保存扰动实验汇总"""
    rows = []
    for val, results in all_results.items():
        for algo, r in results.items():
            rows.append({
                'perturb_value': val,
                'algorithm': algo,
                'avg_latency_ms': round(r['avg_latency_ms'], 2),
                'success_rate': round(r['success_rate'], 4),
                'sla_violation_rate': round(r['sla_violation_rate'], 4),
                'node_utilization_avg': round(r['node_utilization_avg'], 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f'perturb_{perturb}_summary.csv'), index=False)


def _plot_perturbation_curve(all_results: dict, output_dir: str, perturb: str, values: list):
    """绘制扰动曲线（对齐 GSTC Fig.4 类似）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：时延 vs 扰动值
    ax = axes[0]
    for algo, results_by_val in _transpose_perturbation(all_results).items():
        x = list(all_results.keys())
        y = [results_by_val[v]['avg_latency_ms'] for v in x]
        ax.plot(x, y, marker='o', label=algo, linewidth=2)
    ax.set_xlabel(perturb.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel('Avg E2E Latency (ms)', fontsize=11)
    ax.set_title(f'Latency vs {perturb.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右图：成功率 vs 扰动值
    ax = axes[1]
    for algo, results_by_val in _transpose_perturbation(all_results).items():
        x = list(all_results.keys())
        y = [results_by_val[v]['success_rate'] for v in x]
        ax.plot(x, y, marker='s', label=algo, linewidth=2)
    ax.set_xlabel(perturb.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel('Success Rate', fontsize=11)
    ax.set_title(f'Success Rate vs {perturb.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig_perturb_{perturb}.png'), dpi=150)
    plt.close()


def _transpose_perturbation(all_results: dict) -> dict:
    """转置扰动结果：{algo: {val: result}}"""
    transposed = {}
    for val, results in all_results.items():
        for algo, r in results.items():
            if algo not in transposed:
                transposed[algo] = {}
            transposed[algo][val] = r
    return transposed


# ============================================================
# 等时延资源效率实验
# ============================================================

def run_equal_latency_experiment(config: Config, target_latency: float = 50.0,
                                  topology_scale: str = 'small',
                                  algorithms=None) -> dict:
    """
    等时延资源效率对比（对齐 GSTC Fig.8）

    方法：先运行 OURS 确定基准时延，然后调节各基线达到相近时延，最后比较资源消耗

    注意：当前实现直接比较同拓扑下各算法的资源利用率
    （严格实现需要迭代调节基线参数达到目标时延）
    """
    if algorithms is None:
        algorithms = list(ALGORITHM_MAP.keys())

    tables = load_architecture_tables(config.data_path)

    print(f"\n{'='*70}")
    print(f"等时延资源效率实验: 目标时延={target_latency}ms, 拓扑={topology_scale}")
    print(f"{'='*70}")

    results = {}
    ours_result = None

    # 先运行 OURS（用自己的 topo）
    print(f"\n  [OURS] (基准)...", end=' ', flush=True)
    t0 = time.time()
    np.random.seed(config.seed)
    import random; random.seed(config.seed)
    topo = Topology(config, scale=topology_scale)
    sim = Simulator(config, topo, tables, *ALGORITHM_MAP['OURS'], 'steady')
    r = sim.run()
    results['OURS'] = r
    ours_result = r
    print(f"lat={r['avg_latency_ms']:.1f}ms util={r['node_utilization_avg']:.3f} ({time.time()-t0:.0f}s)")

    # 运行其他算法（每个用自己的 topo）
    for algo_name in algorithms:
        if algo_name == 'OURS':
            continue
        print(f"  [{algo_name}] ...", end=' ', flush=True)
        t0 = time.time()
        np.random.seed(config.seed)
        random.seed(config.seed)
        topo = Topology(config, scale=topology_scale)
        sim = Simulator(config, topo, tables, *ALGORITHM_MAP[algo_name], 'steady')
        r = sim.run()
        results[algo_name] = r

        # 计算相对于 OURS 的时延和资源差异
        lat_diff = r['avg_latency_ms'] - ours_result['avg_latency_ms']
        util_diff = (r['node_utilization_avg'] - ours_result['node_utilization_avg']) / ours_result['node_utilization_avg'] * 100
        print(f"lat={r['avg_latency_ms']:.1f}ms (diff={lat_diff:+.1f}) "
              f"util={r['node_utilization_avg']:.3f} (diff={util_diff:+.1f}%) ({time.time()-t0:.0f}s)")

    # 保存和绘图
    save_results(results, config.output_dir, f'equal_latency_{topology_scale}.csv')
    _plot_equal_latency_comparison(results, config.output_dir, target_latency)

    return results


def _plot_equal_latency_comparison(results: dict, output_dir: str, target_latency: float):
    """绘制等时延资源效率对比（对齐 GSTC Fig.8 风格）"""
    algos = list(results.keys())
    latencies = [results[a]['avg_latency_ms'] for a in algos]
    utils = [results[a]['node_utilization_avg'] for a in algos]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：时延对比
    ax = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(algos)))
    bars = ax.bar(algos, latencies, color=colors)
    ax.axhline(y=target_latency, color='red', linestyle='--', label=f'Target: {target_latency}ms')
    ax.set_ylabel('Avg E2E Latency (ms)', fontsize=11)
    ax.set_title('Latency Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    for bar, v in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.1f}',
                ha='center', va='bottom', fontsize=8)

    # 右图：资源利用率对比
    ax = axes[1]
    bars = ax.bar(algos, utils, color=colors)
    ax.set_ylabel('Node Memory Utilization', fontsize=11)
    ax.set_title('Resource Efficiency Comparison', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar, v in zip(bars, utils):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.2f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_equal_latency_comparison.png'), dpi=150)
    plt.close()


# ============================================================
# 规模扩展实验
# ============================================================

def run_scale_experiment(config: Config, scales: list = None,
                          traffic_mode: str = 'tidal',
                          algorithms=None) -> dict:
    """
    规模扩展实验（对齐 GSTC Fig.5-7）

    在小/中/大三种网络规模下进行时序评估
    """
    if scales is None:
        scales = ['small', 'medium', 'large']
    if algorithms is None:
        algorithms = list(ALGORITHM_MAP.keys())

    tables = load_architecture_tables(config.data_path)
    all_results = {}

    print(f"\n{'='*70}")
    print(f"规模扩展实验: 拓扑={scales}, 流量={traffic_mode}")
    print(f"{'='*70}")

    for scale in scales:
        scale_name = {'small': 'N=15', 'medium': 'N=30', 'large': 'N=65'}.get(scale, scale)
        print(f"\n### 规模: {scale_name}")

        scale_results = {}

        for algo_name in algorithms:
            print(f"  [{algo_name}] ...", end=' ', flush=True)
            t0 = time.time()
            np.random.seed(config.seed)
            import random; random.seed(config.seed)
            topo = Topology(config, scale=scale)
            sim = Simulator(config, topo, tables, *ALGORITHM_MAP[algo_name], traffic_mode)
            r = sim.run()
            scale_results[algo_name] = r
            print(f"lat={r['avg_latency_ms']:.1f}ms succ={r['success_rate']:.3f} ({time.time()-t0:.0f}s)")

        all_results[scale] = scale_results
        save_results(scale_results, config.output_dir, f'results_{scale}_{traffic_mode}.csv')

    # 绘制规模对比
    _plot_scale_comparison(all_results, config.output_dir, traffic_mode)

    return all_results


def _plot_scale_comparison(all_results: dict, output_dir: str, traffic_mode: str):
    """绘制规模扩展对比"""
    scales = list(all_results.keys())
    scale_labels = {'small': 'N=15', 'medium': 'N=30', 'large': 'N=65'}
    algos = list(list(all_results.values())[0].keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['avg_latency_ms', 'success_rate', 'node_utilization_avg']
    titles = ['Avg E2E Latency (ms)', 'Success Rate', 'Node Utilization']

    for ax, metric, title in zip(axes, metrics, titles):
        x = np.arange(len(scales))
        width = 0.12
        colors = plt.cm.Set2(np.linspace(0, 1, len(algos)))

        for i, algo in enumerate(algos):
            vals = [all_results[s][algo][metric] for s in scales]
            ax.bar(x + i * width, vals, width, label=algo, color=colors[i])

        ax.set_xlabel('Network Scale', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (len(algos) - 1) / 2)
        ax.set_xticklabels([scale_labels[s] for s in scales])
        ax.legend(fontsize=7, loc='best')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig_scale_comparison_{traffic_mode}.png'), dpi=150)
    plt.close()


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
    print("算法: OURS (globecom.tex) | 基线: HEURISTIC_A | GREEDY_B | STATIC | RESOURCE_FIRST | ACCURACY_FIRST")
    print("实验类型: 动态负载 | 单变量扰动 | 等时延资源效率 | 规模扩展")
    print("=" * 70)

    # ---- 动态负载实验（原有）----
    print("\n" + "="*70)
    print("Part 1: 动态负载实验")
    print("="*70)
    experiments = [
        ('small_steady',  'small',  'steady'),
        ('small_tidal',   'small',  'tidal'),
        ('small_burst',   'small',  'burst'),
        ('medium_tidal',  'medium', 'tidal'),
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

    # ---- 单变量扰动实验（新增，对齐 GSTC Fig.4）----
    print("\n" + "="*70)
    print("Part 2: 单变量扰动实验（静态部署验证）")
    print("="*70)

    print("\n### 到达率扰动 (Exp-S1)")
    run_perturbation_experiment(cfg, perturb='arrival_rate',
                                values=[1, 2, 3, 5, 8, 12],
                                topology_scale='small')

    print("\n### 请求长度扰动 (Exp-S2)")
    run_perturbation_experiment(cfg, perturb='request_length',
                                values=[2, 5, 10, 20, 50],
                                topology_scale='small')

    print("\n### 请求类型数扰动 (Exp-S3)")
    run_perturbation_experiment(cfg, perturb='n_task_types',
                                values=[1, 2, 3, 5, 7],
                                topology_scale='small')

    # ---- 等时延资源效率实验（新增，对齐 GSTC Fig.8）----
    print("\n" + "="*70)
    print("Part 3: 等时延资源效率实验")
    print("="*70)
    run_equal_latency_experiment(cfg, target_latency=50.0, topology_scale='small')

    # ---- 规模扩展实验（新增，对齐 GSTC Fig.5-7）----
    print("\n" + "="*70)
    print("Part 4: 规模扩展实验")
    print("="*70)
    run_scale_experiment(cfg, scales=['small', 'medium', 'large'], traffic_mode='tidal')

    print("\n实验完成！结果保存在:", cfg.output_dir)


if __name__ == '__main__':
    main()
