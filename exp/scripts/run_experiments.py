"""
实验启动脚本 - 统一入口

提供 4 种实验类型的快捷运行方式：
    python run_experiments.py [mode] [options]

用法示例:
    python run_experiments.py quick          # 快速验证（10 slots）
    python run_experiments.py dynamic       # 动态负载实验
    python run_experiments.py perturb ar     # 到达率扰动
    python run_experiments.py perturb len    # 请求长度扰动
    python run_experiments.py perturb task  # 请求类型数扰动
    python run_experiments.py equal          # 等时延资源效率
    python run_experiments.py scale          # 规模扩展
    python run_experiments.py all            # 运行全部实验
"""

import sys
import os

# 确保脚本目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfg import Config
from exp import (
    run_experiment, run_perturbation_experiment,
    run_equal_latency_experiment, run_scale_experiment,
    save_results, save_latencies, plot_comparison_bar,
    plot_cdf, print_summary
)


def _make_cfg(n_slots: int = 100) -> Config:
    cfg = Config()
    cfg.n_slots = n_slots
    os.makedirs(cfg.output_dir, exist_ok=True)
    return cfg


def run_quick(cfg: Config = None):
    """快速验证：10 slots，验证代码是否正常"""
    if cfg is None:
        cfg = _make_cfg(10)
    print("\n" + "="*70)
    print("快速验证模式（10 slots）")
    print("="*70)

    print("\n--- 动态负载（steady）---")
    r = run_experiment(cfg, 'small', 'steady')
    print_summary(r, 'quick_steady')

    print("\n--- 扰动（到达率 1,3,5）---")
    run_perturbation_experiment(cfg, perturb='arrival_rate', values=[1, 3, 5])

    print("\n--- 规模（仅 small）---")
    run_scale_experiment(cfg, scales=['small'], traffic_mode='tidal')

    print("\n[OK] 快速验证完成！")


def run_dynamic(cfg: Config = None):
    """动态负载实验：steady / tidal / burst + small/medium/large"""
    if cfg is None:
        cfg = _make_cfg(100)

    print("\n" + "="*70)
    print("动态负载实验")
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
        r = run_experiment(cfg, scale, mode)
        save_results(r, cfg.output_dir, f'results_{exp_name}.csv')
        save_latencies(r, cfg.output_dir, exp_name)
        plot_comparison_bar(r, cfg.output_dir, f'fig_bar_{exp_name}.png')
        plot_cdf(r, cfg.output_dir, f'fig_cdf_{exp_name}.png')
        print_summary(r, exp_name)

    print(f"\n[OK] 动态负载实验完成，结果保存在: {cfg.output_dir}")


def run_perturb(perturb_type: str = 'ar', cfg: Config = None):
    """
    单变量扰动实验

    perturb_type:
        'ar'   - 到达率扰动 (arrival_rate): [1, 2, 3, 5, 8, 12]
        'len'  - 请求长度扰动 (request_length): [2, 5, 10, 20, 50]
        'task' - 请求类型数扰动 (n_task_types): [1, 2, 3, 5, 7]
    """
    if cfg is None:
        cfg = _make_cfg(100)

    perturb_map = {
        'ar':   ('arrival_rate',    [1, 2, 3, 5, 8, 12],   '到达率扰动 (arrival_rate)'),
        'len':  ('request_length',  [2, 5, 10, 20, 50],    '请求长度扰动 (request_length)'),
        'task': ('n_task_types',    [1, 2, 3, 5, 7],       '请求类型数扰动 (n_task_types)'),
    }

    if perturb_type not in perturb_map:
        print(f"[ERROR] 未知的扰动类型: {perturb_type}")
        print(f"可用类型: {list(perturb_map.keys())}")
        return

    perturb_name, perturb_values, description = perturb_map[perturb_type]

    print("\n" + "="*70)
    print(f"单变量扰动实验: {description}")
    print("="*70)

    run_perturbation_experiment(
        cfg, perturb=perturb_name, values=perturb_values, topology_scale='small'
    )

    print(f"\n[OK] {description}完成，结果保存在: {cfg.output_dir}")


def run_equal_latency(target_latency: float = 50.0, cfg: Config = None):
    """等时延资源效率实验"""
    if cfg is None:
        cfg = _make_cfg(100)

    print("\n" + "="*70)
    print(f"等时延资源效率实验 (目标时延={target_latency}ms)")
    print("="*70)

    run_equal_latency_experiment(cfg, target_latency=target_latency, topology_scale='small')

    print(f"\n[OK] 等时延资源效率实验完成，结果保存在: {cfg.output_dir}")


def run_scale(scales: list = None, traffic_mode: str = 'tidal', cfg: Config = None):
    """规模扩展实验"""
    if cfg is None:
        cfg = _make_cfg(100)
    if scales is None:
        scales = ['small', 'medium', 'large']

    print("\n" + "="*70)
    print(f"规模扩展实验: 拓扑={scales}, 流量={traffic_mode}")
    print("="*70)

    run_scale_experiment(cfg, scales=scales, traffic_mode=traffic_mode)

    print(f"\n[OK] 规模扩展实验完成，结果保存在: {cfg.output_dir}")


def run_all(cfg: Config = None):
    """运行全部实验"""
    if cfg is None:
        cfg = _make_cfg(100)

    print("\n" + "="*70)
    print("全量实验（动态负载 + 扰动 + 等时延 + 规模扩展）")
    print("="*70)

    # Part 1: 动态负载
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
        r = run_experiment(cfg, scale, mode)
        save_results(r, cfg.output_dir, f'results_{exp_name}.csv')
        save_latencies(r, cfg.output_dir, exp_name)
        plot_comparison_bar(r, cfg.output_dir, f'fig_bar_{exp_name}.png')
        plot_cdf(r, cfg.output_dir, f'fig_cdf_{exp_name}.png')
        print_summary(r, exp_name)

    # Part 2: 单变量扰动
    print("\n" + "="*70)
    print("Part 2: 单变量扰动实验")
    print("="*70)
    print("\n--- 到达率扰动 ---")
    run_perturbation_experiment(cfg, perturb='arrival_rate', values=[1, 2, 3, 5, 8, 12])
    print("\n--- 请求长度扰动 ---")
    run_perturbation_experiment(cfg, perturb='request_length', values=[2, 5, 10, 20, 50])
    print("\n--- 请求类型数扰动 ---")
    run_perturbation_experiment(cfg, perturb='n_task_types', values=[1, 2, 3, 5, 7])

    # Part 3: 等时延资源效率
    print("\n" + "="*70)
    print("Part 3: 等时延资源效率实验")
    print("="*70)
    run_equal_latency_experiment(cfg, target_latency=50.0, topology_scale='small')

    # Part 4: 规模扩展
    print("\n" + "="*70)
    print("Part 4: 规模扩展实验")
    print("="*70)
    run_scale_experiment(cfg, scales=['small', 'medium', 'large'], traffic_mode='tidal')

    print("\n" + "="*70)
    print(f"[OK] 全部实验完成！结果保存在: {cfg.output_dir}")
    print("="*70)


# ============================================================
# 命令行入口
# ============================================================

def print_usage():
    print(__doc__)
    print("\n快捷命令对照表：")
    print("  python run_experiments.py quick          # 快速验证")
    print("  python run_experiments.py dynamic        # 动态负载实验")
    print("  python run_experiments.py perturb ar     # 到达率扰动")
    print("  python run_experiments.py perturb len    # 请求长度扰动")
    print("  python run_experiments.py perturb task   # 请求类型数扰动")
    print("  python run_experiments.py equal          # 等时延资源效率")
    print("  python run_experiments.py scale          # 规模扩展")
    print("  python run_experiments.py all            # 运行全部实验")


if __name__ == '__main__':
    argv = sys.argv[1:]

    if len(argv) == 0:
        print_usage()
    elif argv[0] == 'quick':
        run_quick()
    elif argv[0] == 'dynamic':
        run_dynamic()
    elif argv[0] == 'perturb':
        if len(argv) < 2:
            print("[ERROR] perturb 需要子类型: ar / len / task")
            print_usage()
        else:
            run_perturb(argv[1])
    elif argv[0] == 'equal':
        target = float(argv[1]) if len(argv) > 1 else 50.0
        run_equal_latency(target)
    elif argv[0] == 'scale':
        run_scale()
    elif argv[0] == 'all':
        run_all()
    else:
        print(f"[ERROR] 未知命令: {argv[0]}")
        print_usage()
