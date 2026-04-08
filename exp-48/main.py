"""
微服务路由与部署实验 - 主入口

三个扰动实验 x 四种算法:
  1. 到达率扰动:  total_rate = 100, 200, ..., 600
  2. 链长扰动:    chain_length = 3, 4, ..., 8
  3. 服务类型扰动: n_service_types = 10, 20, ..., 60

算法:
  RLS  - 随机部署 + 按比例路由 (基线)
  FFD  - First Fit Decreasing 部署 + 按比例路由
  DRS  - 贪心部署 + 平方根加权概率路由
  LEGO - 负载均衡部署 + 确定性均匀路由
"""
from algorithm import ALL_ALGORITHMS
from experiment import (
    ExperimentRunner,
    make_arrival_rate_experiment,
    make_chain_length_experiment,
    make_service_type_experiment,
)


ALGO_NAMES = ["Random-Proportional", "FFD", "DRS", "LEGO"]

METRICS = [
    ("avg_delay",       "Delay(s)",  "{:.3f}"),
    ("avg_comp_delay",  "Comp(s)",   "{:.3f}"),
    ("avg_comm_delay",  "Comm(s)",   "{:.4f}"),
    ("total_penalty",   "Penalty",   "{:.3f}"),
    ("stable_chains",   "Stable",    None),
    ("cpu_utilization", "CPU_Util",  "{:.1%}"),
    ("gpu_utilization", "GPU_Util",  "{:.1%}"),
]


def _fmt_metric(key, result):
    """格式化单个指标"""
    if key == "stable_chains":
        return f"{result['stable_chains']}/{result['total_chains']}"
    v = result.get(key, 0)
    for mkey, _, mfmt in METRICS:
        if mkey == key:
            return "OVERLOAD" if v > 1e6 else mfmt.format(v)
    return str(v)


def _print_matrix(runner, sweep_key, sweep_vals, algo_names):
    """每个扫描值打印一个矩阵: 行=算法, 列=指标"""
    for val in sweep_vals:
        # 收集该扫描值下各算法的结果
        rows = []
        for name in algo_names:
            hits = [r for r in runner.results
                    if r[sweep_key] == val and r["algorithm"] == name]
            rows.append((name, hits[0] if hits else None))

        # 计算各列宽度
        name_w = max(len(n) for n in algo_names)
        col_ws = []
        for mkey, mname, _ in METRICS:
            w = max(len(mname), 7)
            for _, r in rows:
                if r:
                    w = max(w, len(_fmt_metric(mkey, r)))
            col_ws.append(w + 1)

        # 打印标题
        print(f"\n  >> {sweep_key} = {val}")

        # 表头
        header = f"  {'Algorithm':<{name_w}} |"
        for (_, mname, _), cw in zip(METRICS, col_ws):
            header += f" {mname:>{cw}}"
        print(header)

        # 分隔线
        sep = f"  {'-' * name_w}-+"
        for cw in col_ws:
            sep += "-" * (cw + 1)
        print(sep)

        # 数据行
        for aname, r in rows:
            line = f"  {aname:<{name_w}} |"
            if r is None:
                for cw in col_ws:
                    line += f" {'N/A':>{cw}}"
            else:
                for (mkey, _, _), cw in zip(METRICS, col_ws):
                    line += f" {_fmt_metric(mkey, r):>{cw}}"
            print(line)


def main():
    seed = 42
    algorithms = [a for a in ALL_ALGORITHMS if a.name in ALGO_NAMES]

    print()
    print("=" * 72)
    print("  Microservice Routing & Deployment Experiment")
    print("  Jackson Queuing Network / M/M/c Delay Model")
    print("=" * 72)
    print(f"  Algorithms : {', '.join(a.name for a in algorithms)}")
    print(f"  Seed       : {seed}")

    experiments = [
        ("Arrival Rate Sweep",  make_arrival_rate_experiment),
        ("Chain Length Sweep",   make_chain_length_experiment),
        ("Service Type Sweep",  make_service_type_experiment),
    ]

    all_paths = []

    for idx, (title, make_fn) in enumerate(experiments, 1):
        exp = make_fn()
        runner = ExperimentRunner(exp, algorithms)

        print()
        print("-" * 72)
        print(f"  Experiment {idx}/3: {title}")
        print(f"  Sweep: {exp.sweep_key} in {exp.sweep_values}")
        print("-" * 72)

        runner.run(seed=seed, verbose=False)

        _print_matrix(
            runner,
            sweep_key=exp.sweep_key,
            sweep_vals=exp.sweep_values,
            algo_names=[a.name for a in algorithms],
        )

        path = runner.save_csv()
        all_paths.append(path)
        print(f"\n  -> {path}")

    print()
    print("=" * 72)
    print("  All experiments completed.")
    for p in all_paths:
        print(f"    {p}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
