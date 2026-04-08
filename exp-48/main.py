"""
微服务路由与部署实验 - 主入口

三个扰动实验 × 五种算法:
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
    print("=" * 72)

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

        # 按 (sweep_value, algorithm) 分组打印结果表格
        sweep_key = exp.sweep_key
        sweep_vals = exp.sweep_values
        algo_names = [a.name for a in algorithms]

        # 表头
        col_w = 14
        header = f"  {sweep_key:>14s} |"
        for name in algo_names:
            header += f" {name:>{col_w}s}"
        print(header)
        sep_len = 16 + 1 + (col_w + 1) * len(algo_names)
        print("  " + "-" * sep_len)

        for val in sweep_vals:
            row = f"  {val:>14d} |"
            for name in algo_names:
                match = [r for r in runner.results
                         if r[sweep_key] == val and r["algorithm"] == name]
                if match:
                    r = match[0]
                    d = r["avg_delay"]
                    s = r["stable_chains"]
                    t = r["total_chains"]
                    d_str = f"{d:.3f}" if d < 1e6 else "OVERLOAD"
                    row += f" {d_str:>{col_w - 3}s} {s}/{t}"
                else:
                    row += f" {'N/A':>{col_w}s}"
            print(row)

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
