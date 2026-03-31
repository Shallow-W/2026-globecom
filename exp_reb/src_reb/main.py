# ==========================================
# 主入口：运行实验
# ==========================================

"""
使用说明:
  # 基本对比实验
  python src_reb/main.py

  # 指定算法
  python src_reb/main.py --algo our ffd-m drs

  # 扰动实验
  python src_reb/main.py --perturb arrival_rate 100,200,300,400,500
  python src_reb/main.py --perturb chain_length 3,4,5,6,7
  python src_reb/main.py --perturb num_types 10,20,30,40,50
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np

# 添加src_reb到路径（向上两级到exp_reb根目录）
_src_reb_dir = os.path.dirname(os.path.abspath(__file__))
_exp_reb_root = os.path.dirname(_src_reb_dir)
sys.path.insert(0, _exp_reb_root)

from src_reb.config import DEFAULT_CONFIG, EXPERIMENTS
from src_reb.runner import DataGenerator, ExperimentRunner


def print_banner():
    print("=" * 60)
    print("  服务部署与路由实验框架 (src_reb)")
    print("  Deployment & Routing Experiment Framework")
    print("=" * 60)


def print_results(results: list):
    """打印实验结果"""
    print("\n实验结果:")
    print("-" * 120)
    print(f"{'算法':<12} {'总延迟(ms)':<14} {'排队(ms)':<12} {'通信(ms)':<12} "
          f"{'惩罚':<12} {'QoS':<10} {'内存利用':<10} {'成功率':<10} {'节点数':<8}")
    print("-" * 120)

    for r in results:
        lat = r.get("avg_latency")
        lat_str = f"{lat:.2f}" if isinstance(lat, (int, float)) and lat != float('inf') else "Inf"

        q = r.get("avg_queuing", 0)
        comm = r.get("avg_communication", 0)
        penalty = r.get("total_penalty", 0)
        qos = r.get("avg_qos", 0)
        mem = r.get("mem_utilization", 0)
        success = r.get("success_rate", 0)
        cost = r.get("deployment_cost", 0)

        print(
            f"{r['algorithm']:<12} {lat_str:<14} {q:<12.2f} {comm:<12.2f} "
            f"{penalty:<12.1f} {qos:<10.4f} {mem:<10.1%} {success:<10.1%} {cost:<8}"
        )
    print("-" * 120)


def save_results(results: list, output_dir: str = "results"):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)

    def to_native(val):
        """Convert numpy/Python types to native Python types for JSON serialization"""
        if val is None:
            return None
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, (np.ndarray,)):
            return val.tolist()
        return val

    serializable = []
    for r in results:
        lat = r.get("avg_latency")
        serializable.append({
            "algorithm": r["algorithm"],
            "avg_latency": to_native(lat) if isinstance(lat, (int, float)) and lat != float('inf') else None,
            "avg_queuing": to_native(r.get("avg_queuing", 0)),
            "avg_communication": to_native(r.get("avg_communication", 0)),
            "avg_qos": to_native(r.get("avg_qos", 0)),
            "total_penalty": to_native(r.get("total_penalty", 0)),
            "mem_utilization": to_native(r.get("mem_utilization", 0)),
            "success_rate": to_native(r.get("success_rate", 0)),
            "deployment_cost": to_native(r.get("deployment_cost", 0)),
            "fitness": to_native(r.get("fitness", 0)),
            "param": r.get("param"),
            "value": to_native(r.get("value")),
        })

    # JSON
    json_path = os.path.join(output_dir, "experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    # CSV
    csv_path = os.path.join(output_dir, "experiment_summary.csv")
    df = pd.DataFrame(serializable)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  CSV: {csv_path}")


def run_comparison(algorithms: list, seed: int, excel_path: str = None):
    """运行对比实验"""
    print_banner()
    print(f"\n[1] 生成测试数据 (seed={seed})...")

    config = DEFAULT_CONFIG.copy()
    config["seed"] = seed
    if excel_path:
        config["excel_model_path"] = excel_path

    generator = DataGenerator(seed=seed)
    topology, services, chains, model_lib = generator.generate_all(config)

    print(f"    拓扑: {len(topology.nodes)} 节点")
    print(f"    服务: {len(services)} 个")
    print(f"    服务链: {len(chains)} 条")

    print(f"\n[2] 运行实验: {algorithms}...")
    runner = ExperimentRunner(config)
    results = runner.run_comparison(topology, services, chains, model_lib, algorithms)

    print_results(results)
    return results


def run_perturbation(param_name: str, param_values: list,
                     algorithms: list, seed: int, excel_path: str = None):
    """运行扰动实验"""
    print_banner()
    print(f"\n扰动实验: {param_name} = {param_values}")

    config = DEFAULT_CONFIG.copy()
    config["seed"] = seed
    if excel_path:
        config["excel_model_path"] = excel_path

    runner = ExperimentRunner(config)

    # 按param分组打印
    results = runner.run_perturbation(config, param_name, param_values, algorithms)

    unique_values = sorted(set(r["value"] for r in results))

    for val in unique_values:
        print(f"\n{'=' * 60}")
        print(f"{param_name} = {val}")
        val_results = [r for r in results if r["value"] == val]
        print_results(val_results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="服务部署与路由实验 (src_reb)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--algo", type=str, nargs="+",
        default=["our"],
        help="算法列表: our, ffd-m, drs, lego, cds-m, random-m, greedy-m"
    )
    parser.add_argument(
        "--perturb", type=str, nargs="+",
        help="扰动实验: --perturb param_name value1,value2,..."
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--excel", type=str, default=None,
        help="Excel模型库路径"
    )
    parser.add_argument("--output", type=str, default="results", help="输出目录")

    args = parser.parse_args()

    algorithms = args.algo

    # 扰动实验
    if args.perturb and len(args.perturb) >= 2:
        param_name = args.perturb[0]
        param_values = [float(v) for v in args.perturb[1].split(",")]
        results = run_perturbation(param_name, param_values, algorithms, args.seed, args.excel)
    else:
        results = run_comparison(algorithms, args.seed, args.excel)

    # 保存
    save_results(results, args.output)
    print(f"\n结果已保存到: {args.output}/")


if __name__ == "__main__":
    main()
