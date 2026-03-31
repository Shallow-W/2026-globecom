"""
==============================================================
  服务部署与路由实验 - 主入口
==============================================================

使用方法:
  # 基本对比实验
  python run_experiment.py

  # 指定算法
  python run_experiment.py --algo ffd-m drs lego our

  # 扰动实验（与exp_2保持一致）
  # 到达率扰动（固定: num_types=10, length=4）
  python run_experiment.py --perturb arrival_rate 100,200,300,400,500,600,700,800

  # 链长度扰动（固定: num_types=10, rate=200）
  python run_experiment.py --perturb chain_length 3,4,5,6,7,8,9,10

  # 服务种类扰动（固定: length=4, rate=200）
  python run_experiment.py --perturb n_task_types 10,20,30,40,50,60,70,80

  # 完整示例
  python run_experiment.py --algo ffd-m drs lego our --perturb arrival_rate 100,200,400,600,800 --seed 42

算法列表:
  ffd-m      - First Fit Decreasing (论文baseline)
  drs        - Dynamic Resource Scheduling (论文baseline)
  lego       - Locality Enhanced Greedy (论文baseline)
  cds-m      - Co-Located Deployment (对比用)
  random-m   - Random Deployment
  greedy-m   - Simple Greedy
  our        - Our算法 (动态模型搜索)

扰动实验参数:
  arrival_rate  - 到达率 λ 变化 (固定链结构)
  n_task_types   - 任务类型数量变化
  chain_length   - 服务链长度变化

==============================================================
"""

import sys
import os
import math

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.default_config import DEFAULT_CONFIG
from experiments.generator import DataGenerator
from experiments.runner import ExperimentRunner
from experiments.validator import ResultValidator


def print_banner():
    """打印实验banner"""
    print("=" * 60)
    print("  服务部署与路由实验框架")
    print("  Deployment & Routing Experiment Framework")
    print("=" * 60)


# 固定Excel模型库路径
DEFAULT_EXCEL_PATH = (
    "d:/Item/lab/2026globcom/exp_reb/data/evaluation_tables_20260325_163931.xlsx"
)

# 算法列表
AVAILABLE_ALGORITHMS = ["ffd-m", "drs", "lego", "cds-m", "random-m", "greedy-m", "our"]


def run_comparison_experiment(algorithms=None, seed=42, excel_path=None):
    """
    运行算法对比实验

    Args:
        algorithms: 算法列表
        seed: 随机种子
        excel_path: Excel模型库路径 (用于Our算法)
    """
    if algorithms is None:
        algorithms = ["ffd-m", "drs", "lego"]

    print_banner()
    print(f"\n[1] 生成测试数据 (seed={seed})...")

    # 使用默认配置生成数据
    config = DEFAULT_CONFIG.copy()
    config["seed"] = seed

    generator = DataGenerator(seed=seed)
    topology, services, chains = generator.generate_all(config)

    print(f"    拓扑: {len(topology.nodes)} 节点, {len(topology.links)} 链路")
    print(f"    服务: {len(services)} 个")
    print(f"    服务链: {len(chains)} 条")

    # 打印服务链详情
    print("\n    服务链详情:")
    for chain in chains:
        print(
            f"      {chain.chain_id}: λ={chain.arrival_rate:.2f}, "
            f"延迟约束={chain.max_latency}ms, "
            f"服务数={len(chain.services)}"
        )

    # 如果包含our算法，使用Excel路径
    if "our" in algorithms:
        excel_path = excel_path or DEFAULT_EXCEL_PATH
        config["excel_model_path"] = excel_path
        print(f"\n[Our算法] Excel模型库: {excel_path}")

    print(f"\n[2] 运行实验: {algorithms}...")

    runner = ExperimentRunner(config)
    results = runner.run_comparison(topology, services, chains, algorithms)

    print("\n[3] 实验结果:")
    print("-" * 110)
    print(f"{'算法':<12} {'总延迟(ms)':<14} {'排队(ms)':<12} {'通信(ms)':<12} {'惩罚':<10} {'内存利用':<10} {'成功率':<10} {'节点数':<8}")
    print("-" * 110)

    for r in results:
        lat = r.get("avg_latency")
        if lat is None or (isinstance(lat, float) and math.isnan(lat)):
            lat_str = "N/A"
        elif isinstance(lat, float) and math.isinf(lat):
            lat_str = "Inf"
        else:
            lat_str = f"{lat:.2f}"

        q = r.get("avg_queuing", 0)
        comm = r.get("avg_communication", 0)
        penalty = r.get("total_penalty", 0)
        mem = r.get("mem_utilization", 0)
        success = r.get("success_rate", 0)
        cost = r.get("deployment_cost", 0)

        print(
            f"{r['algorithm']:<12} {lat_str:<14} {q:<12.2f} {comm:<12.2f} "
            f"{penalty:<10.1f} {mem:<10.1%} {success:<10.1%} {cost:<8}"
        )

    print("-" * 70)

    # 验证部署
    print("\n[4] 验证部署合法性:")
    validator = ResultValidator()
    for r in results:
        plan = r["deployment_plan"]
        valid, errors = validator.validate_deployment(plan, topology, services, chains)
        status = "[OK]" if valid else "[X]"
        print(f"    {r['algorithm']:<12}: {status}")
        if errors:
            for err in errors[:3]:
                print(f"        - {err}")

    return results


def run_perturbation_experiment(
    param_name, param_values, algorithms=None, seed=42, excel_path=None
):
    """
    运行扰动实验 (保持链结构不变，只改变指定参数)

    Args:
        param_name: 扰动参数名 (arrival_rate, n_task_types)
        param_values: 参数值列表
        algorithms: 算法列表
        seed: 随机种子
        excel_path: Excel模型库路径
    """
    if algorithms is None:
        algorithms = ["ffd-m", "drs", "lego"]

    print_banner()
    print(f"\n扰动实验: {param_name} = {param_values}")

    config = DEFAULT_CONFIG.copy()
    config["seed"] = seed

    # 如果包含our算法，使用Excel路径
    if "our" in algorithms:
        excel_path = excel_path or DEFAULT_EXCEL_PATH
        config["excel_model_path"] = excel_path
        print(f"[Our算法] Excel模型库: {excel_path}")

    generator = DataGenerator(seed=seed)
    runner = ExperimentRunner(config)

    results = runner.run_perturbation(
        base_config=config,
        param_name=param_name,
        param_values=param_values,
        algorithms=algorithms,
        generator=generator,
    )

    # 按参数值分组显示
    print("\n结果汇总:")
    print("-" * 110)

    unique_values = sorted(set(r["value"] for r in results))

    for val in unique_values:
        print(f"\n{param_name} = {val}:")
        val_results = [r for r in results if r["value"] == val]
        print(f"{'算法':<12} {'总延迟(ms)':<14} {'排队(ms)':<12} {'通信(ms)':<12} {'惩罚':<10} {'内存利用':<10} {'成功率':<10} {'节点数':<8}")
        print("-" * 82)
        for r in val_results:
            lat = r.get("avg_latency")
            if lat is None or (isinstance(lat, float) and math.isnan(lat)):
                lat_str = "N/A"
            elif isinstance(lat, float) and math.isinf(lat):
                lat_str = "Inf"
            else:
                lat_str = f"{lat:.2f}"
            q = r.get("avg_queuing", 0)
            comm = r.get("avg_communication", 0)
            penalty = r.get("total_penalty", 0)
            mem = r.get("mem_utilization", 0)
            success = r.get("success_rate", 0)
            cost = r.get("deployment_cost", 0)
            print(
                f"{r['algorithm']:<12} {lat_str:<14} {q:<12.2f} {comm:<12.2f} "
                f"{penalty:<10.1f} {mem:<10.1%} {success:<10.1%} {cost:<8}"
            )

    return results


def save_results(results, output_dir):
    """保存结果到文件"""
    import json
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # 准备可序列化数据
    serializable = []
    for r in results:
        lat = r["avg_latency"]
        serializable.append(
            {
                "algorithm": r["algorithm"],
                "avg_latency": lat if lat == lat else None,  # NaN -> None
                "avg_queuing": r.get("avg_queuing", 0),
                "avg_processing": r.get("avg_processing", 0),
                "avg_communication": r.get("avg_communication", 0),
                "total_penalty": r.get("total_penalty", 0),
                "mem_utilization": r.get("mem_utilization", 0),
                "success_rate": r["success_rate"],
                "deployment_cost": r["deployment_cost"],
                "param": r.get("param"),
                "value": r.get("value"),
            }
        )

    # 保存JSON
    json_path = os.path.join(output_dir, "experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    # 保存CSV
    csv_path = os.path.join(output_dir, "experiment_summary.csv")
    df = pd.DataFrame(serializable)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  CSV: {csv_path}")


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="服务部署与路由实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
算法列表:
  ffd-m      - First Fit Decreasing (论文baseline)
  drs        - Dynamic Resource Scheduling (论文baseline)
  lego       - Locality Enhanced Greedy (论文baseline)
  cds-m      - Co-Located Deployment (对比用)
  random-m   - Random Deployment
  greedy-m   - Simple Greedy
  our        - Our算法 (动态模型搜索)

示例:
  python run_experiment.py
  python run_experiment.py --algo ffd-m drs lego
  python run_experiment.py --perturb arrival_rate 100,200,400,600,800
  python run_experiment.py --perturb chain_length 3,4,5,6,7,8,9,10
  python run_experiment.py --perturb n_task_types 10,20,30,40,50,60,70,80
  python run_experiment.py --algo ffd-m drs lego our --perturb arrival_rate 100,200,400,800 --seed 42
        """,
    )

    parser.add_argument(
        "--algo",
        type=str,
        nargs="+",
        choices=AVAILABLE_ALGORITHMS,
        help="要运行的算法列表",
    )
    parser.add_argument(
        "--perturb",
        type=str,
        nargs="+",
        help="扰动实验: --perturb param_name value1,value2,...",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument(
        "--excel", type=str, default=None, help="Excel模型库路径 (用于Our算法)"
    )
    parser.add_argument("--output", type=str, default="results", help="结果输出目录")

    args = parser.parse_args()

    algorithms = args.algo if args.algo else ["ffd-m", "drs", "lego", "cds-m", "random-m", "greedy-m", "our"]

    # 检查算法是否有效
    invalid_algos = [a for a in algorithms if a not in AVAILABLE_ALGORITHMS]
    if invalid_algos:
        print(f"[错误] 无效算法: {invalid_algos}")
        print(f"可用算法: {AVAILABLE_ALGORITHMS}")
        return

    # 扰动实验
    if args.perturb and len(args.perturb) >= 2:
        param_name = args.perturb[0]
        param_values = [float(v) for v in args.perturb[1].split(",")]
        results = run_perturbation_experiment(
            param_name, param_values, algorithms, args.seed, args.excel
        )
    # 普通实验
    else:
        results = run_comparison_experiment(algorithms, args.seed, args.excel)

    # 保存结果
    if results:
        save_results(results, args.output)
        print(f"\n结果已保存到: {args.output}/")


if __name__ == "__main__":
    main()
