"""
==============================================================
  服务部署与路由实验 - 主入口
==============================================================

使用方法:
  python run_experiment.py                    # 运行基本实验
  python run_experiment.py --algo ffd-m cds-m our  # 指定算法
  python run_experiment.py --perturb arrival_rate 1,20,50,100  # 扰动实验
python run_experiment.py --perturb arrival_rate 1,20,50,100  # 扰动实验
算法列表:
  ffd-m      - First Fit Decreasing (固定Model-M)
  random-m   - Random Deployment (固定Model-M)
  greedy-m   - Simple Greedy (固定Model-M)
  cds-m      - Co-Located Deployment (固定Model-M)
  our        - Our算法 (动态模型选择, 需要Excel模型库)

扰动实验参数:
  arrival_rate       - 到达率 λ 变化
  n_task_types       - 任务类型数量变化
  request_length     - 请求长度变化
==============================================================
"""

import sys
import os

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


def run_comparison_experiment(algorithms=None, seed=42, excel_path=None):
    """
    运行算法对比实验

    Args:
        algorithms: 算法列表, 如 ["ffd-m", "cds-m", "our"]
        seed: 随机种子
        excel_path: Excel模型库路径 (用于Our算法)
    """
    if algorithms is None:
        algorithms = ["ffd-m", "cds-m", "random-m", "greedy-m"]

    print_banner()
    print(f"\n[1] 生成测试数据 (seed={seed})...")

    # 使用默认配置生成数据
    config = DEFAULT_CONFIG.copy()
    config["seed"] = seed

    # 如果有Excel路径且包含our算法，添加到配置
    if excel_path and "our" in algorithms:
        config["excel_model_path"] = excel_path
        print(f"[Our算法] Excel模型库: {excel_path}")

    generator = DataGenerator(seed=seed)
    topology, services, chains = generator.generate_all(config)

    print(f"    拓扑: {len(topology.nodes)} 节点, {len(topology.links)} 链路")
    print(f"    服务: {len(services)} 个")
    print(f"    服务链: {len(chains)} 条")

    # 打印服务链详情
    print("\n    服务链详情:")
    for chain in chains:
        print(f"      {chain.chain_id}: λ={chain.arrival_rate:.2f}, "
              f"延迟约束={chain.max_latency}ms, "
              f"服务数={len(chain.services)}")

    print(f"\n[2] 运行实验: {algorithms}...")

    runner = ExperimentRunner(config)
    results = runner.run_comparison(topology, services, chains, algorithms)

    print("\n[3] 实验结果:")
    print("-" * 70)
    print(f"{'算法':<15} {'平均延迟(ms)':<18} {'成功率':<12} {'使用节点数':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['algorithm']:<15} {r['avg_latency']:<18.2f} "
              f"{r['success_rate']:<12.2%} {r['deployment_cost']:<10}")

    print("-" * 70)

    # 验证部署
    print("\n[4] 验证部署合法性:")
    validator = ResultValidator()
    for r in results:
        plan = r["deployment_plan"]
        valid, errors = validator.validate_deployment(plan, topology, services, chains)
        status = "[OK] VALID" if valid else "[X] INVALID"
        print(f"    {r['algorithm']}: {status}")
        if errors:
            for err in errors[:3]:  # 只显示前3个错误
                print(f"        - {err}")

    return results


def run_perturbation_experiment(param_name, param_values, algorithms=None, seed=42, excel_path=None):
    """
    运行扰动实验

    Args:
        param_name: 扰动参数名 (arrival_rate, n_task_types, request_length)
        param_values: 参数值列表
        algorithms: 算法列表
        seed: 随机种子
        excel_path: Excel模型库路径 (用于Our算法)
    """
    if algorithms is None:
        algorithms = ["ffd-m", "cds-m", "greedy-m"]

    print_banner()
    print(f"\n扰动实验: {param_name} = {param_values}")

    config = DEFAULT_CONFIG.copy()
    config["seed"] = seed

    # 如果有Excel路径且包含our算法，添加到配置
    if excel_path and "our" in algorithms:
        config["excel_model_path"] = excel_path
        print(f"[Our算法] Excel模型库: {excel_path}")

    generator = DataGenerator(seed=seed)
    runner = ExperimentRunner(config)

    results = runner.run_perturbation(
        base_config=config,
        param_name=param_name,
        param_values=param_values,
        algorithms=algorithms,
        generator=generator
    )

    # 按参数值分组显示
    print("\n结果汇总:")
    print("-" * 80)

    # 获取该参数的唯一值
    unique_values = sorted(set(r["value"] for r in results))

    for val in unique_values:
        print(f"\n{param_name} = {val}:")
        val_results = [r for r in results if r["value"] == val]
        print(f"{'算法':<15} {'平均延迟(ms)':<18} {'成功率':<12} {'使用节点数':<10}")
        print("-" * 55)
        for r in val_results:
            print(f"{r['algorithm']:<15} {r['avg_latency']:<18.2f} "
                  f"{r['success_rate']:<12.2%} {r['deployment_cost']:<10}")

    return results


def run_our_algorithm_with_excel(excel_path, algorithms=None, seed=42):
    """
    运行包含Our算法的实验 (需要Excel模型库)

    Args:
        excel_path: Excel模型库路径
        algorithms: 算法列表
        seed: 随机种子
    """
    if algorithms is None:
        algorithms = ["ffd-m", "cds-m", "our"]

    print_banner()
    print(f"\n[注意] Our算法需要Excel模型库: {excel_path}")

    if not os.path.exists(excel_path):
        print(f"\n[错误] Excel文件不存在: {excel_path}")
        print("请先准备模型库文件，或不使用our算法")
        return None

    # 检查Our算法是否在列表中
    if "our" in algorithms:
        print("\n加载Our算法配置...")

        # 读取Our算法的配置
        from algorithms.deployment.ours import OurAlgorithm

        # Our算法需要excel_path
        our_config = {
            "excel_model_path": excel_path
        }

        # 可以在此处覆盖runner的算法创建逻辑
        # 目前runner暂不支持Our算法，需要手动调用
        print("    Our算法已配置")

    # 暂时先用baseline算法
    baseline_algos = [a for a in algorithms if a != "our"]
    if baseline_algos:
        print(f"\n运行Baseline算法: {baseline_algos}")
        results = run_comparison_experiment(baseline_algos, seed)
    else:
        results = []

    print("\n[提示] Our算法需要在runner中手动配置ModelSearcher")
    print("示例代码:")
    print("""
    from algorithms.deployment.ours import OurAlgorithm, ModelSearcher

    # 创建模型搜索器
    searcher = ModelSearcher(excel_path)
    our_alg = OurAlgorithm({"excel_model_path": excel_path})

    # 手动运行
    deployment_plan = our_alg.deploy(topology, services, chains)
    routing_plan = our_alg.solve(topology, services, chains)
    """)

    return results


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="服务部署与路由实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_experiment.py
  python run_experiment.py --algo ffd-m cds-m random-m
  python run_experiment.py --perturb arrival_rate 1,10,50,100
  python run_experiment.py --algo ffd-m our --excel ../data/model_library.xlsx
        """
    )

    parser.add_argument("--algo", type=str, nargs="+",
                       help="要运行的算法列表: ffd-m, random-m, greedy-m, cds-m, our")
    parser.add_argument("--perturb", type=str, nargs="+",
                       help="扰动实验: --perturb param_name value1,value2,value3")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    parser.add_argument("--excel", type=str, default=None,
                       help="Excel模型库路径 (用于Our算法)")
    parser.add_argument("--output", type=str, default="results",
                       help="结果输出目录")

    args = parser.parse_args()

    algorithms = args.algo if args.algo else ["ffd-m", "cds-m", "random-m", "greedy-m"]

    # 扰动实验
    if args.perturb and len(args.perturb) >= 2:
        param_name = args.perturb[0]
        param_values = [float(v) for v in args.perturb[1].split(",")]
        results = run_perturbation_experiment(
            param_name, param_values, algorithms, args.seed, args.excel
        )
    # 普通实验 (或Our算法对比，需要Excel)
    else:
        if "our" in algorithms and not args.excel:
            print("\n[警告] Our算法需要Excel模型库，使用 --excel 参数指定")
            print("       暂时从算法列表中移除our")
            algorithms = [a for a in algorithms if a != "our"]
        results = run_comparison_experiment(algorithms, args.seed, args.excel)

    # 保存结果
    if results:
        save_results(results, args.output)
        print(f"\n结果已保存到: {args.output}/")


def save_results(results, output_dir):
    """保存结果到文件"""
    import json
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存JSON
    json_path = os.path.join(output_dir, "experiment_results.json")
    serializable = []
    for r in results:
        serializable.append({
            "algorithm": r["algorithm"],
            "avg_latency": r["avg_latency"],
            "success_rate": r["success_rate"],
            "deployment_cost": r["deployment_cost"],
            "param": r.get("param"),
            "value": r.get("value"),
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    # 2. 保存CSV汇总
    csv_path = os.path.join(output_dir, "experiment_summary.csv")
    df = pd.DataFrame(serializable)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
