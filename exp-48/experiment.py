"""
实验运行器：三个扰动实验的通用执行引擎

实验与算法完全解耦:
  - Experiment 只定义"扫描哪个参数、范围是什么、其他参数固定为什么"
  - AlgorithmSuite 只负责"给定输入怎么部署和路由"
  - ExperimentRunner 把两者组合起来运行

新增实验 -> 添加 Experiment 配置
新增算法 -> 在 algorithm.py 注册 AlgorithmSuite
"""
import csv
import os
import numpy as np

from network import EdgeNetwork
from service import generate_services, generate_service_chains, compute_aggregate_arrival_rates
from evaluation import evaluate
from config import (
    DEFAULT_TOTAL_RATE, DEFAULT_CHAIN_LENGTH, DEFAULT_N_SERVICE_TYPES,
    DEFAULT_SCALE, DEFAULT_N_CHAINS, NETWORK_SCALES,
    ARRIVAL_RATE_SWEEP, CHAIN_LENGTH_SWEEP, SERVICE_TYPE_SWEEP,
)


# ============================================================
# 实验定义
# ============================================================

class Experiment:
    """
    单个扰动实验的配置。

    属性:
        name:        实验名称
        sweep_key:   扫描参数名 ('total_rate' / 'chain_length' / 'n_service_types')
        sweep_values: 扫描参数值列表
        fixed:       固定参数字典
    """

    def __init__(self, name, sweep_key, sweep_values, fixed=None):
        self.name = name
        self.sweep_key = sweep_key
        self.sweep_values = sweep_values
        self.fixed = fixed or {}

    def build_params(self, sweep_value):
        """合并固定参数和当前扫描值，返回完整参数字典"""
        params = dict(self.fixed)
        params[self.sweep_key] = sweep_value
        return params

    def __repr__(self):
        return f"Experiment('{self.name}', sweep={self.sweep_key}, values={self.sweep_values})"


# ============================================================
# 三个预设扰动实验
# ============================================================

def make_arrival_rate_experiment(
    rates=None,
    scale=DEFAULT_SCALE,
    chain_length=DEFAULT_CHAIN_LENGTH,
    n_service_types=DEFAULT_N_SERVICE_TYPES,
    n_chains=DEFAULT_N_CHAINS,
):
    """实验 1: 扫描到达率"""
    if rates is None:
        rates = ARRIVAL_RATE_SWEEP
    return Experiment(
        name="arrival_rate",
        sweep_key="total_rate",
        sweep_values=rates,
        fixed={
            "scale": scale,
            "chain_length": chain_length,
            "n_service_types": n_service_types,
            "n_chains": n_chains,
        },
    )


def make_chain_length_experiment(
    lengths=None,
    scale=DEFAULT_SCALE,
    total_rate=DEFAULT_TOTAL_RATE,
    n_service_types=DEFAULT_N_SERVICE_TYPES,
    n_chains=DEFAULT_N_CHAINS,
):
    """实验 2: 扫描链长度"""
    if lengths is None:
        lengths = CHAIN_LENGTH_SWEEP
    return Experiment(
        name="chain_length",
        sweep_key="chain_length",
        sweep_values=lengths,
        fixed={
            "scale": scale,
            "total_rate": total_rate,
            "n_service_types": n_service_types,
            "n_chains": n_chains,
        },
    )


def make_service_type_experiment(
    n_types_list=None,
    scale=DEFAULT_SCALE,
    total_rate=DEFAULT_TOTAL_RATE,
    chain_length=DEFAULT_CHAIN_LENGTH,
    n_chains=DEFAULT_N_CHAINS,
):
    """实验 3: 扫描服务类型数"""
    if n_types_list is None:
        n_types_list = SERVICE_TYPE_SWEEP
    return Experiment(
        name="service_type",
        sweep_key="n_service_types",
        sweep_values=n_types_list,
        fixed={
            "scale": scale,
            "total_rate": total_rate,
            "chain_length": chain_length,
            "n_chains": n_chains,
        },
    )


# ============================================================
# 单次运行
# ============================================================

def run_single(algorithm, params, seed=None):
    """
    用指定算法跑一次完整流程。

    参数:
        algorithm: AlgorithmSuite 实例
        params:    dict, 包含 scale, total_rate, chain_length, n_service_types, n_chains
        seed:      随机种子

    返回:
        dict, 包含参数 + 全部性能指标
    """
    scale = params["scale"]
    total_rate = params["total_rate"]
    chain_length = params["chain_length"]
    n_service_types = params["n_service_types"]
    n_chains = params.get("n_chains", DEFAULT_N_CHAINS)

    # 1. 生成场景
    network = EdgeNetwork(scale=scale, seed=seed)
    services = generate_services(n_service_types, seed=seed)
    chains = generate_service_chains(n_chains, n_service_types, chain_length, total_rate, seed=seed)
    lambda_s = compute_aggregate_arrival_rates(chains, len(services))

    # 2. 算法求解
    deployment, routing = algorithm.solve(network, services, lambda_s, seed=seed)

    # 3. 评估
    metrics = evaluate(deployment, services, chains, network, routing=routing)

    return {
        "algorithm": algorithm.name,
        "scale": scale,
        "n_nodes": network.n_nodes,
        "total_rate": total_rate,
        "chain_length": chain_length,
        "n_service_types": n_service_types,
        "n_chains": n_chains,
        **metrics,
    }


# ============================================================
# 实验运行器
# ============================================================

class ExperimentRunner:
    """
    将一个扰动实验 × 一个/多个算法 组合运行。

    用法:
        runner = ExperimentRunner(experiment, algorithms=[algo1, algo2])
        results = runner.run(seed=42)
        runner.save_csv("results/")
    """

    def __init__(self, experiment, algorithms):
        self.experiment = experiment
        self.algorithms = algorithms if isinstance(algorithms, list) else [algorithms]
        self.results = []

    def run(self, seed=None, verbose=True):
        """运行全部 (参数值 x 算法) 组合"""
        self.results = []

        if verbose:
            print(f"\n--- Experiment: {self.experiment.name} "
                  f"(sweep: {self.experiment.sweep_key}) ---")

        for algo in self.algorithms:
            if verbose:
                print(f"\n  Algorithm: {algo.name}")

            for val in self.experiment.sweep_values:
                params = self.experiment.build_params(val)
                r = run_single(algo, params, seed=seed)
                self.results.append(r)

                if verbose:
                    stable = f"{r['stable_chains']}/{r['total_chains']}"
                    print(f"    {self.experiment.sweep_key}={val:>4}: "
                          f"delay={r['avg_delay']:.4f}s, "
                          f"comp={r['avg_comp_delay']:.4f}s, "
                          f"comm={r['avg_comm_delay']:.4f}s, "
                          f"stable={stable}")

        return self.results

    def save_csv(self, output_dir="results"):
        """保存结果为 CSV"""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.experiment.name}.csv"
        filepath = os.path.join(output_dir, filename)

        if not self.results:
            return filepath

        keys = [k for k in self.results[0].keys() if k != "chain_details"]
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.results)

        return filepath

    def summary(self):
        """按算法分组打印摘要"""
        print(f"\nSummary: {self.experiment.name}")
        for algo in self.algorithms:
            algo_results = [r for r in self.results if r["algorithm"] == algo.name]
            if not algo_results:
                continue
            delays = [r["avg_delay"] for r in algo_results]
            comp = [r["avg_comp_delay"] for r in algo_results]
            comm = [r["avg_comm_delay"] for r in algo_results]
            print(f"  {algo.name}: "
                  f"delay=[{min(delays):.4f}, {max(delays):.4f}], "
                  f"avg_comp={np.mean(comp):.4f}, avg_comm={np.mean(comm):.4f}")
