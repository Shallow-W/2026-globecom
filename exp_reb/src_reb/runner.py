# ==========================================
# 实验运行层：使用旧框架的baseline + 新框架的GA
# ==========================================

import sys
import os
import random
import numpy as np
from typing import List, Dict, Any, Optional

# 导入旧框架的类（用于baseline算法）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core.topology.node import Node
from core.topology.link import Link
from core.topology.topology import Topology
from core.service.microservice import MicroService, ModelVersion
from core.service.chain import ServiceChain
from core.service.deployment import DeploymentPlan
from core.queueing.analyzer import QueueingNetworkAnalyzer

# 导入baseline算法
from algorithms.deployment.baselines import (
    FirstFitDecreasingM,
    RandomDeploymentM,
    SimpleGreedyM,
    CoLocatedDeploymentM,
    LEGOAlgorithm,
    DRSAlgorithm,
)

# 导入我们的GA
from .ga import GeneticAlgorithm
from .config import N_NODES, N_VERSIONS
from .data_model import ModelLibrary


class DataGenerator:
    """
    数据生成器：生成旧框架对象类型

    生成: Topology, Dict[str, MicroService], List[ServiceChain], ModelLibrary
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self._mock_library: Optional[ModelLibrary] = None

    def generate_all(self, config: Dict[str, Any]) -> tuple:
        """
        生成所有数据（使用旧框架对象类型）

        Returns:
            (topology, services, chains, model_library)
        """
        num_nodes = config.get("num_nodes", N_NODES)
        num_types = config.get("num_types", config.get("num_task_types", 10))
        chain_length = config.get("chain_length", 4)
        num_chains = config.get("num_chains", 4)
        total_arrival_rate = config.get("total_arrival_rate", 200)

        # 拓扑
        topology = self._generate_topology(num_nodes)

        # 服务
        services = self._generate_services(num_types)

        # 服务链
        chains = self._generate_chains(
            num_chains=num_chains,
            num_types=num_types,
            chain_length=chain_length,
            total_arrival_rate=total_arrival_rate,
            max_latency=config.get("max_latency", 1000.0)
        )

        # 模型库
        model_lib = self._get_model_library(config)

        return topology, services, chains, model_lib

    def _generate_topology(self, num_nodes: int) -> Topology:
        """生成网络拓扑"""
        topo = Topology()
        for i in range(num_nodes):
            topo.add_node(Node(f"edge_{i}", cpu_cores=32, gpu_memory=4096))
        # 全连接，链路延迟20ms
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    topo.add_link(Link(f"edge_{i}", f"edge_{j}", delay=0.02))
        return topo

    def _generate_services(self, num_types: int) -> Dict[str, MicroService]:
        """生成服务定义（版本ID与baseline兼容：Model-H/M/L）"""
        # mu = flops / NODE_FLOPS_CAPACITY (requests per second)
        # NODE_FLOPS_CAPACITY = 200 * 10^9 = 200 GFLOPs/s
        # Model-H (10M FLOPs): 10M / 200G = 0.00005s = 0.05ms -> mu=20000 req/s
        # Model-M (5M FLOPs): 5M / 200G = 0.000025s = 0.025ms -> mu=40000 req/s
        # Model-L (1M FLOPs): 1M / 200G = 0.000005s = 0.005ms -> mu=200000 req/s
        version_names = ["Model-H", "Model-M", "Model-L"]
        version_mus = [20000.0, 40000.0, 200000.0]  # req/s
        version_params = [10_000_000, 5_000_000, 1_000_000]  # 参数量
        version_accuracies = [0.62, 0.53, 0.45]

        services = {}
        for i in range(num_types):
            svc = MicroService(service_id=f"s{i}")
            svc.versions = {}
            for v_idx, vname in enumerate(version_names):
                mv = ModelVersion(
                    version_id=vname,
                    mu=version_mus[v_idx],
                    accuracy=version_accuracies[v_idx],
                    cpu_per_instance=1,
                    gpu_per_instance=1024,
                    model_params=version_params[v_idx],
                )
                svc.versions[vname] = mv
            svc.requires_gpu = any(v.gpu_per_instance > 0 for v in svc.versions.values())
            services[f"s{i}"] = svc
        return services

    def _generate_chains(self,
                        num_chains: int,
                        num_types: int,
                        chain_length: int,
                        total_arrival_rate: float,
                        max_latency: float = 1000.0) -> List[ServiceChain]:
        """生成服务链（与exp_2一致: Dirichlet到达率 + choices有放回采样）"""
        service_pool = [f"s{i}" for i in range(num_types)]
        task_types = ["class_scene", "class_object", "room_layout", "jigsaw",
                      "segmentsemantic", "normal", "autoencoder"]

        # Dirichlet分配到达率
        raw_rates = np.random.dirichlet(np.ones(num_chains)) * total_arrival_rate
        rates = np.maximum(1, np.round(raw_rates)).astype(int)
        rates[0] += int(total_arrival_rate) - np.sum(rates)

        chains = []
        for i in range(num_chains):
            # 有放回采样
            services = self.rng.choices(service_pool, k=int(chain_length))
            chains.append(ServiceChain(
                chain_id=f"chain_{i}",
                services=services,
                arrival_rate=float(rates[i]),
                max_latency=max_latency,
                task_type=self.rng.choice(task_types)
            ))
        return chains

    def _get_model_library(self, config: Dict[str, Any]) -> ModelLibrary:
        """获取模型库"""
        excel_path = config.get("excel_model_path")
        if excel_path and os.path.exists(excel_path):
            lib = ModelLibrary(excel_path)
            lib.load_from_excel(excel_path, max_required_tasks=80)
            return lib
        else:
            if self._mock_library is None:
                self._mock_library = ModelLibrary()
                self._mock_library._generate_mock_data(80)
            return self._mock_library


class ExperimentRunner:
    """
    实验运行器

    - our: 使用新框架的GA算法（evaluator.py）
    - baseline: 使用旧框架的算法 + QueueingNetworkAnalyzer
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rng = random.Random(config.get("seed", 42))

        # 旧框架分析器（用于评估所有算法）
        self._analyzer: Optional[QueueingNetworkAnalyzer] = None

    def _get_analyzer(self, topology: Topology,
                      services: Dict[str, MicroService]) -> QueueingNetworkAnalyzer:
        """获取或创建分析器"""
        if self._analyzer is None:
            self._analyzer = QueueingNetworkAnalyzer(topology, services)
        else:
            # 更新引用
            self._analyzer.topology = topology
            self._analyzer.services = services
        return self._analyzer

    def _evaluate_plan(self,
                       plan: DeploymentPlan,
                       topology: Topology,
                       services: Dict[str, MicroService],
                       chains: List[ServiceChain]) -> Dict[str, Any]:
        """评估一个部署方案（使用旧框架的QueueingNetworkAnalyzer）"""
        analyzer = self._get_analyzer(topology, services)
        total_arrival = sum(c.arrival_rate for c in chains)

        # 计算每条链的延迟
        chain_results = []
        for chain in chains:
            lat = analyzer.calc_chain_latency(chain, plan)
            weight = chain.arrival_rate / total_arrival

            lat_val = lat["total"]
            if lat_val == float('inf'):
                lat_val = 10000.0

            chain_results.append({
                "chain_id": chain.chain_id,
                "arrival_rate": chain.arrival_rate,
                "weight": weight,
                "queuing_delay": lat["queuing"],
                "comm_delay": lat["communication"],
                "total_delay": lat_val,
                "penalty": lat.get("penalty", 0),
                "success": lat_val <= chain.max_latency and lat_val < 10000,
            })

        # 加权聚合
        avg_latency = sum(r["total_delay"] * r["weight"] for r in chain_results)
        avg_queuing = sum(r["queuing_delay"] * r["weight"] for r in chain_results)
        avg_comm = sum(r["comm_delay"] * r["weight"] for r in chain_results)
        total_penalty = sum(r["penalty"] * r["weight"] for r in chain_results)
        success_rate = sum(1 for r in chain_results if r["success"]) / len(chain_results)

        # 内存利用率
        mem_util = analyzer.calc_mem_utilization(plan)

        # 部署代价
        cost = len(set(n for (_, n) in plan.placement.keys()))

        return {
            "avg_latency": avg_latency,
            "avg_queuing": avg_queuing,
            "avg_communication": avg_comm,
            "total_penalty": total_penalty,
            "mem_utilization": mem_util,
            "success_rate": success_rate,
            "deployment_cost": cost,
            "chain_results": chain_results,
        }

    def _run_our_algorithm(self,
                           topology: Topology,
                           services: Dict[str, MicroService],
                           chains: List[ServiceChain],
                           model_library: ModelLibrary) -> Dict[str, Any]:
        """运行我们的GA算法"""
        config = self.config
        num_chains = len(chains)
        total_arrival = sum(c.arrival_rate for c in chains)
        chain_length = config.get("chain_length", 4)

        service_ids = list(services.keys())

        # 代理知识
        proxy_knowledge = {}
        for s_idx, s_id in enumerate(service_ids):
            versions = model_library.get_versions(s_id)
            if len(versions) > 0:
                best_v = max(range(len(versions)), key=lambda v: versions[v].normalized_qos)
                proxy_knowledge[s_idx] = best_v

        ga = GeneticAlgorithm(
            num_services=len(service_ids),
            num_nodes=len(topology.nodes),
            num_versions=N_VERSIONS,
            pop_size=config.get("pop_size", 40),
            generations=config.get("generations", 50),
            seed=config.get("seed", 42)
        )

        # 每条链独立优化
        chain_results = []
        for chain in chains:
            best = ga.run(
                model_library=model_library,
                service_ids=service_ids,
                chain_length=chain_length,
                arrival_rate=chain.arrival_rate,
                proxy_knowledge=proxy_knowledge
            )
            chain_results.append({
                "chain_id": chain.chain_id,
                "arrival_rate": chain.arrival_rate,
                "weight": chain.arrival_rate / total_arrival,
                "queuing_delay": best.queuing_delay,
                "comm_delay": best.comm_delay,
                "total_delay": best.queuing_delay + best.comm_delay,
                "penalty": best.penalty,
                "success": (best.queuing_delay + best.comm_delay) <= chain.max_latency,
            })

        # 加权聚合
        avg_latency = sum(r["total_delay"] * r["weight"] for r in chain_results)
        avg_queuing = sum(r["queuing_delay"] * r["weight"] for r in chain_results)
        avg_comm = sum(r["comm_delay"] * r["weight"] for r in chain_results)
        total_penalty = sum(r["penalty"] * r["weight"] for r in chain_results)
        success_rate = sum(1 for r in chain_results if r["success"]) / len(chain_results)
        avg_mem = sum(best.mem_util for _ in chain_results) / len(chain_results)

        return {
            "algorithm": "our",
            "avg_latency": avg_latency,
            "avg_queuing": avg_queuing,
            "avg_communication": avg_comm,
            "total_penalty": total_penalty,
            "mem_utilization": avg_mem,
            "success_rate": success_rate,
            "deployment_cost": sum(r["total_delay"] > 0 for r in chain_results),
            "chain_results": chain_results,
        }

    def run_single_algorithm(self,
                              algorithm_name: str,
                              topology: Topology,
                              services: Dict[str, MicroService],
                              chains: List[ServiceChain],
                              model_library: ModelLibrary) -> Dict[str, Any]:
        """运行单个算法"""
        alg_lower = algorithm_name.lower()

        # 我们的GA算法
        if alg_lower == "our":
            return self._run_our_algorithm(topology, services, chains, model_library)

        # Baseline算法
        alg = self._create_baseline_algorithm(alg_lower)

        # 重置拓扑状态
        self._reset_topology_state(topology)

        # 执行部署
        plan = alg.deploy(topology, services, chains)

        # 评估
        result = self._evaluate_plan(plan, topology, services, chains)
        result["algorithm"] = algorithm_name
        result["deployment_plan"] = plan

        return result

    def _create_baseline_algorithm(self, name: str):
        """创建baseline算法实例"""
        if name == "ffd-m":
            return FirstFitDecreasingM()
        elif name == "random-m":
            return RandomDeploymentM()
        elif name == "greedy-m":
            return SimpleGreedyM()
        elif name == "cds-m":
            return CoLocatedDeploymentM()
        elif name == "lego":
            return LEGOAlgorithm()
        elif name == "drs":
            return DRSAlgorithm()
        else:
            raise ValueError(f"Unknown algorithm: {name}")

    def _reset_topology_state(self, topology: Topology):
        """重置拓扑节点状态"""
        for node_id, node in topology.nodes.items():
            if hasattr(node, 'used_cpu'):
                node.used_cpu = 0
            if hasattr(node, 'used_gpu'):
                node.used_gpu = 0
            if hasattr(node, 'deployed_services'):
                node.deployed_services = set()

    def run_comparison(self,
                       topology: Topology,
                       services: Dict[str, MicroService],
                       chains: List[ServiceChain],
                       model_library: ModelLibrary,
                       algorithms: List[str]) -> List[Dict[str, Any]]:
        """运行多算法对比"""
        results = []
        # 重置分析器
        self._analyzer = None
        for alg in algorithms:
            result = self.run_single_algorithm(alg, topology, services, chains, model_library)
            results.append(result)
        return results

    def run_perturbation(self,
                         base_config: Dict[str, Any],
                         param_name: str,
                         param_values: List[Any],
                         algorithms: List[str]) -> List[Dict[str, Any]]:
        """运行扰动实验"""
        from .config import EXPERIMENTS

        results = []
        generator = DataGenerator(seed=base_config.get("seed", 42))

        fixed = EXPERIMENTS.get(param_name, {}).get("fixed", {})

        for value in param_values:
            # 构建当前配置
            config = base_config.copy()
            config.update(fixed)
            config[param_name] = int(value) if param_name == "num_types" or param_name == "chain_length" else float(value)

            # 生成新数据
            topo, servs, chns, lib = generator.generate_all(config)

            # 重置分析器
            self._analyzer = None

            # 运行对比
            alg_results = self.run_comparison(topo, servs, chns, lib, algorithms)
            for r in alg_results:
                r["param"] = param_name
                r["value"] = value
            results.extend(alg_results)

        return results
