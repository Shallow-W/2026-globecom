"""Experiment runner for executing deployment algorithms and analyzing results."""

from typing import Dict, List, Any, Optional

from core.topology.topology import Topology
from core.service.microservice import MicroService
from core.service.chain import ServiceChain
from core.service.deployment import DeploymentPlan
from core.queueing.analyzer import QueueingNetworkAnalyzer

from algorithms.deployment.baselines import (
    FirstFitDecreasingM,
    RandomDeploymentM,
    SimpleGreedyM,
    CoLocatedDeploymentM,
    LEGOAlgorithm,
    DRSAlgorithm,
)
from algorithms.deployment.ours import OurAlgorithm


class ExperimentRunner:
    """
    实验运行器

    流程:
    1. 加载配置
    2. 生成拓扑和服务
    3. 运行各种部署算法
    4. 用排队模型计算指标
    5. 汇总结果
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize experiment runner.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results: List[Dict[str, Any]] = []

    def run_single(self, algorithm_name: str,
                   topology: Topology,
                   services: Dict[str, MicroService],
                   chains: List[ServiceChain],
                   version_map: Dict[str, str] = None) -> Dict[str, Any]:
        """
        运行单个实验。

        Args:
            algorithm_name: Name of the deployment algorithm
            topology: Network topology
            services: Service configurations
            chains: Service chain list
            version_map: Optional mapping of service_id to version_id

        Returns:
            Dict containing experiment results:
                - algorithm: Algorithm name
                - deployment_plan: The deployment plan
                - chain_latencies: Latency for each chain
                - avg_latency: Average latency across chains
                - resource_utilization: CPU utilization per node
                - success_rate: Ratio of chains meeting latency constraints
                - deployment_cost: Number of nodes used
        """
        # Reset topology node state before each algorithm run
        # (some algorithms modify node.used_cpu/used_gpu directly)
        self._reset_topology_state(topology)

        # Get deployment algorithm
        alg = self._create_deployment_algorithm(algorithm_name)

        # Execute deployment
        plan = alg.deploy(topology, services, chains)

        # Queueing analysis
        analyzer = QueueingNetworkAnalyzer(topology, services)

        # Calculate latency for each chain (record arrival_rate for weighting)
        chain_latencies = []
        for chain in chains:
            lat = analyzer.calc_chain_latency(chain, plan, version_map)
            chain_latencies.append({
                "chain_id": chain.chain_id,
                "arrival_rate": chain.arrival_rate,
                **lat
            })

        # ============================================================
        # 与exp_2一致的到达率加权聚合
        # ============================================================
        total_arrival = sum(c["arrival_rate"] for c in chain_latencies)

        # 加权求和：每条链按 weight = rate / total_rate 加权
        weighted_lat = 0.0
        weighted_queuing = 0.0
        weighted_comm = 0.0
        weighted_penalty = 0.0

        for c, chain in zip(chain_latencies, chains):
            weight = chain.arrival_rate / total_arrival
            lat = c["total"]
            if lat == float('inf'):
                lat = 10000.0  # 与exp_2一致：inf给个大值但继续加权
            weighted_lat += lat * weight
            weighted_queuing += c["queuing"] * weight
            weighted_comm += c["communication"] * weight
            weighted_penalty += c.get("penalty", 0) * weight

        avg_lat = weighted_lat
        avg_queuing = weighted_queuing
        avg_communication = weighted_comm
        avg_processing = 0.0  # 已合并到 queuing

        # Success rate（与之前一致：满足延迟约束的链比例）
        success = 0
        for c, chain in zip(chain_latencies, chains):
            if c.get("unrealizable") or c["total"] == float('inf'):
                continue
            if c["total"] <= chain.max_latency:
                success += 1
        success_rate = success / len(chains) if chains else 0

        # Resource utilization (global memory utilization, 与exp_2一致)
        mem_util = analyzer.calc_mem_utilization(plan)

        # Deployment cost (number of nodes used)
        cost = len(set(n for (_, n) in plan.placement.keys()))

        return {
            "algorithm": algorithm_name,
            "deployment_plan": plan,
            "chain_latencies": chain_latencies,
            "avg_latency": avg_lat,
            "avg_queuing": avg_queuing,
            "avg_processing": avg_processing,
            "avg_communication": avg_communication,
            "total_penalty": weighted_penalty,
            "mem_utilization": mem_util,
            "success_rate": success_rate,
            "deployment_cost": cost
        }

    def run_comparison(self,
                       topology: Topology,
                       services: Dict[str, MicroService],
                       chains: List[ServiceChain],
                       algorithms: List[str]) -> List[Dict[str, Any]]:
        """
        运行多算法对比实验。

        Args:
            topology: Network topology
            services: Service configurations
            chains: Service chain list
            algorithms: List of algorithm names

        Returns:
            List of experiment results for each algorithm
        """
        results = []
        for alg_name in algorithms:
            result = self.run_single(alg_name, topology, services, chains)
            results.append(result)
        return results

    def run_perturbation(self,
                         base_config: Dict[str, Any],
                         param_name: str,
                         param_values: List[Any],
                         algorithms: List[str],
                         generator: 'DataGenerator') -> List[Dict[str, Any]]:
        """
        运行扰动实验。

        Varies a single parameter while keeping others constant.

        Args:
            base_config: Base configuration dict
            param_name: Name of parameter to vary
            param_values: List of values to test for the parameter
            algorithms: List of algorithm names to test
            generator: DataGenerator instance for generating test data

        Returns:
            List of experiment results with param and value fields added
        """
        results = []

        # Generate base data once
        base_topology, base_services, base_chains = generator.generate_all(base_config)

        # Keep topology and services fixed, only vary chain parameters
        for value in param_values:
            # Clone chains with perturbed parameter
            if param_name == "arrival_rate":
                # Only vary arrival_rate, keep chain structure same
                perturbed_chains = self._perturb_chains_arrival_rate(base_chains, value)
            elif param_name == "n_task_types":
                # Regenerate chains with different task types
                perturbed_chains = self._perturb_chains_task_types(base_chains, value, base_config)
            elif param_name == "chain_length":
                # Regenerate chains with different length
                perturbed_chains = self._perturb_chains_length(base_chains, int(value), base_config, generator)
            else:
                # For other parameters, regenerate normally
                config = base_config.copy()
                config[param_name] = value
                _, _, perturbed_chains = generator.generate_all(config)

            # Run comparison with perturbed chains
            exp_results = self.run_comparison(base_topology, base_services, perturbed_chains, algorithms)
            for r in exp_results:
                r["param"] = param_name
                r["value"] = value
            results.extend(exp_results)

        return results

    def _perturb_chains_arrival_rate(self, base_chains, total_arrival_rate: float):
        """Create new chains with different total arrival rate using Dirichlet distribution."""
        import numpy as np
        num_chains = len(base_chains)
        raw_rates = np.random.dirichlet(np.ones(num_chains)) * total_arrival_rate
        rates = np.maximum(1, np.round(raw_rates)).astype(int)
        rates[0] += int(total_arrival_rate) - np.sum(rates)

        perturbed_chains = []
        for chain, rate in zip(base_chains, rates):
            new_chain = ServiceChain(
                chain_id=chain.chain_id,
                services=chain.services.copy(),
                arrival_rate=float(rate),
                max_latency=chain.max_latency,
                task_type=chain.task_type
            )
            perturbed_chains.append(new_chain)
        return perturbed_chains

    def _perturb_chains_task_types(self, base_chains, n_types: int, config: Dict):
        """
        Create new chains with different number of task types.

        Chains are regenerated: each chain samples microservices from a pool of
        'n_types' distinct microservice IDs, keeping arrival_rate, max_latency,
        and chain_id from base_chains.
        """
        import random

        # 可用微服务池：s0, s1, ..., s{max-1}
        available_service_ids = [f"s{i}" for i in range(n_types)]

        perturbed_chains = []
        for chain in base_chains:
            length = len(chain.services)
            # 采样服务（有放回，与exp_2一致）
            selected = random.choices(available_service_ids, k=length)
            # task_type 随机分配（与exp_2保持一致）
            task_types = config.get("task_types",
                ["class_scene", "class_object", "room_layout", "jigsaw",
                 "segmentsemantic", "normal", "autoencoder"])
            new_task_type = random.choice(task_types)

            new_chain = ServiceChain(
                chain_id=chain.chain_id,
                services=selected,
                arrival_rate=chain.arrival_rate,
                max_latency=chain.max_latency,
                task_type=new_task_type
            )
            perturbed_chains.append(new_chain)
        return perturbed_chains

    def _perturb_chains_length(self, base_chains, new_length: int, config: Dict, generator):
        """Regenerate chains with different length (sampling with replacement, like exp_2)."""
        import random

        service_ids = [f"s{i}" for i in range(config.get("num_services", 10))]
        perturbed_chains = []

        for chain in base_chains:
            # 有放回采样（与exp_2的 random.choices 一致）
            new_services = random.choices(service_ids, k=new_length)
            new_chain = ServiceChain(
                chain_id=chain.chain_id,
                services=new_services,
                arrival_rate=chain.arrival_rate,
                max_latency=chain.max_latency,
                task_type=chain.task_type
            )
            perturbed_chains.append(new_chain)
        return perturbed_chains

    def _reset_topology_state(self, topology: Topology):
        """Reset node state in topology to initial values."""
        for node_id, node in topology.nodes.items():
            if hasattr(node, 'used_cpu'):
                node.used_cpu = 0
            if hasattr(node, 'used_gpu'):
                node.used_gpu = 0
            if hasattr(node, 'deployed_services'):
                node.deployed_services = set()

    def _create_deployment_algorithm(self, name: str):
        """
        Create a deployment algorithm instance by name.

        Args:
            name: Algorithm name (e.g., "ffd-m", "random-m", "cds-m", "greedy-m")

        Returns:
            DeploymentAlgorithm instance

        Raises:
            ValueError: If algorithm name is unknown
        """
        name_lower = name.lower()

        if name_lower == "ffd-m":
            return FirstFitDecreasingM()
        elif name_lower == "random-m":
            return RandomDeploymentM()
        elif name_lower == "greedy-m":
            return SimpleGreedyM()
        elif name_lower == "cds-m":
            return CoLocatedDeploymentM()
        elif name_lower == "lego":
            return LEGOAlgorithm()
        elif name_lower == "drs":
            return DRSAlgorithm()
        elif name_lower == "our":
            # Our算法需要excel_model_path
            excel_path = self.config.get("excel_model_path")
            if excel_path:
                return OurAlgorithm({"excel_model_path": excel_path})
            else:
                raise ValueError("Our algorithm requires excel_model_path in config")
        else:
            raise ValueError(f"Unknown algorithm: {name}. "
                           f"Available: ffd-m, random-m, greedy-m, cds-m, lego, our")
