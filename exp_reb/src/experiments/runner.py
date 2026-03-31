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
        # Get deployment algorithm
        alg = self._create_deployment_algorithm(algorithm_name)

        # Execute deployment
        plan = alg.deploy(topology, services, chains)

        # Queueing analysis
        analyzer = QueueingNetworkAnalyzer(topology, services)

        # Calculate latency for each chain
        chain_latencies = []
        for chain in chains:
            lat = analyzer.calc_chain_latency(chain, plan, version_map)
            chain_latencies.append({
                "chain_id": chain.chain_id,
                **lat
            })

        # Aggregate metrics
        total_lat = sum(c["total"] for c in chain_latencies)
        avg_lat = total_lat / len(chain_latencies) if chain_latencies else 0

        # Resource utilization
        util = analyzer.calc_resource_utilization(plan)

        # Success rate (chains meeting latency constraint)
        success = 0
        for c in chain_latencies:
            chain_constraint = next(
                (ch.max_latency for ch in chains if ch.chain_id == c["chain_id"]),
                float('inf')
            )
            if c["total"] <= chain_constraint:
                success += 1
        success_rate = success / len(chains) if chains else 0

        # Deployment cost (number of nodes used)
        cost = len(set(n for (_, n) in plan.placement.keys()))

        return {
            "algorithm": algorithm_name,
            "deployment_plan": plan,
            "chain_latencies": chain_latencies,
            "avg_latency": avg_lat,
            "resource_utilization": util,
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

    def _perturb_chains_arrival_rate(self, base_chains, arrival_rate: float):
        """Create new chains with different arrival rate but same structure."""
        from copy import deepcopy
        perturbed_chains = []
        for chain in base_chains:
            new_chain = ServiceChain(
                chain_id=chain.chain_id,
                services=chain.services.copy(),
                arrival_rate=arrival_rate,
                max_latency=chain.max_latency,
                task_type=chain.task_type
            )
            perturbed_chains.append(new_chain)
        return perturbed_chains

    def _perturb_chains_task_types(self, base_chains, n_types: int, config: Dict):
        """Create new chains with different number of task types."""
        task_types = config.get("task_types",
            ["class_scene", "class_object", "room_layout", "jigsaw",
             "segmentsemantic", "normal", "autoencoder"])
        selected_types = task_types[:int(n_types)]

        perturbed_chains = []
        for chain in base_chains:
            import random
            new_task_type = random.choice(selected_types)
            new_chain = ServiceChain(
                chain_id=chain.chain_id,
                services=chain.services.copy(),
                arrival_rate=chain.arrival_rate,
                max_latency=chain.max_latency,
                task_type=new_task_type
            )
            perturbed_chains.append(new_chain)
        return perturbed_chains

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
