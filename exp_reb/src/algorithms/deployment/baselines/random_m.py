"""
Random deployment with fixed Model-M (Random-M).

Baseline algorithm that:
1. Calculates resource demand for each service using Model-M
2. Randomly selects nodes to deploy services
"""

import random
from typing import Dict, List, Any
from algorithms.deployment.base import DeploymentAlgorithm
from core.service.deployment import DeploymentPlan


class RandomDeploymentM(DeploymentAlgorithm):
    """
    Random deployment with fixed Model-M.

    Strategy:
    1. Calculate resource demand for each service using Model-M parameters
    2. Shuffle services in random order
    3. For each service, randomly select a node that can accommodate it

    Model-M parameters: mu=10, accuracy=0.53, cpu=1, gpu=1024
    """

    def __init__(self, config: Dict[str, Any] = None, seed: int = None):
        """
        Initialize Random-M algorithm.

        Args:
            config: Optional configuration dict.
            seed: Random seed for reproducibility.
        """
        super().__init__(config)
        self.seed = seed if seed is not None else self.config.get("seed", 42)
        self.rng = random.Random(self.seed)

    def deploy(self, topology: Any,
               services: Dict[str, Any],
               chains: List[Any]) -> 'DeploymentPlan':
        """
        Execute Random-M deployment.

        Args:
            topology: Network topology with nodes dict.
            services: Service configurations {service_id: Service}.
            chains: Service chain list for calculating load.

        Returns:
            DeploymentPlan: Deployment plan.
        """
        plan = DeploymentPlan()
        fixed_version = "Model-M"

        # Model-M parameters
        model_m_cpu = 1
        model_m_gpu = 1024
        model_m_mu = 10

        # Step 1: Calculate resource demand for each service
        service_demand: Dict[str, Dict[str, Any]] = {}
        for service_id, svc in services.items():
            # Calculate total arrival rate for this service from all chains
            total_rate = 0.0
            for chain in chains:
                if hasattr(chain, 'services') and service_id in chain.services:
                    total_rate += chain.arrival_rate

            # Calculate required instances based on arrival rate and service rate
            if total_rate > 0 and model_m_mu > 0:
                instances = max(1, int(total_rate / model_m_mu) + 1)
            else:
                instances = 1

            cpu_needed = instances * model_m_cpu
            gpu_needed = instances * model_m_gpu

            service_demand[service_id] = {
                "version": fixed_version,
                "instances": instances,
                "cpu": cpu_needed,
                "gpu": gpu_needed,
            }

        # Step 2: Shuffle services in random order
        service_ids = list(service_demand.keys())
        self.rng.shuffle(service_ids)

        # Step 3: Track remaining resources on each node
        node_remaining: Dict[str, Dict[str, int]] = {}
        nodes = topology.nodes if hasattr(topology, 'nodes') else topology.get('nodes', {})
        node_ids = list(nodes.keys())
        for node_id in node_ids:
            node = nodes[node_id]
            cpu = node.cpu_cores if hasattr(node, 'cpu_cores') else node.get('cpu_cores', node.get('cpu', 0))
            gpu = node.gpu_memory if hasattr(node, 'gpu_memory') else node.get('gpu_memory', node.get('gpu', 0))
            node_remaining[node_id] = {
                'cpu': cpu,
                'gpu': gpu,
            }

        # Step 4: Random deployment
        for service_id in service_ids:
            demand = service_demand[service_id]
            cpu_needed = demand["cpu"]
            gpu_needed = demand["gpu"]

            # Find all nodes that can accommodate this service
            candidate_nodes = [
                node_id for node_id, remaining in node_remaining.items()
                if remaining['cpu'] >= cpu_needed and remaining['gpu'] >= gpu_needed
            ]

            if candidate_nodes:
                # Randomly select one
                selected_node = self.rng.choice(candidate_nodes)
                plan.add(service_id, selected_node, fixed_version, demand["instances"])
                node_remaining[selected_node]['cpu'] -= cpu_needed
                node_remaining[selected_node]['gpu'] -= gpu_needed

        return plan


if __name__ == "__main__":
    # Simple test
    topo = {
        "nodes": {
            "n0": {"cpu_cores": 16, "gpu_memory": 4096},
            "n1": {"cpu_cores": 16, "gpu_memory": 4096},
            "n2": {"cpu_cores": 8, "gpu_memory": 2048},
        }
    }

    services = {
        "s0": {"service_id": "s0"},
        "s1": {"service_id": "s1"},
        "s2": {"service_id": "s2"},
    }

    chains = [
        {"chain_id": "c0", "services": ["s0", "s1"], "arrival_rate": 10},
        {"chain_id": "c1", "services": ["s1", "s2"], "arrival_rate": 5},
    ]

    algo = RandomDeploymentM(seed=42)
    plan = algo.deploy(topo, services, chains)
    print(f"Random-M Plan: {plan}")
    print(f"Version usage: {plan.get_version_usage()}")
