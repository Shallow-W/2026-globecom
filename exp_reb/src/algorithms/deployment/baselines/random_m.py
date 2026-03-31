"""
Random deployment with fixed Model-M (Random-M).

Baseline algorithm that:
1. Calculates resource demand for each service using Model-M
2. Randomly selects nodes to deploy services
"""

import random
from typing import Dict, List, Any, Tuple
from algorithms.deployment.base import DeploymentAlgorithm


class DeploymentPlan:
    """
    Simple deployment plan for baseline algorithms.

    placement: {(service_id, node_id): {version_id: count}}
    """

    def __init__(self):
        self.placement: Dict[Tuple[str, str], Dict[str, int]] = {}

    def add(self, service_id: str, node_id: str, version_id: str, count: int = 1):
        """Add deployment instance."""
        key = (service_id, node_id)
        if key not in self.placement:
            self.placement[key] = {}
        self.placement[key][version_id] = self.placement[key].get(version_id, 0) + count

    def get_service_instances(self, service_id: str, version_id: str = None) -> int:
        """Get total instances of a service (optionally for specific version)."""
        if version_id:
            return sum(v.get(version_id, 0) for (s, _), v in self.placement.items() if s == service_id)
        return sum(sum(v.values()) for (s, _), v in self.placement.items() if s == service_id)

    def get_node_instances(self, node_id: str) -> Dict[str, Dict[str, int]]:
        """Get services deployed on a node."""
        return {s: v for (s, n), v in self.placement.items() if n == node_id}

    def get_version_usage(self) -> Dict[str, int]:
        """Count total instances per version."""
        usage: Dict[str, int] = {}
        for (s, n), versions in self.placement.items():
            for vid, count in versions.items():
                usage[vid] = usage.get(vid, 0) + count
        return usage

    def __repr__(self) -> str:
        return f"DeploymentPlan(placement={self.placement})"


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
               chains: List[Any]) -> DeploymentPlan:
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
                if service_id in chain.get('services', []):
                    total_rate += chain.get('arrival_rate', 0)

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
        node_ids = list(topology.get('nodes', {}).keys())
        for node_id in node_ids:
            node = topology['nodes'][node_id]
            node_remaining[node_id] = {
                'cpu': node.get('cpu_cores', node.get('cpu', 0)),
                'gpu': node.get('gpu_memory', node.get('gpu', 0)),
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
