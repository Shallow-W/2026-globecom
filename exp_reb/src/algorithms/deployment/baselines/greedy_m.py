"""
Simple greedy deployment with fixed Model-M (Greedy-M).

Baseline algorithm that:
1. Calculates resource demand for each service using Model-M
2. For each service, selects the node with most remaining resources
"""

from typing import Dict, List, Any
from algorithms.deployment.base import DeploymentAlgorithm
from core.service.deployment import DeploymentPlan


class SimpleGreedyM(DeploymentAlgorithm):
    """
    Simple greedy deployment with fixed Model-M.

    Strategy:
    1. Calculate resource demand for each service using Model-M parameters
    2. For each service, select the node with the most remaining resources
       that can accommodate the service
    3. This maximizes resource utilization packing

    Model-M parameters: mu=10, accuracy=0.53, cpu=1, gpu=1024
    """

    def deploy(self, topology: Any,
               services: Dict[str, Any],
               chains: List[Any]) -> 'DeploymentPlan':
        """
        Execute Greedy-M deployment.

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

        # Step 2: Sort services by total resource demand (descending)
        sorted_services = sorted(
            service_demand.items(),
            key=lambda x: -(x[1]["cpu"] + x[1]["gpu"])
        )

        # Step 3: Track remaining resources on each node
        node_remaining: Dict[str, Dict[str, int]] = {}
        nodes = topology.nodes if hasattr(topology, 'nodes') else topology.get('nodes', {})
        for node_id, node in nodes.items():
            cpu = node.cpu_cores if hasattr(node, 'cpu_cores') else node.get('cpu_cores', node.get('cpu', 0))
            gpu = node.gpu_memory if hasattr(node, 'gpu_memory') else node.get('gpu_memory', node.get('gpu', 0))
            node_remaining[node_id] = {
                'cpu': cpu,
                'gpu': gpu,
            }

        # Step 4: Greedy deployment - select node with most remaining resources
        for service_id, demand in sorted_services:
            cpu_needed = demand["cpu"]
            gpu_needed = demand["gpu"]

            # Find node with maximum remaining resources that can accommodate
            best_node = None
            best_remaining = -1

            for node_id, remaining in node_remaining.items():
                if remaining['cpu'] >= cpu_needed and remaining['gpu'] >= gpu_needed:
                    # Calculate total remaining resources
                    total_remaining = remaining['cpu'] + remaining['gpu']
                    if total_remaining > best_remaining:
                        best_remaining = total_remaining
                        best_node = node_id

            if best_node:
                # Deploy to best node
                plan.add(service_id, best_node, fixed_version, demand["instances"])
                node_remaining[best_node]['cpu'] -= cpu_needed
                node_remaining[best_node]['gpu'] -= gpu_needed

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

    algo = SimpleGreedyM()
    plan = algo.deploy(topo, services, chains)
    print(f"Greedy-M Plan: {plan}")
    print(f"Version usage: {plan.get_version_usage()}")
