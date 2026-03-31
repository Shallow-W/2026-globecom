"""
Co-located deployment with fixed Model-M (CDS-M).

Baseline algorithm that:
1. Groups services by service chain
2. Deploys all services in a chain to the same node (if possible)
3. Uses fixed Model-M parameters
"""

from typing import Dict, List, Any, Set
from algorithms.deployment.base import DeploymentAlgorithm
from core.service.deployment import DeploymentPlan


class CoLocatedDeploymentM(DeploymentAlgorithm):
    """
    Co-located Deployment with fixed Model-M.

    Strategy:
    1. Group services by service chain
    2. For each chain, find a node that can accommodate ALL services in the chain
    3. Deploy all chain services to the same node to minimize communication delay
    4. If no single node can fit the entire chain, deploy using First Fit Decreasing

    Model-M parameters: mu=10, accuracy=0.53, cpu=1, gpu=1024
    """

    def deploy(self, topology: Any,
               services: Dict[str, Any],
               chains: List[Any]) -> 'DeploymentPlan':
        """
        Execute CDS-M deployment.

        Args:
            topology: Network topology with nodes dict.
            services: Service configurations {service_id: Service}.
            chains: Service chain list.

        Returns:
            DeploymentPlan: Deployment plan.
        """
        plan = DeploymentPlan()
        fixed_version = "Model-M"

        # Model-M parameters
        model_m_cpu = 1
        model_m_gpu = 1024
        model_m_mu = 10

        # Track remaining resources on each node
        node_remaining: Dict[str, Dict[str, int]] = {}
        nodes = topology.nodes if hasattr(topology, 'nodes') else topology.get('nodes', {})
        for node_id, node in nodes.items():
            cpu = node.cpu_cores if hasattr(node, 'cpu_cores') else node.get('cpu_cores', node.get('cpu', 0))
            gpu = node.gpu_memory if hasattr(node, 'gpu_memory') else node.get('gpu_memory', node.get('gpu', 0))
            node_remaining[node_id] = {
                'cpu': cpu,
                'gpu': gpu,
            }

        # Track which services have been deployed
        deployed_services: Set[str] = set()

        # Step 1: Group services by chain and calculate chain resource demand
        chain_services: Dict[str, Dict[str, Any]] = {}
        for chain in chains:
            chain_id = chain.chain_id if hasattr(chain, 'chain_id') else id(chain)
            chain_service_list = chain.services if hasattr(chain, 'services') else chain.get('services', [])

            # Calculate total arrival rate for this chain
            total_rate = chain.arrival_rate if hasattr(chain, 'arrival_rate') else chain.get('arrival_rate', 0)

            # Calculate resource demand for each service in the chain
            service_demands = {}
            for service_id in chain_service_list:
                if service_id in deployed_services:
                    continue

                # Calculate required instances
                if total_rate > 0 and model_m_mu > 0:
                    instances = max(1, int(total_rate / model_m_mu) + 1)
                else:
                    instances = 1

                cpu_needed = instances * model_m_cpu
                gpu_needed = instances * model_m_gpu

                service_demands[service_id] = {
                    "instances": instances,
                    "cpu": cpu_needed,
                    "gpu": gpu_needed,
                }

            chain_services[chain_id] = {
                "services": chain_service_list,
                "demands": service_demands,
                "arrival_rate": total_rate,
            }

        # Step 2: Try to co-locate each chain's services on a single node
        for chain_id, chain_data in chain_services.items():
            demands = chain_data["demands"]
            if not demands:
                continue

            # Calculate total chain demand
            total_cpu = sum(d["cpu"] for d in demands.values())
            total_gpu = sum(d["gpu"] for d in demands.values())

            # Find a node that can accommodate the entire chain
            coLocated_node = None
            for node_id, remaining in node_remaining.items():
                if remaining['cpu'] >= total_cpu and remaining['gpu'] >= total_gpu:
                    coLocated_node = node_id
                    break

            if coLocated_node:
                # Deploy all chain services to the same node
                for service_id, demand in demands.items():
                    plan.add(service_id, coLocated_node, fixed_version, demand["instances"])
                    node_remaining[coLocated_node]['cpu'] -= demand["cpu"]
                    node_remaining[coLocated_node]['gpu'] -= demand["gpu"]
                    deployed_services.add(service_id)
            else:
                # Fallback: First Fit Decreasing for remaining services
                self._deploy_ffd(
                    plan, demands, node_remaining,
                    deployed_services, fixed_version,
                    model_m_cpu, model_m_gpu
                )

        # Step 3: Deploy any remaining services that weren't in chains
        all_service_ids = set(services.keys())
        remaining_services = all_service_ids - deployed_services

        if remaining_services:
            remaining_demands = {}
            for service_id in remaining_services:
                # Calculate demand for orphan services
                total_rate = 0.0
                for chain in chains:
                    if hasattr(chain, 'services') and service_id in chain.services:
                        total_rate += chain.arrival_rate

                if total_rate > 0 and model_m_mu > 0:
                    instances = max(1, int(total_rate / model_m_mu) + 1)
                else:
                    instances = 1

                remaining_demands[service_id] = {
                    "instances": instances,
                    "cpu": instances * model_m_cpu,
                    "gpu": instances * model_m_gpu,
                }

            self._deploy_ffd(
                plan, remaining_demands, node_remaining,
                deployed_services, fixed_version,
                model_m_cpu, model_m_gpu
            )

        return plan

    def _deploy_ffd(self, plan: DeploymentPlan,
                    demands: Dict[str, Dict[str, Any]],
                    node_remaining: Dict[str, Dict[str, int]],
                    deployed: Set[str],
                    version: str,
                    cpu_per_inst: int, gpu_per_inst: int):
        """
        Helper: First Fit Decreasing deployment for a set of services.
        """
        # Sort by total resource demand (descending)
        sorted_demands = sorted(
            demands.items(),
            key=lambda x: -(x[1]["cpu"] + x[1]["gpu"])
        )

        for service_id, demand in sorted_demands:
            if service_id in deployed:
                continue

            cpu_needed = demand["cpu"]
            gpu_needed = demand["gpu"]

            # Find first node that can accommodate
            for node_id, remaining in node_remaining.items():
                if remaining['cpu'] >= cpu_needed and remaining['gpu'] >= gpu_needed:
                    plan.add(service_id, node_id, version, demand["instances"])
                    remaining['cpu'] -= cpu_needed
                    remaining['gpu'] -= gpu_needed
                    deployed.add(service_id)
                    break


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
        "s3": {"service_id": "s3"},
    }

    chains = [
        {"chain_id": "c0", "services": ["s0", "s1"], "arrival_rate": 10},
        {"chain_id": "c1", "services": ["s2", "s3"], "arrival_rate": 5},
    ]

    algo = CoLocatedDeploymentM()
    plan = algo.deploy(topo, services, chains)
    print(f"CDS-M Plan: {plan}")
    print(f"Version usage: {plan.get_version_usage()}")

    # Check co-location
    for node_id in topo["nodes"]:
        instances = plan.get_node_instances(node_id)
        if instances:
            print(f"Node {node_id}: {instances}")
