"""DRS baseline algorithm - Dynamic Resource Scheduling."""

from typing import Dict, List, Any, Optional
import random

from algorithms.deployment.base import DeploymentAlgorithm
from core.topology.topology import Topology
from core.service.microservice import MicroService
from core.service.chain import ServiceChain
from core.service.deployment import DeploymentPlan


class DRSAlgorithm(DeploymentAlgorithm):
    """
    DRS (Dynamic Resource Scheduling) Baseline Algorithm.

    Based on the paper:
    - Has probabilistic request routing
    - Requires searching in large feasible space
    - Uses iterative optimization

    Strategy:
    1. Initialize with a random feasible deployment
    2. Iteratively improve by moving services to better nodes
    3. Consider both resource utilization and communication cost
    4. Fixed Model-M version for all deployments
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DRS algorithm.

        Args:
            config: Optional configuration dict with:
                - max_iterations: Maximum iterations (default: 50)
                - seed: Random seed
        """
        super().__init__(config)
        self.fixed_version = "Model-M"
        self.max_iterations = config.get("max_iterations", 50) if config else 50
        self.seed = config.get("seed", random.randint(0, 10000)) if config else random.randint(0, 10000)

    def deploy(self, topology: Topology,
              services: Dict[str, MicroService],
              chains: List[ServiceChain]) -> DeploymentPlan:
        """
        Execute DRS deployment algorithm.

        Args:
            topology: Network topology.
            services: Service configurations.
            chains: Service chain list.

        Returns:
            DeploymentPlan: Deployment plan.
        """
        rng = random.Random(self.seed)
        plan = DeploymentPlan()
        version = self.fixed_version

        # Get all service IDs
        all_services = list(services.keys())

        # Step 1: Initialize - randomly deploy services to feasible nodes
        self._initialize_deployment(plan, topology, services, chains, version, rng)

        # Step 2: Iterative improvement
        for iteration in range(self.max_iterations):
            improved = self._iterate(plan, topology, services, version, rng)
            if not improved:
                break

        return plan

    def _initialize_deployment(self, plan: DeploymentPlan,
                              topology: Topology,
                              services: Dict[str, MicroService],
                              chains: List[ServiceChain],
                              version: str,
                              rng: random.Random):
        """Initialize with random feasible deployment."""
        # Calculate total arrival rate per service
        service_rates = {}
        for chain in chains:
            for service_id in chain.services:
                service_rates[service_id] = service_rates.get(service_id, 0) + chain.arrival_rate

        for service_id, service in services.items():
            ver = service.get_version(version)
            if not ver:
                continue

            cpu_needed = ver.cpu_per_instance
            gpu_needed = ver.gpu_per_instance

            # Calculate number of instances needed based on arrival rate
            total_rate = service_rates.get(service_id, 0)
            # μ = ver.mu (requests per second per instance)
            # Need ceil(λ/μ) + margin instances to keep ρ < 0.9
            if ver.mu > 0:
                base_instances = int(total_rate / ver.mu) + 1
                # Add margin to keep utilization below 90%
                margin = 2 if total_rate > 20 else 1
                min_instances = max(1, base_instances + margin)
            else:
                min_instances = 1

            # Find all feasible nodes
            feasible_nodes = [
                node_id for node_id, node in topology.nodes.items()
                if node.can_deploy(cpu_needed * min_instances, gpu_needed * min_instances)
            ]

            if feasible_nodes:
                # Randomly select a node
                selected_node = rng.choice(feasible_nodes)
                plan.add(service_id, selected_node, version, min_instances)

                # Update node state
                node = topology.nodes[selected_node]
                node.used_cpu += cpu_needed * min_instances
                node.used_gpu += gpu_needed * min_instances

    def _iterate(self, plan: DeploymentPlan,
                topology: Topology,
                services: Dict[str, MicroService],
                version: str,
                rng: random.Random) -> bool:
        """
        Single iteration of improvement.

        Returns:
            True if improvement was made, False otherwise.
        """
        # Randomly select a service to potentially move
        deployed_services = list(set(s for (s, _), v in plan.placement.items()
                                    if v and sum(v.values()) > 0))
        if not deployed_services:
            return False

        service_id = rng.choice(deployed_services)

        # Find current node
        current_node = None
        for (s, n), v in plan.placement.items():
            if s == service_id and v and sum(v.values()) > 0:
                current_node = n
                break

        if not current_node:
            return False

        ver = services[service_id].get_version(version)
        if not ver:
            return False

        cpu_needed = ver.cpu_per_instance
        gpu_needed = ver.gpu_per_instance

        # Find all other feasible nodes
        feasible_nodes = [
            node_id for node_id, node in topology.nodes.items()
            if node_id != current_node and node.can_deploy(cpu_needed, gpu_needed)
        ]

        if not feasible_nodes:
            return False

        # Randomly select a target node
        target_node = rng.choice(feasible_nodes)

        # Calculate current and target load
        current_load = self._estimate_node_load(current_node, topology)
        target_load = self._estimate_node_load(target_node, topology)

        # Prefer moving to less loaded node
        if target_load < current_load:
            # Get current instances on this node
            current_instances = 0
            if (service_id, current_node) in plan.placement:
                current_instances = plan.placement[(service_id, current_node)].get(version, 0)

            if current_instances > 0:
                # Move all instances to target
                plan.placement[(service_id, current_node)][version] = 0
                plan.add(service_id, target_node, version, current_instances)
                return True

        return False

    def _estimate_node_load(self, node_id: str, topology: Topology) -> float:
        """Estimate node load (0-1 scale)."""
        node = topology.nodes.get(node_id)
        if not node:
            return 1.0

        cpu_load = node.used_cpu / node.cpu_cores if node.cpu_cores > 0 else 1.0
        gpu_load = node.used_gpu / node.gpu_memory if node.gpu_memory > 0 else 0.0

        return max(cpu_load, gpu_load)
