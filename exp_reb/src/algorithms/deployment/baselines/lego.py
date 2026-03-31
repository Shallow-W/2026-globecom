"""LEGO baseline algorithm - Locality Enhanced Greedy Optimization."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from algorithms.deployment.base import DeploymentAlgorithm
from core.topology.topology import Topology
from core.service.microservice import MicroService
from core.service.chain import ServiceChain
from core.service.deployment import DeploymentPlan


class LEGOAlgorithm(DeploymentAlgorithm):
    """
    LEGO (Locality Enhanced Greedy Optimization) Baseline Algorithm.

    Strategy:
    1. Prefer deploying services to nodes that already have services from the same chain
    2. Use greedy selection based on node load and communication cost
    3. Fixed Model-M version for all deployments

    This algorithm minimizes inter-node communication by preferring
    co-location of related services, while using greedy node selection.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LEGO algorithm.

        Args:
            config: Optional configuration dict.
        """
        super().__init__(config)
        self.fixed_version = "Model-M"

    def deploy(self, topology: Topology,
              services: Dict[str, MicroService],
              chains: List[ServiceChain]) -> DeploymentPlan:
        """
        Execute LEGO deployment algorithm.

        Args:
            topology: Network topology.
            services: Service configurations.
            chains: Service chain list.

        Returns:
            DeploymentPlan: Deployment plan.
        """
        plan = DeploymentPlan()
        version = self.fixed_version

        # Calculate total arrival rate per service
        service_rates = {}
        for chain in chains:
            for service_id in chain.services:
                service_rates[service_id] = service_rates.get(service_id, 0) + chain.arrival_rate

        # Build a map of chain_id -> set of nodes that already have services from this chain
        chain_nodes: Dict[str, set] = {c.chain_id: set() for c in chains}

        # Track deployed services to avoid redeploying
        deployed_services = set()

        # Process each chain
        for chain in chains:
            for service_id in chain.services:
                # Skip if service already deployed
                if service_id in deployed_services:
                    continue

                # Find best node for this service
                node_id = self._select_node(
                    service_id=service_id,
                    chain_id=chain.chain_id,
                    chain_nodes=chain_nodes,
                    topology=topology,
                    services=services,
                    plan=plan
                )

                if node_id:
                    # Calculate instances needed
                    ver = services[service_id].get_version(version)
                    total_rate = service_rates.get(service_id, 0)
                    if ver and ver.mu > 0:
                        base_instances = int(total_rate / ver.mu) + 1
                        margin = 2 if total_rate > 20 else 1
                        instances = max(1, base_instances + margin)
                    else:
                        instances = 1

                    # Calculate resource requirements for all instances
                    cpu_for_instances = ver.cpu_per_instance * instances if ver else instances
                    gpu_for_instances = ver.gpu_per_instance * instances if ver else 0

                    # Recheck: node must have capacity for ALL instances
                    node = topology.nodes[node_id]
                    if not node.can_deploy(cpu_for_instances, gpu_for_instances):
                        # Not enough capacity, find another node
                        node_id = self._select_node_with_capacity(
                            service_id, cpu_for_instances, gpu_for_instances,
                            topology, chain_nodes, chain.chain_id
                        )
                        if not node_id:
                            continue  # Skip this service if no node can hold all instances

                    # Add deployment
                    plan.add(service_id, node_id, version, instances)
                    chain_nodes[chain.chain_id].add(node_id)

                    # Update node state
                    node = topology.nodes[node_id]
                    if not hasattr(node, 'deployed_services'):
                        node.deployed_services = set()
                    node.deployed_services.add(service_id)
                    node.used_cpu += ver.cpu_per_instance * instances if ver else instances
                    node.used_gpu += ver.gpu_per_instance * instances if ver else 0

                    # Mark service as deployed
                    deployed_services.add(service_id)

        return plan

    def _select_node(self, service_id: str,
                    chain_id: str,
                    chain_nodes: Dict[str, set],
                    topology: Topology,
                    services: Dict[str, MicroService],
                    plan: DeploymentPlan) -> Optional[str]:
        """
        Select best node for service deployment.

        Strategy:
        1. Prefer nodes that already have services from the same chain (locality)
        2. Among those, prefer nodes with lower load
        3. If no locality match, select least loaded node with sufficient resources
        """
        service = services.get(service_id)
        if not service:
            return None

        version = service.get_version(self.fixed_version)
        if not version:
            return None

        cpu_needed = version.cpu_per_instance
        gpu_needed = version.gpu_per_instance

        # First, try to find a node with locality (same chain already deployed)
        locality_nodes = chain_nodes.get(chain_id, set())
        if locality_nodes:
            # Check each locality node for resource availability
            best_locality_node = None
            best_load = float('inf')

            for node_id in locality_nodes:
                node = topology.nodes.get(node_id)
                if not node:
                    continue

                # Check if node has capacity
                if not node.can_deploy(cpu_needed, gpu_needed):
                    continue

                # Calculate load (lower is better)
                load = getattr(node, 'load', 0.5)
                if load < best_load:
                    best_load = load
                    best_locality_node = node_id

            if best_locality_node:
                return best_locality_node

        # No locality match or no capacity, select least loaded node with resources
        best_node = None
        best_score = float('inf')

        for node_id, node in topology.nodes.items():
            if not node.can_deploy(cpu_needed, gpu_needed):
                continue

            # Score: prefer lower load and prefer nodes with existing chain services
            load = getattr(node, 'load', 0.5)
            has_chain_service = 1 if node_id in locality_nodes else 0
            # Lower score is better: prioritize locality, then load
            score = -has_chain_service * 1000 + load

            if score < best_score:
                best_score = score
                best_node = node_id

        return best_node

    def _select_node_with_capacity(self, service_id: str,
                                     cpu_required: float,
                                     gpu_required: float,
                                     topology: Topology,
                                     chain_nodes: Dict[str, set],
                                     chain_id: str) -> Optional[str]:
        """
        Select a node with capacity for the required instances.
        Used when initial locality node doesn't have enough capacity.
        """
        best_node = None
        best_score = float('inf')

        for node_id, node in topology.nodes.items():
            if not node.can_deploy(cpu_required, gpu_required):
                continue

            load = getattr(node, 'load', 0.5)
            has_chain_service = 1 if node_id in chain_nodes.get(chain_id, set()) else 0
            # Lower score is better
            score = -has_chain_service * 1000 + load

            if score < best_score:
                best_score = score
                best_node = node_id

        return best_node
