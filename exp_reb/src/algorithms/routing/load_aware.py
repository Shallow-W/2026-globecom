"""Load-aware routing algorithm for Our algorithm."""

from typing import Dict, List, Tuple, Any

from algorithms.routing.base import RoutingAlgorithm


class LoadAwareRouting(RoutingAlgorithm):
    """
    Load-aware routing for Our algorithm.

    Strategy:
    - Considers both node load and path delay
    - Routes requests to nodes with lower load and shorter paths
    - Weight = 1.0 / (load * path_delay + epsilon)

    This is used by the Our algorithm for dynamic load balancing
    across deployed service instances.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize load-aware routing.

        Args:
            config: Optional config with:
                - epsilon: Small constant to avoid division by zero (default: 1e-6)
                - load_weight: Weight factor for load in routing decision (default: 1.0)
                - delay_weight: Weight factor for delay in routing decision (default: 1.0)
        """
        super().__init__(config)
        self.epsilon = self.config.get('epsilon', 1e-6)
        self.load_weight = self.config.get('load_weight', 1.0)
        self.delay_weight = self.config.get('delay_weight', 1.0)

    def route(self, chain: Any,
              deployment_plan: Any,
              topology: Any) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute load-aware routing for a service chain.

        Args:
            chain: ServiceChain with services list, arrival_rate, entry_node.
            deployment_plan: DeploymentPlan with placements.
            topology: Topology with nodes, links, and communication delays.

        Returns:
            Dict mapping service_id to list of (node_id, probability).
        """
        routing = {}
        services = getattr(chain, 'services', [])
        entry_node = getattr(chain, 'entry_node', None)
        arrival_rate = getattr(chain, 'arrival_rate', 1.0)

        if not services:
            return routing

        # Determine starting node
        if entry_node is None:
            # Try to find a reasonable starting point
            first_service = services[0]
            candidates = self._get_candidate_nodes(first_service, deployment_plan)
            entry_node = candidates[0] if candidates else None

        current_node = entry_node

        for service_id in services:
            # Get candidate nodes for this service
            candidates = self._get_candidate_nodes(service_id, deployment_plan)

            if not candidates:
                continue

            # Calculate routing weights based on load and delay
            weights = []
            for node_id in candidates:
                # Get node load factor
                load = self._get_node_load(node_id, topology, service_id, arrival_rate)

                # Get path delay from current node
                path_delay = self._get_delay(current_node, node_id, topology)

                # Combined routing weight: prefer low load and short delay
                # weight = 1.0 / (load^load_weight * path_delay^delay_weight + epsilon)
                combined_cost = (load ** self.load_weight) * (path_delay ** self.delay_weight)
                weight = 1.0 / (combined_cost + self.epsilon)

                weights.append((node_id, weight))

            # Normalize to probabilities
            routing[service_id] = self._normalize_weights(weights)

            # Update current node (use expected node based on probabilities)
            if routing[service_id]:
                # Use the most likely node for next hop calculation
                current_node = routing[service_id][0][0]

        return routing

    def _get_node_load(self, node_id: str, topology: Any,
                      service_id: str, arrival_rate: float) -> float:
        """
        Get the load factor for a node.

        For a node with deployed service, load is estimated based on:
        - Current CPU/GPU utilization
        - Arrival rate for the service

        Returns:
            Load factor (0.0 = idle, 1.0 = fully loaded).
        """
        if hasattr(topology, 'nodes') and node_id in topology.nodes:
            node = topology.nodes[node_id]

            # Get CPU utilization
            cpu_util = 0.0
            if hasattr(node, 'cpu_cores') and node.cpu_cores > 0:
                used_cpu = getattr(node, 'used_cpu', 0)
                cpu_util = used_cpu / node.cpu_cores

            # Get GPU utilization if applicable
            gpu_util = 0.0
            if hasattr(node, 'gpu_memory') and node.gpu_memory > 0:
                used_gpu = getattr(node, 'used_gpu', 0)
                gpu_util = used_gpu / node.gpu_memory

            # Combined load as weighted average
            load = 0.7 * cpu_util + 0.3 * gpu_util if node.gpu_memory > 0 else cpu_util

            # Scale with arrival rate (higher arrival rate = higher effective load)
            if arrival_rate > 0:
                load = load * (1.0 + arrival_rate / 100.0)

            return max(0.0, min(1.0, load))

        # Default load for unknown nodes
        return 0.5

    def _get_delay(self, src: str, dst: str, topology: Any) -> float:
        """Get communication delay between two nodes."""
        if src == dst:
            return 0.0
        if hasattr(topology, 'get_communication_delay'):
            return topology.get_communication_delay(src, dst)
        elif hasattr(topology, 'get_shortest_path'):
            path = topology.get_shortest_path(src, dst)
            if len(path) < 2:
                return 0.0
            # Sum up link delays
            total = 0.0
            if hasattr(topology, 'link_map'):
                for i in range(len(path) - 1):
                    link = topology.link_map.get((path[i], path[i+1]))
                    if link:
                        total += link.delay
            return total
        return 1.0  # Default delay if unknown
