"""Shortest path routing algorithm."""

from typing import Dict, List, Tuple, Any

from algorithms.routing.base import RoutingAlgorithm


class ShortestPathRouting(RoutingAlgorithm):
    """
    Shortest path routing - routes requests via the minimum delay path.

    Strategy:
    1. For each service in the chain, select the node that is
       closest (minimum communication delay) to the previous service's node.
    2. This minimizes total communication delay.
    """

    def route(self, chain: Any,
              deployment_plan: Any,
              topology: Any) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute shortest path routing for a service chain.

        Args:
            chain: ServiceChain with services list and entry_node.
            deployment_plan: DeploymentPlan with placements.
            topology: Topology with nodes and communication delays.

        Returns:
            Dict mapping service_id to list of (node_id, probability).
        """
        routing = {}
        services = getattr(chain, 'services', [])
        entry_node = getattr(chain, 'entry_node', None)

        if not services:
            return routing

        # Start from entry node or first candidate
        if entry_node is None:
            # Find first service's candidates and pick the one in the topology
            first_service = services[0]
            candidates = self._get_candidate_nodes(first_service, deployment_plan)
            if candidates:
                current_node = candidates[0]
            else:
                return routing
        else:
            current_node = entry_node

        for service_id in services:
            # Get candidate nodes for this service
            candidates = self._get_candidate_nodes(service_id, deployment_plan)

            if not candidates:
                continue

            # Find the node with minimum delay from current position
            best_node = None
            best_delay = float('inf')

            for node_id in candidates:
                delay = self._get_delay(current_node, node_id, topology)
                if delay < best_delay:
                    best_delay = delay
                    best_node = node_id

            if best_node is None:
                # Fallback to first candidate
                best_node = candidates[0]

            # Route all traffic to the best node (probability = 1.0)
            routing[service_id] = [(best_node, 1.0)]

            # Move to this node for next hop
            current_node = best_node

        return routing

    def _get_delay(self, src: str, dst: str, topology: Any) -> float:
        """Get communication delay between two nodes."""
        if src == dst:
            return 0.0
        if hasattr(topology, 'get_communication_delay'):
            return topology.get_communication_delay(src, dst)
        return float('inf')
