"""Routing algorithm base class."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional


class RoutingAlgorithm(ABC):
    """
    Base class for all routing algorithms.

    Routing algorithms determine how requests are routed through
    the service chain - which nodes to use for each service in the chain.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize routing algorithm.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}

    @abstractmethod
    def route(self, chain: Any,
              deployment_plan: Any,
              topology: Any) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute routing for a service chain.

        Args:
            chain: ServiceChain object with services list and arrival_rate.
            deployment_plan: DeploymentPlan with service placements.
            topology: Topology object with nodes and links.

        Returns:
            Dict[str, List[Tuple[node_id, probability]]]:
                Mapping from service_id to list of (node_id, probability) tuples.
                Probabilities should sum to 1.0 for each service.
        """
        pass

    def _get_candidate_nodes(self, service_id: str, deployment_plan: Any) -> List[str]:
        """
        Get list of nodes where a service is deployed.

        Args:
            service_id: Service identifier.
            deployment_plan: DeploymentPlan object.

        Returns:
            List of node IDs where the service is deployed.
        """
        if hasattr(deployment_plan, 'placement'):
            # DeploymentPlan with placement dict
            nodes = []
            for (s, n), versions in deployment_plan.placement.items():
                if s == service_id and versions:
                    nodes.append(n)
            return nodes
        return []

    def _normalize_weights(self, weights: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Normalize routing weights to probabilities.

        Args:
            weights: List of (node_id, weight) tuples.

        Returns:
            List of (node_id, probability) tuples with sum = 1.0.
        """
        if not weights:
            return []
        total = sum(w for _, w in weights)
        if total <= 0:
            return [(n, 1.0/len(weights)) for n, _ in weights]
        return [(n, w/total) for n, w in weights]
