"""Deployment algorithm base class."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class DeploymentAlgorithm(ABC):
    """
    Base class for all deployment algorithms.

    Baseline algorithms use fixed Model-M version.
    Our algorithm uses dynamic model selection.
    """

    # Fixed Model-M parameters (from architecture doc)
    MODEL_M = {
        "version_id": "Model-M",
        "mu": 10,
        "accuracy": 0.53,
        "cpu_per_instance": 1,
        "gpu_per_instance": 1024,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize deployment algorithm.

        Args:
            config: Optional configuration dict.
                   fixed_version: Version to use for baselines (default: "Model-M")
        """
        self.config = config or {}
        self.fixed_version = self.config.get("fixed_version", "Model-M")

    @abstractmethod
    def deploy(self, topology: Any,
               services: Dict[str, Any],
               chains: List[Any]) -> Any:
        """
        Execute deployment algorithm.

        Args:
            topology: Network topology with nodes.
            services: Service configurations {service_id: Service}.
            chains: Service chain list.

        Returns:
            DeploymentPlan: Deployment plan containing placement information.
        """
        pass

    def get_version_for_service(self, service_id: str, load: float,
                                services: Dict[str, Any]) -> str:
        """
        Get model version for a service.

        Baseline algorithms: fixed returns Model-M.
        Our algorithm: dynamic selection based on load.

        Args:
            service_id: Service identifier.
            load: Current load/arrival rate.
            services: Service configurations.

        Returns:
            str: Version ID to use.
        """
        return self.fixed_version

    def validate(self, plan: Any,
                 topology: Any,
                 services: Dict[str, Any]) -> bool:
        """
        Validate deployment plan.

        Args:
            plan: DeploymentPlan to validate.
            topology: Network topology.
            services: Service configurations.

        Returns:
            bool: True if valid, False otherwise.
        """
        if plan is None:
            return False
        if hasattr(plan, 'validate'):
            return plan.validate(topology, services)
        return True
