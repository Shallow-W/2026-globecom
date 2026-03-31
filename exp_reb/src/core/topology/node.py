"""Node class for computation nodes with CPU + GPU resources."""

from typing import Dict


class Node:
    """计算节点（支持CPU + GPU异构资源）"""

    def __init__(self, node_id: str, cpu_cores: int, gpu_memory: int = 0):
        """
        Initialize a computation node.

        Args:
            node_id: Unique identifier for the node
            cpu_cores: Number of CPU cores available
            gpu_memory: GPU memory in MB (0 if no GPU)
        """
        self.node_id = node_id
        self.cpu_cores = cpu_cores
        self.gpu_memory = gpu_memory
        self.used_cpu = 0
        self.used_gpu = 0
        self.services: Dict[str, int] = {}  # {service_id: instance_count}

    @property
    def available_cpu(self) -> int:
        """Get available CPU cores."""
        return self.cpu_cores - self.used_cpu

    @property
    def available_gpu(self) -> int:
        """Get available GPU memory in MB."""
        return self.gpu_memory - self.used_gpu

    def can_deploy(self, cpu_needed: int, gpu_needed: int = 0) -> bool:
        """
        Check if the node can deploy with given resource requirements.

        Args:
            cpu_needed: CPU cores required
            gpu_needed: GPU memory in MB required

        Returns:
            True if both CPU and GPU resources are sufficient
        """
        return self.available_cpu >= cpu_needed and self.available_gpu >= gpu_needed

    def deploy(self, service_id: str, instances: int,
               cpu_per_instance: float, gpu_per_instance: int = 0) -> None:
        """
        Deploy service instances to this node.

        Args:
            service_id: ID of the service being deployed
            instances: Number of instances to deploy
            cpu_per_instance: CPU cores per instance
            gpu_per_instance: GPU memory in MB per instance
        """
        self.services[service_id] = instances
        self.used_cpu += int(instances * cpu_per_instance)
        self.used_gpu += instances * gpu_per_instance

    def remove_service(self, service_id: str) -> None:
        """Remove a service from this node."""
        if service_id in self.services:
            del self.services[service_id]

    def __repr__(self) -> str:
        return (f"Node(node_id='{self.node_id}', cpu_cores={self.cpu_cores}, "
                f"gpu_memory={self.gpu_memory}, used_cpu={self.used_cpu}, "
                f"used_gpu={self.used_gpu})")
