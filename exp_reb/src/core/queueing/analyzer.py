"""Queueing network analyzer for calculating end-to-end latency."""

from typing import Dict, Optional, List, TYPE_CHECKING

from .mmc import MMCQueue

if TYPE_CHECKING:
    from ..topology.topology import Topology
    from ..service.microservice import MicroService
    from ..service.chain import ServiceChain
    from ..service.deployment import DeploymentPlan


class QueueingNetworkAnalyzer:
    """
    排队网络分析器

    给定部署方案，计算服务链的端到端延迟等指标
    """

    def __init__(self, topology: 'Topology', services: Dict[str, 'MicroService']):
        """
        Initialize the analyzer.

        Args:
            topology: Network topology
            services: Dictionary of services {service_id: MicroService}
        """
        self.topology = topology
        self.services = services

    def calc_chain_latency(self, chain: 'ServiceChain',
                          deployment: 'DeploymentPlan',
                          version_map: Dict[str, str] = None) -> Dict[str, float]:
        """
        计算服务链的端到端延迟

        延迟组成:
        - 排队延迟: M/M/C 模型的 Wq
        - 处理延迟: Σ (1/μ_i)
        - 通信延迟: 服务间传输延迟

        Args:
            chain: Service chain to analyze
            deployment: Deployment plan
            version_map: Optional mapping of service_id to version_id

        Returns:
            {
                "total": 总延迟 (ms),
                "queuing": 排队延迟 (ms),
                "processing": 处理延迟 (ms),
                "communication": 通信延迟 (ms)
            }
        """
        total_queuing = 0.0
        total_processing = 0.0
        total_comm = 0.0

        # Use default version if not specified
        if version_map is None:
            version_map = {}

        for i, service_id in enumerate(chain.services):
            # 获取服务部署的节点
            node_id = self._find_service_node(service_id, deployment)
            if not node_id:
                # Service not deployed - this chain is UNREALIZABLE
                return {
                    "total": float('inf'),  # Mark as failed
                    "queuing": float('inf'),
                    "processing": float('inf'),
                    "communication": float('inf'),
                    "unrealizable": True
                }

            service_cfg = self.services[service_id]

            # Get deployed version from plan
            actual_version_id = self._get_deployed_version(service_id, deployment)

            # Get version - first try version_map, then actual deployed version, then fallback
            version_id = version_map.get(service_id, actual_version_id or "Model-M")
            version = service_cfg.get_version(version_id) if version_id else None

            if not version:
                # Try actual deployed version
                if actual_version_id:
                    version = service_cfg.get_version(actual_version_id)
            if not version:
                # Fallback to default Model-M
                version_id = "Model-M"
                version = service_cfg.get_version(version_id)
            if not version:
                # Last resort: use default mu=10
                version = type('Version', (), {'mu': 10, 'accuracy': 0.5})()

            # Get instances - use actual deployed version (not version_id) for lookup
            lookup_version = actual_version_id or version_id
            instances = deployment.get_service_instances(service_id, lookup_version)
            if instances <= 0:
                # Try getting total instances regardless of version
                instances = deployment.get_service_instances(service_id)
            if instances <= 0:
                instances = 1

            # M/M/C 排队延迟
            mmc = MMCQueue(
                arrival_rate=chain.arrival_rate,
                service_rate=version.mu,
                num_servers=instances
            )
            calc = mmc.calc()

            total_queuing += calc["Wq"]
            total_processing += calc["W"] - calc["Wq"]  # 1/μ

            # 通信延迟 (服务i到服务i+1)
            if i < chain.length - 1:
                next_service = chain.services[i + 1]
                next_node = self._find_service_node(next_service, deployment)
                if next_node and next_node != node_id:
                    total_comm += self.topology.get_communication_delay(node_id, next_node)

        return {
            "total": total_queuing + total_processing + total_comm,
            "queuing": total_queuing,
            "processing": total_processing,
            "communication": total_comm
        }

    def calc_resource_utilization(self, deployment: 'DeploymentPlan') -> Dict[str, float]:
        """计算各节点资源利用率"""
        utils = {}
        for node_id, node in self.topology.nodes.items():
            used = deployment.get_node_cpu_usage(node_id, self.services)
            utils[node_id] = used / node.cpu_cores if node.cpu_cores > 0 else 0
        return utils

    def _find_service_node(self, service_id: str, deployment: 'DeploymentPlan') -> Optional[str]:
        """查找服务部署的节点"""
        for (s, n), versions in deployment.placement.items():
            if s == service_id and versions and sum(versions.values()) > 0:
                return n
        return None

    def _get_deployed_version(self, service_id: str, deployment: 'DeploymentPlan') -> Optional[str]:
        """获取服务部署时使用的版本ID"""
        for (s, n), versions in deployment.placement.items():
            if s == service_id and versions and sum(versions.values()) > 0:
                # 返回第一个有部署的版本
                for vid, count in versions.items():
                    if count > 0:
                        return vid
        return None

    def __repr__(self) -> str:
        return f"QueueingNetworkAnalyzer(services={len(self.services)})"
