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
                continue

            service_cfg = self.services[service_id]

            # Get version for this service
            version_id = version_map.get(service_id, "Model-M")
            version = service_cfg.get_version(version_id)
            if not version:
                version_id = "Model-M"
                version = service_cfg.get_version(version_id)

            instances = deployment.get_service_instances(service_id, version_id)
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

    def __repr__(self) -> str:
        return f"QueueingNetworkAnalyzer(services={len(self.services)})"
