"""DeploymentPlan class for modeling service deployment."""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from .microservice import MicroService


@dataclass
class DeploymentPlan:
    """
    部署方案 (包含模型版本信息)

    Our算法: 动态选择模型版本
    Baseline算法: 固定使用Model-M
    """
    # 部署映射: {(service_id, node_id): {version_id: count}}
    placement: Dict[Tuple[str, str], Dict[str, int]] = field(default_factory=dict)
    # 每个服务的mu值 (用于Our算法动态选择的模型)
    service_mu: Dict[str, float] = field(default_factory=dict)

    def add(self, service_id: str, node_id: str, version_id: str, count: int = 1, mu: float = None) -> None:
        """添加部署实例"""
        key = (service_id, node_id)
        if key not in self.placement:
            self.placement[key] = {}
        self.placement[key][version_id] = self.placement[key].get(version_id, 0) + count
        # 记录mu值 (用于延迟计算)
        if mu is not None:
            self.service_mu[service_id] = mu

    def get_service_instances(self, service_id: str, version_id: str = None) -> int:
        """获取服务总实例数 (可指定版本)"""
        if version_id:
            return sum(v.get(version_id, 0) for (s, _), v in self.placement.items() if s == service_id)
        return sum(sum(v.values()) for (s, _), v in self.placement.items() if s == service_id)

    def get_node_instances(self, node_id: str) -> Dict[str, Dict[str, int]]:
        """获取节点上部署的服务及版本"""
        return {s: v for (s, n), v in self.placement.items() if n == node_id}

    def get_version_usage(self) -> Dict[str, int]:
        """统计各版本使用量"""
        usage = {}
        for (s, n), versions in self.placement.items():
            for vid, count in versions.items():
                usage[vid] = usage.get(vid, 0) + count
        return usage

    def get_node_services(self, node_id: str) -> Dict[str, int]:
        """节点上部署的服务"""
        return {s: sum(v.values()) for (s, n), v in self.placement.items() if n == node_id}

    def get_node_cpu_usage(self, node_id: str, services: Dict[str, MicroService]) -> int:
        """节点CPU使用量"""
        total = 0
        for (s, n), versions in self.placement.items():
            if n == node_id and s in services:
                for vid, count in versions.items():
                    version = services[s].get_version(vid)
                    if version:
                        total += int(count * version.cpu_per_instance)
        return total

    def get_node_gpu_usage(self, node_id: str, services: Dict[str, MicroService]) -> int:
        """节点GPU使用量 (MB)"""
        total = 0
        for (s, n), versions in self.placement.items():
            if n == node_id and s in services:
                for vid, count in versions.items():
                    version = services[s].get_version(vid)
                    if version:
                        total += count * version.gpu_per_instance
        return total

    def validate(self, topology: 'Topology', services: Dict[str, MicroService]) -> bool:
        """验证部署合法性（CPU + GPU）"""
        for node_id, node in topology.nodes.items():
            cpu_used = self.get_node_cpu_usage(node_id, services)
            gpu_used = self.get_node_gpu_usage(node_id, services)
            if cpu_used > node.cpu_cores or gpu_used > node.gpu_memory:
                return False
        return True

    def __repr__(self) -> str:
        return f"DeploymentPlan(placement_size={len(self.placement)})"
