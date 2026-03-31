"""Queueing network analyzer for calculating end-to-end latency."""

from typing import Dict, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..topology.topology import Topology
    from ..service.microservice import MicroService
    from ..service.chain import ServiceChain
    from ..service.deployment import DeploymentPlan


class QueueingNetworkAnalyzer:
    """
    排队网络分析器（与exp_2一致的per-instance M/M/1模型）

    给定部署方案，计算服务链的端到端延迟等指标
    """

    # 全局节点算力上限（与exp_2一致），用于从flops计算mu
    NODE_FLOPS_CAPACITY = 200 * 10 ** 9  # 200 GFLOPs
    # 跨节点通信延迟（与exp_2一致）
    COMM_DELAY_CROSS_NODE = 0.02  # 20ms

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

        延迟组成（与exp_2一致）:
        - 排队延迟: per-instance M/M/1 模型的 E[W]
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
        total_penalty = 0.0  # 拥塞惩罚（与exp_2一致）
        total_comm = 0.0

        if version_map is None:
            version_map = {}

        # 计算该链的总到达率（所有服务的Lambda_s）
        # 注意：同一条链中每个服务的到达率 = chain.arrival_rate
        chain_lambda = chain.arrival_rate

        # === 构建 (service, node, version) -> instances 的路由概率矩阵 ===
        # 第一步：收集该链所有服务在所有(node,version)的实例分布
        service_deployments = {}  # {service_id: {node_id: {version_id: count}}}

        for (s, n), versions in deployment.placement.items():
            if s not in service_deployments:
                service_deployments[s] = {}
            if n not in service_deployments[s]:
                service_deployments[s][n] = {}
            for vid, cnt in versions.items():
                if cnt > 0:
                    service_deployments[s][n][vid] = cnt

        # 第二步：遍历链中每个服务，计算加权延迟
        for i, service_id in enumerate(chain.services):
            # 检查服务是否已部署
            if service_id not in service_deployments:
                return {
                    "total": float('inf'),
                    "queuing": float('inf'),
                    "processing": float('inf'),
                    "communication": float('inf'),
                    "penalty": float('inf'),
                    "unrealizable": True
                }

            node_versions = service_deployments[service_id]
            total_instances = sum(sum(v.values()) for v in node_versions.values())

            if total_instances == 0:
                return {
                    "total": float('inf'),
                    "queuing": float('inf'),
                    "processing": float('inf'),
                    "communication": float('inf'),
                    "penalty": float('inf'),
                    "unrealizable": True
                }

            # 获取该服务的mu（优先用算法动态计算的，否则用version的mu）
            mu = deployment.service_mu.get(service_id)
            if mu is None:
                # 尝试从已部署的版本获取
                for n, vers in node_versions.items():
                    for vid, cnt in vers.items():
                        if cnt > 0 and service_id in self.services:
                            v = self.services[service_id].get_version(vid)
                            if v:
                                mu = v.mu
                                break
                    if mu:
                        break
                if mu is None:
                    mu = 10.0  # fallback

            # per-location M/M/1 排队延迟（与exp_2一致）
            # 注意：服务时间 1/μ 只加一次（所有location共享同一服务）
            # 排队延迟才是per-location的
            proc = 1000.0 / mu  # 处理时间 (ms)
            expected_task_delay = proc  # 基准处理时间只加一次

            for node_id, versions in node_versions.items():
                for version_id, cnt in versions.items():
                    if cnt <= 0:
                        continue

                    # 路由概率 = 该location实例数 / 总实例数
                    p = cnt / total_instances
                    lam = chain_lambda * p  # 分配到该location的到达率
                    rate_per_inst = lam / cnt  # 每个实例的到达率

                    # M/M/1 排队延迟（与exp_2一致）
                    if rate_per_inst >= mu:
                        # 拥塞：exp_2用1ms保底delay + 惩罚项
                        overload_penalty = rate_per_inst - mu + 1.0
                        total_penalty += 100.0 * overload_penalty
                        delay_node = 1.0
                    else:
                        delay_node = 1000.0 / (mu - rate_per_inst)  # ms

                    expected_task_delay += p * delay_node

            total_queuing += expected_task_delay

            # 通信延迟（服务i到服务i+1，跨节点才有）
            # 只需要算一次，不是按version算
            if i < chain.length - 1:
                next_service = chain.services[i + 1]
                curr_node = self._find_service_node(service_id, deployment)
                next_node = self._find_service_node(next_service, deployment)
                if next_node and curr_node and next_node != curr_node:
                    total_comm += self.COMM_DELAY_CROSS_NODE

        return {
            "total": total_queuing + total_comm,
            "queuing": total_queuing,
            "processing": 0.0,  # 已合并到 queuing
            "communication": total_comm,
            "penalty": total_penalty
        }

    def calc_resource_utilization(self, deployment: 'DeploymentPlan') -> Dict[str, float]:
        """计算各节点CPU利用率（与exp_2的内存利用率概念不同）"""
        utils = {}
        for node_id, node in self.topology.nodes.items():
            used = deployment.get_node_cpu_usage(node_id, self.services)
            utils[node_id] = used / node.cpu_cores if node.cpu_cores > 0 else 0
        return utils

    def calc_mem_utilization(self, deployment: 'DeploymentPlan') -> Dict[str, float]:
        """
        计算全局内存利用率（与exp_2一致）。

        mem_utilization = Σ(所有已部署实例的model_params) / (节点数 × MAX_NODE_PARAMS)
        """
        MAX_NODE_PARAMS = 150_000_000  # 与exp_2一致：150M Params
        n_nodes = len(self.topology.nodes)

        total_params_used = 0.0
        for (s, n), versions in deployment.placement.items():
            if s not in self.services:
                continue
            for vid, cnt in versions.items():
                if cnt <= 0:
                    continue
                version = self.services[s].get_version(vid)
                if version and version.model_params > 0:
                    total_params_used += cnt * version.model_params
                elif version:
                    # fallback: 用gpu_per_instance作为近似
                    total_params_used += cnt * version.gpu_per_instance

        total_capacity = n_nodes * MAX_NODE_PARAMS
        return total_params_used / total_capacity if total_capacity > 0 else 0.0

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
                for vid, count in versions.items():
                    if count > 0:
                        return vid
        return None

    def __repr__(self) -> str:
        return f"QueueingNetworkAnalyzer(services={len(self.services)})"

