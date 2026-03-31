"""Result validator for verifying deployment and routing feasibility."""

from typing import Dict, List, Any, Optional, Tuple

from core.topology.topology import Topology
from core.service.microservice import MicroService
from core.service.chain import ServiceChain
from core.service.deployment import DeploymentPlan


class ResultValidator:
    """结果验证器 - 验证部署和路由的合法性"""

    def validate_deployment(self, plan: DeploymentPlan,
                          topology: Topology,
                          services: Dict[str, MicroService],
                          chains: List[ServiceChain]) -> Tuple[bool, List[str]]:
        """
        验证部署方案的合法性。

        检查项:
        1. 资源约束: CPU/GPU使用不超过节点容量
        2. 服务可达性: 每条链的服务都已被部署
        3. 部署完整性: 所有服务都已被部署

        Args:
            plan: 部署方案
            topology: 网络拓扑
            services: 服务配置
            chains: 服务链列表

        Returns:
            Tuple[bool, List[str]]: (是否合法, 错误信息列表)
        """
        errors = []

        # Check 1: Resource constraints
        resource_errors = self._validate_resource_constraints(plan, topology, services)
        errors.extend(resource_errors)

        # Check 2: Service reachability for each chain
        reachability_errors = self._validate_service_reachability(plan, services, chains)
        errors.extend(reachability_errors)

        # Check 3: All services deployed
        completeness_errors = self._validate_deployment_completeness(plan, services, chains)
        errors.extend(completeness_errors)

        return len(errors) == 0, errors

    def _validate_resource_constraints(self, plan: DeploymentPlan,
                                       topology: Topology,
                                       services: Dict[str, MicroService]) -> List[str]:
        """验证资源约束"""
        errors = []

        for node_id, node in topology.nodes.items():
            cpu_used = plan.get_node_cpu_usage(node_id, services)
            gpu_used = plan.get_node_gpu_usage(node_id, services)

            if cpu_used > node.cpu_cores:
                errors.append(
                    f"Node {node_id}: CPU usage ({cpu_used}) exceeds capacity ({node.cpu_cores})"
                )

            if gpu_used > node.gpu_memory:
                errors.append(
                    f"Node {node_id}: GPU usage ({gpu_used}) exceeds capacity ({node.gpu_memory})"
                )

        return errors

    def _validate_service_reachability(self, plan: DeploymentPlan,
                                       services: Dict[str, MicroService],
                                       chains: List[ServiceChain]) -> List[str]:
        """验证每条链的服务是否都已部署"""
        errors = []

        for chain in chains:
            for service_id in chain.services:
                instances = plan.get_service_instances(service_id)
                if instances <= 0:
                    errors.append(
                        f"Chain {chain.chain_id}: Service {service_id} not deployed"
                    )

        return errors

    def _validate_deployment_completeness(self, plan: DeploymentPlan,
                                          services: Dict[str, MicroService],
                                          chains: List[ServiceChain]) -> List[str]:
        """验证所有服务是否都已部署"""
        errors = []

        # Collect all services referenced in chains
        referenced_services = set()
        for chain in chains:
            referenced_services.update(chain.services)

        # Check each referenced service
        for service_id in referenced_services:
            instances = plan.get_service_instances(service_id)
            if instances <= 0:
                errors.append(
                    f"Service {service_id} referenced in chains but not deployed"
                )

        return errors

    def validate_routing(self, plan: DeploymentPlan,
                        topology: Topology,
                        chains: List[ServiceChain]) -> Tuple[bool, List[str]]:
        """
        验证路由方案的可行性。

        检查项:
        1. 路径存在性: 任意相邻服务间存在可达路径
        2. 路径连通性: 从入口到出口的完整路径可达

        Args:
            plan: 部署方案
            topology: 网络拓扑
            chains: 服务链列表

        Returns:
            Tuple[bool, List[str]]: (是否可行, 错误信息列表)
        """
        errors = []

        for chain in chains:
            # Get deployment nodes for each service in chain
            node_sequence = []
            for service_id in chain.services:
                node_id = self._find_service_node(plan, service_id)
                if node_id:
                    node_sequence.append((service_id, node_id))
                else:
                    errors.append(
                        f"Chain {chain.chain_id}: Service {service_id} has no deployment"
                    )
                    break

            # Check connectivity between consecutive services
            for i in range(len(node_sequence) - 1):
                src_service, src_node = node_sequence[i]
                dst_service, dst_node = node_sequence[i + 1]

                if src_node != dst_node:
                    # Check if path exists
                    path = topology.get_shortest_path(src_node, dst_node)
                    if not path:
                        errors.append(
                            f"Chain {chain.chain_id}: No path from {src_service} on {src_node} "
                            f"to {dst_service} on {dst_node}"
                        )

        return len(errors) == 0, errors

    def _find_service_node(self, plan: DeploymentPlan, service_id: str) -> Optional[str]:
        """查找服务部署的节点"""
        for (s, n), versions in plan.placement.items():
            if s == service_id and versions and sum(versions.values()) > 0:
                return n
        return None

    def validate_all(self, plan: DeploymentPlan,
                    topology: Topology,
                    services: Dict[str, MicroService],
                    chains: List[ServiceChain]) -> Tuple[bool, List[str]]:
        """
        执行所有验证检查。

        Args:
            plan: 部署方案
            topology: 网络拓扑
            services: 服务配置
            chains: 服务链列表

        Returns:
            Tuple[bool, List[str]]: (是否通过所有验证, 所有错误信息)
        """
        all_errors = []

        # Validate deployment
        deploy_valid, deploy_errors = self.validate_deployment(
            plan, topology, services, chains
        )
        all_errors.extend(deploy_errors)

        # Validate routing
        routing_valid, routing_errors = self.validate_routing(
            plan, topology, chains
        )
        all_errors.extend(routing_errors)

        return len(all_errors) == 0, all_errors
