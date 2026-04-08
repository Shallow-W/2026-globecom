"""
请求路由算法：决定请求在各服务实例间的转发路径

路由表: 对于每对 (source_node, service)，给出转发到各目标节点的概率。
"""
import numpy as np


class RoutingTable:
    """路由表：记录每对 (源节点, 服务) 的转发概率分布"""

    def __init__(self, n_nodes, n_services):
        self.n_nodes = n_nodes
        self.n_services = n_services
        self.table = {}

    def set_route(self, source_node, service, target_probs):
        self.table[(source_node, service)] = np.array(target_probs)

    def get_route(self, source_node, service):
        return self.table.get((source_node, service), None)


def proportional_routing(deployment, seed=None):
    """
    按实例数比例路由：P[v] = X[v][s] / sum(X[:,s])
    直接从部署矩阵推导，无需额外决策。
    """
    rt = RoutingTable(deployment.n_nodes, deployment.n_services)

    for s in range(deployment.n_services):
        probs = deployment.routing_probabilities(s)
        for v in range(deployment.n_nodes):
            rt.set_route(v, s, probs)

    return rt


def random_routing(deployment, seed=None):
    """
    随机路由基线：仅在有实例的节点间随机分配转发概率。
    """
    rng = np.random.default_rng(seed)
    rt = RoutingTable(deployment.n_nodes, deployment.n_services)

    for s in range(deployment.n_services):
        active = np.where(deployment.X[:, s] > 0)[0]
        if len(active) == 0:
            continue

        raw = rng.uniform(0.1, 1.0, size=len(active))
        probs_full = np.zeros(deployment.n_nodes)
        probs_full[active] = raw / raw.sum()

        for v in range(deployment.n_nodes):
            rt.set_route(v, s, probs_full)

    return rt


def lego_routing(deployment, seed=None):
    """
    LEGO 路由算法：确定性均匀路由策略。

    LEGO 采用"确定性路由策略 + 均衡部署"的思路：
    对每个服务，在所有部署了该服务实例的节点间均匀分配流量，
    即每个活跃节点获得相同的转发概率 P[v] = 1 / count(active_nodes)。
    该策略不考虑各节点上的实例数量差异，而是强制均分，
    配合负载均衡的部署策略使用。

    参数:
        deployment: Deployment 实例
        seed:       随机种子（保留接口一致性，LEGO 路由是确定性的）
    """
    rt = RoutingTable(deployment.n_nodes, deployment.n_services)

    for s in range(deployment.n_services):
        # 找到部署了服务 s 实例的活跃节点
        active = np.where(deployment.X[:, s] > 0)[0]
        if len(active) == 0:
            continue

        # 均匀分布：每个活跃节点获得相同概率
        prob = 1.0 / len(active)
        probs_full = np.zeros(deployment.n_nodes)
        probs_full[active] = prob

        # 所有源节点使用相同的路由概率
        for v in range(deployment.n_nodes):
            rt.set_route(v, s, probs_full.copy())

    return rt


def drs_routing(deployment, seed=None):
    """
    DRS（确定性路由方案）路由：基于平方根加权的概率路由。

    对每个服务 s:
      1. 找到活跃节点（X[:,s] > 0 的节点）
      2. 计算权重 weight[v] = sqrt(X[v][s])，即实例数的平方根
      3. 路由概率 P[v] = sqrt(X[v][s]) / sum(sqrt(X[:,s]))

    平方根加权比纯比例路由更加均衡，避免实例数多的节点承担过多流量，
    同时仍保留按实例数加权的基本特性。
    所有源节点共享相同的路由概率（源节点无关）。

    参数:
        deployment: Deployment 实例
        seed:       随机种子（保留接口一致性，本策略为确定性）
    """
    rt = RoutingTable(deployment.n_nodes, deployment.n_services)

    for s in range(deployment.n_services):
        # 计算每个节点的平方根权重
        sqrt_weights = np.sqrt(deployment.X[:, s].astype(float))
        total_weight = np.sum(sqrt_weights)

        if total_weight <= 0:
            # 该服务无任何活跃实例，跳过
            continue

        # 归一化为路由概率
        probs = sqrt_weights / total_weight

        # 所有源节点设置相同的路由概率（源节点无关）
        for v in range(deployment.n_nodes):
            rt.set_route(v, s, probs)

    return rt
