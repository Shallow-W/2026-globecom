"""
算法策略层：将部署算法 + 路由算法封装为可插拔的策略对象

使用方式:
    algo = AlgorithmSuite("my_algo", my_deploy_fn, my_route_fn)
    deployment, routing = algo.solve(network, services, lambda_s, seed=42)

新增算法只需:
    1. 在 deployment.py 中写部署函数 (network, services, lambda_s, seed) -> (Deployment, cpu_rem, gpu_rem)
    2. 在 routing.py 中写路由函数 (deployment, seed) -> RoutingTable
    3. 在这里注册一个 AlgorithmSuite 实例
"""
from deployment import random_deployment, ffd_deployment, drs_deployment, lego_deployment
from routing import proportional_routing, random_routing, drs_routing, lego_routing


class AlgorithmSuite:
    """
    算法策略：封装一组 (部署算法, 路由算法) 的完整方案。

    属性:
        name:       算法名称 (用于结果标注)
        deploy_fn:  部署函数, 签名 (network, services, lambda_s, seed) -> (Deployment, cpu_rem, gpu_rem)
        route_fn:   路由函数, 签名 (deployment, seed) -> RoutingTable
    """

    def __init__(self, name, deploy_fn, route_fn):
        self.name = name
        self.deploy_fn = deploy_fn
        self.route_fn = route_fn

    def solve(self, network, services, lambda_s, seed=None):
        """
        执行部署 + 路由，返回 (deployment, routing)。
        """
        deployment, cpu_rem, gpu_rem = self.deploy_fn(
            network, services, lambda_s=lambda_s, seed=seed
        )
        routing = self.route_fn(deployment, seed=seed)
        return deployment, routing

    def __repr__(self):
        return f"AlgorithmSuite('{self.name}')"


# ============================================================
# 注册的算法策略
# ============================================================

# RLS: 随机部署 + 按比例路由 (默认基线)
ALGO_RANDOM_PROPORTIONAL = AlgorithmSuite(
    name="Random-Proportional",
    deploy_fn=random_deployment,
    route_fn=proportional_routing,
)

# 随机部署 + 随机路由
ALGO_RANDOM_RANDOM = AlgorithmSuite(
    name="Random-Random",
    deploy_fn=random_deployment,
    route_fn=random_routing,
)

# FFD: First Fit Decreasing 部署 + 按比例路由
ALGO_FFD = AlgorithmSuite(
    name="FFD",
    deploy_fn=ffd_deployment,
    route_fn=proportional_routing,
)

# DRS: 贪心部署 + 按比例路由
ALGO_DRS = AlgorithmSuite(
    name="DRS",
    deploy_fn=drs_deployment,
    route_fn=proportional_routing,
)

# LEGO: Round-Robin 均衡部署 + 按比例路由
ALGO_LEGO = AlgorithmSuite(
    name="LEGO",
    deploy_fn=lego_deployment,
    route_fn=proportional_routing,
)

# 全部已注册算法 (方便遍历)
ALL_ALGORITHMS = [
    ALGO_RANDOM_PROPORTIONAL,
    ALGO_RANDOM_RANDOM,
    ALGO_FFD,
    ALGO_DRS,
    ALGO_LEGO,
]


def get_algorithm(name):
    """按名称查找已注册算法"""
    for algo in ALL_ALGORITHMS:
        if algo.name == name:
            return algo
    raise ValueError(f"Unknown algorithm: {name}. Available: {[a.name for a in ALL_ALGORITHMS]}")
