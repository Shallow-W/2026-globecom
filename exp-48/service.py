"""
微服务与服务链定义：服务属性、链路生成、聚合到达率计算
"""
import numpy as np
from config import SERVICE_RATE_RANGE, GPU_PER_INSTANCE_RANGE


class Service:
    """单个微服务：拥有服务率和 GPU 资源需求"""
    def __init__(self, service_id, service_rate, gpu_per_instance):
        self.id = service_id
        self.service_rate = service_rate              # mu: 每核心处理速率
        self.gpu_per_instance = gpu_per_instance      # 每实例 GPU 占用 (MB)

    def __repr__(self):
        return f"Service(id={self.id}, mu={self.service_rate:.2f}, gpu={self.gpu_per_instance}MB)"


class ServiceChain:
    """服务链：有序微服务序列 + 到达率"""
    def __init__(self, chain_id, service_ids, arrival_rate):
        self.id = chain_id
        self.services = service_ids                   # list[int]: 服务 ID 序列
        self.arrival_rate = arrival_rate               # lambda: Poisson 到达率

    def __repr__(self):
        return f"Chain(id={self.id}, svc={self.services}, rate={self.arrival_rate:.1f})"


def generate_services(n_types, seed=None):
    """随机生成 n_types 个微服务"""
    rng = np.random.default_rng(seed)
    services = []
    for i in range(n_types):
        mu = rng.uniform(*SERVICE_RATE_RANGE)
        gpu = int(rng.integers(GPU_PER_INSTANCE_RANGE[0], GPU_PER_INSTANCE_RANGE[1]))
        services.append(Service(i, mu, gpu))
    return services


def generate_service_chains(n_chains, n_service_types, chain_length, total_rate, seed=None):
    """
    生成 n_chains 条服务链，每条链从 n_service_types 中随机抽取 chain_length 个服务。
    总到达率 total_rate 按 Dirichlet 分布分配给各链。
    """
    rng = np.random.default_rng(seed)
    rates = rng.dirichlet(np.ones(n_chains)) * total_rate

    chains = []
    for i in range(n_chains):
        seq = rng.integers(0, n_service_types, size=chain_length).tolist()
        chains.append(ServiceChain(i, seq, rates[i]))
    return chains


def compute_aggregate_arrival_rates(chains, n_services):
    """计算每个微服务在所有链中的聚合到达率 lambda_s"""
    lambda_s = np.zeros(n_services)
    for chain in chains:
        for sid in chain.services:
            lambda_s[sid] += chain.arrival_rate
    return lambda_s
