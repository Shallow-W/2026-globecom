"""
评估模块：给定部署 + 路由方案，计算系统整体性能指标
"""
import numpy as np
from service import compute_aggregate_arrival_rates
from queuing import compute_chain_delay


def evaluate(deployment, services, chains, network, routing=None):
    """
    评估部署方案的系统性能。

    参数:
        deployment: Deployment 实例
        services:   Service 列表
        chains:     ServiceChain 列表
        network:    EdgeNetwork 实例
        routing:    RoutingTable 实例 (可选, 若提供则使用其中的路由概率)

    返回指标:
        avg_delay:       加权平均端到端时延
        avg_comp_delay:  加权平均计算时延
        avg_comm_delay:  加权平均通信时延
        total_penalty:   不稳定惩罚
        stable_chains:   稳定链数
        total_chains:    总链数
        cpu_utilization: CPU 利用率
        gpu_utilization: GPU 利用率
        chain_details:   各链详细结果
    """
    lambda_s = compute_aggregate_arrival_rates(chains, len(services))
    total_rate = sum(c.arrival_rate for c in chains)

    total_delay = 0.0
    total_comp = 0.0
    total_comm = 0.0
    total_penalty = 0.0
    n_stable = 0

    chain_details = []

    for chain in chains:
        result = compute_chain_delay(chain, services, deployment, network, lambda_s, routing=routing)
        weight = chain.arrival_rate / total_rate

        total_delay += weight * result['total']
        total_comp += weight * result['comp']
        total_comm += weight * result['comm']
        total_penalty += weight * result['penalty']

        if result['stable']:
            n_stable += 1

        chain_details.append({
            'chain_id': chain.id,
            'weight': weight,
            **result,
        })

    # 资源利用率
    total_cpu_used = float(np.sum(deployment.X))
    total_cpu_cap = float(np.sum(network.cpu_capacity))

    total_gpu_used = sum(
        deployment.X[v][s] * services[s].gpu_per_instance
        for v in range(deployment.n_nodes)
        for s in range(len(services))
    )
    total_gpu_cap = float(np.sum(network.gpu_capacity))

    return {
        'avg_delay': total_delay,
        'avg_comp_delay': total_comp,
        'avg_comm_delay': total_comm,
        'total_penalty': total_penalty,
        'stable_chains': n_stable,
        'total_chains': len(chains),
        'cpu_utilization': total_cpu_used / total_cpu_cap if total_cpu_cap > 0 else 0,
        'gpu_utilization': total_gpu_used / total_gpu_cap if total_gpu_cap > 0 else 0,
        'chain_details': chain_details,
    }
