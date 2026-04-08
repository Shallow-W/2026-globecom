"""
排队论时延计算：基于 M/M/c 模型的 Jackson 网络分解

核心公式:
  rho = lambda / (c * mu)                服务强度
  P_0 = [...]^{-1}                       系统空闲概率
  L_q = (a^c * rho) / (c! * (1-rho)^2) * P_0    平均队列长度
  W_q = L_q / lambda                     平均排队等待时间
  W   = W_q + 1/mu                       平均响应时间 (排队 + 服务)
"""
import numpy as np
from math import factorial
from config import INSTABILITY_PENALTY_FACTOR, INSTABILITY_PENALTY_BASE, INSTABILITY_FIXED_DELAY


def mm_c_response_time(arrival_rate, service_rate, n_servers):
    """
    M/M/c 排队模型：计算平均响应时间。

    参数:
        arrival_rate: 到达率 lambda
        service_rate: 服务率 mu (每核心)
        n_servers:    并行服务台数 c (即该节点上该服务的实例数)

    返回:
        (response_time, is_stable)
    """
    if n_servers <= 0 or arrival_rate <= 0:
        return 0.0, True

    lam = arrival_rate
    mu = service_rate
    c = int(n_servers)

    rho = lam / (c * mu)                   # 服务强度

    if rho >= 1.0:
        return INSTABILITY_FIXED_DELAY, False

    a = lam / mu                            # 到达负载

    # 计算 P_0 (系统空闲概率)
    sum_terms = sum((a ** k) / factorial(k) for k in range(c))
    last_term = (a ** c) / (factorial(c) * (1.0 - rho))
    P_0 = 1.0 / (sum_terms + last_term)

    # 平均队列长度 L_q
    L_q = ((a ** c) * rho) / (factorial(c) * (1.0 - rho) ** 2) * P_0

    # 平均响应时间 W = W_q + 1/mu
    W_q = L_q / lam
    W = W_q + 1.0 / mu

    return W, True


def compute_chain_delay(chain, services, deployment, network, lambda_s, routing=None):
    """
    计算单条服务链的端到端时延 (Jackson 网络分解).

    对链中每个服务:
      1. 路由概率 P[v] 优先从 routing 表读取，否则回退到 deployment.routing_probabilities(s)
      2. 各节点 M/M/c 排队时延
      3. 期望计算时延 = sum(P[v] * W[v])

    相邻服务间通信时延:
      E[comm] = sum P[t_i on v1] * P[t_{i+1} on v2] * comm[v1][v2],  v1 != v2

    返回: dict 包含 total / comp / comm / penalty / stable
    """
    comp_delay = 0.0
    comm_delay = 0.0
    penalty = 0.0
    n_unstable = 0

    prev_node_probs = None

    for i, sid in enumerate(chain.services):
        svc = services[sid]
        total_inst = deployment.total_instances(sid)

        if total_inst == 0:
            return {
                'total': float('inf'),
                'comp': float('inf'),
                'comm': float('inf'),
                'penalty': float('inf'),
                'stable': False,
            }

        # 优先使用路由表中的概率，否则回退到按比例路由
        if routing is not None:
            route = routing.get_route(0, sid)  # 源节点无关时取任一即可
            node_probs = route if route is not None else deployment.routing_probabilities(sid)
        else:
            node_probs = deployment.routing_probabilities(sid)
        lam_s = lambda_s[sid]

        # 期望计算时延
        svc_delay = 0.0
        for v in range(deployment.n_nodes):
            n_inst = deployment.X[v][sid]
            if n_inst == 0:
                continue

            lam_v = lam_s * node_probs[v]          # 到达节点 v 的到达率
            delay_v, stable = mm_c_response_time(lam_v, svc.service_rate, n_inst)

            if not stable:
                n_unstable += 1
                excess = lam_v / (n_inst * svc.service_rate) - 1.0
                penalty += INSTABILITY_PENALTY_FACTOR * (excess + INSTABILITY_PENALTY_BASE)

            svc_delay += node_probs[v] * delay_v

        comp_delay += svc_delay

        # 通信时延: 与前一个服务的跨节点通信
        if prev_node_probs is not None and i > 0:
            for v1 in range(deployment.n_nodes):
                if prev_node_probs[v1] == 0:
                    continue
                for v2 in range(deployment.n_nodes):
                    if node_probs[v2] == 0 or v1 == v2:
                        continue
                    comm_delay += prev_node_probs[v1] * node_probs[v2] * network.comm_delay[v1][v2]

        prev_node_probs = node_probs

    total = comp_delay + comm_delay + penalty

    return {
        'total': total,
        'comp': comp_delay,
        'comm': comm_delay,
        'penalty': penalty,
        'stable': n_unstable == 0,
    }
