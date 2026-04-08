"""
动态实验：模拟时变负载与资源容量波动
每个时间步重新部署并评估
"""
import numpy as np
from network import EdgeNetwork
from service import generate_services, generate_service_chains, compute_aggregate_arrival_rates
from deployment import random_deployment
from routing import proportional_routing
from evaluation import evaluate
from config import NETWORK_SCALES, DYNAMIC_N_STEPS, DYNAMIC_SCALE_CONFIG, DEFAULT_N_CHAINS


def run_dynamic_experiment(scale, n_steps=None, n_chains=DEFAULT_N_CHAINS, seed=None):
    """
    运行动态实验。

    每个时间步:
      1. 随机扰动负载因子、CPU/GPU 容量
      2. 重新部署服务实例
      3. 计算并记录时延指标
    """
    rng = np.random.default_rng(seed)
    if n_steps is None:
        n_steps = DYNAMIC_N_STEPS

    cfg = DYNAMIC_SCALE_CONFIG[scale]
    params = NETWORK_SCALES[scale]

    # 初始化 (固定)
    network = EdgeNetwork(scale=scale, seed=seed)
    services = generate_services(cfg['n_types'], seed=seed)
    base_chains = generate_service_chains(
        n_chains, cfg['n_types'], cfg['chain_length'], cfg['base_rate'], seed=seed
    )

    results = []

    for step in range(n_steps):
        # 扰动因子
        load_factor = rng.uniform(0.5, 1.5)
        cpu_factor = rng.uniform(0.7, 1.3)
        gpu_factor = rng.uniform(0.7, 1.3)

        # 扰动网络容量
        network.perturb_capacity(cpu_factor, gpu_factor)

        # 扰动到达率
        chains = []
        for bc in base_chains:
            chains.append(type(bc)(bc.id, list(bc.services), bc.arrival_rate * load_factor))

        # 聚合到达率
        lambda_s = compute_aggregate_arrival_rates(chains, len(services))

        # 部署 + 路由
        deployment, _, _ = random_deployment(network, services, lambda_s, seed=seed + step)
        routing = proportional_routing(deployment)

        # 评估
        metrics = evaluate(deployment, services, chains, network, routing=routing)

        results.append({
            'step': step,
            'load_factor': round(load_factor, 3),
            'cpu_factor': round(cpu_factor, 3),
            'gpu_factor': round(gpu_factor, 3),
            'avg_delay': metrics['avg_delay'],
            'avg_comp_delay': metrics['avg_comp_delay'],
            'avg_comm_delay': metrics['avg_comm_delay'],
            'total_penalty': metrics['total_penalty'],
            'stable_chains': metrics['stable_chains'],
            'total_chains': metrics['total_chains'],
            'cpu_utilization': metrics['cpu_utilization'],
            'gpu_utilization': metrics['gpu_utilization'],
        })

        if step % 10 == 0:
            print(f"  step {step:>3d}: delay={metrics['avg_delay']:.4f}s, "
                  f"load={load_factor:.2f}, cpu_f={cpu_factor:.2f}, gpu_f={gpu_factor:.2f}, "
                  f"stable={metrics['stable_chains']}/{metrics['total_chains']}")

    return results


def run_all_scales(seed=None):
    """三种规模分别运行动态实验"""
    all_results = {}
    for scale in ['small', 'medium', 'large']:
        print(f"\n=== Dynamic: {scale} scale ===")
        results = run_dynamic_experiment(scale, seed=seed)
        all_results[scale] = results

        avg_delay = np.mean([r['avg_delay'] for r in results])
        avg_comp = np.mean([r['avg_comp_delay'] for r in results])
        avg_comm = np.mean([r['avg_comm_delay'] for r in results])
        print(f"  >> Avg: delay={avg_delay:.4f}s, comp={avg_comp:.4f}s, comm={avg_comm:.4f}s")

    return all_results
