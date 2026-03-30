"""
流量生成模块 - 对齐论文 A trace-driven + 论文 B 周期性
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from topo import Topology


class TrafficGenerator:
    """流量生成器：支持平稳/潮汐/突发三种模式"""

    def __init__(self, config, topology: Topology, tasks: List[str]):
        self.config = config
        self.topology = topology
        self.tasks = tasks
        self.n_edge = len([n for n in topology.nodes.values() if n.node_type == 'edge'])

        # 历史 λ 用于计算 λ_th
        self.node_lambda_history: Dict[int, List[float]] = defaultdict(list)
        self.lambda_th: Dict[int, float] = {}

    def generate_slot(self, t: int, mode: str = 'steady') -> Dict[int, List[Tuple[str, int]]]:
        """
        生成时隙 t 的请求分配。

        Returns:
            {node_nid: [(task_name, service_duration_slots), ...]}
        """
        requests = defaultdict(list)

        for nid, node in self.topology.nodes.items():
            if node.node_type == 'cloud':
                continue

            # 到达率 λ(t) - 单位: req/s
            # λ_base 设为 3.0 req/s（Poisson采样），保证小型节点也能稳定服务
            if mode == 'steady':
                lam = self.config.lambda_base * (0.8 + 0.4 * np.random.random())
            elif mode == 'tidal':
                period = 100  # 对齐论文 B 100s 周期
                lam = self.config.lambda_base * (1.0 + 0.8 * np.sin(2 * np.pi * t / period))
            elif mode == 'burst':
                if 20 <= (t % 100) <= 40:
                    lam = 15.0  # 突发高峰（仍保证大型节点 μ > λ）
                else:
                    lam = 3.0
            else:
                lam = self.config.lambda_base

            lam = max(0.5, lam)
            self.node_lambda_history[nid].append(lam)

            # 更新 λ_th（70% 分位，对齐论文 B）
            history = self.node_lambda_history[nid][-100:]
            if len(history) >= 10:
                self.lambda_th[nid] = np.percentile(history, self.config.lambda_th_percentile * 100)
            else:
                self.lambda_th[nid] = self.config.lambda_base

            # Poisson 采样每时隙请求数
            n_requests = np.random.poisson(lam * self.config.slot_duration_s)
            n_requests = min(n_requests, 200)  # 上限避免爆炸

            for _ in range(n_requests):
                task = self.tasks[np.random.randint(len(self.tasks))]
                duration = max(1, int(np.random.exponential(5.0)))
                requests[nid].append((task, duration))

        return dict(requests)
