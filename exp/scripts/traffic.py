"""
流量生成模块 - 对齐论文 A trace-driven + 论文 B 周期性
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from topo import Topology


class TrafficGenerator:
    """流量生成器：支持平稳/潮汐/突发三种模式 + 单变量扰动"""

    def __init__(self, config, topology: Topology, tasks: List[str]):
        self.config = config
        self.topology = topology
        self.tasks = tasks
        self.n_edge = len([n for n in topology.nodes.values() if n.node_type == 'edge'])

        # 历史 λ 用于计算 λ_th
        self.node_lambda_history: Dict[int, List[float]] = defaultdict(list)
        self.lambda_th: Dict[int, float] = {}

        # 当前扰动参数（支持运行时覆盖）
        self._override_lambda: float = None
        self._override_request_length: int = None
        self._override_n_tasks: int = None

    def set_perturbation(self, *, lambda_override: float = None,
                         request_length: int = None, n_tasks: int = None):
        """设置单变量扰动参数"""
        self._override_lambda = lambda_override
        self._override_request_length = request_length
        self._override_n_tasks = n_tasks

    def clear_perturbation(self):
        """清除扰动设置"""
        self._override_lambda = None
        self._override_request_length = None
        self._override_n_tasks = None

    def generate_slot(self, t: int, mode: str = 'steady') -> Dict[int, List[Tuple[str, int]]]:
        """
        生成时隙 t 的请求分配。

        Returns:
            {node_nid: [(task_name, service_duration_slots), ...]}
        """
        requests = defaultdict(list)

        # 确定激活的任务类型
        if self._override_n_tasks is not None and self._override_n_tasks < len(self.tasks):
            active_tasks = self.tasks[:self._override_n_tasks]
        else:
            active_tasks = self.tasks

        # 确定请求长度
        req_length = (self._override_request_length if self._override_request_length is not None
                      else self.config.request_length_base)

        for nid, node in self.topology.nodes.items():
            if node.node_type == 'cloud':
                continue

            # 到达率 λ(t) - 单位: req/s
            # 优先使用 override 值
            if self._override_lambda is not None:
                lam = self._override_lambda
            elif mode == 'steady':
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
                task = active_tasks[np.random.randint(len(active_tasks))]
                duration = max(1, int(np.random.exponential(req_length)))
                requests[nid].append((task, duration))

        return dict(requests)
