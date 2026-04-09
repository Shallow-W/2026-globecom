"""
边缘计算网络拓扑：异构节点资源与通信时延矩阵
"""
import numpy as np
from config import NETWORK_SCALES, COMM_DELAY_MIN, COMM_DELAY_MAX


class EdgeNetwork:
    """
    异构边缘计算网络。每个节点的 CPU 核心数和 GPU 显存独立随机生成。
    """

    def __init__(self, scale='medium', seed=None):
        rng = np.random.default_rng(seed)
        params = NETWORK_SCALES[scale]

        self.scale = scale
        self.n_nodes = params['n_nodes']

        # 每个节点独立采样资源 (异构)
        cpu_lo, cpu_hi = params['cpu_range']
        gpu_lo, gpu_hi = params['gpu_range']

        # CPU: 整数，从 [cpu_lo, cpu_hi] 均匀采样
        self.base_cpu = rng.integers(cpu_lo, cpu_hi + 1, size=self.n_nodes).astype(float)
        # GPU: 整数且 256 对齐，从 [gpu_lo, gpu_hi] 均匀采样
        self.base_gpu = (rng.integers(gpu_lo, gpu_hi + 1, size=self.n_nodes) // 256 * 256).astype(float)

        self.cpu_capacity = self.base_cpu.copy()
        self.gpu_capacity = self.base_gpu.copy()

        # 通信时延矩阵 (对称)
        self.comm_delay = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                d = rng.uniform(COMM_DELAY_MIN, COMM_DELAY_MAX)
                self.comm_delay[i][j] = d
                self.comm_delay[j][i] = d

    def reset_resources(self):
        self.cpu_capacity = self.base_cpu.copy()
        self.gpu_capacity = self.base_gpu.copy()

    def perturb_capacity(self, cpu_factor, gpu_factor):
        """动态实验：按比例扰动各节点资源"""
        self.cpu_capacity = self.base_cpu * cpu_factor
        self.gpu_capacity = self.base_gpu * gpu_factor

    def summary(self):
        print(f"Network: {self.scale}, {self.n_nodes} nodes (heterogeneous)")
        for v in range(self.n_nodes):
            print(f"  Node {v}: CPU={int(self.base_cpu[v])}, GPU={int(self.base_gpu[v])} MB")
        print(f"  Comm delay matrix:\n{self.comm_delay}")
