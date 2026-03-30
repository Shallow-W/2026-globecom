"""
拓扑构建模块 - 对齐论文 A (Waxman) 和论文 B (分层树状)
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, List


@dataclass
class Node:
    """边缘/云端节点"""
    nid: int
    node_type: str  # 'edge' or 'cloud'
    memory_mb: float
    gflops: float
    x: float = 0.0
    y: float = 0.0

    # 运行时状态
    used_memory_mb: float = 0.0
    lambda_arrival: float = 0.0  # 到达率 req/s
    queue_length: int = 0
    deployed_archs: List[str] = field(default_factory=list)
    cache: Set[str] = field(default_factory=set)

    # 统计
    served_count: int = 0
    violated_count: int = 0
    total_latency_ms: float = 0.0


class Topology:
    """网络拓扑（支持小拓扑 Waxman + 大拓扑分层树状）"""

    def __init__(self, config, scale: str = 'small'):
        self.config = config
        self.scale = scale
        self.nodes: Dict[int, Node] = {}
        self.delay_matrix: Dict[tuple, float] = {}
        self.cloud_nid: int = 0

        if scale == 'small':
            self._build_small()
        else:
            self._build_large()

    def _build_small(self):
        """对齐论文 A: Atlanta 15 节点 Waxman 拓扑"""
        n = self.config.n_small
        alpha, beta = 0.5, 0.2

        # 节点地理坐标
        coords = [(random.random(), random.random()) for _ in range(n)]
        max_dist = np.sqrt(2)

        # Waxman 随机边生成
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                prob = beta * np.exp(-d / (alpha * max_dist))
                if random.random() < prob:
                    delay_ms = 1.0 + d * 9.0  # 1-10ms
                    self.delay_matrix[(i, j)] = delay_ms
                    self.delay_matrix[(j, i)] = delay_ms

        # 保证连通：连接最近邻
        for i in range(n):
            j = (i + 1) % n
            if (i, j) not in self.delay_matrix:
                d = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                delay_ms = 1.0 + d * 9.0
                self.delay_matrix[(i, j)] = delay_ms
                self.delay_matrix[(j, i)] = delay_ms

        # 节点资源分配
        nid = 1  # 0 留给云端
        node_idx = 0
        for count, mem, gflops in self.config.node_types:
            for _ in range(count):
                if node_idx < n:
                    self.nodes[nid] = Node(
                        nid=nid, node_type='edge',
                        memory_mb=mem, gflops=gflops,
                        x=coords[node_idx][0], y=coords[node_idx][1]
                    )
                    nid += 1
                    node_idx += 1

        # 云端节点
        self.cloud_nid = 0
        self.nodes[self.cloud_nid] = Node(
            nid=0, node_type='cloud',
            memory_mb=65536, gflops=10000,
            x=0.5, y=0.5
        )

        # 云边延迟（20ms，对齐论文 A 的 5% 约束）
        for nid_edge in self.nodes:
            if nid_edge != self.cloud_nid:
                self.delay_matrix[(self.cloud_nid, nid_edge)] = self.config.L_cloud_ms
                self.delay_matrix[(nid_edge, self.cloud_nid)] = self.config.L_cloud_ms

    def _build_large(self):
        """对齐论文 B: ta2 65 节点分层拓扑"""
        n = self.config.n_large
        nodes_per_region = 8
        cloud_nid = n

        # 8 区域 × 8 节点
        for rid in range(8):
            for nid_in_region in range(nodes_per_region):
                nid = rid * nodes_per_region + nid_in_region
                x = (rid + nid_in_region * 0.1) / 8.0
                y = 0.5
                mem, gflops = 4096, 30.0
                self.nodes[nid] = Node(nid=nid, node_type='edge',
                                        memory_mb=mem, gflops=gflops, x=x, y=y)

        # 云端
        self.cloud_nid = cloud_nid
        self.nodes[self.cloud_nid] = Node(nid=cloud_nid, node_type='cloud',
                                           memory_mb=65536, gflops=10000, x=0.5, y=0.5)

        # 延迟矩阵：区域内 2ms，区域间 10ms
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    ri = i // nodes_per_region
                    rj = j // nodes_per_region
                    self.delay_matrix[(i, j)] = 2.0 if ri == rj else 10.0

        # 云边延迟
        for i in range(n):
            self.delay_matrix[(i, self.cloud_nid)] = self.config.L_cloud_ms
            self.delay_matrix[(self.cloud_nid, i)] = self.config.L_cloud_ms

    def get_delay(self, i: int, j: int) -> float:
        """获取两节点间延迟（ms）"""
        return self.delay_matrix.get((i, j), 100.0)
