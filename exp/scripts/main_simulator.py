"""
云边协同部署与路由仿真器
对齐论文 A (TPDS 2023) 和论文 B (TSC 2024) 实验设计
实现 6 种算法：OURS, HEURISTIC_A, GREEDY_B, STATIC, RESOURCE_FIRST, ACCURACY_FIRST
"""

import numpy as np
import pandas as pd
import random
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import heapq
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置区
# ============================================================

@dataclass
class Config:
    # 基础路径
    data_path: str = "D:/Item/lab/2026globcom/exp/data/evaluation_tables_20260325_163931.xlsx"
    output_dir: str = "D:/Item/lab/2026globcom/exp/results/"

    # 拓扑参数（对齐论文 A: 15节点小拓扑 / 论文 B: 65节点大拓扑）
    n_small: int = 15
    n_large: int = 65
    n_cloud: int = 1  # 云端节点

    # 节点异构资源（对齐论文 A: 8-64 核，映射到 GFLOPS 和内存）
    node_types: List[Tuple[int, int, int]] = None  # (count, memory_mb, gflops)

    # SLA 参数（对齐论文 A/B）
    T_SLA_ms: float = 200.0  # 通用 SLA
    T_SLA_jigsaw_ms: float = 100.0  # jigsaw 时间敏感

    # 流量参数（对齐论文 A/B）
    # 调整为合理范围: λ 应 < 节点服务率 μ = C_max / F_flops
    # 小型节点 10GF: μ≈4 req/s (中等架构); 大型节点 100GF: μ≈40 req/s
    # 因此 λ_base 设为 3.0 req/s (保证小型节点也能服务)
    lambda_base: float = 3.0  # 基准到达率 req/s
    lambda_th_percentile: float = 0.7  # λ_th 分位点

    # 拉取参数
    B_cloud_mbps: float = 100.0  # 云边带宽 Mbps
    L_cloud_ms: float = 20.0  # 云边时延 ms
    rho_pull: float = 1e-6  # 参数量到传输时间系数 (params / B_cloud)

    # 时隙
    slot_duration_s: float = 1.0
    n_slots: int = 500  # 仿真时隙数

    # 缓存
    cache_k: int = 10  # 每节点缓存架构数

    # 权重参数
    alpha: float = 1.0  # 延迟惩罚系数
    beta: float = 1.0  # 内存惩罚系数
    theta1: float = 0.5  # 精度权重
    theta2: float = 0.35  # 延迟权重
    theta3: float = 0.15  # 内存权重

    # 随机种子
    seed: int = 42

    def __post_init__(self):
        if self.node_types is None:
            # 小型5个(2GB,10GF), 中型7个(4GB,30GF), 大型3个(8GB,100GF) 对齐论文A
            self.node_types = [(5, 2048, 10), (7, 4096, 30), (3, 8192, 100)]


# ============================================================
# 数据加载
# ============================================================

def load_architecture_tables(data_path: str) -> Dict[str, pd.DataFrame]:
    """加载 xlsx 的 7 张任务表，做 min-max 归一化"""
    tables = pd.read_excel(data_path, sheet_name=None)

    normalized = {}
    for task_name, df in tables.items():
        df = df.copy()
        # 基础字段
        df['arch_id'] = df['architecture']
        df['task'] = task_name

        # min-max 归一化（全局跨表，用于跨任务比较时）
        for col in ['proxy_score', 'flops', 'model_params']:
            v_min, v_max = df[col].min(), df[col].max()
            if v_max > v_min:
                df[f'{col}_norm'] = (df[col] - v_min) / (v_max - v_min)
            else:
                df[f'{col}_norm'] = 0.5

        # epochs 归一化（处理 '-' 值）
        df['epochs_val'] = pd.to_numeric(df['epochs_to_reach_avg_final_performance'], errors='coerce')
        df['epochs_val'] = df['epochs_val'].fillna(0)
        e_max = df['epochs_val'].max()
        if e_max > 0:
            df['epochs_norm'] = df['epochs_val'] / e_max
        else:
            df['epochs_norm'] = 0.0

        normalized[task_name] = df

    return normalized


# ============================================================
# 拓扑构建（对齐论文 A: Waxman, 论文 B: 分层树状）
# ============================================================

@dataclass
class Node:
    nid: int
    node_type: str  # 'edge' or 'cloud'
    memory_mb: int
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
    def __init__(self, config: Config, scale: str = 'small'):
        self.config = config
        self.scale = scale
        self.nodes: Dict[int, Node] = {}
        self.delay_matrix: Dict[Tuple[int, int], float] = {}
        self.cloud_nid: int = 0

        if scale == 'small':
            self._build_small()
        else:
            self._build_large()

    def _build_small(self):
        """对齐论文 A: Atlanta 15 节点 Waxman 拓扑"""
        n = self.config.n_small
        alpha, beta = 0.5, 0.2

        # 生成节点坐标
        coords = [(random.random(), random.random()) for _ in range(n)]
        max_dist = np.sqrt(2)

        # 构建边和延迟矩阵（全连接近似，用概率连通）
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                prob = beta * np.exp(-d / (alpha * max_dist))
                if random.random() < prob:
                    delay_ms = 1.0 + d * 9.0  # 1-10ms
                    edges.append((i, j, delay_ms))
                    self.delay_matrix[(i, j)] = delay_ms
                    self.delay_matrix[(j, i)] = delay_ms

        # 保证连通：找最大生成树，然后补充一些边
        if len(edges) < n - 1:
            for i in range(n):
                j = (i + 1) % n
                d = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                delay_ms = 1.0 + d * 9.0
                self.delay_matrix[(i, j)] = delay_ms
                self.delay_matrix[(j, i)] = delay_ms

        # 节点资源分配（按 config 的三档）
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

        self.cloud_nid = 0
        self.nodes[self.cloud_nid] = Node(
            nid=0, node_type='cloud',
            memory_mb=65536, gflops=10000,
            x=0.5, y=0.5
        )

        # 云边时延（所有边缘到云端 20ms，对齐论文 A 的 5% 约束）
        for nid_node in self.nodes:
            if nid_node != self.cloud_nid:
                self.delay_matrix[(self.cloud_nid, nid_node)] = self.config.L_cloud_ms
                self.delay_matrix[(nid_node, self.cloud_nid)] = self.config.L_cloud_ms

    def _build_large(self):
        """对齐论文 B: ta2 65节点分层拓扑"""
        n = self.config.n_large
        regions = 8
        nodes_per_region = 8
        cloud_nid = n  # 最后一个是云

        # 分层：区域内部 2ms，区域间 10ms
        for rid in range(regions):
            for nid_in_region in range(nodes_per_region):
                nid = rid * nodes_per_region + nid_in_region
                x = (rid + nid_in_region * 0.1) / regions
                y = 0.5
                mem, gflops = 4096, 30  # 统一中型
                self.nodes[nid] = Node(nid=nid, node_type='edge',
                                       memory_mb=mem, gflops=gflops, x=x, y=y)

        self.cloud_nid = cloud_nid
        self.nodes[self.cloud_nid] = Node(nid=cloud_nid, node_type='cloud',
                                           memory_mb=65536, gflops=10000, x=0.5, y=0.5)

        # 延迟矩阵
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    ri, rj = i // nodes_per_region, j // nodes_per_region
                    if ri == rj:
                        self.delay_matrix[(i, j)] = 2.0  # 区域内
                    else:
                        self.delay_matrix[(i, j)] = 10.0  # 区域间

        # 边缘到云端
        for i in range(n):
            self.delay_matrix[(i, self.cloud_nid)] = self.config.L_cloud_ms
            self.delay_matrix[(self.cloud_nid, i)] = self.config.L_cloud_ms

    def get_delay(self, i: int, j: int) -> float:
        return self.delay_matrix.get((i, j), 100.0)  # 默认 100ms


# ============================================================
# 流量生成器（对齐论文 A trace-driven + 论文 B 周期性）
# ============================================================

class TrafficGenerator:
    def __init__(self, config: Config, topology: Topology, tasks: List[str]):
        self.config = config
        self.topology = topology
        self.tasks = tasks
        self.n_edge_nodes = len([n for n in topology.nodes.values() if n.node_type == 'edge'])

        # 预计算每节点历史 λ，用于求 λ_th
        self.node_lambda_history: Dict[int, List[float]] = defaultdict(list)
        self.lambda_th: Dict[int, float] = {}

    def generate_slot(self, t: int, mode: str = 'steady') -> Dict[int, List[Tuple[str, float]]]:
        """
        生成时隙 t 的请求分配。
        返回: {node_nid: [(task_name, duration_slots), ...]}
        """
        requests = defaultdict(list)

        for nid, node in self.topology.nodes.items():
            if node.node_type == 'cloud':
                continue

            # 到达率 λ(t)
            if mode == 'steady':
                # 平稳流：Poisson(λ_base)，对齐论文 A
                lam = self.config.lambda_base * (0.8 + 0.4 * random.random())
            elif mode == 'tidal':
                # 潮汐流：对齐论文 B 100s 周期
                period = 100
                lam = self.config.lambda_base * (1.0 + 0.8 * np.sin(2 * np.pi * t / period))
            elif mode == 'burst':
                # 突发流：对齐论文 A 请求规模（压缩到可服务范围）
                if 20 <= (t % 100) <= 40:
                    lam = 15.0  # 突发高峰（仍保证 μ > λ）
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

            # Poisson 采样请求数
            n_requests = np.random.poisson(lam * self.config.slot_duration_s)
            n_requests = min(n_requests, 100)  # 上限避免爆炸

            for _ in range(n_requests):
                task = random.choice(self.tasks)
                duration = max(1, int(np.random.exponential(5.0)))  # 服务时长 1-10 slot
                requests[nid].append((task, duration))

        return dict(requests)


# ============================================================
# 算法实现
# ============================================================

class DeploymentAlgorithm:
    """部署算法基类"""

    def __init__(self, config: Config):
        self.config = config

    def select_arch(self, node: Node, candidates: List[dict], t: int,
                    topo: Topology, tables: Dict[str, pd.DataFrame]) -> Optional[str]:
        raise NotImplementedError

    def compute_F_max(self, node: Node, t: int, traffic_gen: TrafficGenerator) -> float:
        """动态算力红线（论文核心公式）"""
        lam = node.lambda_arrival
        if lam <= 0:
            lam = 1.0
        F_max = node.gflops / (lam + 1.0 / (self._get_T_SLA(node) / 1000.0))
        return max(F_max, 1.0)  # 最小 1 GFLOPS

    def _get_T_SLA(self, node: Node) -> float:
        return self.config.T_SLA_jigsaw_ms  # 保守用严格 SLA

    def filter_candidates(self, node: Node, task: str, t: int,
                          topo: Topology, tables: Dict[str, pd.DataFrame]) -> List[dict]:
        """硬约束过滤，返回满足内存+算力红线的候选列表（向量化优化）"""
        df = tables[task]
        F_max = self.compute_F_max(node, t, None)
        M_avail = node.memory_mb - node.used_memory_mb

        # 向量化过滤（避免 iterrows）
        # params_mb = model_params / (1024*1024)
        # flops_gf = flops / 1e9
        params_mb = df['model_params'].values / (1024 * 1024)
        flops_gf = df['flops'].values / 1e9

        mask = (params_mb <= M_avail) & (flops_gf <= F_max)
        filtered = df[mask]

        if len(filtered) == 0:
            return []

        # 转成 list of dicts（只取需要的列）
        return [
            {
                'arch_id': row['arch_id'],
                'proxy_score': row['proxy_score_norm'],
                'flops': row['flops'],
                'flops_norm': row['flops_norm'],
                'params': row['model_params'],
                'params_norm': row['model_params_norm'],
                'proxy_raw': row['proxy_score'],
                'task_perf': row['task_final_performance'],
            }
            for _, row in filtered.iterrows()
        ]


class OursCEDR(DeploymentAlgorithm):
    """OURS: 你的论文算法 - 动态启发式部署"""

    def select_arch(self, node: Node, task: str, t: int,
                    topo: Topology, tables: Dict[str, pd.DataFrame],
                    traffic_gen: TrafficGenerator) -> Optional[dict]:

        candidates = self.filter_candidates(node, task, t, topo, tables)
        if not candidates:
            return None

        lam = node.lambda_arrival
        lam_th = traffic_gen.lambda_th.get(node.nid, self.config.lambda_base)

        # 动态权重（论文公式）
        w2 = self.config.alpha * np.exp(max(0, (lam - lam_th) / lam_th))
        mem_used_ratio = node.used_memory_mb / node.memory_mb if node.memory_mb > 0 else 0
        w3 = self.config.beta * mem_used_ratio
        w1 = 1.0 / (1.0 + w2 + w3)

        best_arch = None
        best_utility = -float('inf')

        for c in candidates:
            # 效用函数
            utility = (w1 * c['proxy_score']
                       - w2 * c['flops_norm']
                       - w3 * c['params_norm'])

            if utility > best_utility:
                best_utility = utility
                best_arch = c

        return best_arch


class HeuristicA(DeploymentAlgorithm):
    """论文 A 算法：按服务实例数量降序 -> 部署到平均距离最短节点"""

    def select_arch(self, node: Node, task: str, t: int,
                    topo: Topology, tables: Dict[str, pd.DataFrame],
                    traffic_gen: TrafficGenerator) -> Optional[dict]:

        candidates = self.filter_candidates(node, task, t, topo, tables)
        if not candidates:
            return None

        # 论文 A 思路：选 proxy_score 最高的（对应"平均距离最短"的代理）
        # 简化：用 proxy_score 排序，选择最高的
        best = max(candidates, key=lambda c: c['proxy_score'])
        return best


class GreedyB(DeploymentAlgorithm):
    """论文 B 算法：贪心分配 -> 资源利用率最低节点"""

    def select_arch(self, node: Node, task: str, t: int,
                    topo: Topology, tables: Dict[str, pd.DataFrame],
                    traffic_gen: TrafficGenerator) -> Optional[dict]:

        candidates = self.filter_candidates(node, task, t, topo, tables)
        if not candidates:
            return None

        # 论文 B 思路：贪心选资源利用率最低 -> 即选 flops+params 最小的（轻量优先）
        best = min(candidates, key=lambda c: c['flops_norm'] + c['params_norm'])
        return best


class StaticBestProxy(DeploymentAlgorithm):
    """Static-BestProxy: 全程固定 proxy_score 最高的架构"""

    def __init__(self, config: Config):
        self.config = config
        self.fixed_archs: Dict[str, dict] = {}  # task -> arch dict

    def select_arch(self, node: Node, task: str, t: int,
                    topo: Topology, tables: Dict[str, pd.DataFrame],
                    traffic_gen: TrafficGenerator) -> Optional[dict]:

        if task not in self.fixed_archs:
            df = tables[task]
            best_row = df.loc[df['proxy_score'].idxmax()]
            self.fixed_archs[task] = {
                'arch_id': best_row['arch_id'],
                'proxy_score': best_row['proxy_score_norm'],
                'flops': best_row['flops'],
                'flops_norm': best_row['flops_norm'],
                'params': best_row['model_params'],
                'params_norm': best_row['params_norm'],
                'task_perf': best_row['task_final_performance'],
            }

        return self.fixed_archs[task]


class ResourceFirst(DeploymentAlgorithm):
    """Resource-First: 仅按 flops+params 最小优先"""

    def select_arch(self, node: Node, task: str, t: int,
                    topo: Topology, tables: Dict[str, pd.DataFrame],
                    traffic_gen: TrafficGenerator) -> Optional[dict]:

        candidates = self.filter_candidates(node, task, t, topo, tables)
        if not candidates:
            return None

        best = min(candidates, key=lambda c: c['flops_norm'] + c['params_norm'])
        return best


class AccuracyFirst(DeploymentAlgorithm):
    """Accuracy-First: 仅按 proxy_score 最大优先，超约束则回退"""

    def select_arch(self, node: Node, task: str, t: int,
                    topo: Topology, tables: Dict[str, pd.DataFrame],
                    traffic_gen: TrafficGenerator) -> Optional[dict]:

        candidates = self.filter_candidates(node, task, t, topo, tables)
        if not candidates:
            # 回退：放宽约束找任意候选
            df = tables[task]
            if len(df) > 0:
                row = df.iloc[0]
                return {
                    'arch_id': row['arch_id'],
                    'proxy_score': row['proxy_score_norm'],
                    'flops': row['flops'],
                    'flops_norm': row['flops_norm'],
                    'params': row['model_params'],
                    'params_norm': row['model_params_norm'],
                    'task_perf': row['task_final_performance'],
                }
            return None

        best = max(candidates, key=lambda c: c['proxy_score'])
        return best


# ============================================================
# 路由算法
# ============================================================

class RoutingAlgorithm:
    """路由算法基类"""

    def __init__(self, config: Config):
        self.config = config

    def route(self, request_task: str, src_node: Node,
              topo: Topology, tables: Dict[str, pd.DataFrame],
              deploy_algo: DeploymentAlgorithm, t: int,
              traffic_gen: TrafficGenerator) -> Tuple[Optional[Node], Optional[dict], float]:
        """
        返回: (目标节点, 选中架构, 路由时延ms)
        """
        raise NotImplementedError

    def dijkstra(self, src: int, dst: int, topo: Topology) -> float:
        """Dijkstra 最短路径（论文 A 路由）"""
        if src == dst:
            return 0.0

        dist = {src: 0.0}
        pq = [(0.0, src)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == dst:
                return d

            for v in topo.nodes:
                if v not in visited:
                    w = topo.get_delay(u, v)
                    nd = d + w
                    if nd < dist.get(v, float('inf')):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))

        return float('inf')


class RoutingOURS(RoutingAlgorithm):
    """OURS 联合路由效用最大化"""

    def route(self, request_task: str, src_node: Node,
              topo: Topology, tables: Dict[str, pd.DataFrame],
              deploy_algo: DeploymentAlgorithm, t: int,
              traffic_gen: TrafficGenerator,
              slot_candidates: Dict[Tuple[int, str], List[dict]] = None) -> Tuple[Optional[Node], Optional[dict], float]:

        if slot_candidates is None:
            slot_candidates = {}

        candidates = []
        for nid, node in topo.nodes.items():
            if node.node_type != 'edge':
                continue

            cands = slot_candidates.get((nid, request_task), [])
            if not cands:
                continue

            # OURS: 从候选中选择效用最高的
            arch = self._select_best_arch(node, cands, traffic_gen)
            if arch is None:
                continue

            # 端到端时延估算
            L_user = topo.get_delay(src_node.nid, nid)
            mu = node.gflops / (arch['flops'] / 1e9)
            lam = max(node.lambda_arrival, 0.1)
            T_queue = 1000.0 / max(mu - lam, 0.1) if mu > lam else 10000.0
            T_sla = 200.0
            if T_queue > T_sla:
                continue

            is_cached = arch['arch_id'] in node.cache
            D_pull = 0.0 if is_cached else (
                self.config.rho_pull * arch['params'] / self.config.B_cloud_mbps * 1000.0
                + topo.get_delay(nid, topo.cloud_nid)
            )
            R_total = L_user + T_queue + D_pull

            # 联合效用
            utility = (self.config.theta1 * arch['proxy_score']
                       - self.config.theta2 * (R_total / 200.0)
                       - self.config.theta3 * arch['params_norm'])

            candidates.append((utility, nid, arch, R_total))

        if not candidates:
            return None, None, float('inf')

        best = max(candidates, key=lambda x: x[0])
        best_node = topo.nodes[best[1]]
        return best_node, best[2], best[3]

    def _select_best_arch(self, node: Node, candidates: List[dict],
                          traffic_gen: TrafficGenerator) -> Optional[dict]:
        """从预过滤候选中用效用函数选最优"""
        lam = node.lambda_arrival
        lam_th = traffic_gen.lambda_th.get(node.nid, self.config.lambda_base)

        w2 = self.config.alpha * np.exp(max(0, (lam - lam_th) / lam_th))
        mem_used_ratio = node.used_memory_mb / node.memory_mb if node.memory_mb > 0 else 0
        w3 = self.config.beta * mem_used_ratio
        w1 = 1.0 / (1.0 + w2 + w3)

        best_arch = None
        best_utility = -float('inf')

        for c in candidates:
            utility = (w1 * c['proxy_score']
                       - w2 * c['flops_norm']
                       - w3 * c['params_norm'])
            if utility > best_utility:
                best_utility = utility
                best_arch = c

        return best_arch


class RoutingHeuristicA(RoutingAlgorithm):
    """论文 A 路由: Dijkstra 最短路径"""

    def route(self, request_task: str, src_node: Node,
              topo: Topology, tables: Dict[str, pd.DataFrame],
              deploy_algo: DeploymentAlgorithm, t: int,
              traffic_gen: TrafficGenerator,
              slot_candidates: Dict[Tuple[int, str], List[dict]] = None) -> Tuple[Optional[Node], Optional[dict], float]:

        if slot_candidates is None:
            slot_candidates = {}

        # Dijkstra 找所有边缘节点的最短时延
        edge_nodes = [(nid, node) for nid, node in topo.nodes.items() if node.node_type == 'edge']

        best_nid, best_dist = None, float('inf')
        for nid, node in edge_nodes:
            d = self.dijkstra(src_node.nid, nid, topo)
            if d < best_dist:
                best_dist = d
                best_nid = nid

        if best_nid is None:
            return None, None, float('inf')

        target_node = topo.nodes[best_nid]
        cands = slot_candidates.get((best_nid, request_task), [])
        if not cands:
            return target_node, None, best_dist

        # 论文A: 选 proxy_score 最高的
        arch = max(cands, key=lambda c: c['proxy_score'])
        return target_node, arch, best_dist


class RoutingGreedyB(RoutingAlgorithm):
    """论文 B 路由: 最近节点转发"""

    def route(self, request_task: str, src_node: Node,
              topo: Topology, tables: Dict[str, pd.DataFrame],
              deploy_algo: DeploymentAlgorithm, t: int,
              traffic_gen: TrafficGenerator,
              slot_candidates: Dict[Tuple[int, str], List[dict]] = None) -> Tuple[Optional[Node], Optional[dict], float]:

        if slot_candidates is None:
            slot_candidates = {}

        # 找最近节点
        best_nid, best_dist = None, float('inf')
        for nid, node in topo.nodes.items():
            if node.node_type != 'edge':
                continue
            d = topo.get_delay(src_node.nid, nid)
            if d < best_dist:
                best_dist = d
                best_nid = nid

        if best_nid is None:
            return None, None, float('inf')

        target_node = topo.nodes[best_nid]
        cands = slot_candidates.get((best_nid, request_task), [])
        if not cands:
            return target_node, None, best_dist

        # 论文B: 选 flops+params 最小的（轻量优先）
        arch = min(cands, key=lambda c: c['flops_norm'] + c['params_norm'])
        return target_node, arch, best_dist


# ============================================================
# 仿真器
# ============================================================

class Simulator:
    def __init__(self, config: Config, topology: Topology,
                 tables: Dict[str, pd.DataFrame],
                 deploy_class, routing_class,
                 traffic_mode: str = 'steady'):
        self.config = config
        self.topology = topology
        self.tables = tables
        self.tasks = list(tables.keys())
        self.traffic_gen = TrafficGenerator(config, topology, self.tasks)
        self.traffic_mode = traffic_mode

        self.deploy_algo = deploy_class(config)
        self.routing_algo = routing_class(config)

        # 统计
        self.stats = {
            'total_requests': 0,
            'success_requests': 0,
            'failed_requests': 0,
            'sla_violations': 0,
            'total_latency_ms': 0.0,
            'pull_count': 0,
            'pull_delay_ms': 0.0,
            'latencies': [],
            'per_task_perf': defaultdict(list),
            'slot_throughput': [],
            'slot_sla_violation': [],
        }

    def run(self):
        config = self.config

        for t in range(config.n_slots):
            # 1. 生成流量
            requests_by_node = self.traffic_gen.generate_slot(t, self.traffic_mode)

            # 2. 更新节点到达率
            for nid, node in self.topology.nodes.items():
                if nid in requests_by_node:
                    node.lambda_arrival = len(requests_by_node[nid]) / config.slot_duration_s
                else:
                    node.lambda_arrival = max(0, node.lambda_arrival - 5.0)

            # 3. 预计算每节点每任务的候选架构（每时隙只计算一次）
            slot_candidates: Dict[Tuple[int, str], List[dict]] = {}
            for nid, node in self.topology.nodes.items():
                if node.node_type != 'edge':
                    continue
                for task_name in self.tasks:
                    try:
                        cands = self.deploy_algo.filter_candidates(node, task_name, t, self.topology, self.tables)
                        slot_candidates[(nid, task_name)] = cands
                    except Exception:
                        slot_candidates[(nid, task_name)] = []

            # 4. 处理请求
            slot_served = 0
            slot_violated = 0

            for src_nid, reqs in requests_by_node.items():
                src_node = self.topology.nodes.get(src_nid)
                if src_node is None:
                    continue

                for task, duration in reqs:
                    self.stats['total_requests'] += 1

                    # 路由决策
                    target_node, arch, route_delay = self.routing_algo.route(
                        task, src_node, self.topology, self.tables,
                        self.deploy_algo, t, self.traffic_gen,
                        slot_candidates
                    )

                    if target_node is None or arch is None:
                        self.stats['failed_requests'] += 1
                        continue

                    # 检查/更新缓存
                    is_cached = arch['arch_id'] in target_node.cache
                    if not is_cached:
                        # 拉取
                        self.stats['pull_count'] += 1
                        pull_delay = (config.rho_pull * arch['params'] / config.B_cloud_mbps * 1000.0
                                      + self.topology.get_delay(target_node.nid, self.topology.cloud_nid))
                        self.stats['pull_delay_ms'] += pull_delay

                        # 更新缓存（LRU 简化）
                        if len(target_node.cache) >= config.cache_k:
                            target_node.cache.pop()
                        target_node.cache.add(arch['arch_id'])

                    # 部署（如果该节点尚未部署此架构）
                    if arch['arch_id'] not in target_node.deployed_archs:
                        target_node.deployed_archs.append(arch['arch_id'])
                        target_node.used_memory_mb += arch['params'] / (1024 * 1024)

                    # 服务时延（M/M/1 排队，cap 防止爆炸）
                    mu = target_node.gflops / (arch['flops'] / 1e9)  # req/s
                    lam = max(target_node.lambda_arrival, 0.1)
                    if mu > lam:
                        T_service = 1000.0 / (mu - lam)  # ms
                    else:
                        T_service = 10000.0  # 超出稳定区，惩罚值（不代表真实排队）
                    total_latency = route_delay + T_service

                    # SLA 检查
                    T_SLA = config.T_SLA_jigsaw_ms if task == 'jigsaw' else config.T_SLA_ms
                    if total_latency > T_SLA:
                        slot_violated += 1
                        self.stats['sla_violations'] += 1
                        target_node.violated_count += 1
                    else:
                        slot_served += 1
                        self.stats['success_requests'] += 1

                    self.stats['total_latency_ms'] += total_latency
                    self.stats['latencies'].append(total_latency)
                    target_node.served_count += 1
                    target_node.total_latency_ms += total_latency
                    self.stats['per_task_perf'][task].append(arch['task_perf'])

            # 4. 时隙统计
            self.stats['slot_throughput'].append(slot_served)
            self.stats['slot_sla_violation'].append(slot_violated)

            # 5. 清理已结束请求的队列（简化）
            for node in self.topology.nodes.values():
                if node.queue_length > 0:
                    node.queue_length = max(0, node.queue_length - int(node.gflops / 10))

        return self._collect_results()

    def _collect_results(self) -> dict:
        s = self.stats
        n = self.config.n_slots

        avg_latency = s['total_latency_ms'] / max(s['success_requests'], 1)
        success_rate = s['success_requests'] / max(s['total_requests'], 1)
        sla_viol_rate = s['sla_violations'] / max(s['total_requests'], 1)
        throughput = s['success_requests'] / n
        avg_pull_delay = s['pull_delay_ms'] / max(s['pull_count'], 1)

        # 节点利用率
        utilizations = []
        for node in self.topology.nodes.values():
            if node.node_type == 'edge':
                util = node.used_memory_mb / node.memory_mb if node.memory_mb > 0 else 0
                utilizations.append(util)

        # 分任务性能
        task_perf_avg = {task: np.mean(perfs) if perfs else 0
                         for task, perfs in s['per_task_perf'].items()}

        # 时延 CDF
        latencies_sorted = sorted(s['latencies'])
        cdf_x = np.linspace(0, 1, len(latencies_sorted) + 1)
        cdf_y = np.concatenate([[0], latencies_sorted])

        return {
            'avg_latency_ms': avg_latency,
            'success_rate': success_rate,
            'sla_violation_rate': sla_viol_rate,
            'throughput': throughput,
            'avg_pull_delay_ms': avg_pull_delay,
            'pull_count': s['pull_count'],
            'node_utilization_avg': np.mean(utilizations) if utilizations else 0,
            'node_utilization_std': np.std(utilizations) if utilizations else 0,
            'task_performance': task_perf_avg,
            'latencies': s['latencies'],
            'latencies_cdf_x': cdf_x.tolist(),
            'latencies_cdf_y': cdf_y.tolist(),
            'slot_throughput': s['slot_throughput'],
            'slot_sla_violation': s['slot_sla_violation'],
            'total_requests': s['total_requests'],
        }


# ============================================================
# 实验运行器
# ============================================================

ALGORITHM_MAP = {
    'OURS': (OursCEDR, RoutingOURS),
    'HEURISTIC_A': (HeuristicA, RoutingHeuristicA),
    'GREEDY_B': (GreedyB, RoutingGreedyB),
    'STATIC': (StaticBestProxy, RoutingGreedyB),
    'RESOURCE_FIRST': (ResourceFirst, RoutingGreedyB),
    'ACCURACY_FIRST': (AccuracyFirst, RoutingGreedyB),
}


def run_experiment(config: Config, topology_scale: str = 'small',
                   traffic_mode: str = 'steady',
                   algorithms: List[str] = None) -> Dict[str, dict]:
    """运行一组算法的对比实验"""

    tables = load_architecture_tables(config.data_path)
    if algorithms is None:
        algorithms = list(ALGORITHM_MAP.keys())

    results = {}

    for algo_name in algorithms:
        deploy_cls, route_cls = ALGORITHM_MAP[algo_name]
        topo = Topology(config, scale=topology_scale)

        np.random.seed(config.seed)
        random.seed(config.seed)

        sim = Simulator(config, topo, tables, deploy_cls, route_cls, traffic_mode)
        results[algo_name] = sim.run()
        print(f"[{algo_name}] lat={results[algo_name]['avg_latency_ms']:.2f}ms "
              f"succ={results[algo_name]['success_rate']:.3f} "
              f"sla_viol={results[algo_name]['sla_violation_rate']:.3f} "
              f"pull={results[algo_name]['pull_count']}")

    return results


def run_sensitivity_analysis(config: Config, param_name: str, param_values: List,
                              topology_scale: str = 'small',
                              traffic_mode: str = 'steady') -> Dict:
    """参数敏感性分析"""
    tables = load_architecture_tables(config.data_path)
    topo = Topology(config, scale=topology_scale)

    results = {}

    for val in param_values:
        old_val = getattr(config, param_name)
        setattr(config, param_name, val)

        np.random.seed(config.seed)
        random.seed(config.seed)

        deploy_cls, route_cls = ALGORITHM_MAP['OURS']
        sim = Simulator(config, topo, tables, deploy_cls, route_cls, traffic_mode)
        results[val] = sim.run()

        setattr(config, param_name, old_val)

    return results


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("云边协同部署与路由仿真实验")
    print("对齐论文 A (TPDS 2023) 和论文 B (TSC 2024)")
    print("=" * 60)

    # 实验 1: 小拓扑 + 平稳流（对齐论文 A 基础设置）
    print("\n### 实验 1: 小拓扑 (N=15) + 平稳流量 ###")
    results_steady = run_experiment(config, 'small', 'steady')
    save_results(results_steady, config.output_dir, 'exp1_steady_small.csv')

    # 实验 2: 小拓扑 + 潮汐流（对齐论文 B 周期性）
    print("\n### 实验 2: 小拓扑 (N=15) + 潮汐流量 ###")
    results_tidal = run_experiment(config, 'small', 'tidal')
    save_results(results_tidal, config.output_dir, 'exp2_tidal_small.csv')

    # 实验 3: 小拓扑 + 突发流（对齐论文 A 请求规模）
    print("\n### 实验 3: 小拓扑 (N=15) + 突发流量 ###")
    results_burst = run_experiment(config, 'small', 'burst')
    save_results(results_burst, config.output_dir, 'exp3_burst_small.csv')

    # 实验 4: 大拓扑 + 潮汐流（对齐论文 B ta2 65节点）
    print("\n### 实验 4: 大拓扑 (N=65) + 潮汐流量 ###")
    results_large = run_experiment(config, 'large', 'tidal')
    save_results(results_large, config.output_dir, 'exp4_large_tidal.csv')

    # 实验 5: 扫参 - 到达率敏感性（对齐论文 A/B 到达率扫参）
    print("\n### 实验 5: 到达率敏感性分析 ###")
    config_n = Config()
    orig_lambda = config_n.lambda_base
    sensitivity_results = {}
    for lam in [10, 30, 50, 100, 150, 200]:
        config_n.lambda_base = float(lam)
        r = run_experiment(config_n, 'small', 'steady', ['OURS'])
        sensitivity_results[lam] = r['OURS']
    config_n.lambda_base = orig_lambda

    # 生成图表
    print("\n### 生成图表 ###")
    plot_comparison(results_steady, config.output_dir, 'comparison_steady.png')
    plot_cdf(results_steady, config.output_dir, 'latency_cdf.png')
    plot_sensitivity(sensitivity_results, config.output_dir, 'sensitivity_lambda.png')

    print(f"\n实验完成! 结果保存在 {config.output_dir}")
    print_summary_table(results_steady, results_tidal, results_burst, results_large)


def save_results(results: Dict, output_dir: str, filename: str):
    """保存结果到 CSV"""
    rows = []
    for algo, r in results.items():
        row = {
            'algorithm': algo,
            'avg_latency_ms': r['avg_latency_ms'],
            'success_rate': r['success_rate'],
            'sla_violation_rate': r['sla_violation_rate'],
            'throughput': r['throughput'],
            'avg_pull_delay_ms': r['avg_pull_delay_ms'],
            'pull_count': r['pull_count'],
            'node_utilization_avg': r['node_utilization_avg'],
            'node_utilization_std': r['node_utilization_std'],
            'total_requests': r['total_requests'],
        }
        for task, perf in r['task_performance'].items():
            row[f'perf_{task}'] = perf
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, filename), index=False)


def plot_comparison(results: Dict, output_dir: str, filename: str):
    """柱状图对比（论文 A 风格）"""
    algos = list(results.keys())
    metrics = ['avg_latency_ms', 'success_rate', 'sla_violation_rate', 'throughput']
    titles = ['Avg E2E Latency (ms)', 'Success Rate', 'SLA Violation Rate', 'Throughput']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, len(algos)))

    for ax, metric, title in zip(axes.flat, metrics, titles):
        vals = [results[a][metric] for a in algos]
        bars = ax.bar(algos, vals, color=colors)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(vals),
                     f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


def plot_cdf(results: Dict, output_dir: str, filename: str):
    """时延 CDF（论文 B 风格）"""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for (algo, r), color in zip(results.items(), colors):
        x, y = r['latencies_cdf_x'], r['latencies_cdf_y']
        ax.plot(y, x, label=algo, color=color, linewidth=2)

    ax.set_xlabel('E2E Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('Latency CDF Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


def plot_sensitivity(sensitivity_results: Dict, output_dir: str, filename: str):
    """敏感性分析曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    lambdas = sorted(sensitivity_results.keys())
    latencies = [sensitivity_results[l]['avg_latency_ms'] for l in lambdas]
    sla_viols = [sensitivity_results[l]['sla_violation_rate'] for l in lambdas]

    ax2 = ax.twinx()
    l1, = ax.plot(lambdas, latencies, 'b-o', label='Avg Latency', linewidth=2)
    l2, = ax2.plot(lambdas, sla_viols, 'r-s', label='SLA Violation', linewidth=2)

    ax.set_xlabel('Arrival Rate λ (req/s)')
    ax.set_ylabel('Avg Latency (ms)', color='b')
    ax2.set_ylabel('SLA Violation Rate', color='r')
    ax.legend(handles=[l1, l2], loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.title('Sensitivity to Arrival Rate (OURS Algorithm)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


def print_summary_table(r1, r2, r3, r4):
    """打印汇总对比表"""
    print("\n" + "=" * 90)
    print("实验结果汇总")
    print("=" * 90)
    print(f"{'Algorithm':<20} {'Latency(ms)':<15} {'SuccessRate':<15} {'SLAViol':<15} {'Throughput':<15}")
    print("-" * 90)

    for algo in ['OURS', 'HEURISTIC_A', 'GREEDY_B', 'STATIC', 'RESOURCE_FIRST', 'ACCURACY_FIRST']:
        r = r1.get(algo, {})
        if r:
            print(f"{algo:<20} {r['avg_latency_ms']:<15.2f} {r['success_rate']:<15.3f} "
                  f"{r['sla_violation_rate']:<15.3f} {r['throughput']:<15.1f}")

    print("\n--- 潮汐流 (论文 B 风格) ---")
    for algo in ['OURS', 'HEURISTIC_A', 'GREEDY_B', 'STATIC', 'RESOURCE_FIRST', 'ACCURACY_FIRST']:
        r = r2.get(algo, {})
        if r:
            print(f"{algo:<20} {r['avg_latency_ms']:<15.2f} {r['success_rate']:<15.3f} "
                  f"{r['sla_violation_rate']:<15.3f} {r['throughput']:<15.1f}")

    print("\n--- 突发流 ---")
    for algo in ['OURS', 'HEURISTIC_A', 'GREEDY_B', 'STATIC', 'RESOURCE_FIRST', 'ACCURACY_FIRST']:
        r = r3.get(algo, {})
        if r:
            print(f"{algo:<20} {r['avg_latency_ms']:<15.2f} {r['success_rate']:<15.3f} "
                  f"{r['sla_violation_rate']:<15.3f} {r['throughput']:<15.1f}")

    print("\n--- 大拓扑 N=65 (论文 B 风格) ---")
    for algo in ['OURS', 'HEURISTIC_A', 'GREEDY_B', 'STATIC', 'RESOURCE_FIRST', 'ACCURACY_FIRST']:
        r = r4.get(algo, {})
        if r:
            print(f"{algo:<20} {r['avg_latency_ms']:<15.2f} {r['success_rate']:<15.3f} "
                  f"{r['sla_violation_rate']:<15.3f} {r['throughput']:<15.1f}")
