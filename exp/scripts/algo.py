"""
算法模块 - 部署算法 + 路由算法

包含 6 种算法组合:
  1. OURS           : 动态启发式部署 + 联合路由效用最大化
  2. HEURISTIC_A    : 实例数降序→平均距离最短 + Dijkstra 路由
  3. GREEDY_B       : 资源利用率最低 + 最近节点路由
  4. STATIC         : 固定 proxy_score 最高架构
  5. RESOURCE_FIRST : 仅 flops+params 最小优先
  6. ACCURACY_FIRST : 仅 proxy_score 最大优先

算法设计基于 globecom.tex，实验设置参考论文 A (TPDS 2023) 和论文 B (TSC 2024)。
"""

import numpy as np
import heapq
from typing import List, Dict, Tuple, Optional
from cfg import Config
from data import load_architecture_tables
from topo import Node, Topology
from traffic import TrafficGenerator


# ============================================================
# 部署算法基类
# ============================================================

class DeploymentAlgorithm:
    """部署算法基类"""

    def __init__(self, config: Config):
        self.config = config

    def compute_F_max(self, node: Node, t: int) -> float:
        """
        动态算力红线（论文核心公式）
        F_i_max(t) = C_i_max / (λ_i(t) + 1/T_SLA)
        """
        lam = node.lambda_arrival
        if lam <= 0:
            lam = 0.1
        T_sla_s = self._get_T_SLA(node) / 1000.0
        F_max = node.gflops / (lam + 1.0 / T_sla_s)
        return max(F_max, 1.0)

    def _get_T_SLA(self, node: Node) -> float:
        """SLA 死线（jigsaw 更严格）"""
        return self.config.T_SLA_jigsaw_ms

    def filter_candidates(self, node: Node, task: str, t: int,
                          tables: Dict) -> List[dict]:
        """
        硬约束过滤：返回满足内存+算力红线的候选列表。
        使用 pandas 向量化 + numpy 直接索引（避免 iterrows）。
        """
        df = tables[task]
        F_max = self.compute_F_max(node, t)
        M_avail = node.memory_mb - node.used_memory_mb

        # 向量化布尔过滤
        params_mb = df['model_params'].values / (1024 * 1024)
        flops_gf = df['flops'].values / 1e9

        mask = (params_mb <= M_avail) & (flops_gf <= F_max)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            return []

        # 直接用 numpy 数组索引，避免 iterrows
        arch_ids = df['arch_id'].values[idx]
        proxy_scores = df['proxy_score_norm'].values[idx]
        flops_arr = df['flops'].values[idx]
        flops_norm = df['flops_norm'].values[idx]
        params_arr = df['model_params'].values[idx]
        params_norm = df['model_params_norm'].values[idx]
        proxy_raw = df['proxy_score'].values[idx]
        task_perf = df['task_final_performance'].values[idx]

        return [
            {
                'arch_id': str(arch_ids[i]),
                'proxy_score': float(proxy_scores[i]),
                'flops': float(flops_arr[i]),
                'flops_norm': float(flops_norm[i]),
                'params': float(params_arr[i]),
                'params_norm': float(params_norm[i]),
                'proxy_raw': float(proxy_raw[i]),
                'task_perf': float(task_perf[i]),
            }
            for i in range(len(idx))
        ]

    def select_arch(self, node: Node, task: str, t: int,
                    tables: Dict, candidates: List[dict]) -> Optional[dict]:
        """从候选中选择最优架构（子类实现）"""
        raise NotImplementedError


# ============================================================
# OURS 部署算法（基于 globecom.tex）
# ============================================================

class OursCEDR(DeploymentAlgorithm):
    """
    OURS: 动态启发式部署算法
    - 动态算力红线（论文公式）
    - 硬约束过滤
    - 效用函数 + 动态权重自适应降级
    """

    def select_arch(self, node: Node, task: str, t: int,
                    tables: Dict, candidates: List[dict]) -> Optional[dict]:
        if not candidates:
            return None

        lam = node.lambda_arrival
        lam_th = self.config.lambda_base * self.config.lambda_th_percentile

        # 动态权重（论文公式 7-8）
        w2 = self.config.alpha * np.exp(max(0.0, (lam - lam_th) / lam_th))
        mem_used = node.used_memory_mb / node.memory_mb if node.memory_mb > 0 else 0.0
        w3 = self.config.beta * mem_used
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


# ============================================================
# 论文 A 算法（启发式部署 + Dijkstra 路由）
# ============================================================

class HeuristicA(DeploymentAlgorithm):
    """
    论文 A (TPDS 2023) 部署算法变体:
    - 按服务实例数量降序排列
    - 部署到平均距离最短的节点
    - 简化：选择 proxy_score 最高的候选（对应拓扑亲和性）
    """

    def select_arch(self, node: Node, task: str, t: int,
                    tables: Dict, candidates: List[dict]) -> Optional[dict]:
        if not candidates:
            return None
        # 论文 A: proxy_score 最高 → 对应"平均距离最短"的近似
        return max(candidates, key=lambda c: c['proxy_score'])


# ============================================================
# 论文 B 算法（贪心分配 + 最近节点路由）
# ============================================================

class GreedyB(DeploymentAlgorithm):
    """
    论文 B (TSC 2024) Greedy 部署算法变体:
    - 贪心选择 flops+params 最小的轻量架构
    - 对应"资源利用率最低"的代理
    """

    def select_arch(self, node: Node, task: str, t: int,
                    tables: Dict, candidates: List[dict]) -> Optional[dict]:
        if not candidates:
            return None
        return min(candidates, key=lambda c: c['flops_norm'] + c['params_norm'])


# ============================================================
# 静态/简单基线
# ============================================================

class StaticBestProxy(DeploymentAlgorithm):
    """Static-BestProxy: 全程固定 proxy_score 最高的架构（提前选好）"""

    def __init__(self, config: Config):
        super().__init__(config)
        self.fixed_archs: Dict[str, dict] = {}

    def select_arch(self, node: Node, task: str, t: int,
                    tables: Dict, candidates: List[dict]) -> Optional[dict]:
        if task not in self.fixed_archs and candidates:
            best = max(candidates, key=lambda c: c['proxy_score'])
            self.fixed_archs[task] = best
        return self.fixed_archs.get(task)


class ResourceFirst(DeploymentAlgorithm):
    """Resource-First: 仅按 flops+params 最小优先"""

    def select_arch(self, node: Node, task: str, t: int,
                    tables: Dict, candidates: List[dict]) -> Optional[dict]:
        if not candidates:
            return None
        return min(candidates, key=lambda c: c['flops_norm'] + c['params_norm'])


class AccuracyFirst(DeploymentAlgorithm):
    """Accuracy-First: 仅按 proxy_score 最大优先，超约束则回退"""

    def select_arch(self, node: Node, task: str, t: int,
                    tables: Dict, candidates: List[dict]) -> Optional[dict]:
        if not candidates:
            # 回退：选任意候选（最轻量的）
            if candidates:
                return min(candidates, key=lambda c: c['flops_norm'] + c['params_norm'])
            return None
        return max(candidates, key=lambda c: c['proxy_score'])


# ============================================================
# 路由算法基类
# ============================================================

class RoutingAlgorithm:
    """路由算法基类"""

    def __init__(self, config: Config):
        self.config = config

    def dijkstra(self, src: int, dst: int, topo: Topology) -> float:
        """Dijkstra 最短路径"""
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
                    nd = d + topo.get_delay(u, v)
                    if nd < dist.get(v, float('inf')):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
        return float('inf')

    def route(self, request_task: str, src_node: Node,
              topo: Topology, tables: Dict,
              deploy_algo: DeploymentAlgorithm, t: int,
              traffic_gen: TrafficGenerator,
              slot_candidates: Dict[Tuple[int, str], List[dict]]) -> Tuple[Optional[Node], Optional[dict], float]:
        """
        路由决策。
        Returns: (目标节点, 选中架构, 路由时延ms)
        """
        raise NotImplementedError


# ============================================================
# OURS 路由算法
# ============================================================

class RoutingOURS(RoutingAlgorithm):
    """OURS 联合路由效用最大化"""

    def route(self, request_task: str, src_node: Node,
              topo: Topology, tables: Dict,
              deploy_algo: DeploymentAlgorithm, t: int,
              traffic_gen: TrafficGenerator,
              slot_candidates: Dict[Tuple[int, str], List[dict]]) -> Tuple[Optional[Node], Optional[dict], float]:

        candidates = []
        for nid, node in topo.nodes.items():
            if node.node_type != 'edge':
                continue

            cands = slot_candidates.get((nid, request_task), [])
            if not cands:
                continue

            arch = deploy_algo.select_arch(node, request_task, t, tables, cands)
            if arch is None:
                continue

            # 排队时延（M/M/1）
            mu = node.gflops / (arch['flops'] / 1e9)
            lam = max(node.lambda_arrival, 0.1)
            T_queue = 1000.0 / max(mu - lam, 0.1) if mu > lam else 10000.0

            # SLA 预检（论文 A 约束过滤）
            if T_queue > self.config.T_SLA_ms:
                continue

            # 云边拉取开销
            is_cached = arch['arch_id'] in node.cache
            if is_cached:
                D_pull = 0.0
            else:
                D_pull = (self.config.rho_pull * arch['params'] / self.config.B_cloud_mbps * 1000.0
                          + topo.get_delay(nid, topo.cloud_nid))

            # 端到端时延
            L_user = topo.get_delay(src_node.nid, nid)
            R_total = L_user + T_queue + D_pull

            # 联合效用函数（论文公式 9）
            utility = (self.config.theta1 * arch['proxy_score']
                       - self.config.theta2 * (R_total / self.config.T_SLA_ms)
                       - self.config.theta3 * arch['params_norm'])

            candidates.append((utility, nid, arch, R_total))

        if not candidates:
            return None, None, float('inf')

        best = max(candidates, key=lambda x: x[0])
        return topo.nodes[best[1]], best[2], best[3]


# ============================================================
# 论文 A 路由算法（Dijkstra 最短路径）
# ============================================================

class RoutingHeuristicA(RoutingAlgorithm):
    """论文 A 路由: Dijkstra 最短路径"""

    def route(self, request_task: str, src_node: Node,
              topo: Topology, tables: Dict,
              deploy_algo: DeploymentAlgorithm, t: int,
              traffic_gen: TrafficGenerator,
              slot_candidates: Dict[Tuple[int, str], List[dict]]) -> Tuple[Optional[Node], Optional[dict], float]:

        # 找所有边缘节点中 Dijkstra 最短路径
        best_nid, best_dist = None, float('inf')
        for nid, node in topo.nodes.items():
            if node.node_type != 'edge':
                continue
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

        arch = deploy_algo.select_arch(target_node, request_task, t, tables, cands)
        return target_node, arch, best_dist


# ============================================================
# 论文 B 路由算法（最近节点）
# ============================================================

class RoutingGreedyB(RoutingAlgorithm):
    """论文 B 路由: 最近节点转发"""

    def route(self, request_task: str, src_node: Node,
              topo: Topology, tables: Dict,
              deploy_algo: DeploymentAlgorithm, t: int,
              traffic_gen: TrafficGenerator,
              slot_candidates: Dict[Tuple[int, str], List[dict]]) -> Tuple[Optional[Node], Optional[dict], float]:

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

        arch = deploy_algo.select_arch(target_node, request_task, t, tables, cands)
        return target_node, arch, best_dist


# ============================================================
# 算法注册表
# ============================================================

ALGORITHM_MAP = {
    'OURS':           (OursCEDR,           RoutingOURS),
    'HEURISTIC_A':    (HeuristicA,         RoutingHeuristicA),
    'GREEDY_B':       (GreedyB,            RoutingGreedyB),
    'STATIC':         (StaticBestProxy,    RoutingGreedyB),
    'RESOURCE_FIRST': (ResourceFirst,       RoutingGreedyB),
    'ACCURACY_FIRST': (AccuracyFirst,      RoutingGreedyB),
}
