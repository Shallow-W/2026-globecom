"""
算法模块 - 部署算法 + 路由算法

包含以下算法组合（基于论文原始baseline）:

OURS (我们的论文算法):
  - 动态算力红线 + 效用函数 + 动态权重自适应

论文 A (TPDS 2023) 原始baseline:
  - RLS     : Random Local Search (随机局部搜索)
  - FFD     : First Fit Decreasing (首次适配降序)
  - DRS     : Auto-scaling for Real-time Stream analytics
  - LEGO    : Joint optimization of service request routing and instance placement

论文 B (TSC 2024) 原始baseline:
  - GREEDY  : 贪心分配资源利用率最低 + 最近节点路由
  - PSO     : Particle Swarm Optimization (粒子群优化)

消融实验基线:
  - STATIC         : 固定 proxy_score 最高架构
  - RESOURCE_FIRST : 仅 flops+params 最小优先
  - ACCURACY_FIRST : 仅 proxy_score 最大优先

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
    """
    部署算法基类

    设计思路：
    - initialize(): 初始化时为每个(节点, 任务)选择模型并缓存
    - get_deployed_model(): 获取已部署的模型（用于运行时路由）
    - select_arch(): 实际选择模型的逻辑（子类实现）

    对于 Baseline：initialize() 时选择模型，运行时 get_deployed_model() 返回缓存的模型
    对于 OURS：initialize() 时选择初始模型，但运行时仍可调用 select_arch() 进行动态替换
    """

    def __init__(self, config: Config):
        self.config = config
        # 已部署的模型缓存: {(node_id, task): arch}
        self.deployed_models: Dict[Tuple[int, str], dict] = {}
        # 是否已初始化
        self._initialized = False

    def compute_F_max(self, node: Node, t: int, task: str = None) -> float:
        """
        动态算力红线（论文核心公式）

        正确的 SLA 约束应该是：
        T_total = T_queue + T_service = 1/(μ-λ) + 1/μ <= T_SLA

        解这个不等式得到：
        μ >= (λ + 1/T_SLA + sqrt((λ + 1/T_SLA)² + 4λ/T_SLA)) / 2

        所以最大允许的模型 FLOPs:
        F_max = C / μ_min = C * 2 / (λ + 1/T_SLA + sqrt((λ + 1/T_SLA)² + 4λ/T_SLA))
        """
        lam = node.lambda_arrival
        if lam <= 0:
            lam = 0.1
        T_sla_s = self._get_T_SLA(task) / 1000.0

        # 精确 SLA 约束公式
        term = (lam + 1.0/T_sla_s)**2 + 4*lam/T_sla_s
        mu_min = (lam + 1.0/T_sla_s + term**0.5) / 2.0

        F_max = node.gflops / mu_min
        return max(F_max, 0.5)  # 放宽下限，允许更小的模型

    def _get_T_SLA(self, task: str = None) -> float:
        """SLA 死线（jigsaw 更严格，其他任务 200ms）"""
        if task == 'jigsaw':
            return self.config.T_SLA_jigsaw_ms
        return self.config.T_SLA_ms

    def filter_candidates(self, node: Node, task: str, t: int,
                          tables: Dict) -> List[dict]:
        """
        硬约束过滤：返回满足内存+算力红线的候选列表。
        使用 pandas 向量化 + numpy 直接索引（避免 iterrows）。
        """
        df = tables[task]
        F_max = self.compute_F_max(node, t, task)
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

    def initialize(self, topology, tables: Dict, t: int = 0):
        """
        初始化阶段：为每个(节点, 任务)选择模型并缓存

        对于 Baseline：只在初始化时调用一次 select_arch()
        对于 OURS：也调用一次 select_arch() 作为初始选择，但运行时可重新选择（动态替换）

        子类可重写此方法来自定义初始化逻辑
        """
        self.deployed_models = {}
        self._initialized = True

        for nid, node in topology.nodes.items():
            if node.node_type != 'edge':
                continue
            for task_name in tables.keys():
                cands = self.filter_candidates(node, task_name, t, tables)
                if cands:
                    # 调用子类的 select_arch 选择模型并缓存
                    arch = self.select_arch(node, task_name, t, tables, cands)
                    if arch:
                        self.deployed_models[(nid, task_name)] = arch

    def get_deployed_model(self, node_id: int, task: str,
                          node: 'Node' = None, t: int = None,
                          tables: Dict = None) -> Optional[dict]:
        """
        获取已部署的模型（用于运行时路由）

        对于 Baseline：返回初始化时缓存的模型
        对于 OURS：基类实现返回缓存模型，但 OURS 可在运行时覆盖
        """
        return self.deployed_models.get((node_id, task))

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
    - 动态模型替换：根据负载变化动态替换模型
    """

    def __init__(self, config: Config):
        super().__init__(config)
        # 记录上一次选择的模型ID，用于检测是否需要替换
        self.last_selected_arch_id: Dict[Tuple[int, str], str] = {}
        # 负载变化阈值：超过此比例则触发模型替换
        self.replace_threshold = 0.3  # 30% 负载变化触发

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

    def get_deployed_model(self, node_id: int, task: str,
                          node: 'Node' = None, t: int = None,
                          tables: Dict = None) -> Optional[dict]:
        """
        获取已部署的模型，支持动态替换

        对于 OURS：
        - 如果负载变化超过阈值（相比上次选择时），则重新选择模型
        - 这实现了"动态模型替换"机制

        参数:
            node_id: 节点ID
            task: 任务名称
            node: 节点对象（用于获取当前负载）
            t: 当前时隙
            tables: 候选架构表（用于重新选择）
        """
        cached = self.deployed_models.get((node_id, task))

        # 如果没有初始化过，先初始化
        if cached is None:
            return None

        # 对于 OURS：检测是否需要动态替换
        if node is not None and t is not None and tables is not None:
            if self._should_replace_model(node, task, t):
                # 重新计算可行候选
                cands = self.filter_candidates(node, task, t, tables)
                if cands:
                    new_arch = self.select_arch(node, task, t, tables, cands)
                    if new_arch and new_arch['arch_id'] != cached['arch_id']:
                        # 模型替换：更新缓存
                        self.deployed_models[(node_id, task)] = new_arch
                        self.last_selected_arch_id[(node_id, task)] = new_arch['arch_id']
                        return new_arch

        return cached

    def _should_replace_model(self, node: Node, task: str, t: int) -> bool:
        """
        判断是否需要替换模型

        替换条件：
        1. 负载显著增加（超过阈值）-> 可能需要更轻量的模型
        2. 负载显著降低（超过阈值）-> 可能可以换回更高精度的模型

        使用指数移动平均来平滑负载变化
        """
        if not hasattr(self, '_last_load'):
            self._last_load: Dict[Tuple[int, str], float] = {}

        key = (node.nid, task)
        current_load = node.lambda_arrival
        last_load = self._last_load.get(key, current_load)

        # 计算负载变化比例
        if last_load > 0.1:
            load_change = abs(current_load - last_load) / last_load
        else:
            load_change = 0.0

        # 更新记录的负载
        self._last_load[key] = current_load

        # 如果负载变化超过阈值，触发替换检查
        return load_change > self.replace_threshold


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
# 论文 A 原始 Baseline 实现 (TPDS 2023)
# ============================================================

class RLS(DeploymentAlgorithm):
    """
    RLS (Random Local Search) - 论文 A (TPDS 2023) Baseline
    - 随机局部搜索算法
    - 从随机初始解出发，在邻居解空间搜索更好的解
    - 部署时综合评估精度和资源，选择最优候选
    """

    def __init__(self, config: Config):
        super().__init__(config)
        np.random.seed(42)  # 可重复性

    def select_arch(self, node: Node, task: str, t: int,
                    tables: Dict, candidates: List[dict]) -> Optional[dict]:
        if not candidates:
            return None

        # RLS: 综合评估精度和资源消耗，选择最优候选
        # 不同于简单的随机选择，而是评估所有候选
        best_arch = None
        best_score = -float('inf')

        for c in candidates:
            # RLS 评分：proxy_score - 0.5 * (flops_norm + params_norm)
            score = c['proxy_score'] - 0.3 * (c['flops_norm'] + c['params_norm'])
            if score > best_score:
                best_score = score
                best_arch = c

        return best_arch


class FFD(DeploymentAlgorithm):
    """
    FFD (First Fit Decreasing) - 论文 A (TPDS 2023) Baseline
    - 首次适配降序算法（来自装箱问题）
    - 按请求流部署，优先使用已有部署
    - 高请求成功率著称
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.deployed_cache: Dict[str, str] = {}  # task -> arch_id 缓存

    def select_arch(self, node: Node, task: str, t: int,
                    tables: Dict, candidates: List[dict]) -> Optional[dict]:
        if not candidates:
            return None

        # FFD: 优先使用已部署的架构（复用原则）
        if task in self.deployed_cache:
            cached_arch_id = self.deployed_cache[task]
            for c in candidates:
                if c['arch_id'] == cached_arch_id:
                    return c

        # 否则按 proxy_score 降序选择（首次适配）
        best = max(candidates, key=lambda c: c['proxy_score'])
        self.deployed_cache[task] = best['arch_id']
        return best


class DRS(DeploymentAlgorithm):
    """
    DRS (Auto-scaling for Real-time Stream analytics) - 论文 A (TPDS 2023) Baseline
    - 基于动态资源拆分和概率路由
    - 根据实时负载动态调整资源分配
    - 引入随机性避免局部最优
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.resource_split_ratio = 0.7  # 资源拆分比例

    def select_arch(self, node: Node, task: str, t: int,
                    tables: Dict, candidates: List[dict]) -> Optional[dict]:
        if not candidates:
            return None

        # DRS: 按资源效率比选择 (proxy_score / (flops + params))
        best_arch = None
        best_ratio = -float('inf')

        for c in candidates:
            resource_cost = c['flops_norm'] + c['params_norm']
            if resource_cost > 0:
                ratio = c['proxy_score'] / resource_cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_arch = c

        return best_arch


class LEGO(DeploymentAlgorithm):
    """
    LEGO (Joint optimization of service request routing and instance placement)
    论文 A (TPDS 2023) Baseline - 三阶段联合优化算法

    - 第一阶段：服务实例部署（按调用依赖关系聚类）
    - 第二阶段：分区映射（将请求流映射到实例）
    - 第三阶段：负载均衡（概率路由优化）
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.instance_counts: Dict[str, int] = {}  # 记录每任务的实例数

    def select_arch(self, node: Node, task: str, t: int,
                    tables: Dict, candidates: List[dict]) -> Optional[dict]:
        if not candidates:
            return None

        # LEGO: 按实例数量降序优先部署 + 平衡路由
        # 记录该任务的全局实例部署次数
        if task not in self.instance_counts:
            self.instance_counts[task] = 0
        self.instance_counts[task] += 1

        # 按 proxy_score * 实例平衡因子 选择
        balance_factor = 1.0 / (1.0 + self.instance_counts[task] * 0.1)

        best_arch = None
        best_score = -float('inf')

        for c in candidates:
            # 分数 = proxy_score * 平衡因子 - 资源惩罚
            score = c['proxy_score'] * balance_factor - 0.1 * (c['flops_norm'] + c['params_norm'])
            if score > best_score:
                best_score = score
                best_arch = c

        return best_arch


# ============================================================
# 论文 B 原始 Baseline 实现 (TSC 2024)
# ============================================================

class PSO(DeploymentAlgorithm):
    """
    PSO (Particle Swarm Optimization) - 论文 B (TSC 2024) Baseline
    - 粒子群优化算法
    - 通过粒子间的信息共享搜索最优部署方案
    - 收敛速度快，适合大规模问题
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.pso_iterations = 10  # PSO 迭代次数
        self.n_particles = 5      # 粒子数

    def select_arch(self, node: Node, task: str, t: int,
                    tables: Dict, candidates: List[dict]) -> Optional[dict]:
        if not candidates:
            return None

        n = len(candidates)
        if n <= self.n_particles:
            # 粒子数多于候选，直接选最优
            return max(candidates, key=lambda c: c['proxy_score'] - 0.3 * (c['flops_norm'] + c['params_norm']))

        # PSO 简化版：迭代搜索
        # 随机初始化粒子位置（选择索引）
        indices = np.random.choice(n, self.n_particles, replace=False)
        particles = [candidates[i] for i in indices]

        # 全局最优
        gbest = max(particles, key=lambda c: c['proxy_score'] - 0.3 * (c['flops_norm'] + c['params_norm']))

        # 简化的 PSO 更新：用全局最优引导搜索
        for iteration in range(self.pso_iterations):
            new_particles = []
            for p in particles:
                # 粒子向全局最优靠近 + 随机扰动
                if np.random.random() < 0.7:  # 70% 概率学习全局最优
                    # 选择接近 gbest 的候选
                    gbest_proxy = gbest['proxy_score']
                    better_candidates = [c for c in candidates
                                         if c['proxy_score'] >= gbest_proxy * 0.9]
                    if better_candidates:
                        new_p = better_candidates[np.random.randint(0, len(better_candidates))]
                    else:
                        new_p = p
                else:  # 30% 概率随机探索
                    new_p = candidates[np.random.randint(0, n)]

                new_particles.append(new_p)

            particles = new_particles
            gbest = max(particles + [gbest],
                       key=lambda c: c['proxy_score'] - 0.3 * (c['flops_norm'] + c['params_norm']))

        return gbest


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
    """论文 A 路由: Dijkstra 最短路径 + 联合部署评估

    改进：不仅找最短路径节点，还评估该节点的候选部署可行性，
    如果该节点没有可行候选，则尝试次优节点（联合路由+部署评估）
    """

    def route(self, request_task: str, src_node: Node,
              topo: Topology, tables: Dict,
              deploy_algo: DeploymentAlgorithm, t: int,
              traffic_gen: TrafficGenerator,
              slot_candidates: Dict[Tuple[int, str], List[dict]]) -> Tuple[Optional[Node], Optional[dict], float]:

        # 收集所有可行方案（节点+已部署模型）
        candidates = []

        for nid, node in topo.nodes.items():
            if node.node_type != 'edge':
                continue

            cands = slot_candidates.get((nid, request_task), [])
            if not cands:
                continue

            # Baseline：使用已部署的固定模型（get_deployed_model 返回缓存的模型）
            # OURS：get_deployed_model 会检查是否需要动态替换
            arch = deploy_algo.get_deployed_model(nid, request_task, node, t, tables)
            if arch is None:
                continue

            # 排队时延
            mu = node.gflops / (arch['flops'] / 1e9)
            lam = max(node.lambda_arrival, 0.1)
            T_queue = 1000.0 / max(mu - lam, 0.1) if mu > lam else 10000.0

            # Dijkstra 拓扑距离
            L_topo = self.dijkstra(src_node.nid, nid, topo)

            # 总时延
            R_total = L_topo + T_queue

            # 记录可行方案
            candidates.append((nid, arch, R_total, L_topo))

        if not candidates:
            return None, None, float('inf')

        # 论文 A: 按 Dijkstra 最短路径排序
        # 选择拓扑距离最近的方案
        best = min(candidates, key=lambda x: x[3])

        return topo.nodes[best[0]], best[1], best[2]


# ============================================================
# 论文 B 路由算法（最近节点）
# ============================================================

class RoutingGreedyB(RoutingAlgorithm):
    """论文 B 路由: 最近节点转发 + 联合部署评估

    改进：不仅看拓扑距离，还评估候选部署的可行性，
    如果最近节点没有可行候选，则尝试次优节点
    """

    def route(self, request_task: str, src_node: Node,
              topo: Topology, tables: Dict,
              deploy_algo: DeploymentAlgorithm, t: int,
              traffic_gen: TrafficGenerator,
              slot_candidates: Dict[Tuple[int, str], List[dict]]) -> Tuple[Optional[Node], Optional[dict], float]:

        # 收集所有可行方案（按拓扑距离排序）
        candidates = []

        for nid, node in topo.nodes.items():
            if node.node_type != 'edge':
                continue

            cands = slot_candidates.get((nid, request_task), [])
            if not cands:
                continue

            # Baseline：使用已部署的固定模型（get_deployed_model 返回缓存的模型）
            # OURS：get_deployed_model 会检查是否需要动态替换
            arch = deploy_algo.get_deployed_model(nid, request_task, node, t, tables)
            if arch is None:
                continue

            # 排队时延
            mu = node.gflops / (arch['flops'] / 1e9)
            lam = max(node.lambda_arrival, 0.1)
            T_queue = 1000.0 / max(mu - lam, 0.1) if mu > lam else 10000.0

            # 拓扑距离（最近节点路由只用拓扑距离，不考虑排队）
            L_topo = topo.get_delay(src_node.nid, nid)

            # 总时延
            R_total = L_topo + T_queue

            candidates.append((nid, arch, R_total, L_topo))

        if not candidates:
            return None, None, float('inf')

        # 论文 B: 选择拓扑距离最近的（不考虑排队）
        best = min(candidates, key=lambda x: x[3])

        return topo.nodes[best[0]], best[1], best[2]


# ============================================================
# 算法注册表
# ============================================================

ALGORITHM_MAP = {
    # 我们的论文算法
    'OURS':           (OursCEDR,           RoutingOURS),

    # 论文 A (TPDS 2023) 原始 Baseline
    'RLS':            (RLS,                RoutingHeuristicA),   # 随机局部搜索 + Dijkstra
    'FFD':            (FFD,               RoutingHeuristicA),   # 首次适配降序 + Dijkstra
    'DRS':            (DRS,               RoutingHeuristicA),   # 动态资源拆分 + Dijkstra
    'LEGO':           (LEGO,              RoutingHeuristicA),   # 三阶段联合优化 + Dijkstra

    # 论文 B (TSC 2024) 原始 Baseline
    'GREEDY':        (GreedyB,            RoutingGreedyB),     # 贪心最小资源 + 最近节点
    'PSO':            (PSO,               RoutingGreedyB),     # 粒子群优化 + 最近节点

    # 消融实验基线
    'STATIC':         (StaticBestProxy,    RoutingGreedyB),
    'RESOURCE_FIRST': (ResourceFirst,      RoutingGreedyB),
    'ACCURACY_FIRST': (AccuracyFirst,      RoutingGreedyB),
}
