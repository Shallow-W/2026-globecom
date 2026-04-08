"""
服务部署算法：将微服务实例部署到边缘节点

部署矩阵 X: X[node][service] = 实例数量
约束: CPU 核心数 + GPU 显存
"""
import numpy as np
from math import ceil


class Deployment:
    """部署矩阵，记录每个节点上每个服务的实例数"""

    def __init__(self, n_nodes, n_services):
        self.n_nodes = n_nodes
        self.n_services = n_services
        self.X = np.zeros((n_nodes, n_services), dtype=int)

    def set(self, node, service, count):
        self.X[node][service] = count

    def get(self, node, service):
        return self.X[node][service]

    def total_instances(self, service):
        return int(np.sum(self.X[:, service]))

    def routing_probabilities(self, service):
        """P[v] = X[v][s] / sum(X[:,s])"""
        total = self.total_instances(service)
        if total == 0:
            return np.zeros(self.n_nodes)
        return self.X[:, service] / total

    def total_deployed(self):
        return int(np.sum(self.X))

    def copy(self):
        d = Deployment(self.n_nodes, self.n_services)
        d.X = self.X.copy()
        return d

    def __repr__(self):
        lines = ["Deployment:"]
        for v in range(self.n_nodes):
            deployed = [(s, self.X[v][s]) for s in range(self.n_services) if self.X[v][s] > 0]
            if deployed:
                lines.append(f"  Node {v}: {deployed}")
        return "\n".join(lines)


def _find_feasible_node(rng, n_nodes, cpu_remain, gpu_remain, svc):
    """找到可以放置 svc 一个实例的节点，返回节点索引或 -1"""
    candidates = [
        v for v in range(n_nodes)
        if cpu_remain[v] >= 1 and gpu_remain[v] >= svc.gpu_per_instance
    ]
    if not candidates:
        return -1
    return int(rng.choice(candidates))


def _required_instances(lambda_s, services, margin=1.2):
    """计算每个服务所需实例数 N_m = ceil(lambda_m / mu_m * margin)，保证 rho 远离 1"""
    required = []
    for m in range(len(services)):
        if lambda_s is not None and lambda_s[m] > 0:
            required.append(max(1, int(ceil(lambda_s[m] / services[m].service_rate * margin))))
        else:
            required.append(1)
    return required


def random_deployment(network, services, lambda_s=None, seed=None):
    """
    随机部署基线 (RLS)：将微服务实例分配到满足资源约束的节点。

    策略:
      1. 计算每个服务所需实例数 N_m = ceil(lambda_m / mu_m * margin)
      2. 按 N_m 降序排列，优先满足高需求服务
      3. 随机选择可行节点放置实例

    参数:
        network:  EdgeNetwork 实例
        services: Service 列表
        lambda_s: 每服务聚合到达率 (可选)
        seed:     随机种子
    """
    rng = np.random.default_rng(seed)
    n_nodes = network.n_nodes
    n_services = len(services)
    deployment = Deployment(n_nodes, n_services)

    cpu_remain = network.cpu_capacity.copy()
    gpu_remain = network.gpu_capacity.copy()

    # 计算所需实例数并按降序排列
    required = _required_instances(lambda_s, services)
    svc_order = sorted(range(n_services), key=lambda i: required[i], reverse=True)

    # 按需求依次部署，随机选择可行节点
    for svc_id in svc_order:
        svc = services[svc_id]
        remaining = required[svc_id]
        for _ in range(remaining):
            v = _find_feasible_node(rng, n_nodes, cpu_remain, gpu_remain, svc)
            if v < 0:
                break
            deployment.X[v][svc_id] += 1
            cpu_remain[v] -= 1
            gpu_remain[v] -= svc.gpu_per_instance

    return deployment, cpu_remain, gpu_remain


def ffd_deployment(network, services, lambda_s=None, seed=None):
    """
    First Fit Decreasing (FFD) 部署算法：按所需实例数降序排列服务，
    依次将每个服务的实例装入节点（First Fit 策略）。

    算法步骤:
      1. 计算每个服务所需实例数 N_m = ceil(lambda_m / mu_m * margin)
      2. 按 N_m 降序排列所有服务（Decreasing 部分）
      3. First Fit 策略逐节点放入

    参数:
        network:  EdgeNetwork 实例
        services: Service 列表
        lambda_s: 每服务聚合到达率 (可选)
        seed:     随机种子（FFD 为确定性算法，此参数保留接口兼容）

    返回:
        (deployment, cpu_remain, gpu_remain)
    """
    n_nodes = network.n_nodes
    n_services = len(services)
    deployment = Deployment(n_nodes, n_services)

    cpu_remain = network.cpu_capacity.copy()
    gpu_remain = network.gpu_capacity.copy()

    # 计算每个服务所需的实例数（含安全余量）
    N = _required_instances(lambda_s, services)

    # --- 步骤 2: 按 N_m 降序排列服务（Decreasing） ---
    sorted_indices = sorted(range(n_services), key=lambda m: N[m], reverse=True)

    # --- 步骤 3 & 4: First Fit 策略逐服务部署 ---
    for m in sorted_indices:
        svc = services[m]
        remaining = N[m]  # 还需要放置的实例数
        # 保证至少 1 个实例
        remaining = max(remaining, 1)

        # 按节点编号顺序遍历（First Fit）
        for v in range(n_nodes):
            if remaining <= 0:
                break
            # 当前节点最多能放多少个该服务的实例
            fit_by_cpu = int(cpu_remain[v])
            fit_by_gpu = int(gpu_remain[v] // svc.gpu_per_instance)
            can_place = min(fit_by_cpu, fit_by_gpu, remaining)

            if can_place > 0:
                deployment.X[v][m] += can_place
                cpu_remain[v] -= can_place
                gpu_remain[v] -= can_place * svc.gpu_per_instance
                remaining -= can_place

    return deployment, cpu_remain, gpu_remain


def lego_deployment(network, services, lambda_s=None, seed=None):
    """
    LEGO 部署算法：Round-Robin 均衡部署。

    LEGO (Load-balanced Efficient Grid Optimization) 的核心思想是
    将实例均匀分布到各可行节点，使每个节点上的实例数尽量接近，
    配合均匀路由策略实现跨节点负载均衡。

    算法流程:
      1. 计算每个服务所需实例数 N_m = ceil(lambda_m / mu_m * margin)
      2. 按 N_m 降序排列服务
      3. Round-Robin 放置：轮流在可行节点间依次放置实例，
         使得各节点实例数尽可能均衡

    参数:
        network:  EdgeNetwork 实例
        services: Service 列表
        lambda_s: 每服务聚合到达率 (可选)；若为 None 则每服务默认 1 个实例
        seed:     随机种子（保留接口一致性，LEGO 本身是确定性的）
    """
    n_nodes = network.n_nodes
    n_services = len(services)
    deployment = Deployment(n_nodes, n_services)

    cpu_remain = network.cpu_capacity.copy()
    gpu_remain = network.gpu_capacity.copy()

    # ---------- 阶段 1：计算每个服务所需的实例数（含安全余量）----------
    N_m = _required_instances(lambda_s, services)

    # ---------- 阶段 2：按 N_m 降序排列服务 ----------
    svc_order = sorted(range(n_services), key=lambda i: N_m[i], reverse=True)

    # ---------- 阶段 3：Round-Robin 均衡放置 ----------
    for svc_id in svc_order:
        svc = services[svc_id]
        remaining = N_m[svc_id]

        # 收集初始可行节点
        feasible = [v for v in range(n_nodes)
                    if cpu_remain[v] >= 1 and gpu_remain[v] >= svc.gpu_per_instance]
        if not feasible:
            continue

        node_idx = 0
        while remaining > 0:
            v = feasible[node_idx % len(feasible)]

            # 检查当前节点是否仍然可行
            if cpu_remain[v] >= 1 and gpu_remain[v] >= svc.gpu_per_instance:
                deployment.X[v][svc_id] += 1
                cpu_remain[v] -= 1
                gpu_remain[v] -= svc.gpu_per_instance
                remaining -= 1
                node_idx += 1
            else:
                # 移除不可行节点
                feasible.remove(v)
                if not feasible:
                    break

    return deployment, cpu_remain, gpu_remain


def _find_greedy_node(n_nodes, cpu_remain, gpu_remain, svc):
    """
    贪心选择：在满足资源约束的节点中，选择剩余 CPU 最多的节点。
    用于 DRS（确定性路由方案）部署策略。
    返回节点索引，若无可用节点则返回 -1。
    """
    best_v = -1
    best_cpu = -1
    for v in range(n_nodes):
        if cpu_remain[v] >= 1 and gpu_remain[v] >= svc.gpu_per_instance:
            if cpu_remain[v] > best_cpu:
                best_cpu = cpu_remain[v]
                best_v = v
    return best_v


def drs_deployment(network, services, lambda_s=None, seed=None):
    """
    DRS（确定性路由方案）部署：贪心策略，优先将实例放置在剩余资源最多的节点。

    算法流程:
      1. 计算每个服务所需实例数 N_m = ceil(lambda_m / mu_m * margin)
      2. 按所需实例数降序排列服务
      3. 逐个放置实例，贪心选择剩余 CPU 最多的节点

    参数:
        network:  EdgeNetwork 实例
        services: Service 列表
        lambda_s: 每服务聚合到达率 (可选)
        seed:     随机种子（本策略为确定性，保留接口一致性）
    """
    n_nodes = network.n_nodes
    n_services = len(services)
    deployment = Deployment(n_nodes, n_services)

    cpu_remain = network.cpu_capacity.copy()
    gpu_remain = network.gpu_capacity.copy()

    # 计算所需实例数（含安全余量）
    required = _required_instances(lambda_s, services)

    # 按所需实例数降序排列
    svc_order = sorted(range(n_services), key=lambda i: required[i], reverse=True)

    # 贪心放置：每个实例放在剩余 CPU 最多的节点
    for svc_id in svc_order:
        svc = services[svc_id]
        remaining = required[svc_id]
        for _ in range(remaining):
            v = _find_greedy_node(n_nodes, cpu_remain, gpu_remain, svc)
            if v < 0:
                break
            deployment.X[v][svc_id] += 1
            cpu_remain[v] -= 1
            gpu_remain[v] -= svc.gpu_per_instance

    return deployment, cpu_remain, gpu_remain
