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


def random_deployment(network, services, lambda_s=None, seed=None):
    """
    随机部署基线：将微服务实例分配到满足资源约束的节点。

    策略:
      1. 第一轮：保证每个服务至少部署 1 个实例
      2. 第二轮：根据负载按比例分配剩余资源

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

    # 第一轮：保证每个服务至少 1 个实例
    order = rng.permutation(n_services)
    for idx in order:
        svc = services[idx]
        v = _find_feasible_node(rng, n_nodes, cpu_remain, gpu_remain, svc)
        if v >= 0:
            deployment.X[v][svc.id] += 1
            cpu_remain[v] -= 1
            gpu_remain[v] -= svc.gpu_per_instance

    # 第二轮：按负载比例分配剩余资源
    if lambda_s is not None:
        total_lambda = max(np.sum(lambda_s), 1e-9)
        # 按到达率降序排列
        svc_order = sorted(range(n_services), key=lambda i: lambda_s[i], reverse=True)
        # 每个服务可追加的实例数 (与负载成比例，但不超过可用资源)
        for svc_id in svc_order:
            svc = services[svc_id]
            if lambda_s[svc_id] <= 0:
                continue
            # 目标额外实例：按负载比例分配剩余 CPU 的一定比例
            share = lambda_s[svc_id] / total_lambda
            remaining_cpu = max(int(np.sum(cpu_remain)), 0)
            extra_target = max(1, int(remaining_cpu * share))
            for _ in range(extra_target):
                v = _find_feasible_node(rng, n_nodes, cpu_remain, gpu_remain, svc)
                if v < 0:
                    break
                deployment.X[v][svc_id] += 1
                cpu_remain[v] -= 1
                gpu_remain[v] -= svc.gpu_per_instance
    else:
        # 无负载信息时，随机追加 1-2 个实例
        for svc in services:
            for _ in range(rng.integers(0, 3)):
                v = _find_feasible_node(rng, n_nodes, cpu_remain, gpu_remain, svc)
                if v < 0:
                    break
                deployment.X[v][svc.id] += 1
                cpu_remain[v] -= 1
                gpu_remain[v] -= svc.gpu_per_instance

    return deployment, cpu_remain, gpu_remain


def ffd_deployment(network, services, lambda_s=None, seed=None):
    """
    First Fit Decreasing (FFD) 部署算法：按所需实例数降序排列服务，
    依次将每个服务的实例装入节点（First Fit 策略）。

    算法步骤:
      1. 计算每个服务所需的最少实例数 N_m = ceil(lambda_m / mu_m)，
         保证 rho < 1（系统稳定性）。若 lambda_s 为 None 或到达率为 0，
         则默认 N_m = 1。
      2. 按 N_m 降序排列所有服务（Decreasing 部分）。
      3. 对每个服务依次尝试放入节点（First Fit 策略）:
         - 按节点编号 0, 1, 2, ... 顺序遍历
         - 在当前节点尽可能多地放置实例:
           min(剩余CPU, 剩余GPU // gpu_per_instance, 尚需放置数)
         - 当前节点放不下更多时，转向下一个节点
      4. 保证每个服务至少部署 1 个实例（即使计算得到的 N_m 为 0）。

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

    # --- 步骤 1: 计算每个服务所需的实例数 N_m ---
    N = []
    for m in range(n_services):
        svc = services[m]
        if lambda_s is None or lambda_s[m] <= 0:
            # 无负载信息或到达率为 0，默认 1 个实例
            N.append(1)
        else:
            # N_m = ceil(lambda_m / mu_m)，保证 rho = lambda / (N*mu) < 1
            N.append(max(1, ceil(lambda_s[m] / svc.service_rate)))

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
    LEGO 部署算法：三阶段负载均衡部署。

    LEGO (Load-balanced Efficient Grid Optimization) 的核心思想是
    将实例优先放置到剩余资源最多的节点，从而实现跨节点的负载均衡。
    这与 First Fit（首次适应）策略相反。

    三阶段流程:
      1. 实例创建：根据服务到达率计算所需实例数 N_m = ceil(lambda_m / mu_m)
      2. 按需实例数降序排列服务，逐个放置实例
      3. 每个实例放置到当前剩余资源最多的节点（最多剩余 CPU 优先），
         实现负载均衡

    参数:
        network:  EdgeNetwork 实例
        services: Service 列表
        lambda_s: 每服务聚合到达率 (可选)；若为 None 则每服务默认 1 个实例
        seed:     随机种子（保留接口一致性，LEGO 本身是确定性的）
    """
    rng = np.random.default_rng(seed)
    n_nodes = network.n_nodes
    n_services = len(services)
    deployment = Deployment(n_nodes, n_services)

    cpu_remain = network.cpu_capacity.copy()
    gpu_remain = network.gpu_capacity.copy()

    # ---------- 阶段 1：计算每个服务所需的实例数 ----------
    N_m = []
    for svc in services:
        if lambda_s is not None and lambda_s[svc.id] > 0:
            # N_m = ceil(lambda_m / mu_m)，mu_m 取服务处理率的倒数对应单实例吞吐
            instances_needed = int(ceil(lambda_s[svc.id] / svc.service_rate))
        else:
            instances_needed = 1
        N_m.append(instances_needed)

    # ---------- 阶段 2：按 N_m 降序排列服务 ----------
    svc_order = sorted(range(n_services), key=lambda i: N_m[i], reverse=True)

    # ---------- 阶段 3：逐个放置实例，选择剩余资源最多的节点 ----------
    for svc_id in svc_order:
        svc = services[svc_id]
        remaining = N_m[svc_id]

        for _ in range(remaining):
            # 扫描所有节点，找到资源充足且剩余 CPU 最大的节点
            best_node = -1
            best_cpu = -1

            for v in range(n_nodes):
                # 检查资源是否足够放置一个实例
                if cpu_remain[v] >= 1 and gpu_remain[v] >= svc.gpu_per_instance:
                    if cpu_remain[v] > best_cpu:
                        best_cpu = cpu_remain[v]
                        best_node = v

            if best_node < 0:
                # 没有可行节点，停止放置该服务的更多实例
                break

            # 放置实例并更新剩余资源
            deployment.X[best_node][svc_id] += 1
            cpu_remain[best_node] -= 1
            gpu_remain[best_node] -= svc.gpu_per_instance

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
      1. 计算每个服务所需实例数 N_m = ceil(lambda_m / mu_m)
         无到达率信息时默认 1 个实例
      2. 按到达率降序排列服务（无到达率时按服务率升序，慢速优先）
      3. 第一轮：保证每个服务至少 1 个实例，选择剩余 CPU 最多的节点
      4. 第二轮：按负载比例分配额外实例，同样贪心选择剩余 CPU 最多的节点

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

    # 步骤 1：计算每个服务所需的实例数
    if lambda_s is not None:
        required = np.array(
            [max(1, int(ceil(lambda_s[m] / services[m].service_rate)))
             for m in range(n_services)]
        )
    else:
        # 无到达率信息时，默认每服务 1 个实例
        required = np.ones(n_services, dtype=int)

    # 步骤 2：按到达率降序排列；无到达率时按服务率升序（慢速优先）
    if lambda_s is not None:
        svc_order = sorted(range(n_services), key=lambda i: lambda_s[i], reverse=True)
    else:
        svc_order = sorted(range(n_services), key=lambda i: services[i].service_rate)

    # 第一轮：保证每个服务至少 1 个实例
    for idx in svc_order:
        svc = services[idx]
        v = _find_greedy_node(n_nodes, cpu_remain, gpu_remain, svc)
        if v >= 0:
            deployment.X[v][svc.id] += 1
            cpu_remain[v] -= 1
            gpu_remain[v] -= svc.gpu_per_instance

    # 第二轮：按负载比例分配剩余实例
    if lambda_s is not None:
        total_lambda = max(np.sum(lambda_s), 1e-9)
        for svc_id in svc_order:
            svc = services[svc_id]
            if lambda_s[svc_id] <= 0:
                continue
            # 已部署 1 个实例，还需部署 (required - 1) 个
            remaining_instances = max(0, required[svc_id] - 1)
            if remaining_instances == 0:
                continue
            # 按负载比例确定额外可分配的实例数上限
            share = lambda_s[svc_id] / total_lambda
            remaining_cpu = max(int(np.sum(cpu_remain)), 0)
            extra_target = min(remaining_instances, max(1, int(remaining_cpu * share)))
            for _ in range(extra_target):
                v = _find_greedy_node(n_nodes, cpu_remain, gpu_remain, svc)
                if v < 0:
                    break
                deployment.X[v][svc_id] += 1
                cpu_remain[v] -= 1
                gpu_remain[v] -= svc.gpu_per_instance
    else:
        # 无负载信息时，为每个服务在剩余资源最多的节点上追加 1 个实例
        for idx in svc_order:
            svc = services[idx]
            v = _find_greedy_node(n_nodes, cpu_remain, gpu_remain, svc)
            if v >= 0:
                deployment.X[v][svc.id] += 1
                cpu_remain[v] -= 1
                gpu_remain[v] -= svc.gpu_per_instance

    return deployment, cpu_remain, gpu_remain
