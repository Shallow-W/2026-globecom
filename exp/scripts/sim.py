"""
仿真器模块 - 运行仿真并收集统计指标
"""

from collections import defaultdict
from typing import Dict, Tuple, List
import numpy as np
from cfg import Config
from data import load_architecture_tables
from topo import Topology
from traffic import TrafficGenerator
from algo import DeploymentAlgorithm, RoutingAlgorithm


class Simulator:
    """仿真器：每时隙执行"流量生成 → 候选预计算 → 请求路由 → 统计"流程"""

    def __init__(self, config: Config, topology: Topology,
                 tables: Dict, deploy_class, routing_class,
                 traffic_mode: str = 'steady'):
        self.config = config
        self.topology = topology
        self.tables = tables
        self.tasks = list(tables.keys())
        self.traffic_gen = TrafficGenerator(config, topology, self.tasks)
        self.traffic_mode = traffic_mode

        self.deploy_algo: DeploymentAlgorithm = deploy_class(config)
        self.routing_algo: RoutingAlgorithm = routing_class(config)

        # 统计计数器
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
        }

    def run(self) -> Dict:
        """运行完整仿真"""
        cfg = self.config

        # 初始化部署算法（选择初始模型）
        self.deploy_algo.initialize(self.topology, self.tables, t=0)

        for t in range(cfg.n_slots):
            # 1. 流量生成
            requests_by_node = self.traffic_gen.generate_slot(t, self.traffic_mode)

            # 2. 更新节点的活跃请求跟踪
            #    - 递减所有 active_requests 的剩余时隙
            #    - 移除已完成的请求
            #    - 计算有效到达率（已有请求占用 + 新到达）
            for nid, node in self.topology.nodes.items():
                if node.node_type == 'edge':
                    # 递减已占用请求的剩余时长，移除完成的
                    node.active_requests = [
                        (rem - 1, task) for rem, task in node.active_requests if rem > 1
                    ]
                    # 有效负载 = 已有积压 + 新到达
                    pending = len(node.active_requests)
                    if nid in requests_by_node:
                        new_arrivals = len(requests_by_node[nid])
                        node.lambda_arrival = (pending + new_arrivals) / cfg.slot_duration_s
                    else:
                        node.lambda_arrival = max(0.1, pending / cfg.slot_duration_s)

            # 3. 预计算每节点每任务的候选架构（每时隙只过滤一次，避免重复遍历 4096 行）
            # 注意：对于baseline，已部署的固定模型应始终可用，即使当前负载很高
            slot_candidates: Dict[Tuple[int, str], List] = {}
            for nid, node in self.topology.nodes.items():
                if node.node_type != 'edge':
                    continue
                for task_name in self.tasks:
                    try:
                        cands = self.deploy_algo.filter_candidates(node, task_name, t, self.tables)
                        # 确保已部署的固定模型也在候选列表中（用于baseline的固定模型）
                        deployed = self.deploy_algo.deployed_models.get((nid, task_name))
                        if deployed and deployed not in cands:
                            cands = cands + [deployed]
                        slot_candidates[(nid, task_name)] = cands
                    except Exception:
                        slot_candidates[(nid, task_name)] = []

            # 4. 处理请求
            for src_nid, reqs in requests_by_node.items():
                src_node = self.topology.nodes.get(src_nid)
                if src_node is None:
                    continue

                for task, duration in reqs:
                    self.stats['total_requests'] += 1

                    # 路由决策
                    target_node, arch, route_delay = self.routing_algo.route(
                        task, src_node, self.topology, self.tables,
                        self.deploy_algo, t, self.traffic_gen, slot_candidates
                    )

                    if target_node is None or arch is None:
                        self.stats['failed_requests'] += 1
                        # 失败请求计入5s超时惩罚时延
                        TIMEOUT_MS = 5000.0
                        self.stats['total_latency_ms'] += TIMEOUT_MS
                        self.stats['latencies'].append(TIMEOUT_MS)
                        continue

                    # 缓存/拉取处理（LRU策略）
                    params_mb = arch['params'] / (1024 * 1024)
                    is_cached = arch['arch_id'] in target_node.cache
                    if not is_cached:
                        self.stats['pull_count'] += 1
                        pull_delay = (cfg.rho_pull * arch['params'] / cfg.B_cloud_mbps * 1000.0
                                      + self.topology.get_delay(target_node.nid, self.topology.cloud_nid))
                        self.stats['pull_delay_ms'] += pull_delay
                        if len(target_node.cache) >= cfg.cache_k:
                            evicted_id, evicted_params = target_node.cache.popitem(last=False)
                            evicted_params_mb = evicted_params / (1024 * 1024)
                            target_node.used_memory_mb = max(0.0, target_node.used_memory_mb - evicted_params_mb)
                        # LRU: 新访问移到末尾（最新使用）
                        target_node.cache[arch['arch_id']] = arch['params']
                        target_node.used_memory_mb += params_mb
                    else:
                        # 已缓存：移到末尾（最新使用）
                        target_node.cache.move_to_end(arch['arch_id'])

                    # 部署状态更新（避免重复计入）
                    if arch['arch_id'] not in target_node.deployed_archs:
                        target_node.deployed_archs.append(arch['arch_id'])

                    # 请求占用该节点 N 个时隙（request_length 影响的时长）
                    target_node.active_requests.append((duration, task))

                    # 总延迟 = 路由返回的 L_topo + T_queue（路由决策时已算好）
                    total_latency = route_delay

                    # SLA 检查
                    T_SLA = cfg.T_SLA_jigsaw_ms if task == 'jigsaw' else cfg.T_SLA_ms
                    if total_latency > T_SLA:
                        self.stats['sla_violations'] += 1
                        target_node.violated_count += 1
                    else:
                        self.stats['success_requests'] += 1

                    # 统计
                    self.stats['total_latency_ms'] += total_latency
                    self.stats['latencies'].append(total_latency)
                    target_node.served_count += 1
                    target_node.total_latency_ms += total_latency
                    self.stats['per_task_perf'][task].append(arch['task_perf'])

        return self._collect_results()

    def _collect_results(self) -> Dict:
        """汇总指标"""
        s = self.stats
        n = self.config.n_slots

        total = max(s['total_requests'], 1)
        avg_lat = s['total_latency_ms'] / total

        # 时延 CDF
        lats = sorted(s['latencies']) if s['latencies'] else [0]
        cdf_x = np.linspace(0, 1, len(lats) + 1)
        cdf_y = [0.0] + lats

        # 分任务性能
        task_perf = {task: np.mean(perfs) if perfs else 0
                      for task, perfs in s['per_task_perf'].items()}

        # 节点利用率
        utils = []
        for node in self.topology.nodes.values():
            if node.node_type == 'edge':
                u = node.used_memory_mb / node.memory_mb if node.memory_mb > 0 else 0
                utils.append(u)

        return {
            'avg_latency_ms': avg_lat,
            'success_rate': s['success_requests'] / total,
            'sla_violation_rate': s['sla_violations'] / total,
            'throughput': s['success_requests'] / n,
            'avg_pull_delay_ms': s['pull_delay_ms'] / max(s['pull_count'], 1),
            'pull_count': s['pull_count'],
            'node_utilization_avg': np.mean(utils) if utils else 0,
            'node_utilization_std': np.std(utils) if utils else 0,
            'task_performance': task_perf,
            'latencies': s['latencies'],
            'latencies_cdf_x': cdf_x.tolist(),
            'latencies_cdf_y': cdf_y,
            'slot_throughput': [],  # 可扩展
            'total_requests': s['total_requests'],
        }
