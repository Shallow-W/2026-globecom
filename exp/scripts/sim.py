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

            # 2. 更新节点到达率（使用配置的ar值，保持恒定）
            # Poisson随机性已经体现在实际请求数上，lambda保持为配置的固定值
            for nid, node in self.topology.nodes.items():
                if nid in requests_by_node:
                    node.lambda_arrival = len(requests_by_node[nid]) / cfg.slot_duration_s
                else:
                    node.lambda_arrival = self.traffic_gen._override_lambda or cfg.lambda_base

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

                for task, _ in reqs:
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

                    # 缓存/拉取处理
                    is_cached = arch['arch_id'] in target_node.cache
                    if not is_cached:
                        self.stats['pull_count'] += 1
                        pull_delay = (cfg.rho_pull * arch['params'] / cfg.B_cloud_mbps * 1000.0
                                      + self.topology.get_delay(target_node.nid, self.topology.cloud_nid))
                        self.stats['pull_delay_ms'] += pull_delay
                        if len(target_node.cache) >= cfg.cache_k:
                            target_node.cache.pop()
                        target_node.cache.add(arch['arch_id'])

                    # 部署状态更新
                    if arch['arch_id'] not in target_node.deployed_archs:
                        target_node.deployed_archs.append(arch['arch_id'])
                        target_node.used_memory_mb += arch['params'] / (1024 * 1024)

                    # 总延迟直接使用 route_delay（已包含 L_topo + T_queue）
                    # 避免重复计算 T_queue
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
