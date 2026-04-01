# ==========================================
# 评估层：染色体编码、延迟计算、QoS、内存利用率
# ==========================================

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import (
    N_NODES, N_VERSIONS, NODE_FLOPS_CAPACITY, MAX_NODE_PARAMS,
    COMM_DELAY_CROSS_NODE, WEIGHT_DELAY, CONGESTION_PENALTY, OCCUPANCY_PENALTY
)
from .data_model import ModelLibrary, ModelVersion


# 默认版本名称映射（与baseline的fixed_version兼容）
DEFAULT_VERSION_NAMES = ["Model-H", "Model-M", "Model-L"]


@dataclass
class ChromosomeEncoder:
    """
    染色体编码器：将部署方案编码为整数矩阵

    X[s, e, v] = 在节点e上使用版本v部署服务s的实例数量
    - s: 服务索引 (0 ~ num_services-1)
    - e: 节点索引 (0 ~ num_nodes-1)
    - v: 版本索引 (0 ~ num_versions-1)，v=1 对应 "Model-M"（baseline使用的版本）
    """
    num_services: int
    num_nodes: int
    num_versions: int
    version_names: List[str] = None  # 版本ID列表

    def __post_init__(self):
        self.shape = (self.num_services, self.num_nodes, self.num_versions)
        if self.version_names is None:
            # 默认：索引1 -> "Model-M"（中间版本，与baseline固定版本对应）
            self.version_names = DEFAULT_VERSION_NAMES[:self.num_versions]

    def _vid_to_idx(self, vid: str) -> int:
        """将version_id转换为索引"""
        if vid in self.version_names:
            return self.version_names.index(vid)
        # fallback: 尝试解析 "v0", "v1" 格式
        if '_' in vid:
            return int(vid.split('_')[1])
        return int(vid[1:]) if vid[1:].isdigit() else 0

    def _idx_to_vid(self, v_idx: int) -> str:
        """将索引转换为version_id"""
        if v_idx < len(self.version_names):
            return self.version_names[v_idx]
        return f"v{v_idx}"

    def encode(self, placement: Dict[Tuple[str, str], Dict[str, int]]) -> np.ndarray:
        """
        将部署方案(placement)编码为染色体矩阵

        Args:
            placement: {(service_id, node_id): {version_id: count}}

        Returns:
            X: np.ndarray of shape (num_services, num_nodes, num_versions)
        """
        X = np.zeros(self.shape, dtype=np.int32)
        for (s, n), versions in placement.items():
            s_idx = int(s.split('_')[1]) if '_' in s else int(s[1:])
            n_idx = int(n.split('_')[1]) if '_' in n else int(n[1:])
            for vid, cnt in versions.items():
                v_idx = self._vid_to_idx(vid)
                if v_idx < self.num_versions:
                    X[s_idx, n_idx, v_idx] = cnt
        return X

    def decode(self, X: np.ndarray, service_ids: List[str], node_ids: List[str]) -> Dict[Tuple[str, str], Dict[str, int]]:
        """
        将染色体矩阵解码为部署方案

        Args:
            X: np.ndarray of shape (num_services, num_nodes, num_versions)
            service_ids: List of service_id strings
            node_ids: List of node_id strings

        Returns:
            placement: {(service_id, node_id): {version_id: count}}
        """
        placement = {}
        for s_idx in range(self.num_services):
            for n_idx in range(self.num_nodes):
                for v_idx in range(self.num_versions):
                    cnt = X[s_idx, n_idx, v_idx]
                    if cnt > 0:
                        s = service_ids[s_idx]
                        n = node_ids[n_idx]
                        v = self._idx_to_vid(v_idx)
                        placement[(s, n)] = placement.get((s, n), {})
                        placement[(s, n)][v] = cnt
        return placement


class RoutingMatrix:
    """
    路由概率矩阵

    p[s, e, v] = 服务s在(node=e, version=v)被选中的概率
              = X[s, e, v] / Σ X[s, e', v']  (服务s在所有节点的实例总数)
    """

    @staticmethod
    def compute(X: np.ndarray) -> np.ndarray:
        """
        从染色体矩阵计算路由概率矩阵

        Args:
            X: np.ndarray of shape (num_services, num_nodes, num_versions)

        Returns:
            p: np.ndarray of same shape, row-stochastic (sum over e,v = 1)
        """
        total_per_service = X.sum(axis=(1, 2), keepdims=True)  # (num_services, 1, 1)
        # 避免除零
        total_per_service = np.maximum(total_per_service, 1e-10)
        p = X / total_per_service
        return p


class DelayCalculator:
    """
    延迟计算器：per-instance M/M/1 排队模型

    公式（与exp_2一致）:
    - rate_per_inst = Lambda_s × p[s,e,v] / X[s,e,v]
    - if rate_per_inst >= mu: delay = 1ms + penalty
    - else: delay = 1 / (mu - rate_per_inst)  (单位: ms)
    - total_delay = Σ (p[s,e,v] × delay[s,e,v]) + comm_delay
    """

    def __init__(self, model_library: ModelLibrary, chain_lambda: float):
        self.model_library = model_library
        self.chain_lambda = chain_lambda  # 该服务链的总到达率

    def calc_service_delay(self, s_idx: int, X: np.ndarray, p: np.ndarray,
                           service_ids: List[str]) -> Tuple[float, float]:
        """
        计算单个服务的延迟

        Args:
            s_idx: 服务索引
            X: 染色体矩阵 (num_services, num_nodes, num_versions)
            p: 路由概率矩阵
            service_ids: 服务ID列表

        Returns:
            (delay, penalty): 该服务的加权延迟和拥塞惩罚
        """
        total_instances = X[s_idx].sum()  # 服务s在所有节点的实例总数

        if total_instances == 0:
            return float('inf'), 0.0

        service_id = service_ids[s_idx]
        versions = self.model_library.get_versions(service_id)

        if len(versions) == 0:
            return float('inf'), 0.0

        total_delay = 0.0
        total_penalty = 0.0

        # 固定mu值（与旧框架services定义一致）: Model-H=20000, Model-M=40000, Model-L=200000
        # 注意：这些是req/s，不是per-ms
        version_mus = [20000.0, 40000.0, 200000.0]

        # 遍历所有(node, version)计算加权延迟
        for e_idx in range(X.shape[1]):
            for v_idx in range(X.shape[2]):
                cnt = X[s_idx, e_idx, v_idx]
                if cnt <= 0 or v_idx >= len(versions):
                    continue

                prob = p[s_idx, e_idx, v_idx]  # 路由概率
                # chain_lambda is per-second; convert to per-ms: divide by 1000
                rate_per_inst = (self.chain_lambda / 1000.0) * prob / cnt  # 每个实例的到达率 (per ms)

                # 使用固定mu值（req/s），转为per-ms用于计算
                mu = version_mus[v_idx] if v_idx < len(version_mus) else version_mus[-1]
                v_mu = mu / 1000.0  # 转为per-ms

                if rate_per_inst >= v_mu:
                    overload = rate_per_inst - v_mu
                    total_penalty += 100.0 * overload
                    delay_node = 1.0  # 保底延迟 1ms
                else:
                    delay_node = 1000.0 / (v_mu - rate_per_inst)

                total_delay += prob * delay_node

        return total_delay, total_penalty

    def calc_chain_delay(self, X: np.ndarray, p: np.ndarray,
                         service_ids: List[str], chain_services: List[str]) -> Dict[str, float]:
        """
        计算整条服务链的延迟

        Args:
            X: 染色体矩阵
            p: 路由概率矩阵
            service_ids: 服务ID列表 (完整列表)
            chain_services: 链实际使用的服务列表

        Returns:
            {
                "queuing": 排队延迟 (ms),
                "communication": 通信延迟 (ms),
                "penalty": 拥塞惩罚,
                "total": 总延迟 (ms)
            }
        """
        total_queuing = 0.0
        total_penalty = 0.0

        # 计算排队延迟：遍历链中每个服务，找到其在service_ids中的索引
        for task_name in chain_services:
            s_idx = service_ids.index(task_name)  # 找到正确的服务索引
            delay, penalty = self.calc_service_delay(s_idx, X, p, service_ids)
            if delay == float('inf'):
                return {
                    "queuing": float('inf'),
                    "communication": 0.0,
                    "penalty": float('inf'),
                    "total": float('inf')
                }
            total_queuing += delay
            total_penalty += penalty

        # 通信延迟：链上相邻服务跨节点才有
        # 按照静态实验的逻辑：计算相邻服务在不同节点上的概率
        comm_delay = 0.0
        for i in range(len(chain_services) - 1):
            # 获取相邻两个服务的索引
            s1_name = chain_services[i]
            s2_name = chain_services[i + 1]
            s1_idx = service_ids.index(s1_name)
            s2_idx = service_ids.index(s2_name)

            # 计算每个服务在各个节点上的边缘概率
            p_node_t1 = np.sum(p[s1_idx, :, :], axis=1)  # shape: (num_nodes,)
            p_node_t2 = np.sum(p[s2_idx, :, :], axis=1)  # shape: (num_nodes,)

            # 跨节点通信概率 = sum over all (e1 != e2) of p_t1[e1] * p_t2[e2]
            for e1 in range(X.shape[1]):
                for e2 in range(X.shape[1]):
                    if e1 != e2:
                        comm_delay += p_node_t1[e1] * p_node_t2[e2] * COMM_DELAY_CROSS_NODE * 1000.0  # 转为ms

        return {
            "queuing": total_queuing,
            "communication": comm_delay,
            "penalty": total_penalty,
            "total": total_queuing + comm_delay
        }


class QoSCalculator:
    """
    QoS计算器：基于路由概率和质量向量的加权QoS

    与exp_2一致：QoS_s = Σ p[s,e,v] × qos[s,v]
    其中 qos[s,v] = normalized_qos of version v
    """

    @staticmethod
    def compute(X: np.ndarray, p: np.ndarray,
                model_library: ModelLibrary,
                service_ids: List[str],
                chain_length: int = None) -> float:
        """
        计算加权QoS（平均值，0~1范围）

        Args:
            X: 染色体矩阵
            p: 路由概率矩阵
            model_library: 模型库
            service_ids: 服务ID列表
            chain_length: 链中实际服务数量（用于限制范围）

        Returns:
            加权QoS平均值 (0~1)
        """
        if chain_length is None:
            chain_length = len(service_ids)

        total_qos = 0.0
        n_deployed = 0

        for s_idx in range(min(chain_length, len(service_ids))):
            service_id = service_ids[s_idx]
            versions = model_library.get_versions(service_id)
            total_inst = X[s_idx].sum()

            if total_inst == 0 or len(versions) == 0:
                continue

            n_deployed += 1
            for e_idx in range(X.shape[1]):
                for v_idx in range(X.shape[2]):
                    cnt = X[s_idx, e_idx, v_idx]
                    if cnt <= 0 or v_idx >= len(versions):
                        continue

                    prob = p[s_idx, e_idx, v_idx]
                    qos = versions[v_idx].normalized_qos
                    total_qos += prob * qos

        if n_deployed == 0:
            return 0.0

        return total_qos / n_deployed  # 平均值，限制在[0,1]


class MemCalculator:
    """
    内存利用率计算器

    mem_utilization = Σ X[s,e,v] × model_params[s,v] / (num_nodes × MAX_NODE_PARAMS)
    """

    @staticmethod
    def compute(X: np.ndarray,
                model_library: ModelLibrary,
                service_ids: List[str],
                num_nodes: int) -> Tuple[float, float]:
        """
        计算内存利用率

        Args:
            X: 染色体矩阵
            model_library: 模型库
            service_ids: 服务ID列表
            num_nodes: 节点数量

        Returns:
            (mem_utilization, overflow_penalty): 内存利用率和溢出惩罚
        """
        total_params = 0.0
        overflow_penalty = 0.0

        for s_idx, service_id in enumerate(service_ids):
            versions = model_library.get_versions(service_id)

            for e_idx in range(X.shape[1]):
                for v_idx in range(X.shape[2]):
                    cnt = X[s_idx, e_idx, v_idx]
                    if cnt <= 0 or v_idx >= len(versions):
                        continue

                    model_params = versions[v_idx].model_params
                    total_params += cnt * model_params

                    # 检查单节点内存约束
                    node_params = 0.0
                    for sv in range(X.shape[0]):
                        for vv in range(X.shape[2]):
                            node_params += X[sv, e_idx, vv] * (
                                versions[min(vv, len(versions)-1)].model_params
                                if vv < len(versions) else 0
                            )

                    if node_params > MAX_NODE_PARAMS:
                        overflow_penalty += OCCUPANCY_PENALTY * (
                            node_params - MAX_NODE_PARAMS
                        )

        total_capacity = num_nodes * MAX_NODE_PARAMS
        mem_util = total_params / total_capacity if total_capacity > 0 else 0.0

        return mem_util, overflow_penalty


class FitnessAggregator:
    """
    适应度聚合器

    fitness = total_qos - WEIGHT_DELAY × total_delay - total_penalty
           = total_qos - 5.0 × total_delay - (congestion_penalty + overflow_penalty)
    """

    def __init__(self, weight_delay: float = WEIGHT_DELAY,
                 congestion_penalty: float = CONGESTION_PENALTY,
                 occupancy_penalty: float = OCCUPANCY_PENALTY):
        self.weight_delay = weight_delay
        self.congestion_penalty = congestion_penalty
        self.occupancy_penalty = occupancy_penalty

    def aggregate(self,
                  qos: float,
                  queuing_delay: float,
                  comm_delay: float,
                  congestion_penalty: float,
                  overflow_penalty: float) -> float:
        """
        计算适应度

        Args:
            qos: 加权QoS
            queuing_delay: 排队延迟 (ms)
            comm_delay: 通信延迟 (ms)
            congestion_penalty: 拥塞惩罚
            overflow_penalty: 内存溢出惩罚

        Returns:
            fitness: 适应度值
        """
        if queuing_delay == float('inf') or comm_delay == float('inf'):
            return -1e10  # 不可行解

        total_delay = (queuing_delay + comm_delay) / 1000.0  # 转为秒
        penalty = congestion_penalty + overflow_penalty

        fitness = qos - self.weight_delay * total_delay - penalty
        return fitness

    def aggregate_chain_list(self,
                              chain_results: List[Dict],
                              total_arrival_rate: float) -> Dict[str, float]:
        """
        聚合多条服务链的结果（到达率加权）

        Args:
            chain_results: List[{
                "qos": float,
                "queuing_delay": float,
                "comm_delay": float,
                "congestion_penalty": float,
                "overflow_penalty": float,
                "arrival_rate": float
            }]
            total_arrival_rate: 所有链的总到达率

        Returns:
            聚合后的指标字典
        """
        weighted_qos = 0.0
        weighted_queuing = 0.0
        weighted_comm = 0.0
        total_penalty = 0.0

        for cr in chain_results:
            weight = cr["arrival_rate"] / total_arrival_rate
            weighted_qos += cr["qos"] * weight
            weighted_queuing += cr["queuing_delay"] * weight
            weighted_comm += cr["comm_delay"] * weight
            total_penalty += cr["congestion_penalty"] * weight + cr["overflow_penalty"] * weight

        fitness = self.aggregate(
            weighted_qos, weighted_queuing, weighted_comm,
            total_penalty, 0.0
        )

        return {
            "fitness": fitness,
            "qos": weighted_qos,
            "queuing_delay": weighted_queuing,
            "comm_delay": weighted_comm,
            "total_penalty": total_penalty,
            "total_delay": (weighted_queuing + weighted_comm) / 1000.0
        }
