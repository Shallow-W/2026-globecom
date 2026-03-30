"""
配置模块 - 所有可调参数集中管理
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Config:
    """仿真全局配置"""

    # 基础路径
    data_path: str = "D:/Item/lab/2026globcom/exp/data/evaluation_tables_20260325_163931.xlsx"
    output_dir: str = "D:/Item/lab/2026globcom/exp/results/"

    # 拓扑参数（对齐论文 A: 15节点小拓扑 / 论文 B: 65节点大拓扑 / 新增30节点中拓扑）
    n_small: int = 15
    n_medium: int = 30
    n_large: int = 65

    # 节点异构资源（对齐论文 A: 8-64 核 → GFLOPS 和内存）
    # 格式: (节点数量, 内存MB, 峰值算力GFLOPS)
    node_types: List[Tuple[int, int, float]] = None

    # 中拓扑节点配置（统一中型）
    node_types_medium: List[Tuple[int, int, float]] = None

    # SLA 参数（对齐论文 A/B）
    T_SLA_ms: float = 5000.0  # 通用 SLA
    T_SLA_jigsaw_ms: float = 5000.0  # jigsaw 严格 SLA

    # 流量参数
    # 注意: λ 应保证系统稳定，即 λ < min(μ) 对所有节点-架构组合
    # 小型节点 10GF + 轻量架构 μ≈100 req/s → 稳定 λ < 1 req/s
    # 小型节点 10GF + 中等架构 μ≈4 req/s → 稳定 λ < 2 req/s
    # 大型节点 100GF + 中等架构 μ≈40 req/s → 稳定 λ < 10 req/s
    lambda_base: float = 3.0  # 基准到达率 req/s（每节点，每时隙 = Poisson(λ_base)）
    lambda_th_percentile: float = 0.7  # λ_th 分位点（用于动态权重触发）

    # 单变量扰动参数（静态部署实验）
    request_length_base: int = 5  # 基准请求服务时长（slots）
    n_task_types: int = 7  # 激活的任务类型数（1-7）

    # 云边参数
    B_cloud_mbps: float = 100.0  # 云边带宽 Mbps
    L_cloud_ms: float = 20.0  # 云边时延 ms
    rho_pull: float = 1e-6  # 参数量→传输时间系数 (params/B_cloud * 1000)

    # 仿真参数
    slot_duration_s: float = 1.0  # 时隙长度（秒）
    n_slots: int = 100  # 默认仿真时隙数
    cache_k: int = 10  # 每节点缓存架构数
    seed: int = 42  # 随机种子

    # OURS 权重参数
    alpha: float = 1.0  # 延迟惩罚系数（动态权重公式）
    beta: float = 1.0  # 内存惩罚系数
    theta1: float = 0.5  # 精度权重（路由效用）
    theta2: float = 0.35  # 延迟权重（路由效用）
    theta3: float = 0.15  # 内存权重（路由效用）

    def __post_init__(self):
        if self.node_types is None:
            # 小型5个(2GB,10GF), 中型7个(4GB,30GF), 大型3个(8GB,100GF)
            self.node_types = [
                (5, 2048, 10.0),
                (7, 4096, 30.0),
                (3, 8192, 100.0),
            ]
        if self.node_types_medium is None:
            # 中拓扑：统一中型节点 4GB, 30GF
            self.node_types_medium = [
                (self.n_medium, 4096, 30.0),
            ]
