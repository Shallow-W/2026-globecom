"""
全局配置参数：微服务路由与部署实验框架
基于 Jackson 排队网络 / M/M/c 模型
"""

# ========== 网络拓扑参数 (异构) ==========
NETWORK_SCALES = {
    "small": {"n_nodes": 3, "cpu_range": (8, 32), "gpu_range": (4096, 16384)},
    "medium": {"n_nodes": 5, "cpu_range": (8, 64), "gpu_range": (4096, 32768)},
    "large": {"n_nodes": 7, "cpu_range": (8, 128), "gpu_range": (4096, 65536)},
}

# 节点间通信时延范围 (秒)
COMM_DELAY_MIN = 0.005
COMM_DELAY_MAX = 0.050

# ========== 微服务参数 ==========
SERVICE_RATE_RANGE = (20.0, 80.0)  # 服务率 mu (请求/秒/核心)
GPU_PER_INSTANCE_RANGE = (128, 1024)  # 每实例 GPU 占用 (MB)

# ========== 实验扫描参数 ==========
ARRIVAL_RATE_SWEEP = [100, 200, 300, 400, 500, 600]
CHAIN_LENGTH_SWEEP = [3, 4, 5, 6, 7, 8]
SERVICE_TYPE_SWEEP = [10, 20, 30, 40, 50, 60]

# 默认实验参数
DEFAULT_TOTAL_RATE = 400
DEFAULT_CHAIN_LENGTH = 6
DEFAULT_N_SERVICE_TYPES = 15
DEFAULT_SCALE = "medium"
DEFAULT_N_CHAINS = 4

# ========== 稳定性惩罚 ==========
# 惩罚公式: PENALTY_FACTOR * (rho - 1 + PENALTY_BASE), 对每个过载的 (服务,节点) 对
INSTABILITY_PENALTY_FACTOR = 2.0  # 惩罚倍数, 越大对过载惩罚越重
INSTABILITY_PENALTY_BASE = 0.1  # 过载基础惩罚 (保证 rho 刚好=1 时也有惩罚)
INSTABILITY_FIXED_DELAY = 1.0  # 过载时的固定时延 (秒)

# ========== 动态实验参数 ==========
DYNAMIC_N_STEPS = 50
DYNAMIC_SCALE_CONFIG = {
    "small": {"chain_length": 3, "n_types": 5, "base_rate": 100},
    "medium": {"chain_length": 5, "n_types": 10, "base_rate": 200},
    "large": {"chain_length": 7, "n_types": 20, "base_rate": 300},
}
