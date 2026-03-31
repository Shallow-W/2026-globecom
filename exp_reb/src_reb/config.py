# ==========================================
# 全局常量与默认配置
# ==========================================

# 边缘环境全局常量（与exp_2一致）
N_NODES = 3                        # 边缘节点数量
N_VERSIONS = 3                     # 每个任务的候选模型版本数（与baseline的Model-H/M/L一致）
NODE_FLOPS_CAPACITY = 200 * 10**9  # 单节点算力上限 200 GFLOPs/s
MAX_NODE_PARAMS = 150_000_000      # 单节点内存容量 150M params
COMM_DELAY_CROSS_NODE = 0.02      # 跨节点通信延迟 0.02s (20ms)

# 适应度权重（与exp_2一致）
WEIGHT_DELAY = 5.0                # 延迟惩罚系数
CONGESTION_PENALTY = 1000.0        # 拥塞惩罚系数（每超1单位+1000）
OCCUPANCY_PENALTY = 1e6            # 内存溢出惩罚（每次溢出+1e6）

# 遗传算法参数
POP_SIZE = 40
GENERATIONS = 50
MUTATION_RATE = 0.4
CROSSOVER_RATE = 0.8

# 扰动实验配置（与exp_2一致）
EXPERIMENTS = {
    "arrival_rate": {
        "param": "total_arrival_rate",
        "values": [100, 200, 300, 400, 500, 600, 700, 800],
        "fixed": {"num_types": 10, "length": 4}
    },
    "chain_length": {
        "param": "length",
        "values": [3, 4, 5, 6, 7, 8, 9, 10],
        "fixed": {"num_types": 10, "total_arrival_rate": 200}
    },
    "num_types": {
        "param": "num_types",
        "values": [10, 20, 30, 40, 50, 60, 70, 80],
        "fixed": {"length": 4, "total_arrival_rate": 200}
    }
}

# 默认实验参数
DEFAULT_CONFIG = {
    "num_nodes": N_NODES,
    "n_versions": N_VERSIONS,
    "num_chains": 4,               # 固定4条链
    "num_task_types": 10,          # 默认任务池大小
    "chain_length": 4,            # 默认链长度
    "total_arrival_rate": 200,     # 默认总到达率
    "generations": GENERATIONS,
    "pop_size": POP_SIZE,
}
