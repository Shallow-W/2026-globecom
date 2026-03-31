"""Default configuration for experiments."""

DEFAULT_CONFIG = {
    # Topology
    "num_nodes": 20,
    "cpu_cores_range": [8, 64],  # CPU核心数范围
    "gpu_memory_range": [0, 16384],  # GPU内存范围 (MB)
    "gpu_equipped_nodes_ratio": 0.5,  # 配备GPU的节点比例
    "link_delay_range": [0.1, 5.0],  # ms
    # Service (with multi-version models)
    "num_services": 10,
    "num_gpu_services": 3,  # 需要GPU的服务数量
    # 模型版本配置 (参考Excel数据)
    "model_versions": {
        "Model-H": {
            "mu_range": [3, 7],  # 处理速率范围
            "accuracy": 0.62,  # 固定精度
            "cpu_range": [1, 2],
            "gpu_range": [1024, 4096],  # GPU内存需求
        },
        "Model-M": {
            "mu_range": [8, 12],
            "accuracy": 0.53,
            "cpu_range": [1, 2],
            "gpu_range": [512, 2048],
        },
        "Model-L": {
            "mu_range": [15, 25],
            "accuracy": 0.45,
            "cpu_range": [1, 1],
            "gpu_range": [256, 1024],
        },
    },
    # Service chain
    "num_chains": 30,
    "arrival_rate_range": [1, 50],  # 到达率 λ
    "chain_length_range": [2, 5],
    "max_latency": 500.0,  # ms (SLA延迟约束)
    # Experiment
    # Our算法: 动态模型选择
    # Baseline算法: 固定使用 Model-M
    "algorithms": [
        {"name": "ffd-m", "type": "baseline", "fixed_version": "Model-M"},
        {"name": "cds-m", "type": "baseline", "fixed_version": "Model-M"},
        {"name": "random-m", "type": "baseline", "fixed_version": "Model-M"},
        {"name": "greedy-m", "type": "baseline", "fixed_version": "Model-M"},
        # {"name": "our", "type": "dynamic", "load_thresholds": [0.3, 0.7]},
    ],
    # Random seed for reproducibility
    "seed": 42,
}
