"""Data generator for creating test topologies, services, and service chains."""

import random
from typing import Dict, List, Any

from core.topology.node import Node
from core.topology.link import Link
from core.topology.topology import Topology
from core.service.microservice import MicroService, ModelVersion, MODEL_VERSION_CONFIGS
from core.service.chain import ServiceChain


class DataGenerator:
    """测试数据生成器"""

    def __init__(self, seed: int = 42):
        """
        Initialize data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_topology(self, config: Dict[str, Any]) -> Topology:
        """
        Generate network topology with nodes, links, and resources.

        Args:
            config: Configuration dict with keys:
                - num_nodes: Number of nodes (default: 20)
                - cpu_cores_range: [min, max] CPU cores per node
                - gpu_memory_range: [min, max] GPU memory in MB
                - gpu_equipped_nodes_ratio: Ratio of nodes with GPU
                - link_delay_range: [min, max] link delay in ms

        Returns:
            Topology: Generated network topology
        """
        num_nodes = config.get("num_nodes", 20)
        cpu_range = config.get("cpu_cores_range", [8, 64])
        gpu_range = config.get("gpu_memory_range", [0, 16384])
        gpu_ratio = config.get("gpu_equipped_nodes_ratio", 0.5)
        delay_range = config.get("link_delay_range", [0.1, 5.0])

        topo = Topology()

        # Determine which nodes have GPU
        num_gpu_nodes = int(num_nodes * gpu_ratio)
        gpu_node_ids = self.rng.sample(range(num_nodes), num_gpu_nodes) if num_gpu_nodes > 0 else []

        # Add nodes
        for i in range(num_nodes):
            cpu = self.rng.randint(cpu_range[0], cpu_range[1])
            # Assign GPU to some nodes
            if i in gpu_node_ids:
                gpu = self.rng.randint(gpu_range[0] // 1024, gpu_range[1] // 1024) * 1024
            else:
                gpu = 0
            topo.add_node(Node(f"n{i}", cpu, gpu))

        # Add links (full mesh for simplicity)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                delay = self.rng.uniform(delay_range[0], delay_range[1])
                topo.add_link(Link(f"n{i}", f"n{j}", delay))
                topo.add_link(Link(f"n{j}", f"n{i}", delay))

        return topo

    def generate_services(self, config: Dict[str, Any]) -> Dict[str, MicroService]:
        """
        Generate microservice configurations with multi-version models.

        Args:
            config: Configuration dict with keys:
                - num_services: Number of services
                - num_gpu_services: Number of services requiring GPU
                - model_versions: Model version configurations

        Returns:
            Dict[str, MicroService]: Dictionary of services {service_id: MicroService}
        """
        num_services = config.get("num_services", 10)
        num_gpu_services = config.get("num_gpu_services", 3)
        model_versions_cfg = config.get("model_versions", MODEL_VERSION_CONFIGS)

        services = {}
        service_ids = [f"s{i}" for i in range(num_services)]

        # Determine which services require GPU
        gpu_service_ids = set(self.rng.sample(service_ids, min(num_gpu_services, num_services)))

        for service_id in service_ids:
            requires_gpu = service_id in gpu_service_ids
            versions = {}

            for vid, cfg in model_versions_cfg.items():
                mu_range = cfg.get("mu_range", [5, 10])
                mu = self.rng.uniform(mu_range[0], mu_range[1])
                cpu_range = cfg.get("cpu_range", [1, 2])
                cpu = self.rng.randint(cpu_range[0], cpu_range[1])

                # GPU services need GPU memory, others don't
                if requires_gpu:
                    gpu_range = cfg.get("gpu_range", [512, 2048])
                    gpu = self.rng.randint(gpu_range[0], gpu_range[1])
                else:
                    gpu = 0

                versions[vid] = ModelVersion(
                    version_id=vid,
                    mu=mu,
                    accuracy=cfg.get("accuracy", 0.5),
                    cpu_per_instance=cpu,
                    gpu_per_instance=gpu,
                    model_params=cfg.get("params", 0)
                )

            services[service_id] = MicroService(
                service_id=service_id,
                versions=versions,
                requires_gpu=requires_gpu
            )

        return services

    def generate_chains(self, config: Dict[str, Any],
                       services: Dict[str, MicroService]) -> List[ServiceChain]:
        """
        Generate service chains with arrival rates and latency constraints.

        Args:
            config: Configuration dict with keys:
                - num_chains: Number of service chains (default: 4)
                - total_arrival_rate: Total arrival rate λ (sum across all chains)
                - chain_length_range: [min, max] chain length
                - max_latency: Maximum latency constraint in ms
                - task_types: List of task types (default: Excel sheet names)
            services: Dictionary of available services

        Returns:
            List[ServiceChain]: List of generated service chains
        """
        num_chains = config.get("num_chains", 4)
        total_rate = config.get("total_arrival_rate", 200)
        num_task_types = config.get("num_task_types", 10)  # 限制任务池大小
        length_range = config.get("chain_length_range", [2, 5])
        if isinstance(length_range, (int, float)):
            length_range = [length_range, length_range]
        max_latency = config.get("max_latency", 100.0)

        # 任务类型列表 (对应Excel中的sheet名)
        task_types = config.get("task_types",
            ["class_scene", "class_object", "room_layout", "jigsaw",
             "segmentsemantic", "normal", "autoencoder"])

        # Dirichlet 分配到达率（与exp_2一致）
        import numpy as np
        raw_rates = np.random.dirichlet(np.ones(num_chains)) * total_rate
        rates = np.maximum(1, np.round(raw_rates)).astype(int)
        rates[0] += total_rate - np.sum(rates)  # 保证总和精确等于total_rate

        chains = []
        # 只使用前 num_task_types 个微服务（与exp_2的 current_tasks = available_tasks[:num_types] 一致）
        all_service_ids = list(services.keys())
        service_ids = all_service_ids[:num_task_types]

        for i in range(num_chains):
            # Random chain length
            length = self.rng.randint(length_range[0], min(length_range[1], len(service_ids)))

            # Randomly select services WITH replacement (与exp_2的 random.choices 一致)
            chain_services = self.rng.choices(service_ids, k=length)

            # Dirichlet 分配的到达率
            rate = float(rates[i])

            # Random task type
            task_type = self.rng.choice(task_types)

            chains.append(ServiceChain(
                chain_id=f"c{i}",
                services=chain_services,
                arrival_rate=rate,
                max_latency=max_latency,
                task_type=task_type
            ))

        return chains

    def generate_all(self, config: Dict[str, Any]) -> tuple:
        """
        Generate topology, services, and chains together.

        Args:
            config: Configuration dict

        Returns:
            Tuple[Topology, Dict[str, MicroService], List[ServiceChain]]:
                Generated topology, services, and chains
        """
        topology = self.generate_topology(config)
        services = self.generate_services(config)
        chains = self.generate_chains(config, services)
        return topology, services, chains
