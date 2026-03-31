# ==========================================
# 遗传算法层：种群初始化、选择、交叉、变异、修复
# ==========================================

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

from .config import POP_SIZE, GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, N_NODES, N_VERSIONS
from .evaluator import ChromosomeEncoder, RoutingMatrix, DelayCalculator, QoSCalculator, MemCalculator, FitnessAggregator


@dataclass
class Individual:
    """遗传算法个体"""
    X: np.ndarray  # 染色体矩阵 (num_services, num_nodes, num_versions)
    fitness: float
    qos: float = 0.0
    queuing_delay: float = 0.0
    comm_delay: float = 0.0
    mem_util: float = 0.0
    penalty: float = 0.0

    def __lt__(self, other):
        return self.fitness < other.fitness  # 降序排列


class GeneticAlgorithm:
    """
    遗传算法求解服务部署优化问题

    关键设计（与exp_2一致）:
    - 染色体: X[s,e,v] 整数矩阵
    - 初始种群: 贪心 + 随机混合
    - 修复算子: 处理服务宕机和内存溢出
    - 自适应变异: 差值变异 (rand/1/bin)
    - 精英保留: 保留top-k最优个体
    """

    def __init__(self,
                 num_services: int,
                 num_nodes: int = N_NODES,
                 num_versions: int = N_VERSIONS,
                 pop_size: int = POP_SIZE,
                 generations: int = GENERATIONS,
                 mutation_rate: float = MUTATION_RATE,
                 crossover_rate: float = CROSSOVER_RATE,
                 elite_size: int = 2,
                 seed: int = 42):
        self.num_services = num_services
        self.num_nodes = num_nodes
        self.num_versions = num_versions
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.seed = seed

        self.encoder = ChromosomeEncoder(num_services, num_nodes, num_versions)
        self.router = RoutingMatrix()
        self.qos_calc = QoSCalculator()
        self.mem_calc = MemCalculator()
        self.fitness_agg = FitnessAggregator()

        self.rng = np.random.RandomState(seed)
        self.best_individual: Optional[Individual] = None

    def initialize_population(self,
                              model_library,
                              service_ids: List[str],
                              chain_length: int,
                              arrival_rate: float,
                              proxy_knowledge: Dict = None) -> List[Individual]:
        """
        初始化种群：混合贪心和随机

        Args:
            model_library: 模型库
            service_ids: 服务ID列表
            chain_length: 链长度
            arrival_rate: 到达率
            proxy_knowledge: 代理知识 {s_idx: best_v_idx}

        Returns:
            个体列表
        """
        population = []

        # 贪心个体：每个服务选最佳版本，分布在不同节点
        greedy = self._create_greedy_individual(
            model_library, service_ids, chain_length, arrival_rate, proxy_knowledge
        )
        population.append(greedy)

        # 更多贪心变种
        for _ in range(3):
            greedy_var = self._create_greedy_individual(
                model_library, service_ids, chain_length, arrival_rate, proxy_knowledge, variance=True
            )
            population.append(greedy_var)

        # 随机个体填充
        while len(population) < self.pop_size:
            random_ind = self._create_random_individual()
            population.append(random_ind)

        return population[:self.pop_size]

    def _create_greedy_individual(self,
                                  model_library,
                                  service_ids: List[str],
                                  chain_length: int,
                                  arrival_rate: float,
                                  proxy_knowledge: Optional[Dict] = None,
                                  variance: bool = False) -> Individual:
        """创建贪心个体"""
        X = np.zeros((self.num_services, self.num_nodes, self.num_versions), dtype=np.int32)

        for s_idx in range(min(chain_length, len(service_ids))):
            service_id = service_ids[s_idx]
            versions = model_library.get_versions(service_id)

            if len(versions) == 0:
                continue

            # 选择最佳版本（按normalized_qos，选最高的那个映射到num_versions）
            if proxy_knowledge and s_idx in proxy_knowledge:
                best_v = proxy_knowledge[s_idx]
            else:
                # 默认选最高normalized_qos的版本
                best_v = max(range(len(versions)), key=lambda v: versions[v].normalized_qos)

            # 映射到GA的version维度（ clamp到0~num_versions-1）
            best_v = min(best_v, self.num_versions - 1)

            # 选择最少负载的节点
            node_loads = [X[s_idx, e, :].sum() for e in range(self.num_nodes)]
            best_e = min(range(self.num_nodes), key=lambda e: node_loads[e])

            # 部署1个实例
            X[s_idx, best_e, best_v] = 1

        return self._evaluate(X, model_library, service_ids, chain_length, arrival_rate)

    def _create_random_individual(self) -> Individual:
        """创建随机个体"""
        X = np.zeros((self.num_services, self.num_nodes, self.num_versions), dtype=np.int32)

        # 每个服务随机部署1-3个实例到随机节点
        for s_idx in range(self.num_services):
            n_instances = self.rng.randint(1, 4)
            for _ in range(n_instances):
                e = self.rng.randint(0, self.num_nodes)
                v = self.rng.randint(0, self.num_versions)
                X[s_idx, e, v] += 1

        return Individual(X=X, fitness=0.0)

    def _evaluate(self, X: np.ndarray,
                  model_library,
                  service_ids: List[str],
                  chain_length: int,
                  arrival_rate: float) -> Individual:
        """评估一个个体的适应度"""
        p = self.router.compute(X)

        # QoS
        qos = self.qos_calc.compute(X, p, model_library, service_ids, chain_length)

        # 延迟
        delay_calc = DelayCalculator(model_library, arrival_rate)
        delay_result = delay_calc.calc_chain_delay(X, p, service_ids, chain_length)

        # 内存
        mem_util, overflow_penalty = self.mem_calc.compute(X, model_library, service_ids, self.num_nodes)

        # 适应度
        fitness = self.fitness_agg.aggregate(
            qos=qos,
            queuing_delay=delay_result["queuing"],
            comm_delay=delay_result["communication"],
            congestion_penalty=delay_result["penalty"],
            overflow_penalty=overflow_penalty
        )

        return Individual(
            X=X,
            fitness=fitness,
            qos=qos,
            queuing_delay=delay_result["queuing"],
            comm_delay=delay_result["communication"],
            mem_util=mem_util,
            penalty=delay_result["penalty"] + overflow_penalty
        )

    def select(self, population: List[Individual], model_library,
               service_ids: List[str], chain_length: int, arrival_rate: float) -> List[Individual]:
        """
        锦标赛选择
        """
        selected = []
        tournament_size = 3

        for _ in range(self.pop_size):
            # 随机选择tournament_size个个体
            candidates = self.rng.choice(len(population), tournament_size, replace=False)
            best = max(candidates, key=lambda i: population[i].fitness)
            selected.append(population[best])

        return selected

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        两点交叉：对染色体矩阵进行交叉

        Returns:
            (child1, child2)
        """
        if self.rng.rand() > self.crossover_rate:
            return parent1, parent2

        X1, X2 = parent1.X.copy(), parent2.X.copy()

        # 两点交叉：随机选择s维和e维的切点
        s_cut = self.rng.randint(1, self.num_services)
        e_cut = self.rng.randint(1, self.num_nodes)

        # 交换子矩阵
        X1[:s_cut, :e_cut, :], X2[:s_cut, :e_cut, :] = (
            X2[:s_cut, :e_cut, :].copy(), X1[:s_cut, :e_cut, :].copy()
        )

        child1 = Individual(X=X1, fitness=0.0)
        child2 = Individual(X=X2, fitness=0.0)

        return child1, child2

    def mutate(self, individual: Individual,
               population: List[Individual],
               model_library,
               service_ids: List[str],
               chain_length: int,
               arrival_rate: float,
               generation: int) -> Individual:
        """
        自适应变异：差值变异 (rand/1/bin)

        变异策略（与exp_2一致）:
        - 以mutation_rate概率触发
        - 差值变异: X = X_best + F × (X_r1 - X_r2)
        - F = 0.5 × (1 + generation/generations)  随代数增加而减小
        """
        if self.rng.rand() > self.mutation_rate:
            return individual

        X = individual.X.copy()

        # 自适应F
        F = 0.5 * (1.0 + generation / self.generations)

        # rand/1/bin 差值变异
        if len(population) >= 3:  # 需要至少3个个体
            indices = self.rng.choice(len(population), 3, replace=False)
            r1, r2, r3 = [population[i].X for i in indices]

            # 差值变异
            mutant = r3 + F * (r1 - r2)
            mutant = np.maximum(mutant, 0).astype(np.int32)  # 确保非负

            # 二项式交叉
            trial = X.copy()
            j_rand = self.rng.randint(0, self.num_services * self.num_nodes * self.num_versions)
            j = 0
            for s in range(self.num_services):
                for e in range(self.num_nodes):
                    for v in range(self.num_versions):
                        if j == j_rand or self.rng.rand() < 0.5:
                            trial[s, e, v] = mutant[s, e, v]
                        j += 1

            X = trial

        return self._evaluate(X, model_library, service_ids, chain_length, arrival_rate)

    def repair(self, individual: Individual,
               model_library,
               service_ids: List[str]) -> Individual:
        """
        修复算子（与exp_2一致）

        1. 服务宕机修复: 如果某服务没有任何实例，添加1个实例到最少负载节点（最小版本）
        2. 内存溢出修复: 如果某节点内存超限，降级或删除最重实例
        """
        X = individual.X.copy()
        MAX_NODE_PARAMS = 150_000_000

        # 1. 服务宕机修复
        for s_idx in range(min(len(service_ids), self.num_services)):
            total_inst = X[s_idx].sum()
            if total_inst == 0:
                versions = model_library.get_versions(service_ids[s_idx])
                if len(versions) > 0:
                    # 选参数量最少的版本（ clamp到num_versions）
                    min_v = min(range(len(versions)), key=lambda v: versions[v].model_params)
                    min_v = min(min_v, self.num_versions - 1)
                    node_loads = [X[s_idx, e, :].sum() for e in range(self.num_nodes)]
                    min_e = min(range(self.num_nodes), key=lambda e: node_loads[e])
                    X[s_idx, min_e, min_v] = 1

        # 2. 内存溢出修复
        for e_idx in range(self.num_nodes):
            node_params = 0.0
            for s_idx in range(self.num_services):
                for v_idx in range(self.num_versions):
                    cnt = X[s_idx, e_idx, v_idx]
                    if cnt > 0 and s_idx < len(service_ids):
                        versions = model_library.get_versions(service_ids[s_idx])
                        if v_idx < len(versions):
                            node_params += cnt * versions[v_idx].model_params

            if node_params > MAX_NODE_PARAMS:
                # 按参数量从大到小删除实例
                instances = []
                for s_idx in range(self.num_services):
                    for v_idx in range(self.num_versions):
                        cnt = X[s_idx, e_idx, v_idx]
                        if cnt > 0 and s_idx < len(service_ids):
                            versions = model_library.get_versions(service_ids[s_idx])
                            if v_idx < len(versions):
                                instances.append((versions[v_idx].model_params, s_idx, v_idx, cnt))

                instances.sort(reverse=True)

                excess = node_params - MAX_NODE_PARAMS
                for params, s_idx, v_idx, cnt in instances:
                    if excess <= 0:
                        break
                    # 删除该版本的所有实例
                    removed = min(cnt, int(np.ceil(excess / params)) if params > 0 else cnt)
                    X[s_idx, e_idx, v_idx] -= removed
                    excess -= removed * params

        return self._evaluate(X, model_library, service_ids, self.num_services, individual.queuing_delay)

    def run(self,
            model_library,
            service_ids: List[str],
            chain_length: int,
            arrival_rate: float,
            proxy_knowledge: Dict = None) -> Individual:
        """
        运行遗传算法

        Args:
            model_library: 模型库
            service_ids: 服务ID列表
            chain_length: 链长度
            arrival_rate: 到达率
            proxy_knowledge: 代理知识

        Returns:
            最优个体
        """
        # 初始化
        self.population = self.initialize_population(
            model_library, service_ids, chain_length, arrival_rate, proxy_knowledge
        )

        # 评估初始种群
        for i in range(len(self.population)):
            X = self.population[i].X
            p = self.router.compute(X)
            qos = self.qos_calc.compute(X, p, model_library, service_ids, chain_length)
            delay_calc = DelayCalculator(model_library, arrival_rate)
            delay_result = delay_calc.calc_chain_delay(X, p, service_ids, chain_length)
            mem_util, overflow_penalty = self.mem_calc.compute(X, model_library, service_ids, self.num_nodes)
            fitness = self.fitness_agg.aggregate(
                qos, delay_result["queuing"], delay_result["communication"],
                delay_result["penalty"], overflow_penalty
            )
            self.population[i] = Individual(
                X=X, fitness=fitness, qos=qos,
                queuing_delay=delay_result["queuing"],
                comm_delay=delay_result["communication"],
                mem_util=mem_util,
                penalty=delay_result["penalty"] + overflow_penalty
            )

        # 精英个体
        elite = []

        for gen in range(self.generations):
            # 锦标赛选择
            selected = self.select(self.population, model_library, service_ids, chain_length, arrival_rate)

            # 交叉
            offsprings = []
            for i in range(0, len(selected) - 1, 2):
                c1, c2 = self.crossover(selected[i], selected[i + 1])
                offsprings.extend([c1, c2])

            # 变异
            for i in range(len(offsprings)):
                offsprings[i] = self.mutate(
                    offsprings[i], selected, model_library, service_ids, chain_length, arrival_rate, gen
                )

            # 修复
            for i in range(len(offsprings)):
                offsprings[i] = self.repair(offsprings[i], model_library, service_ids)

            # 合并
            self.population.extend(offsprings)

            # 评估
            for i in range(len(self.population)):
                if self.population[i].fitness == 0:  # 未评估
                    X = self.population[i].X
                    p = self.router.compute(X)
                    qos = self.qos_calc.compute(X, p, model_library, service_ids, chain_length)
                    delay_calc = DelayCalculator(model_library, arrival_rate)
                    delay_result = delay_calc.calc_chain_delay(X, p, service_ids, chain_length)
                    mem_util, overflow_penalty = self.mem_calc.compute(X, model_library, service_ids, self.num_nodes)
                    fitness = self.fitness_agg.aggregate(
                        qos, delay_result["queuing"], delay_result["communication"],
                        delay_result["penalty"], overflow_penalty
                    )
                    self.population[i] = Individual(
                        X=X, fitness=fitness, qos=qos,
                        queuing_delay=delay_result["queuing"],
                        comm_delay=delay_result["communication"],
                        mem_util=mem_util,
                        penalty=delay_result["penalty"] + overflow_penalty
                    )

            # 精英保留
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            elite = self.population[:self.elite_size]

            # 截断选择
            self.population = self.population[:self.pop_size]

            # 记录最优
            if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
                self.best_individual = self.population[0]

        return self.best_individual
