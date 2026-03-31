"""Deployment algorithms: Our method and multiple baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from .domain import ExperimentContext, flatten_matrix, reshape_individual
from .evaluator import evaluate_matrix


class DeploymentAlgorithm(ABC):
    """Base class for deployment algorithms."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def deploy(self, ctx: ExperimentContext, rng: np.random.Generator) -> np.ndarray:
        """Return flattened deployment genes."""


def _safe_version_idx(ctx: ExperimentContext, task: str, idx: int) -> int:
    max_valid = min(ctx.n_versions - 1, len(ctx.tasks_data[task]) - 1)
    return max(0, min(idx, max_valid))


def build_proxy_knowledge(ctx: ExperimentContext) -> Dict[str, Dict[str, int]]:
    """Build per-task version hints used by all heuristics."""

    proxy: Dict[str, Dict[str, int]] = {}
    for task in ctx.current_tasks:
        versions = ctx.tasks_data[task]
        qos_idx = int(np.argmax([v["normalized_qos"] for v in versions]))
        flops_idx = int(np.argmin([v["flops"] for v in versions]))
        params_idx = int(np.argmin([v["model_params"] for v in versions]))
        mid_idx = len(versions) // 2
        proxy[task] = {
            "qos": _safe_version_idx(ctx, task, qos_idx),
            "flops": _safe_version_idx(ctx, task, flops_idx),
            "params": _safe_version_idx(ctx, task, params_idx),
            "mid": _safe_version_idx(ctx, task, mid_idx),
        }
    return proxy


def required_instances(ctx: ExperimentContext, task: str, version_idx: int) -> int:
    """Estimate stable instance count for a task/version pair."""

    lam = float(ctx.lambda_s[task])
    flops = float(ctx.tasks_data[task][version_idx]["flops"])
    mu = ctx.node_flops_capacity / flops if flops > 0 else 1.0
    if mu <= 0:
        return 1
    # Keep utilization below ~0.85.
    est = int(np.ceil(lam / (0.85 * mu))) if lam > 0 else 1
    return max(1, est)


def repair_individual(
    individual: np.ndarray,
    ctx: ExperimentContext,
    proxy_knowledge: Dict[str, Dict[str, int]],
) -> np.ndarray:
    """Strictly enforce node memory limits and avoid service outage."""

    x = reshape_individual(individual, ctx)
    node_usage = np.zeros(ctx.n_nodes, dtype=float)

    for s_idx, task in enumerate(ctx.current_tasks):
        for e in range(ctx.n_nodes):
            for v in range(ctx.n_versions):
                if v < len(ctx.tasks_data[task]) and x[s_idx, e, v] > 0:
                    node_usage[e] += x[s_idx, e, v] * ctx.tasks_data[task][v]["model_params"]

    # Ensure each task has at least one instance.
    for s_idx, task in enumerate(ctx.current_tasks):
        if np.sum(x[s_idx]) == 0:
            min_v = proxy_knowledge[task]["params"]
            req = ctx.tasks_data[task][min_v]["model_params"]
            best_node = int(np.argmin(node_usage))
            x[s_idx, best_node, min_v] += 1
            node_usage[best_node] += req

    # Hard memory cap.
    for e in range(ctx.n_nodes):
        while node_usage[e] > ctx.max_node_params:
            max_p = -1.0
            t_s = -1
            t_v = -1
            for s_idx, task in enumerate(ctx.current_tasks):
                for v in range(ctx.n_versions):
                    if v >= len(ctx.tasks_data[task]):
                        continue
                    if x[s_idx, e, v] > 0:
                        p_size = ctx.tasks_data[task][v]["model_params"]
                        if p_size > max_p:
                            max_p = p_size
                            t_s = s_idx
                            t_v = v

            if t_s == -1:
                break

            task = ctx.current_tasks[t_s]
            min_v = proxy_knowledge[task]["params"]
            min_p = ctx.tasks_data[task][min_v]["model_params"]

            if t_v != min_v and max_p > min_p:
                x[t_s, e, t_v] -= 1
                x[t_s, e, min_v] += 1
                node_usage[e] -= (max_p - min_p)
            else:
                x[t_s, e, t_v] -= 1
                node_usage[e] -= max_p

    return flatten_matrix(x)


class BaselineAlgorithm(DeploymentAlgorithm):
    """Common baseline implementation with different placement policies."""

    def __init__(self, name: str, policy: str):
        super().__init__(name)
        self.policy = policy

    def deploy(self, ctx: ExperimentContext, rng: np.random.Generator) -> np.ndarray:
        proxy = build_proxy_knowledge(ctx)
        x = np.zeros((ctx.n_tasks, ctx.n_nodes, ctx.n_versions), dtype=int)
        node_usage = np.zeros(ctx.n_nodes, dtype=float)

        tasks_sorted = sorted(ctx.current_tasks, key=lambda t: ctx.lambda_s[t], reverse=True)

        # Precompute chain-locality map for CDS/LEGO.
        task_chain_neighbors: Dict[str, set] = {t: set() for t in ctx.current_tasks}
        for uc in ctx.user_chains:
            chain = uc["chain"]
            for i, t in enumerate(chain):
                if i > 0:
                    task_chain_neighbors[t].add(chain[i - 1])
                if i < len(chain) - 1:
                    task_chain_neighbors[t].add(chain[i + 1])

        for task in tasks_sorted:
            s_idx = ctx.current_tasks.index(task)
            v_idx = proxy[task]["mid"]
            model_params = float(ctx.tasks_data[task][v_idx]["model_params"])
            n_inst = required_instances(ctx, task, v_idx)

            for _ in range(n_inst):
                candidates = [
                    e
                    for e in range(ctx.n_nodes)
                    if node_usage[e] + model_params <= ctx.max_node_params
                ]
                if not candidates:
                    break

                chosen = self._select_node(
                    task=task,
                    candidates=candidates,
                    node_usage=node_usage,
                    x=x,
                    ctx=ctx,
                    task_chain_neighbors=task_chain_neighbors,
                    rng=rng,
                )
                x[s_idx, chosen, v_idx] += 1
                node_usage[chosen] += model_params

            # Guarantee at least one instance before final repair.
            if np.sum(x[s_idx]) == 0:
                best_node = int(np.argmin(node_usage))
                x[s_idx, best_node, v_idx] += 1
                node_usage[best_node] += model_params

        if self.policy == "drs":
            self._drs_refine(x, node_usage, ctx, proxy, rng)

        return repair_individual(flatten_matrix(x), ctx, proxy)

    def _select_node(
        self,
        task: str,
        candidates: List[int],
        node_usage: np.ndarray,
        x: np.ndarray,
        ctx: ExperimentContext,
        task_chain_neighbors: Dict[str, set],
        rng: np.random.Generator,
    ) -> int:
        if self.policy == "random":
            return int(rng.choice(candidates))

        if self.policy == "first_fit":
            return int(min(candidates))

        if self.policy == "greedy":
            free_caps = [ctx.max_node_params - node_usage[e] for e in candidates]
            return int(candidates[int(np.argmax(free_caps))])

        if self.policy in {"cds", "lego"}:
            best_node = candidates[0]
            best_score = -1e18
            for e in candidates:
                locality = 0
                for neighbor in task_chain_neighbors.get(task, set()):
                    n_idx = ctx.current_tasks.index(neighbor)
                    if np.sum(x[n_idx, e, :]) > 0:
                        locality += 1

                util = node_usage[e] / ctx.max_node_params
                if self.policy == "cds":
                    score = 1000.0 * locality - util
                else:
                    score = 1000.0 * locality - 2.0 * util

                if score > best_score:
                    best_score = score
                    best_node = e
            return int(best_node)

        # drs initialization defaults to random.
        return int(rng.choice(candidates))

    def _drs_refine(
        self,
        x: np.ndarray,
        node_usage: np.ndarray,
        ctx: ExperimentContext,
        proxy: Dict[str, Dict[str, int]],
        rng: np.random.Generator,
    ) -> None:
        """Simple iterative move refinement for DRS baseline."""

        for _ in range(40):
            utilization = node_usage / ctx.max_node_params
            src = int(np.argmax(utilization))
            dst = int(np.argmin(utilization))
            if utilization[src] - utilization[dst] < 0.05:
                break

            movable = []
            for s_idx, task in enumerate(ctx.current_tasks):
                for v in range(ctx.n_versions):
                    if v >= len(ctx.tasks_data[task]):
                        continue
                    if x[s_idx, src, v] > 0:
                        p = ctx.tasks_data[task][v]["model_params"]
                        if node_usage[dst] + p <= ctx.max_node_params:
                            movable.append((s_idx, v, p))

            if not movable:
                break

            s_idx, v_idx, p_size = movable[int(rng.integers(0, len(movable)))]
            x[s_idx, src, v_idx] -= 1
            x[s_idx, dst, v_idx] += 1
            node_usage[src] -= p_size
            node_usage[dst] += p_size


class OurAlgorithm(DeploymentAlgorithm):
    """Proxy-driven evolutionary deployment algorithm."""

    def __init__(self, generations: int = 40, pop_size: int = 36):
        super().__init__("our")
        self.generations = generations
        self.pop_size = pop_size

    def deploy(self, ctx: ExperimentContext, rng: np.random.Generator) -> np.ndarray:
        proxy = build_proxy_knowledge(ctx)
        pop: List[np.ndarray] = []

        # Seed 1: tiny-model-safe individuals.
        ind_tiny = np.zeros((ctx.n_tasks, ctx.n_nodes, ctx.n_versions), dtype=int)
        for s_idx, task in enumerate(ctx.current_tasks):
            node = int(rng.integers(0, ctx.n_nodes))
            ind_tiny[s_idx, node, proxy[task]["params"]] = 1
        for _ in range(max(1, int(self.pop_size * 0.15))):
            pop.append(repair_individual(flatten_matrix(ind_tiny), ctx, proxy))

        # Seed 2: qos-first individuals.
        ind_qos = np.zeros((ctx.n_tasks, ctx.n_nodes, ctx.n_versions), dtype=int)
        for s_idx, task in enumerate(ctx.current_tasks):
            node = int(rng.integers(0, ctx.n_nodes))
            ind_qos[s_idx, node, proxy[task]["qos"]] = 1
        for _ in range(max(1, int(self.pop_size * 0.15))):
            pop.append(repair_individual(flatten_matrix(ind_qos), ctx, proxy))

        # Seed 3: random individuals.
        while len(pop) < self.pop_size:
            ind = rng.integers(0, 2, size=ctx.genes_len, dtype=int)
            x = reshape_individual(ind, ctx)
            for s_idx, task in enumerate(ctx.current_tasks):
                if np.sum(x[s_idx]) == 0:
                    node = int(rng.integers(0, ctx.n_nodes))
                    v = int(rng.integers(0, min(ctx.n_versions, len(ctx.tasks_data[task]))))
                    x[s_idx, node, v] = 1
            pop.append(repair_individual(flatten_matrix(x), ctx, proxy))

        best_fit = -1e30
        best_individual = pop[0].copy()

        for _ in range(self.generations):
            evaluations = [evaluate_matrix(ind, ctx) for ind in pop]

            for ind, ev in zip(pop, evaluations):
                if ev["fitness"] > best_fit:
                    best_fit = ev["fitness"]
                    best_individual = ind.copy()

            # Tournament selection.
            new_pop: List[np.ndarray] = []
            for _ in range(self.pop_size):
                i1, i2 = rng.choice(self.pop_size, size=2, replace=False)
                winner = pop[int(i1)] if evaluations[int(i1)]["fitness"] > evaluations[int(i2)]["fitness"] else pop[int(i2)]
                new_pop.append(winner.copy())

            # Two-point crossover.
            for i in range(0, self.pop_size - 1, 2):
                if rng.random() < 0.8:
                    pt1 = int(rng.integers(1, max(2, ctx.genes_len // 2)))
                    pt2 = int(rng.integers(max(pt1 + 1, ctx.genes_len // 2), ctx.genes_len))
                    seg1 = new_pop[i][pt1:pt2].copy()
                    seg2 = new_pop[i + 1][pt1:pt2].copy()
                    new_pop[i][pt1:pt2] = seg2
                    new_pop[i + 1][pt1:pt2] = seg1

            # Status-aware mutation.
            for i in range(self.pop_size):
                if rng.random() < 0.4:
                    status = evaluations[i]["status"]
                    x = reshape_individual(new_pop[i], ctx)
                    s_mut = int(rng.integers(0, ctx.n_tasks))
                    task = ctx.current_tasks[s_mut]
                    e_mut = int(rng.integers(0, ctx.n_nodes))

                    if status["congested"] and not status["oom"]:
                        x[s_mut, e_mut, proxy[task]["qos"]] += 1
                    elif status["congested"] and status["oom"]:
                        x[s_mut, e_mut, :] = 0
                        x[s_mut, e_mut, proxy[task]["flops"]] = 1
                    elif status["oom"] and not status["congested"]:
                        x[s_mut, e_mut, :] = 0
                        x[s_mut, e_mut, proxy[task]["params"]] = 1
                    else:
                        if rng.random() < 0.6:
                            x[s_mut, e_mut, proxy[task]["qos"]] += 1
                        else:
                            vmax = min(ctx.n_versions, len(ctx.tasks_data[task]))
                            x[s_mut, e_mut, int(rng.integers(0, vmax))] += 1

                    new_pop[i] = flatten_matrix(x)

                new_pop[i] = repair_individual(new_pop[i], ctx, proxy)

            pop = new_pop

        return best_individual


def create_algorithm(name: str, our_generations: int = 40, our_pop_size: int = 36) -> DeploymentAlgorithm:
    """Factory for all supported algorithms."""

    name = name.lower()
    if name == "our":
        return OurAlgorithm(generations=our_generations, pop_size=our_pop_size)
    if name == "ffd-m":
        return BaselineAlgorithm(name="ffd-m", policy="first_fit")
    if name == "cds-m":
        return BaselineAlgorithm(name="cds-m", policy="cds")
    if name == "random-m":
        return BaselineAlgorithm(name="random-m", policy="random")
    if name == "greedy-m":
        return BaselineAlgorithm(name="greedy-m", policy="greedy")
    if name == "lego":
        return BaselineAlgorithm(name="lego", policy="lego")
    if name == "drs":
        return BaselineAlgorithm(name="drs", policy="drs")

    raise ValueError(f"Unknown algorithm: {name}")
