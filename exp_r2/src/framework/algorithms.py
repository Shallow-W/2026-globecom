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
                    node_usage[e] += (
                        x[s_idx, e, v] * ctx.tasks_data[task][v]["model_params"]
                    )

    def _task_totals() -> np.ndarray:
        return np.sum(x, axis=(1, 2)).astype(int)

    def _place_min_instance(task_idx: int) -> bool:
        """Place one minimal-param instance while trying to keep memory feasible."""

        task = ctx.current_tasks[task_idx]
        min_v = proxy_knowledge[task]["params"]
        req = float(ctx.tasks_data[task][min_v]["model_params"])

        free_caps = ctx.max_node_params - node_usage
        candidates = np.where(free_caps >= req)[0]
        if candidates.size > 0:
            local = free_caps[candidates]
            best_node = int(candidates[int(np.argmax(local))])
            x[task_idx, best_node, min_v] += 1
            node_usage[best_node] += req
            return True

        # No immediate room: free memory on the node with the largest remaining room
        # by downgrading/removing redundant instances only.
        best_node = int(np.argmax(free_caps))
        max_rounds = int(np.sum(x[:, best_node, :])) + 1
        for _ in range(max_rounds):
            if ctx.max_node_params - node_usage[best_node] >= req:
                x[task_idx, best_node, min_v] += 1
                node_usage[best_node] += req
                return True

            totals = _task_totals()
            donor = None
            donor_gain = -1.0

            for s2, task2 in enumerate(ctx.current_tasks):
                if totals[s2] <= 1:
                    continue
                for v2 in range(ctx.n_versions):
                    if v2 >= len(ctx.tasks_data[task2]):
                        continue
                    if x[s2, best_node, v2] <= 0:
                        continue

                    p_size = float(ctx.tasks_data[task2][v2]["model_params"])
                    min_v2 = proxy_knowledge[task2]["params"]
                    min_p2 = float(ctx.tasks_data[task2][min_v2]["model_params"])
                    gain = p_size - min_p2 if (v2 != min_v2 and p_size > min_p2) else p_size

                    if gain > donor_gain:
                        donor_gain = gain
                        donor = (s2, v2, min_v2, p_size, min_p2)

            if donor is None:
                break

            s2, v2, min_v2, p_size, min_p2 = donor
            if v2 != min_v2 and p_size > min_p2:
                x[s2, best_node, v2] -= 1
                x[s2, best_node, min_v2] += 1
                node_usage[best_node] -= p_size - min_p2
            else:
                x[s2, best_node, v2] -= 1
                node_usage[best_node] -= p_size

        return False

    # Ensure each task has at least one instance.
    for s_idx, task in enumerate(ctx.current_tasks):
        if np.sum(x[s_idx]) == 0:
            _place_min_instance(s_idx)

    task_totals = _task_totals()

    # Hard memory cap.
    for e in range(ctx.n_nodes):
        while node_usage[e] > ctx.max_node_params:
            max_p = -1.0
            t_s = -1
            t_v = -1

            # Prefer removing/downgrading redundant instances first.
            for only_redundant in (True, False):
                for s_idx, task in enumerate(ctx.current_tasks):
                    if only_redundant and task_totals[s_idx] <= 1:
                        continue
                    for v in range(ctx.n_versions):
                        if v >= len(ctx.tasks_data[task]):
                            continue
                        if x[s_idx, e, v] > 0:
                            p_size = float(ctx.tasks_data[task][v]["model_params"])
                            if p_size > max_p:
                                max_p = p_size
                                t_s = s_idx
                                t_v = v
                if t_s != -1:
                    break

            if t_s == -1:
                break

            task = ctx.current_tasks[t_s]
            min_v = proxy_knowledge[task]["params"]
            min_p = float(ctx.tasks_data[task][min_v]["model_params"])

            if t_v != min_v and max_p > min_p:
                x[t_s, e, t_v] -= 1
                x[t_s, e, min_v] += 1
                node_usage[e] -= max_p - min_p
            else:
                x[t_s, e, t_v] -= 1
                node_usage[e] -= max_p
                task_totals[t_s] = max(0, int(task_totals[t_s]) - 1)

    # Final coverage pass: memory trimming above may drop a task to zero.
    task_totals = _task_totals()
    for s_idx in np.where(task_totals == 0)[0].tolist():
        placed = _place_min_instance(int(s_idx))
        if placed:
            task_totals[s_idx] = 1

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

        tasks_sorted = sorted(
            ctx.current_tasks, key=lambda t: ctx.lambda_s[t], reverse=True
        )

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
            # Intentionally use lightweight low-quality baseline models.
            v_idx = proxy[task]["params"]
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
    """Load-aware dynamic deployment using proxy-guided model switching."""

    def __init__(self, generations: int = 40, pop_size: int = 36):
        super().__init__("our")
        self.generations = generations
        self.pop_size = pop_size
        self.low_load_threshold = 0.70
        self.high_load_threshold = 1.10
        self.max_refine_steps = max(60, generations * 2)
        self.qos_delay_tradeoff = 1.20

    @staticmethod
    def _build_task_neighbors(ctx: ExperimentContext) -> Dict[str, set]:
        neighbors: Dict[str, set] = {t: set() for t in ctx.current_tasks}
        for uc in ctx.user_chains:
            chain = uc["chain"]
            for i, t in enumerate(chain):
                if i > 0:
                    neighbors[t].add(chain[i - 1])
                if i < len(chain) - 1:
                    neighbors[t].add(chain[i + 1])
        return neighbors

    @staticmethod
    def _node_usage_from_matrix(x: np.ndarray, ctx: ExperimentContext) -> np.ndarray:
        usage = np.zeros(ctx.n_nodes, dtype=float)
        for e in range(ctx.n_nodes):
            for s_idx, task in enumerate(ctx.current_tasks):
                for v in range(ctx.n_versions):
                    if v >= len(ctx.tasks_data[task]):
                        continue
                    if x[s_idx, e, v] > 0:
                        usage[e] += (
                            float(x[s_idx, e, v])
                            * float(ctx.tasks_data[task][v]["model_params"])
                        )
        return usage

    @staticmethod
    def _task_pressure(ctx: ExperimentContext, task: str, version_idx: int) -> float:
        flops = float(ctx.tasks_data[task][version_idx]["flops"])
        if flops <= 0:
            return 0.0
        mu = float(ctx.node_flops_capacity) / flops
        if mu <= 0:
            return float("inf")
        lam = float(ctx.lambda_s[task])
        return lam / (float(ctx.n_nodes) * mu)

    def _preferred_versions(
        self,
        ctx: ExperimentContext,
        task: str,
        proxy: Dict[str, Dict[str, int]],
    ) -> List[int]:
        qos_v = int(proxy[task]["qos"])
        mid_v = int(proxy[task]["mid"])
        flops_v = int(proxy[task]["flops"])
        params_v = int(proxy[task]["params"])

        p_qos = self._task_pressure(ctx, task, qos_v)
        if p_qos <= self.low_load_threshold:
            order = [qos_v, mid_v, flops_v, params_v]
        elif p_qos <= self.high_load_threshold:
            order = [qos_v, mid_v, flops_v, params_v]
        else:
            order = [mid_v, flops_v, qos_v, params_v]

        dedup: List[int] = []
        for v in order:
            if v not in dedup:
                dedup.append(v)
        return dedup

    def _is_better(self, candidate: Dict[str, float], current: Dict[str, float]) -> bool:
        eps = 1e-9
        p_new = float(candidate["penalty"])
        p_old = float(current["penalty"])
        if p_new < p_old - eps:
            return True
        if p_new > p_old + eps:
            return False

        q_new = float(candidate["avg_qos"])
        q_old = float(current["avg_qos"])
        d_new = float(candidate["total_delay"])
        d_old = float(current["total_delay"])

        # When both solutions are feasible, prioritize QoS with bounded delay increase.
        if p_new <= eps and p_old <= eps:
            if q_new > q_old + eps:
                base_delay = max(d_old, eps)
                if d_new <= base_delay * self.qos_delay_tradeoff:
                    return True
            if q_new < q_old - eps and d_new >= d_old - eps:
                return False

            if d_new < d_old - eps:
                return True
            if d_new > d_old + eps:
                return False

            if q_new > q_old + eps:
                return True
            if q_new < q_old - eps:
                return False
        else:
            # If both are infeasible with similar penalty, reduce delay first then QoS.
            if d_new < d_old - eps:
                return True
            if d_new > d_old + eps:
                return False

            if q_new > q_old + eps:
                return True
            if q_new < q_old - eps:
                return False

        return float(candidate["fitness"]) > float(current["fitness"]) + eps

    @staticmethod
    def _choose_node(
        task: str,
        candidates: List[int],
        node_usage: np.ndarray,
        x: np.ndarray,
        ctx: ExperimentContext,
        task_chain_neighbors: Dict[str, set],
        task_to_idx: Dict[str, int],
    ) -> int:
        best_node = int(candidates[0])
        best_score = -1e18

        for e in candidates:
            locality = 0
            for neighbor in task_chain_neighbors.get(task, set()):
                n_idx = task_to_idx[neighbor]
                if np.sum(x[n_idx, e, :]) > 0:
                    locality += 1

            util = node_usage[e] / float(ctx.max_node_params)
            free_ratio = (float(ctx.max_node_params) - node_usage[e]) / float(
                ctx.max_node_params
            )
            score = 1000.0 * locality + 2.0 * free_ratio - util
            if score > best_score:
                best_score = score
                best_node = int(e)

        return best_node

    def _try_place_task(
        self,
        task: str,
        task_idx: int,
        version_idx: int,
        x: np.ndarray,
        node_usage: np.ndarray,
        ctx: ExperimentContext,
        task_chain_neighbors: Dict[str, set],
        task_to_idx: Dict[str, int],
    ) -> bool:
        model_params = float(ctx.tasks_data[task][version_idx]["model_params"])
        n_inst = required_instances(ctx, task, version_idx)

        for _ in range(n_inst):
            candidates = [
                e
                for e in range(ctx.n_nodes)
                if node_usage[e] + model_params <= ctx.max_node_params
            ]
            if not candidates:
                return False

            chosen = self._choose_node(
                task=task,
                candidates=candidates,
                node_usage=node_usage,
                x=x,
                ctx=ctx,
                task_chain_neighbors=task_chain_neighbors,
                task_to_idx=task_to_idx,
            )
            x[task_idx, chosen, version_idx] += 1
            node_usage[chosen] += model_params

        return True

    def deploy(self, ctx: ExperimentContext, rng: np.random.Generator) -> np.ndarray:
        proxy = build_proxy_knowledge(ctx)
        task_chain_neighbors = self._build_task_neighbors(ctx)
        task_to_idx = {task: i for i, task in enumerate(ctx.current_tasks)}

        x = np.zeros((ctx.n_tasks, ctx.n_nodes, ctx.n_versions), dtype=int)
        node_usage = np.zeros(ctx.n_nodes, dtype=float)

        tasks_sorted = sorted(
            ctx.current_tasks, key=lambda t: float(ctx.lambda_s[t]), reverse=True
        )

        # Initial dynamic deployment: low-load tasks try high-QoS versions,
        # high-load tasks prefer faster versions.
        for task in tasks_sorted:
            s_idx = task_to_idx[task]
            placed = False

            for v_idx in self._preferred_versions(ctx, task, proxy):
                snapshot_x = x.copy()
                snapshot_usage = node_usage.copy()
                ok = self._try_place_task(
                    task=task,
                    task_idx=s_idx,
                    version_idx=v_idx,
                    x=x,
                    node_usage=node_usage,
                    ctx=ctx,
                    task_chain_neighbors=task_chain_neighbors,
                    task_to_idx=task_to_idx,
                )
                if ok:
                    placed = True
                    break
                x = snapshot_x
                node_usage = snapshot_usage

            if not placed:
                fallback = [
                    proxy[task]["flops"],
                    proxy[task]["params"],
                    proxy[task]["mid"],
                    proxy[task]["qos"],
                ]
                used: List[int] = []
                for v_idx in fallback:
                    if v_idx in used:
                        continue
                    used.append(int(v_idx))

                    model_params = float(ctx.tasks_data[task][v_idx]["model_params"])
                    candidates = [
                        e
                        for e in range(ctx.n_nodes)
                        if node_usage[e] + model_params <= ctx.max_node_params
                    ]
                    if not candidates:
                        continue
                    chosen = self._choose_node(
                        task=task,
                        candidates=candidates,
                        node_usage=node_usage,
                        x=x,
                        ctx=ctx,
                        task_chain_neighbors=task_chain_neighbors,
                        task_to_idx=task_to_idx,
                    )
                    x[s_idx, chosen, int(v_idx)] += 1
                    node_usage[chosen] += model_params
                    placed = True
                    break

            if not placed:
                # Final fallback: force one tiny-model instance, then rely on repair.
                v_idx = int(proxy[task]["params"])
                best_node = int(np.argmax(ctx.max_node_params - node_usage))
                x[s_idx, best_node, v_idx] += 1
                node_usage[best_node] += float(ctx.tasks_data[task][v_idx]["model_params"])

        best_individual = repair_individual(flatten_matrix(x), ctx, proxy)
        best_eval = evaluate_matrix(best_individual, ctx)

        # Feasibility-first local refinement:
        # 1) when congested -> add fast instances for heavy tasks,
        # 2) when stable -> opportunistically upgrade light tasks to QoS versions.
        for _ in range(self.max_refine_steps):
            improved = False

            if best_eval["status"]["congested"]:
                task_order = sorted(
                    ctx.current_tasks,
                    key=lambda t: float(ctx.lambda_s[t]),
                    reverse=True,
                )

                x_cur = reshape_individual(best_individual, ctx).copy()
                node_usage_cur = self._node_usage_from_matrix(x_cur, ctx)

                for task in task_order:
                    s_idx = task_to_idx[task]
                    try_versions: List[int] = []
                    for v_try in [proxy[task]["flops"], proxy[task]["mid"]]:
                        v_try = int(v_try)
                        if v_try not in try_versions:
                            try_versions.append(v_try)

                    for v_add in try_versions:
                        model_params = float(ctx.tasks_data[task][v_add]["model_params"])
                        candidates = [
                            e
                            for e in range(ctx.n_nodes)
                            if node_usage_cur[e] + model_params <= ctx.max_node_params
                        ]
                        if not candidates:
                            continue

                        chosen = self._choose_node(
                            task=task,
                            candidates=candidates,
                            node_usage=node_usage_cur,
                            x=x_cur,
                            ctx=ctx,
                            task_chain_neighbors=task_chain_neighbors,
                            task_to_idx=task_to_idx,
                        )

                        cand_x = x_cur.copy()
                        cand_x[s_idx, chosen, v_add] += 1
                        cand_ind = repair_individual(flatten_matrix(cand_x), ctx, proxy)
                        cand_eval = evaluate_matrix(cand_ind, ctx)

                        if self._is_better(cand_eval, best_eval):
                            best_individual = cand_ind
                            best_eval = cand_eval
                            improved = True
                            break

                    if improved:
                        break

            else:
                task_order = sorted(
                    ctx.current_tasks,
                    key=lambda t: float(ctx.lambda_s[t]),
                )

                for task in task_order:
                    qos_v = int(proxy[task]["qos"])
                    if self._task_pressure(ctx, task, qos_v) > self.high_load_threshold:
                        continue

                    s_idx = task_to_idx[task]
                    x_cur = reshape_individual(best_individual, ctx).copy()

                    best_loc = None
                    best_delta = -1e18
                    qos_n = float(ctx.tasks_data[task][qos_v]["normalized_qos"])
                    for e in range(ctx.n_nodes):
                        for v in range(ctx.n_versions):
                            if v >= len(ctx.tasks_data[task]):
                                continue
                            if v == qos_v or x_cur[s_idx, e, v] <= 0:
                                continue
                            cur_n = float(ctx.tasks_data[task][v]["normalized_qos"])
                            delta = qos_n - cur_n
                            if delta > best_delta:
                                best_delta = delta
                                best_loc = (e, v)

                    if best_loc is None or best_delta <= 0:
                        continue

                    e_chosen, v_old = best_loc
                    x_cur[s_idx, e_chosen, v_old] -= 1
                    x_cur[s_idx, e_chosen, qos_v] += 1

                    cand_ind = repair_individual(flatten_matrix(x_cur), ctx, proxy)
                    cand_eval = evaluate_matrix(cand_ind, ctx)
                    if self._is_better(cand_eval, best_eval):
                        best_individual = cand_ind
                        best_eval = cand_eval
                        improved = True
                        break

            if not improved:
                break

        return best_individual


def create_algorithm(
    name: str, our_generations: int = 40, our_pop_size: int = 36
) -> DeploymentAlgorithm:
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
