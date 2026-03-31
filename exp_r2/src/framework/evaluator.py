"""Unified evaluator used by all deployment algorithms."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .domain import ExperimentContext, reshape_individual


def evaluate_matrix(individual: np.ndarray, ctx: ExperimentContext) -> Dict[str, Any]:
    """Evaluate one deployment matrix with the same metric structure as the static main."""

    x = reshape_individual(individual, ctx)

    total_qos = 0.0
    penalty_params = 0.0
    penalty_delay = 0.0

    total_comm_delay = 0.0
    total_comp_delay = 0.0
    total_params_used = 0.0

    # Node memory budget check.
    for e in range(ctx.n_nodes):
        node_params_used = 0.0
        for s_idx, task in enumerate(ctx.current_tasks):
            for v in range(ctx.n_versions):
                if v < len(ctx.tasks_data[task]):
                    node_params_used += (
                        float(x[s_idx, e, v]) * ctx.tasks_data[task][v]["model_params"]
                    )
        total_params_used += node_params_used
        if node_params_used > ctx.max_node_params:
            penalty_params += 1e6

    mem_utilization = total_params_used / (ctx.n_nodes * ctx.max_node_params)

    p = np.zeros((ctx.n_tasks, ctx.n_nodes, ctx.n_versions), dtype=float)

    for s_idx, task in enumerate(ctx.current_tasks):
        if ctx.lambda_s[task] == 0:
            continue
        task_total_instances = float(np.sum(x[s_idx, :, :]))
        if task_total_instances == 0:
            penalty_delay += 1000.0
            continue
        for e in range(ctx.n_nodes):
            for v in range(ctx.n_versions):
                if v >= len(ctx.tasks_data[task]):
                    continue
                p[s_idx, e, v] = float(x[s_idx, e, v]) / task_total_instances

    for uc in ctx.user_chains:
        chain = uc["chain"]
        weight = float(uc["rate"]) / ctx.total_arrival_rate if ctx.total_arrival_rate > 0 else 0.0

        chain_qos = 0.0
        chain_comp = 0.0
        chain_comm = 0.0

        for task in chain:
            s_idx = ctx.current_tasks.index(task)
            expected_qos = 0.0
            expected_task_delay = 0.0

            for e in range(ctx.n_nodes):
                for v in range(ctx.n_versions):
                    if v >= len(ctx.tasks_data[task]):
                        continue

                    prob = p[s_idx, e, v]
                    if prob <= 0:
                        continue

                    expected_qos += prob * ctx.tasks_data[task][v]["normalized_qos"]
                    lam = ctx.lambda_s[task] * prob
                    inst = x[s_idx, e, v]
                    if lam > 0 and inst > 0:
                        mu = ctx.node_flops_capacity / ctx.tasks_data[task][v]["flops"]
                        rate_per_inst = lam / float(inst)
                        if rate_per_inst >= mu:
                            penalty_delay += 1000.0 * (float(rate_per_inst) - float(mu) + 1.0)
                            delay_node = 1.0
                        else:
                            delay_node = 1.0 / (mu - rate_per_inst)
                        expected_task_delay += prob * delay_node

            chain_qos += expected_qos
            chain_comp += expected_task_delay

        for i in range(len(chain) - 1):
            s1_idx = ctx.current_tasks.index(chain[i])
            s2_idx = ctx.current_tasks.index(chain[i + 1])
            p_node_t1 = np.sum(p[s1_idx, :, :], axis=1)
            p_node_t2 = np.sum(p[s2_idx, :, :], axis=1)
            for e1 in range(ctx.n_nodes):
                for e2 in range(ctx.n_nodes):
                    if e1 != e2:
                        chain_comm += (
                            p_node_t1[e1] * p_node_t2[e2] * ctx.comm_delay_cross_node
                        )

        total_qos += chain_qos * weight
        total_comp_delay += chain_comp * weight
        total_comm_delay += chain_comm * weight

    total_delay = total_comp_delay + total_comm_delay
    total_penalty = penalty_params + penalty_delay
    fitness = total_qos - 5.0 * total_delay - total_penalty

    return {
        "fitness": fitness,
        "total_delay": total_delay,
        "avg_qos": total_qos,
        "penalty": total_penalty,
        "comp_delay": total_comp_delay,
        "comm_delay": total_comm_delay,
        "mem_utilization": mem_utilization,
        "status": {
            "congested": penalty_delay > 0,
            "oom": penalty_params > 0,
        },
    }
