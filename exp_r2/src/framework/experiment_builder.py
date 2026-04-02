"""Build experiment contexts from perturbation settings."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np

from .constants import (
    DEFAULT_COMM_DELAY_CROSS_NODE,
    DEFAULT_MAX_NODE_PARAMS,
    DEFAULT_NODE_FLOPS_CAPACITY,
    DEFAULT_NUM_NODES,
    DEFAULT_NUM_VERSIONS,
)
from .domain import ExperimentContext


def generate_user_chains(
    available_tasks: List[str],
    num_types: int,
    length: int,
    total_rate: int,
    num_chains: int = 4,
    rng_seed: int | None = None,
    task_pool_size: int | None = None,
    target_unique_tasks: int | None = None,
    sample_without_replacement_per_chain: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Generate chain templates and per-chain arrival rates."""

    rng = np.random.default_rng(rng_seed)
    chains: List[Dict[str, Any]] = []
    pool_size = int(task_pool_size) if task_pool_size is not None else int(num_types)
    pool_size = max(1, min(pool_size, len(available_tasks)))
    current_tasks = available_tasks[:pool_size]

    rates = rng.dirichlet(np.ones(num_chains)) * total_rate
    rates = np.maximum(1, np.round(rates)).astype(int)
    rates[0] += total_rate - int(np.sum(rates))

    if sample_without_replacement_per_chain:
        if length > len(current_tasks):
            raise ValueError(
                "length cannot exceed task pool size when sampling without replacement"
            )
        for i in range(num_chains):
            indices = rng.choice(len(current_tasks), size=length, replace=False)
            chain = [current_tasks[int(idx)] for idx in indices]
            chains.append({"chain": chain, "rate": int(rates[i])})
    elif target_unique_tasks is not None:
        total_positions = int(num_chains) * int(length)
        unique_count = int(target_unique_tasks)
        unique_count = max(1, min(unique_count, len(current_tasks), total_positions))

        selected_indices = rng.choice(
            len(current_tasks), size=unique_count, replace=False
        )
        selected_tasks = [current_tasks[int(idx)] for idx in selected_indices]

        chain_tasks_flat = selected_tasks.copy()
        if total_positions > unique_count:
            extra_indices = rng.integers(
                0, unique_count, size=total_positions - unique_count
            )
            chain_tasks_flat.extend(selected_tasks[int(idx)] for idx in extra_indices)
        rng.shuffle(chain_tasks_flat)

        for i in range(num_chains):
            start = i * length
            end = start + length
            chains.append({"chain": chain_tasks_flat[start:end], "rate": int(rates[i])})
    else:
        for i in range(num_chains):
            indices = rng.integers(0, len(current_tasks), size=length)
            chain = [current_tasks[idx] for idx in indices]
            chains.append({"chain": chain, "rate": int(rates[i])})

    return chains, current_tasks


def scale_user_chain_rates(
    user_chains: List[Dict[str, Any]], total_rate: int
) -> List[Dict[str, Any]]:
    """Keep chain structure and rescale per-chain rates to a new total rate."""

    if total_rate <= 0:
        raise ValueError("total_rate must be positive")

    chains = deepcopy(user_chains)
    if not chains:
        return chains

    old_rates = np.array([max(0, int(uc.get("rate", 0))) for uc in chains], dtype=float)
    if float(np.sum(old_rates)) <= 0:
        old_rates = np.ones(len(chains), dtype=float)

    scaled = old_rates / float(np.sum(old_rates)) * float(total_rate)
    new_rates = np.maximum(1, np.round(scaled)).astype(int)
    new_rates[0] += int(total_rate) - int(np.sum(new_rates))

    if new_rates[0] < 1:
        deficit = 1 - int(new_rates[0])
        new_rates[0] = 1
        for i in range(1, len(new_rates)):
            if deficit <= 0:
                break
            take = min(deficit, max(0, int(new_rates[i]) - 1))
            new_rates[i] -= take
            deficit -= take

    for i, uc in enumerate(chains):
        uc["rate"] = int(new_rates[i])

    return chains


def build_context(
    tasks_list: List[str],
    tasks_data: Dict[str, List[Dict[str, Any]]],
    num_types: int,
    length: int,
    total_rate: int,
    num_chains: int = 4,
    rng_seed: int | None = None,
    n_nodes: int = DEFAULT_NUM_NODES,
    n_versions: int = DEFAULT_NUM_VERSIONS,
    node_flops_capacity: float = DEFAULT_NODE_FLOPS_CAPACITY,
    max_node_params: int = DEFAULT_MAX_NODE_PARAMS,
    comm_delay_cross_node: float = DEFAULT_COMM_DELAY_CROSS_NODE,
    task_pool_size: int | None = None,
    target_unique_tasks: int | None = None,
    sample_without_replacement_per_chain: bool = False,
) -> ExperimentContext:
    """Construct a complete context object used by all algorithms."""

    user_chains, current_tasks = generate_user_chains(
        available_tasks=tasks_list,
        num_types=num_types,
        length=length,
        total_rate=total_rate,
        num_chains=num_chains,
        rng_seed=rng_seed,
        task_pool_size=task_pool_size,
        target_unique_tasks=target_unique_tasks,
        sample_without_replacement_per_chain=sample_without_replacement_per_chain,
    )

    total_arrival_rate = float(sum(uc["rate"] for uc in user_chains))
    lambda_s = {t: 0.0 for t in current_tasks}
    for uc in user_chains:
        for task in uc["chain"]:
            lambda_s[task] += float(uc["rate"])

    return ExperimentContext(
        current_tasks=current_tasks,
        tasks_data=tasks_data,
        user_chains=user_chains,
        lambda_s=lambda_s,
        total_arrival_rate=total_arrival_rate,
        n_nodes=n_nodes,
        n_versions=n_versions,
        node_flops_capacity=node_flops_capacity,
        max_node_params=max_node_params,
        comm_delay_cross_node=comm_delay_cross_node,
    )


def build_context_from_user_chains(
    tasks_data: Dict[str, List[Dict[str, Any]]],
    user_chains: List[Dict[str, Any]],
    n_nodes: int = DEFAULT_NUM_NODES,
    n_versions: int = DEFAULT_NUM_VERSIONS,
    node_flops_capacity: float = DEFAULT_NODE_FLOPS_CAPACITY,
    max_node_params: int = DEFAULT_MAX_NODE_PARAMS,
    comm_delay_cross_node: float = DEFAULT_COMM_DELAY_CROSS_NODE,
) -> ExperimentContext:
    """Construct context from a predefined set of user chains."""

    chains = deepcopy(user_chains)
    current_tasks = list(dict.fromkeys(task for uc in chains for task in uc["chain"]))

    total_arrival_rate = float(sum(uc["rate"] for uc in chains))
    lambda_s = {t: 0.0 for t in current_tasks}
    for uc in chains:
        for task in uc["chain"]:
            lambda_s[task] += float(uc["rate"])

    return ExperimentContext(
        current_tasks=current_tasks,
        tasks_data=tasks_data,
        user_chains=chains,
        lambda_s=lambda_s,
        total_arrival_rate=total_arrival_rate,
        n_nodes=n_nodes,
        n_versions=n_versions,
        node_flops_capacity=node_flops_capacity,
        max_node_params=max_node_params,
        comm_delay_cross_node=comm_delay_cross_node,
    )
