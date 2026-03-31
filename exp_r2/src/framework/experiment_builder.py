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
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Generate chain templates and per-chain arrival rates."""

    rng = np.random.default_rng(rng_seed)
    chains: List[Dict[str, Any]] = []
    current_tasks = available_tasks[:num_types]

    rates = rng.dirichlet(np.ones(num_chains)) * total_rate
    rates = np.maximum(1, np.round(rates)).astype(int)
    rates[0] += total_rate - int(np.sum(rates))

    for i in range(num_chains):
        indices = rng.integers(0, len(current_tasks), size=length)
        chain = [current_tasks[idx] for idx in indices]
        chains.append({"chain": chain, "rate": int(rates[i])})

    return chains, current_tasks


def scale_user_chain_rates(user_chains: List[Dict[str, Any]], total_rate: int) -> List[Dict[str, Any]]:
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
) -> ExperimentContext:
    """Construct a complete context object used by all algorithms."""

    user_chains, current_tasks = generate_user_chains(
        available_tasks=tasks_list,
        num_types=num_types,
        length=length,
        total_rate=total_rate,
        num_chains=num_chains,
        rng_seed=rng_seed,
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
