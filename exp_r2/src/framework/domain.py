"""Core data structures for experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class ExperimentContext:
    """All data needed by deployment algorithms and evaluators."""

    current_tasks: List[str]
    tasks_data: Dict[str, List[Dict[str, Any]]]
    user_chains: List[Dict[str, Any]]
    lambda_s: Dict[str, float]
    total_arrival_rate: float
    n_nodes: int
    n_versions: int
    node_flops_capacity: float
    max_node_params: int
    comm_delay_cross_node: float

    def task_index(self, task: str) -> int:
        return self.current_tasks.index(task)

    @property
    def n_tasks(self) -> int:
        return len(self.current_tasks)

    @property
    def genes_len(self) -> int:
        return self.n_tasks * self.n_nodes * self.n_versions


def reshape_individual(individual: np.ndarray | List[int], ctx: ExperimentContext) -> np.ndarray:
    """Reshape 1D genes to (task, node, version)."""
    return np.asarray(individual, dtype=int).reshape((ctx.n_tasks, ctx.n_nodes, ctx.n_versions))


def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    """Flatten (task, node, version) matrix."""
    return np.asarray(matrix, dtype=int).reshape(-1)
