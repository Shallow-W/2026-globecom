"""Baseline deployment algorithms (fixed Model-M)."""

from algorithms.deployment.baselines.ffd_m import FirstFitDecreasingM
from algorithms.deployment.baselines.random_m import RandomDeploymentM
from algorithms.deployment.baselines.greedy_m import SimpleGreedyM
from algorithms.deployment.baselines.cds_m import CoLocatedDeploymentM

__all__ = [
    "FirstFitDecreasingM",
    "RandomDeploymentM",
    "SimpleGreedyM",
    "CoLocatedDeploymentM",
]
