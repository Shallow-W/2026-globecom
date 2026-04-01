"""Experiment runner for comparison and perturbation workflows."""

from __future__ import annotations

from typing import Any, Dict, List

from .algorithms import create_algorithm
from .evaluator import evaluate_matrix
from .experiment_builder import (
    build_context,
    build_context_from_user_chains,
    generate_user_chains,
    scale_user_chain_rates,
)


class ExperimentRunner:
    """Run algorithm comparison and perturbation experiments."""

    def __init__(self, our_generations: int = 40, our_pop_size: int = 36):
        self.our_generations = our_generations
        self.our_pop_size = our_pop_size

    def run_comparison(
        self,
        ctx,
        algorithms: List[str],
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Run all algorithms on one fixed experiment context."""

        import numpy as np

        results: List[Dict[str, Any]] = []
        base_rng = np.random.default_rng(seed)

        for algo_name in algorithms:
            algo_rng = np.random.default_rng(int(base_rng.integers(0, 1_000_000_000)))
            algo = create_algorithm(
                algo_name,
                our_generations=self.our_generations,
                our_pop_size=self.our_pop_size,
            )
            genes = algo.deploy(ctx, algo_rng)
            metrics = evaluate_matrix(genes, ctx)

            results.append(
                {
                    "Algorithm": algo_name,
                    "Total_Delay_D": metrics["total_delay"],
                    "Avg_QoS_Q": metrics["avg_qos"],
                    "Comp_Delay": metrics["comp_delay"],
                    "Comm_Delay": metrics["comm_delay"],
                    "Mem_Utilization": metrics["mem_utilization"],
                    "Penalty_Score": metrics["penalty"],
                    "Fitness": metrics["fitness"],
                    "Converged_Status": metrics["status"],
                }
            )

        return results

    def run_perturbation(
        self,
        tasks_list,
        tasks_data,
        param_name: str,
        param_values: List[int],
        algorithms: List[str],
        base_num_types: int = 10,
        base_length: int = 4,
        base_rate: int = 200,
        seed: int = 42,
        fixed_arrival_chains: bool = True,
        ntask_mode: str = "unique_in_chains",
        ntask_pool_size: int = 80,
        num_chains: int = 10,
    ) -> List[Dict[str, Any]]:
        """Run one perturbation axis while keeping the others fixed."""

        rows: List[Dict[str, Any]] = []

        base_user_chains = None
        if param_name == "arrival_rate" and fixed_arrival_chains:
            base_user_chains, _ = generate_user_chains(
                available_tasks=tasks_list,
                num_types=base_num_types,
                length=base_length,
                total_rate=base_rate,
                num_chains=num_chains,
                rng_seed=seed,
            )

        for idx, value in enumerate(param_values):
            num_types = base_num_types
            length = base_length
            total_rate = base_rate
            task_pool_size = None
            target_unique_tasks = None
            sample_without_replacement_per_chain = False

            if param_name == "arrival_rate":
                total_rate = int(value)
            elif param_name == "chain_length":
                length = int(value)
            elif param_name == "n_task_types":
                num_types = int(value)
                if ntask_mode == "unique_in_chains":
                    task_pool_size = max(1, int(ntask_pool_size))
                    target_unique_tasks = int(value)
                elif ntask_mode == "unique_chain_exact":
                    task_pool_size = int(value)
                    length = int(value)
                    sample_without_replacement_per_chain = True
            else:
                raise ValueError(f"Unsupported perturbation parameter: {param_name}")

            if param_name == "arrival_rate" and fixed_arrival_chains and base_user_chains is not None:
                user_chains = scale_user_chain_rates(base_user_chains, total_rate=total_rate)
                ctx = build_context_from_user_chains(
                    tasks_data=tasks_data,
                    user_chains=user_chains,
                )
            else:
                ctx = build_context(
                    tasks_list=tasks_list,
                    tasks_data=tasks_data,
                    num_types=num_types,
                    length=length,
                    total_rate=total_rate,
                    num_chains=num_chains,
                    rng_seed=seed + idx,
                    task_pool_size=task_pool_size,
                    target_unique_tasks=target_unique_tasks,
                    sample_without_replacement_per_chain=sample_without_replacement_per_chain,
                )

            comparison = self.run_comparison(ctx, algorithms=algorithms, seed=seed + idx)
            for row in comparison:
                row["Experiment"] = param_name
                row["Variable_Value"] = value
            rows.extend(comparison)

        return rows
