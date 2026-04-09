"""Experiment 5: Resource Constraint Scanning.

Core experiment: Our algorithm adapts model selection to resource constraints
while baselines use a fixed model configuration. At low resources, the
customization advantage is most visible -- ours flexibly fits, baselines are
stuck with a fixed (non-optimal) model.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_DIR)

from src.framework.constants import (
    DEFAULT_ALGORITHMS,
    DEFAULT_MAX_NODE_PARAMS,
    DEFAULT_NODE_FLOPS_CAPACITY,
)
from src.framework.data_loader import load_and_prepare_data
from src.framework.experiment_builder import build_context
from src.framework.domain import ExperimentContext, reshape_individual
from src.framework.algorithms import create_algorithm
from src.framework.evaluator import evaluate_matrix

METRIC_COLUMNS = [
    "Total_Delay_D",
    "Effective_Delay",
    "Avg_QoS_Q",
    "Comp_Delay",
    "Comm_Delay",
    "Mem_Utilization",
    "Penalty_Score",
    "Fitness",
]

RESOURCE_FACTORS = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]

BASE_CONFIG = {
    "n_nodes": 3,
    "n_task_types": 10,
    "chain_length": 4,
    "arrival_rate": 600,
    "num_chains": 10,
}

# Baselines use fixed mid-quality model (simulating one-size-fits-all deployment)
BASELINE_VERSION_KEY = "mid"


def compute_effective_delay(genes: np.ndarray, ctx: ExperimentContext) -> float:
    """Continuous delay that reflects congestion severity.

    Standard evaluator caps congested delay at 1.0 and puts severity into
    penalty.  This function uses a continuous M/M/1-style delay where
    overloaded nodes show high delay proportional to their overload ratio.
    """
    x = reshape_individual(genes, ctx)

    # Build routing probability matrix p[s, e, v]
    p = np.zeros((ctx.n_tasks, ctx.n_nodes, ctx.n_versions), dtype=float)
    for s_idx, task in enumerate(ctx.current_tasks):
        if ctx.lambda_s[task] == 0:
            continue
        task_total = float(np.sum(x[s_idx, :, :]))
        if task_total == 0:
            continue
        for e in range(ctx.n_nodes):
            for v in range(ctx.n_versions):
                if v < len(ctx.tasks_data[task]):
                    p[s_idx, e, v] = float(x[s_idx, e, v]) / task_total

    total_comp = 0.0
    total_comm = 0.0

    for uc in ctx.user_chains:
        chain = uc["chain"]
        w = (
            float(uc["rate"]) / ctx.total_arrival_rate
            if ctx.total_arrival_rate > 0
            else 0.0
        )

        chain_comp = 0.0
        chain_comm = 0.0

        for task in chain:
            s_idx = ctx.current_tasks.index(task)
            expected_delay = 0.0

            for e in range(ctx.n_nodes):
                for v in range(ctx.n_versions):
                    if v >= len(ctx.tasks_data[task]):
                        continue
                    prob = p[s_idx, e, v]
                    if prob <= 0:
                        continue

                    lam = ctx.lambda_s[task] * prob
                    inst = x[s_idx, e, v]
                    if lam > 0 and inst > 0:
                        flops = float(ctx.tasks_data[task][v]["flops"])
                        mu = ctx.node_flops_capacity / flops if flops > 0 else 1.0
                        rate_per_inst = lam / float(inst)

                        if rate_per_inst >= mu:
                            # Overloaded: delay proportional to overload ratio
                            delay_node = rate_per_inst / mu
                        else:
                            # Stable: standard M/M/1 queuing delay
                            delay_node = 1.0 / (mu - rate_per_inst)

                        expected_delay += prob * delay_node

            chain_comp += expected_delay

        for i in range(len(chain) - 1):
            s1 = ctx.current_tasks.index(chain[i])
            s2 = ctx.current_tasks.index(chain[i + 1])
            pn1 = np.sum(p[s1, :, :], axis=1)
            pn2 = np.sum(p[s2, :, :], axis=1)
            for e1 in range(ctx.n_nodes):
                for e2 in range(ctx.n_nodes):
                    if e1 != e2:
                        chain_comm += pn1[e1] * pn2[e2] * ctx.comm_delay_cross_node

        total_comp += chain_comp * w
        total_comm += chain_comm * w

    return total_comp + total_comm


def save_rows(rows, path: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def run_resource_scan(
    tasks_list,
    tasks_data,
    algorithms: List[str],
    factors: List[float],
    replicates: int,
    seed: int,
    our_generations: int,
    our_pop_size: int,
    baseline_version: str = "mid",
) -> list:
    rows = []
    base_rng = np.random.default_rng(seed)

    for fi, factor in enumerate(factors):
        scaled_flops = DEFAULT_NODE_FLOPS_CAPACITY * factor
        scaled_params = int(DEFAULT_MAX_NODE_PARAMS * factor)

        print(
            f"\nresource_factor={factor:.1f}x | "
            f"flops={scaled_flops:.2e} params={scaled_params:,}"
        )

        for rep in range(replicates):
            rep_seed = seed + fi * 10_000 + rep
            ctx = build_context(
                tasks_list=tasks_list,
                tasks_data=tasks_data,
                num_types=BASE_CONFIG["n_task_types"],
                length=BASE_CONFIG["chain_length"],
                total_rate=BASE_CONFIG["arrival_rate"],
                num_chains=BASE_CONFIG["num_chains"],
                rng_seed=rep_seed,
                n_nodes=BASE_CONFIG["n_nodes"],
                node_flops_capacity=scaled_flops,
                max_node_params=scaled_params,
            )

            algo_rng_base = np.random.default_rng(rep_seed)

            for algo_name in algorithms:
                is_ours = algo_name.lower() == "our"
                version_key = "params" if is_ours else baseline_version
                algo = create_algorithm(
                    algo_name,
                    our_generations=our_generations,
                    our_pop_size=our_pop_size,
                    baseline_version_key=version_key,
                )
                algo_rng = np.random.default_rng(
                    int(algo_rng_base.integers(0, 1_000_000_000))
                )
                genes = algo.deploy(ctx, algo_rng)
                metrics = evaluate_matrix(genes, ctx)
                eff_delay = compute_effective_delay(genes, ctx)

                rows.append(
                    {
                        "Algorithm": algo_name,
                        "Total_Delay_D": metrics["total_delay"],
                        "Effective_Delay": eff_delay,
                        "Avg_QoS_Q": metrics["avg_qos"],
                        "Comp_Delay": metrics["comp_delay"],
                        "Comm_Delay": metrics["comm_delay"],
                        "Mem_Utilization": metrics["mem_utilization"],
                        "Penalty_Score": metrics["penalty"],
                        "Fitness": metrics["fitness"],
                        "Converged_Status": metrics["status"],
                        "Experiment": "resource_factor",
                        "Variable_Value": factor,
                        "Resource_Factor": factor,
                        "Replicate": rep,
                        "Replicate_Seed": rep_seed,
                    }
                )

    return rows


def print_scan_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results.")
        return

    summary = (
        df.groupby(["Resource_Factor", "Algorithm"], as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values(["Resource_Factor", "Algorithm"])
    )
    print("\n=== Resource Factor Scan Summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp5: resource constraint scanning"
    )
    parser.add_argument(
        "--excel",
        type=str,
        default=os.path.join(
            PROJECT_DIR, "data", "evaluation_tables_20260325_163931.xlsx"
        ),
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=DEFAULT_ALGORITHMS
    )
    parser.add_argument(
        "--factors",
        type=str,
        default=",".join(str(f) for f in RESOURCE_FACTORS),
        help="Comma-separated resource factors",
    )
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--our-generations", type=int, default=40)
    parser.add_argument("--our-pop-size", type=int, default=36)
    parser.add_argument("--max-required-tasks", type=int, default=80)
    parser.add_argument(
        "--baseline-version",
        type=str,
        default=BASELINE_VERSION_KEY,
        help="Fixed model version for baselines: params/mid/qos/flops",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "results"),
    )
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()

    factors = [float(v.strip()) for v in args.factors.split(",") if v.strip()]
    replicates = 1 if args.quick else args.replicates

    print(f"Loading model library: {args.excel}")
    tasks_list, tasks_data = load_and_prepare_data(
        excel_file=args.excel,
        max_required_tasks=args.max_required_tasks,
    )
    print(f"Task pool: {len(tasks_list)}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Baseline version key: {args.baseline_version}")
    print(f"Factors: {factors}")
    print(f"Replicates: {replicates}")

    rows = run_resource_scan(
        tasks_list=tasks_list,
        tasks_data=tasks_data,
        algorithms=args.algorithms,
        factors=factors,
        replicates=replicates,
        seed=args.seed,
        our_generations=args.our_generations,
        our_pop_size=args.our_pop_size,
        baseline_version=args.baseline_version,
    )

    raw_path = os.path.join(args.output_dir, "exp5_resource_scan.csv")
    df = save_rows(rows, raw_path)

    summary = (
        df.groupby(["Resource_Factor", "Algorithm"], as_index=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns.to_flat_index()
    ]
    summary_path = os.path.join(args.output_dir, "exp5_resource_scan_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")
    print_scan_summary(df)


if __name__ == "__main__":
    main()
