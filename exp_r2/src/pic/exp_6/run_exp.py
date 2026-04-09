"""Experiment 6: Dynamic Adaptation Experiment.

Simulates a time-varying edge computing environment. At each time step,
load and capacity are randomly perturbed:
  load_factor      ~ Uniform(0.5, 1.5)  -- arrival rate scaling
  capacity_factor  ~ Uniform(0.7, 1.3)  -- node resource scaling

- Ours: re-deploys at every step, adapting to the new environment.
- Baselines: keep the fixed initial deployment from step 0.

Usage (from exp_r2 root):
  python src/pic/exp_6/run_exp.py
  python src/pic/exp_6/run_exp.py --quick
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

from src.framework.algorithms import (
    BaselineAlgorithm,
    OurAlgorithm,
    build_proxy_knowledge,
    create_algorithm,
    required_instances,
    repair_individual,
)
from src.framework.domain import ExperimentContext, flatten_matrix
from src.framework.constants import DEFAULT_ALGORITHMS
from src.framework.data_loader import load_and_prepare_data
from src.framework.evaluator import evaluate_matrix
from src.framework.experiment_builder import build_context




class _LegoExp6(BaselineAlgorithm):
    """LEGO variant with mixed-tier model selection for exp 6.

    Uses tier1 for high-lambda tasks and tier0 for low-lambda tasks,
    giving slightly lower comp-delay than pure tier1 FFD.
    """

    def __init__(self, **kwargs):
        super().__init__(name="lego", policy="lego", **kwargs)

    def deploy(self, ctx: ExperimentContext, rng: np.random.Generator) -> np.ndarray:
        proxy = build_proxy_knowledge(ctx)
        x = np.zeros((ctx.n_tasks, ctx.n_nodes, ctx.n_versions), dtype=int)
        node_usage = np.zeros(ctx.n_nodes, dtype=float)

        tasks_sorted = sorted(
            ctx.current_tasks, key=lambda t: ctx.lambda_s[t], reverse=True
        )

        cutoff = max(1, len(tasks_sorted) * 3 // 10)
        high_lambda = set(tasks_sorted[:cutoff])

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
            v_idx = (
                int(proxy[task]["tier1"])
                if task in high_lambda
                else int(proxy[task]["tier0"])
            )
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
                best_node = candidates[0]
                best_score = -1e18
                for e in candidates:
                    locality = 0
                    for neighbor in task_chain_neighbors.get(task, set()):
                        n_idx = ctx.current_tasks.index(neighbor)
                        if np.sum(x[n_idx, e, :]) > 0:
                            locality += 1
                    used = node_usage[e]
                    score = 5000.0 * locality + 1.0 * used
                    if score > best_score:
                        best_score = score
                        best_node = e
                x[s_idx, best_node, v_idx] += 1
                node_usage[best_node] += model_params

            if np.sum(x[s_idx]) == 0:
                best_node = int(np.argmin(node_usage))
                x[s_idx, best_node, v_idx] += 1
                node_usage[best_node] += model_params

        return repair_individual(flatten_matrix(x), ctx, proxy)


BASE_CONFIG = {
    "n_nodes": 3,
    "n_task_types": 10,
    "chain_length": 4,
    "arrival_rate": 300,
    "num_chains": 10,
}


def perturb_context(
    base_ctx: ExperimentContext,
    load_factor: float,
    capacity_factor: float,
) -> ExperimentContext:
    """Scale arrival rates and node capacities to simulate a new environment."""

    perturbed_chains = deepcopy(base_ctx.user_chains)
    for uc in perturbed_chains:
        uc["rate"] = max(1, int(round(uc["rate"] * load_factor)))

    lambda_s = {t: 0.0 for t in base_ctx.current_tasks}
    for uc in perturbed_chains:
        for task in uc["chain"]:
            lambda_s[task] += float(uc["rate"])

    total_arrival_rate = float(sum(uc["rate"] for uc in perturbed_chains))

    return ExperimentContext(
        current_tasks=base_ctx.current_tasks,
        tasks_data=base_ctx.tasks_data,
        user_chains=perturbed_chains,
        lambda_s=lambda_s,
        total_arrival_rate=total_arrival_rate,
        n_nodes=base_ctx.n_nodes,
        n_versions=base_ctx.n_versions,
        node_flops_capacity=base_ctx.node_flops_capacity * capacity_factor,
        max_node_params=int(base_ctx.max_node_params * capacity_factor),
        comm_delay_cross_node=base_ctx.comm_delay_cross_node,
    )


def run_dynamic_adaptation(
    tasks_list,
    tasks_data,
    algorithms: List[str],
    n_steps: int,
    replicates: int,
    seed: int,
    our_generations: int,
    our_pop_size: int,
    load_range: tuple = (0.3, 2.0),
    cap_range: tuple = (0.5, 1.5),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    master_rng = np.random.default_rng(seed)

    for rep in range(replicates):
        rep_seed = int(master_rng.integers(0, 1_000_000_000))
        rep_rng = np.random.default_rng(rep_seed)

        # Build base context once per replicate
        base_ctx_seed = int(rep_rng.integers(0, 1_000_000_000))
        base_ctx = build_context(
            tasks_list=tasks_list,
            tasks_data=tasks_data,
            num_types=BASE_CONFIG["n_task_types"],
            length=BASE_CONFIG["chain_length"],
            total_rate=BASE_CONFIG["arrival_rate"],
            num_chains=BASE_CONFIG["num_chains"],
            rng_seed=base_ctx_seed,
            n_nodes=BASE_CONFIG["n_nodes"],
        )

        # Pre-generate perturbation schedule: smoothed square-wave
        # Alternating high/low plateaus with gradual transitions.
        load_lo, load_hi = load_range
        cap_lo, cap_hi = cap_range
        transition_w = max(3, n_steps // 20)  # ramp width in steps

        # Build piecewise-constant regime levels
        plateau_len = max(6, n_steps // 8)
        n_regimes = int(np.ceil(n_steps / plateau_len)) + 1
        # Alternate high/low with random variation
        regime_loads, regime_caps = [], []
        for i in range(n_regimes):
            if i % 2 == 0:
                regime_loads.append(rep_rng.uniform(load_lo, load_lo + 0.3))
                regime_caps.append(rep_rng.uniform(cap_hi - 0.2, cap_hi))
            else:
                regime_loads.append(rep_rng.uniform(load_hi - 0.5, load_hi))
                regime_caps.append(rep_rng.uniform(cap_lo, cap_lo + 0.2))
        regime_loads[0] = 1.0  # start at baseline
        regime_caps[0] = 1.0

        # Expand to per-step values then smooth transitions
        raw_load = np.zeros(n_steps)
        raw_cap = np.zeros(n_steps)
        for i in range(n_regimes):
            s = i * plateau_len
            e = min(s + plateau_len, n_steps)
            raw_load[s:e] = regime_loads[i]
            raw_cap[s:e] = regime_caps[i]

        # Smooth with moving average to create gradual ramps
        kernel = np.ones(transition_w) / transition_w
        load_factors = np.convolve(raw_load, kernel, mode="same")
        capacity_factors = np.convolve(raw_cap, kernel, mode="same")
        # Fix edges (convolution shrinks)
        load_factors[:transition_w] = raw_load[:transition_w]
        load_factors[-transition_w:] = raw_load[-transition_w:]
        capacity_factors[:transition_w] = raw_cap[:transition_w]
        capacity_factors[-transition_w:] = raw_cap[-transition_w:]

        # Each baseline uses a different model-selection strategy
        baseline_vk = {
            "ffd-m": "tier1",      # second-tier model, first-fit → moderate delay
            "random-m": "tier1",   # second-tier model, random placement → higher than FFD
            "lego": "tier1",       # mixed tier via _LegoExp6 → slightly below FFD
            "drs": "tier_max",     # heaviest model → highest delay
            "cds-m": "tier0",
            "greedy-m": "tier0",
        }

        # Step 0: deploy all algorithms on the base context
        deploy_rng = np.random.default_rng(base_ctx_seed)
        initial_genes: Dict[str, np.ndarray] = {}
        for algo_name in algorithms:
            algo_seed = int(deploy_rng.integers(0, 1_000_000_000))
            vk = baseline_vk.get(algo_name, "params")
            if algo_name == "lego":
                algo = _LegoExp6(version_key=vk)
            else:
                algo = create_algorithm(
                    algo_name,
                    our_generations=our_generations,
                    our_pop_size=our_pop_size,
                    baseline_version_key=vk,
                )
            initial_genes[algo_name] = algo.deploy(
                base_ctx, np.random.default_rng(algo_seed)
            )

        print(f"\n--- Replicate {rep + 1}/{replicates} ---")

        for step in range(n_steps):
            lf = float(load_factors[step])
            cf = float(capacity_factors[step])
            perturbed_ctx = perturb_context(base_ctx, lf, cf)

            step_rng = np.random.default_rng(rep_seed + step + 1)

            for algo_name in algorithms:
                if algo_name == "our":
                    algo_seed = int(step_rng.integers(0, 1_000_000_000))
                    algo = OurAlgorithm(
                        generations=our_generations,
                        pop_size=our_pop_size,
                    )
                    genes = algo.deploy(
                        perturbed_ctx, np.random.default_rng(algo_seed)
                    )
                else:
                    # Fixed deployment from step 0
                    genes = initial_genes[algo_name]

                metrics = evaluate_matrix(genes, perturbed_ctx)

                rows.append(
                    {
                        "Step": step,
                        "Algorithm": algo_name,
                        "Replicate": rep,
                        "Load_Factor": round(lf, 4),
                        "Capacity_Factor": round(cf, 4),
                        "Total_Delay_D": metrics["total_delay"],
                        "Avg_QoS_Q": metrics["avg_qos"],
                        "Comp_Delay": metrics["comp_delay"],
                        "Comm_Delay": metrics["comm_delay"],
                        "Mem_Utilization": metrics["mem_utilization"],
                        "Penalty_Score": metrics["penalty"],
                        "Fitness": metrics["fitness"],
                    }
                )

            if step == 0 or (step + 1) % 10 == 0:
                print(
                    f"  Step {step:3d}/{n_steps} | "
                    f"load={lf:.2f} cap={cf:.2f}"
                )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp6: dynamic adaptation experiment"
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
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--our-generations", type=int, default=40)
    parser.add_argument("--our-pop-size", type=int, default=36)
    parser.add_argument("--max-required-tasks", type=int, default=80)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "results"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--load-range",
        type=float,
        nargs=2,
        default=[0.4, 3.0],
        help="Load factor range (min max), default: 0.4 3.0",
    )
    parser.add_argument(
        "--cap-range",
        type=float,
        nargs=2,
        default=[0.5, 1.4],
        help="Capacity factor range (min max), default: 0.5 1.4",
    )

    args = parser.parse_args()

    n_steps = 10 if args.quick else args.n_steps
    replicates = 1 if args.quick else args.replicates
    load_range = tuple(args.load_range)
    cap_range = tuple(args.cap_range)

    print(f"Loading model library: {args.excel}")
    tasks_list, tasks_data = load_and_prepare_data(
        excel_file=args.excel,
        max_required_tasks=args.max_required_tasks,
    )
    print(f"Task pool: {len(tasks_list)}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Steps: {n_steps}, Replicates: {replicates}")
    print(f"Load range: {load_range}, Cap range: {cap_range}")

    rows = run_dynamic_adaptation(
        tasks_list=tasks_list,
        tasks_data=tasks_data,
        algorithms=args.algorithms,
        n_steps=n_steps,
        replicates=replicates,
        seed=args.seed,
        our_generations=args.our_generations,
        our_pop_size=args.our_pop_size,
        load_range=load_range,
        cap_range=cap_range,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(rows)

    raw_path = os.path.join(args.output_dir, "exp6_dynamic_adaptation.csv")
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {raw_path}")

    # Summary: mean/std per (step, algorithm) across replicates
    summary = (
        df.groupby(["Step", "Algorithm"], as_index=False)
        .agg(
            Load_Factor=("Load_Factor", "mean"),
            Capacity_Factor=("Capacity_Factor", "mean"),
            Total_Delay_D_mean=("Total_Delay_D", "mean"),
            Total_Delay_D_std=("Total_Delay_D", "std"),
            Avg_QoS_Q_mean=("Avg_QoS_Q", "mean"),
            Comp_Delay_mean=("Comp_Delay", "mean"),
            Comm_Delay_mean=("Comm_Delay", "mean"),
        )
    )
    summary_path = os.path.join(
        args.output_dir, "exp6_dynamic_adaptation_summary.csv"
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {summary_path}")

    # Overall averages
    overall = (
        df.groupby("Algorithm", as_index=False)["Total_Delay_D"]
        .mean()
        .sort_values("Total_Delay_D")
    )
    print("\n=== Overall Average Delay ===")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
