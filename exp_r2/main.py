"""CLI entry for the modular deployment-routing experiment framework."""

from __future__ import annotations

import argparse
import os
from typing import List, Set

import pandas as pd


"""

ar: python .\main.py --mode single --perturb arrival_rate --values 200,300,400,500,600,700,800 --replicates 10 --replicate-targets arrival_rate --arrival-chain-mode fixed --arrival-base-ntask 7 --arrival-base-chainlen 5 
ntask: python .\main.py --mode single --perturb n_task_types --values 3,4,5,6,7,8,9,10 --replicates 10 --ntask-mode unique_chain_exact --ntask-base-rate 400 --num-chains 10
chainlen: python .\main.py --mode single --perturb chain_length --values 3,4,5,6,7,8 --replicates 5 --chain-base-rate 350 --num-chains 10

"""
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from src.framework.constants import (
    DEFAULT_ALGORITHMS,
    DEFAULT_LENGTH_VALUES,
    DEFAULT_RATE_VALUES,
    DEFAULT_TASK_TYPE_VALUES,
)
from src.framework.data_loader import load_and_prepare_data
from src.framework.runner import ExperimentRunner


METRIC_COLUMNS = [
    "Total_Delay_D",
    "Avg_QoS_Q",
    "Comp_Delay",
    "Comm_Delay",
    "Mem_Utilization",
    "Penalty_Score",
    "Fitness",
]


def parse_int_list(raw: str) -> List[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def parse_str_set(raw: str) -> Set[str]:
    return {v.strip() for v in raw.split(",") if v.strip()}


def save_rows(rows, path: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def print_brief_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results.")
        return

    summary = (
        df.groupby(["Experiment", "Algorithm"], as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values(["Experiment", "Algorithm"])
    )

    print("\n=== Overall Average Summary (Experiment + Algorithm) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _display_value(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def print_round_details(df: pd.DataFrame) -> None:
    """Print per-round results for each perturbation value."""

    if df.empty:
        return

    experiments = sorted(df["Experiment"].unique().tolist())
    for exp in experiments:
        exp_df = df[df["Experiment"] == exp]
        values = sorted(exp_df["Variable_Value"].unique().tolist())

        print(f"\n=== Per-round Results: {exp} ===")
        for value in values:
            value_df = exp_df[exp_df["Variable_Value"] == value]
            round_df = (
                value_df.groupby("Algorithm", as_index=False)[METRIC_COLUMNS]
                .mean()
                .sort_values("Algorithm")
                .reset_index(drop=True)
            )
            run_counts = (
                value_df.groupby("Algorithm", as_index=False)
                .size()
                .rename(columns={"size": "Runs"})
            )
            round_df = round_df.merge(run_counts, on="Algorithm", how="left")
            print(f"\n{exp} = {_display_value(value)}")
            print(round_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def summarize_by_value(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Experiment", "Variable_Value", "Algorithm"], as_index=False)[
            METRIC_COLUMNS
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns.to_flat_index()
    ]

    summary = summary.rename(
        columns={
            "Experiment_": "Experiment",
            "Variable_Value_": "Variable_Value",
            "Algorithm_": "Algorithm",
        }
    )
    std_cols = [c for c in summary.columns if c.endswith("_std")]
    if std_cols:
        summary[std_cols] = summary[std_cols].fillna(0.0)
    return summary.sort_values(
        ["Experiment", "Variable_Value", "Algorithm"]
    ).reset_index(drop=True)


def run_perturbation_with_replicates(
    runner: ExperimentRunner,
    tasks_list,
    tasks_data,
    param_name: str,
    param_values: List[int],
    algorithms: List[str],
    seed: int,
    fixed_arrival_chains: bool,
    replicates: int,
    base_rate: int,
    ntask_mode: str,
    ntask_pool_size: int,
    num_chains: int,
    base_num_types: int,
    base_chain_length: int,
):
    rows = []
    for rep in range(replicates):
        rep_seed = seed + rep
        rep_rows = runner.run_perturbation(
            tasks_list=tasks_list,
            tasks_data=tasks_data,
            param_name=param_name,
            param_values=param_values,
            algorithms=algorithms,
            base_num_types=base_num_types,
            base_length=base_chain_length,
            base_rate=base_rate,
            seed=rep_seed,
            fixed_arrival_chains=fixed_arrival_chains,
            ntask_mode=ntask_mode,
            ntask_pool_size=ntask_pool_size,
            num_chains=num_chains,
        )
        for row in rep_rows:
            row["Replicate"] = rep
            row["Replicate_Seed"] = rep_seed
        rows.extend(rep_rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deployment and routing experiment framework"
    )
    parser.add_argument(
        "--excel",
        type=str,
        default=os.path.join(
            CURRENT_DIR, "data", "evaluation_tables_20260325_163931.xlsx"
        ),
        help="Path to Excel model library",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        help="Algorithms to run (our ffd-m cds-m random-m greedy-m lego drs)",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "single"],
        default="all",
        help="Run all three perturbations or one custom perturbation",
    )
    parser.add_argument(
        "--perturb",
        choices=["arrival_rate", "chain_length", "n_task_types"],
        default="arrival_rate",
        help="Perturbation axis when mode=single",
    )
    parser.add_argument(
        "--values",
        type=str,
        default="100,200,300,400,500,600,700,800",
        help="Comma-separated values when mode=single",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--chain-base-rate",
        type=int,
        default=300,
        help="Base arrival rate used when perturbing chain_length",
    )
    parser.add_argument(
        "--arrival-base-ntask",
        type=int,
        default=10,
        help="Base n_task_types used when perturbing arrival_rate",
    )
    parser.add_argument(
        "--arrival-base-chainlen",
        type=int,
        default=4,
        help="Base chain_length used when perturbing arrival_rate",
    )
    parser.add_argument(
        "--ntask-base-rate",
        type=int,
        default=300,
        help="Base arrival rate used when perturbing n_task_types",
    )
    parser.add_argument(
        "--ntask-mode",
        choices=["pool_size", "unique_in_chains", "unique_chain_exact"],
        default="unique_in_chains",
        help="Interpret n_task_types as pool size, unique task count in chains, or exact unique-chain mode (length=n_task_types)",
    )
    parser.add_argument(
        "--ntask-pool-size",
        type=int,
        default=80,
        help="Fixed candidate task pool size when --ntask-mode=unique_in_chains",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=10,
        help="Number of generated user-chain templates per experiment",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of random-seed repeats per perturbation",
    )
    parser.add_argument(
        "--replicate-targets",
        type=str,
        default="chain_length,n_task_types",
        help="Comma-separated perturbations using multi-seed repeats",
    )
    parser.add_argument(
        "--arrival-chain-mode",
        choices=["fixed", "regenerate"],
        default="fixed",
        help="When perturb=arrival_rate, keep chain templates fixed or regenerate per value",
    )
    parser.add_argument(
        "--our-generations", type=int, default=40, help="Evolution generations for Our"
    )
    parser.add_argument(
        "--our-pop-size", type=int, default=36, help="Population size for Our"
    )
    parser.add_argument(
        "--max-required-tasks",
        type=int,
        default=80,
        help="Expand task pool to this size by copy augmentation",
    )
    parser.add_argument(
        "--output-dir", type=str, default=os.path.join(CURRENT_DIR, "results")
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller perturbation grids for smoke checks",
    )

    args = parser.parse_args()

    print(f"Loading model library: {args.excel}")
    tasks_list, tasks_data = load_and_prepare_data(
        excel_file=args.excel,
        max_required_tasks=args.max_required_tasks,
    )
    print(f"Task pool size: {len(tasks_list)}")
    print(f"Algorithms: {args.algorithms}")

    runner = ExperimentRunner(
        our_generations=args.our_generations,
        our_pop_size=args.our_pop_size,
    )

    all_rows = []
    fixed_arrival_chains = args.arrival_chain_mode == "fixed"
    replicate_targets = parse_str_set(args.replicate_targets)

    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")
    if args.chain_base_rate < 1 or args.ntask_base_rate < 1:
        raise ValueError("--chain-base-rate and --ntask-base-rate must be >= 1")
    if args.arrival_base_ntask < 1 or args.arrival_base_chainlen < 1:
        raise ValueError(
            "--arrival-base-ntask and --arrival-base-chainlen must be >= 1"
        )
    if args.ntask_pool_size < 1:
        raise ValueError("--ntask-pool-size must be >= 1")
    if args.num_chains < 1:
        raise ValueError("--num-chains must be >= 1")

    base_rate_by_param = {
        "arrival_rate": 200,
        "chain_length": args.chain_base_rate,
        "n_task_types": args.ntask_base_rate,
    }

    if args.mode == "single":
        values = parse_int_list(args.values)
        if args.quick:
            values = values[: min(3, len(values))]

        replicate_count = args.replicates if args.perturb in replicate_targets else 1
        rows = run_perturbation_with_replicates(
            runner=runner,
            tasks_list=tasks_list,
            tasks_data=tasks_data,
            param_name=args.perturb,
            param_values=values,
            algorithms=args.algorithms,
            seed=args.seed,
            fixed_arrival_chains=fixed_arrival_chains,
            replicates=replicate_count,
            base_rate=base_rate_by_param[args.perturb],
            ntask_mode=args.ntask_mode,
            ntask_pool_size=args.ntask_pool_size,
            num_chains=args.num_chains,
            base_num_types=args.arrival_base_ntask,
            base_chain_length=args.arrival_base_chainlen,
        )
        all_rows.extend(rows)

        out_path = os.path.join(args.output_dir, f"perturb_{args.perturb}.csv")
        perturb_df = save_rows(rows, out_path)
        print(f"Saved: {out_path}")
        summary_value_path = os.path.join(
            args.output_dir, f"perturb_{args.perturb}_summary_mean_std.csv"
        )
        summarize_by_value(perturb_df).to_csv(
            summary_value_path, index=False, encoding="utf-8-sig"
        )
        print(f"Saved: {summary_value_path}")

    else:
        experiments = [
            ("arrival_rate", DEFAULT_RATE_VALUES),
            ("chain_length", DEFAULT_LENGTH_VALUES),
            ("n_task_types", DEFAULT_TASK_TYPE_VALUES),
        ]

        for param, values in experiments:
            local_values = values[:3] if args.quick else values
            print(f"\nRunning perturbation: {param} = {local_values}")
            replicate_count = args.replicates if param in replicate_targets else 1
            rows = run_perturbation_with_replicates(
                runner=runner,
                tasks_list=tasks_list,
                tasks_data=tasks_data,
                param_name=param,
                param_values=local_values,
                algorithms=args.algorithms,
                seed=args.seed,
                fixed_arrival_chains=fixed_arrival_chains,
                replicates=replicate_count,
                base_rate=base_rate_by_param[param],
                ntask_mode=args.ntask_mode,
                ntask_pool_size=args.ntask_pool_size,
                num_chains=args.num_chains,
                base_num_types=args.arrival_base_ntask,
                base_chain_length=args.arrival_base_chainlen,
            )
            all_rows.extend(rows)

            out_path = os.path.join(args.output_dir, f"perturb_{param}.csv")
            perturb_df = save_rows(rows, out_path)
            print(f"Saved: {out_path}")
            summary_value_path = os.path.join(
                args.output_dir, f"perturb_{param}_summary_mean_std.csv"
            )
            summarize_by_value(perturb_df).to_csv(
                summary_value_path, index=False, encoding="utf-8-sig"
            )
            print(f"Saved: {summary_value_path}")

    all_df = save_rows(all_rows, os.path.join(args.output_dir, "all_results.csv"))

    summary_df = (
        all_df.groupby(["Experiment", "Algorithm"], as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values(["Experiment", "Algorithm"])
    )
    summary_path = os.path.join(args.output_dir, "summary_by_experiment_algorithm.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {os.path.join(args.output_dir, 'all_results.csv')}")
    print(f"Saved: {summary_path}")
    overall_by_value_path = os.path.join(
        args.output_dir, "summary_by_value_algorithm_mean_std.csv"
    )
    summarize_by_value(all_df).to_csv(
        overall_by_value_path, index=False, encoding="utf-8-sig"
    )
    print(f"Saved: {overall_by_value_path}")
    print_round_details(all_df)
    print_brief_summary(all_df)


if __name__ == "__main__":
    main()
