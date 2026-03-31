"""CLI entry for the modular deployment-routing experiment framework."""

from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from src.framework.constants import (
    DEFAULT_ALGORITHMS,
    DEFAULT_LENGTH_VALUES,
    DEFAULT_RATE_VALUES,
    DEFAULT_TASK_TYPE_VALUES,
)
from src.framework.data_loader import load_and_prepare_data
from src.framework.runner import ExperimentRunner


def parse_int_list(raw: str) -> List[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


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
        df.groupby(["Experiment", "Algorithm"], as_index=False)[
            [
                "Total_Delay_D",
                "Avg_QoS_Q",
                "Comp_Delay",
                "Comm_Delay",
                "Mem_Utilization",
                "Penalty_Score",
                "Fitness",
            ]
        ]
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
    metric_cols = [
        "Algorithm",
        "Total_Delay_D",
        "Avg_QoS_Q",
        "Comp_Delay",
        "Comm_Delay",
        "Mem_Utilization",
        "Penalty_Score",
        "Fitness",
    ]

    for exp in experiments:
        exp_df = df[df["Experiment"] == exp]
        values = sorted(exp_df["Variable_Value"].unique().tolist())

        print(f"\n=== Per-round Results: {exp} ===")
        for value in values:
            round_df = (
                exp_df[exp_df["Variable_Value"] == value][metric_cols]
                .sort_values("Algorithm")
                .reset_index(drop=True)
            )
            print(f"\n{exp} = {_display_value(value)}")
            print(round_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Deployment and routing experiment framework")
    parser.add_argument(
        "--excel",
        type=str,
        default=os.path.join(CURRENT_DIR, "data", "evaluation_tables_20260325_163931.xlsx"),
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
    parser.add_argument("--our-generations", type=int, default=40, help="Evolution generations for Our")
    parser.add_argument("--our-pop-size", type=int, default=36, help="Population size for Our")
    parser.add_argument(
        "--max-required-tasks",
        type=int,
        default=80,
        help="Expand task pool to this size by copy augmentation",
    )
    parser.add_argument("--output-dir", type=str, default=os.path.join(CURRENT_DIR, "results"))
    parser.add_argument("--quick", action="store_true", help="Use smaller perturbation grids for smoke checks")

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

    if args.mode == "single":
        values = parse_int_list(args.values)
        if args.quick:
            values = values[: min(3, len(values))]

        rows = runner.run_perturbation(
            tasks_list=tasks_list,
            tasks_data=tasks_data,
            param_name=args.perturb,
            param_values=values,
            algorithms=args.algorithms,
            seed=args.seed,
        )
        all_rows.extend(rows)

        out_path = os.path.join(args.output_dir, f"perturb_{args.perturb}.csv")
        save_rows(rows, out_path)
        print(f"Saved: {out_path}")

    else:
        experiments = [
            ("arrival_rate", DEFAULT_RATE_VALUES),
            ("chain_length", DEFAULT_LENGTH_VALUES),
            ("n_task_types", DEFAULT_TASK_TYPE_VALUES),
        ]

        for param, values in experiments:
            local_values = values[:3] if args.quick else values
            print(f"\nRunning perturbation: {param} = {local_values}")
            rows = runner.run_perturbation(
                tasks_list=tasks_list,
                tasks_data=tasks_data,
                param_name=param,
                param_values=local_values,
                algorithms=args.algorithms,
                seed=args.seed,
            )
            all_rows.extend(rows)

            out_path = os.path.join(args.output_dir, f"perturb_{param}.csv")
            save_rows(rows, out_path)
            print(f"Saved: {out_path}")

    all_df = save_rows(all_rows, os.path.join(args.output_dir, "all_results.csv"))

    summary_df = (
        all_df.groupby(["Experiment", "Algorithm"], as_index=False)[
            [
                "Total_Delay_D",
                "Avg_QoS_Q",
                "Comp_Delay",
                "Comm_Delay",
                "Mem_Utilization",
                "Penalty_Score",
                "Fitness",
            ]
        ]
        .mean()
        .sort_values(["Experiment", "Algorithm"])
    )
    summary_path = os.path.join(args.output_dir, "summary_by_experiment_algorithm.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {os.path.join(args.output_dir, 'all_results.csv')}")
    print(f"Saved: {summary_path}")
    print_round_details(all_df)
    print_brief_summary(all_df)


if __name__ == "__main__":
    main()
