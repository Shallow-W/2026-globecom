"""CLI entry for three-scale algorithm stability experiments."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from src.framework.constants import DEFAULT_ALGORITHMS
from src.framework.data_loader import load_and_prepare_data
from src.framework.experiment_builder import build_context
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

SCALE_PROFILES: Dict[str, Dict[str, int]] = {
    "small": {
        "n_nodes": 3,
        "n_task_types": 3,
        "chain_length": 3,
        "arrival_rate": 100,
    },
    "medium": {
        "n_nodes": 5,
        "n_task_types": 5,
        "chain_length": 5,
        "arrival_rate": 300,
    },
    "large": {
        "n_nodes": 7,
        "n_task_types": 7,
        "chain_length": 7,
        "arrival_rate": 350,
    },
}


def parse_str_list(raw: str) -> List[str]:
    return [v.strip() for v in raw.split(",") if v.strip()]


def save_rows(rows, path: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def summarize_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Scale", "Algorithm"])[METRIC_COLUMNS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns.to_flat_index()
    ]
    summary = summary.rename(
        columns={
            "Scale_": "Scale",
            "Algorithm_": "Algorithm",
        }
    )

    std_cols = [c for c in summary.columns if c.endswith("_std")]
    if std_cols:
        summary[std_cols] = summary[std_cols].fillna(0.0)

    return summary.sort_values(["Scale", "Algorithm"]).reset_index(drop=True)


def build_stability_metrics(
    raw_df: pd.DataFrame, summary_df: pd.DataFrame
) -> pd.DataFrame:
    stability = summary_df.copy()

    for metric in METRIC_COLUMNS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        cv_col = f"{metric}_cv"

        if mean_col in stability.columns and std_col in stability.columns:
            mean_values = stability[mean_col].to_numpy(dtype=float)
            std_values = stability[std_col].to_numpy(dtype=float)
            cv_values = np.zeros_like(mean_values)

            denom = np.abs(mean_values)
            nz_mask = denom > 0.0
            cv_values[nz_mask] = std_values[nz_mask] / denom[nz_mask]
            cv_values[np.abs(cv_values) < 1e-15] = 0.0
            stability[cv_col] = cv_values

    failure = (
        raw_df.assign(Failed=raw_df["Penalty_Score"] > 0)
        .groupby(["Scale", "Algorithm"], as_index=False)["Failed"]
        .mean()
        .rename(columns={"Failed": "Failure_Rate"})
    )

    stability = stability.merge(failure, on=["Scale", "Algorithm"], how="left")
    stability["Failure_Rate"] = stability["Failure_Rate"].fillna(0.0)

    keep_cols = ["Scale", "Algorithm", "Failure_Rate"]
    keep_cols.extend([c for c in stability.columns if c.endswith("_cv")])
    run_count_cols = [
        c for c in stability.columns if c.endswith("_count") and c.startswith("Fitness")
    ]
    if run_count_cols:
        run_col = run_count_cols[0]
        stability = stability.rename(columns={run_col: "Runs"})
        keep_cols.append("Runs")

    return (
        stability[keep_cols].sort_values(["Scale", "Algorithm"]).reset_index(drop=True)
    )


def print_scale_summary(raw_df: pd.DataFrame) -> None:
    if raw_df.empty:
        print("No results.")
        return

    summary = (
        raw_df.groupby(["Scale", "Algorithm"], as_index=False)[METRIC_COLUMNS]
        .mean()
        .sort_values(["Scale", "Algorithm"])
    )

    print("\n=== Scale Stability Mean Summary (Scale + Algorithm) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def run_scale_stability(
    runner: ExperimentRunner,
    tasks_list,
    tasks_data,
    algorithms: List[str],
    scale_names: List[str],
    replicates: int,
    seed: int,
    num_chains: int,
):
    rows = []

    for scale_idx, scale_name in enumerate(scale_names):
        if scale_name not in SCALE_PROFILES:
            raise ValueError(f"Unsupported scale: {scale_name}")

        cfg = SCALE_PROFILES[scale_name]
        print(
            f"\nRunning scale={scale_name} | "
            f"nodes={cfg['n_nodes']} ntask={cfg['n_task_types']} "
            f"chainlen={cfg['chain_length']} ar={cfg['arrival_rate']}"
        )

        for rep in range(replicates):
            rep_seed = seed + scale_idx * 10_000 + rep
            ctx = build_context(
                tasks_list=tasks_list,
                tasks_data=tasks_data,
                num_types=cfg["n_task_types"],
                length=cfg["chain_length"],
                total_rate=cfg["arrival_rate"],
                num_chains=num_chains,
                rng_seed=rep_seed,
                n_nodes=cfg["n_nodes"],
            )

            comparison = runner.run_comparison(
                ctx=ctx,
                algorithms=algorithms,
                seed=rep_seed,
            )
            for row in comparison:
                row["Experiment"] = "scale_stability"
                row["Variable_Value"] = scale_name
                row["Scale"] = scale_name
                row["n_nodes"] = cfg["n_nodes"]
                row["n_task_types"] = cfg["n_task_types"]
                row["chain_length"] = cfg["chain_length"]
                row["arrival_rate"] = cfg["arrival_rate"]
                row["Replicate"] = rep
                row["Replicate_Seed"] = rep_seed
            rows.extend(comparison)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-scale stability experiment")
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
        help="Algorithms to run",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="small,medium,large",
        help="Comma-separated scales to run",
    )
    parser.add_argument("--replicates", type=int, default=10, help="Repeat times")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--num-chains",
        type=int,
        default=10,
        help="Number of generated user-chain templates per experiment",
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
        "--output-dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "results"),
        help="Output directory",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smoke check with one replicate",
    )

    args = parser.parse_args()

    if args.replicates < 1:
        raise ValueError("--replicates must be >= 1")
    if args.num_chains < 1:
        raise ValueError("--num-chains must be >= 1")

    scales = parse_str_list(args.scales)
    if not scales:
        raise ValueError("--scales cannot be empty")

    bad_scales = [s for s in scales if s not in SCALE_PROFILES]
    if bad_scales:
        raise ValueError(
            f"Unsupported scales: {bad_scales}. Allowed: {list(SCALE_PROFILES.keys())}"
        )

    replicates = 1 if args.quick else args.replicates

    print(f"Loading model library: {args.excel}")
    tasks_list, tasks_data = load_and_prepare_data(
        excel_file=args.excel,
        max_required_tasks=args.max_required_tasks,
    )
    print(f"Task pool size: {len(tasks_list)}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Scales: {scales}")
    print(f"Replicates per scale: {replicates}")

    runner = ExperimentRunner(
        our_generations=args.our_generations,
        our_pop_size=args.our_pop_size,
    )

    rows = run_scale_stability(
        runner=runner,
        tasks_list=tasks_list,
        tasks_data=tasks_data,
        algorithms=args.algorithms,
        scale_names=scales,
        replicates=replicates,
        seed=args.seed,
        num_chains=args.num_chains,
    )

    raw_path = os.path.join(args.output_dir, "scale_stability_results.csv")
    raw_df = save_rows(rows, raw_path)

    summary_df = summarize_mean_std(raw_df)
    summary_path = os.path.join(args.output_dir, "scale_stability_summary_mean_std.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    stability_df = build_stability_metrics(raw_df, summary_df)
    stability_path = os.path.join(
        args.output_dir, "scale_stability_stability_metrics.csv"
    )
    stability_df.to_csv(stability_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {stability_path}")

    print_scale_summary(raw_df)


if __name__ == "__main__":
    main()
