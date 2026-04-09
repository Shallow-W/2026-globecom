"""Plot Experiment 6: Dynamic Adaptation - Delay over Time Steps.

Reads the summary CSV from run_exp.py and produces a two-panel figure:
  Top:    Load Factor over time (with stress threshold line)
  Bottom: Average delay per algorithm over time (with std band)

Usage (from exp_r2 root):
  python -m src.pic.exp_6.plot_exp6
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(CURRENT_DIR, "results", "exp6_dynamic_adaptation_summary.csv")

# ============ Plot Style ============
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }
)

ALGORITHM_ORDER = ["our", "ffd-m", "lego", "drs", "random-m"]

DISPLAY_LABELS = {
    "our": "OURS",
    "ffd-m": "FFD",
    "lego": "LEGO",
    "drs": "DRS",
    "random-m": "RANDOM",
}

COLORS = {
    "our": "#d62728",
    "ffd-m": "#2ca02c",
    "lego": "#9467bd",
    "drs": "#ff7f0e",
    "random-m": "#7f7f7f",
}

MARKERS = {
    "our": "*",
    "ffd-m": "^",
    "lego": "p",
    "drs": "s",
    "random-m": "v",
}

STRESS_THRESHOLD = 1.2  # load_factor above this triggers stress


def _setup_ticks(ax):
    ax.tick_params(axis="both", labelsize=14)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname("Times New Roman")


def main():
    df = pd.read_csv(DATA_PATH)

    # Get load factor series (same for all algorithms per step)
    load_df = df[df["Algorithm"] == df["Algorithm"].iloc[0]].sort_values("Step")

    fig, (ax_load, ax_delay) = plt.subplots(
        2, 1, figsize=(12, 7), height_ratios=[1, 3], sharex=True
    )

    # ---- Top panel: Load Factor ----
    steps = load_df["Step"].values
    lf = load_df["Load_Factor"].values

    ax_load.fill_between(
        steps, 1.0, lf, where=(lf >= 1.0), color="#ffcccc", alpha=0.6, label=""
    )
    ax_load.fill_between(
        steps, 1.0, lf, where=(lf < 1.0), color="#ccffcc", alpha=0.6, label=""
    )
    ax_load.plot(steps, lf, color="#555555", linewidth=1.5, marker="o", markersize=3)
    ax_load.axhline(
        y=STRESS_THRESHOLD, color="#d62728", linewidth=1, linestyle=":", alpha=0.7
    )
    ax_load.axhline(y=1.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_load.set_ylabel("Load Factor", fontsize=14, fontname="Times New Roman")
    ax_load.set_ylim(0.0, 3.5)
    _setup_ticks(ax_load)
    ax_load.text(
        steps[-1] + 0.5,
        STRESS_THRESHOLD,
        "stress",
        fontsize=10,
        color="#d62728",
        va="center",
        fontname="Times New Roman",
    )

    # ---- Bottom panel: Delay ----
    for algo in ALGORITHM_ORDER:
        algo_data = df[df["Algorithm"] == algo].sort_values("Step")
        if algo_data.empty:
            continue

        label = DISPLAY_LABELS.get(algo, algo)
        color = COLORS.get(algo, None)
        marker = MARKERS.get(algo, "o")
        n_pts = len(algo_data)

        ax_delay.plot(
            algo_data["Step"],
            algo_data["Total_Delay_D_mean"],
            marker=marker,
            label=label,
            color=color,
            linewidth=2,
            linestyle="--",
            markersize=5,
            markevery=max(1, n_pts // 12),
        )

    ax_delay.set_xlabel("Time Step", fontsize=20, fontname="Times New Roman")
    ax_delay.set_ylabel(
        "Average End-to-End Latency (s)", fontsize=16, fontname="Times New Roman"
    )
    _setup_ticks(ax_delay)

    legend = ax_delay.legend(
        loc="upper left",
        fontsize=12,
        prop={"family": "Times New Roman", "size": 12},
    )
    for text in legend.get_texts():
        if text.get_text() == "OURS":
            text.set_fontweight("bold")

    ax_delay.grid(True, color="#EBEBEB", linewidth=0.8)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)

    fig.savefig(
        os.path.join(CURRENT_DIR, "exp6_dynamic_adaptation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(CURRENT_DIR, "exp6_dynamic_adaptation.eps"),
        format="eps",
        bbox_inches="tight",
    )
    print(f"Plot saved to {CURRENT_DIR}")


if __name__ == "__main__":
    main()
