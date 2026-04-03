import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "scale_stability_summary_mean_std.csv"
OUTPUT_PNG = BASE_DIR / "scale_fitness.png"
OUTPUT_EPS = BASE_DIR / "scale_fitness.eps"


PLOT_CONFIG = {
    "fig_size": (7.16, 4.2),
    "xlabel_fontsize": 14,
    "ylabel_fontsize": 14,
    "title_fontsize": 14,
    "tick_labelsize": 12,
    "legend_fontsize": 11,
    "legend_framealpha": 1.0,
    "legend_edgecolor": "#FFFFFF",
    "legend_facecolor": "#F8F8F8",
    "grid_linewidth": 0.8,
    "grid_color": "#EBEBEB",
    "dpi": 300,
    "bar_width": 0.16,
    "bar_group_gap": 1.35,
    "error_cap_size": 3,
}


COLORS = {
    "CDS": "#8ECFC9",
    "RLS": "#EEC994",
    "FFD": "#AFDEF3",
    "DRS": "#A5A3C3",
    "LEGO": "#FFE6B7",
    "GREEDY": "#F5EFBA",
    "OURS": "#FA7F6F",
}


ALGORITHM_ORDER = ["CDS", "DRS", "FFD", "GREEDY", "LEGO", "RLS", "OURS"]
EXCLUDED_ALGORITHMS = {"CDS", "GREEDY"}

ALGORITHM_MAP = {
    "cds-m": "CDS",
    "drs": "DRS",
    "ffd-m": "FFD",
    "greedy-m": "GREEDY",
    "lego": "LEGO",
    "random-m": "RLS",
    "our": "OURS",
    "ours": "OURS",
}

SCALE_ORDER = ["small", "medium", "large"]
SCALE_LABELS = {
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
}


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["Algorithm_Norm"] = (
        df["Algorithm"].astype(str).str.strip().str.lower().map(ALGORITHM_MAP)
    )
    df["Scale_Norm"] = df["Scale"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["Algorithm_Norm"]).copy()

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

    fig, ax = plt.subplots(1, 1, figsize=PLOT_CONFIG["fig_size"])

    available_algorithms = set(df["Algorithm_Norm"].unique())
    algorithms = [
        algo
        for algo in ALGORITHM_ORDER
        if algo in available_algorithms and algo not in EXCLUDED_ALGORITHMS
    ]

    available_scales = set(df["Scale_Norm"].unique())
    scales = [s for s in SCALE_ORDER if s in available_scales]

    if not algorithms:
        raise ValueError("No valid algorithms found in CSV for plotting.")
    if not scales:
        raise ValueError("No valid scales found in CSV for plotting.")

    n_algos = len(algorithms)
    base_positions = np.arange(len(scales)) * PLOT_CONFIG["bar_group_gap"]
    centered_offsets = np.arange(n_algos) - (n_algos - 1) / 2
    positions = [
        base_positions + offset * PLOT_CONFIG["bar_width"] for offset in centered_offsets
    ]

    all_bar_values = []
    for i, algo in enumerate(algorithms):
        algo_data = (
            df[df["Algorithm_Norm"] == algo].set_index("Scale_Norm").reindex(scales)
        )
        mean_values = algo_data["Fitness_mean"].to_numpy(dtype=float)
        std_values = algo_data["Fitness_std"].to_numpy(dtype=float)
        valid_mask = ~np.isnan(mean_values)

        if valid_mask.any():
            all_bar_values.extend((mean_values[valid_mask] - std_values[valid_mask]).tolist())
            all_bar_values.extend((mean_values[valid_mask] + std_values[valid_mask]).tolist())

        ax.bar(
            positions[i][valid_mask],
            mean_values[valid_mask],
            width=PLOT_CONFIG["bar_width"],
            yerr=std_values[valid_mask],
            capsize=PLOT_CONFIG["error_cap_size"],
            label=algo,
            color=COLORS.get(algo),
            edgecolor="none",
            error_kw={"elinewidth": 0.9, "ecolor": "#555555"},
        )

    ax.set_xlabel(
        "Scale",
        fontsize=PLOT_CONFIG["xlabel_fontsize"],
        fontname="Times New Roman",
    )
    ax.set_ylabel(
        "Fitness",
        fontsize=PLOT_CONFIG["ylabel_fontsize"],
        fontname="Times New Roman",
    )
    ax.set_title(
        "Fitness under Different Scales",
        fontsize=PLOT_CONFIG["title_fontsize"],
        fontweight="bold",
        fontname="Times New Roman",
    )

    ax.set_xticks(base_positions)
    ax.set_xticklabels([SCALE_LABELS[s] for s in scales])

    legend = ax.legend(
        loc="upper left",
        fontsize=PLOT_CONFIG["legend_fontsize"],
        prop={"family": "Times New Roman", "size": PLOT_CONFIG["legend_fontsize"]},
        frameon=False,
        framealpha=PLOT_CONFIG["legend_framealpha"],
        facecolor=PLOT_CONFIG["legend_facecolor"],
        edgecolor=PLOT_CONFIG["legend_edgecolor"],
        fancybox=False,
    )
    for text in legend.get_texts():
        if text.get_text() == "OURS":
            text.set_fontweight("bold")

    if all_bar_values:
        y_min = min(all_bar_values)
        y_max = max(all_bar_values)
        y_margin = (y_max - y_min) * 0.12 if y_max > y_min else 0.1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.grid(
        True,
        axis="y",
        color=PLOT_CONFIG["grid_color"],
        linewidth=PLOT_CONFIG["grid_linewidth"],
    )
    ax.tick_params(axis="both", labelsize=PLOT_CONFIG["tick_labelsize"])
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname("Times New Roman")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.savefig(OUTPUT_EPS, format="eps", bbox_inches="tight")
    print(f"图表已保存至: {OUTPUT_PNG}")
    print(f"图表已保存至: {OUTPUT_EPS}")


if __name__ == "__main__":
    main()