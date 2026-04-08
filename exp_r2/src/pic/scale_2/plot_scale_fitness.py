import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_CANDIDATES = [
    BASE_DIR / "scale_stability_results.csv",
    BASE_DIR.parents[2] / "results" / "scale_stability_results.csv",
]
OUTPUT_PNG = BASE_DIR / "scale_fitness.png"
OUTPUT_EPS = BASE_DIR / "scale_fitness.eps"


PLOT_CONFIG = {
    "fig_size": (7.16, 4.2),
    "ylabel_fontsize": 14,
    "tick_labelsize": 12,
    "legend_fontsize": 11,
    "grid_linewidth": 0.8,
    "grid_color": "#EBEBEB",
    "dpi": 300,
    "line_width_baseline": 1.6,
    "line_width_ours": 2.4,
    "marker_size_baseline": 5.5,
    "marker_size_ours": 7.5,
    "ribbon_alpha_baseline": 0.15,
    "ribbon_alpha_ours": 0.25,
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

MARKERS = {
    "DRS": "s",
    "FFD": "^",
    "LEGO": "D",
    "RLS": "o",
    "OURS": "*",
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


def _resolve_raw_data_path() -> Path:
    for candidate in RAW_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No raw scale stability CSV found. Tried: "
        + ", ".join(str(p) for p in RAW_DATA_CANDIDATES)
    )


def main() -> None:
    data_path = _resolve_raw_data_path()
    df = pd.read_csv(data_path)

    if "Fitness" not in df.columns:
        raise ValueError("Raw CSV must contain a 'Fitness' column for plotting.")

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

    x_positions = np.arange(len(scales))

    # Compute statistics for each algorithm
    stats = {}
    for algo in algorithms:
        algo_stats = (
            df[df["Algorithm_Norm"] == algo]
            .groupby("Scale_Norm")["Fitness"]
            .agg(["mean", "std"])
            .reindex(scales)
        )
        stats[algo] = algo_stats

    # Draw baselines first (so OURS is on top)
    baselines = [a for a in algorithms if a != "OURS"]
    draw_order = baselines + (["OURS"] if "OURS" in algorithms else [])

    legend_handles = []
    all_values = []

    for algo in draw_order:
        algo_stats = stats[algo]
        mean_values = algo_stats["mean"].to_numpy(dtype=float)
        std_values = algo_stats["std"].to_numpy(dtype=float).copy()
        std_values = np.where(np.isnan(std_values), 0.0, std_values)

        valid_mask = ~np.isnan(mean_values)
        if not valid_mask.any():
            continue

        x_valid = x_positions[valid_mask]
        mean_valid = mean_values[valid_mask]
        std_valid = std_values[valid_mask]

        all_values.extend((mean_valid - std_valid).tolist())
        all_values.extend((mean_valid + std_valid).tolist())

        color = COLORS.get(algo, "#5A5A5A")
        is_ours = algo == "OURS"
        lw = PLOT_CONFIG["line_width_ours"] if is_ours else PLOT_CONFIG["line_width_baseline"]
        ms = PLOT_CONFIG["marker_size_ours"] if is_ours else PLOT_CONFIG["marker_size_baseline"]
        alpha = PLOT_CONFIG["ribbon_alpha_ours"] if is_ours else PLOT_CONFIG["ribbon_alpha_baseline"]
        marker = MARKERS.get(algo, "o")

        # CI ribbon
        ax.fill_between(
            x_valid,
            mean_valid - std_valid,
            mean_valid + std_valid,
            color=color,
            alpha=alpha,
            linewidth=0,
        )

        # Line + markers
        ax.plot(
            x_valid,
            mean_valid,
            color=color,
            marker=marker,
            markersize=ms,
            linewidth=lw,
            markeredgewidth=0.0,
            markeredgecolor=color,
            label=algo,
            zorder=5 if is_ours else 3,
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                color=color,
                marker=marker,
                markersize=ms * 0.7,
                linewidth=lw * 0.7,
                markeredgewidth=0.0,
                label=algo,
            )
        )

    ax.set_ylabel(
        "Fitness",
        fontsize=PLOT_CONFIG["ylabel_fontsize"],
        fontname="Times New Roman",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([SCALE_LABELS[s] for s in scales])

    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=PLOT_CONFIG["legend_fontsize"],
        prop={"family": "Times New Roman", "size": PLOT_CONFIG["legend_fontsize"]},
        frameon=False,
        fancybox=False,
    )
    for text in legend.get_texts():
        if text.get_text() == "OURS":
            text.set_fontweight("bold")

    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_margin = (y_max - y_min) * 0.10 if y_max > y_min else 0.1
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
    print(f"使用数据源: {data_path}")
    print(f"图表已保存至: {OUTPUT_PNG}")
    print(f"图表已保存至: {OUTPUT_EPS}")


if __name__ == "__main__":
    main()
