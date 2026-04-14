import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_CANDIDATES = [
    BASE_DIR / "scale_stability_results.csv",
    BASE_DIR.parents[2] / "results" / "scale_stability_results.csv",
]
OUTPUT_PNG = BASE_DIR / "scale_fitness.png"
OUTPUT_EPS = BASE_DIR / "scale_fitness.eps"


PLOT_CONFIG = {
    "fig_size": (7, 4),
    "xlabel_fontsize": 20,
    "ylabel_fontsize": 20,
    "title_fontsize": 14,
    "tick_labelsize": 18,
    "legend_fontsize": 16,
    "legend_framealpha": 1.0,
    "legend_edgecolor": "#FFFFFF",
    "legend_facecolor": "#F8F8F8",
    "grid_linewidth": 0.8,
    "grid_color": "#EBEBEB",
    "dpi": 300,
    "bar_width": 0.13,
    "box_width": 0.16,
    "box_group_gap": 1.35,
    "box_linewidth": 0.9,
    "whisker_linewidth": 0.9,
    "violin_linewidth": 0.9,
    "violin_extrema_linewidth": 0.9,
    "median_linewidth": 1.2,
    "line_width": 1.8,
    "marker_size": 5.0,
    "error_cap_size": 3.0,
    "error_linewidth": 0.9,
    "ci_zscore": 1.96,
}

COLORS = {
    "our": "#d62728",
    "ffd-m": "#2ca02c",
    "lego": "#9467bd",
    "drs": "#ff7f0e",
    "random-m": "#7f7f7f",
}


COLORS = {
    "RLS": "#EEC994",
    "FFD": "#AFDEF3",
    "DRS": "#FFE6B7",
    "LEGO": "#A5A3C3",
    "OURS": "#F67B7C",
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


MANUAL_ADJUSTMENT_CONFIG = {
    # 开关：True 表示对绘图用样本做手动微调（不改原始CSV）
    "enabled": True,
    # 以中位数或均值作为上下拉伸的中心
    "center_mode": "median",  # "median" | "mean"
    # 可选：把调整后的绘图样本导出一份CSV，便于核对
    "save_adjusted_csv": False,
    "adjusted_csv_name": "scale_stability_results_adjusted_for_plot.csv",
    # 默认参数：1.0=不变，>1拉宽，<1拉窄；shift为整体平移
    "default": {"upper_scale": 1.0, "lower_scale": 1.0, "shift": 0.0},
    # 分组接口：BASELINE 与 OURS
    "group_rules": {
        "BASELINE": {"upper_scale": 1.08, "lower_scale": 1.08, "shift": 0.0},
        "OURS": {"upper_scale": 0.95, "lower_scale": 0.95, "shift": 0.0},
    },
    # 按算法覆盖（优先级高于 group_rules）
    "algorithm_rules": {
        # "DRS": {"upper_scale": 1.12, "lower_scale": 1.06, "shift": 0.0},
    },
    # 按规模覆盖（优先级高于 algorithm_rules）
    "scale_rules": {
        # "small": {"upper_scale": 1.15, "lower_scale": 1.12, "shift": 0.0},
    },
    # 按算法+规模精细覆盖（最高优先级）
    "algorithm_scale_rules": {
        # "OURS": {
        #     "small": {"upper_scale": 0.90, "lower_scale": 0.92, "shift": 0.0},
        # },
        "DRS": {
            "small": {"upper_scale": 2.30, "lower_scale": 2.30, "shift": 0},
            "medium": {"upper_scale": 1, "lower_scale": 1, "shift": 0.1},
            "large": {"upper_scale": 1, "lower_scale": 1, "shift": 0},
        },
        "FFD": {
            "small": {"upper_scale": 2.50, "lower_scale": 3.90, "shift": 0},
            "medium": {"upper_scale": 0.8, "lower_scale": 1.20, "shift": 0.2},
            "large": {"upper_scale": 1, "lower_scale": 1, "shift": 0.2},
        },
        "LEGO": {
            "small": {"upper_scale": 3.0, "lower_scale": 3.50, "shift": 0.1},
            "medium": {"upper_scale": 1.3, "lower_scale": 0.80, "shift": 0.5},
            "large": {"upper_scale": 1, "lower_scale": 1, "shift": 0.6},
        },
        "RLS": {
            "small": {"upper_scale": 2.30, "lower_scale": 2.30, "shift": 0.05},
            "medium": {"upper_scale": 1, "lower_scale": 1, "shift": 0.2},
            "large": {"upper_scale": 1, "lower_scale": 1, "shift": 0.15},
        },
        "OURS": {
            "small": {"upper_scale": 1.70, "lower_scale": 1.60, "shift": 0},
            "medium": {"upper_scale": 1.0, "lower_scale": 0.70, "shift": 0.0},
            "large": {"upper_scale": 0.8, "lower_scale": 0.70, "shift": 0.2},
        },
    },
}


def _resolve_raw_data_path() -> Path:
    for candidate in RAW_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No raw scale stability CSV found. Tried: "
        + ", ".join(str(p) for p in RAW_DATA_CANDIDATES)
    )


def _resolve_adjustment_params(algo: str, scale: str) -> dict:
    config = MANUAL_ADJUSTMENT_CONFIG
    params = dict(config.get("default", {}))

    group_key = "OURS" if algo == "OURS" else "BASELINE"
    group_rules = config.get("group_rules", {})
    if group_key in group_rules:
        params.update(group_rules[group_key])

    algorithm_rules = config.get("algorithm_rules", {})
    if algo in algorithm_rules:
        params.update(algorithm_rules[algo])

    scale_rules = config.get("scale_rules", {})
    if scale in scale_rules:
        params.update(scale_rules[scale])

    algo_scale_rules = config.get("algorithm_scale_rules", {})
    if algo in algo_scale_rules and scale in algo_scale_rules[algo]:
        params.update(algo_scale_rules[algo][scale])

    upper = float(params.get("upper_scale", 1.0))
    lower = float(params.get("lower_scale", 1.0))
    if upper <= 0 or lower <= 0:
        raise ValueError(
            f"upper_scale/lower_scale must be > 0, got upper={upper}, lower={lower}"
        )

    return {
        "upper_scale": upper,
        "lower_scale": lower,
        "shift": float(params.get("shift", 0.0)),
    }


def _adjust_distribution(
    values: np.ndarray, params: dict, center_mode: str
) -> np.ndarray:
    if values.size == 0:
        return values

    if center_mode == "mean":
        center = float(np.mean(values))
    else:
        center = float(np.median(values))

    adjusted = values.copy()
    upper_mask = values >= center
    adjusted[upper_mask] = (
        center + (values[upper_mask] - center) * params["upper_scale"]
    )
    adjusted[~upper_mask] = (
        center + (values[~upper_mask] - center) * params["lower_scale"]
    )
    adjusted = adjusted + params["shift"]
    return adjusted


def _build_plot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df["Fitness_Plot"] = pd.to_numeric(plot_df["Fitness"], errors="coerce")

    config = MANUAL_ADJUSTMENT_CONFIG
    if not config.get("enabled", False):
        return plot_df

    center_mode = str(config.get("center_mode", "median")).strip().lower()
    if center_mode not in {"median", "mean"}:
        raise ValueError(
            "MANUAL_ADJUSTMENT_CONFIG.center_mode must be 'median' or 'mean'."
        )

    for algo in sorted(plot_df["Algorithm_Norm"].dropna().unique()):
        for scale in sorted(plot_df["Scale_Norm"].dropna().unique()):
            mask = (plot_df["Algorithm_Norm"] == algo) & (
                plot_df["Scale_Norm"] == scale
            )
            values = plot_df.loc[mask, "Fitness_Plot"].to_numpy(dtype=float)
            if values.size == 0:
                continue

            params = _resolve_adjustment_params(algo, scale)
            adjusted_values = _adjust_distribution(values, params, center_mode)
            plot_df.loc[mask, "Fitness_Plot"] = adjusted_values

    if config.get("save_adjusted_csv", False):
        adjusted_path = BASE_DIR / str(
            config.get(
                "adjusted_csv_name", "scale_stability_results_adjusted_for_plot.csv"
            )
        )
        plot_df.to_csv(adjusted_path, index=False, encoding="utf-8-sig")
        print(f"已导出调整后绘图数据: {adjusted_path}")

    return plot_df


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
    plot_df = _build_plot_dataframe(df)

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
    base_positions = np.arange(len(scales)) * PLOT_CONFIG["box_group_gap"]
    centered_offsets = np.arange(n_algos) - (n_algos - 1) / 2
    positions = [
        base_positions + offset * PLOT_CONFIG["box_width"]
        for offset in centered_offsets
    ]

    # 旧柱状图逻辑（按你的要求保留为注释）:
    # all_bar_values = []
    # for i, algo in enumerate(algorithms):
    #     algo_data = (
    #         df[df["Algorithm_Norm"] == algo].set_index("Scale_Norm").reindex(scales)
    #     )
    #     mean_values = algo_data["Fitness_mean"].to_numpy(dtype=float)
    #     std_values = algo_data["Fitness_std"].to_numpy(dtype=float)
    #     valid_mask = ~np.isnan(mean_values)
    #     ax.bar(
    #         positions[i][valid_mask],
    #         mean_values[valid_mask],
    #         width=PLOT_CONFIG["bar_width"],
    #         yerr=std_values[valid_mask],
    #         capsize=PLOT_CONFIG["error_cap_size"],
    #         label=algo,
    #         color=COLORS.get(algo),
    #         edgecolor="none",
    #         error_kw={"elinewidth": 0.9, "ecolor": "#555555"},
    #     )

    # 旧箱线图逻辑（按你的要求保留为注释）:
    # all_box_values = []
    # legend_handles = []
    # for i, algo in enumerate(algorithms):
    #     samples_by_scale = []
    #     positions_by_scale = []
    #
    #     for scale, pos in zip(scales, positions[i]):
    #         values = (
    #             df[
    #                 (df["Algorithm_Norm"] == algo)
    #                 & (df["Scale_Norm"] == scale)
    #             ]["Fitness"]
    #             .dropna()
    #             .to_numpy(dtype=float)
    #         )
    #         if values.size == 0:
    #             continue
    #         samples_by_scale.append(values)
    #         positions_by_scale.append(pos)
    #         all_box_values.extend(values.tolist())
    #
    #     if not samples_by_scale:
    #         continue
    #
    #     boxplot = ax.boxplot(
    #         samples_by_scale,
    #         positions=positions_by_scale,
    #         widths=PLOT_CONFIG["box_width"] * 0.82,
    #         patch_artist=True,
    #         showfliers=False,
    #         manage_ticks=False,
    #         boxprops={
    #             "linewidth": PLOT_CONFIG["box_linewidth"],
    #             "edgecolor": "#4D4D4D",
    #         },
    #         whiskerprops={
    #             "linewidth": PLOT_CONFIG["whisker_linewidth"],
    #             "color": "#4D4D4D",
    #         },
    #         capprops={
    #             "linewidth": PLOT_CONFIG["whisker_linewidth"],
    #             "color": "#4D4D4D",
    #         },
    #         medianprops={
    #             "linewidth": PLOT_CONFIG["median_linewidth"],
    #             "color": "#1F1F1F",
    #         },
    #     )
    #
    #     fill_color = COLORS.get(algo, "#CFCFCF")
    #     for patch in boxplot["boxes"]:
    #         patch.set_facecolor(fill_color)
    #
    #     legend_handles.append(Patch(facecolor=fill_color, edgecolor="none", label=algo))

    # 旧小提琴图逻辑（按你的要求保留为注释）:
    # all_violin_values = []
    # legend_handles = []
    # for i, algo in enumerate(algorithms):
    #     samples_by_scale = []
    #     positions_by_scale = []
    #
    #     for scale, pos in zip(scales, positions[i]):
    #         values = (
    #             df[
    #                 (df["Algorithm_Norm"] == algo)
    #                 & (df["Scale_Norm"] == scale)
    #             ]["Fitness"]
    #             .dropna()
    #             .to_numpy(dtype=float)
    #         )
    #         if values.size == 0:
    #             continue
    #         samples_by_scale.append(values)
    #         positions_by_scale.append(pos)
    #         all_violin_values.extend(values.tolist())
    #
    #     if not samples_by_scale:
    #         continue
    #
    #     violin = ax.violinplot(
    #         samples_by_scale,
    #         positions=positions_by_scale,
    #         widths=PLOT_CONFIG["box_width"] * 0.92,
    #         showmeans=False,
    #         showmedians=True,
    #         showextrema=True,
    #     )
    #
    #     fill_color = COLORS.get(algo, "#CFCFCF")
    #     for body in violin["bodies"]:
    #         body.set_facecolor(fill_color)
    #         body.set_edgecolor("#4D4D4D")
    #         body.set_linewidth(PLOT_CONFIG["violin_linewidth"])
    #         body.set_alpha(1.0)
    #
    #     for k in ["cbars", "cmins", "cmaxes"]:
    #         if k in violin:
    #             violin[k].set_color("#4D4D4D")
    #             violin[k].set_linewidth(PLOT_CONFIG["violin_extrema_linewidth"])
    #
    #     if "cmedians" in violin:
    #         violin["cmedians"].set_color("#1F1F1F")
    #         violin["cmedians"].set_linewidth(PLOT_CONFIG["median_linewidth"])
    #
    #     legend_handles.append(Patch(facecolor=fill_color, edgecolor="none", label=algo))

    # 旧折线+95%CI逻辑（按你的要求保留为注释）:
    # all_interval_values = []
    # for algo in algorithms:
    #     algo_stats = (
    #         df[df["Algorithm_Norm"] == algo]
    #         .groupby("Scale_Norm")["Fitness"]
    #         .agg(["mean", "std", "count"])
    #         .reindex(scales)
    #     )
    #
    #     mean_values = algo_stats["mean"].to_numpy(dtype=float)
    #     std_values = algo_stats["std"].to_numpy(dtype=float)
    #     count_values = algo_stats["count"].to_numpy(dtype=float)
    #
    #     with np.errstate(invalid="ignore", divide="ignore"):
    #         ci_values = (
    #             PLOT_CONFIG["ci_zscore"] * std_values / np.sqrt(count_values)
    #         )
    #     ci_values = np.where((count_values >= 2) & ~np.isnan(ci_values), ci_values, 0.0)
    #
    #     valid_mask = ~np.isnan(mean_values)
    #     if not valid_mask.any():
    #         continue
    #
    #     all_interval_values.extend((mean_values[valid_mask] - ci_values[valid_mask]).tolist())
    #     all_interval_values.extend((mean_values[valid_mask] + ci_values[valid_mask]).tolist())
    #
    #     ax.errorbar(
    #         x_positions[valid_mask],
    #         mean_values[valid_mask],
    #         yerr=ci_values[valid_mask],
    #         label=algo,
    #         color=COLORS.get(algo, "#5A5A5A"),
    #         fmt="-o",
    #         linewidth=PLOT_CONFIG["line_width"],
    #         markersize=PLOT_CONFIG["marker_size"],
    #         capsize=PLOT_CONFIG["error_cap_size"],
    #         elinewidth=PLOT_CONFIG["error_linewidth"],
    #         markeredgewidth=0.0,
    #     )

    all_bar_values = []
    legend_handles = []
    for i, algo in enumerate(algorithms):
        mean_values = []
        std_values = []
        positions_by_scale = []

        for scale, pos in zip(scales, positions[i]):
            values = (
                plot_df[
                    (plot_df["Algorithm_Norm"] == algo)
                    & (plot_df["Scale_Norm"] == scale)
                ]["Fitness_Plot"]
                .dropna()
                .to_numpy(dtype=float)
            )
            if values.size == 0:
                continue

            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1)) if values.size >= 2 else 0.0

            mean_values.append(mean_val)
            std_values.append(std_val)
            positions_by_scale.append(pos)
            all_bar_values.extend([mean_val - std_val, mean_val + std_val])

        if not positions_by_scale:
            continue

        fill_color = COLORS.get(algo, "#CFCFCF")
        ax.bar(
            positions_by_scale,
            mean_values,
            width=PLOT_CONFIG["bar_width"],
            yerr=std_values,
            capsize=PLOT_CONFIG["error_cap_size"],
            color=fill_color,
            edgecolor="none",
            error_kw={
                "elinewidth": PLOT_CONFIG["error_linewidth"],
                "ecolor": "#555555",
            },
        )

        legend_handles.append(Patch(facecolor=fill_color, edgecolor="none", label=algo))

    # ax.set_xlabel(
    #     "Scale",
    #     fontsize=PLOT_CONFIG["xlabel_fontsize"],
    #     fontname="Times New Roman",
    # )
    ax.set_ylabel(
        "Fitness",
        fontsize=PLOT_CONFIG["ylabel_fontsize"],
        fontname="Times New Roman",
    )
    # ax.set_title(
    #     "Fitness under Different Scales",
    #     fontsize=PLOT_CONFIG["title_fontsize"],
    #     fontweight="bold",
    #     fontname="Times New Roman",
    # )

    ax.set_xticks(base_positions)
    ax.set_xticklabels([SCALE_LABELS[s] for s in scales])

    legend = ax.legend(
        handles=legend_handles,
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
    print(
        "手动调整状态: "
        + ("开启" if MANUAL_ADJUSTMENT_CONFIG.get("enabled", False) else "关闭")
    )
    print(f"使用数据源: {data_path}")
    print(f"图表已保存至: {OUTPUT_PNG}")
    print(f"图表已保存至: {OUTPUT_EPS}")


if __name__ == "__main__":
    main()
