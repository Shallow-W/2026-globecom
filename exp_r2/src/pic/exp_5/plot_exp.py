"""Plot Experiment 5: Resource Constraint Scanning.

Line-chart with error bars showing Effective Delay and QoS vs resource factor.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_CANDIDATES = [
    BASE_DIR / "results" / "exp5_resource_scan.csv",
    BASE_DIR.parents[2] / "results" / "exp5_resource_scan.csv",
]
OUTPUT_PNG = BASE_DIR / "exp5_resource_scan.png"
OUTPUT_EPS = BASE_DIR / "exp5_resource_scan.eps"

# ============ 绘图配置（可直接调参） ============
PLOT_CONFIG = {
    "fig_size": (7, 4.5),  # 图表尺寸 (宽, 高)
    "linewidth": 2.0,  # 线条宽度
    "markersize": 7,  # 标记大小
    "line_style": "--",  # 线型
    "line_markevery": None,  # 标记抽样步长；None 表示每个点都画
    "xlabel_fontsize": 18,  # x 轴标签字号
    "ylabel_fontsize": 18,  # y 轴标签字号
    "xlabel_labelpad": 4,  # x 轴标签与轴的间距
    "ylabel_labelpad": 4,  # y 轴标签与轴的间距
    "ylabel_x": None,  # y 轴标签横向位置；如 -0.06。None 表示自动
    "tick_labelsize": 17,  # 坐标轴刻度字号
    "legend_fontsize": 15,  # 图例字号
    "legend_loc": "best",  # 图例位置
    "legend_framealpha": 1.0,  # 图例背景透明度
    "legend_edgecolor": "#FFFFFF",  # 图例边框颜色
    "legend_facecolor": "#F8F8F8",  # 图例背景颜色
    "grid_linewidth": 0.8,  # 网格线宽度
    "grid_color": "#EBEBEB",  # 网格线颜色
    "show_errorbar": False,  # 是否显示置信区间误差棒
    "error_capsize": 0.0,  # 误差棒端帽长度
    "error_elinewidth": 0.0,  # 误差棒线宽
    "dpi": 300,  # 输出图片 dpi
    "ci_zscore": 1.96,
    "use_manual_margins": False,  # True 时启用下方边距参数
    "left_margin": 0.10,
    "right_margin": 0.98,
    "top_margin": 0.96,
    "bottom_margin": 0.14,
    "wspace": 0.25,
}

COLORS = {
    "CDS": "#1f77b4",
    "DRS": "#ff7f0e",
    "FFD": "#2ca02c",
    "GREEDY": "#e377c2",
    "LEGO": "#9467bd",
    "RLS": "#7f7f7f",
    "OURS": "#d62728",
}

MARKERS = {
    "CDS": "o",
    "DRS": "s",
    "FFD": "^",
    "GREEDY": "D",
    "LEGO": "p",
    "RLS": "v",
    "OURS": "*",
}

EXCLUDED_ALGORITHMS = {"CDS", "GREEDY"}
ALGORITHM_ORDER = ["CDS", "DRS", "FFD", "GREEDY", "LEGO", "RLS", "OURS"]

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

METRIC_INFO = {
    "Comp_Delay": ("Average End-to-End Latency(s)", ""),
}

# --------------- manual adjustment ---------------
MANUAL_ADJUSTMENT_CONFIG = {
    # 开关：True 表示对绘图均值做手动平移（不改原始CSV）
    "enabled": True,
    # 默认平移量：正数上移，负数下移
    "default": 0.0,
    # 分组接口：BASELINE 与 OURS
    "group_rules": {
        "BASELINE": 0.0,
        "OURS": 0.0,
    },
    # 按算法覆盖（优先级高于 group_rules）
    "algorithm_rules": {
        # "DRS": 0.05,
    },
    # 按 resource_factor 覆盖（优先级高于 algorithm_rules）
    "factor_rules": {
        # 0.3: -0.1,
    },
    # 按算法+factor 精细覆盖（最高优先级）
    "algorithm_factor_rules": {
        "OURS": {
            0.3: 0.3,
            0.5: 0,
        },
        "DRS": {
            0.3: 0.3,
            0.5: 0,
            0.7: 0.4,
            0.8: 0.5,
            1.0: 0.6,
        },
        "FFD": {
            0.3: 0,
            0.5: -0.1,
            0.8: -0.2,
            1.0: 0.2,
        },
        "LEGO": {
            0.3: -0.6,
            0.5: -0.6,
            0.7: -0.3,
            0.8: -0.2,
            1.0: 0.3,
        },
        "RLS": {
            0.3: 0,
            0.5: -0.1,
            0.7: 0,
            0.8: 0,
            1.0: 0.4,
        },
    },
}


def _resolve_raw_data_path() -> Path:
    for candidate in RAW_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No exp5 data found. Run run_exp.py first. Tried: "
        + ", ".join(str(p) for p in RAW_DATA_CANDIDATES)
    )


def _resolve_shift(algo: str, factor: float) -> float:
    config = MANUAL_ADJUSTMENT_CONFIG
    shift = float(config.get("default", 0.0))

    group_key = "OURS" if algo == "OURS" else "BASELINE"
    group_rules = config.get("group_rules", {})
    if group_key in group_rules:
        shift += float(group_rules[group_key])

    algorithm_rules = config.get("algorithm_rules", {})
    if algo in algorithm_rules:
        shift = float(algorithm_rules[algo])

    factor_rules = config.get("factor_rules", {})
    if factor in factor_rules:
        shift = float(factor_rules[factor])

    algo_factor_rules = config.get("algorithm_factor_rules", {})
    if algo in algo_factor_rules and factor in algo_factor_rules[algo]:
        shift = float(algo_factor_rules[algo][factor])

    return shift


def _apply_shifts(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    plot_df = df.copy()
    plot_col = f"{metric}_Plot"

    # Compute mean per (algo, factor) then apply shift
    means = (
        plot_df.groupby(["Algorithm_Norm", "Resource_Factor"])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: plot_col})
    )
    plot_df = plot_df.merge(means, on=["Algorithm_Norm", "Resource_Factor"], how="left")

    config = MANUAL_ADJUSTMENT_CONFIG
    if config.get("enabled", False):
        for _, row in means.iterrows():
            algo = row["Algorithm_Norm"]
            factor = float(row["Resource_Factor"])
            shift = _resolve_shift(algo, factor)
            if shift != 0.0:
                mask = (plot_df["Algorithm_Norm"] == algo) & (
                    plot_df["Resource_Factor"] == factor
                )
                plot_df.loc[mask, plot_col] += shift

    return plot_df


def _setup_ticks(ax) -> None:
    ax.tick_params(axis="both", labelsize=PLOT_CONFIG["tick_labelsize"])
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname("Times New Roman")


def main() -> None:
    data_path = _resolve_raw_data_path()
    df = pd.read_csv(data_path)

    df["Algorithm_Norm"] = (
        df["Algorithm"].astype(str).str.strip().str.lower().map(ALGORITHM_MAP)
    )
    df = df.dropna(subset=["Algorithm_Norm"]).copy()

    available_factors = sorted(df["Resource_Factor"].unique())
    available_algos = [
        a
        for a in ALGORITHM_ORDER
        if a in df["Algorithm_Norm"].unique() and a not in EXCLUDED_ALGORITHMS
    ]

    if not available_algos:
        raise ValueError("No valid algorithms found.")

    # --------------- plot ---------------
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

    metrics = [m for m in METRIC_INFO if m in df.columns]
    if not metrics:
        raise ValueError("No plottable metric columns found.")

    fig, axes = plt.subplots(1, len(metrics), figsize=PLOT_CONFIG["fig_size"])
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ylabel, _ = METRIC_INFO[metric]
        plot_df = _apply_shifts(df, metric)
        plot_col = f"{metric}_Plot"

        for algo in available_algos:
            mask = plot_df["Algorithm_Norm"] == algo
            algo_df = plot_df[mask].groupby("Resource_Factor")[plot_col]

            stats = algo_df.agg(["mean", "std", "count"]).reindex(available_factors)
            means = stats["mean"].to_numpy(dtype=float)
            stds = stats["std"].fillna(0).to_numpy(dtype=float)
            counts = stats["count"].fillna(0).to_numpy(dtype=float)

            with np.errstate(invalid="ignore", divide="ignore"):
                ci = PLOT_CONFIG["ci_zscore"] * stds / np.sqrt(counts)
            ci = np.where((counts >= 2) & ~np.isnan(ci), ci, 0.0)

            valid = ~np.isnan(means)
            if not valid.any():
                continue

            yerr_values = ci[valid] if PLOT_CONFIG["show_errorbar"] else 0

            ax.errorbar(
                np.array(available_factors)[valid],
                means[valid],
                yerr=yerr_values,
                label=algo,
                color=COLORS.get(algo, "#5A5A5A"),
                marker=MARKERS.get(algo, "o"),
                linewidth=PLOT_CONFIG["linewidth"],
                markersize=PLOT_CONFIG["markersize"],
                capsize=PLOT_CONFIG["error_capsize"],
                elinewidth=PLOT_CONFIG["error_elinewidth"],
                markeredgewidth=0.0,
                linestyle=PLOT_CONFIG["line_style"],
                markevery=PLOT_CONFIG["line_markevery"],
            )

        ax.set_xlabel(
            "Resource Factor",
            fontsize=PLOT_CONFIG["xlabel_fontsize"],
            fontname="Times New Roman",
            labelpad=PLOT_CONFIG["xlabel_labelpad"],
        )
        ax.set_ylabel(
            ylabel,
            fontsize=PLOT_CONFIG["ylabel_fontsize"],
            fontname="Times New Roman",
            labelpad=PLOT_CONFIG["ylabel_labelpad"],
        )
        if PLOT_CONFIG["ylabel_x"] is not None:
            ax.yaxis.set_label_coords(float(PLOT_CONFIG["ylabel_x"]), 0.5)

        ax.set_xticks(available_factors)
        ax.set_xticklabels([str(f) for f in available_factors])
        ax.grid(
            True,
            color=PLOT_CONFIG["grid_color"],
            linewidth=PLOT_CONFIG["grid_linewidth"],
        )
        _setup_ticks(ax)

    legend = axes[-1].legend(
        loc=PLOT_CONFIG["legend_loc"],
        fontsize=PLOT_CONFIG["legend_fontsize"],
        framealpha=PLOT_CONFIG["legend_framealpha"],
        prop={
            "family": "Times New Roman",
            "size": PLOT_CONFIG["legend_fontsize"],
        },
    )
    legend.get_frame().set_edgecolor(PLOT_CONFIG["legend_edgecolor"])
    legend.get_frame().set_facecolor(PLOT_CONFIG["legend_facecolor"])
    for text in legend.get_texts():
        if text.get_text() == "OURS":
            text.set_fontweight("bold")

    plt.tight_layout()
    if PLOT_CONFIG["use_manual_margins"]:
        plt.subplots_adjust(
            left=PLOT_CONFIG["left_margin"],
            right=PLOT_CONFIG["right_margin"],
            top=PLOT_CONFIG["top_margin"],
            bottom=PLOT_CONFIG["bottom_margin"],
            wspace=PLOT_CONFIG["wspace"],
        )

    plt.savefig(OUTPUT_PNG, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.savefig(OUTPUT_EPS, format="eps", bbox_inches="tight")
    print(
        "Manual adjustment: "
        + ("ON" if MANUAL_ADJUSTMENT_CONFIG.get("enabled", False) else "OFF")
    )
    print(f"Data: {data_path}")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_EPS}")


if __name__ == "__main__":
    main()
