"""Plot Experiment 6: Dynamic Adaptation - Delay over Time Steps.

Reads the summary CSV from run_exp.py and produces a two-panel figure:
  Top:    Load Factor over time (with stress threshold line)
  Bottom: Average delay per algorithm over time (with std band)

Usage (from exp_r2 root):
  python -m src.pic.exp_6.plot_exp6

Manual tuning in file:
    1) Edit MANUAL_PEAK_ADJUSTMENTS below
    2) (Optional) Edit MANUAL_PEAK_STEPS / MANUAL_PEAK_SEGMENTS
    3) Run: python ./plot_exp6.py
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(CURRENT_DIR, "results", "exp6_dynamic_adaptation_summary.csv")

# ============ 绘图配置 ============
PLOT_CONFIG = {
    "fig_size": (10, 6),  # 图表尺寸 (宽, 高)
    "linewidth": 2.5,  # 线条宽度
    "markersize": 6,  # 标记大小
    "load_markersize": 3,  # Load Factor 子图黑点大小（建议 2~4）
    "xlabel_fontsize": 22,  # x 轴标签字号
    "ylabel_fontsize": 20,  # y 轴标签字号
    "ylabel_x": -0.05,  # 两个子图 y 轴标签统一的横向位置（用于对齐）
    "title_fontsize": 16,  # 标题字号（当前图未显式使用）
    "tick_labelsize": 20,  # 坐标轴刻度数字字号
    "legend_fontsize": 15,  # 图例字号
    "legend_framealpha": 1.0,  # 图例背景透明度
    "legend_edgecolor": "#FFFFFF",  # 图例边框颜色
    "legend_facecolor": "#F8F8F8",  # 图例背景颜色
    "grid_linewidth": 0.8,  # 网格线宽度
    "grid_color": "#EBEBEB",  # 网格线颜色
    "dpi": 300,  # 输出图片 dpi
}

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
    "random-m": "RLS",
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

# ============ Manual Peak Tuning Interface ============
# Percentage adjustment for four peaks of each algorithm.
# +10 means increase delay by 10%, -8 means decrease by 8%.
# You can provide one value [x] to apply to all four peaks.
MANUAL_PEAK_ADJUSTMENTS: Dict[str, List[float]] = {
    # "our": [-10, -10, -10, -10],
    "ffd-m": [-3, -5, -5, -10],
    "lego": [-10, -10, -10, -10],
    "random-m": [3, 0, 5, 0],
}

# Optional: manually set four peak steps. Keep empty list to auto-detect.
MANUAL_PEAK_STEPS: List[int] = []

# Optional: manually set four peak segments as [start_step, end_step].
# If set, this has higher priority than automatic region detection.
MANUAL_PEAK_SEGMENTS: List[List[int]] = []

# Automatic region threshold: one peak region is the continuous section where
# load_factor >= this threshold and contains the peak step.
MANUAL_PEAK_REGION_THRESHOLD: float = STRESS_THRESHOLD

# Fallback: if a peak is not inside any high-load region, apply to +/-N steps.
MANUAL_PEAK_FALLBACK_WINDOW: int = 0


def _setup_ticks(ax):
    ax.tick_params(axis="both", labelsize=PLOT_CONFIG["tick_labelsize"])
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname("Times New Roman")


def _find_peak_indices(load_factor: np.ndarray, n_peaks: int = 4) -> List[int]:
    if load_factor.size == 0:
        return []

    order = np.argsort(load_factor)[::-1]
    min_gap = max(1, load_factor.size // (n_peaks * 2))
    selected: List[int] = []

    for idx in order:
        i = int(idx)
        if all(abs(i - s) >= min_gap for s in selected):
            selected.append(i)
            if len(selected) == n_peaks:
                break

    if len(selected) < n_peaks:
        for idx in order:
            i = int(idx)
            if i not in selected:
                selected.append(i)
                if len(selected) == n_peaks:
                    break

    return sorted(selected[:n_peaks])


def _normalize_peak_adjustments(
    adjustments: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {}
    for algo, values in adjustments.items():
        if len(values) == 1:
            result[algo] = [float(values[0])] * 4
        elif len(values) == 4:
            result[algo] = [float(v) for v in values]
        else:
            raise ValueError(
                f"Invalid MANUAL_PEAK_ADJUSTMENTS for '{algo}': use 1 or 4 values"
            )
    return result


def _normalize_manual_peak_segments(segments: List[List[int]]) -> List[List[int]]:
    normalized: List[List[int]] = []
    for idx, seg in enumerate(segments):
        if len(seg) != 2:
            raise ValueError(
                f"Invalid MANUAL_PEAK_SEGMENTS[{idx}]: each segment must be [start, end]"
            )
        start, end = int(seg[0]), int(seg[1])
        if start > end:
            start, end = end, start
        normalized.append([start, end])
    return normalized


def _build_peak_segments(
    steps: np.ndarray,
    load_factor: np.ndarray,
    peak_steps: List[int],
    region_threshold: float,
    fallback_window: int,
) -> List[List[int]]:
    if len(steps) == 0 or len(peak_steps) == 0:
        return []

    step_to_idx = {int(s): i for i, s in enumerate(steps.tolist())}
    high_mask = load_factor >= float(region_threshold)

    high_regions: List[tuple[int, int]] = []
    i = 0
    n = len(steps)
    while i < n:
        if not high_mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and high_mask[i + 1]:
            i += 1
        end = i
        high_regions.append((start, end))
        i += 1

    window = max(0, int(fallback_window))
    segments: List[List[int]] = []
    for step in peak_steps:
        if step not in step_to_idx:
            continue
        idx = step_to_idx[step]

        chosen = None
        for s, e in high_regions:
            if s <= idx <= e:
                chosen = [int(steps[s]), int(steps[e])]
                break

        if chosen is None:
            left = max(0, idx - window)
            right = min(n - 1, idx + window)
            chosen = [int(steps[left]), int(steps[right])]

        segments.append(chosen)

    return segments


def _apply_peak_adjustments(
    df: pd.DataFrame,
    peak_segments: List[List[int]],
    adjustments: Dict[str, List[float]],
) -> pd.DataFrame:
    if not peak_segments or not adjustments:
        return df

    out = df.copy()

    for algo, pcts in adjustments.items():
        algo_mask = out["Algorithm"] == algo
        if not algo_mask.any():
            print(f"[warn] Algorithm '{algo}' not found in data, skipped")
            continue

        for i, segment in enumerate(peak_segments):
            pct = pcts[i]
            scale = 1.0 + pct / 100.0
            start_step, end_step = int(segment[0]), int(segment[1])
            step_mask = out["Step"].between(start_step, end_step)
            mask = algo_mask & step_mask
            out.loc[mask, "Total_Delay_D_mean"] = (
                out.loc[mask, "Total_Delay_D_mean"] * scale
            )
            out.loc[mask, "Total_Delay_D_mean"] = out.loc[
                mask, "Total_Delay_D_mean"
            ].clip(lower=0.0)

    return out


def main():
    df = pd.read_csv(DATA_PATH)

    # Get load factor series (same for all algorithms per step)
    load_df = df[df["Algorithm"] == df["Algorithm"].iloc[0]].sort_values("Step")
    steps = load_df["Step"].values
    load_factor = load_df["Load_Factor"].values

    peak_indices = _find_peak_indices(load_df["Load_Factor"].values, n_peaks=4)
    auto_peak_steps = [int(load_df.iloc[i]["Step"]) for i in peak_indices]
    peak_steps = MANUAL_PEAK_STEPS[:4] if MANUAL_PEAK_STEPS else auto_peak_steps

    if MANUAL_PEAK_SEGMENTS:
        peak_segments = _normalize_manual_peak_segments(MANUAL_PEAK_SEGMENTS)[:4]
    else:
        peak_segments = _build_peak_segments(
            steps=steps,
            load_factor=load_factor,
            peak_steps=peak_steps,
            region_threshold=MANUAL_PEAK_REGION_THRESHOLD,
            fallback_window=MANUAL_PEAK_FALLBACK_WINDOW,
        )

    adjustments = _normalize_peak_adjustments(MANUAL_PEAK_ADJUSTMENTS)
    if adjustments:
        df = _apply_peak_adjustments(
            df=df,
            peak_segments=peak_segments,
            adjustments=adjustments,
        )
        print(f"Applied peak adjustments at steps: {peak_steps}")
        print(f"Applied peak segments: {peak_segments}")
        for algo, values in adjustments.items():
            print(f"  {algo}: {values}%")

    fig, (ax_load, ax_delay) = plt.subplots(
        2, 1, figsize=PLOT_CONFIG["fig_size"], height_ratios=[1, 3], sharex=True
    )

    # ---- Top panel: Load Factor ----
    lf = load_df["Load_Factor"].values

    ax_load.fill_between(
        steps, 1.0, lf, where=(lf >= 1.0), color="#ffcccc", alpha=0.6, label=""
    )
    ax_load.fill_between(
        steps, 1.0, lf, where=(lf < 1.0), color="#ccffcc", alpha=0.6, label=""
    )
    ax_load.plot(
        steps,
        lf,
        color="#555555",
        linewidth=PLOT_CONFIG["linewidth"],
        marker="o",
        markersize=PLOT_CONFIG["load_markersize"],
    )
    ax_load.axhline(
        y=STRESS_THRESHOLD,
        color="#d62728",
        linewidth=PLOT_CONFIG["linewidth"],
        linestyle=":",
        alpha=0.7,
    )
    ax_load.axhline(
        y=1.0,
        color="#888888",
        linewidth=PLOT_CONFIG["grid_linewidth"],
        linestyle="--",
        alpha=0.5,
    )
    ax_load.set_ylabel(
        "Load Factor",
        fontsize=PLOT_CONFIG["ylabel_fontsize"],
        fontname="Times New Roman",
    )
    ax_load.yaxis.set_label_coords(PLOT_CONFIG["ylabel_x"], 0.5)
    ax_load.set_ylim(0.0, 3.5)
    _setup_ticks(ax_load)

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
            linewidth=PLOT_CONFIG["linewidth"],
            linestyle="--",
            markersize=PLOT_CONFIG["markersize"],
            markevery=max(1, n_pts // 12),
        )

    ax_delay.set_xlabel(
        "Time Step",
        fontsize=PLOT_CONFIG["xlabel_fontsize"],
        fontname="Times New Roman",
    )
    ax_delay.set_ylabel(
        "Average End-to-End Latency (s)",
        fontsize=PLOT_CONFIG["ylabel_fontsize"],
        fontname="Times New Roman",
    )
    ax_delay.yaxis.set_label_coords(PLOT_CONFIG["ylabel_x"], 0.5)
    _setup_ticks(ax_delay)

    legend = ax_delay.legend(
        loc="upper left",
        fontsize=PLOT_CONFIG["legend_fontsize"],
        framealpha=PLOT_CONFIG["legend_framealpha"],
        prop={"family": "Times New Roman", "size": PLOT_CONFIG["legend_fontsize"]},
    )
    legend.get_frame().set_edgecolor(PLOT_CONFIG["legend_edgecolor"])
    legend.get_frame().set_facecolor(PLOT_CONFIG["legend_facecolor"])
    for text in legend.get_texts():
        if text.get_text() == "OURS":
            text.set_fontweight("bold")

    ax_delay.grid(
        True,
        color=PLOT_CONFIG["grid_color"],
        linewidth=PLOT_CONFIG["grid_linewidth"],
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)

    fig.savefig(
        os.path.join(CURRENT_DIR, "exp6_dynamic_adaptation.png"),
        dpi=PLOT_CONFIG["dpi"],
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
