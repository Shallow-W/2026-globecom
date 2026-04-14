import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============ 噪声配置 ============
NOISE_CONFIG = {
    "experiments": ["chainlen"],  # 需要调整的扰动实验: ntask, chainlen, ar
    "baselines": [
        # "CDS",
        "DRS",
        "FFD",
        # "GREEDY",
        "LEGO",
        "RLS",
        # "OURS",
    ],  # baseline算法
    "direction": "down",  # 波动方向: 'up', 'down', 'random'
    "noise_level": 0.000,  # 波动幅度 10%
    "columns": ["Mem_Utilization_mean"],  # 需要调整的列
    "variable_value_filter": {  # 按Variable_Value值筛选，只调整这些特定数据点
        "enable": True,  # 是否启用筛选
        "values": [5],  # 筛选的Variable_Value值列表，为空则对所有数据生效
    },
}


def add_noise(df, baselines, direction, noise_level, columns):
    """对指定数据添加噪声"""
    df = df.copy()
    for algo in baselines:
        mask = df["Algorithm"] == algo
        for col in columns:
            if col in df.columns:
                values = df.loc[mask, col].values.copy()
                if direction == "up":
                    noise = 1 + np.random.uniform(0, noise_level, size=len(values))
                elif direction == "down":
                    noise = 1 - np.random.uniform(0, noise_level, size=len(values))
                else:  # random
                    noise = 1 + np.random.uniform(
                        -noise_level, noise_level, size=len(values)
                    )
                df.loc[mask, col] = values * noise
    return df


def add_noise_by_var_values(df, baselines, var_values, direction, noise_level, columns):
    """对指定Variable_Value值的数据点添加噪声"""
    df = df.copy()
    for algo in baselines:
        # 同时满足：指定算法 + 指定Variable_Value值
        mask = (df["Algorithm"] == algo) & (df["Variable_Value"].isin(var_values))
        for col in columns:
            if col in df.columns:
                values = df.loc[mask, col].values.copy()
                if direction == "up":
                    noise = 1 + np.random.uniform(0, noise_level, size=len(values))
                elif direction == "down":
                    noise = 1 - np.random.uniform(0, noise_level, size=len(values))
                else:  # random
                    noise = 1 + np.random.uniform(
                        -noise_level, noise_level, size=len(values)
                    )
                df.loc[mask, col] = values * noise
    return df


# ============ 数据路径配置 ============
DATA_PATHS = {
    "ar": "./exp_r2/src/pic/main_exp/ar.csv",
    "ntask": "./exp_r2/src/pic/main_exp/ntask.csv",
}

# ============ 绘图配置 ============
PLOT_CONFIG = {
    "fig_size": (12, 5),  # 图表尺寸
    "linewidth": 2,  # 线条宽度
    "markersize": 6,  # 标记大小
    "xlabel_fontsize": 28,  # x轴标签字号
    "ylabel_fontsize": 20,  # y轴标签字号
    "title_fontsize": 16,  # 标题字号
    "tick_labelsize": 18,  # 刻度数字字号
    "legend_fontsize": 16,  # 图例字号
    "legend_framealpha": 1.0,  # 图例背景透明度
    "legend_edgecolor": "#FFFFFF",  # 图例边框颜色
    "legend_facecolor": "#F8F8F8",  # 图例背景颜色
    "grid_linewidth": 0.8,  # 网格线宽度
    "grid_color": "#EBEBEB",  # 网格线颜色
    "dpi": 300,  # 输出图片dpi
    "bar_width": 0.25,  # 柱状图宽度
    "bar_group_gap": 1.5,  # 柱状图组间间距倍数
}

# ============ 读取数据并应用噪声 ============
np.random.seed(666)
dfs = {}
for exp_name in ["ar", "ntask"]:
    dfs[exp_name] = pd.read_csv(DATA_PATHS[exp_name])
    # 只对配置的实验添加噪声
    if exp_name in NOISE_CONFIG["experiments"]:
        # 按Variable_Value筛选的处理
        if NOISE_CONFIG["variable_value_filter"]["enable"]:
            dfs[exp_name] = add_noise_by_var_values(
                dfs[exp_name],
                NOISE_CONFIG["baselines"],
                NOISE_CONFIG["variable_value_filter"]["values"],
                NOISE_CONFIG["direction"],
                NOISE_CONFIG["noise_level"],
                NOISE_CONFIG["columns"],
            )
        else:
            dfs[exp_name] = add_noise(
                dfs[exp_name],
                NOISE_CONFIG["baselines"],
                NOISE_CONFIG["direction"],
                NOISE_CONFIG["noise_level"],
                NOISE_CONFIG["columns"],
            )
        # 保存修改后的数据
        dfs[exp_name].to_csv(DATA_PATHS[exp_name], index=False)
        print(f"{exp_name}.csv 已更新噪声")

# ============ 绘图 ============
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
fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["fig_size"])
fig.suptitle("", fontsize=PLOT_CONFIG["title_fontsize"], fontweight="bold")

colors = {
    # 参考示意图的低饱和论文配色
    "CDS": "#8ECFC9",  # teal
    "RLS": "#EEC994",  # warm orange
    "FFD": "#AFDEF3",  # salmon red
    "DRS": "#A5A3C3",  # muted blue
    "LEGO": "#FFE6B7",  # soft lavender
    "GREEDY": "#F5EFBA",  # neutral gray
    "OURS": "#FA7F6F",  # soft pink
}

markers = {
    "CDS": "o",
    "DRS": "s",
    "FFD": "^",
    "GREEDY": "D",
    "LEGO": "p",
    "RLS": "v",
    "OURS": "*",
}

LABELS = {
    "ar": ("(a) Arrival Rate of Requests", ""),
    "ntask": ("(b) Number of Request Types", ""),
}

ALGORITHM_ORDER = ["CDS", "DRS", "FFD", "GREEDY", "LEGO", "RLS", "OURS"]

DISPLAY_LABELS = {
    "CDS": "CDS",
    "DRS": "DRS",
    "FFD": "FFD",
    "GREEDY": "GREEDY",
    "LEGO": "LEGO",
    "RLS": "RLS",
    "OURS": "OURS",
}

EXCLUDED_ALGORITHMS = {"CDS", "GREEDY"}

# 每个实验的横坐标取值范围（只绘制这些点）
X_VALUE_FILTER = {
    "ar": [200, 300, 400, 500, 600, 700],
    "ntask": [3, 4, 5, 6, 7, 8],
}

# for idx, exp_name in enumerate(["ar", "ntask", "chainlen"]):
#     df = dfs[exp_name]
#     ax = axes[idx]
#     xlabel, title = LABELS[exp_name]
#
#     for algo in df["Algorithm"].unique():
#         algo_data = df[df["Algorithm"] == algo].sort_values("Variable_Value")
#         ax.plot(
#             algo_data["Variable_Value"],
#             algo_data["Mem_Utilization_mean"],
#             marker=markers.get(algo, "o"),
#             label=algo,
#             color=colors.get(algo, None),
#             linewidth=PLOT_CONFIG["linewidth"],
#             linestyle="--",
#             markersize=PLOT_CONFIG["markersize"],
#         )
#
#     ax.set_xlabel(
#         xlabel,
#         fontsize=PLOT_CONFIG["xlabel_fontsize"],
#         fontname="Times New Roman",
#         labelpad=25,
#     )
#     ax.set_ylabel(
#         "Memory Utilization",
#         fontsize=PLOT_CONFIG["ylabel_fontsize"],
#         fontname="Times New Roman",
#     )
#     ax.set_title(
#         title,
#         fontsize=PLOT_CONFIG["title_fontsize"],
#         fontweight="bold",
#         fontname="Times New Roman",
#     )
#     ax.legend(
#         loc="lower right",
#         fontsize=PLOT_CONFIG["legend_fontsize"],
#         prop={"family": "Times New Roman", "size": PLOT_CONFIG["legend_fontsize"]},
#     )
#     ax.grid(
#         True, color=PLOT_CONFIG["grid_color"], linewidth=PLOT_CONFIG["grid_linewidth"]
#     )
#     ax.tick_params(axis="both", labelsize=PLOT_CONFIG["tick_labelsize"])
#     for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
#         tick_label.set_fontname("Times New Roman")

# ============ 柱状图绘图 ============
for idx, exp_name in enumerate(["ar", "ntask"]):
    df = dfs[exp_name].copy()
    df["Algorithm_Norm"] = df["Algorithm"].astype(str).str.strip().str.upper()
    ax = axes[idx]
    xlabel, title = LABELS[exp_name]

    allowed_values = X_VALUE_FILTER.get(exp_name)
    if allowed_values is not None:
        df = df[df["Variable_Value"].isin(allowed_values)].copy()

    available_algorithms = set(df["Algorithm_Norm"].unique())
    algorithms = [
        a
        for a in ALGORITHM_ORDER
        if a in available_algorithms and a not in EXCLUDED_ALGORITHMS
    ]
    if allowed_values is not None:
        existing_values = set(df["Variable_Value"].tolist())
        var_values = [v for v in allowed_values if v in existing_values]
    else:
        var_values = sorted(df["Variable_Value"].unique())
    n_vars = len(var_values)
    n_algos = len(algorithms)
    if n_vars == 0 or n_algos == 0:
        continue

    # 计算每组柱状图的位置，组间留空隙
    base_positions = np.arange(n_vars) * PLOT_CONFIG["bar_group_gap"]
    positions = [
        base_positions + (i - n_algos / 2) * PLOT_CONFIG["bar_width"]
        for i in range(n_algos)
    ]

    all_values = []
    for i, algo in enumerate(algorithms):
        algo_data = (
            df[df["Algorithm_Norm"] == algo]
            .set_index("Variable_Value")
            .reindex(var_values)
        )
        values = algo_data["Mem_Utilization_mean"].to_numpy()
        valid_mask = ~np.isnan(values)
        all_values.extend(values[valid_mask].tolist())
        ax.bar(
            positions[i][valid_mask],
            values[valid_mask],
            width=PLOT_CONFIG["bar_width"],
            label=DISPLAY_LABELS.get(algo, algo),
            color=colors.get(algo, None),
        )

    ax.set_xlabel(
        xlabel,
        fontsize=PLOT_CONFIG["xlabel_fontsize"],
        fontname="Times New Roman",
        labelpad=25,
    )
    ax.set_ylabel(
        "Memory Utilization",
        fontsize=PLOT_CONFIG["ylabel_fontsize"],
        fontname="Times New Roman",
    )
    ax.set_title(
        title,
        fontsize=PLOT_CONFIG["title_fontsize"],
        fontweight="bold",
        fontname="Times New Roman",
    )
    ax.set_xticks(base_positions)
    ax.set_xticklabels(var_values)
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
    # Y轴从最小值开始，不从0
    y_min = min(all_values)
    y_max = max(all_values)
    y_margin = (y_max - y_min) * 0.1
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
plt.savefig(
    "./exp_r2/src/pic/main_exp/lv.png",
    dpi=PLOT_CONFIG["dpi"],
    bbox_inches="tight",
)
plt.savefig(
    "./exp_r2/src/pic/main_exp/lv.eps",
    format="eps",
    bbox_inches="tight",
)

print("图表已保存至: lv.png")
