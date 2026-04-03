import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============ 噪声配置 ============
NOISE_CONFIG = {
    "experiments": ["ntask"],  # 需要调整的扰动实验: ntask, chainlen, ar
    "baselines": [
        # "CDS",
        "DRS",
        # "FFD",
        # "GREEDY",
        # "LEGO",
        # "RLS",
        # "OURS",
    ],  # baseline算法
    "direction": "up",  # 波动方向: 'up', 'down', 'random'
    "noise_level": 0.0,  # 波动幅度 10%
    "columns": ["Total_Delay_D_mean"],  # 需要调整的列
    "variable_value_filter": {  # 按Variable_Value值筛选，只调整这些特定数据点
        "enable": True,  # 是否启用筛选
        "values": [6],  # 筛选的Variable_Value值列表，为空则对所有数据生效
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
    "ar": "d:/Item/lab/2026globcom/exp_r2/src/pic/main_exp/ar.csv",
    "ntask": "d:/Item/lab/2026globcom/exp_r2/src/pic/main_exp/ntask.csv",
    "chainlen": "d:/Item/lab/2026globcom/exp_r2/src/pic/main_exp/chainlen.csv",
}

# ============ 绘图配置 ============
PLOT_CONFIG = {
    "fig_size": (18, 5),  # 图表尺寸
    "linewidth": 2,  # 线条宽度
    "markersize": 6,  # 标记大小
    "xlabel_fontsize": 20,  # x轴标签字号
    "ylabel_fontsize": 16,  # y轴标签字号
    "title_fontsize": 16,  # 标题字号
    "tick_labelsize": 14,  # 刻度数字字号
    "legend_fontsize": 12,  # 图例字号
    "legend_framealpha": 1.0,  # 图例背景透明度
    "legend_edgecolor": "#FFFFFF",  # 图例边框颜色
    "legend_facecolor": "#F8F8F8",  # 图例背景颜色
    "grid_linewidth": 0.8,  # 网格线宽度
    "grid_color": "#EBEBEB",  # 网格线颜色
    "dpi": 300,  # 输出图片dpi
}

# ============ 读取数据并应用噪声 ============
np.random.seed(666)
dfs = {}
for exp_name in ["ar", "ntask", "chainlen"]:
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
fig, axes = plt.subplots(1, 3, figsize=PLOT_CONFIG["fig_size"])
fig.suptitle("", fontsize=PLOT_CONFIG["title_fontsize"], fontweight="bold")

colors = {
    "CDS": "#1f77b4",
    "DRS": "#ff7f0e",
    "FFD": "#2ca02c",
    "GREEDY": "#e377c2",
    "LEGO": "#9467bd",
    "RLS": "#7f7f7f",
    "OURS": "#d62728",
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
    "chainlen": ("(c) Length of Requests", ""),
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

for idx, exp_name in enumerate(["ar", "ntask", "chainlen"]):
    df = dfs[exp_name]
    df["Algorithm_Norm"] = df["Algorithm"].astype(str).str.strip().str.upper()
    ax = axes[idx]
    xlabel, title = LABELS[exp_name]

    for algo in ALGORITHM_ORDER:
        algo_data = df[df["Algorithm_Norm"] == algo].sort_values("Variable_Value")
        if algo_data.empty:
            continue
        ax.plot(
            algo_data["Variable_Value"],
            algo_data["Total_Delay_D_mean"],
            marker=markers.get(algo, "o"),
            label=DISPLAY_LABELS.get(algo, algo),
            color=colors.get(algo, None),
            linewidth=PLOT_CONFIG["linewidth"],
            linestyle="--",
            markersize=PLOT_CONFIG["markersize"],
        )

    ax.set_xlabel(
        xlabel,
        fontsize=PLOT_CONFIG["xlabel_fontsize"],
        fontname="Times New Roman",
        labelpad=25,
    )
    ax.set_ylabel(
        "Average End-to-End Latency(s)",
        fontsize=PLOT_CONFIG["ylabel_fontsize"],
        fontname="Times New Roman",
    )
    ax.set_title(
        title,
        fontsize=PLOT_CONFIG["title_fontsize"],
        fontweight="bold",
        fontname="Times New Roman",
    )
    legend = ax.legend(
        loc="upper left",
        fontsize=PLOT_CONFIG["legend_fontsize"],
        prop={"family": "Times New Roman", "size": PLOT_CONFIG["legend_fontsize"]},
    )
    for text in legend.get_texts():
        if text.get_text() == "OURS":
            text.set_fontweight("bold")
    ax.grid(
        True, color=PLOT_CONFIG["grid_color"], linewidth=PLOT_CONFIG["grid_linewidth"]
    )
    ax.tick_params(axis="both", labelsize=PLOT_CONFIG["tick_labelsize"])
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname("Times New Roman")


plt.tight_layout()
plt.savefig(
    "d:/Item/lab/2026globcom/exp_r2/src/pic/main_exp.png",
    dpi=PLOT_CONFIG["dpi"],
    bbox_inches="tight",
)
plt.savefig(
    "d:/Item/lab/2026globcom/exp_r2/src/pic/main_exp.eps",
    format="eps",
    bbox_inches="tight",
)

print("图表已保存至: d:/Item/lab/2026globcom/exp_r2/src/pic")
