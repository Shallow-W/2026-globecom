import pandas as pd
import matplotlib.pyplot as plt

# 读取三个实验数据
df_ar = pd.read_csv("./ar.csv")
df_ntask = pd.read_csv("./ntask.csv")
df_chain = pd.read_csv("./chainlen.csv")

# 设置图表样式
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "Total Delay Under Different Perturbations", fontsize=16, fontweight="bold"
)

# 定义颜色
colors = {
    "cds-m": "#1f77b4",
    "drs": "#ff7f0e",
    "ffd-m": "#2ca02c",
    "greedy-m": "#d62728",
    "lego": "#9467bd",
    "our": "#e377c2",
    "random-m": "#7f7f7f",
}

# 获取所有算法
algorithms = df_ar["Algorithm"].unique()


def plot_total_delay(ax, df, xlabel, title):
    for algo in algorithms:
        algo_data = df[df["Algorithm"] == algo].sort_values("Variable_Value")
        ax.plot(
            algo_data["Variable_Value"],
            algo_data["Total_Delay_D_mean"],
            marker="o",
            label=algo,
            color=colors.get(algo, None),
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Total Delay (s)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


# 1. Arrival Rate 实验
plot_total_delay(axes[0], df_ar, "Arrival Rate", "Arrival Rate Perturbation")

# 2. N_task_types 实验
plot_total_delay(axes[1], df_ntask, "Number of Task Types", "Task Types Perturbation")

# 3. Chain Length 实验
plot_total_delay(axes[2], df_chain, "Chain Length", "Chain Length Perturbation")

plt.tight_layout()
plt.savefig("./all_perturbations.png", dpi=150, bbox_inches="tight")
print("图表已保存至: ./all_perturbations.png")
plt.show()