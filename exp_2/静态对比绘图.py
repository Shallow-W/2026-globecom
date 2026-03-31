import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 全局设置：字体及字号，全部使用英文
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 18

algorithms = {
    "Our": "静态时延能效.csv",
    "THREE-STAGE": "胡毅静态.csv",
    "DA-RSPPO": "王良源静态.csv",
    "Greedy": "贪心静态.csv",
    "GMDA-RMPR": "启发静态.csv"
}

experiments = {
    '不同服务类型数': ('Comparison of Service Type Number', 'Number of Service Types'),
    '不同服务链长度': ('Comparison of Service Chain Length', 'Service Chain Length'),
    '不同请求到达率': ('Comparison of Request Arrival Rate', 'Request Arrival Rate')
}

# 采用 Nature 期刊经典配色 (ggsci NPG palette)
colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
markers = ['*', 'o', 's', '^', 'D']
lines = ['-', '--', '-.', ':', '-']

avg_data = []

# ==========================================
# 1. 拆分绘制折线图 (共 6 张)
# ==========================================
for exp_name, (title_base, xlabel) in experiments.items():
    # 拆分：创建两个独立的Figure，一个画Delay，一个画QoS
    fig_delay, ax_delay = plt.subplots(figsize=(8, 6))
    fig_qos, ax_qos = plt.subplots(figsize=(8, 6))

    for idx, (algo, file_path) in enumerate(algorithms.items()):
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            df_exp = df[df['Experiment'] == exp_name].copy()
            df_exp.sort_values(by='Variable_Value', inplace=True)

            if not df_exp.empty:
                x = df_exp['Variable_Value']

                # 单独绘制时延
                ax_delay.plot(x, df_exp['Total_Delay_D'], label=algo, color=colors[idx],
                              marker=markers[idx], linestyle=lines[idx], linewidth=2.5, markersize=8)

                # 单独绘制效能
                ax_qos.plot(x, df_exp['Avg_QoS_Q'], label=algo, color=colors[idx],
                            marker=markers[idx], linestyle=lines[idx], linewidth=2.5, markersize=8)

                # 收集当前算法在当前实验下全部自变量的平均值数据，用于后续柱状图
                avg_data.append({
                    'Experiment': exp_name,
                    'Algorithm': algo,
                    'Avg_Delay': df_exp['Total_Delay_D'].mean(),
                    'Avg_QoS': df_exp['Avg_QoS_Q'].mean()
                })
        else:
            pass # 避免过多打印

    # ========== 格式化并保存时延折线图 ==========
    ax_delay.set_xlabel(xlabel, fontweight='bold')
    ax_delay.set_ylabel('Total Delay (s)', fontweight='bold')
    ax_delay.set_title(f'{title_base}\nTotal Delay', fontweight='bold')
    ax_delay.grid(True, linestyle='--', alpha=0.6)
    ax_delay.legend(loc='upper right', framealpha=0.85)
    fig_delay.tight_layout()
    fig_delay.savefig(f'静态{exp_name}时延对比.png', dpi=300)
    plt.close(fig_delay)

    # ========== 格式化并保存效能折线图 ==========
    ax_qos.set_xlabel(xlabel, fontweight='bold')
    ax_qos.set_ylabel('Average QoS', fontweight='bold')
    ax_qos.set_title(f'{title_base}\nAverage QoS', fontweight='bold')
    ax_qos.grid(True, linestyle='--', alpha=0.6)
    ax_qos.legend(loc='upper right', framealpha=0.85)
    fig_qos.tight_layout()
    fig_qos.savefig(f'静态{exp_name}效能对比.png', dpi=300)
    plt.close(fig_qos)

# ==========================================
# 2. 绘制独立的总体平均值对比柱状图 (共 2 张)
# ==========================================
df_avg = pd.DataFrame(avg_data)

exp_keys = list(experiments.keys())
x_pos = np.arange(len(exp_keys))
width = 0.15  # 柱子宽度

# 准备数据字典
delay_data_dict = {algo: [] for algo in algorithms.keys()}
qos_data_dict = {algo: [] for algo in algorithms.keys()}

for i, algo in enumerate(algorithms.keys()):
    for exp in exp_keys:
        if not df_avg.empty:
            row = df_avg[(df_avg['Experiment'] == exp) & (df_avg['Algorithm'] == algo)]
            if not row.empty:
                delay_data_dict[algo].append(row['Avg_Delay'].values[0])
                qos_data_dict[algo].append(row['Avg_QoS'].values[0])
            else:
                delay_data_dict[algo].append(0)
                qos_data_dict[algo].append(0)
        else:
            delay_data_dict[algo].append(0)
            qos_data_dict[algo].append(0)

# ================= 独立时延柱状图 =================
fig_bar_delay, ax_bar_delay = plt.subplots(figsize=(8, 6))

for i, algo in enumerate(algorithms.keys()):
    offset = (i - len(algorithms) / 2 + 0.5) * width
    ax_bar_delay.bar(x_pos + offset, delay_data_dict[algo], width, label=algo, color=colors[i], edgecolor='black', linewidth=1.2, zorder=3)

ax_bar_delay.set_xticks(x_pos)
ax_bar_delay.set_xticklabels(['Service Types', 'Chain Length', 'Arrival Rate'], fontweight='bold')
ax_bar_delay.set_ylabel('Average Total Delay (s)', fontweight='bold')
ax_bar_delay.set_title('Overall Average Delay Comparison', fontweight='bold')
ax_bar_delay.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
ax_bar_delay.legend(loc='best', framealpha=0.85)

fig_bar_delay.tight_layout()
fig_bar_delay.savefig('静态总体平均时延柱状图.png', dpi=300)
plt.close(fig_bar_delay)

# ================= 独立效能柱状图 =================
fig_bar_qos, ax_bar_qos = plt.subplots(figsize=(8, 6))

for i, algo in enumerate(algorithms.keys()):
    offset = (i - len(algorithms) / 2 + 0.5) * width
    ax_bar_qos.bar(x_pos + offset, qos_data_dict[algo], width, label=algo, color=colors[i], edgecolor='black', linewidth=1.2, zorder=3)

ax_bar_qos.set_xticks(x_pos)
ax_bar_qos.set_xticklabels(['Service Types', 'Chain Length', 'Arrival Rate'], fontweight='bold')
ax_bar_qos.set_ylabel('Average QoS', fontweight='bold')
ax_bar_qos.set_title('Overall Average QoS Comparison', fontweight='bold')
ax_bar_qos.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
ax_bar_qos.legend(loc='best', framealpha=0.85)

fig_bar_qos.tight_layout()
fig_bar_qos.savefig('静态总体平均效能柱状图.png', dpi=300)
plt.close(fig_bar_qos)

print("拆分折线图(6张)及分离的平均值对比柱状图(2张)已全部生成！共计8张图。")