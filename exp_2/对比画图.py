import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import os

# 设置中文字体，防止图表乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_dynamic_robustness():
    # 定义要读取的文件和对应的标签及颜色、样式
    configs = [
        # ================= PD-HEA 算法 (我们的) =================
        {'file': 'Dynamic_Results_小规模.csv', 'label': 'PD-HEA (小)', 'color': '#1f77b4', 'marker': 'o',
         'linestyle': '-', 'algo': 'pdhea', 'fill': True},
        {'file': 'Dynamic_Results_中等规模.csv', 'label': 'PD-HEA (中)', 'color': '#ff7f0e', 'marker': 's',
         'linestyle': '-', 'algo': 'pdhea', 'fill': True},
        {'file': 'Dynamic_Results_大规模.csv', 'label': 'PD-HEA (大)', 'color': '#2ca02c', 'marker': '^',
         'linestyle': '-', 'algo': 'pdhea', 'fill': True},

        # ================= Baseline 算法 (对比算法) =================
        {'file': '胡毅动态小规模.csv', 'label': 'Baseline (小)', 'color': '#1f77b4', 'marker': 'o', 'linestyle': '--',
         'algo': 'baseline', 'fill': False},
        {'file': '胡毅动态中规模.csv', 'label': 'Baseline (中)', 'color': '#ff7f0e', 'marker': 's', 'linestyle': '--',
         'algo': 'baseline', 'fill': False},
        {'file': '胡毅动态大规模.csv', 'label': 'Baseline (大)', 'color': '#2ca02c', 'marker': '^', 'linestyle': '--',
         'algo': 'baseline', 'fill': False}
    ]

    valid_configs = []
    # 读取并验证数据
    for cfg in configs:
        if os.path.exists(cfg['file']):
            cfg['data'] = pd.read_csv(cfg['file'])
            valid_configs.append(cfg)
        else:
            print(f"警告: 找不到文件 [{cfg['file']}]，将跳过绘制该曲线。")

    if not valid_configs:
        print("错误: 没有找到任何数据文件，无法绘图！")
        return

    # 创建画布 (包含上下两个子图，稍微加宽以容纳右侧图例)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    plt.subplots_adjust(hspace=0.15)

    # ---------------------------------------------------------
    # 子图 1: 平均端到端时延对比 (Avg Delay)
    # ---------------------------------------------------------
    for cfg in valid_configs:
        df = cfg['data']
        steps = df['Time_Step']
        delays = df['Avg_Delay(s)']

        # 根据配置决定 marker 是否实心
        mfc = cfg['color'] if cfg['fill'] else 'none'

        # 绘制时延主折线
        ax1.plot(steps, delays, label=cfg['label'], color=cfg['color'],
                 marker=cfg['marker'], markerfacecolor=mfc, markersize=5,
                 linestyle=cfg['linestyle'], linewidth=1.8, alpha=0.85)

        # 找出发生拥塞惩罚的点，差异化标记
        penalties = df['Penalty(0=Safe)']
        congested_mask = penalties > 0
        congested_steps = steps[congested_mask]
        congested_delays = delays[congested_mask]

        if not congested_steps.empty:
            if cfg['algo'] == 'pdhea':
                ax1.scatter(congested_steps, congested_delays, color='red', marker='*', s=120, zorder=5)
            else:
                # Baseline的崩溃点使用红叉 (使用大写字母X避免乱码)
                ax1.scatter(congested_steps, congested_delays, color='darkred', marker='x', s=60, linewidths=1.5,
                            zorder=5)

    ax1.set_ylabel("平均端到端时延 D (秒)", fontsize=13, fontweight='bold')
    ax1.set_title("动态剧变环境下 PD-HEA 与 Baseline 时延及效能双向鲁棒性对比", fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 放置图例到坐标轴外部右侧
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, title="时延曲线 (Delay)", title_fontsize=12)

    # 添加红星和红叉的图例说明
    ax1.text(0.01, 0.88, "★ 红星: PD-HEA 偶发拥塞\n X  红叉: Baseline 严重拥塞/宕机",
             transform=ax1.transAxes, color='darkred', fontsize=11,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    # ---------------------------------------------------------
    # 子图 2: 系统平均服务效能对比 (Avg QoS)
    # ---------------------------------------------------------
    for cfg in valid_configs:
        df = cfg['data']
        steps = df['Time_Step']
        qos = df['Avg_QoS']

        mfc = cfg['color'] if cfg['fill'] else 'none'
        ax2.plot(steps, qos, label=cfg['label'], color=cfg['color'],
                 marker=cfg['marker'], markerfacecolor=mfc, markersize=5,
                 linestyle=cfg['linestyle'], linewidth=1.8, alpha=0.85)

    ax2.set_xlabel("环境动态变化时间步 (Time Step)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("系统平均服务效能 Q", fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 放置图例到坐标轴外部右侧
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, title="效能曲线 (QoS)", title_fontsize=12)

    # 设置 X 轴刻度，使其每 5 个步长显示一次
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.set_xlim(0, 51)

    # ---------------------------------------------------------
    # 保存文件
    # ---------------------------------------------------------
    # 使用 bbox_inches='tight' 确保外部的图例不会被裁剪掉
    plt.savefig("Multi_Scale_Robustness_Comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("Multi_Scale_Robustness_Comparison.png", dpi=300, bbox_inches='tight')
    print("绘制完成！对比高清图表已保存为 'Multi_Scale_Robustness_Comparison.pdf' 和 '.png'。")


if __name__ == "__main__":
    plot_dynamic_robustness()