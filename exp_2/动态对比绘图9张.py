import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import os
import numpy as np

# 设置英文字体 (优先使用 Arial)，并保留中文字体作为系统文件名的 fallback
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Times New Roman', 'SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_dynamic_robustness_separated():
    # 定义不同规模的配置文件映射
    scales = [
        {
            'name': '小规模',
            'en_name': 'Small Scale',
            'pdhea_file': 'Dynamic_Results_小规模.csv',
            'hu_file': '胡毅动态小规模.csv',
            'wang_file': '王良源动态小规模.csv',
            'he_file': '启发式动态小规模.csv',
            'gr_file': '贪心动态小规模.csv'
        },
        {
            'name': '中规模',
            'en_name': 'Small Scale',
            'pdhea_file': 'Dynamic_Results_中等规模.csv',
            'hu_file': '胡毅动态中规模.csv',
            'wang_file': '王良源动态中规模.csv',
            'he_file': '启发式动态中规模.csv',
            'gr_file': '贪心动态中规模.csv'
        },
        {
            'name': '大规模',
            'en_name': 'Large Scale',
            'pdhea_file': 'Dynamic_Results_大规模.csv',
            'hu_file': '胡毅动态大规模.csv',
            'wang_file': '王良源动态大规模.csv',
            'he_file': '启发式动态大规模.csv',
            'gr_file': '贪心动态大规模.csv'
        }
    ]

    # 统一使用 Nature 配色 (与静态折线图保持一致)
    colors = {
        'Our': '#E64B35',
        'THREE-STAGE': '#4DBBD5',
        'DA-RSPPO': '#00A087',
        'GMDA-RMPR': '#3C5488',
        'Greedy': '#F39B7F'
    }

    markers = {
        'Our': '*',
        'THREE-STAGE': 'o',
        'DA-RSPPO': 's',
        'GMDA-RMPR': '^',
        'Greedy': 'D'
    }

    lines = {
        'Our': '-',
        'THREE-STAGE': '--',
        'DA-RSPPO': '-.',
        'GMDA-RMPR': ':',
        'Greedy': '-'
    }

    for scale in scales:
        pdhea_file = scale['pdhea_file']
        hu_file = scale['hu_file']
        wang_file = scale['wang_file']
        he_file = scale['he_file']
        gr_file = scale['gr_file']
        scale_name = scale['name']
        en_name = scale['en_name']

        # 检查所有5个文件是否存在
        files_to_check = [pdhea_file, hu_file, wang_file, he_file, gr_file]
        missing_files = [f for f in files_to_check if not os.path.exists(f)]

        if missing_files:
            print(f"⚠️ Warning: Missing files for [{scale_name}]: {missing_files}, skipping plot!")
            continue

        # 读取数据并截取前 15 代环境变化
        df_pdhea = pd.read_csv(pdhea_file).head(15)
        df_hu = pd.read_csv(hu_file).head(15)
        df_wang = pd.read_csv(wang_file).head(15)
        df_he = pd.read_csv(he_file).head(15)
        df_gr = pd.read_csv(gr_file).head(15)

        steps = df_pdhea['Time_Step']

        # =========================================================
        # 1. 绘制：动态时延对比曲线 (Delay Comparison)
        # =========================================================
        plt.figure(figsize=(10, 6))

        # 绘制主折线
        plt.plot(steps, df_pdhea['Avg_Delay(s)'], label='Our', color=colors['Our'],
                 marker=markers['Our'], linestyle=lines['Our'], linewidth=2.5, markersize=8, alpha=0.85)
        plt.plot(steps, df_hu['Avg_Delay(s)'], label='THREE-STAGE', color=colors['THREE-STAGE'],
                 marker=markers['THREE-STAGE'], markerfacecolor='none', linestyle=lines['THREE-STAGE'], linewidth=2.5,
                 markersize=8, alpha=0.85)
        plt.plot(steps, df_wang['Avg_Delay(s)'], label='DA-RSPPO', color=colors['DA-RSPPO'],
                 marker=markers['DA-RSPPO'], markerfacecolor='none', linestyle=lines['DA-RSPPO'], linewidth=2.5,
                 markersize=8, alpha=0.85)
        plt.plot(steps, df_he['Avg_Delay(s)'], label='GMDA-RMPR', color=colors['GMDA-RMPR'],
                 marker=markers['GMDA-RMPR'], markerfacecolor='none', linestyle=lines['GMDA-RMPR'], linewidth=2.5,
                 markersize=8, alpha=0.85)
        plt.plot(steps, df_gr['Avg_Delay(s)'], label='Greedy', color=colors['Greedy'],
                 marker=markers['Greedy'], markerfacecolor='none', linestyle=lines['Greedy'], linewidth=2.5,
                 markersize=8, alpha=0.85)

        # 标记各算法的拥塞惩罚点 (统一使用 'X' 图标加黑色描边以增强辨识度)
        if (df_pdhea['Penalty(0=Safe)'] > 0).any():
            plt.scatter(steps[df_pdhea['Penalty(0=Safe)'] > 0],
                        df_pdhea.loc[df_pdhea['Penalty(0=Safe)'] > 0, 'Avg_Delay(s)'],
                        color=colors['Our'], marker='X', s=150, zorder=5, label='Our (Cong)', edgecolor='black')

        if (df_hu['Penalty(0=Safe)'] > 0).any():
            plt.scatter(steps[df_hu['Penalty(0=Safe)'] > 0], df_hu.loc[df_hu['Penalty(0=Safe)'] > 0, 'Avg_Delay(s)'],
                        color=colors['THREE-STAGE'], marker='X', s=120, zorder=5, label='THREE-STAGE (Cong)',
                        edgecolor='black')

        if (df_wang['Penalty(0=Safe)'] > 0).any():
            plt.scatter(steps[df_wang['Penalty(0=Safe)'] > 0],
                        df_wang.loc[df_wang['Penalty(0=Safe)'] > 0, 'Avg_Delay(s)'],
                        color=colors['DA-RSPPO'], marker='X', s=120, zorder=5, label='DA-RSPPO (Cong)',
                        edgecolor='black')

        if (df_he['Penalty(0=Safe)'] > 0).any():
            plt.scatter(steps[df_he['Penalty(0=Safe)'] > 0], df_he.loc[df_he['Penalty(0=Safe)'] > 0, 'Avg_Delay(s)'],
                        color=colors['GMDA-RMPR'], marker='X', s=120, zorder=5, label='GMDA-RMPR (Cong)',
                        edgecolor='black')

        if (df_gr['Penalty(0=Safe)'] > 0).any():
            plt.scatter(steps[df_gr['Penalty(0=Safe)'] > 0], df_gr.loc[df_gr['Penalty(0=Safe)'] > 0, 'Avg_Delay(s)'],
                        color=colors['Greedy'], marker='X', s=120, zorder=5, label='Greedy (Cong)', edgecolor='black')

        # 图表装饰 (全英文)
        plt.xlabel("Time Step", fontsize=18, fontweight='bold')
        plt.ylabel("Average End-to-End Delay (s)", fontsize=18, fontweight='bold')
        plt.title(f"Dynamic Delay Comparison ({en_name})", fontsize=18, fontweight='bold', pad=15)

        # 图例放置在右上角 (包含拥塞提示，双列排版)
        plt.legend(loc='upper right', fontsize=15, ncol=2)
        plt.grid(True, linestyle='--', alpha=0.6)
        # 将 X 轴的刻度间隔设置为 1
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
        # 限制 X 轴的显示范围到前 15 代
        plt.xlim(0.5, 15.5)

        # 保存时延图片
        delay_filename = f"{scale_name}动态时延对比曲线"
        plt.savefig(f"{delay_filename}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Generated: {delay_filename}")

        # =========================================================
        # 2. 绘制：动态准确率对比曲线 (Accuracy Comparison)
        # =========================================================
        if all('Avg_Accuracy' in df.columns for df in [df_pdhea, df_hu, df_wang, df_he, df_gr]):
            plt.figure(figsize=(10, 6))

            plt.plot(steps, df_pdhea['Avg_Accuracy'], label='Our', color=colors['Our'],
                     marker=markers['Our'], linestyle=lines['Our'], linewidth=2.5, markersize=8, alpha=0.85)
            plt.plot(steps, df_hu['Avg_Accuracy'], label='THREE-STAGE', color=colors['THREE-STAGE'],
                     marker=markers['THREE-STAGE'], markerfacecolor='none', linestyle=lines['THREE-STAGE'],
                     linewidth=2.5, markersize=8, alpha=0.85)
            plt.plot(steps, df_wang['Avg_Accuracy'], label='DA-RSPPO', color=colors['DA-RSPPO'],
                     marker=markers['DA-RSPPO'], markerfacecolor='none', linestyle=lines['DA-RSPPO'], linewidth=2.5,
                     markersize=8, alpha=0.85)
            plt.plot(steps, df_he['Avg_Accuracy'], label='GMDA-RMPR', color=colors['GMDA-RMPR'],
                     marker=markers['GMDA-RMPR'], markerfacecolor='none', linestyle=lines['GMDA-RMPR'], linewidth=2.5,
                     markersize=8, alpha=0.85)
            plt.plot(steps, df_gr['Avg_Accuracy'], label='Greedy', color=colors['Greedy'],
                     marker=markers['Greedy'], markerfacecolor='none', linestyle=lines['Greedy'], linewidth=2.5,
                     markersize=8, alpha=0.85)

            # 图表装饰
            plt.xlabel("Time Step", fontsize=18, fontweight='bold')
            plt.ylabel("Average Service Accuracy", fontsize=18, fontweight='bold')
            plt.title(f"Dynamic Accuracy Comparison ({en_name})", fontsize=18, fontweight='bold', pad=15)

            # 图例在右上角
            plt.legend(loc='upper right', fontsize=15, ncol=2)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
            plt.xlim(0.5, 15.5)

            acc_filename = f"{scale_name}动态时准确率对比曲线"
            plt.savefig(f"{acc_filename}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Generated: {acc_filename}")

        # =========================================================
        # 3. 绘制：动态效能对比曲线 (Avg QoS Comparison)
        # =========================================================
        if all('Avg_QoS' in df.columns for df in [df_pdhea, df_hu, df_wang, df_he, df_gr]):
            plt.figure(figsize=(10, 6))

            plt.plot(steps, df_pdhea['Avg_QoS'], label='Our', color=colors['Our'],
                     marker=markers['Our'], linestyle=lines['Our'], linewidth=2.5, markersize=8, alpha=0.85)
            plt.plot(steps, df_hu['Avg_QoS'], label='THREE-STAGE', color=colors['THREE-STAGE'],
                     marker=markers['THREE-STAGE'], markerfacecolor='none', linestyle=lines['THREE-STAGE'],
                     linewidth=2.5, markersize=8, alpha=0.85)
            plt.plot(steps, df_wang['Avg_QoS'], label='DA-RSPPO', color=colors['DA-RSPPO'],
                     marker=markers['DA-RSPPO'], markerfacecolor='none', linestyle=lines['DA-RSPPO'], linewidth=2.5,
                     markersize=8, alpha=0.85)
            plt.plot(steps, df_he['Avg_QoS'], label='GMDA-RMPR', color=colors['GMDA-RMPR'],
                     marker=markers['GMDA-RMPR'], markerfacecolor='none', linestyle=lines['GMDA-RMPR'], linewidth=2.5,
                     markersize=8, alpha=0.85)
            plt.plot(steps, df_gr['Avg_QoS'], label='Greedy', color=colors['Greedy'],
                     marker=markers['Greedy'], markerfacecolor='none', linestyle=lines['Greedy'], linewidth=2.5,
                     markersize=8, alpha=0.85)

            # 图表装饰
            plt.xlabel("Time Step", fontsize=18, fontweight='bold')
            plt.ylabel("Average QoS", fontsize=18, fontweight='bold')
            plt.title(f"Dynamic QoS Comparison ({en_name})", fontsize=18, fontweight='bold', pad=15)

            # 图例在右上角
            plt.legend(loc='upper right', fontsize=15, ncol=2)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
            plt.xlim(0.5, 15.5)

            qos_filename = f"{scale_name}动态时效能对比曲线"
            plt.savefig(f"{qos_filename}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Generated: {qos_filename}")

    print("\n🎉 All separated comparison plots are successfully generated and saved!")


if __name__ == "__main__":
    plot_dynamic_robustness_separated()