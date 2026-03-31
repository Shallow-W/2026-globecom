"""Visualizer module for plotting experiment results."""

import os
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class Visualizer:
    """可视化"""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        初始化可视化器

        Args:
            style: matplotlib 样式
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        self.fig_dpi = 100
        self.fig_size = (10, 6)

    @staticmethod
    def plot_algorithm_comparison(results: List[Dict],
                                  output_dir: str,
                                  metrics: Optional[List[str]] = None) -> str:
        """
        绘制算法对比图

        Args:
            results: 实验结果列表
            output_dir: 输出目录
            metrics: 要绘制的指标列表，默认包含延迟、成功率和资源利用率

        Returns:
            str: 保存的图片路径
        """
        os.makedirs(output_dir, exist_ok=True)

        if metrics is None:
            metrics = ["avg_latency", "success_rate", "avg_utilization"]

        df = Evaluator.summarize(results)
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        metric_labels = {
            "avg_latency": "Average Latency (ms)",
            "success_rate": "Success Rate",
            "avg_utilization": "Resource Utilization",
            "deployment_cost": "Deployment Cost"
        }

        metric_colors = {
            "avg_latency": "steelblue",
            "success_rate": "green",
            "avg_utilization": "orange",
            "deployment_cost": "purple"
        }

        for i, metric in enumerate(metrics):
            ax = axes[i]
            if metric not in df.columns:
                continue

            # 按指标值排序
            df_sorted = df.sort_values(metric, ascending=(metric != "success_rate"))
            algorithms = df_sorted["algorithm"].tolist()
            values = df_sorted[metric].tolist()

            bars = ax.bar(range(len(algorithms)), values,
                         color=metric_colors.get(metric, "steelblue"))

            ax.set_xticks(range(len(algorithms)))
            ax.set_xticklabels(algorithms, rotation=45, ha='right')
            ax.set_ylabel(metric_labels.get(metric, metric))
            ax.set_title(f"{metric_labels.get(metric, metric)} Comparison")

            # 添加数值标签
            for j, (alg, val) in enumerate(zip(algorithms, values)):
                if metric == "success_rate":
                    label = f"{val:.2%}"
                else:
                    label = f"{val:.2f}"
                ax.text(j, val, label, ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        output_path = os.path.join(output_dir, "algorithm_comparison.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path

    @staticmethod
    def plot_perturbation(results: List[Dict],
                         output_dir: str,
                         metric: str = "avg_latency") -> str:
        """
        绘制扰动实验结果图

        Args:
            results: 扰动实验结果列表，包含 param 和 value 字段
            output_dir: 输出目录
            metric: 要绘制的指标

        Returns:
            str: 保存的图片路径
        """
        os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(results)
        if df.empty or "value" not in df.columns or "algorithm" not in df.columns:
            return ""

        # 创建透视表
        pivot = df.pivot_table(values=metric,
                              index="value",
                              columns="algorithm",
                              aggfunc="mean")

        fig, ax = plt.subplots(figsize=(10, 6))

        pivot.plot(ax=ax, marker='o', linewidth=2)

        ax.set_xlabel(df["param"].iloc[0] if "param" in df.columns else "Parameter Value")
        ax.set_ylabel(f"{metric.replace('_', ' ').title()} ({'ms' if metric == 'avg_latency' else ''})")
        ax.set_title(f"Perturbation Analysis: {df['param'].iloc[0] if 'param' in df.columns else metric}")

        # 设置x轴为数值型刻度
        if pivot.index.dtype == 'object':
            try:
                pivot.index = pivot.index.astype(float)
                pivot = pivot.sort_index()
            except:
                pass

        ax.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, "perturbation.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path

    @staticmethod
    def plot_latency_breakdown(chain_latencies: List[Dict],
                               output_dir: str,
                               chain_id: Optional[str] = None) -> str:
        """
        绘制延迟分解图（排队、处理、通信延迟）

        Args:
            chain_latencies: 服务链延迟数据列表
            output_dir: 输出目录
            chain_id: 可选，指定服务链ID

        Returns:
            str: 保存的图片路径
        """
        os.makedirs(output_dir, exist_ok=True)

        if not chain_latencies:
            return ""

        # 筛选指定链的数据
        if chain_id:
            filtered = [c for c in chain_latencies if c.get("chain_id") == chain_id]
            if not filtered:
                filtered = chain_latencies
        else:
            filtered = chain_latencies

        df = pd.DataFrame(filtered)

        if df.empty:
            return ""

        components = ["queuing", "processing", "communication"]
        labels = ["Queuing Delay", "Processing Delay", "Communication Delay"]
        colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]

        fig, ax = plt.subplots(figsize=(10, 6))

        # 堆叠柱状图
        x = range(len(df))
        bottom = np.zeros(len(df))

        for comp, label, color in zip(components, labels, colors):
            if comp in df.columns:
                values = df[comp].fillna(0).values
                ax.bar(x, values, bottom=bottom, label=label, color=color)
                bottom += values

        chain_labels = df["chain_id"].tolist() if "chain_id" in df.columns else [f"Chain {i}" for i in range(len(df))]
        ax.set_xticks(x)
        ax.set_xticklabels(chain_labels, rotation=45, ha='right')
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency Breakdown by Service Chain")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = os.path.join(output_dir, "latency_breakdown.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path

    @staticmethod
    def plot_resource_utilization(utilization: Dict[str, float],
                                 output_dir: str,
                                 title: str = "Resource Utilization") -> str:
        """
        绘制资源利用率图

        Args:
            utilization: 节点资源利用率字典 {node_id: utilization}
            output_dir: 输出目录
            title: 图表标题

        Returns:
            str: 保存的图片路径
        """
        os.makedirs(output_dir, exist_ok=True)

        if not utilization:
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))

        nodes = list(utilization.keys())
        values = list(utilization.values())

        # 按利用率排序
        sorted_pairs = sorted(zip(nodes, values), key=lambda x: x[1], reverse=True)
        nodes, values = zip(*sorted_pairs)

        colors = plt.cm.RdYlGn_r(np.array(values) / max(values) if max(values) > 0 else values)

        bars = ax.bar(range(len(nodes)), values, color=colors)

        ax.set_xticks(range(len(nodes)))
        ax.set_xticklabels(nodes, rotation=45, ha='right')
        ax.set_ylabel("Utilization")
        ax.set_title(title)
        ax.axhline(y=0.8, color='r', linestyle='--', label="80% threshold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = os.path.join(output_dir, "resource_utilization.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path


# Import Evaluator for use in plot_algorithm_comparison
from .evaluator import Evaluator
