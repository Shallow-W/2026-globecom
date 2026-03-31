"""Reporter module for generating experiment reports."""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from .evaluator import Evaluator
from .visualizer import Visualizer


class Reporter:
    """实验报告生成器"""

    def __init__(self, output_dir: str = "results"):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.visualizer = Visualizer()
        self.evaluator = Evaluator()

    def generate_report(self,
                       results: List[Dict],
                       experiment_name: str = "experiment",
                       include_visualizations: bool = True,
                       include_detailed_metrics: bool = True) -> Dict[str, str]:
        """
        生成实验报告

        Args:
            results: 实验结果列表
            experiment_name: 实验名称
            include_visualizations: 是否包含可视化图表
            include_detailed_metrics: 是否包含详细指标

        Returns:
            Dict[str, str]: 生成的报告文件路径
                - summary_csv: 汇总CSV路径
                - comparison_csv: 对比CSV路径
                - comparison_plot: 对比图路径
                - perturbation_plot: 扰动图路径
                - report_json: JSON报告路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(self.output_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)

        output_files = {}

        # 1. 生成汇总DataFrame
        summary_df = self.evaluator.summarize(results)
        summary_csv_path = os.path.join(exp_dir, "summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        output_files["summary_csv"] = summary_csv_path

        # 2. 生成算法对比统计
        if include_detailed_metrics:
            comparison_df = self.evaluator.compare_all_metrics(results)
            comparison_csv_path = os.path.join(exp_dir, "algorithm_comparison.csv")
            comparison_df.to_csv(comparison_csv_path)
            output_files["comparison_csv"] = comparison_csv_path

        # 3. 生成算法对比图
        if include_visualizations:
            try:
                comparison_plot = self.visualizer.plot_algorithm_comparison(
                    results, exp_dir
                )
                output_files["comparison_plot"] = comparison_plot
            except Exception as e:
                print(f"Warning: Failed to generate comparison plot: {e}")

        # 4. 检查是否有扰动实验数据并生成扰动图
        has_perturbation = any("param" in r and "value" in r for r in results)
        if has_perturbation and include_visualizations:
            try:
                perturbation_plot = self.visualizer.plot_perturbation(
                    results, exp_dir
                )
                output_files["perturbation_plot"] = perturbation_plot
            except Exception as e:
                print(f"Warning: Failed to generate perturbation plot: {e}")

        # 5. 生成JSON报告
        report_data = self._build_json_report(
            results, summary_df, experiment_name, include_detailed_metrics
        )
        report_json_path = os.path.join(exp_dir, "report.json")
        with open(report_json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
        output_files["report_json"] = report_json_path

        # 6. 生成Markdown格式报告
        md_report_path = self._generate_markdown_report(
            results, summary_df, report_data, exp_dir, experiment_name
        )
        output_files["report_md"] = md_report_path

        return output_files

    def _build_json_report(self,
                          results: List[Dict],
                          summary_df: pd.DataFrame,
                          experiment_name: str,
                          include_detailed: bool) -> Dict:
        """构建JSON报告数据"""
        report = {
            "experiment_name": experiment_name,
            "generated_at": datetime.now().isoformat(),
            "num_experiments": len(results),
            "algorithms": summary_df["algorithm"].tolist() if not summary_df.empty else [],
            "summary": {
                "best_latency": {
                    "algorithm": summary_df.loc[summary_df["avg_latency"].idxmin(), "algorithm"]
                    if not summary_df.empty else None,
                    "value": float(summary_df["avg_latency"].min()) if not summary_df.empty else None
                },
                "best_success_rate": {
                    "algorithm": summary_df.loc[summary_df["success_rate"].idxmax(), "algorithm"]
                    if not summary_df.empty else None,
                    "value": float(summary_df["success_rate"].max()) if not summary_df.empty else None
                }
            }
        }

        if include_detailed:
            report["metrics"] = {
                metric: self.evaluator.compare_algorithms(results, metric)
                for metric in ["avg_latency", "success_rate", "deployment_cost", "avg_utilization"]
            }

        return report

    def _generate_markdown_report(self,
                                  results: List[Dict],
                                  summary_df: pd.DataFrame,
                                  report_data: Dict,
                                  output_dir: str,
                                  experiment_name: str) -> str:
        """生成Markdown格式报告"""
        md_lines = [
            f"# {experiment_name} 实验报告",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. 实验概述",
            "",
            f"- 实验次数: {report_data['num_experiments']}",
            f"- 算法数量: {len(report_data['algorithms'])}",
            f"- 算法列表: {', '.join(report_data['algorithms'])}",
            "",
            "## 2. 算法性能汇总",
            "",
        ]

        # 添加汇总表格
        if not summary_df.empty:
            md_lines.append("| Algorithm | Avg Latency (ms) | Success Rate | Deployment Cost | Avg Utilization |")
            md_lines.append("|-----------|-----------------|--------------|-----------------|-----------------|")
            for _, row in summary_df.iterrows():
                md_lines.append(
                    f"| {row['algorithm']} | {row['avg_latency']:.2f} | "
                    f"{row['success_rate']:.2%} | {row['deployment_cost']} | "
                    f"{row['avg_utilization']:.2%} |"
                )
            md_lines.append("")

        # 添加最佳算法推荐
        md_lines.append("## 3. 最佳算法推荐")
        md_lines.append("")
        best_latency = report_data["summary"]["best_latency"]
        best_success = report_data["summary"]["best_success_rate"]
        md_lines.append(f"- **最低延迟**: {best_latency['algorithm']} ({best_latency['value']:.2f} ms)")
        md_lines.append(f"- **最高成功率**: {best_success['algorithm']} ({best_success['value']:.2%})")
        md_lines.append("")

        # 添加生成的文件列表
        md_lines.append("## 4. 生成的文件")
        md_lines.append("")
        md_lines.append("| 文件类型 | 说明 |")
        md_lines.append("|---------|------|")
        md_lines.append("| summary.csv | 实验结果汇总 |")
        if include_detailed_metrics := "metrics" in report_data:
            md_lines.append("| algorithm_comparison.csv | 算法详细对比 |")
        md_lines.append("| comparison_plot.png | 算法性能对比图 |")
        if any("param" in r for r in results):
            md_lines.append("| perturbation.png | 扰动实验结果图 |")
        md_lines.append("| report.json | JSON格式报告 |")

        md_path = os.path.join(output_dir, "REPORT.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))

        return md_path

    @staticmethod
    def save_results(results: List[Dict], output_path: str) -> str:
        """
        保存实验结果到JSON文件

        Args:
            results: 实验结果列表
            output_path: 输出文件路径

        Returns:
            str: 保存的文件路径
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        return output_path

    @staticmethod
    def load_results(input_path: str) -> List[Dict]:
        """
        从JSON文件加载实验结果

        Args:
            input_path: 输入文件路径

        Returns:
            List[Dict]: 实验结果列表
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
