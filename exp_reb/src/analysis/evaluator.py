"""Evaluator module for computing and comparing algorithm metrics."""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd


class Evaluator:
    """指标计算与评估"""

    @staticmethod
    def summarize(results: List[Dict]) -> pd.DataFrame:
        """
        汇总实验结果为DataFrame

        Args:
            results: 实验结果列表，每个元素包含:
                - algorithm: 算法名称
                - avg_latency: 平均延迟
                - success_rate: 成功率
                - deployment_cost: 部署成本
                - resource_utilization: 资源利用率字典

        Returns:
            pd.DataFrame: 汇总后的结果表
        """
        rows = []
        for r in results:
            rows.append({
                "algorithm": r["algorithm"],
                "avg_latency": r["avg_latency"],
                "success_rate": r["success_rate"],
                "deployment_cost": r["deployment_cost"],
                "avg_utilization": np.mean(list(r["resource_utilization"].values()))
                if r.get("resource_utilization") else 0.0
            })
        return pd.DataFrame(rows)

    @staticmethod
    def compare_algorithms(results: List[Dict],
                          metric: str = "avg_latency") -> Dict:
        """
        对比各算法在指定指标上的表现

        Args:
            results: 实验结果列表
            metric: 要对比的指标名称，可选值:
                - "avg_latency": 平均延迟
                - "success_rate": 成功率
                - "deployment_cost": 部署成本
                - "avg_utilization": 平均资源利用率

        Returns:
            Dict: 各算法的统计信息，格式为
                {
                    "algorithm_name": {
                        "mean": 平均值,
                        "std": 标准差,
                        "min": 最小值,
                        "max": 最大值,
                        "median": 中位数
                    }
                }
        """
        alg_metrics = {}
        for r in results:
            alg = r["algorithm"]
            if alg not in alg_metrics:
                alg_metrics[alg] = []
            alg_metrics[alg].append(r[metric])

        # 计算统计量
        summary = {}
        for alg, values in alg_metrics.items():
            if len(values) == 0:
                continue
            summary[alg] = {
                "mean": np.mean(values),
                "std": np.std(values) if len(values) > 1 else 0.0,
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        return summary

    @staticmethod
    def compare_all_metrics(results: List[Dict]) -> pd.DataFrame:
        """
        对比所有指标

        Args:
            results: 实验结果列表

        Returns:
            pd.DataFrame: 包含所有算法所有指标的对比表
        """
        metrics = ["avg_latency", "success_rate", "deployment_cost", "avg_utilization"]
        all_summaries = {}

        for metric in metrics:
            summary = Evaluator.compare_algorithms(results, metric)
            for alg, stats in summary.items():
                if alg not in all_summaries:
                    all_summaries[alg] = {}
                all_summaries[alg][f"{metric}_mean"] = stats["mean"]
                all_summaries[alg][f"{metric}_std"] = stats["std"]

        return pd.DataFrame.from_dict(all_summaries, orient="index")

    @staticmethod
    def rank_algorithms(results: List[Dict],
                       metric: str = "avg_latency",
                       higher_is_better: bool = False) -> pd.DataFrame:
        """
        对算法进行排名

        Args:
            results: 实验结果列表
            metric: 排名依据的指标
            higher_is_better: True表示该指标越大越好（如成功率），
                             False表示越小越好（如延迟）

        Returns:
            pd.DataFrame: 排名表
        """
        summary = Evaluator.compare_algorithms(results, metric)

        rank_data = []
        for alg, stats in summary.items():
            rank_data.append({
                "algorithm": alg,
                "mean": stats["mean"],
                "std": stats["std"]
            })

        df = pd.DataFrame(rank_data)

        if higher_is_better:
            df = df.sort_values("mean", ascending=False)
        else:
            df = df.sort_values("mean", ascending=True)

        df["rank"] = range(1, len(df) + 1)
        return df[["rank", "algorithm", "mean", "std"]]
