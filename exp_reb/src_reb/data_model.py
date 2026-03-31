# ==========================================
# 数据层：Excel模型库读取与归一化
# ==========================================

import pandas as pd
import numpy as np
import random
from typing import Dict, List
from dataclasses import dataclass

# 全局常量
N_VERSIONS = 3  # 每个任务固定3个候选版本（Model-H/M/L）


@dataclass
class ModelVersion:
    """单个模型版本"""
    model_id: str           # 架构标识
    proxy_score: float      # 代理分数（越高越好）
    model_params: int       # 参数量（内存占用）
    flops: int              # 计算量（越低越快）
    normalized_qos: float   # 归一化QoS [0.1, 1.0]
    raw_performance: float  # 原始精度（未归一化）


class ModelLibrary:
    """
    模型库：读取Excel并计算归一化QoS

    每个task对应Excel一个sheet，读取字段：
    - architecture: 模型架构名
    - proxy_score: 代理分数
    - model_params: 参数量
    - flops: 计算量
    - task_final_performance: 最终精度
    """

    def __init__(self, excel_path: str = None):
        self.excel_path = excel_path
        self.task_data: Dict[str, List[ModelVersion]] = {}
        self.available_tasks: List[str] = []

    def load_from_excel(self, excel_file: str = None, max_required_tasks: int = 50):
        """从Excel加载数据并归一化"""
        path = excel_file or self.excel_path
        if path is None:
            raise ValueError("必须提供Excel文件路径")

        try:
            xl = pd.ExcelFile(path)
            sheet_names = xl.sheet_names
        except FileNotFoundError:
            print(f"警告: 未找到 {path}，使用模拟数据")
            self._generate_mock_data(max_required_tasks)
            return

        tasks_data = {}
        for task_name in sheet_names:
            df = pd.read_excel(path, sheet_name=task_name)
            top5 = df.sort_values(by="proxy_score", ascending=False).head(N_VERSIONS).copy()

            min_q = df['task_final_performance'].min()
            max_q = df['task_final_performance'].max()
            if max_q > min_q:
                top5['normalized_qos'] = 0.1 + 0.9 * (top5['task_final_performance'] - min_q) / (max_q - min_q)
            else:
                top5['normalized_qos'] = 1.0

            tasks_data[task_name] = [
                ModelVersion(
                    model_id=row['architecture'],
                    proxy_score=float(row['proxy_score']),
                    model_params=int(row['model_params']),
                    flops=int(row['flops']),
                    normalized_qos=float(row['normalized_qos']),
                    raw_performance=float(row['task_final_performance'])
                )
                for _, row in top5.iterrows()
            ]

        self.available_tasks = list(tasks_data.keys())
        self.task_data = tasks_data
        self._expand_to_required(max_required_tasks)
        print(f"数据加载完毕: {len(self.available_tasks)} 个任务类型")

    def _generate_mock_data(self, max_required: int):
        """生成模拟数据（当Excel不存在时）"""
        random.seed(42)
        np.random.seed(42)
        # 使用 s0, s1, ... 命名，与服务ID对应
        mock_tasks = [f"s{i}" for i in range(max(7, max_required))]

        tasks_data = {}
        for task_name in mock_tasks:
            tasks_data[task_name] = [
                ModelVersion(
                    model_id=f"arch_{v}",
                    proxy_score=random.uniform(0.5, 2.0),
                    model_params=random.randint(1_000_000, 20_000_000),
                    flops=random.randint(100_000_000, 2_000_000_000),
                    normalized_qos=random.uniform(0.5, 1.0),
                    raw_performance=random.uniform(0.5, 0.9)
                )
                for v in range(N_VERSIONS)
            ]

        self.task_data = tasks_data
        self.available_tasks = list(tasks_data.keys())
        self._expand_to_required(max_required)

    def _expand_to_required(self, max_required: int):
        """数据扩充：将任务池扩展到max_required个类型"""
        original = list(self.task_data.keys())
        idx = 0
        while len(self.available_tasks) < max_required:
            base = original[idx % len(original)]
            new_name = f"{base}_copy_{len(self.available_tasks)}"
            self.task_data[new_name] = self.task_data[base]
            self.available_tasks.append(new_name)
            idx += 1

    def get_versions(self, task_name: str) -> List[ModelVersion]:
        """获取任务的所有模型版本"""
        return self.task_data.get(task_name, [])

    def get_task_pool(self, num_types: int) -> List[str]:
        """获取任务池（前num_types个）"""
        return self.available_tasks[:num_types]
