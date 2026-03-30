"""
数据加载模块 - 读取 xlsx 并做归一化预处理
"""

import pandas as pd
import numpy as np
from typing import Dict


def load_architecture_tables(data_path: str) -> Dict[str, pd.DataFrame]:
    """
    加载 xlsx 的 7 张任务表，对每个任务的 proxy_score/flops/model_params 做 min-max 归一化。

    Returns:
        dict: {task_name: DataFrame}，每张表新增 _norm 列
    """
    tables = pd.read_excel(data_path, sheet_name=None)

    normalized = {}
    for task_name, df in tables.items():
        df = df.copy()

        # 基本字段
        df['arch_id'] = df['architecture']
        df['task'] = task_name

        # 对每个任务表内做 min-max 归一化（用于同任务内跨架构比较）
        for col in ['proxy_score', 'flops', 'model_params']:
            v_min, v_max = df[col].min(), df[col].max()
            if v_max > v_min:
                df[f'{col}_norm'] = (df[col] - v_min) / (v_max - v_min)
            else:
                df[f'{col}_norm'] = 0.5

        # epochs 处理（处理 '-' 值）
        df['epochs_val'] = pd.to_numeric(
            df['epochs_to_reach_avg_final_performance'], errors='coerce'
        ).fillna(0)
        e_max = df['epochs_val'].max()
        df['epochs_norm'] = df['epochs_val'] / e_max if e_max > 0 else 0.0

        normalized[task_name] = df

    return normalized


def get_task_metric(task: str) -> str:
    """返回任务对应的评估指标名称"""
    metric_map = {
        'class_scene': 'top1',
        'class_object': 'top1',
        'room_layout': 'neg_loss',
        'jigsaw': 'top1',
        'segmentsemantic': 'mIOU',
        'normal': 'ssim',
        'autoencoder': 'ssim',
    }
    return metric_map.get(task, 'unknown')
