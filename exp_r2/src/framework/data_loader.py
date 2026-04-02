"""Load model tables and expand task pools."""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Tuple

import pandas as pd


def load_and_prepare_data(
    excel_file: str,
    max_required_tasks: int = 80,
    top_k: int = 5,
) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]]]:
    """Load model candidates from Excel and expand to the required task count."""

    tasks_data: Dict[str, List[Dict[str, Any]]] = {}

    try:
        xl = pd.ExcelFile(excel_file)
        for task_name in xl.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=task_name)
            topk = df.sort_values(by="proxy_score", ascending=False).head(top_k).copy()

            min_q = df["task_final_performance"].min()
            max_q = df["task_final_performance"].max()
            if max_q > min_q:
                topk["normalized_qos"] = 0.1 + 0.9 * (
                    (topk["task_final_performance"] - min_q) / (max_q - min_q)
                )
            else:
                topk["normalized_qos"] = 1.0

            tasks_data[task_name] = topk[
                [
                    "architecture",
                    "proxy_score",
                    "task_final_performance",
                    "normalized_qos",
                    "model_params",
                    "flops",
                ]
            ].to_dict("records")
    except FileNotFoundError:
        for i in range(7):
            task_name = f"mock_task_{i}"
            tasks_data[task_name] = [
                {
                    "architecture": f"arch_{v}",
                    "proxy_score": random.uniform(0.5, 2.0),
                    "task_final_performance": random.uniform(0.5, 0.9),
                    "normalized_qos": random.uniform(0.5, 1.0),
                    "model_params": random.randint(1_000_000, 20_000_000),
                    "flops": random.randint(100_000_000, 2_000_000_000),
                }
                for v in range(top_k)
            ]

    original_tasks = list(tasks_data.keys())
    if not original_tasks:
        raise RuntimeError(
            "No tasks loaded from Excel and no fallback tasks available."
        )

    tasks_list = copy.deepcopy(original_tasks)
    idx = 0
    while len(tasks_list) < max_required_tasks:
        base_task = original_tasks[idx % len(original_tasks)]
        new_task_name = f"{base_task}_copy_{len(tasks_list)}"
        tasks_data[new_task_name] = copy.deepcopy(tasks_data[base_task])
        tasks_list.append(new_task_name)
        idx += 1

    return tasks_list, tasks_data
