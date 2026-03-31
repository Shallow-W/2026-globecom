# src_reb package
"""
服务部署与路由实验框架 (重构版)

模块:
- config: 全局常量和配置
- data_model: Excel模型库读取与归一化
- evaluator: 评估层（染色体编码、延迟计算、QoS、内存利用率）
- ga: 遗传算法（初始化、选择、交叉、变异、修复）
- runner: 实验运行层（扰动实验、对比实验）
- main: 主入口
"""

from .config import *
from .data_model import ModelLibrary, ModelVersion
from .evaluator import (
    ChromosomeEncoder, RoutingMatrix, DelayCalculator,
    QoSCalculator, MemCalculator, FitnessAggregator
)
from .ga import GeneticAlgorithm, Individual
from .runner import ExperimentRunner, DataGenerator

__all__ = [
    "ModelLibrary", "ModelVersion",
    "ChromosomeEncoder", "RoutingMatrix", "DelayCalculator",
    "QoSCalculator", "MemCalculator", "FitnessAggregator",
    "GeneticAlgorithm", "Individual",
    "ExperimentRunner", "DataGenerator",
]
