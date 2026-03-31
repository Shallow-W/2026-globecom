"""MicroService class with multi-version model support."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ModelVersion:
    """
    模型版本定义

    每个AI服务有多个模型版本可选:
    - 高精度模型 (Model-H): μ低(慢), 精度高
    - 中精度模型 (Model-M): μ中等, 精度中等
    - 低精度模型 (Model-L): μ高(快), 精度低
    """
    version_id: str              # 版本标识 "Model-H", "Model-M", "Model-L"
    mu: float                   # 处理速率 (请求/秒)
    accuracy: float             # 精度指标 (0-1)
    cpu_per_instance: float     # CPU需求
    gpu_per_instance: int       # GPU内存需求 (MB)

    def calc_service_time(self) -> float:
        """平均服务时间 = 1000/μ (ms)"""
        return 1000.0 / self.mu


# 预定义版本配置
MODEL_VERSION_CONFIGS = {
    "Model-H": {"mu_range": [3, 7], "accuracy": 0.62, "cpu": 2, "gpu": 2048},
    "Model-M": {"mu_range": [8, 12], "accuracy": 0.53, "cpu": 1, "gpu": 1024},
    "Model-L": {"mu_range": [15, 25], "accuracy": 0.45, "cpu": 1, "gpu": 512},
}


@dataclass
class MicroService:
    """
    AI服务定义 (支持多模型版本)

    与 Service 的区别:
    - 包含多个 ModelVersion
    - 部署时需选择使用哪个版本
    """
    service_id: str
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    requires_gpu: bool = False

    def __post_init__(self):
        """Initialize derived fields after construction."""
        if not self.versions:
            # Use default model versions
            self.versions = {
                vid: ModelVersion(
                    version_id=vid,
                    mu=5.0,  # default
                    accuracy=MODEL_VERSION_CONFIGS[vid]["accuracy"],
                    cpu_per_instance=MODEL_VERSION_CONFIGS[vid]["cpu"],
                    gpu_per_instance=MODEL_VERSION_CONFIGS[vid]["gpu"]
                )
                for vid in ["Model-H", "Model-M", "Model-L"]
            }
        self.requires_gpu = any(v.gpu_per_instance > 0 for v in self.versions.values())

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """获取指定版本"""
        return self.versions.get(version_id)

    def get_default_version(self) -> Optional[ModelVersion]:
        """获取默认版本 (Model-M)"""
        return self.versions.get("Model-M")

    def get_version_ids(self) -> list:
        """Get all available version IDs."""
        return list(self.versions.keys())

    def calc_service_time(self, version_id: str = "Model-M") -> float:
        """计算指定版本的服务时间"""
        v = self.get_version(version_id)
        return v.calc_service_time() if v else float('inf')

    def get_processing_rate(self, version_id: str = "Model-M") -> float:
        """Get processing rate (mu) for specified version."""
        v = self.get_version(version_id)
        return v.mu if v else 0.0

    def get_cpu_per_instance(self, version_id: str = "Model-M") -> float:
        """Get CPU requirement per instance for specified version."""
        v = self.get_version(version_id)
        return v.cpu_per_instance if v else 0.0

    def get_gpu_per_instance(self, version_id: str = "Model-M") -> int:
        """Get GPU memory requirement per instance for specified version."""
        v = self.get_version(version_id)
        return v.gpu_per_instance if v else 0

    def __repr__(self) -> str:
        return f"MicroService(service_id='{self.service_id}', versions={list(self.versions.keys())})"
