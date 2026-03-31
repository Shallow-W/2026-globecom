"""ServiceChain class for modeling request service chains."""

from dataclasses import dataclass
from typing import List


@dataclass
class ServiceChain:
    """服务链/请求类型"""
    chain_id: str
    services: List[str]           # 服务ID顺序列表 [s1, s2, ..., sk]
    arrival_rate: float           # 到达率 λ (请求/秒)
    max_latency: float           # 最大容忍延迟 (ms)

    def __post_init__(self):
        """Initialize derived fields after construction."""
        self.length = len(self.services)

    def __repr__(self) -> str:
        return (f"ServiceChain(chain_id='{self.chain_id}', "
                f"services={self.services}, arrival_rate={self.arrival_rate}, "
                f"max_latency={self.max_latency})")
