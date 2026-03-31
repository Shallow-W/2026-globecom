"""M/M/C queueing model with Erlang-C formula."""

import math
from typing import Dict


class MMCQueue:
    """
    M/M/C 排队模型 (无限容量)

    输入: 到达率λ, 服务率μ, 服务器数c
    输出: 等待概率、平均队列长度、平均等待时间、平均响应时间
    """

    def __init__(self, arrival_rate: float, service_rate: float, num_servers: int):
        """
        Initialize M/M/C queue.

        Args:
            arrival_rate: Arrival rate λ (requests/second)
            service_rate: Service rate μ per server (requests/second)
            num_servers: Number of servers c
        """
        self.lambda_ = arrival_rate   # λ
        self.mu = service_rate        # μ (每个服务器)
        self.c = num_servers          # c

    @property
    def rho(self) -> float:
        """服务强度 ρ = λ / (c * μ)"""
        return self.lambda_ / (self.c * self.mu)

    def is_stable(self) -> bool:
        """Check if queue is stable (rho < 1)."""
        return self.rho < 1.0

    def erlang_c(self) -> float:
        """
        Erlang-C 公式: 等待概率 Pw
        Pw = ((cρ)^c / c!) / (Σ_{k=0}^{c-1} ((cρ)^k / k!) + ((cρ)^c / c!) / (1-ρ))
        """
        if self.rho >= 1.0:
            return 1.0
        c = self.c
        rho = self.rho
        crho = c * rho

        # 数值稳定性保护: 当 rho 接近1或crho很大时，概率接近1
        # crho^crho 会在溢出前快速变得非常大
        if rho > 0.95 or crho > 50:
            return 1.0

        try:
            # 分子
            num = crho ** c / self._factorial(c)

            # 分母前半部分 (k=0 到 c-1)
            sum_part = sum(crho ** k / self._factorial(k) for k in range(c))

            # 分母后半部分
            denom = sum_part + crho ** c / (self._factorial(c) * (1 - rho))

            return num / denom
        except OverflowError:
            # 数值溢出时，返回接近1的概率 (系统高负载)
            return 1.0

    def avg_queue_length(self) -> float:
        """平均队列长度 Lq = (cρ)^c * ρ * Pw / (c! * (1-ρ)^2)"""
        if not self.is_stable():
            return float('inf')
        c = self.c
        rho = self.rho
        pw = self.erlang_c()

        # 数值稳定性: 当 rho 接近1时返回大值
        if rho > 0.95 or (c * rho) > 50:
            return float('inf')

        try:
            return ((c * rho) ** c * rho * pw) / (self._factorial(c) * (1 - rho) ** 2)
        except OverflowError:
            return float('inf')

    def avg_waiting_time(self) -> float:
        """平均等待时间 Wq = Lq / λ (ms)"""
        if self.lambda_ <= 0:
            return 0.0
        return (self.avg_queue_length() / self.lambda_) * 1000  # 转换为ms

    def avg_response_time(self) -> float:
        """平均响应时间 W = Wq + 1/μ (ms)"""
        return self.avg_waiting_time() + (1000.0 / self.mu)

    def calc(self) -> Dict[str, float]:
        """计算所有指标"""
        return {
            "rho": self.rho,
            "pw": self.erlang_c(),
            "Lq": self.avg_queue_length(),
            "Wq": self.avg_waiting_time(),
            "W": self.avg_response_time(),
            "stable": self.is_stable()
        }

    @staticmethod
    def _factorial(n: int) -> float:
        """Calculate factorial of n."""
        if n <= 1:
            return 1.0
        result = 1.0
        for i in range(2, n + 1):
            result *= i
        return result

    def __repr__(self) -> str:
        return (f"MMCQueue(lambda={self.lambda_}, mu={self.mu}, "
                f"c={self.c}, rho={self.rho:.4f})")
