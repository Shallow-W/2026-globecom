# 云边协同部署与路由实验设计文档

> 本文档描述基于论文算法（globecom.tex）的仿真实验完整设计，对齐论文 A (TPDS 2023) 和论文 B (TSC 2024) 的实验方法。

---

## 1. 实验目标

1. 验证联合部署+路由策略相比分离策略能提升请求成功率、降低 SLA 违约率
2. 验证动态算力红线和自适应权重机制在流量波动下的有效性
3. 对比 6 种算法在不同流量模式（平稳/潮汐/突发）下的表现
4. 分析云边拉取开销对系统可用性的影响

---

## 2. 评测指标

| 指标 | 说明 |
|------|------|
| **Success Rate** | 请求成功率 = 成功服务请求数 / 总请求数 |
| **Avg E2E Latency (ms)** | 平均端到端时延（含排队 + 推理 + 拉取） |
| **SLA Violation Rate** | SLA 违约率 = 超时请求数 / 总请求数 |
| **Throughput (req/s)** | 单位时间成功服务请求数 |
| **Pull Count** | 云边拉取总次数 |
| **Avg Pull Delay (ms)** | 平均拉取延迟 |
| **Node Utilization** | 节点内存利用率（均值/标准差） |
| **Per-Task Performance** | 分任务真实性能（task_final_performance）均值 |

---

## 3. 数据与任务

**数据源**: `exp/data/evaluation_tables_20260325_163931.xlsx`

| 任务 | 评估指标 | SLA (ms) |
|------|---------|----------|
| class_scene | top1 ↑ | 200 |
| class_object | top1 ↑ | 200 |
| room_layout | neg_loss ↓ | 200 |
| jigsaw | top1 ↑ | 100（严格） |
| segmentsemantic | mIOU ↑ | 200 |
| normal | ssim ↑ | 200 |
| autoencoder | ssim ↑ | 200 |

**候选架构**: 每任务 4096 个
**零成本代理**: 直接使用 xlsx 中的 `proxy_score` 列（已由 GSTC 评估完成）

---

## 4. 网络拓扑

### 4.1 小拓扑（N=15，对齐论文 A Atlanta）
- 15 个边缘节点 + 1 云端节点
- Waxman 随机图（α=0.5, β=0.2），延迟 1-10ms
- 云边延迟：20ms
- 节点异构三档：
  - 小型 5 个：2GB 内存, 10 GFLOPS
  - 中型 7 个：4GB 内存, 30 GFLOPS
  - 大型 3 个：8GB 内存, 100 GFLOPS

### 4.2 大拓扑（N=65，对齐论文 B ta2）
- 65 个边缘节点 + 1 云端节点
- 分层树状：8 区域 × 8 节点
- 区域内延迟 2ms，区域间延迟 10ms
- 云边延迟：20ms
- 统一中型节点配置

---

## 5. 流量模型

| 模式 | 描述 | 对齐文献 |
|------|------|---------|
| **steady** | Poisson(λ=3 req/s)，低方差平稳流 | 论文 A 基础设置 |
| **tidal** | λ(t) = 3 × (1 + 0.8 sin(2πt/100))，周期性潮汐 | 论文 B 100s 周期 |
| **burst** | 20-40 时隙高峰 λ=15，其余 λ=3，突发冲击 | 论文 A 请求规模 |

每时隙（1 秒）按 Poisson(λ_avg) 采样新请求，服务时长指数分布（均值 5 时隙）。

---

## 6. 算法实现（6 种）

### 6.1 OURS（你的论文算法）

**部署阶段**：
1. 动态算力红线：$F_i^{max}(t) = C_i^{max} / (\lambda_i(t) + 1/T_{SLA})$
2. 硬约束过滤：剔除 $P_k^{model} > M_i^{max}(t)$ 或 $F_k^{flops} > F_i^{max}(t)$ 的候选
3. 效用函数打分：
   - $w_2(t) = \alpha \cdot \exp(\max(0, (\lambda - \lambda_{th}) / \lambda_{th}))$
   - $w_3(t) = \beta \cdot M_{used} / M_{total}$
   - $w_1(t) = 1 / (1 + w_2 + w_3)$
   - $U = w_1 \cdot \widehat{S}_k^{proxy} - w_2 \cdot \widehat{F}_k - w_3 \cdot \widehat{P}_k$

**路由阶段**：联合效用最大化
- $U_{route} = \theta_1 \cdot \widehat{S} - \theta_2 \cdot \widehat{R}_{E2E} - \theta_3 \cdot \widehat{P}$
- 端到端时延含排队 + 推理 + 可选云边拉取

### 6.2 HEURISTIC_A（论文 A 基线）
- **部署**：按 proxy_score 最高选择（近似"平均距离最短"）
- **路由**：Dijkstra 最短路径

### 6.3 GREEDY_B（论文 B 基线）
- **部署**：贪心选 flops+params 最小的轻量架构
- **路由**：最近节点转发

### 6.4 STATIC
- 全程固定每任务 proxy_score 最高的架构，不做动态重配置

### 6.5 RESOURCE_FIRST
- 仅按 flops+params 最小优先，忽略 proxy_score

### 6.6 ACCURACY_FIRST
- 仅按 proxy_score 最大优先，超约束则回退

---

## 7. 排队模型

使用 M/M/1 近似：
- 服务率：$\mu_{i,k} = C_i^{max} / F_k^{flops}$（req/s）
- 到达率：$\lambda_i(t)$（每时隙更新）
- 稳定条件：$\mu > \lambda$
- 排队时延：$T_{queue} = 1 / (\mu - \lambda)$（秒），超出稳定区则标记 SLA 违约

---

## 8. 实验分组

| 实验 | 拓扑 | 流量 | 目的 |
|------|------|------|------|
| Exp 1 | N=15 | steady | 基础性能对比 |
| Exp 2 | N=15 | tidal | 周期性负载下稳定性 |
| Exp 3 | N=15 | burst | 突发流量冲击测试 |
| Exp 4 | N=65 | tidal | 规模化验证 |

---

## 9. 代码文件结构

```
exp/scripts/
├── cfg.py         # Config 类：所有可调参数（节点资源、λ、SLA、权重）
├── data.py        # load_architecture_tables()：加载 xlsx 并做 min-max 归一化
├── topo.py        # Topology 类：Waxman 小拓扑 / 分层大拓扑
├── traffic.py     # TrafficGenerator 类：steady / tidal / burst 流量生成
├── algo.py        # 6 种部署算法 + 3 种路由算法的实现
├── sim.py         # Simulator 类：仿真主循环（候选预计算 → 路由 → 统计）
├── exp.py         # 实验运行器 + CSV/图表输出
└── main_simulator.py  # （旧版，可忽略）
```

**模块依赖**：
```
cfg.py
  └── data.py
        └── topo.py
              └── traffic.py
                    └── algo.py
                          └── sim.py
                                └── exp.py
```

---

## 10. 快速运行

```bash
cd exp/scripts
python -c "
from cfg import Config
from data import load_architecture_tables
from topo import Topology
from algo import ALGORITHM_MAP
from sim import Simulator
import numpy as np, random

cfg = Config()
cfg.n_slots = 20
tables = load_architecture_tables(cfg.data_path)

for algo_name, (dc, rc) in ALGORITHM_MAP.items():
    np.random.seed(42); random.seed(42)
    topo = Topology(cfg, 'small')
    sim = Simulator(cfg, topo, tables, dc, rc, 'steady')
    r = sim.run()
    print(f'{algo_name}: lat={r[\"avg_latency_ms\"]:.1f}ms succ={r[\"success_rate\"]:.3f}')
"
```

---

## 11. 典型结果（20 slots, N=15, steady）

| 算法 | 时延 (ms) | 成功率 | SLA 违约率 | 拉取次数 |
|------|----------|--------|-----------|---------|
| **OURS** | 66.7 | **97.5%** | 2.5% | 14 |
| HEURISTIC_A | **8.0** | 67.5% | 0.0% | 15 |
| GREEDY_B | 13.7 | 63.6% | 0.8% | 11 |
| STATIC | 16.4 | 63.6% | 0.8% | 11 |
| RESOURCE_FIRST | 13.7 | 63.6% | 0.8% | 11 |
| ACCURACY_FIRST | 16.4 | 63.6% | 0.8% | 11 |

**结论**：OURS 通过动态权重自适应降级，在保持较高成功率（97.5%）的同时实现了精度与资源的平衡；HEURISTIC_A 虽然时延最低但成功率仅 67.5%；其他基线均收敛到 63.6%。

---

## 12. 调参建议

| 参数 | 建议范围 | 说明 |
|------|---------|------|
| λ_base | 1.0 ~ 5.0 | 保证小型节点也能稳定服务 |
| α | 0.5, 1.0, 2.0 | 延迟惩罚系数 |
| β | 0.5, 1.0, 2.0 | 内存惩罚系数 |
| θ1,θ2,θ3 | 0.5, 0.35, 0.15（默认） | 路由效用权重 |
| cache_k | 5, 10, 20 | 每节点缓存架构数 |
| T_SLA | 100 ~ 300 ms | 根据任务类型调整 |
