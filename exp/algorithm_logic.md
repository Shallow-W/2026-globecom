# 云边协同部署与路由算法逻辑

> **重要说明：** 零成本代理（GSTC）已在 xlsx 的 `proxy_score` 列中给出，无需额外实现。
> 仿真直接使用 proxy_score 作为精度潜力评分。

## 0. 代码文件结构

```
exp/scripts/
├── cfg.py         # 配置参数（节点资源、SLA、权重等）
├── data.py        # 数据加载与归一化
├── topo.py        # 网络拓扑构建（Waxman / 分层树状）
├── traffic.py     # 流量生成器（平稳 / 潮汐 / 突发）
├── algo.py        # 部署算法 + 路由算法（6种组合）
├── sim.py         # 仿真器（每时隙预计算候选 + 路由 + 统计）
├── exp.py         # 实验运行器 + 图表生成
└── main_simulator.py  # （旧版，可忽略）
```

---

## 1. 目标与输入

本算法面向论文场景：在严格 SLA 下，联合完成"子网部署决策 + 请求路由决策"，兼顾精度、时延、内存、算力和云边拉取开销。对齐论文 A (TPDS 2023) 和论文 B (TSC 2024) 的实验设计。

---

## 2. 数据与任务设置

数据源：`exp/data/evaluation_tables_20260325_163931.xlsx`

**7 个任务表：**
- class_scene (top1)
- class_object (top1)
- room_layout (neg_loss)
- jigsaw (top1)
- segmentsemantic (mIOU)
- normal (ssim)
- autoencoder (ssim)

**候选架构：每表 4096 个**，字段映射：
- `architecture` → 架构唯一编码
- `proxy_score` → $S_k^{proxy}$（在线选择主评分）
- `task_final_performance` → 离线真实性能（评估指标）
- `model_params` → $P_k^{model}$（内存占用代理）
- `flops` → $F_k^{flops}$（算力开销代理）
- `epochs_to_reach_avg_final_performance` → 可选收敛稳定性代理

---

## 3. 网络拓扑（对齐论文 A 和论文 B）

### 小拓扑（对齐论文 A：Atlanta，15 节点）
```
N = 15 节点，按地理坐标构建 Waxman 随机图
边(i,j)概率：P(i,j) = β * exp(-d(i,j) / (α * L))
α = 0.5, β = 0.2（保证连通性）
节点间时延 L_{i,j}：按欧氏距离 × 0.01 ms（边延迟 1-10ms 量级）
云端节点 id=0，到所有边缘节点时延 20ms（模拟跨域回程）
```

### 大拓扑（对齐论文 B：ta2，65 节点）
```
N = 65 节点 + 560 用户设备
骨干网：分层树状结构，根为云端
边缘节点按区域分组（8 区域 × 8 节点 + 1 云）
区域内部延迟 2ms，区域间延迟 10ms
```

### 节点异构资源（对齐论文 A 的处理器核心 8-64）
```
三档边缘节点容量：
- 小型（M_i_max=2GB, C_i_max=10 GFLOPS）：5 个
- 中型（M_i_max=4GB, C_i_max=30 GFLOPS）：7 个
- 大型（M_i_max=8GB, C_i_max=100 GFLOPS）：3 个
```

---

## 4. 流量模型（对齐论文 A/B 的 trace-driven + 动态到达率）

### 流量类型（与论文 A/B 一致）

**1. 平稳流（Poisson，对齐论文 A）**
```
λ_i(t) 在 [10, 50] req/s 区间随机设定
每个请求持续 1-5 个时隙
```

**2. 潮汐流（周期性，对齐论文 B 的长时隙观测）**
```
λ_i(t) = λ_base × (1 + 0.8 × sin(2πt/T_period))
T_period = 100 时隙（模拟流量潮汐）
高峰时段 λ_max = 150 req/s
```

**3. 突发流（脉冲，对齐论文 A 的请求规模 10-1000）**
```
在特定 20 时隙窗口注入高峰
λ_burst = 200 req/s（模拟突发流量）
其余时段 λ_normal = 30 req/s
```

### 请求到达（对齐论文 A/B 的仿真方式）
```
每时隙新到达请求数：Poisson(λ_avg)
请求类型：随机从 7 个任务中均匀采样
每个请求的服务时长：指数分布，均值 5 时隙
SLA 死线 T_SLA：200ms（通用）/ 100ms（时间敏感任务如 jigsaw）
```

---

## 5. 核心算法实现（对齐两篇论文 + 你的论文）

### 5.1OURS（你的论文 CEDR）

**部署阶段：**
1. 状态感知：采集 $M_i^{max}(t), \lambda_i(t), C_i^{max}$
2. 动态算力红线：
   $F_i^{max}(t) = C_i^{max} / (\lambda_i(t) + 1/T_{SLA})$
3. 硬约束过滤：剔除 $P_k^{model} > M_i^{max}(t)$ 或 $F_k^{flops} > F_i^{max}(t)$ 的候选
4. 动态效用打分：
   - $w_2(t) = \alpha \cdot \exp(\max(0, (\lambda_i(t) - \lambda_{th}) / \lambda_{th}))$
   - $w_3(t) = \beta \cdot M_{used}(t) / M_{total}$
   - $w_1(t) = 1 / (1 + w_2(t) + w_3(t))$
   - $U_{dep}(i,k,t) = w_1(t) \widehat{S}_k - w_2(t) \widehat{F}_k - w_3(t) \widehat{P}_k$

**路由阶段：**
- 联合路由效用：$U_{route}(i,k,t) = \theta_1 \widehat{S}_k - \theta_2 \widehat{R}_{i,k}(t) - \theta_3 \widehat{P}_k$
- 端到端时延：$R_{i,k}(t) = L_{user,i}(t) + T_{i,k}(t) + \mathbb{1}[k \notin cache_i] \cdot D_{pull}(i,k,t)$
- 排队服务时延：$T_{i,k}(t) = 1 / (\mu_{i,k} - \lambda_i(t)), \mu_{i,k} = C_i^{max} / F_k^{flops}$
- 拉取开销：$D_{pull}(i,k,t) = \rho \cdot P_k^{model} / B_{cloud,i}(t) + L_{cloud,i}(t)$

**输出：** 联合最大化 $\arg\max_{i,k} U_{route}(i,k,t)$

---

### 5.2 基线1：论文 A 算法（启发式部署 + Dijkstra 路由）

**对齐论文 A 的 GMDA-RMPR 思路：**

**部署阶段（GMDA 变体）：**
```
Step 1: 计算每种服务（任务类型）的总请求量 Q_q = sum(λ_q(t))
Step 2: 按 Q_q 降序排列服务（与服务实例数量排序对应）
Step 3: 对每个服务 q，找到与其他节点平均距离最短的节点 i*：
        i* = argmin_i (sum_j L_{i,j} / N)
Step 4: 若节点 i* 资源不足（不满足硬约束），顺延到次优节点
```

**路由阶段（RMPR 变体）：**
```
对每个请求，使用 Dijkstra 算法从用户节点出发找到目标服务实例的最短路径
路径选择标准：最小化累计端到端时延
若同一节点有多个服务实例，按负载均衡分发
```

---

### 5.3 基线2：论文 B 算法（贪心分配 + 最近节点路由）

**对齐论文 B 的 Greedy 思路：**

**部署阶段：**
```
每到来一个新服务实例请求：
  找到当前资源利用率最低的节点：
  i* = argmin_i (used_resources_i / total_resources_i)
  将服务实例部署在 i*
```

**路由阶段：**
```
每个请求直接路由到物理距离最近的节点：
i* = argmin_i L_{user,i}(t)
若该节点无目标服务实例，则路由失败（请求丢弃）
```

---

### 5.4 基线3：Static-BestProxy

```
全程固定选择每任务 proxy_score 最高的架构
不做任何动态重配置
每节点仅保留一个"最优架构"
```

### 5.5 基线4：Resource-First

```
仅按 flops 与 params 最小优先选择
忽略 proxy_score
在满足硬约束的候选中，选择 P_k + F_k 最小的架构
```

### 5.6 基线5：Accuracy-First

```
仅按 proxy_score 最大优先
超约束（内存/算力）则回退到次优
直到找到满足硬约束的候选
```

---

## 6. 评测指标（对齐论文 A/B 的五维评估）

### 主指标（论文 A 的核心指标）
1. **Avg E2E Latency** - 平均端到端时延（含排队、推理、拉取）
2. **Request Success Rate** - 请求成功率（被成功服务的比例）
3. **Deployment Cost** - 部署成本（激活节点数 × 部署实例数）
4. **SLA Violation Rate** - 超过 T_SLA 的请求比例
5. **Throughput** - 单位时间完成请求数

### 辅助指标（论文 B 的扩展指标）
6. **Node Utilization** - 节点资源利用率（CPU/内存）
7. **Pull/Reload Overhead** - 云边拉取次数与平均重载延迟
8. **Iteration/Runtime Overhead** - 算法执行时间开销
9. **Stability (论文 B)** - 时隙间指标方差（稳定性）
10. **Per-Task Performance** - 按任务类型分组的真实性能均值

---

## 7. 实验参数设置

### 7.1 基础参数（来自论文 A/B 经验值）

| 参数 | 值 | 来源 |
|------|-----|------|
| T_SLA | 200ms（通用）/ 100ms（jigsaw） | 论文 A/B 通用设置 |
| λ_th | λ历史均值的70%分位 | 论文 B 临界阈值 |
| 云边带宽 B_cloud | 100 Mbps（高配）/ 50 Mbps（低配） | 类比论文 B |
| 云边时延 L_cloud | 20ms（区域内）/ 50ms（区域间） | 类比论文 A 5%约束 |
| ρ（传输系数） | 1.0（未压缩）/ 0.3（量化压缩） | 论文 B |
| 时隙长度 | 1 秒 | 论文 B 100s周期对应 |
| α（延迟惩罚系数） | 1.0 | 论文 B 权重默认 |
| β（内存惩罚系数） | 1.0 | 论文 B 权重默认 |
| θ_1,θ_2,θ_3 | 0.5, 0.35, 0.15 | 论文 B 效用加权 |

### 7.2 扫参变量（对齐论文 A/B 的多维扫参）

**按论文 A/B 推荐的核心扫参：**
- 请求规模：100 / 500 / 1000 / 5000 并发请求
- 节点数量：15（小拓扑）/ 65（大拓扑）
- 到达率 λ：10 / 50 / 100 / 200 req/s
- 节点容量档位：低 / 中 / 高 三档
- 流量模式：平稳 / 潮汐 / 突发
- 缓存容量 K：5 / 10 / 20 个架构/节点

---

## 8. 仿真流程（每时隙执行）

```text
Algorithm CEDR_Simulator
Input: arch_tables[7], topology, traffic_generator, algorithm

for each time slot t:
    # 1. 流量生成（对齐论文 A trace-driven）
    requests[t] = generate_traffic(t, traffic_mode)

    # 2. 节点状态更新
    for each node i:
        update_M_i_max(t)    # 内存抖动 ±10%
        update_lambda_i(t)    # 按流量模式更新到达率
        compute_F_i_max(t)    # 动态算力红线

    # 3. 部署决策（按算法类型）
    if algorithm == "OURS":
        deploy_cedr(nodes, arch_tables)
    elif algorithm == "HEURISTIC_A":
        deploy_gmda(nodes, requests)  # 论文A: 实例数降序→平均距离最短
    elif algorithm == "GREEDY_B":
        deploy_greedy(nodes, requests)  # 论文B: 资源利用率最低
    elif algorithm == "STATIC":
        deploy_static(nodes, arch_tables)
    elif algorithm == "RESOURCE_FIRST":
        deploy_resource_first(nodes, arch_tables)
    elif algorithm == "ACCURACY_FIRST":
        deploy_accuracy_first(nodes, arch_tables)

    # 4. 路由决策（按算法类型）
    for each request r in requests[t]:
        if algorithm == "OURS":
            route_cedr(r, nodes)
        elif algorithm == "HEURISTIC_A":
            route_dijkstra(r, nodes)  # 论文A: Dijkstra最短路径
        elif algorithm == "GREEDY_B":
            route_nearest(r, nodes)   # 论文B: 最近节点
        else:
            route_default(r, nodes)

        # 5. 服务与统计
        if service_success(r):
            record_latency(r)
            record_success()
        else:
            record_failure()

    # 6. 更新缓存状态
    update_cache_states(nodes)

    # 7. 汇总时隙统计
    record_slot_metrics(t, nodes)

return metrics (latency, success_rate, cost, sla_violation, util, overhead)
```

---

## 9. 与 xlsx 数据的直接映射

| xlsx 字段 | 算法使用方式 |
|-----------|------------|
| architecture | 架构唯一ID，用于缓存键 |
| proxy_score | 在线选择主评分 $S_k^{proxy}$ |
| task_final_performance | 离线评估ground truth，按任务类型分metric |
| model_params | 内存约束 $P_k^{model}$ |
| flops | 算力约束 $F_k^{flops}$ |
| epochs_to_reach_avg | 可选：收敛稳定性惩罚 |

**注意：** 所有 proxy_score / flops / params 在跨表联合评估前，需对每个任务表做 min-max 归一化（全局归一化到 [0,1]），保证跨任务的公平比较。

---

## 10. 输出格式

仿真完成后输出：
- `exp/results/metrics_summary.csv` - 主指标对比表
- `exp/results/latency_cdf.png` - 时延CDF分布（论文B风格）
- `exp/results/timeseries.png` - 关键指标时间序列（论文B风格）
- `exp/results/algorithm_comparison.png` - 柱状图对比（论文A风格）
- `exp/results/sensitivity_analysis.png` - 扫参敏感性分析
- `exp/results/per_task_performance.csv` - 分任务性能详情
