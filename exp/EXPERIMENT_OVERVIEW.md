# 云边协同部署与路由实验整体逻辑文档

> 本文档从顶层视角描述实验的整体逻辑、数据流向、模块关系，便于理解实验框架的设计思路。

---

## 1. 实验要回答的核心问题

本实验旨在验证**联合部署+路由协同优化**相比分离策略的系统收益。具体回答：

1. **静态部署**：当网络负载固定时，我们的算法（OURS）是否能在时延、成功率、资源利用率上优于基线？
2. **鲁棒性**：当流量波动（到达率变化、请求长度变化、任务数变化）时，各算法的表现是否稳定？
3. **资源效率**：在同等服务质量（时延约束）下，各算法消耗的资源是否有差异？
4. **规模扩展**：算法在小/中/大三档网络规模下是否都能保持有效性？

---

## 2. 实验类型总览

| 实验类型 | 目的 | 对应 GSTC 论文 |
|---------|------|--------------|
| **动态负载实验** | 验证基础性能（steady/tidal/burst 三种流量模式） | Fig.5-7 动态时序评估 |
| **单变量扰动实验** | 逐一改变单一变量，观察各算法鲁棒性 | Fig.4 静态部署单变量扰动 |
| **等时延资源效率实验** | 同等时延目标下比较资源消耗 | Fig.8 资源效率对比 |
| **规模扩展实验** | 验证算法在 N=15/30/65 下都有效 | 多规模验证 |

---

## 3. 数据流与模块关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           实验数据流                                      │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  evaluation  │  xlsx: 每任务 4096 个候选架构
    │  _tables     │  包含: arch_id, flops, params, proxy_score,
    │  .xlsx       │         task_final_performance, proxy_score_norm 等
    └──────┬───────┘
           │ load_architecture_tables()
           │ (data.py, 返回 Dict[task_name, DataFrame])
           ▼
    ┌──────────────┐
    │    cfg.py    │  Config: 所有参数（节点资源、流量、权重、SLA）
    │   (单例)     │  参数可运行时覆盖（扰动实验时）
    └──────┬───────┘
           │
           ├────────────────────────────────────────┐
           │                │                       │
           ▼                ▼                       ▼
    ┌──────────────┐  ┌──────────────┐      ┌──────────────┐
    │   topo.py    │  │ traffic.py   │      │   algo.py    │
    │              │  │              │      │              │
    │ Topology:    │  │ TrafficGen:  │      │ 6 部署算法   │
    │ ·small(N=15) │  │ ·steady      │      │ 3 路由算法   │
    │ ·medium(N=30)│  │ ·tidal       │      │              │
    │ ·large(N=65) │  │ ·burst       │      │ 组合成       │
    │              │  │              │      │ ALGORITHM_MAP│
    │ Node:        │  │ + 扰动接口:  │      │              │
    │ ·memory_mb   │  │ ·lambda      │      │              │
    │ ·gflops      │  │ ·req_length  │      │              │
    │ ·cache       │  │ ·n_tasks     │      │              │
    │ ·used_mem    │  │              │      │              │
    └──────┬───────┘  └──────┬───────┘      └──────┬───────┘
           │                 │                       │
           │                 │                       │
           └────────┬────────┴───────────────────────┘
                    │
                    ▼
           ┌─────────────────┐
           │     sim.py      │
           │   Simulator     │
           │                 │
           │ 每时隙循环:     │
           │ 1. 流量生成     │
           │ 2. 更新 λ_i(t) │
           │ 3. 预计算候选   │
           │ 4. 请求路由     │
           │ 5. 更新统计     │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │    exp.py       │
           │                 │
           │ 4 种实验入口:   │
           │ ·run_experiment │
           │ ·run_perturb..  │
           │ ·run_equal_lat..│
           │ ·run_scale..    │
           │                 │
           │ 输出:           │
           │ ·results_*.csv   │
           │ ·fig_*.png      │
           │ ·perturb_*.csv  │
           └─────────────────┘
```

---

## 4. 各模块职责

### 4.1 `cfg.py` - 配置中心

```python
Config {
    # 拓扑
    n_small=15, n_medium=30, n_large=65
    node_types=[(5,2048,10), (7,4096,30), (3,8192,100)]  # 小拓扑异构节点

    # 流量
    lambda_base=3.0        # 基准到达率 (req/s)
    request_length_base=5  # 基准请求服务时长 (slots)
    n_task_types=7         # 激活的任务类型数

    # SLA
    T_SLA_ms=200, T_SLA_jigsaw_ms=100

    # OURS 权重
    alpha=1.0, beta=1.0, theta1=0.5, theta2=0.35, theta3=0.15
}
```

**关键设计**：所有可调参数集中在 `Config`，便于做参数敏感性分析。

---

### 4.2 `data.py` - 数据加载与归一化

输入：xlsx 文件（每任务 4096 架构）
输出：`Dict[task_name, DataFrame]`

```python
{
    'class_scene': DataFrame {
        arch_id, flops, model_params, proxy_score,
        task_final_performance,  # 真实性能（用于对比）
        proxy_score_norm,        # min-max 归一化代理分
        flops_norm, params_norm  # min-max 归一化资源
    },
    ...
}
```

**关键设计**：数据在加载时完成归一化，后续算法直接使用归一化值做比较，避免量纲不一致。

---

### 4.3 `topo.py` - 拓扑构建

| 拓扑 | 规模 | 结构 | 延迟 | 节点配置 |
|------|------|------|------|---------|
| small | 15+1 节点 | Waxman 随机图 | 1-10ms + 云边 20ms | 异构三档 |
| medium | 30+1 节点 | 分层树状 4×7 | 区域 2ms / 区间 8ms / 云边 20ms | 统一中型 |
| large | 65+1 节点 | 分层树状 8×8 | 区域 2ms / 区间 10ms / 云边 20ms | 统一中型 |

**节点运行时状态**（每个时隙更新）：
- `lambda_arrival`：当前到达率
- `used_memory_mb`：已用内存
- `cache`：已缓存架构 ID 集合
- `deployed_archs`：已部署架构列表

---

### 4.4 `traffic.py` - 流量生成

**三种模式**：

1. **steady（平稳）**：`λ(t) = λ_base * (0.8 + 0.4 * random)`，每时隙 Poisson 采样
2. **tidal（潮汐）**：`λ(t) = λ_base * (1 + 0.8 * sin(2πt/100))`，100 秒周期
3. **burst（突发）**：20-40 时隙高峰 λ=15，其余 λ=3

**扰动接口**（支持运行时叠加）：

```python
traffic_gen.set_perturbation(
    lambda_override=8.0,      # 覆盖到达率
    request_length=20,        # 覆盖请求服务时长
    n_tasks=3                 # 覆盖激活任务数
)
traffic_gen.clear_perturbation()  # 清除扰动
```

**λ_th 动态计算**：每个节点维护历史 λ 记录，取 70% 分位作为动态权重触发阈值。

---

### 4.5 `algo.py` - 算法实现

#### 部署算法（6 种）

| 算法 | 选择策略 | 说明 |
|------|---------|------|
| **OURS** | 效用函数 $U = w_1 S - w_2 F - w_3 P$ | 动态权重自适应 |
| HEURISTIC_A | proxy_score 最高 | 论文 A 基线 |
| GREEDY_B | flops+params 最小 | 论文 B 基线 |
| STATIC | 固定 proxy_score 最高 | 不做动态调整 |
| RESOURCE_FIRST | 仅 flops+params 最小 | 忽略代理分 |
| ACCURACY_FIRST | 仅 proxy_score 最高 | 超约束则回退 |

**OURS 动态权重公式**（论文核心）：

```
w2 = α * exp(max(0, (λ - λ_th) / λ_th))   # 延迟惩罚
w3 = β * (M_used / M_total)                # 内存惩罚
w1 = 1 / (1 + w2 + w3)                     # 精度归一化
```

**OURS 部署流程**：
```
1. 动态算力红线: F_max = gflops / (λ + 1/T_SLA)
2. 硬约束过滤:   剔除 params > 可用内存 或 flops > F_max 的候选
3. 效用打分:     计算所有候选的 U，选择最高
```

#### 路由算法（3 种）

| 算法 | 策略 |
|------|------|
| **OURS** | 联合效用最大化 $U_{route} = θ_1 S - θ_2 R_{E2E} - θ_3 P$ |
| HEURISTIC_A | Dijkstra 最短路径 |
| GREEDY_B | 最近节点转发（仅看拓扑距离） |

---

### 4.6 `sim.py` - 仿真器（核心引擎）

每时隙执行：

```
┌─────────────────────────────────────────────────────┐
│  for t in range(n_slots):                           │
│                                                     │
│  1. 流量生成                                         │
│     requests = traffic_gen.generate_slot(t, mode)   │
│     例: {node_1: [(class_scene, 5), ...], ...}     │
│                                                     │
│  2. 更新节点到达率                                   │
│     node.lambda_arrival = len(requests[nid]) / dt  │
│                                                     │
│  3. 预计算候选架构（每节点 × 每任务）                 │
│     slot_candidates[(nid, task)] =                  │
│       deploy_algo.filter_candidates(node, task, t)  │
│     避免每个请求都遍历 4096 架构                      │
│                                                     │
│  4. 处理请求                                         │
│     for src_nid, reqs in requests:                  │
│       for task, duration in reqs:                  │
│         target, arch, route_delay =                 │
│           routing_algo.route(...)                   │
│         if target is None: 失败计数++               │
│         else:                                       │
│           · 检查/更新缓存                            │
│           · 计算 M/M/1 排队时延                      │
│           · SLA 检查                                │
│           · 更新统计计数器                          │
│                                                     │
│  5. 时隙结束                                         │
│     · 内存释放（duration 耗尽）                     │
│     · 可选：重置 used_memory（简化处理）             │
└─────────────────────────────────────────────────────┘
```

**排队模型（M/M/1）**：

```python
μ = node.gflops / (arch.flops / 1e9)   # 服务率 (req/s)
λ = node.lambda_arrival                 # 到达率 (req/s)
T_queue = 1000 / max(μ - λ, 0.1)       # 排队时延 (ms)
if μ <= λ: 标记 SLA 违约
```

**云边拉取**：

```python
if arch_id not in node.cache:
    D_pull = params / B_cloud * 1000 + L_cloud  # 传输 + 拓扑延迟
    更新统计: pull_count++, pull_delay += D_pull
    更新缓存: cache.add(arch_id)
else:
    D_pull = 0
```

---

### 4.7 `exp.py` - 实验入口与输出

#### 实验 1：动态负载实验

```python
run_experiment(cfg, topology_scale, traffic_mode)
→ 对每种 (算法, 拓扑, 流量) 组合运行仿真
→ 输出: results_*.csv, fig_bar_*.png, fig_cdf_*.png
```

#### 实验 2：单变量扰动实验

```python
run_perturbation_experiment(cfg, perturb='arrival_rate', values=[1,2,3,5,8,12])
→ 固定其他参数，仅改变一个变量
→ 观察: 时延、成功率随扰动值的变化曲线
→ 输出: perturb_*.csv, perturb_*_summary.csv, fig_perturb_*.png

支持的 perturb 类型:
  'arrival_rate'     → λ = [1, 2, 3, 5, 8, 12] req/s
  'request_length'   → 服务时长 = [2, 5, 10, 20, 50] slots
  'n_task_types'     → 任务数 = [1, 2, 3, 5, 7]
```

#### 实验 3：等时延资源效率实验

```python
run_equal_latency_experiment(cfg, target_latency=50.0)
→ 先运行 OURS 获取基准时延
→ 对比同拓扑下各算法的资源利用率
→ 输出: equal_latency_*.csv, fig_equal_latency_comparison.png
```

#### 实验 4：规模扩展实验

```python
run_scale_experiment(cfg, scales=['small', 'medium', 'large'])
→ 在 N=15, 30, 65 三种规模下运行仿真
→ 输出: results_*_*.csv, fig_scale_comparison_*.png
```

---

## 5. 关键设计决策

### 5.1 为什么候选架构要预计算？

每个时隙有数百个请求，每个请求如果都遍历 4096 个候选架构复杂度为 `O(请求数 × 任务数 × 候选数)`，会非常慢。

优化策略：**每个时隙只预计算一次 `(node, task) → candidates`**，后续请求直接查表复用：

```python
# 优化前: 每个请求都遍历
for request in all_requests:
    candidates = filter_candidates(node, task, tables)  # 4096 次

# 优化后: 每时隙只计算一次
for nid, node in topology.nodes:
    for task in tasks:
        slot_candidates[(nid, task)] = filter_candidates(...)  # 预计算
for request in all_requests:
    candidates = slot_candidates[(nid, task)]  # 直接查表 O(1)
```

### 5.2 为什么用 M/M/1 近似而不是精确排队？

精确排队论需要跟踪每个请求的到达时间、服务时间、剩余服务时间，复杂度高。M/M/1 近似通过稳态假设提供解析解，适合大规模仿真：

```python
T_queue = 1000 / max(μ - λ, ε)  # ms
```

当 `μ <= λ`（不稳定区）时，直接标记 SLA 违约，避免数值爆炸。

### 5.3 为什么节点内存是累积的而不是真正释放的？

简化处理：简化实现中内存一旦分配就不释放（只在统计中记录）。更精确的做法是按 `duration` 计时释放，但这需要额外的生命周期管理。当前的简化在仿真时长足够长（数百时隙）时，误差可接受。

---

## 6. 输出文件说明

```
exp/results/
├── results_small_steady.csv       # 主指标 CSV
├── results_small_steady.csv
├── perturb_arrival_rate_summary.csv  # 扰动汇总
├── perturb_arrival_rate_val=1.csv
├── ...
├── fig_bar_*.png                  # 柱状图对比
├── fig_cdf_*.png                  # 时延 CDF
├── fig_perturb_*.png              # 扰动曲线
├── fig_equal_latency_comparison.png
└── fig_scale_comparison_*.png
```

---

## 7. 如何运行实验

```bash
cd exp/scripts

# 运行全部实验（动态负载 + 扰动 + 等时延 + 规模扩展）
python exp.py

# 只运行特定实验
python -c "
from exp import run_perturbation_experiment, run_equal_latency_experiment, run_scale_experiment
from cfg import Config
cfg = Config(); cfg.n_slots = 100

# 到达率扰动
run_perturbation_experiment(cfg, perturb='arrival_rate', values=[1,2,3,5,8,12])

# 请求长度扰动
run_perturbation_experiment(cfg, perturb='request_length', values=[2,5,10,20,50])

# 任务数扰动
run_perturbation_experiment(cfg, perturb='n_task_types', values=[1,2,3,5,7])

# 等时延资源效率
run_equal_latency_experiment(cfg, target_latency=50.0)

# 规模扩展
run_scale_experiment(cfg, scales=['small', 'medium', 'large'])
"
```

---

## 8. 实验结果解读指南

### 8.1 动态负载实验

关注：
- **OURS 成功率**是否显著高于基线（如 >90% vs <70%）
- **OURS SLA 违约率**是否显著低于基线
- **时延**是否在 SLA 约束内（<200ms，jigsaw <100ms）

### 8.2 单变量扰动实验

关注：
- **曲线斜率**：OURS 的时延随扰动值增加是否比基线更平缓（鲁棒性）
- **拐点**：各算法在哪个扰动值下开始急剧恶化
- **成功率崩溃点**：基线在 λ≥5 时成功率是否急剧下降

### 8.3 等时延资源效率实验

关注：
- 当时延相近时，**OURS 的内存利用率**是否更低
- 相对节省比例（应 >40%，对齐 GSTC 论文结果）

### 8.4 规模扩展实验

关注：
- **时延增长是否线性**：OURS 应比基线更平缓
- **成功率是否稳定**：OURS 应在所有规模下保持 >90%

---

## 9. 扩展方向

当前框架预留了以下扩展点：

1. **动态替换机制**：LAMRA 风格的运行时架构替换（当前 STATIC 是固定部署，OURS 是每时隙重新评估）
2. **多目标优化**：Pareto 前沿可视化（当前是加权求和）
3. **真实轨迹回放**：替代 Poisson 流量，使用真实请求 trace
4. **能耗模型**：将 GFLOPS 转为功耗，加入能耗指标
