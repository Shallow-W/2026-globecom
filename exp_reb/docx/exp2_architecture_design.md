# exp_2 架构设计文档（待迁移到 exp_reb）

> 本文档详细记录 exp_2 (`main G-（静态-改进算法-定）(1).py`) 的核心架构和设计决策，作为重建 exp_reb 的参考依据。

---

## 一、全局常量定义

```python
N_nodes = 3                        # 边缘节点数量（固定3个）
N_versions = 5                     # 每个任务的候选模型版本数（固定5个）
NODE_FLOPS_CAPACITY = 200 * 10**9 # 单节点算力上限 200 GFLOPs/s
MAX_NODE_PARAMS = 150_000_000     # 单节点内存容量 150M params
COMM_DELAY_CROSS_NODE = 0.02      # 跨节点通信延迟 0.02s (20ms)
```

**设计要点**：
- 节点数 N_nodes 是固定的，不随实验参数变化
- 模型版本数 N_versions 固定为 5，与 Excel 中的 top-5 模型对应
- 通信延迟是固定的全局常量，不区分具体链路

---

## 二、数据层

### 2.1 Excel 数据模型

每个 task（如 `class_scene`）对应 Excel 中的一个 sheet，读取字段：

| 字段 | 含义 | 用途 |
|------|------|------|
| `architecture` | 模型架构名 | 版本标识 |
| `proxy_score` | 代理分数 | 选模型时参考 |
| `model_params` | 参数量 | 内存占用计算 |
| `flops` | 计算量 | 服务率 μ 的计算 |
| `task_final_performance` | 最终精度 | 归一化 QoS 的原始值 |

### 2.2 数据归一化

```python
normalized_qos = 0.1 + 0.9 * (task_final_performance - min) / (max - min)
```

- 最低精度 → 0.1，最高精度 → 1.0
- 保留 0.1 保底，防止归一化后精度为 0

### 2.3 数据扩充

```python
original_tasks = list(tasks_data.keys())
tasks_list = copy.deepcopy(original_tasks)
while len(tasks_list) < max_required_tasks:
    # 循环复用原始任务类型，复制其模型数据
    base_task = original_tasks[idx % len(original_tasks)]
    tasks_data[new_task_name] = copy.deepcopy(tasks_data[base_task])
```

- 目的：支持 num_types > 7（原始任务类型数）的场景
- 每个扩展任务复制原始任务的 5 个模型版本数据
- `max_required_tasks=50` 支持最大 50 种任务类型

---

## 三、服务链生成

### 3.1 函数签名

```python
def generate_user_chains(available_tasks, num_types, length, total_rate):
    chains = []
    current_tasks = available_tasks[:num_types]   # 取前 num_types 个任务
    num_chains = 4                                # 固定 4 条链
    rates = np.random.dirichlet(...) * total_rate # Dirichlet 分配到达率
    rates = np.maximum(1, np.round(rates)).astype(int)
    rates[0] += total_rate - np.sum(rates)        # 保证总和精确等于 total_rate
    for i in range(num_chains):
        chain = random.choices(current_tasks, k=length)  # 有放回采样
        chains.append({'chain': chain, 'rate': rates[i]})
    return chains, current_tasks
```

### 3.2 关键设计点

| 设计 | 值 | 说明 |
|------|-----|------|
| `num_chains` | **4** | 固定，简化实验变量 |
| `num_types` | 可变 | 任务池大小，对应实验中的 S |
| `length` | 可变 | 链长度，对应实验中的 K |
| `total_rate` | 可变 | 总到达率，对应实验中的 λ |
| 任务采样方式 | `choices` | **有放回**，同一任务可在链中重复出现 |
| 到达率分配 | Dirichlet | 数学上更合理的比例分配，保证每条链都有流量 |
| 每链至少 1 | `np.maximum(1, ...)` | 确保最小到达率为 1 |

### 3.3 到达率 Dirichlet 分配详解

```python
rates = np.random.dirichlet(np.ones(4)) * total_rate
rates = np.maximum(1, np.round(rates)).astype(int)
rates[0] += total_rate - np.sum(rates)
```

- Dirichlet(α=[1,1,1,1]) 等价于均匀分布
- round 后可能损失精度，用最后一项补齐差值
- 确保：Σ rates[i] = total_rate，rate[i] ≥ 1

---

## 四、适应度评估（核心）

### 4.1 染色体编码

```python
X = np.array(individual).reshape((n_tasks, N_nodes, N_versions))
# X[s_idx, e, v] = 在节点 e 上部署任务 s 使用版本 v 的实例数（整数）
```

**解读**：三维 0/1 整数矩阵，值为实例数量（可大于 1 表示多实例部署）。

### 4.2 路由概率矩阵 p

```python
p = np.zeros((n_tasks, N_nodes, N_versions))
for s_idx, task in enumerate(current_tasks):
    task_total_instances = np.sum(X[s_idx, :, :])
    for e in range(N_nodes):
        for v in range(N_versions):
            p[s_idx, e, v] = X[s_idx, e, v] / task_total_instances
```

- `p[s,e,v]` = 任务 s 的请求被路由到 (节点e, 版本v) 的概率
- 概率 = 该位置的实例数 / 总实例数
- 本质上是**按实例数比例加权**的概率路由

### 4.3 任务到达率 Lambda_s

```python
Lambda_s = {t: 0 for t in current_tasks}
for uc in user_chains:
    for task in uc['chain']:
        Lambda_s[task] += uc['rate']
```

- 每条链的到达率 `uc['rate']` 累加到链中每个任务
- 注意：同一任务可出现在多条链中，其总到达率 = Σ 各链中该任务的到达率

### 4.4 单任务 QoS 期望

```python
expected_qos = Σ_{e,v} p[s_idx, e, v] × normalized_qos[s, v]
```

按概率加权求和，得到该任务所有可能部署位置和服务版本的 QoS 期望。

### 4.5 排队时延（M/M/1 模型）

```python
mu = NODE_FLOPS_CAPACITY / tasks_data[task][v]['flops']  # 服务率（个/秒）
rate_per_inst = lam / X[s_idx, e, v]                      # 每实例到达率
# lam = Lambda_s[task] × p[s_idx, e, v]

if rate_per_inst >= mu:
    delay_node = 1.0                           # 拥塞保底 1ms
    penalty_delay += 1000 × (rate_per_inst - mu + 1.0)
else:
    delay_node = 1.0 / (mu - rate_per_inst)    # M/M/1 公式
```

**关键**：`rate_per_inst` 是**每实例到达率**，不是总到达率，这是 M/M/1 模型正确的输入。

**拥塞判定**：`rate_per_inst >= mu` 时触发拥塞惩罚。

### 4.6 通信时延

```python
for i in range(len(chain) - 1):
    p_node_t1 = Σ_v p[s1_idx, :, v]  # 任务t1在各节点的概率之和
    p_node_t2 = Σ_v p[s2_idx, :, v]  # 任务t2在各节点的概率之和
    for e1 in range(N_nodes):
        for e2 in range(N_nodes):
            if e1 != e2:  # 只有跨节点才计通信延迟
                chain_comm += p_node_t1[e1] × p_node_t2[e2] × COMM_DELAY_CROSS_NODE
```

- 先按节点汇聚概率（忽略版本差异）
- 跨节点概率乘积 × 固定通信延迟
- 同节点通信延迟 = 0

### 4.7 最终适应度

```python
total_delay = total_comp_delay + total_comm_delay
total_penalty = penalty_params + penalty_delay
fitness = total_qos - 5.0 × total_delay - total_penalty
```

**到达率加权**在链级别已完成：
```python
for uc in user_chains:
    weight = uc['rate'] / total_arrival_rate
    total_qos += chain_qos × weight       # QoS 加权求和
    total_comp_delay += chain_comp × weight
    total_comm_delay += chain_comm × weight
```

### 4.8 内存利用率

```python
total_params_used = Σ_{e,s,v} X[s,e,v] × model_params[s,v]
mem_utilization = total_params_used / (N_nodes × MAX_NODE_PARAMS)
```

- 分母：节点数 × 单节点容量 = 全局总容量
- 分子：所有已部署实例的参数量之和

---

## 五、适应度返回值

```python
return fitness, total_delay, total_qos, total_penalty,
       total_comp_delay, total_comm_delay, mem_utilization, status
```

| 返回值 | 含义 | 单位 |
|--------|------|------|
| `fitness` | 适应度 = Q - 5×D - penalty | 无 |
| `total_delay` | 总时延 = 排队 + 通信 | 秒 |
| `total_qos` | 加权 QoS | 无 |
| `total_penalty` | 总惩罚 = 参数惩罚 + 延迟惩罚 | 无 |
| `total_comp_delay` | 计算（排队）时延 | 秒 |
| `total_comm_delay` | 通信时延 | 秒 |
| `mem_utilization` | 全局内存利用率 | 0~1 |
| `status` | 状态标志 | dict |

---

## 六、遗传算法核心

### 6.1 基因结构

```
个体长度 = n_tasks × N_nodes × N_versions
每个基因值 = 整数（0 或 1，有时更大表示多实例）
```

### 6.2 初始种群

```python
# 10% 用最小模型初始化
ind_tiny[s_idx, node, proxy_knowledge[task]['params']] = 1

# 10% 用最高精度模型初始化
ind_qos[s_idx, node, proxy_knowledge[task]['qos']] = 1

# 其余随机
ind = np.random.randint(0, 2, size=GENES_LEN)
```

**proxy_knowledge 预计算**：
```python
proxy_knowledge[task] = {
    'qos':    argmax normalized_qos,   # 最高精度版本索引
    'flops':  argmin flops,            # 最快版本索引
    'params': argmin model_params       # 最小版本索引
}
```

### 6.3 修复算子（核心！）

**问题**：变异/交叉可能产生内存溢出或服务断供的无效解。

**解决**：每次变异后强制修复：

```python
def repair_individual(X_ind):
    # Step 1: 服务断供修复
    for s_idx in range(n_tasks):
        if np.sum(X[s_idx]) == 0:  # 该任务没有任何实例
            best_node = argmin(node_usage)
            X[s_idx, best_node, proxy_knowledge[task]['params']] = 1  # 部署最小模型

    # Step 2: 内存超限修复
    for e in range(N_nodes):
        while node_usage[e] > MAX_NODE_PARAMS:
            # 揪出占用最大的实例
            # 优先降级（换小模型），否则删除实例
```

**修复策略优先级**：
1. 能降级就不删除（保住服务可用性）
2. 降级到最小模型（proxy_knowledge['params']）
3. 无论如何保住物理内存不爆

### 6.4 变异策略（自适应）

```python
if status['congested'] and not status['oom']:
    # 拥塞但内存够 → 横向扩展最高精度模型
    X[s, e, proxy_knowledge[task]['qos']] += 1
elif status['congested'] and status['oom']:
    # 拥塞且内存爆 → 删除所有实例，部署最快模型
    X[s, e, :] = 0
    X[s, e, proxy_knowledge[task]['flops']] = 1
elif status['oom'] and not status['congested']:
    # 内存爆但不拥塞 → 删除所有实例，部署最小模型
    X[s, e, :] = 0
    X[s, e, proxy_knowledge[task]['params']] = 1
else:
    # 正常状态 → 60%概率部署高精度模型，40%随机
```

---

## 七、扰动实验设计

### 7.1 三个扰动因子

| 实验 | 固定参数 | 扰动参数 | 扰动范围 |
|------|---------|---------|---------|
| 实验1（到达率） | num_types=10, length=4 | total_rate | 100,200,300,400,500,600,700,800 |
| 实验2（链长度） | num_types=10, rate=200 | length | 3,4,5,6,7,8,9,10 |
| 实验3（服务种类） | length=4, rate=200 | num_types | 10,20,30,40,50,60,70,80 |

### 7.2 扰动时服务池控制

```python
# 实验3：num_types=80 时
current_tasks = available_tasks[:80]  # 从任务池前80个中采样
# available_tasks 在 load_and_prepare_data 时已扩展到 max_required_tasks=50
# 但 num_types 最大只到 80，所以任务池够用
```

---

## 八、输出指标体系

### 8.1 每行输出格式

```python
D={delay:.4f}s (排队={comp_d:.4f}s, 通信={comm_d:.4f}s) | Q={qos:.4f} | 内存利用={mem_u:.2%} | 违规惩罚={pen:.1f}
```

### 8.2 CSV 列

```python
'Experiment': 实验名称（三个实验）
'Variable_Value': 扰动参数值
'Total_Delay_D': 总时延 (s)
'Avg_QoS_Q': 加权 QoS
'Comp_Delay': 计算（排队）时延 (s)
'Comm_Delay': 通信时延 (s)
'Mem_Utilization': 内存利用率
'Penalty_Score': 总惩罚分
```

### 8.3 Summary 输出

按实验分组求各项均值：

```python
summary_df = df_results.groupby('Experiment').mean().reset_index()
```

---

## 九、与 exp_reb 当前实现的对比

| 维度 | exp_2 (正确) | exp_reb (需改造) |
|------|-------------|-----------------|
| 排队模型 | per-instance M/M/1 | MMCQueue (有bug) |
| 通信延迟 | 概率矩阵 × 固定延迟 | 拓扑最短路径 (不对) |
| 到达率分配 | Dirichlet 分配 | uniform 独立 |
| 内存利用率 | Σ params / (N×MAX) | CPU 利用率 (错) |
| 扰动实验 | 固定基线参数，只变一个 | 链结构随机变化 |
| 适应度 | 加权求和 | 简单平均 |
| QoS 优化 | normalized_qos 加权 | 无 (固定0.53) |
| 修复算子 | 主动修复 | 无 |
| 模型选择 | proxy 引导 | 随机/贪婪 |

---

## 十、重建 exp_reb 的关键设计建议

### 10.1 架构原则

1. **评估模型与算法分离**：评估模型只计算指标，不负责搜索
2. **联合优化目标**：`fitness = QoS - 5×Delay - Penalty`，不要分层求解
3. **路由概率显式化**：通过 `p[s,e,v]` 矩阵表达请求分配，清晰且可调试
4. **约束处理前置**：用硬约束过滤替代事后修复（如 Our 算法的 utility 函数）

### 10.2 核心类设计建议

```
DataLayer
├── ModelLibrary        # Excel 读取 + normalized_qos 计算
├── TaskPool            # 任务池，支持按 num_types 切片
└── ServiceChainGen    # Dirichlet 到达率分配 + choices 采样

EvaluationModel
├── ChromosomeEncoder    # X[s,e,v] 整数矩阵
├── RoutingMatrix       # p[s,e,v] 概率矩阵
├── DelayCalculator    # per-instance M/M/1 + 拥塞惩罚
├── QoSCalculator      # normalized_qos 加权求和
├── MemCalculator       # Σ params / 全局容量
└── FitnessAggregator  # 到达率加权求和

AlgorithmModule
├── RepairOperator      # 物理约束修复
├── MutationStrategy    # 自适应变异（根据状态）
└── GA / Heuristic     # 搜索策略

ExperimentModule
├── PerturbationRunner  # 三因子扰动
├── MetricsRecorder    # 全指标记录
└── ResultExporter    # CSV + JSON
```

### 10.3 扰动实验配置模板

```python
EXPERIMENTS = {
    "arrival_rate": {
        "param": "total_arrival_rate",
        "values": [100, 200, 300, 400, 500, 600, 700, 800],
        "fixed": {"num_types": 10, "length": 4}
    },
    "chain_length": {
        "param": "length",
        "values": [3, 4, 5, 6, 7, 8, 9, 10],
        "fixed": {"num_types": 10, "total_arrival_rate": 200}
    },
    "num_types": {
        "param": "num_types",
        "values": [10, 20, 30, 40, 50, 60, 70, 80],
        "fixed": {"length": 4, "total_arrival_rate": 200}
    }
}
```

### 10.4 关键公式速查

```python
# 1. mu 计算
mu = NODE_FLOPS_CAPACITY / flops

# 2. per-instance 到达率
rate_per_inst = Lambda_s[s] × p[s,e,v] / X[s,e,v]

# 3. M/M/1 排队时延
delay = 1 / (mu - rate_per_inst)  if rate_per_inst < mu else 1.0

# 4. 拥塞惩罚
penalty_delay += 1000 × (rate_per_inst - mu + 1.0)

# 5. 内存利用率
mem_util = Σ X[s,e,v] × params[s,v] / (N_nodes × MAX_NODE_PARAMS)

# 6. 到达率权重
weight = chain_rate / total_arrival_rate

# 7. 适应度
fitness = total_qos - 5.0 × total_delay - total_penalty
```
