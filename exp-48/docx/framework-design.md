# 微服务路由与部署实验框架 — 设计文档

## 1. 概述

本实验框架用于研究边缘计算环境中微服务的**联合部署与路由优化**问题。给定一个边缘节点网络和一组微服务链 (Service Chain)，框架的目标是评估不同部署策略与路由策略下系统的端到端时延性能。

框架的核心特征：

- **解析计算**：基于 Jackson 排队网络和 M/M/c 排队模型，直接通过公式计算时延，不依赖离散事件仿真
- **模块化架构**：网络拓扑、微服务定义、部署算法、路由算法、时延计算、实验运行各自独立，可单独替换
- **静态 + 动态实验**：支持参数扫描式静态实验和时变环境下的动态实验

---

## 2. 系统模型

### 2.1 网络拓扑

边缘计算网络建模为无向图 G = (V, E)：

- V = {v_1, v_2, ..., v_N}：N 个边缘节点（对称多核处理器）
- E：节点间通信链路，每条链路具有传播时延 D_{i,j}

每个节点 v 拥有两类资源：

| 资源 | 含义 | 约束 |
|------|------|------|
| CPU 核心数 C_v | 可并行运行的最大容器（实例）数 | sum_s X[v][s] <= C_v |
| GPU 显存 G_v (MB) | 每个实例独占一部分 GPU 显存 | sum_s X[v][s] * g_s <= G_v |

节点间通信时延 D_{i,j} 在 [COMM_DELAY_MIN, COMM_DELAY_MAX] 范围内随机生成，满足对称性 (D_{i,j} = D_{j,i})，对角线为零（同节点通信不计时延）。

代码实现：`network.py` → `EdgeNetwork` 类

```python
class EdgeNetwork:
    n_nodes          # 节点数
    cpu_capacity     # 各节点 CPU 核心数, shape (n_nodes,)
    gpu_capacity     # 各节点 GPU 显存 (MB), shape (n_nodes,)
    comm_delay       # 通信时延矩阵 (秒), shape (n_nodes, n_nodes), 对称
```

三种预设网络规模：

| 规模 | 节点数 | CPU/节点 | GPU/节点 (MB) |
|------|--------|----------|---------------|
| small | 3 | 16 | 8,192 |
| medium | 5 | 32 | 16,384 |
| large | 7 | 64 | 32,768 |

### 2.2 微服务模型

系统中有 M 种不同的微服务类型。每个微服务 m 具有如下属性：

| 属性 | 符号 | 说明 |
|------|------|------|
| 服务率 | mu_m | 单核心每秒可处理的请求数，服从 (5.0, 20.0) 均匀分布 |
| GPU 需求 | g_m | 每个实例独占的 GPU 显存 (MB)，服从 [128, 1024] 均匀分布 |

代码实现：`service.py` → `Service` 类

```python
class Service:
    id                # 服务编号
    service_rate      # mu: 单核心服务率
    gpu_per_instance  # g: 每实例 GPU 占用 (MB)
```

### 2.3 服务链模型

每个用户请求建模为一条**线性服务链** (Linear Service Chain)，即微服务的有序序列：

```
请求 r = (M_r, lambda_r)
```

- M_r = {m_1, m_2, ..., m_L}：长度为 L 的微服务序列，按严格顺序执行
- lambda_r：Poisson 到达率（请求/秒）

同一微服务可出现在多条链中（**服务复用**），此时该微服务的聚合到达率为所有包含它的链的到达率之和：

```
lambda_s[m] = sum_{r: m in M_r} lambda_r
```

代码实现：`service.py` → `ServiceChain` 类，`generate_service_chains()` 函数

链的生成规则：
1. 总到达率 total_rate 按 Dirichlet 分布分配给 n_chains 条链
2. 每条链的每个位置从 n_service_types 种服务中均匀随机抽取

---

## 3. 排队模型与时延计算

### 3.1 Jackson 排队网络分解

整个微服务系统建模为**开环 Jackson 排队网络**。根据 Jackson 定理，该网络可以分解为独立的 M/M/c 排队子系统，每个子系统对应一个节点上的一种微服务实例组。

分解的关键假设：
- 外部到达为 Poisson 过程
- 每个核心的服务时间服从指数分布
- 网络是无环开环的（请求最终离开系统）

### 3.2 M/M/c 排队模型

对节点 v 上的微服务 m，其 c = X[v][m] 个实例构成一个 M/M/c 排队系统：

**输入参数**：
- lambda_v：到达节点 v 的到达率 = lambda_s[m] * P[v]
- mu_m：单核心服务率
- c = X[v][m]：该节点上该服务的实例数

**计算公式**：

```
服务强度:       rho = lambda_v / (c * mu_m)           要求 rho < 1
到达负载:       a = lambda_v / mu_m
空闲概率:       P_0 = 1 / [sum_{k=0}^{c-1} a^k/k! + a^c / (c! * (1-rho))]
平均队列长度:   L_q = (a^c * rho) / (c! * (1-rho)^2) * P_0
平均排队等待:   W_q = L_q / lambda_v
平均响应时间:   W = W_q + 1/mu_m
```

**稳定性处理**：
- 若 rho >= 1（系统过载），返回固定时延 1.0s 并标记为不稳定
- 同时施加惩罚：penalty = 5.0 * (rho - 1 + 0.1)

代码实现：`queuing.py` → `mm_c_response_time()` 函数

### 3.3 单条链的端到端时延

对于链 r = {m_1, m_2, ..., m_L}，端到端时延由三部分组成：

#### (a) 计算时延（排队 + 服务）

对链中第 i 个服务 m_i：

```
E[W_comp(m_i)] = sum_v P_i[v] * W(lambda_s[m_i] * P_i[v], mu_{m_i}, X[v][m_i])
```

其中 P_i[v] = X[v][m_i] / sum(X[:, m_i]) 为路由概率。

链的总计算时延：

```
W_comp = sum_{i=1}^{L} E[W_comp(m_i)]
```

#### (b) 通信时延

相邻服务 m_i 和 m_{i+1} 之间，若处理它们的节点不同，产生跨节点通信时延：

```
E[W_comm(m_i, m_{i+1})] = sum_{v1 != v2} P_i[v1] * P_{i+1}[v2] * D[v1][v2]
```

若 m_i 和 m_{i+1} 在同一节点 (v1 = v2)，通信时延为零。

链的总通信时延：

```
W_comm = sum_{i=1}^{L-1} E[W_comm(m_i, m_{i+1})]
```

#### (c) 不稳定惩罚

对于任何 rho >= 1 的服务-节点对，施加惩罚：

```
penalty = sum 5.0 * (rho - 1 + 0.1)    对所有不稳定的 (service, node) 对
```

#### (d) 链总时延

```
T_chain = W_comp + W_comm + penalty
```

代码实现：`queuing.py` → `compute_chain_delay()` 函数

### 3.4 系统平均时延

所有链的加权平均时延，权重为各链到达率占总到达率的比例：

```
E[T_avg] = sum_r (lambda_r / Lambda_total) * T_r
```

其中 Lambda_total = sum_r lambda_r。

代码实现：`evaluation.py` → `evaluate()` 函数

---

## 4. 部署模型

### 4.1 部署矩阵

部署决策用矩阵 X 表示：

```
X[v][s] = 节点 v 上服务 s 的实例数量, shape: (n_nodes, n_service_types)
```

约束条件：

```
sum_s X[v][s] <= C_v              (CPU 约束: 每节点实例总数不超过核心数)
sum_s X[v][s] * g_s <= G_v        (GPU 约束: 每节点 GPU 占用不超过显存总量)
```

代码实现：`deployment.py` → `Deployment` 类

```python
class Deployment:
    X                    # 部署矩阵, shape (n_nodes, n_services), dtype=int
    routing_probabilities(service)  # 返回 P[v] = X[v][s] / sum(X[:,s])
```

### 4.2 随机部署基线 (RLS)

`random_deployment()` 函数实现两阶段部署策略：

**第一轮：保证覆盖**
- 随机打乱服务顺序
- 对每个服务，在满足资源约束的节点中随机选择一个，放置 1 个实例
- 确保所有服务至少有 1 个实例在运行

**第二轮：按负载分配**
- 根据各服务的聚合到达率 lambda_s，计算每个服务应获得的资源份额
- share_s = lambda_s / sum(lambda_s)
- 按到达率降序处理各服务，将剩余 CPU 资源按份额比例分配为额外实例

```
输入: network, services, lambda_s (可选)
输出: deployment, cpu_remaining, gpu_remaining
```

### 4.3 FFD 部署 (First Fit Decreasing)

`ffd_deployment()` 函数实现经典的 FFD 装箱策略，适配微服务场景：

**步骤 1：计算实例需求**
- 对每个服务 m，计算最少实例数 N_m = ceil(lambda_m / mu_m)，保证 rho < 1
- 无负载信息时默认 N_m = 1

**步骤 2：降序排列**
- 按所需实例数 N_m 降序排列所有服务（Decreasing 部分）

**步骤 3：首次适应部署**
- 按节点编号 0, 1, 2, ... 顺序遍历
- 在当前节点尽可能多地放置实例：min(剩余CPU, 剩余GPU // gpu需求, 尚需数)
- 当前节点放不下时移至下一节点

代码实现：`deployment.py` → `ffd_deployment()` 函数

### 4.4 DRS 部署 (Deterministic Routing Scheme)

`drs_deployment()` 函数实现贪心资源优先部署策略：

**步骤 1：计算实例需求**
- 与 FFD 相同：N_m = ceil(lambda_m / mu_m)

**步骤 2：按到达率降序排列**
- 有负载信息时按到达率降序；无负载信息时按服务率升序（慢速优先）

**步骤 3：贪心放置**
- 第一轮：保证每个服务至少 1 个实例，选择剩余 CPU 最多的节点
- 第二轮：按负载比例分配额外实例，同样贪心选择剩余 CPU 最多的节点

代码实现：`deployment.py` → `drs_deployment()` 函数

### 4.5 LEGO 部署 (Load-balanced Efficient Grid Optimization)

`lego_deployment()` 函数实现三阶段负载均衡部署：

**阶段 1：实例创建**
- 计算每个服务所需实例数 N_m = ceil(lambda_m / mu_m)

**阶段 2：按需排序**
- 按 N_m 降序排列服务

**阶段 3：均衡放置**
- 逐个放置实例，每次选择当前剩余 CPU 最多的节点
- 与 FFD（首次适应）相反，LEGO 优先选择资源最充足的节点，实现跨节点负载均衡

代码实现：`deployment.py` → `lego_deployment()` 函数

---

## 5. 路由模型

### 5.1 路由概率

路由决策确定请求到达各服务实例的流量分配。在 Jackson 网络中，路由概率直接从部署矩阵推导：

```
P[v, s] = X[v][s] / sum_v' X[v'][s]
```

即每个服务的流量按实例数比例分配到各节点。

### 5.2 比例路由 (当前默认)

`proportional_routing()` 函数：从部署矩阵直接推导，P[v] = X[v][s] / sum(X[:,s])。对所有源节点返回相同的概率分布（与源无关）。

### 5.3 随机路由 (备选基线)

`random_routing()` 函数：仅在有实例的节点间随机分配概率权重，归一化后得到路由分布。不同运行产生不同分布。

### 5.4 DRS 路由 (平方根加权概率路由)

`drs_routing()` 函数：基于平方根加权的概率路由策略。

```
P[v] = sqrt(X[v][s]) / sum(sqrt(X[:,s]))
```

平方根加权比纯比例路由更加均衡，避免实例数多的节点承担过多流量，同时仍保留按实例数加权的基本特性。所有源节点共享相同的路由概率（源节点无关）。

代码实现：`routing.py` → `drs_routing()` 函数

### 5.5 LEGO 路由 (确定性均匀路由)

`lego_routing()` 函数：确定性均匀路由策略。

```
P[v] = 1 / count(active_nodes)    对所有活跃节点
```

对每个服务，在所有部署了该服务实例的节点间均匀分配流量。不考虑各节点上的实例数量差异，强制均分，配合负载均衡的部署策略使用。

代码实现：`routing.py` → `lego_routing()` 函数

代码实现：`routing.py` → `RoutingTable` 类

---

## 6. 评估模块

`evaluation.py` → `evaluate()` 函数对给定 (部署, 服务, 链, 网络) 计算系统性能指标。

### 6.1 评估流程

```
1. 计算各服务聚合到达率 lambda_s
2. 对每条链:
   a. 调用 compute_chain_delay() 计算端到端时延
   b. 记录计算时延、通信时延、惩罚、稳定性
3. 加权汇总 (权重 = 链到达率 / 总到达率)
4. 计算资源利用率
```

### 6.2 输出指标

| 指标 | 字段名 | 计算方式 |
|------|--------|----------|
| 加权平均端到端时延 | avg_delay | sum(w_r * T_r) |
| 加权平均计算时延 | avg_comp_delay | sum(w_r * W_comp_r) |
| 加权平均通信时延 | avg_comm_delay | sum(w_r * W_comm_r) |
| 加权平均惩罚 | total_penalty | sum(w_r * penalty_r) |
| 稳定链数 | stable_chains | count(T_r.stable == True) |
| 总链数 | total_chains | len(chains) |
| CPU 利用率 | cpu_utilization | sum(X) / sum(C_v) |
| GPU 利用率 | gpu_utilization | sum(X * g_s) / sum(G_v) |

---

## 7. 实验设计

### 7.1 静态实验 (experiment.py)

四组参数扫描实验，每次只变化一个参数，其余保持默认值：

#### 实验 1：到达率扫描

| 参数 | 值 |
|------|-----|
| 扫描变量 | total_rate = 100, 200, 300, 400, 500, 600, 700, 800 |
| 固定参数 | scale=medium, chain_length=6, n_service_types=30 |
| 预期趋势 | 到达率增加 -> 各节点负载增加 -> 排队时延增加 |

#### 实验 2：链长度扫描

| 参数 | 值 |
|------|-----|
| 扫描变量 | chain_length = 3, 4, 5, 6, 7, 8, 9, 10 |
| 固定参数 | scale=medium, total_rate=400, n_service_types=30 |
| 预期趋势 | 链越长 -> 累积计算时延和跨节点通信时延越大 |

#### 实验 3：服务类型数扫描

| 参数 | 值 |
|------|-----|
| 扫描变量 | n_service_types = 10, 20, 30, 40, 50, 60, 70, 80 |
| 固定参数 | scale=medium, total_rate=400, chain_length=6 |
| 预期趋势 | 更多服务类型 -> 每种服务被分配到的实例数更少 -> 可能增加时延 |

#### 实验 4：网络规模对比

| 参数 | 值 |
|------|-----|
| 扫描变量 | scale = small / medium / large |
| 固定参数 | total_rate=400, chain_length=6, n_service_types=30 |
| 预期趋势 | 更大规模网络 -> 更多资源 -> 时延降低 |

### 7.2 动态实验 (experiment_dynamic.py)

模拟时变环境下的系统性能，每个时间步独立运行：

```
初始化 (固定):
  network = EdgeNetwork(scale)
  services = generate_services(n_types)
  base_chains = generate_service_chains(...)

For step = 0, 1, ..., 49:
  1. 生成扰动因子:
     load_factor ~ Uniform(0.5, 1.5)
     cpu_factor  ~ Uniform(0.7, 1.3)
     gpu_factor  ~ Uniform(0.7, 1.3)

  2. 扰动环境:
     network.cpu_capacity = base * cpu_factor
     network.gpu_capacity = base * gpu_factor
     chain.arrival_rate = base_rate * load_factor

  3. 重新部署 (基于新环境)

  4. 计算并记录时延指标
```

三种规模的动态实验配置：

| 规模 | 链长 | 服务类型数 | 基础到达率 | 时间步 |
|------|------|-----------|-----------|--------|
| small | 3 | 5 | 100 | 50 |
| medium | 5 | 10 | 200 | 50 |
| large | 7 | 20 | 300 | 50 |

### 7.3 实验执行入口

`main.py` 按顺序执行全部五组实验，结果保存为 CSV 文件到 `results/` 目录：

```
results/
  arrival_rate_sweep.csv     # 实验 1
  chain_length_sweep.csv     # 实验 2
  service_type_sweep.csv     # 实验 3
  scale_comparison.csv       # 实验 4
  dynamic_small.csv          # 实验 5 - small
  dynamic_medium.csv         # 实验 5 - medium
  dynamic_large.csv          # 实验 5 - large
```

---

## 8. 代码结构

```
exp-48/
|
|-- config.py                 全局配置参数
|   |-- NETWORK_SCALES        网络规模定义 (small/medium/large)
|   |-- SERVICE_RATE_RANGE    服务率范围
|   |-- GPU_PER_INSTANCE_RANGE GPU 需求范围
|   |-- ARRIVAL_RATE_SWEEP    到达率扫描值
|   |-- CHAIN_LENGTH_SWEEP    链长扫描值
|   |-- SERVICE_TYPE_SWEEP    服务类型数扫描值
|   |-- DYNAMIC_*             动态实验参数
|   |-- INSTABILITY_*         稳定性惩罚参数
|
|-- network.py                网络拓扑
|   |-- EdgeNetwork           边缘网络类
|       |-- __init__()        初始化节点资源 + 通信时延矩阵
|       |-- reset_resources() 重置资源到原始值
|       |-- perturb_capacity() 按比例扰动容量 (动态实验用)
|
|-- service.py                微服务与服务链
|   |-- Service               微服务 (id, service_rate, gpu_per_instance)
|   |-- ServiceChain          服务链 (id, services[], arrival_rate)
|   |-- generate_services()   随机生成 n 个微服务
|   |-- generate_service_chains()  生成服务链 + Dirichlet 分配到达率
|   |-- compute_aggregate_arrival_rates()  计算聚合到达率 lambda_s
|
|-- queuing.py                排队论时延计算
|   |-- mm_c_response_time()  M/M/c 单节点响应时间
|   |-- compute_chain_delay() 单条链端到端时延 (Jackson 分解)
|
|-- deployment.py             部署算法
|   |-- Deployment            部署矩阵类 + 路由概率计算
|   |-- random_deployment()   RLS 随机部署基线 (两阶段)
|   |-- ffd_deployment()      FFD 降序首次适应部署
|   |-- drs_deployment()      DRS 贪心资源优先部署
|   |-- lego_deployment()     LEGO 负载均衡部署
|
|-- routing.py                路由算法
|   |-- RoutingTable          路由表类
|   |-- proportional_routing() 按实例数比例路由
|   |-- random_routing()      随机路由基线
|   |-- drs_routing()         DRS 平方根加权概率路由
|   |-- lego_routing()        LEGO 确定性均匀路由
|
|-- evaluation.py             性能评估
|   |-- evaluate()            计算系统全部性能指标
|
|-- experiment.py             静态实验
|   |-- run_single()          运行单次实验
|   |-- run_arrival_rate_sweep()    实验 1
|   |-- run_chain_length_sweep()    实验 2
|   |-- run_service_type_sweep()    实验 3
|   |-- run_scale_comparison()      实验 4
|
|-- experiment_dynamic.py     动态实验
|   |-- run_dynamic_experiment()    单规模动态实验
|   |-- run_all_scales()            三种规模全部运行
|
|-- main.py                   主入口
|
|-- test_framework.py         单元测试 (158 项)
|
|-- results/                  CSV 输出目录
```

---

## 9. 核心数据流

```
                    config.py
                        |
            +-----------+-----------+
            |                       |
        network.py             service.py
     (EdgeNetwork)        (Service, ServiceChain)
            |                       |
            |              compute_aggregate_arrival_rates()
            |                       |
            +-----------+-----------+
                        |
                  deployment.py          routing.py
               (Deployment.X)      (RoutingTable)
                        |                   |
                        +-------+-----------+
                                |
                          queuing.py
                     (M/M/c 延迟计算)
                                |
                         evaluation.py
                       (加权汇总指标)
                                |
                    +-----------+-----------+
                    |                       |
            experiment.py          experiment_dynamic.py
            (静态参数扫描)           (动态时变实验)
                    |                       |
                    +-----------+-----------+
                                |
                            main.py
                                |
                           results/*.csv
```

### 9.1 单次实验的执行流程

`run_single(scale, total_rate, chain_length, n_service_types, seed)` 的完整执行路径：

```
Step 1: EdgeNetwork(scale, seed)
        生成 n_nodes 个节点，每个具有 cpu_capacity 和 gpu_capacity
        生成 comm_delay 通信时延矩阵 (对称, 对角零)

Step 2: generate_services(n_service_types, seed)
        对每种服务: mu ~ Uniform(5, 20), gpu ~ Uniform(128, 1024)

Step 3: generate_service_chains(n_chains, n_service_types, chain_length, total_rate, seed)
        总到达率按 Dirichlet(1,...,1) 分配给各链
        每条链的每个位置随机抽取一种服务

Step 4: compute_aggregate_arrival_rates(chains, n_services)
        lambda_s[s] = sum of chain.rate for chains containing service s

Step 5: random_deployment(network, services, lambda_s, seed)
        第一轮: 每个服务至少 1 实例
        第二轮: 按负载比例分配剩余资源

Step 6: proportional_routing(deployment)
        P[v,s] = X[v,s] / sum(X[:,s])

Step 7: evaluate(deployment, services, chains, network)
        对每条链:
          对链中每个服务:
            对每个有实例的节点:
              mm_c_response_time(lambda_s * P[v], mu, X[v][s])
            期望计算时延 += P[v] * W[v]
          相邻服务通信时延 += P[v1] * P[v2] * comm_delay[v1][v2] (v1!=v2)
        加权平均 (权重 = chain.rate / total_rate)
```

---

## 10. 公式推导与正确性验证

### 10.1 M/M/c 公式推导

M/M/c 排队系统：c 个并行服务台，Poisson 到达 (率 lambda)，指数服务时间 (率 mu)。

**状态转移**：系统中有 n 个顾客的概率 p_n 满足平衡方程：

```
n < c:  lambda * p_n = (n+1) * mu * p_{n+1}     => p_n = (a^n / n!) * p_0
n >= c: lambda * p_n = c * mu * p_{n+1}           => p_n = (a^n / (c! * c^{n-c})) * p_0
```

其中 a = lambda / mu 为到达负载。

**归一化**求 p_0：

```
p_0 = 1 / [sum_{n=0}^{c-1} a^n/n! + a^c / (c! * (1-rho))]
```

其中 rho = lambda / (c * mu) < 1。

**平均队列长度** L_q (等待中的顾客数)：

```
L_q = sum_{n=c+1}^{inf} (n-c) * p_n = (a^c * rho) / (c! * (1-rho)^2) * p_0
```

**平均等待时间**：

```
W_q = L_q / lambda          (Little's Law)
W = W_q + 1/mu              (排队等待 + 服务时间)
```

### 10.2 M/M/1 特例验证

当 c = 1 时：

```
rho = lambda / mu
p_0 = 1 - rho
L_q = rho^2 / (1 - rho)
W_q = rho^2 / ((1-rho) * lambda) = rho / (mu - lambda)
W = rho / (mu - lambda) + 1/mu = lambda / (mu * (mu - lambda)) + 1/mu
  = (lambda + mu - lambda) / (mu * (mu - lambda))
  = 1 / (mu - lambda)
```

这与经典 M/M/1 结果 W = 1/(mu - lambda) 一致。

测试验证 (test_framework.py)：

```
mm_c_response_time(1.0, 2.0, 1) = 1.0    (理论: 1/(2-1) = 1.0)   PASS
mm_c_response_time(0.5, 1.0, 1) = 2.0    (理论: 1/(1-0.5) = 2.0)  PASS
mm_c_response_time(5.0, 10.0, 1) = 0.2   (理论: 1/(10-5) = 0.2)   PASS
```

### 10.3 Jackson 网络分解的正确性

Jackson 定理保证了在以下条件下，排队网络可以分解为独立的 M/M/c 子系统：
1. 外部到达为 Poisson 过程
2. 服务时间服从指数分布
3. 路由概率固定（与系统状态无关）

本框架中：
- 链到达率为 Poisson (满足条件 1)
- 每核心服务时间为指数分布 (满足条件 2)
- 路由概率 P[v,s] = X[v,s] / sum(X[:,s]) 为固定值 (满足条件 3)

因此，各节点上的 M/M/c 队列相互独立，可以分别计算时延后求和。

---

## 11. 扩展指南

### 11.1 替换部署算法

在 `deployment.py` 中新增函数，遵循接口约定：

```python
def my_deployment(network, services, lambda_s, seed=None):
    """
    返回: (deployment, cpu_remaining, gpu_remaining)
    - deployment: Deployment 实例
    - cpu_remaining: np.array, shape (n_nodes,)
    - gpu_remaining: np.array, shape (n_nodes,)
    """
    ...
```

然后在 `experiment.py` 的 `run_single()` 中添加分支：

```python
if deploy_algo == 'my_algo':
    deployment, _, _ = my_deployment(network, services, lambda_s, seed=seed)
```

### 11.2 替换路由算法

在 `routing.py` 中新增函数，遵循接口约定：

```python
def my_routing(deployment, **kwargs):
    """
    返回: RoutingTable 实例
    - table[(source_node, service)] = np.array of probabilities, shape (n_nodes,)
    """
    ...
```

### 11.3 添加新的实验维度

在 `experiment.py` 中添加新的 sweep 函数：

```python
def run_my_sweep(...):
    results = []
    for param in param_list:
        r = run_single(scale, rate, length, types, ...)
        results.append(r)
    return results
```

### 11.4 添加新的性能指标

在 `evaluation.py` 的 `evaluate()` 函数中添加新的计算逻辑，返回值中追加新字段即可。

---

## 12. 依赖与环境

```
Python >= 3.8
numpy
scipy (用于 factorial, 已改用 math.factorial, 可选)
```

无其他外部依赖。

---

## 13. 参数速查表

| 参数 | 默认值 | 位置 |
|------|--------|------|
| 网络规模 | medium (5 nodes, 32 cores, 16384 GPU) | config.py |
| 服务率范围 | (5.0, 20.0) | config.py |
| GPU 需求范围 | (128, 1024) MB | config.py |
| 通信时延范围 | (0.005, 0.050) s | config.py |
| 默认总到达率 | 400 | config.py |
| 默认链长 | 6 | config.py |
| 默认服务类型数 | 30 | config.py |
| 默认链数 | 4 | config.py |
| 不稳定惩罚系数 | 5.0 | config.py |
| 不稳定固定时延 | 1.0 s | config.py |
| 动态实验步数 | 50 | config.py |
| 动态负载因子范围 | Uniform(0.5, 1.5) | experiment_dynamic.py |
| 动态容量因子范围 | Uniform(0.7, 1.3) | experiment_dynamic.py |
