# 实验总结文档（给 ClaudeCode）

> **说明**: 本文档反映 exp/scripts 中实现的实验逻辑，已与 globecom.tex 论文内容对齐。

## 0. 当前实现的算法

### OURS 算法
- **部署**: `OursCEDR` - 动态算力红线 + 效用函数 + 动态权重自适应
- **路由**: `RoutingOURS` - 联合路由效用最大化
- **动态替换机制**: 基于 30% 负载变化阈值触发模型重新选择

### 论文 A 基线 (TPDS 2023)
- **RLS/FFD/DRS/LEGO**: 均固定中等性能模型（proxy_score 50% 分位）
- **路由**: `RoutingHeuristicA` - Dijkstra 最短路径

### 论文 B 基线 (TSC 2024)
- **GREEDY/PSO**: 均固定中等性能模型
- **路由**: `RoutingGreedyB` - 最近节点转发

### 消融实验基线
- **STATIC**: 固定 proxy_score 最高架构
- **RESOURCE_FIRST**: flops+params 最小优先
- **ACCURACY_FIRST**: proxy_score 最大优先

## 1. 实验类型与运行方式

```bash
cd exp/scripts
python run_experiments.py quick          # 快速验证（10 slots）
python run_experiments.py dynamic        # 动态负载实验
python run_experiments.py perturb ar     # 到达率扰动
python run_experiments.py perturb len    # 请求长度扰动
python run_experiments.py perturb task   # 请求类型数扰动
python run_experiments.py equal          # 等时延资源效率
python run_experiments.py scale          # 规模扩展
python run_experiments.py all            # 全部实验
```

## 2. 仿真指标体系

| 指标 | 说明 |
|------|------|
| avg_latency_ms | 平均端到端时延 |
| success_rate | 请求成功率 |
| sla_violation_rate | SLA 违规率 |
| throughput | 系统吞吐量 (req/s) |
| avg_pull_delay_ms | 模型拉取平均时延 |
| pull_count | 模型拉取总次数 |
| node_utilization_avg | 节点内存平均利用率 |
| task_performance | 分任务真实性能 |

## 3. 拓扑配置

| 规模 | 节点数 | 配置 |
|------|--------|------|
| Small | 15 | 5小型(2GB,10GF) + 7中型(4GB,30GF) + 3大型(8GB,100GF) |
| Medium | 30 | 统一中型节点 (4GB, 30GF) |
| Large | 65 | 对齐论文 B ta2 拓扑 |

## 4. 流量模式

- **steady**: 固定到达率 λ=3.0 req/s/节点
- **tidal**: 周期性潮汐流量
- **burst**: 高强度突发流量

## 5. 扰动实验变量

| 变量 | 扰动值 |
|------|--------|
| 到达率 (arrival_rate) | [1, 2, 3, 5, 8, 12] |
| 请求长度 (request_length) | [2, 5, 10, 20, 50] |
| 请求类型数 (n_task_types) | [1, 2, 3, 5, 7] |

## 6. 关键仿真逻辑 (sim.py)

### 每时隙流程
1. 流量生成 (Poisson 随机到达)
2. 候选架构预计算（每节点每任务过滤一次）
3. 请求路由决策
4. 统计收集

### 时延模型
- **排队时延**: M/M/1 有上界模型，$\rho < 0.9$ 时 $T_{queue} = 1000/(μ-λ)$，否则 $T_{queue} = max(2000, 1000/(μ(1-ρ)))$
- **路由时延**: $R_{total} = L_{topo} + T_{queue} + D_{pull}$（云边拉取）
- **失败处理**: 失败请求计入 5000ms 超时惩罚时延

---

## 以下为参考论文原始实验设计（已对齐实现）

## 1. 论文 A：Joint Deployment and Request Routing for Microservice Call Graphs in Data Centers (TPDS 2023)

### 1.1 研究目标与方法框架

- 目标：联合优化微服务部署与请求路由，降低部署成本与端到端时延，并提升请求成功率。
- 方法：两阶段启发式框架 GMDA-RMPR。
- 阶段 1（部署）：GMDA，基于资源拆分与队列建模，决定微服务实例部署。
- 阶段 2（路由）：RMPR，基于请求匹配与分区映射，采用概率路由分发到实例。

### 1.2 实验设置（可复现信息）

1. 流量与调用链设置
- 请求数范围：10 到 1000。
- 微服务规模：文中提到常见规模 2 到 50。
- 实验中请求链长度：最多 14，且主要分布在 2 到 9。
- 输入依据：trace-driven simulation，基于真实数据中心流量特征。

2. 基础设施与拓扑
- 数据中心节点（处理器）数量：10 到 64。
- 每处理器核心资源：8 到 64（可视作服务能力上限）。
- 网络：全连接服务网格/连接图建模；假设交换与带宽足够承载请求。
- 通信时延约束：最大处理器间通信时延不超过用户可容忍时延的 5%。

3. 对比算法（Baselines）
- DRS
- RLS
- FFD
- LEGO（文中提到三阶段算法）
- Ours：GMDA-RMPR

4. 统计方式
- 多次重复仿真并取平均。
- 平均响应时延实验中，提到算法执行 1000 次取平均结果。

### 1.3 评价指标体系

1. Total deployment cost
- 与“已占用微服务容器数 + 激活处理器数”相关。

2. Total number of used processors
- 反映资源占用与部署紧凑性。

3. Average response latency
- 核心时延指标，带 enhancement ratio 定义（相对基线改进率）。

4. Request success rate
- 在时延/资源约束下请求成功服务比例。

5. Iteration cost
- 寻找可行近优解的迭代步数（也可理解为执行开销代理指标）。

### 1.4 关键实验变量（自变量）

- 用户请求规模（number of requests）
- 微服务类型数量（number of microservices）
- 请求到达率或到达间隔（arrival rate/interval）
- 资源拆分因子 $\omega$
- 可用处理器数量

### 1.5 主要结果（文中可读出的量化）

1. 总结性结论（文中摘要与结论一致）
- 平均部署成本下降约 27.4%。
- 平均端到端响应时延下降约 15.1%。
- 请求成功率在资源充足时平均提升约 5.47%。

2. 细项观察（实验图文）
- 相比 DRS/RLS/LEGO，GMDA-RMPR 在“处理器占用数”上长期更低。
- 在请求规模增大和到达率上升时，GMDA-RMPR 的时延曲线更稳定。
- 请求成功率受资源供给和部署策略双重影响，资源不足时各算法均下降，但 GMDA-RMPR 整体保持较高水平。
- 迭代开销方面：DRS 迭代步明显高；FFD/RLS 低但性能折中；GMDA-RMPR 在性能-开销比上更均衡。

### 1.6 可借鉴到你当前实验的点

- 强烈建议保留“部署成本 + 时延 + 成功率 + 资源占用 + 迭代开销”的五维评估，而不是只看精度/时延。
- 建议像该文一样做多维扫参：请求规模、到达率、服务类型数、资源拆分因子。
- 你的框架是云边协同 NAS 选型，可把“处理器数量”替换成“边缘节点数量/容量档位”。

## 2. 论文 B：Joint Task Offloading, Resource Allocation and Model Placement for AI as a Service in 6G Network (TSC 2024)

### 2.1 研究目标与方法框架

- 目标：在 AIaaS 场景联合优化任务卸载、资源分配、模型放置，综合考虑时延与能耗。
- 方法：两时间尺度优化。
- 短时标：任务卸载 + CPU/GPU 频率与资源分配。
- 长时标：模型放置（Model Placement）动态更新。
- 技术路线：Lyapunov 分解 + DA（deferred acceptance）+ 凸优化 + MAB（文中核心是 DA-MAB）。

### 2.2 实验设置（可复现信息）

1. 网络拓扑
- 小拓扑：Atlanta，15 个计算节点。
- 大拓扑：ta2，65 个计算节点 + 560 用户（用于规模化验证）。

2. 任务负载
- 基础场景：210 个任务，分配给 35 个用户。
- 任务分配周期：每 100 秒。
- 用户任务周期：600 秒。
- 对比短时标性能时，进行多时隙观察（图中多次提及 6 个时间槽）。

3. 对比算法
- Greedy
- PSO
- PPO
- Ours：DA-MAB
- 大拓扑扩展：DA-MAB(ta2)
- 长时标比较：用表现最佳的 PPO 作为“有先验知识”对照，与 MAB 比较。

4. 统计方式
- 单时隙和多时隙都做。
- 通过概率密度分布图展示任务完成时延、频率选择分布等统计特性。

5. 参数表
- 文中有 Table II: Experimental Parameters。
- 但当前 OCR 文本未完整读出表格具体数值（疑似表格嵌图）。
- 可通过原 PDF 的 Table II 手工补录具体参数。

### 2.3 评价指标体系

1. Overall weight（核心目标）
- 等价于联合目标函数值，综合了时延与能耗等成本项。

2. Average task completion time / distribution
- 不仅看均值，也看概率密度分布变化。

3. Time / Energy / Weight 随频率上限变化
- 分析计算频率上限对系统代价与效率的耦合影响。

4. Node capacity violation / penalty
- MAB 随尝试轮次增加，超容量惩罚概率下降。

5. 稳定性指标
- 不同传输速率、通信间隔下的时间序列曲线稳定性。

### 2.4 关键实验变量（自变量）

- 数据规模（data size）
- 传输速率（transmission rate）
- CPU/GPU 能耗系数（energy coefficient）
- 频率上限（frequency upper limit）
- 通信间隔（communication interval）
- 每轮尝试次数（MAB tries per round）

### 2.5 主要结果（定性与可读出结论）

1. DA-MAB 总体优于 Greedy/PSO/PPO
- 在多数场景下，overall weight 更低，且随时间波动更小。

2. 对数据量增长更鲁棒
- 数据规模扩大时，PSO/PPO 更易因容量约束触发惩罚，DA-MAB 增长更平缓。

3. 低传输速率下优势更明显
- Greedy 在低速链路场景退化明显；DA-MAB 仍保持较稳表现。

4. 长时标模型放置有价值
- 开启 MAB 动态放置后，在后续时隙性能明显改善，且更稳定。

5. 频率/能耗耦合规律明确
- 提升频率上限可降低计算时延但提高能耗，存在折中区间；DA-MAB 能较稳定落在较优折中点。

### 2.6 可借鉴到你当前实验的点

- 建议把“静态单时隙最优”与“多时隙动态优化”分开评估，避免只看单点性能。
- 建议引入“分布型结果”而非仅均值，例如时延 PDF/CDF、超约束概率。
- 对你的云边模型拉取场景，等价可引入“重载惩罚概率/缓存失效率”作为容量违规的对应指标。

## 3. 两篇论文实验方法对照（可直接映射到你的课题）

1. 共性
- 都是联合优化，不把部署/放置与路由/卸载割裂。
- 都强调动态环境下的鲁棒性（负载变化、链路变化、资源变化）。
- 都采用多基线横向比较，而非只做自家算法消融。

2. 差异
- 论文 A 偏“微服务调用链 + 概率路由 + 容器部署成本”。
- 论文 B 偏“AIaaS 两时间尺度 + 任务卸载 + 计算频率能耗权衡 + 模型放置”。

3. 对你的直接组合建议
- 评估维度用 A 的“部署成本/成功率/时延/资源占用”框架。
- 动态机制用 B 的“两时间尺度 + 分布统计 + 稳定性曲线”框架。
- 在你的 NAS 子网选择中，补上“拉取与热重载开销”并做敏感性分析。

## 4. 给 ClaudeCode 的落地指令建议（可直接引用）

请按以下优先级实现实验：

1. 场景构建
- 多节点异构边缘网络，带动态到达率与链路波动。

2. 算法组
- Ours（联合部署+路由）
- Accuracy-first
- Resource-first
- Deploy-only
- Static-best

3. 指标组
- Avg latency
- SLA violation rate
- Request success rate
- Deployment/reload cost
- Node utilization
- Iteration/runtime overhead

4. 扫参组
- 请求规模
- 到达率/到达间隔
- 节点数量与容量
- 链路带宽与时延
- 拉取成本系数

5. 输出组
- 均值 + 方差/置信区间
- CDF/PDF 曲线
- 时间序列稳定性图
- 关键指标对比表

## 5. 当前信息缺口与补全建议

1. 论文 B 的 Table II 参数值未完整 OCR 出来
- 建议人工从原 PDF 抄录到配置文件，避免复现偏差。

2. 论文 A 的部分图中精确数值仅在图内
- 若要严格复现，需要从图中读点或查补充材料。

3. 你当前数据集与两篇论文任务定义不同
- 应采用“方法迁移复现”而非“数值逐点复现”。即保留实验设计逻辑、指标与趋势对比，而不强求绝对同值。

## 6. 一句话结论（可放在实验设计总述里）

两篇参考文献共同支持这样的实验路线：在动态负载和异构资源条件下，采用联合优化与多时间尺度决策，并用成本-时延-成功率-稳定性的多指标体系评估，能够比静态或分离式策略得到更稳健、更可部署的系统收益。