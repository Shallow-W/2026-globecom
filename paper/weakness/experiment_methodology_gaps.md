# 实验与方法论对照问题清单

> 对照 Sections II-IV（方法论）与 Section VI（实验），整理出以下 7 个对不上的问题。

---

## 1. 优化目标是最大化精度，但没有报告精度结果 [高]

**问题定位：**
- 方法论 Eq.(4): `max sum(x_{i,k}(t) * S_k^{proxy})`，目标函数明确以代理精度分数为优化目标
- 实验部分仅报告了 end-to-end latency (Fig.2) 和 memory utilization (Fig.3)
- 整个 Section VI 没有出现任何 accuracy/precision/mAP 相关的指标

**影响：** 优化目标和实验评估脱节，审稿人会质疑：你的系统选出了"最优"子网，但这个"最优"在精度上到底好多少？

**建议补充：**
1. 在静态性能中增加一个 accuracy 的子图或表格，展示不同到达率下各方法的推理精度
2. 如果用 ImageNet/CIFAR 等标准数据集，报告 top-1 accuracy
3. 可以展示 accuracy vs latency 的 trade-off 曲线（Pareto 前沿）

---

## 2. "毫秒级评估"核心声明缺少数据支撑 [高]

**问题定位：**
- 摘要: "millisecond-level, on-demand model switching capabilities"
- 引言 Contribution 1: "enables millisecond-level performance estimations on edge nodes"
- Section III: 整个章节围绕快速评估展开

但实验部分：
- 没有报告 ZCP 评估的实际耗时（多少 ms）
- 没有与 NAS 梯度搜索耗时做对比
- 没有展示"搜索-决策-拉取"各阶段的耗时分解

**影响：** 这是论文的核心 claim 之一，没有定量数据支撑会让审稿人认为只是理论声明。

**建议补充：**
1. 一个表格对比：ZCP 评估时间 vs 传统 NAS 评估时间（梯度训练）
2. 端到端部署流水线的各阶段耗时分解图：ZCP评估 -> 调度决策 -> 云端权重切片 -> 传输 -> 热重载
3. 说明候选子网数量 K 对评估耗时的影响

---

## 3. Scalability 实验中的 fitness 指标未在方法论中定义 [中]

**问题定位：**
- Section VI-C (Horizontal Scaling): "our method achieves 2.41, 3.33, and 3.94" (median fitness)
- 方法论中定义了效用函数 U(a_k) (Eq.(9))，但没有出现 "fitness" 这个术语
- 审稿人无法判断 fitness 的含义、计算方式和取值范围

**影响：** 指标定义不清，结果不可复现。

**建议修复：**
1. 在实验设置或方法论中明确定义 fitness = U(a_k)，或者给出独立的定义
2. 说明 fitness 的取值范围和物理含义
3. 如果 fitness 就是 U(a_k)，统一使用 utility score 术语

---

## 4. 动态适应实验（Section VI-D）过于单薄 [中]

**问题定位：**
- 这是验证核心动态调度能力的关键实验
- 目前只有一段文字 + 一个图 (Fig.6)
- 方法论中详细描述了动态惩罚权重 (Eq.(10))、GSTC 方案切换 (Section III)、硬约束过滤 (Algorithm 1)，但实验中没有对应展示

**缺失内容：**
1. 具体的流量变化模式（何时升高、何时恢复）未描述
2. 没有展示 w_2(t), w_3(t) 随时间的变化曲线
3. 没有展示系统在不同阶段选择了哪种类型的子网（轻量 vs 重型）
4. 没有与方法论三阶段流程的对应分析
5. 没有展示动态切换的次数和切换带来的瞬时影响

**建议补充：**
1. 增加一个子图展示 traffic lambda(t) 随时间的变化曲线，与延迟曲线并列
2. 增加动态权重变化的可视化
3. 展示子网切换事件的时间线（何时从重型切换到轻量）
4. 适当扩充文字分析，与方法论机制一一对应

---

## 5. SLA 违约率未报告 [中]

**问题定位：**
- 系统模型核心约束: T_req <= T_{SLA} (Eq.(2))
- 方法论围绕 SLA 预算展开: T_{SLA} = T_{e2e}/L，并由此推导 F_i^{max}(t)
- 但实验中只报告了平均延迟，没有报告 SLA 违约率 (violation rate) 或合规率

**影响：**
- 审稿人无法判断硬约束是否真正被满足
- 平均延迟低不代表没有尾部违约（可能有少量请求严重超时）
- 硬约束过滤声称能"严格保证 SLA 延迟边界"，需要实验证据

**建议补充：**
1. 报告各方法在不同到达率下的 SLA 违约率百分比
2. 或者报告延迟的 CDF 曲线（累积分布），标注 SLA 阈值线
3. 至少在文字中说明 T_{e2e} 和 T_{SLA} 的具体设定值

---

## 6. 关键超参数和实验细节缺失 [中]

**问题定位：**
实验设置 (Section VI-A) 缺少多个方法论中涉及的关键参数：

| 参数 | 出处 | 现状 |
|------|------|------|
| lambda_th | Eq.(10) 安全流量阈值 | 未给出具体值 |
| T_{e2e} | 端到端 SLA 目标 | 未明确设定 |
| L | 链路长度 | 说"mostly 2-9"但未说明分布 |
| Supernet 架构 | Section II | 未说明（OFA / MobileNet / 自定义？），K = ? |
| GSTC 权重 (alpha, beta, gamma, delta) | Eq.(8) | 三种方案的具体数值未给出 |
| c_1, c_2 | Eq.(10) 动态惩罚系数 | 未给出 |
| 硬件环境 | Section II-A | M_total, C_max 的具体值未给出 |

**影响：** 结果不可复现，审稿人无法判断参数选择是否合理。

**建议修复：** 在实验设置中增加一个参数表，列出所有关键参数及其取值和含义。

---

## 7. 云边解耦流水线缺少分解验证 [低]

**问题定位：**
- Contribution 3: "Edge nodes run only a minimal ZCP and parser to fetch subnets on demand. This eliminates local supernet storage."
- Section IV 最后一段详细描述了 on-demand fetching 三阶段：搜索 -> 决策 -> 拉取
- 但实验中没有验证这个流程的实际表现

**缺失验证：**
1. 各阶段耗时分解（ZCP评估 / 调度决策 / 云端切片 / 传输 / 热重载）
2. 本地存储开销对比（存储完整 supernet vs 仅存 ZCP+parser 的内存差值）
3. 云边通信延迟对整体部署时间的影响

**建议补充：**
1. 一个条形图或表格，展示一次完整的搜索-拉取-重载流程中各阶段的耗时占比
2. 对比存储需求：完整 supernet 的大小 vs 边缘侧最小部署包的大小
3. 如果篇幅有限，至少在文字中给出数量级说明
