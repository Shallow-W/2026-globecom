# 实验设置与方法论的差距分析

> 基于 globecom.tex 最新版本（含 per-hop SLA 修改）的逐项审查

---

## 一、关键参数缺失（高优先级）

方法论中定义了多个核心参数，实验部分完全没有给出具体数值，导致不可复现。

| 参数 | 出现位置 | 物理含义 | 实验是否给出 |
|------|----------|----------|--------------|
| $T_{e2e}$ | Section II | 端到端 SLA 目标 | 未给出 |
| $\lambda_{th}$ | Eq.(10) | 安全流量阈值 | 未给出 |
| $M_{total}$ | Section II | 节点总内存 | 未给出 |
| $C_i^{max}$ | Section II | 节点峰值算力 (FLOPS) | 未给出 |
| $c_1, c_2$ | Eq.(10) | 惩罚权重系数 | 未给出 |
| $K$ | Section II-B | 候选子网池大小 | 未给出 |
| $N$ | Section II-A | 默认边缘节点数 | 仅在 Scalability 实验中给出 3/5/7，Static 实验未说明 |
| $L$ | Section II-A | 请求链长度 | 实验给出了 2-9 范围，但默认值未说明 |

**建议：** 在 Experimental Settings 末尾加一个参数表，列出所有取值。

---

## 二、Supernet 架构与数据集未交代（高优先级）

整篇论文以 supernet $\mathcal{W}$ 为核心载体，但实验部分完全没有说明：

1. **用了哪个 supernet**（OFA / ProxylessNAS / Once-for-All 变体 / 自定义？）
2. **搜索空间定义**（kernel size 范围、depth 范围、width multiplier 范围、候选子网总数 $K$）
3. **在什么数据集上预训练**（ImageNet / CIFAR-10 / CIFAR-100 / 自定义任务？）
4. **预训练 epoch 数、supernet top-1 精度**

这是 NAS / supernet 类论文审稿人必问的信息，缺失会严重影响可信度。

**建议：** 补充一段类似 "We adopt OFA-style supernet pre-trained on ImageNet, with search space spanning kernel sizes {3,5,7}, depths {2,3,4} per block, and width multipliers {0.5, 0.75, 1.0}, yielding K=XXX candidate subnets."

---

## 三、术语不一致："multi-task model library" vs. Supernet

实验部分第 192 行：

> "We pre-train a supernet offline and build a **multi-task model library** for online deployment."

但全文方法论都在讲**单个 supernet 的子网采样与调度**，"multi-task" 从未在方法论中出现。这会让审稿人困惑：

- 如果是多任务，那不同任务之间的子网如何区分？多任务之间的关系是什么？
- 如果不是多任务，那这个表述就是笔误。

**建议：** 统一为 "a pool of candidate subnets sampled from the pre-trained supernet" 或类似表述，与全文保持一致。

---

## 四、ZCP 评估缺乏验证（高优先级）

Section III 花了整整一节描述 GSTC proxy 的动态权重机制（三种典型场景），但实验部分：

1. **没有验证 ZCP 预测分数与实际推理精度的相关性**
   - NAS 论文通常报告 Kendall $\tau$ 或 Spearman $\rho$ 来证明 proxy 的有效性
   - 完全没有，审稿人会质疑 GSTC proxy 是否真的能准确预测精度
2. **没有说明 $(\alpha, \beta, \gamma, \delta)$ 在实验中的具体配置**
   - 三种场景各用什么权重？
   - 是手动设定还是自动切换？切换条件是什么？
3. **缺少针对 ZCP 组件的消融实验**（见第八条）

**建议：** 至少补充一张 ZCP score vs. 实际 accuracy 的散点图 + 相关系数。

---

## 五、缺少推理精度评估（高优先级）

方法论的优化目标 Eq.(4) 是**最大化精度潜力** $\widehat{S}_k^{proxy}$，但实验的三个 subsection 只报告了：

- End-to-end latency
- Memory utilization
- Fitness（Scalability 实验中）

**完全没有报告实际推理精度**（accuracy / top-1 / mAP / F1 等）。

审稿人的合理质疑：你的方法虽然延迟低、内存好，但精度有没有下降？动态选子网后精度是提升还是损失？

**建议：**
- 补充一张 latency-accuracy trade-off 图（Pareto front）
- 或者在每个实验中同时报告精度指标

---

## 六、基线方法描述不足

实验列了 6 个基线（FFD, CDS, RLS, Greedy, DRS, LEGO），但：

1. **没有一句话描述各基线的做法**（审稿人不一定熟悉所有方法）
2. **只有 LEGO 有引用**，其余 5 个基线都没有 `\cite{}`，读者无法溯源
3. **"They use fixed models" 太笼统** —— 固定的是什么模型？多大？精度多少？

**建议：** 加一个简短的基线描述段落或表格，包含：方法名、简要描述、引用、固定模型配置。

---

## 七、实验环境未说明

缺少以下基本信息：

- **仿真还是真实部署？** 论文没有明确说是模拟实验还是真实系统实验
- **硬件平台？** 如果是真实的，用什么 GPU/CPU？如果是仿真的，用什么仿真器？
- **深度学习框架？** PyTorch / TensorFlow / 其他？
- **重复次数与统计显著性？** Scalability 实验提到 "100 independent trials"，但其他实验没有说明

**建议：** 补充一小段实验环境描述。

---

## 八、缺少消融实验

论文提出三个核心贡献，但实验没有消融分析来拆分各自的贡献：

| 消融变体 | 去掉的组件 | 验证的贡献 |
|----------|-----------|-----------|
| w/o dynamic GSTC | 固定 $(\alpha, \beta, \gamma, \delta)$，不随场景切换 | ZCP 动态适配的效果 |
| w/o hard constraints | 去掉 Eq.(6)(7) 的硬约束过滤 | 硬约束过滤的必要性 |
| w/o dynamic penalty | 固定 $w_2, w_3$，不随 traffic 调节 | 动态惩罚权重的效果 |
| Full method | 所有组件启用 | 完整方法的优越性 |

没有消融实验，审稿人无法判断每个组件是否真的有用，还是整体结果主要来自其中某一个。

**建议：** 至少做 2-3 个消融变体，在主实验的一个 perturbation axis 上对比。

---

## 九、图片引用顺序问题

Fig.1（`fig/delay`）和 Fig.2（`fig/mem_util`）的 `\begin{figure}` 放在 Experimental Settings 的文字**之前**（第 174-190 行），但它们是在 Static Performance 部分（第 198-200 行）才被引用。IEEE 模板中 figure 的浮动位置由 LaTeX 决定，但在源码中把 figure 环境放在引用文字之前可能导致排版错位。

**建议：** 将 figure 环境移到首次引用 `\ref{}` 附近。

---

## 十、Keywords 仍然是模板默认值

第 35-36 行：

```latex
\begin{IEEEkeywords}
component, formatting, style, styling, insert.
\end{IEEEkeywords}
```

**建议：** 替换为与论文内容相关的关键词，如 "Edge AI, Microservice Deployment, Neural Architecture Search, Zero-Cost Proxy, Cloud-Edge Collaboration"。

---

## 十一、标题和作者信息未填写

第 17-26 行仍然是 IEEE 模板的占位符。

---

## 优先级排序

| 优先级 | 问题 | 预计审稿人关注程度 |
|--------|------|-------------------|
| P0 | Supernet 架构与数据集缺失 | 必问 |
| P0 | 推理精度未评估 | 必问 |
| P0 | 关键参数表缺失 | 必问 |
| P1 | ZCP 验证（proxy vs. 实际精度） | 很可能问 |
| P1 | 消融实验缺失 | 很可能问 |
| P1 | 基线描述与引用不足 | 可能问 |
| P2 | 术语不一致（multi-task） | 容易修复 |
| P2 | 实验环境未说明 | 可能问 |
| P2 | Keywords / 标题占位符 | 投稿前必须改 |
| P3 | 图片浮动位置 | 排版细节 |
