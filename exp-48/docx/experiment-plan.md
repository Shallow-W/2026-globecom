# 微服务路由与部署实验计划

## 实验总览

共 6 组实验，6 张图。算法为定制化部署调整策略（Ours），对比基线使用固定模型配置（Baselines）。

| 编号 | 实验名称 | 扫描变量 | 固定参数 | 目的 |
|------|---------|---------|---------|------|
| 1 | 到达率扰动 | total_rate | scale, chain_length, n_service_types | 负载承受能力 |
| 2 | 链长扰动 | chain_length | scale, total_rate, n_service_types | 链复杂度影响 |
| 3 | 服务类型扰动 | n_service_types | scale, total_rate, chain_length | 服务多样性影响 |
| 4 | 网络规模对比 | scale (small/medium/large) | total_rate, chain_length, n_service_types | 可扩展性 |
| 5 | 资源约束扫描 | resource_factor | scale, total_rate, chain_length, n_service_types | 定制化 vs 固定的核心差异 |
| 6 | 动态适应 | time step (时变负载) | scale, chain_length, n_service_types | 定制化的实时适应能力 |

---

## 实验 1：到达率扰动

**扫描变量**：`total_rate = [100, 200, 300, 400, 500, 600]`

**固定参数**：
```
scale = medium (5 节点, 异构)
chain_length = 6
n_service_types = 15
n_chains = 4
```

**预期趋势**：
- 到达率增加 → 各节点负载上升 → 排队时延增加
- 低到达率时所有算法都稳定，差距小
- 高到达率时定制化算法通过调整配置保持稳定，baseline 因固定模型过载

**输出文件**：`results/arrival_rate.csv`

**图表要素**：
- X 轴：total_rate
- Y 轴：avg_delay (s)
- 多条曲线：Ours vs 各 Baseline
- 可加副图：stable_chains 比例

---

## 实验 2：链长扰动

**扫描变量**：`chain_length = [3, 4, 5, 6, 7, 8]`

**固定参数**：
```
scale = medium
total_rate = 400
n_service_types = 15
n_chains = 4
```

**预期趋势**：
- 链越长 → 累积计算时延和跨节点通信时延越大
- 定制化算法在长链中能更好地选择轻量服务、优化通信路径
- 短链时差异不明显，长链时差距拉开

**输出文件**：`results/chain_length.csv`

**图表要素**：
- X 轴：chain_length
- Y 轴：avg_delay (s)
- 多条曲线：Ours vs 各 Baseline

---

## 实验 3：服务类型扰动

**扫描变量**：`n_service_types = [10, 20, 30, 40, 50, 60]`

**固定参数**：
```
scale = medium
total_rate = 400
chain_length = 6
n_chains = 4
```

**预期趋势**：
- 服务类型增多 → 每种服务分到的实例数减少 → 资源争用加剧
- 定制化算法能根据服务特点分配不同规格的模型，资源利用更高效
- Baseline 用统一固定模型，类型多时资源浪费严重

**输出文件**：`results/service_type.csv`

**图表要素**：
- X 轴：n_service_types
- Y 轴：avg_delay (s)
- 多条曲线：Ours vs 各 Baseline

---

## 实验 4：网络规模对比

**扫描变量**：`scale = [small, medium, large]`

| 规模 | 节点数 | CPU 范围 | GPU 范围 (MB) |
|------|--------|----------|---------------|
| small | 3 | (8, 32) | (4096, 16384) |
| medium | 5 | (8, 64) | (4096, 32768) |
| large | 7 | (8, 128) | (4096, 65536) |

**固定参数**：
```
total_rate = 400
chain_length = 6
n_service_types = 15
n_chains = 4
```

**预期趋势**：
- 更大规模 → 更多资源 → 时延降低
- 小规模时资源极度紧张，定制化算法优势突出
- 大规模时资源充裕，差距收窄

**输出文件**：`results/scale_comparison.csv`

**图表要素**：
- X 轴：scale (small / medium / large)
- Y 轴：avg_delay (s)
- 分组柱状图或折线图：Ours vs 各 Baseline

---

## 实验 5：资源约束扫描 (核心实验)

**实验动机**：

Ours 算法可根据部署场景定制化调整服务配置（选择不同规格的模型、动态调整实例数等），Baseline 使用固定的模型配置。资源受限时，定制化优势最为显著——能灵活适配紧张资源，Baseline 被固定模型卡死。

**扫描变量**：`resource_factor = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]`

每个节点的实际容量 = base_capacity * resource_factor，同时作用于 CPU 和 GPU。

**固定参数**：
```
scale = medium
total_rate = 400
chain_length = 6
n_service_types = 15
n_chains = 4
```

**预期趋势**：

```
resource_factor   Ours (定制化)         Baseline (固定)
    0.3x          勉强维持, delay 适中   大量服务无法部署, delay 极高
    0.5x          delay 可接受           过载严重, 系统不稳定
    0.7x          接近最优               仍有部分过载
    1.0x          最优性能               基本可行但资源浪费
    1.5x          性能提升有限           赶上 Ours 水平
    2.0x          两者接近               两者接近
```

- **低资源区 (0.3x-0.7x)**：差距最大，是论文核心亮点
- **中等资源 (1.0x)**：Ours 仍优，定制化带来的效率提升可见
- **高资源区 (1.5x-2.0x)**：差距收窄，说明两者在资源充足时都能工作

**输出文件**：`results/resource_constraint.csv`

**图表要素**：
- X 轴：resource_factor
- Y 轴：avg_delay (s)
- 多条曲线：Ours vs 各 Baseline
- 可标注"资源充足区"和"资源紧张区"的分界线
- 这是论文中最重要的一张图，直接回答定制化部署的价值

**实现方式**：

在 experiment.py 中新增 `make_resource_constraint_experiment()`，扫描 resource_factor。每次运行时调用 `network.perturb_capacity(factor, factor)` 缩放节点资源。

---

## 实验 6：动态适应实验

**实验动机**：

边缘计算环境中的负载和资源是时变的（用户潮汐、硬件波动等）。Ours 算法能在环境变化后即时调整服务配置（换模型规格、增减实例），Baseline 使用固定配置无法响应变化。本实验模拟时变环境，直接展示定制化的动态适应能力。

**扫描方式**：时间步模拟（非参数扫描），每步环境随机扰动

**参数配置**：
```
scale = medium
total_rate (base) = 400
chain_length = 6
n_service_types = 15
n_chains = 4
n_steps = 50
```

**每步扰动方式**：
```
load_factor      ~ Uniform(0.5, 1.5)     # 到达率波动: base_rate * load_factor
capacity_factor  ~ Uniform(0.7, 1.3)     # 资源容量波动: CPU 和 GPU 同比例缩放
```

**各算法行为**：
```
Ours:       每步重新评估 + 定制化调整服务配置 → 适配新环境
Baseline:   固定配置不变 → 用初始部署应对波动的环境
```

**预期趋势**：

```
时间轴示意 (delay vs step):

  Ours:   ─┐     ┌─┐   ┌──    ← 波动但整体受控，快速回落
            │     │ └───┘
  Baseline: ──────╥═══════════  ← 负载突增后持续高位，无法调整
                   ╥
                  负载突增
```

- **稳态期**：两者差距小
- **负载突增 (load_factor > 1.3)**：Ours 快速调整配置，delay 短暂上升后回落；Baseline 无法响应，delay 持续高位
- **负载骤降 (load_factor < 0.7)**：Ours 可释放资源，Baseline 资源闲置浪费
- **整体**：Ours 的平均 delay 明显低于 Baseline，且波动幅度更小

**输出文件**：`results/dynamic_adaptation.csv`

**图表要素**：
- 主图：X 轴 = time step (0-49)，Y 轴 = avg_delay (s)
  - 多条折线：Ours vs 各 Baseline
  - 背景可用浅色带标注 load_factor 变化
- 副图 1：各算法的平均 delay 柱状图 (50 步取均值)
- 副图 2：各算法的 stable_chains 比例随时间变化

**实现方式**：

已有 `experiment_dynamic.py` 中的 `run_dynamic_experiment()` 框架。需要扩展：
1. 接受 `AlgorithmSuite` 参数，支持不同算法
2. Ours 算法每步重新调用 `solve()` 适配新环境
3. Baseline 只在第一步部署，后续步骤保持不变
4. 记录每步的 delay / comp / comm / penalty / stable / load_factor / capacity_factor

---

## 实现进度

| 实验 | 框架代码 | 随机基线数据 | Ours 算法 | Baseline 算法 | 出图 |
|------|---------|-------------|----------|--------------|------|
| 1. 到达率扰动 | done | done | pending | pending | pending |
| 2. 链长扰动 | done | done | pending | pending | pending |
| 3. 服务类型扰动 | done | done | pending | pending | pending |
| 4. 规模对比 | done | done | pending | pending | pending |
| 5. 资源约束扫描 | pending | pending | pending | pending | pending |
| 6. 动态适应 | done (基础框架) | pending | pending | pending | pending |

---

## 算法对比说明

| 算法 | 部署策略 | 服务配置 | 说明 |
|------|---------|---------|------|
| **Ours** | 定制化部署 | 根据场景动态选择模型规格 | 核心算法 |
| Baseline 1 | 固定模型部署 | 统一使用高性能模型 | 资源消耗大，低资源时难以部署 |
| Baseline 2 | 固定模型部署 | 统一使用轻量模型 | 资源占用少，但服务质量低 |
| Baseline 3 (可选) | 随机部署 | 随机选择模型 | 已有随机基线 |
