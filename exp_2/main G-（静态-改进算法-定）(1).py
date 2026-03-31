import pandas as pd
import numpy as np
import random
import copy


# ==========================================
# 1. 代理评估结果注入与数据扩充
# ==========================================
def load_and_prepare_data(excel_file="evaluation_tables.xlsx", max_required_tasks=50):
    tasks_data = {}
    try:
        xl = pd.ExcelFile(excel_file)
        for task_name in xl.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=task_name)
            top5 = df.sort_values(by="proxy_score", ascending=False).head(5).copy()
            min_q = df['task_final_performance'].min()
            max_q = df['task_final_performance'].max()
            if max_q > min_q:
                top5['normalized_qos'] = 0.1 + 0.9 * (top5['task_final_performance'] - min_q) / (max_q - min_q)
            else:
                top5['normalized_qos'] = 1.0

            tasks_data[task_name] = top5[
                ['architecture', 'proxy_score', 'task_final_performance', 'normalized_qos', 'model_params', 'flops']
            ].to_dict('records')
    except FileNotFoundError:
        print(f"警告: 未找到 {excel_file}，将使用模拟数据进行演示。")
        for i in range(7):
            task_name = f"mock_task_{i}"
            tasks_data[task_name] = [
                {'architecture': f'arch_{v}', 'proxy_score': random.uniform(0.5, 2.0),
                 'task_final_performance': random.uniform(0.5, 0.9), 'normalized_qos': random.uniform(0.5, 1.0),
                 'model_params': random.randint(1_000_000, 20_000_000),
                 'flops': random.randint(100_000_000, 2_000_000_000)}
                for v in range(5)
            ]

    original_tasks = list(tasks_data.keys())
    tasks_list = copy.deepcopy(original_tasks)
    idx = 0
    while len(tasks_list) < max_required_tasks:
        base_task = original_tasks[idx % len(original_tasks)]
        new_task_name = f"{base_task}_copy_{len(tasks_list)}"
        tasks_data[new_task_name] = copy.deepcopy(tasks_data[base_task])
        tasks_list.append(new_task_name)
        idx += 1
    return tasks_list, tasks_data


# ==========================================
# 2. 边缘环境与全局参数
# ==========================================
N_nodes = 3
N_versions = 5
NODE_FLOPS_CAPACITY = 200 * 10 ** 9
MAX_NODE_PARAMS = 150_000_000
COMM_DELAY_CROSS_NODE = 0.02


# ==========================================
# 3. 实验场景生成与适应度评估
# ==========================================
def generate_user_chains(available_tasks, num_types, length, total_rate):
    chains = []
    current_tasks = available_tasks[:num_types]
    num_chains = 4
    rates = np.random.dirichlet(np.ones(num_chains)) * total_rate
    rates = np.maximum(1, np.round(rates)).astype(int)
    rates[0] += total_rate - np.sum(rates)
    for i in range(num_chains):
        chain = random.choices(current_tasks, k=length)
        chains.append({'chain': chain, 'rate': rates[i]})
    return chains, current_tasks


def evaluate_system(individual, n_tasks, current_tasks, tasks_data, user_chains, Lambda_s, total_arrival_rate):
    X = np.array(individual).reshape((n_tasks, N_nodes, N_versions))
    total_delay = 0.0
    total_qos = 0.0
    penalty_params = 0.0
    penalty_delay = 0.0

    total_comm_delay = 0.0
    total_comp_delay = 0.0
    total_params_used = 0

    # 1. 计算内存使用率 (有了强制修复算子，这里永远不会溢出)
    for e in range(N_nodes):
        node_params_used = 0
        for s_idx, task in enumerate(current_tasks):
            for v in range(N_versions):
                if v < len(tasks_data[task]):
                    node_params_used += int(X[s_idx, e, v]) * tasks_data[task][v]['model_params']
        total_params_used += node_params_used
        if node_params_used > MAX_NODE_PARAMS:
            penalty_params += 1e6  # 留一个保险丝

    mem_utilization = total_params_used / (N_nodes * MAX_NODE_PARAMS)

    p = np.zeros((n_tasks, N_nodes, N_versions))
    for s_idx, task in enumerate(current_tasks):
        if Lambda_s[task] == 0: continue
        task_total_instances = np.sum(X[s_idx, :, :])
        if task_total_instances == 0:
            penalty_delay += 1000  # 断供惩罚
            continue
        for e in range(N_nodes):
            for v in range(N_versions):
                if v >= len(tasks_data[task]): continue
                p[s_idx, e, v] = X[s_idx, e, v] / task_total_instances

    for uc in user_chains:
        chain = uc['chain']
        weight = uc['rate'] / total_arrival_rate
        chain_qos = 0
        chain_comp = 0
        chain_comm = 0

        for task in chain:
            s_idx = current_tasks.index(task)
            expected_qos = 0
            expected_task_delay = 0
            for e in range(N_nodes):
                for v in range(N_versions):
                    if v < len(tasks_data[task]):
                        expected_qos += p[s_idx, e, v] * tasks_data[task][v]['normalized_qos']
                        lam = Lambda_s[task] * p[s_idx, e, v]
                        if lam > 0:
                            mu = NODE_FLOPS_CAPACITY / tasks_data[task][v]['flops']
                            rate_per_inst = lam / X[s_idx, e, v]
                            if rate_per_inst >= mu:
                                penalty_delay += 1000 * (float(rate_per_inst) - float(mu) + 1.0)
                                delay_node = 1.0
                            else:
                                delay_node = 1.0 / (mu - rate_per_inst)
                            expected_task_delay += p[s_idx, e, v] * delay_node
            chain_qos += expected_qos
            chain_comp += expected_task_delay

        for i in range(len(chain) - 1):
            s1_idx = current_tasks.index(chain[i])
            s2_idx = current_tasks.index(chain[i + 1])
            p_node_t1 = np.sum(p[s1_idx, :, :], axis=1)
            p_node_t2 = np.sum(p[s2_idx, :, :], axis=1)
            for e1 in range(N_nodes):
                for e2 in range(N_nodes):
                    if e1 != e2:
                        chain_comm += p_node_t1[e1] * p_node_t2[e2] * COMM_DELAY_CROSS_NODE

        total_qos += chain_qos * weight
        total_comp_delay += chain_comp * weight
        total_comm_delay += chain_comm * weight

    total_delay = total_comp_delay + total_comm_delay
    total_penalty = penalty_params + penalty_delay
    fitness = total_qos - 5.0 * total_delay - total_penalty

    status = {'congested': penalty_delay > 0, 'oom': penalty_params > 0}
    return fitness, total_delay, total_qos, total_penalty, total_comp_delay, total_comm_delay, mem_utilization, status


# ==========================================
# 4. [算法升级核心] 物理空间强制修复算子 & 进化逻辑
# ==========================================
def run_proxy_driven_evolution(tasks_list, tasks_data, num_types, length, total_rate, generations=50, pop_size=40):
    user_chains, current_tasks = generate_user_chains(tasks_list, num_types, length, total_rate)
    n_tasks = len(current_tasks)

    total_arrival_rate = sum(uc['rate'] for uc in user_chains)
    Lambda_s = {t: 0 for t in current_tasks}
    for uc in user_chains:
        for task in uc['chain']:
            Lambda_s[task] += uc['rate']

    GENES_LEN = n_tasks * N_nodes * N_versions
    proxy_knowledge = {}
    for task in current_tasks:
        versions = tasks_data[task]
        proxy_knowledge[task] = {
            'qos': np.argmax([v['normalized_qos'] for v in versions]),
            'flops': np.argmin([v['flops'] for v in versions]),
            'params': np.argmin([v['model_params'] for v in versions])
        }

    # ---- 核心新增：强制物理容量修复算子 ----
    def repair_individual(X_ind):
        X = X_ind.reshape((n_tasks, N_nodes, N_versions))
        node_usage = np.zeros(N_nodes)
        for s_idx in range(n_tasks):
            for e in range(N_nodes):
                for v in range(N_versions):
                    if X[s_idx, e, v] > 0:
                        node_usage[e] += X[s_idx, e, v] * tasks_data[current_tasks[s_idx]][v]['model_params']

        # 1. 智能保底分配：服务断供时，优先挑选集群中最空闲的节点部署极小模型
        for s_idx in range(n_tasks):
            if np.sum(X[s_idx]) == 0:
                task = current_tasks[s_idx]
                min_v = proxy_knowledge[task]['params']
                req_mem = tasks_data[task][min_v]['model_params']

                best_node = np.argmin(node_usage)
                X[s_idx, best_node, min_v] = 1
                node_usage[best_node] += req_mem

        # 2. 绝对硬约束拦截：对超载的节点疯狂瘦身，压死在100%以内
        for e in range(N_nodes):
            while node_usage[e] > MAX_NODE_PARAMS:
                max_p, t_s, t_v = -1, -1, -1
                # 揪出该节点上占据最多内存的实例
                for s_idx in range(n_tasks):
                    for v in range(N_versions):
                        if X[s_idx, e, v] > 0:
                            p_size = tasks_data[current_tasks[s_idx]][v]['model_params']
                            if p_size > max_p:
                                max_p = p_size
                                t_s, t_v = s_idx, v

                if t_s == -1: break  # 安全锁

                task = current_tasks[t_s]
                min_v = proxy_knowledge[task]['params']
                min_p = tasks_data[task][min_v]['model_params']

                if t_v != min_v and max_p > min_p:
                    # 优先选择降级
                    X[t_s, e, t_v] -= 1
                    X[t_s, e, min_v] += 1
                    node_usage[e] -= (max_p - min_p)
                else:
                    # 已经被压榨到极限了，宁可让服务断供扣1000分，也要保住物理内存不爆
                    X[t_s, e, t_v] -= 1
                    node_usage[e] -= max_p

        return X.flatten()

    # ---------------------------------------------

    pop = []
    ind_tiny = np.zeros((n_tasks, N_nodes, N_versions), dtype=int)
    for s_idx, task in enumerate(current_tasks):
        ind_tiny[s_idx, np.random.randint(N_nodes), proxy_knowledge[task]['params']] = 1
    for _ in range(int(pop_size * 0.1)):
        pop.append(repair_individual(ind_tiny.flatten()))  # 放入种群前强制体检

    ind_qos = np.zeros((n_tasks, N_nodes, N_versions), dtype=int)
    for s_idx, task in enumerate(current_tasks):
        ind_qos[s_idx, np.random.randint(N_nodes), proxy_knowledge[task]['qos']] = 1
    for _ in range(int(pop_size * 0.1)):
        pop.append(repair_individual(ind_qos.flatten()))

    while len(pop) < pop_size:
        ind = np.random.randint(0, 2, size=GENES_LEN)
        X = ind.reshape((n_tasks, N_nodes, N_versions))
        for s in range(n_tasks):
            if np.sum(X[s]) == 0: X[s, np.random.randint(N_nodes), np.random.randint(N_versions)] = 1
        pop.append(repair_individual(X.flatten()))

    best_fit = -1e9
    best_metrics = None

    for gen in range(generations):
        evaluations = [
            evaluate_system(ind, n_tasks, current_tasks, tasks_data, user_chains, Lambda_s, total_arrival_rate) for ind
            in pop
        ]

        for i, (fit, delay, qos, pen, comp_d, comm_d, mem_u, status) in enumerate(evaluations):
            if fit > best_fit:
                best_fit = fit
                best_metrics = (delay, qos, pen, comp_d, comm_d, mem_u)

        new_pop = []
        for _ in range(pop_size):
            i1, i2 = random.sample(range(pop_size), 2)
            new_pop.append(pop[i1].copy() if evaluations[i1][0] > evaluations[i2][0] else pop[i2].copy())

        for i in range(0, pop_size - 1, 2):
            if random.random() < 0.8:
                pt1 = random.randint(1, GENES_LEN // 2)
                pt2 = random.randint(GENES_LEN // 2, GENES_LEN - 1)
                new_pop[i][pt1:pt2], new_pop[i + 1][pt1:pt2] = new_pop[i + 1][pt1:pt2].copy(), new_pop[i][
                                                                                               pt1:pt2].copy()

        for i in range(pop_size):
            if random.random() < 0.4:
                status = evaluations[i][7]
                X = new_pop[i].reshape((n_tasks, N_nodes, N_versions))
                s_mut = random.randint(0, n_tasks - 1)
                task_name = current_tasks[s_mut]
                e_mut = random.randint(0, N_nodes - 1)

                if status['congested'] and not status['oom']:
                    X[s_mut, e_mut, proxy_knowledge[task_name]['qos']] += 1
                elif status['congested'] and status['oom']:
                    X[s_mut, e_mut, :] = 0
                    X[s_mut, e_mut, proxy_knowledge[task_name]['flops']] = 1
                elif status['oom'] and not status['congested']:
                    X[s_mut, e_mut, :] = 0
                    X[s_mut, e_mut, proxy_knowledge[task_name]['params']] = 1
                else:
                    if random.random() < 0.6:
                        X[s_mut, e_mut, proxy_knowledge[task_name]['qos']] += 1
                    else:
                        X[s_mut, e_mut, random.randint(0, N_versions - 1)] += 1

            # 变异完成后，交给修复算子进行严格审查与拦截
            new_pop[i] = repair_individual(new_pop[i])

        pop = new_pop

    return best_metrics


# ==========================================
# 5. 主程序执行并保存 CSV
# ==========================================
if __name__ == "__main__":
    print("正在加载并准备模型数据 (初始化代理评估图谱)...")
    ALL_TASKS, TASKS_DATA = load_and_prepare_data("evaluation_tables.xlsx", max_required_tasks=50)
    print(f"数据准备完毕。包含候选任务总数: {len(ALL_TASKS)}")

    results_data = []


    def record_metrics(experiment_name, var_val, metrics):
        delay, qos, pen, comp_d, comm_d, mem_u = metrics
        print(
            f"-> D={delay:.4f}s (排队={comp_d:.4f}s, 通信={comm_d:.4f}s) | Q={qos:.4f} | 内存利用={mem_u:.2%} | 违规惩罚={pen:.1f}")
        results_data.append({
            'Experiment': experiment_name,
            'Variable_Value': var_val,
            'Total_Delay_D': delay,
            'Avg_QoS_Q': qos,
            'Comp_Delay': comp_d,
            'Comm_Delay': comm_d,
            'Mem_Utilization': mem_u,
            'Penalty_Score': pen
        })


    print("\n" + "=" * 65)
    print(" 实验 1：不同请求到达率对多维指标的影响 (绝对硬约束保护)")
    print(" (固定: 任务种类=10, 链长度=4)")
    print("=" * 65)
    rates = [100, 200, 300, 400, 500, 600, 700, 800]
    for r in rates:
        print(f"请求到达率 λ = {r: <4}", end=" ")
        m = run_proxy_driven_evolution(ALL_TASKS, TASKS_DATA, num_types=10, length=4, total_rate=r)
        record_metrics('不同请求到达率', r, m)

    print("\n" + "=" * 65)
    print(" 实验 2：不同请求链路长度对多维指标的影响 (绝对硬约束保护)")
    print(" (固定: 任务种类=10, 到达率=200)")
    print("=" * 65)
    lengths = [3, 4, 5, 6, 7, 8, 9, 10]
    for l in lengths:
        print(f"服务链长度 K = {l: <4}", end=" ")
        m = run_proxy_driven_evolution(ALL_TASKS, TASKS_DATA, num_types=10, length=l, total_rate=200)
        record_metrics('不同服务链长度', l, m)

    print("\n" + "=" * 65)
    print(" 实验 3：不同服务类型数目对多维指标的影响 (绝对硬约束保护)")
    print(" (固定: 链长度=4, 到达率=200)")
    print("=" * 65)
    types = [10, 20, 30, 40, 50, 60, 70, 80]
    for t in types:
        print(f"服务类型数 S = {t: <4}", end=" ")
        m = run_proxy_driven_evolution(ALL_TASKS, TASKS_DATA, num_types=t, length=4, total_rate=200)
        record_metrics('不同服务类型数', t, m)

    # 导出 CSV
    df_results = pd.DataFrame(results_data)
    df_results.to_csv("静态时延能效.csv", index=False, encoding='utf-8-sig')

    print("\n代理驱动启发式算法 (PD-HEA) 实验验证运行完毕。")
    print("运行结果已保存至: 静态时延能效.csv")

    # ==========================================
    # 6. 新增功能：输出三种环境下的各项平均值汇总
    # ==========================================
    print("\n" + "=" * 90)
    print(" 📊 总结：三种实验环境下各项性能指标的总体平均值")
    print("=" * 90)

    # 按照 'Experiment' 分组求各项指标的均值
    summary_df = df_results.groupby('Experiment').mean().reset_index()

    # 格式化输出表头
    print(
        f"{'实验环境':<16} | {'总时延(s)':<10} | {'效能(QoS)':<10} | {'排队时延(s)':<12} | {'通信时延(s)':<12} | {'内存利用率':<12} | {'违规惩罚'}")
    print("-" * 90)

    # 逐行打印统计结果
    for index, row in summary_df.iterrows():
        exp_name = row['Experiment']
        d = row['Total_Delay_D']
        q = row['Avg_QoS_Q']
        comp_d = row['Comp_Delay']
        comm_d = row['Comm_Delay']
        mem_u = row['Mem_Utilization']
        pen = row['Penalty_Score']

        # 针对中文字符对齐做简单适配，固定宽度显示
        print(
            f"{exp_name:<16} | {d:<13.4f} | {q:<11.4f} | {comp_d:<15.4f} | {comm_d:<15.4f} | {mem_u:<13.2%} | {pen:.1f}")

    print("=" * 90)