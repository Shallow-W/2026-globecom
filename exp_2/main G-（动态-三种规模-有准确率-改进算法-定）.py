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
        print(f"警告: 未找到 {excel_file}，将使用内置模拟数据进行演示。")
        random.seed(42)
        np.random.seed(42)
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
    # 扩充数据直至满足最大所需的任务数 (如20种服务)
    while len(tasks_list) < max_required_tasks:
        base_task = original_tasks[idx % len(original_tasks)]
        new_task_name = f"{base_task}_copy_{len(tasks_list)}"
        tasks_data[new_task_name] = copy.deepcopy(tasks_data[base_task])
        tasks_list.append(new_task_name)
        idx += 1
    return tasks_list, tasks_data


# ==========================================
# 2. 全局基础参数设置
# ==========================================
N_versions = 5
COMM_DELAY_CROSS_NODE = 0.02
BASE_NODE_FLOPS = 200 * 10 ** 9  # 单节点基准算力 200 GFLOPs
BASE_NODE_PARAMS = 150_000_000  # 单节点基准内存容量 150M Params


# ==========================================
# 3. 动态适应度评估 (支持可变节点数 n_nodes)
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


def evaluate_system_dynamic(individual, n_tasks, current_tasks, tasks_data, user_chains, Lambda_s, total_arrival_rate,
                            current_flops_cap, current_params_cap, n_nodes):
    X = np.array(individual).reshape((n_tasks, n_nodes, N_versions))
    total_delay = 0.0
    total_qos = 0.0
    total_accuracy = 0.0  # 记录系统平均真实准确率
    penalty_params = 0.0
    penalty_delay = 0.0

    # 1. 检查各节点的动态内存约束
    for e in range(n_nodes):
        node_params_used = 0
        for s_idx, task in enumerate(current_tasks):
            for v in range(N_versions):
                if v < len(tasks_data[task]):
                    node_params_used += int(X[s_idx, e, v]) * tasks_data[task][v]['model_params']
        if node_params_used > current_params_cap:
            penalty_params += (node_params_used - current_params_cap) / 1e6

    p = np.zeros((n_tasks, n_nodes, N_versions))
    for s_idx, task in enumerate(current_tasks):
        if Lambda_s[task] == 0: continue
        task_total_instances = np.sum(X[s_idx, :, :])
        if task_total_instances == 0:
            penalty_delay += 1000
            continue
        for e in range(n_nodes):
            for v in range(N_versions):
                if v >= len(tasks_data[task]): continue
                p[s_idx, e, v] = X[s_idx, e, v] / task_total_instances

    # 2. 计算动态排队时延、端到端 QoS 与 准确率
    for uc in user_chains:
        chain = uc['chain']
        weight = uc['rate'] / total_arrival_rate
        chain_qos = 0
        chain_acc = 0  # 单条链的准确率期望
        chain_delay = 0

        for task in chain:
            s_idx = current_tasks.index(task)
            expected_qos = 0
            expected_acc = 0  # 单个任务的准确率期望
            expected_task_delay = 0
            for e in range(n_nodes):
                for v in range(N_versions):
                    if v < len(tasks_data[task]):
                        prob = p[s_idx, e, v]
                        expected_qos += prob * tasks_data[task][v]['normalized_qos']
                        expected_acc += prob * tasks_data[task][v]['task_final_performance']

                        lam = Lambda_s[task] * prob
                        if lam > 0:
                            mu = current_flops_cap / tasks_data[task][v]['flops']
                            rate_per_inst = lam / X[s_idx, e, v]
                            if rate_per_inst >= mu:
                                penalty_delay += 100 * (rate_per_inst - mu + 1)
                                delay_node = 1.0
                            else:
                                delay_node = 1.0 / (mu - rate_per_inst)
                            expected_task_delay += prob * delay_node

            chain_qos += expected_qos
            chain_acc += expected_acc  # 累加准确率
            chain_delay += expected_task_delay

        total_qos += chain_qos * weight
        total_accuracy += chain_acc * weight  # 按流量权重累加全局准确率

        chain_comm_delay = 0
        for i in range(len(chain) - 1):
            s1_idx = current_tasks.index(chain[i])
            s2_idx = current_tasks.index(chain[i + 1])
            p_node_t1 = np.sum(p[s1_idx, :, :], axis=1)
            p_node_t2 = np.sum(p[s2_idx, :, :], axis=1)
            for e1 in range(n_nodes):
                for e2 in range(n_nodes):
                    if e1 != e2:
                        chain_comm_delay += p_node_t1[e1] * p_node_t2[e2] * COMM_DELAY_CROSS_NODE
        chain_delay += chain_comm_delay
        total_delay += chain_delay * weight

    total_penalty = penalty_params + penalty_delay
    fitness = total_qos - 5.0 * total_delay - total_penalty
    status = {'congested': penalty_delay > 0, 'oom': penalty_params > 0}

    return fitness, total_delay, total_qos, total_accuracy, total_penalty, status


# ==========================================
# 4. 代理驱动自适应学习引擎 (核心优化)
# ==========================================
def run_dynamic_environment_simulation(tasks_list, tasks_data, n_nodes, num_types, length, base_rate, dynamic_steps=50):
    user_chains, current_tasks = generate_user_chains(tasks_list, num_types, length, base_rate)
    n_tasks = len(current_tasks)
    GENES_LEN = n_tasks * n_nodes * N_versions
    pop_size = 40

    proxy_knowledge = {}
    for task in current_tasks:
        versions = tasks_data[task]
        proxy_knowledge[task] = {
            'qos': np.argmax([v['normalized_qos'] for v in versions]),
            'flops': np.argmin([v['flops'] for v in versions]),
            'params': np.argmin([v['model_params'] for v in versions])
        }

    pop = []
    # 增加极高精度先锋个体的比例，奠定效能基础
    ind_qos = np.zeros((n_tasks, n_nodes, N_versions), dtype=int)
    for s_idx, task in enumerate(current_tasks):
        ind_qos[s_idx, np.random.randint(n_nodes), proxy_knowledge[task]['qos']] = 1
    for _ in range(int(pop_size * 0.2)): pop.append(ind_qos.flatten())

    ind_fast = np.zeros((n_tasks, n_nodes, N_versions), dtype=int)
    for s_idx, task in enumerate(current_tasks):
        ind_fast[s_idx, np.random.randint(n_nodes), proxy_knowledge[task]['flops']] = 2
    for _ in range(int(pop_size * 0.1)): pop.append(ind_fast.flatten())

    while len(pop) < pop_size:
        ind = np.random.randint(0, 3, size=GENES_LEN)
        X = ind.reshape((n_tasks, n_nodes, N_versions))
        for s in range(n_tasks):
            if np.sum(X[s]) == 0: X[s, np.random.randint(n_nodes), np.random.randint(N_versions)] = 1
        pop.append(X.flatten())

    results = []

    for step in range(dynamic_steps):
        current_rate_multiplier = random.uniform(0.5, 1.5)
        current_flops_cap = BASE_NODE_FLOPS * random.uniform(0.7, 1.3)
        current_params_cap = BASE_NODE_PARAMS * random.uniform(0.7, 1.3)

        Lambda_s_dynamic = {t: 0 for t in current_tasks}
        total_arrival_rate_dyn = 0
        for uc in user_chains:
            r = int(uc['rate'] * current_rate_multiplier)
            total_arrival_rate_dyn += r
            for task in uc['chain']:
                Lambda_s_dynamic[task] += r

        best_fit = -1e9
        best_metrics = None

        generations_per_step = 10
        for gen in range(generations_per_step):
            evaluations = [
                evaluate_system_dynamic(ind, n_tasks, current_tasks, tasks_data, user_chains, Lambda_s_dynamic,
                                        total_arrival_rate_dyn, current_flops_cap, current_params_cap, n_nodes) for ind
                in pop
            ]

            for i, (fit, delay, qos, acc, pen, status) in enumerate(evaluations):
                if fit > best_fit:
                    best_fit = fit
                    best_metrics = (delay, qos, acc, pen)

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

            # ==== 核心进化与修复干预 ====
            for i in range(pop_size):
                status = evaluations[i][5]
                X = new_pop[i].reshape((n_tasks, n_nodes, N_versions))

                # 1. 约束感知贪心效能提升 (Greedy Proxy-Score Upgrade)
                # 如果没有触发内存溢出，则尽最大努力将模型版本提升至最高精度
                if not status['oom']:
                    node_usage = np.zeros(n_nodes)
                    for e in range(n_nodes):
                        for s_idx in range(n_tasks):
                            for v in range(N_versions):
                                node_usage[e] += X[s_idx, e, v] * tasks_data[current_tasks[s_idx]][v]['model_params']

                    # 见缝插针地进行升级
                    for s_idx in range(n_tasks):
                        for e in range(n_nodes):
                            deployed_vs = np.where(X[s_idx, e, :] > 0)[0]
                            for v in deployed_vs:
                                best_v = proxy_knowledge[current_tasks[s_idx]]['qos']
                                if v != best_v:
                                    task = current_tasks[s_idx]
                                    mem_diff = tasks_data[task][best_v]['model_params'] - tasks_data[task][v][
                                        'model_params']
                                    # 若当前节点的残余内存容量支持升级至顶配，则强制替换
                                    if node_usage[e] + mem_diff <= current_params_cap:
                                        X[s_idx, e, v] -= 1
                                        X[s_idx, e, best_v] += 1
                                        node_usage[e] += mem_diff

                # 2. 随机干扰与排队降压
                if random.random() < 0.4:
                    s_mut = random.randint(0, n_tasks - 1)
                    task_name = current_tasks[s_mut]
                    e_mut = random.randint(0, n_nodes - 1)

                    if status['congested']:
                        # 面对拥塞优先横向扩展最高精度模型进行分流，而不是牺牲精度
                        X[s_mut, e_mut, proxy_knowledge[task_name]['qos']] += 1
                    elif status['oom']:
                        X[s_mut, e_mut, :] = 0
                        X[s_mut, e_mut, proxy_knowledge[task_name]['params']] += 1
                    else:
                        X[s_mut, e_mut, random.randint(0, N_versions - 1)] += 1

                # 3. 防止服务链断供的最终兜底机制 (优先以最高精度修复)
                for s in range(n_tasks):
                    if np.sum(X[s]) == 0:
                        X[s, np.random.randint(n_nodes), proxy_knowledge[current_tasks[s]]['qos']] = 1

                new_pop[i] = X.flatten()

            pop = new_pop

        results.append({
            'Time_Step': step + 1,
            'Load_Factor': current_rate_multiplier,
            'Avail_Flops_Ratio': current_flops_cap / BASE_NODE_FLOPS,
            'Avail_Params_Ratio': current_params_cap / BASE_NODE_PARAMS,
            'Avg_Delay(s)': best_metrics[0],
            'Avg_QoS': best_metrics[1],
            'Avg_Accuracy': best_metrics[2],
            'Penalty(0=Safe)': best_metrics[3]
        })

    return results


# ==========================================
# 5. 执行：大、中、小 规模对比测试及最终总结
# ==========================================
if __name__ == "__main__":
    print("初始化系统，提取代理先验知识...")
    ALL_TASKS, TASKS_DATA = load_and_prepare_data("evaluation_tables.xlsx")

    scales_config = [
        {'name': '小规模 (Small)', 'n_nodes': 3, 'num_types': 5, 'length': 3, 'base_rate': 100},
        {'name': '中等规模 (Medium)', 'n_nodes': 5, 'num_types': 10, 'length': 5, 'base_rate': 200},
        {'name': '大规模 (Large)', 'n_nodes': 7, 'num_types': 20, 'length': 7, 'base_rate': 300}
    ]

    # 新增：用于记录各规模最终的总体平均统计结果
    overall_summaries = []

    for config in scales_config:
        print("\n" + "=" * 65)
        print(f" 开始验证: {config['name']} 网络环境")
        print(
            f" 参数: {config['n_nodes']}节点 | {config['num_types']}类服务 | 链长={config['length']} | 基准到达率={config['base_rate']}")
        print("=" * 65)

        dynamic_results = run_dynamic_environment_simulation(
            ALL_TASKS, TASKS_DATA,
            n_nodes=config['n_nodes'],
            num_types=config['num_types'],
            length=config['length'],
            base_rate=config['base_rate'],
            dynamic_steps=50
        )

        df_dyn = pd.DataFrame(dynamic_results)
        pd.set_option('display.float_format', lambda x: '%.4f' % x)

        print(df_dyn.head(10).to_string(index=False))
        print("      ... 持续跟踪 ...")
        print(df_dyn.tail(5).to_string(index=False))

        failed_steps = len(df_dyn[df_dyn['Penalty(0=Safe)'] > 0])
        print(f"\n[{config['name']} 结论] 50次剧变中发生拥塞或溢出的次数: {failed_steps} 次。")

        filename = f"Dynamic_Results_{config['name'].split(' ')[0]}.csv"
        df_dyn.to_csv(filename, index=False)
        print(f"-> 详细数据已保存至: {filename}")

        # ==== 核心新增：统计计算 50 步下的整体均值 ====
        avg_delay_overall = df_dyn['Avg_Delay(s)'].mean()
        avg_qos_overall = df_dyn['Avg_QoS'].mean()
        avg_acc_overall = df_dyn['Avg_Accuracy'].mean()

        overall_summaries.append({
            'Scale': config['name'].split(' ')[0],
            'Overall_Avg_Delay': avg_delay_overall,
            'Overall_Avg_QoS': avg_qos_overall,
            'Overall_Avg_Acc': avg_acc_overall
        })

    print("\n所有不同规模的网络动态验证全部运行完毕。")

    # ==========================================
    # 6. 新增功能：输出各规模的全局统计总结表格
    # ==========================================
    print("\n" + "=" * 70)
    print(" 📊 总结：50次环境变化下各规模网络的全局总体表现平均值")
    print("=" * 70)
    print(f"{'网络规模':<10} | {'总平均时延 (s)':<14} | {'总平均综合效能 (QoS)':<18} | {'总平均真实准确率':<15}")
    print("-" * 70)
    for res in overall_summaries:
        print(
            f"{res['Scale']:<12} | {res['Overall_Avg_Delay']:<18.4f} | {res['Overall_Avg_QoS']:<23.4f} | {res['Overall_Avg_Acc']:.4f}")
    print("=" * 70)