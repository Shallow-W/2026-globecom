"""
微服务路由与部署实验框架 - 单元测试
验证各模块计算正确性和框架整体一致性
"""
import sys
import os
import csv
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    NETWORK_SCALES, SERVICE_RATE_RANGE, GPU_PER_INSTANCE_RANGE,
    DEFAULT_TOTAL_RATE, DEFAULT_CHAIN_LENGTH, DEFAULT_N_SERVICE_TYPES,
    INSTABILITY_PENALTY_FACTOR, INSTABILITY_FIXED_DELAY,
)
from network import EdgeNetwork
from service import (
    Service, ServiceChain,
    generate_services, generate_service_chains,
    compute_aggregate_arrival_rates,
)
from deployment import Deployment, random_deployment
from routing import proportional_routing, random_routing
from queuing import mm_c_response_time, compute_chain_delay
from evaluation import evaluate
from algorithm import (
    AlgorithmSuite, ALGO_RANDOM_PROPORTIONAL, ALGO_RANDOM_RANDOM,
    ALL_ALGORITHMS, get_algorithm,
)
from experiment import (
    Experiment, ExperimentRunner, run_single,
    make_arrival_rate_experiment,
    make_chain_length_experiment,
    make_service_type_experiment,
)


passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ============================================================
# 1. M/M/c 排队公式正确性
# ============================================================
def test_queuing():
    print("\n=== 1. M/M/c 排队公式 ===")

    # M/M/1 已知解: W = 1/(mu - lambda)
    W, stable = mm_c_response_time(1.0, 2.0, 1)
    expected = 1.0 / (2.0 - 1.0)
    check("M/M/1: lambda=1, mu=2, c=1 -> W=1.0",
          abs(W - expected) < 1e-10, f"got {W:.6f}, expected {expected:.6f}")

    W, stable = mm_c_response_time(0.5, 1.0, 1)
    expected = 1.0 / (1.0 - 0.5)
    check("M/M/1: lambda=0.5, mu=1, c=1 -> W=2.0",
          abs(W - expected) < 1e-10, f"got {W:.6f}, expected {expected:.6f}")

    W, stable = mm_c_response_time(5.0, 10.0, 1)
    expected = 1.0 / (10.0 - 5.0)
    check("M/M/1: lambda=5, mu=10, c=1 -> W=0.2",
          abs(W - expected) < 1e-10, f"got {W:.6f}, expected {expected:.6f}")

    # 稳定性
    _, stable = mm_c_response_time(2.0, 1.0, 1)
    check("M/M/1 不稳定: lambda=2, mu=1, c=1", not stable)

    _, stable = mm_c_response_time(3.0, 1.0, 2)
    check("M/M/2 不稳定: lambda=3, mu=1, c=2 -> rho=1.5", not stable)

    _, stable = mm_c_response_time(1.9, 1.0, 2)
    check("M/M/2 稳定: lambda=1.9, mu=1, c=2 -> rho=0.95", stable)

    # c 越多时延越小
    W1, _ = mm_c_response_time(3.0, 3.0, 1)
    W2, _ = mm_c_response_time(3.0, 3.0, 2)
    W3, _ = mm_c_response_time(3.0, 3.0, 4)
    check("c=1 不稳定", W1 == INSTABILITY_FIXED_DELAY)
    check("c=4 时延 < c=2 时延", W3 < W2, f"c=2 W={W2:.6f}, c=4 W={W3:.6f}")

    # 边界
    W, stable = mm_c_response_time(0.0, 1.0, 1)
    check("lambda=0 -> delay=0", W == 0.0 and stable)

    W, stable = mm_c_response_time(1.0, 1.0, 0)
    check("c=0 -> delay=0", W == 0.0)

    # W >= 1/mu
    W, _ = mm_c_response_time(1.0, 3.0, 1)
    check("W >= 1/mu", W >= 1.0 / 3.0 - 1e-10)

    # 单调性
    W_low, _ = mm_c_response_time(1.0, 5.0, 1)
    W_high, _ = mm_c_response_time(4.0, 5.0, 1)
    check("负载越高时延越大", W_high > W_low)

    W_c1, _ = mm_c_response_time(2.0, 3.0, 1)
    W_c2, _ = mm_c_response_time(2.0, 3.0, 2)
    W_c3, _ = mm_c_response_time(2.0, 3.0, 3)
    check("c 越多时延越小: 1>2>3", W_c1 > W_c2 > W_c3)


# ============================================================
# 2. 网络拓扑
# ============================================================
def test_network():
    print("\n=== 2. 网络拓扑 ===")

    net = EdgeNetwork(scale='small', seed=42)
    check("small: 3 节点", net.n_nodes == 3)
    check("small: CPU 在范围 [8,32]", all(8 <= c <= 32 for c in net.cpu_capacity))
    check("small: GPU 在范围 [4096,16384]", all(4096 <= g <= 16384 for g in net.gpu_capacity))
    check("异构: 节点 CPU 不全相同", len(set(net.cpu_capacity)) > 1,
          f"cpu={net.cpu_capacity}")
    check("时延矩阵对称", np.allclose(net.comm_delay, net.comm_delay.T))
    check("对角线=0", np.all(np.diag(net.comm_delay) == 0))
    check("非对角线>0", np.all(net.comm_delay[np.triu_indices(net.n_nodes, k=1)] > 0))

    saved_cpu = net.cpu_capacity.copy()
    saved_gpu = net.gpu_capacity.copy()
    net.cpu_capacity *= 0.5
    net.reset_resources()
    check("reset 恢复", np.allclose(net.cpu_capacity, saved_cpu))

    net.perturb_capacity(0.8, 1.2)
    check("perturb CPU: base*0.8", np.allclose(net.cpu_capacity, saved_cpu * 0.8))
    check("perturb GPU: base*1.2", np.allclose(net.gpu_capacity, saved_gpu * 1.2))

    ns = EdgeNetwork(scale='small', seed=0)
    nm = EdgeNetwork(scale='medium', seed=0)
    nl = EdgeNetwork(scale='large', seed=0)
    check("节点数递增", ns.n_nodes < nm.n_nodes < nl.n_nodes)


# ============================================================
# 3. 微服务与服务链
# ============================================================
def test_services():
    print("\n=== 3. 微服务与服务链 ===")

    services = generate_services(20, seed=42)
    check("生成 20 个服务", len(services) == 20)
    check("ID 连续", all(s.id == i for i, s in enumerate(services)))

    for svc in services:
        check(f"svc{svc.id} mu 在范围",
              SERVICE_RATE_RANGE[0] <= svc.service_rate <= SERVICE_RATE_RANGE[1])
        if not (SERVICE_RATE_RANGE[0] <= svc.service_rate <= SERVICE_RATE_RANGE[1]):
            break

    for svc in services[:3]:
        check(f"svc{svc.id} gpu 在范围",
              GPU_PER_INSTANCE_RANGE[0] <= svc.gpu_per_instance <= GPU_PER_INSTANCE_RANGE[1])

    chains = generate_service_chains(4, 20, 6, 400, seed=42)
    check("4 条链", len(chains) == 4)
    check("每条长 6", all(len(c.services) == 6 for c in chains))
    total_rate = sum(c.arrival_rate for c in chains)
    check(f"总到达率=400", abs(total_rate - 400) < 1e-6)
    check("每条>0", all(c.arrival_rate > 0 for c in chains))

    lambda_s = compute_aggregate_arrival_rates(chains, 20)
    check("长度=20", len(lambda_s) == 20)
    check(">=0", all(l >= 0 for l in lambda_s))

    chains_simple = [ServiceChain(0, [0, 1, 2], 10.0), ServiceChain(1, [1, 2, 3], 20.0)]
    lam = compute_aggregate_arrival_rates(chains_simple, 4)
    check("svc0=10", abs(lam[0] - 10) < 1e-10)
    check("svc1=30", abs(lam[1] - 30) < 1e-10)
    check("svc2=30", abs(lam[2] - 30) < 1e-10)
    check("svc3=20", abs(lam[3] - 20) < 1e-10)


# ============================================================
# 4. 部署
# ============================================================
def test_deployment():
    print("\n=== 4. 部署 ===")

    net = EdgeNetwork(scale='small', seed=42)
    services = generate_services(10, seed=42)

    dep, cpu_rem, gpu_rem = random_deployment(net, services, seed=42)
    check("形状正确", dep.X.shape == (3, 10))
    check("每个服务>=1实例",
          all(dep.total_instances(s.id) >= 1 for s in services))

    for v in range(net.n_nodes):
        check(f"node{v} CPU 未超", cpu_rem[v] >= -1e-9)
        check(f"node{v} GPU 未超", gpu_rem[v] >= -1e-9)

    for s in services:
        p = dep.routing_probabilities(s.id)
        if dep.total_instances(s.id) > 0:
            check(f"svc{s.id} 概率和=1", abs(sum(p) - 1.0) < 1e-10)
            check(f"svc{s.id} 概率只在有实例节点",
                  np.count_nonzero(p) == np.count_nonzero(dep.X[:, s.id]))
        else:
            check(f"svc{s.id} 未部署->全0", np.all(p == 0))

    chains = generate_service_chains(4, 10, 4, 200, seed=42)
    lambda_s = compute_aggregate_arrival_rates(chains, 10)
    dep2, _, _ = random_deployment(net, services, lambda_s=lambda_s, seed=42)
    active = [s for s in services if lambda_s[s.id] > 0]
    check("有负载均已部署", all(dep2.total_instances(s.id) >= 1 for s in active))

    inst_no = sum(dep.total_instances(s.id) for s in services)
    inst_with = sum(dep2.total_instances(s.id) for s in services)
    check("带负载实例数>=无负载", inst_with >= inst_no)


# ============================================================
# 5. 路由
# ============================================================
def test_routing():
    print("\n=== 5. 路由 ===")

    dep = Deployment(3, 4)
    dep.X = np.array([[2, 0, 1, 3], [0, 4, 0, 1], [1, 0, 2, 0]])

    rt = proportional_routing(dep)
    p0 = rt.get_route(0, 0)
    check("svc0: 2/3, 0, 1/3", abs(p0[0] - 2/3) < 1e-10 and p0[1] == 0 and abs(p0[2] - 1/3) < 1e-10)

    p3 = rt.get_route(0, 3)
    check("svc3: 3/4, 1/4, 0", abs(p3[0] - 3/4) < 1e-10 and abs(p3[1] - 1/4) < 1e-10)

    check("比例路由与源无关",
          np.allclose(rt.get_route(0, 0), rt.get_route(1, 0)))

    rt_rand = random_routing(dep, seed=42)
    p_r0 = rt_rand.get_route(0, 0)
    check("随机路由: 非零在 node0,2", p_r0[0] > 0 and p_r0[2] > 0 and p_r0[1] == 0)
    check("随机路由: 概率和=1", abs(sum(p_r0) - 1.0) < 1e-10)

    p_r1 = rt_rand.get_route(0, 1)
    check("随机路由 svc1 全到 node1", abs(p_r1[1] - 1.0) < 1e-10)


# ============================================================
# 6. 链时延计算
# ============================================================
def test_chain_delay():
    print("\n=== 6. 链时延计算 ===")

    services = [Service(0, 10.0, 256), Service(1, 10.0, 256), Service(2, 10.0, 256)]
    net = EdgeNetwork(scale='small', seed=42)
    chain = ServiceChain(0, [0, 1, 2], 5.0)

    dep = Deployment(3, 3)
    dep.X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    lambda_s = np.array([5.0, 5.0, 5.0])
    result = compute_chain_delay(chain, services, dep, net, lambda_s)

    check("finite", result['total'] < float('inf'))
    check("stable", result['stable'])
    check("comp>0", result['comp'] > 0)
    check("comm>0", result['comm'] > 0)
    check("penalty=0", result['penalty'] == 0.0)

    expected_comp = 3 * (1.0 / (10.0 - 5.0))
    check(f"comp = 3*0.2 = {expected_comp:.2f}",
          abs(result['comp'] - expected_comp) < 0.01,
          f"got {result['comp']:.6f}")

    dep_same = Deployment(3, 3)
    dep_same.X = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    result_same = compute_chain_delay(chain, services, dep_same, net, lambda_s)
    check("同节点 comm=0", result_same['comm'] == 0.0)

    chain_heavy = ServiceChain(0, [0], 50.0)
    lambda_heavy = np.array([50.0, 0.0, 0.0])
    result_heavy = compute_chain_delay(chain_heavy, services, dep, net, lambda_heavy)
    check("高负载不稳定", not result_heavy['stable'])
    check("高负载有惩罚", result_heavy['penalty'] > 0)

    dep_miss = Deployment(3, 3)
    dep_miss.X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    result_miss = compute_chain_delay(chain, services, dep_miss, net, lambda_s)
    check("未部署->inf", result_miss['total'] == float('inf'))


# ============================================================
# 7. 评估模块
# ============================================================
def test_evaluation():
    print("\n=== 7. 评估模块 ===")

    net = EdgeNetwork(scale='small', seed=42)
    services = generate_services(10, seed=42)
    chains = generate_service_chains(4, 10, 4, 200, seed=42)
    dep, _, _ = random_deployment(net, services, seed=42)
    metrics = evaluate(dep, services, chains, net)

    check("delay 有限正", 0 < metrics['avg_delay'] < float('inf'))
    check("comp>=0", metrics['avg_comp_delay'] >= 0)
    check("comm>=0", metrics['avg_comm_delay'] >= 0)
    check("penalty>=0", metrics['total_penalty'] >= 0)
    check("total_chains=4", metrics['total_chains'] == 4)
    check("stable_chains in [0,4]", 0 <= metrics['stable_chains'] <= 4)
    check("cpu_util in [0,1]", 0 <= metrics['cpu_utilization'] <= 1)
    check("gpu_util in [0,1]", 0 <= metrics['gpu_utilization'] <= 1)
    check("details len=4", len(metrics['chain_details']) == 4)
    check("delay = comp + comm + penalty",
          abs(metrics['avg_delay'] - metrics['avg_comp_delay']
              - metrics['avg_comm_delay'] - metrics['total_penalty']) < 1e-6)


# ============================================================
# 8. 算法策略层
# ============================================================
def test_algorithm():
    print("\n=== 8. 算法策略层 ===")

    # AlgorithmSuite.solve 正常工作
    net = EdgeNetwork(scale='small', seed=42)
    services = generate_services(10, seed=42)
    chains = generate_service_chains(4, 10, 4, 200, seed=42)
    lambda_s = compute_aggregate_arrival_rates(chains, len(services))

    dep, rt = ALGO_RANDOM_PROPORTIONAL.solve(net, services, lambda_s, seed=42)
    check("solve 返回 Deployment", isinstance(dep, Deployment))
    check("solve 返回 RoutingTable", rt is not None)
    check("每个服务已部署",
          all(dep.total_instances(s.id) >= 1 for s in services))

    # 两个不同算法产生不同部署
    dep1, _ = ALGO_RANDOM_PROPORTIONAL.solve(net, services, lambda_s, seed=42)
    dep2, _ = ALGO_RANDOM_RANDOM.solve(net, services, lambda_s, seed=42)
    check("同 seed 同 deploy_fn -> 相同部署", np.array_equal(dep1.X, dep2.X))

    # get_algorithm
    algo = get_algorithm("Random-Proportional")
    check("get_algorithm 正确", algo.name == "Random-Proportional")

    try:
        get_algorithm("nonexistent")
        check("get_algorithm 不存在应报错", False)
    except ValueError:
        check("get_algorithm 不存在报错", True)

    # ALL_ALGORITHMS 都能跑
    check("ALL_ALGORITHMS >= 2", len(ALL_ALGORITHMS) >= 2)
    for algo in ALL_ALGORITHMS:
        dep, rt = algo.solve(net, services, lambda_s, seed=42)
        check(f"{algo.name}: solve 成功", dep is not None)

    # 自定义算法
    def dummy_deploy(network, services, lambda_s=None, seed=None):
        d = Deployment(network.n_nodes, len(services))
        for s in services:
            d.X[0][s.id] = 1
        return d, network.cpu_capacity.copy(), network.gpu_capacity.copy()

    custom = AlgorithmSuite("custom", dummy_deploy, proportional_routing)
    dep_c, _ = custom.solve(net, services, lambda_s, seed=42)
    check("自定义算法: 全部部署在 node0", all(dep_c.X[0][s.id] >= 1 for s in services))


# ============================================================
# 9. 实验定义
# ============================================================
def test_experiment_definition():
    print("\n=== 9. 实验定义 ===")

    exp = make_arrival_rate_experiment()
    check("到达率实验: sweep_key=total_rate", exp.sweep_key == "total_rate")
    check("到达率实验: 名称", exp.name == "arrival_rate")
    check("到达率实验: 有值", len(exp.sweep_values) >= 5)
    check("到达率实验: 固定参数含 scale", "scale" in exp.fixed)
    check("到达率实验: 固定参数含 chain_length", "chain_length" in exp.fixed)

    p = exp.build_params(300)
    check("build_params: total_rate=300", p["total_rate"] == 300)
    check("build_params: 保留固定参数", p["scale"] == "medium")

    exp2 = make_chain_length_experiment()
    check("链长实验: sweep_key", exp2.sweep_key == "chain_length")
    check("链长实验: 有值", len(exp2.sweep_values) >= 5)

    exp3 = make_service_type_experiment()
    check("类型实验: sweep_key", exp3.sweep_key == "n_service_types")
    check("类型实验: 有值", len(exp3.sweep_values) >= 5)

    # 自定义实验
    exp_custom = Experiment("test", "total_rate", [100, 200], {"scale": "small"})
    check("自定义: build_params", exp_custom.build_params(100) == {"scale": "small", "total_rate": 100})


# ============================================================
# 10. run_single (解耦版)
# ============================================================
def test_run_single():
    print("\n=== 10. run_single ===")

    algo = ALGO_RANDOM_PROPORTIONAL
    params = {"scale": "small", "total_rate": 100, "chain_length": 3,
              "n_service_types": 10, "n_chains": 4}

    r = run_single(algo, params, seed=42)
    check("返回 algorithm 字段", r["algorithm"] == "Random-Proportional")
    check("返回 scale", r["scale"] == "small")
    check("返回 n_nodes=3", r["n_nodes"] == 3)
    check("返回 total_rate=100", r["total_rate"] == 100)
    check("返回 chain_length=3", r["chain_length"] == 3)
    check("delay 有限正", 0 < r["avg_delay"] < float('inf'))

    # 可重复
    r2 = run_single(algo, params, seed=42)
    check("同 seed 可重复", r["avg_delay"] == r2["avg_delay"])

    # 不同 seed
    r3 = run_single(algo, params, seed=99)
    check("不同 seed 不同", r["avg_delay"] != r3["avg_delay"])

    # 负载越高时延越大
    params_low = {"scale": "medium", "total_rate": 100, "chain_length": 4,
                  "n_service_types": 20, "n_chains": 4}
    params_high = {"scale": "medium", "total_rate": 800, "chain_length": 4,
                   "n_service_types": 20, "n_chains": 4}
    r_low = run_single(algo, params_low, seed=42)
    r_high = run_single(algo, params_high, seed=42)
    check("高负载>低负载", r_high["avg_delay"] > r_low["avg_delay"],
          f"low={r_low['avg_delay']:.4f}, high={r_high['avg_delay']:.4f}")

    # 大网络时延更低
    params_s = {"scale": "small", "total_rate": 200, "chain_length": 4,
                "n_service_types": 20, "n_chains": 4}
    params_l = {"scale": "large", "total_rate": 200, "chain_length": 4,
                "n_service_types": 20, "n_chains": 4}
    r_s = run_single(algo, params_s, seed=42)
    r_l = run_single(algo, params_l, seed=42)
    check("大规模<=小规模", r_l["avg_delay"] <= r_s["avg_delay"],
          f"small={r_s['avg_delay']:.4f}, large={r_l['avg_delay']:.4f}")


# ============================================================
# 11. ExperimentRunner
# ============================================================
def test_experiment_runner():
    print("\n=== 11. ExperimentRunner ===")

    algo = ALGO_RANDOM_PROPORTIONAL
    exp = Experiment("test_sweep", "total_rate", [50, 100, 200],
                     {"scale": "large", "chain_length": 4,
                      "n_service_types": 20, "n_chains": 4})

    runner = ExperimentRunner(exp, algo)
    results = runner.run(seed=42, verbose=False)

    check("结果数量 = 3 参数值 x 1 算法 = 3", len(results) == 3)
    check("每条结果有 algorithm", all(r["algorithm"] == "Random-Proportional" for r in results))
    check("total_rate 值正确", [r["total_rate"] for r in results] == [50, 100, 200])
    check("所有 delay 有限正", all(0 < r["avg_delay"] < float('inf') for r in results))

    # 单调性
    delays = [r["avg_delay"] for r in results]
    check("到达率递增->时延递增", all(delays[i] <= delays[i+1] for i in range(len(delays)-1)),
          f"delays={[f'{d:.4f}' for d in delays]}")

    # CSV 保存
    with tempfile.TemporaryDirectory() as tmpdir:
        path = runner.save_csv(tmpdir)
        check("CSV 文件存在", os.path.exists(path))

        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        check("CSV 行数=3", len(rows) == 3)
        check("CSV 含 algorithm 列", "algorithm" in rows[0])
        check("CSV 含 avg_delay 列", "avg_delay" in rows[0])

    # 多算法
    runner2 = ExperimentRunner(exp, [ALGO_RANDOM_PROPORTIONAL, ALGO_RANDOM_RANDOM])
    results2 = runner2.run(seed=42, verbose=False)
    check("2 算法 x 3 参数 = 6 条", len(results2) == 6)
    algos_in_result = set(r["algorithm"] for r in results2)
    check("结果含两种算法", len(algos_in_result) == 2)

    # summary 不报错
    runner.summary()
    check("summary 不报错", True)

    # 链长实验单调性
    exp_len = Experiment("test_len", "chain_length", [3, 5, 7, 9],
                         {"scale": "large", "total_rate": 100,
                          "n_service_types": 20, "n_chains": 4})
    runner_len = ExperimentRunner(exp_len, algo)
    results_len = runner_len.run(seed=42, verbose=False)
    delays_len = [r["avg_delay"] for r in results_len]
    check("链长递增->时延递增", all(delays_len[i] <= delays_len[i+1]
          for i in range(len(delays_len)-1)),
          f"delays={[f'{d:.4f}' for d in delays_len]}")


# ============================================================
# 12. 动态实验
# ============================================================
def test_dynamic():
    print("\n=== 12. 动态实验 ===")

    from experiment_dynamic import run_dynamic_experiment

    results = run_dynamic_experiment('small', n_steps=10, seed=42)
    check("10 步", len(results) == 10)
    check("step 0..9", results[0]['step'] == 0 and results[9]['step'] == 9)

    for r in results:
        check(f"step{r['step']}: delay finite",
              0 < r['avg_delay'] < float('inf') or r['stable_chains'] < r['total_chains'])
        check(f"step{r['step']}: load in [0.5,1.5]", 0.5 <= r['load_factor'] <= 1.5)
        check(f"step{r['step']}: cpu_f in [0.7,1.3]", 0.7 <= r['cpu_factor'] <= 1.3)

    low_load = [r for r in results if r['load_factor'] < 0.8]
    high_load = [r for r in results if r['load_factor'] > 1.2]
    if low_load and high_load:
        avg_low = np.mean([r['stable_chains'] for r in low_load])
        avg_high = np.mean([r['stable_chains'] for r in high_load])
        check("低负载稳定性>=高负载", avg_low >= avg_high,
              f"low={avg_low:.2f}, high={avg_high:.2f}")


# ============================================================
# 运行全部
# ============================================================
if __name__ == '__main__':
    test_queuing()
    test_network()
    test_services()
    test_deployment()
    test_routing()
    test_chain_delay()
    test_evaluation()
    test_algorithm()
    test_experiment_definition()
    test_run_single()
    test_experiment_runner()
    test_dynamic()

    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  ALL TESTS PASSED")
    print(f"{'=' * 60}")
