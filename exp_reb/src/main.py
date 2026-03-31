"""Main entry point for running deployment experiments."""

import json
import os
from typing import Dict, Any

from config.default_config import DEFAULT_CONFIG
from experiments.generator import DataGenerator
from experiments.runner import ExperimentRunner
from experiments.validator import ResultValidator


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration.

    Args:
        config_path: Optional path to custom config file

    Returns:
        Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        # Load from custom config file
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
        # Merge with default config
        config = DEFAULT_CONFIG.copy()
        config.update(custom_config)
        return config
    return DEFAULT_CONFIG.copy()


def save_results(results: list, output_dir: str, prefix: str = "experiment"):
    """
    Save experiment results to files.

    Args:
        results: List of experiment results
        output_dir: Output directory path
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    json_path = os.path.join(output_dir, f"{prefix}_results.json")
    # Convert non-serializable objects to strings
    serializable_results = []
    for r in results:
        serializable = {
            "algorithm": r["algorithm"],
            "avg_latency": r["avg_latency"],
            "success_rate": r["success_rate"],
            "deployment_cost": r["deployment_cost"],
            "resource_utilization": r["resource_utilization"],
            "chain_latencies": r["chain_latencies"],
        }
        if "param" in r:
            serializable["param"] = r["param"]
            serializable["value"] = r["value"]
        serializable_results.append(serializable)

    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to {json_path}")


def print_summary(results: list):
    """
    Print a summary table of results.

    Args:
        results: List of experiment results
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<15} {'Avg Latency (ms)':<20} {'Success Rate':<15} {'Nodes Used':<10}")
    print("-" * 80)

    for r in results:
        alg = r.get("algorithm", "unknown")
        latency = r.get("avg_latency", 0)
        success = r.get("success_rate", 0)
        cost = r.get("deployment_cost", 0)

        print(f"{alg:<15} {latency:<20.2f} {success:<15.2%} {cost:<10}")

    print("=" * 80)


def run_basic_experiment():
    """Run a basic single experiment with default config."""
    print("Running basic experiment...")

    # Load config
    config = DEFAULT_CONFIG.copy()

    # Generate data
    generator = DataGenerator(seed=config.get("seed", 42))
    topology, services, chains = generator.generate_all(config)

    print(f"Generated topology: {len(topology.nodes)} nodes, {len(topology.links)} links")
    print(f"Generated {len(services)} services")
    print(f"Generated {len(chains)} service chains")

    # Run experiment
    runner = ExperimentRunner(config)

    # Get algorithm names from config
    algorithm_names = [alg["name"] for alg in config.get("algorithms", [])]
    if not algorithm_names:
        algorithm_names = ["ffd-m", "cds-m", "random-m", "greedy-m"]

    results = runner.run_comparison(topology, services, chains, algorithm_names)

    # Print summary
    print_summary(results)

    # Validate results
    validator = ResultValidator()
    print("\nValidation Results:")
    for r in results:
        plan = r["deployment_plan"]
        valid, errors = validator.validate_deployment(plan, topology, services, chains)
        if valid:
            print(f"  {r['algorithm']}: VALID")
        else:
            print(f"  {r['algorithm']}: INVALID")
            for err in errors:
                print(f"    - {err}")

    return results


def run_perturbation_experiment(param_name: str, param_values: list):
    """
    Run perturbation experiment varying a single parameter.

    Args:
        param_name: Name of parameter to vary
        param_values: List of values to test
    """
    print(f"Running perturbation experiment for {param_name}...")

    # Load config
    config = DEFAULT_CONFIG.copy()

    # Generate base data
    generator = DataGenerator(seed=config.get("seed", 42))

    # Get algorithm names
    algorithm_names = [alg["name"] for alg in config.get("algorithms", [])]
    if not algorithm_names:
        algorithm_names = ["ffd-m", "cds-m", "random-m", "greedy-m"]

    # Run perturbation
    runner = ExperimentRunner(config)
    results = runner.run_perturbation(
        base_config=config,
        param_name=param_name,
        param_values=param_values,
        algorithms=algorithm_names,
        generator=generator
    )

    # Print summary
    print_summary(results)

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run deployment experiments")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to custom config file")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--perturb", type=str, nargs=2, metavar=("PARAM", "VALUES"),
                       help="Run perturbation experiment. Provide param name and comma-separated values")
    parser.add_argument("--algorithm", type=str, nargs="+",
                       help="Specific algorithms to run")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override algorithms if specified
    if args.algorithm:
        config["algorithms"] = [{"name": alg} for alg in args.algorithm]

    # Run experiment
    if args.perturb:
        param_name = args.perturb[0]
        param_values = [float(v) for v in args.perturb[1].split(",")]
        results = run_perturbation_experiment(param_name, param_values)
    else:
        results = run_basic_experiment()

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
