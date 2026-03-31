"""Our algorithm - dynamic model selection and load-aware routing."""

import math
from typing import Dict, List, Any, Optional, Tuple

from algorithms.deployment.base import DeploymentAlgorithm
from algorithms.deployment.model_searcher import ModelSearcher
from algorithms.routing.load_aware import LoadAwareRouting
from core.service.deployment import DeploymentPlan


class OurDeployment:
    """
    Our deployment algorithm - based on globecom.pdf启发式调度.

    Three-stage pipeline:
    1. Hard constraint filtering (memory红线 + compute红线)
    2. Dynamic weighted selection (utility function maximization)
    3. Output deployment plan

    The algorithm dynamically selects model versions based on:
    - Current traffic load (arrival rate)
    - Node memory availability
    - Node compute capability
    - SLA latency constraints
    """

    def __init__(self, model_searcher: ModelSearcher):
        """
        Initialize OurDeployment.

        Args:
            model_searcher: ModelSearcher instance for Excel model lookup.
        """
        self.model_searcher = model_searcher

    def deploy(self, topology: Any,
               services: Dict[str, Any],
               chains: List[Any]) -> DeploymentPlan:
        """
        Execute Our deployment algorithm.

        Args:
            topology: Network topology with nodes dict.
            services: Service configurations {service_id: Service/MicroService}.
            chains: Service chain list with task_type, arrival_rate, max_latency.

        Returns:
            DeploymentPlan: Deployment plan with model selections.
        """
        deployment_plan = DeploymentPlan()

        # Pre-load all task type model libraries
        task_types = set()
        for chain in chains:
            if hasattr(chain, 'task_type') and chain.task_type:
                task_types.add(chain.task_type)
            elif hasattr(chain, 'get') and chain.get('task_type'):
                task_types.add(chain.get('task_type'))

        for task_type in task_types:
            self.model_searcher.load_models(task_type)

        # Get topology nodes
        nodes = topology.get('nodes', {}) if isinstance(topology, dict) else getattr(topology, 'nodes', {})

        # Calculate total arrival rate per service
        service_rates = {}
        for chain in chains:
            for service_id in chain.services:
                service_rates[service_id] = service_rates.get(service_id, 0) + chain.arrival_rate

        # Collect unique task types for each service (use the first chain's task_type)
        service_task_types = {}
        for chain in chains:
            for service_id in chain.services:
                if service_id not in service_task_types:
                    service_task_types[service_id] = getattr(chain, 'task_type', 'default') if isinstance(chain, dict) else getattr(chain, 'task_type', 'default')

        # Deploy each service ONCE (not per chain) to appropriate nodes
        deployed_services = set()
        for service_id, total_rate in service_rates.items():
            if service_id in deployed_services:
                continue
            deployed_services.add(service_id)

            task_type = service_task_types.get(service_id, 'default')

            # Find best node for this service
            best_node = self._find_best_node(
                service_id, task_type, total_rate, 500.0,  # Use max_latency from config
                nodes, services, deployment_plan
            )

            if best_node:
                node_id = best_node['node_id']
                best_model = best_node['model']

                # Calculate instances needed based on arrival rate
                mu_baseline = 10.0

                if mu_baseline > 0:
                    base_instances = int(total_rate / mu_baseline) + 1
                    margin = 2 if total_rate > 20 else 1
                    instances = max(1, base_instances + margin)
                else:
                    instances = 1

                # Add deployment with selected model version
                model_version = best_model.get('architecture', 'Model-M')

                # Calculate mu from flops for queueing analysis
                # Lower flops = higher mu (faster model)
                flops = best_model.get('flops', 1e9)
                if flops < 1e8:  # < 100M FLOPs -> Model-L (fast)
                    mu = 20.0
                    mapped_version = 'Model-L'
                elif flops < 5e8:  # < 500M FLOPs -> Model-M (medium)
                    mu = 10.0
                    mapped_version = 'Model-M'
                else:  # >= 500M FLOPs -> Model-H (slow)
                    mu = 5.0
                    mapped_version = 'Model-H'

                deployment_plan.add(
                    service_id=service_id,
                    node_id=node_id,
                    version_id=mapped_version,  # Use mapped version for lookup
                    count=instances,
                    mu=mu  # Store actual mu for queueing calculation
                )

                # Update node resource usage (critical for correct deployment)
                node = nodes[node_id] if isinstance(nodes, dict) else nodes.get(node_id)
                if node:
                    gpu_per_inst = best_model.get('gpu_memory', 512)
                    cpu_per_inst = best_model.get('cpu_cores', 1)
                    if isinstance(node, dict):
                        node['used_gpu'] = node.get('used_gpu', 0) + gpu_per_inst * instances
                        node['used_cpu'] = node.get('used_cpu', 0) + cpu_per_inst * instances
                    else:
                        node.used_gpu = getattr(node, 'used_gpu', 0) + gpu_per_inst * instances
                        node.used_cpu = getattr(node, 'used_cpu', 0) + cpu_per_inst * instances

        return deployment_plan

    def _find_best_node(self, service_id: str,
                        task_type: str,
                        arrival_rate: float,
                        max_latency: float,
                        nodes: Dict[str, Any],
                        services: Dict[str, Any],
                        deployment_plan: DeploymentPlan) -> Optional[Dict]:
        """
        Find the best node and model for deploying a service.

        Args:
            service_id: Service to deploy.
            task_type: Task type for model search.
            arrival_rate: Request arrival rate.
            max_latency: SLA latency constraint.
            nodes: Topology nodes dict.
            services: Service configurations.
            deployment_plan: Current deployment plan.

        Returns:
            Dict with 'node_id' and 'model' keys, or None if no feasible deployment.
        """
        best_candidate = None
        best_utility = float('-inf')

        for node_id, node in nodes.items():
            # Get node resources
            if isinstance(node, dict):
                M_max = node.get('gpu_memory', node.get('available_memory', node.get('memory', 0)))
                C_max = node.get('cpu_cores', node.get('compute_capability', node.get('C_max', 0)))
                M_total = node.get('gpu_memory', node.get('total_memory', M_max))
                M_used = node.get('used_gpu', node.get('used_memory', 0))
            else:
                # Object with attributes
                M_max = getattr(node, 'available_gpu', getattr(node, 'available_memory', 0))
                if M_max == 0:
                    M_max = getattr(node, 'gpu_memory', getattr(node, 'memory', 0))
                C_max = getattr(node, 'cpu_cores', getattr(node, 'compute_capability', 0))
                M_total = getattr(node, 'gpu_memory', getattr(node, 'total_memory', M_max))
                if M_total == 0:
                    M_total = M_max
                M_used = getattr(node, 'used_gpu', getattr(node, 'used_memory', 0))

            if M_max <= 0:
                M_max = 16384  # Default 16GB GPU memory
            if C_max <= 0:
                C_max = 100  # Default compute units

            # Search for best model at this node
            model = self.model_searcher.get_best_model(
                task_type=task_type,
                M_max=M_max,
                C_max=C_max,
                lambda_t=arrival_rate,
                T_SLA=max_latency,
                M_used=M_used,
                M_total=M_total
            )

            if model and model.get('utility', float('-inf')) > best_utility:
                best_utility = model['utility']
                best_candidate = {
                    'node_id': node_id,
                    'model': model
                }

        return best_candidate


class OurRouting:
    """
    Our routing algorithm - load-aware routing.

    Strategy:
    - Prefers nodes with lower current load
    - Considers path delay between services
    - Uses LoadAwareRouting for dynamic load balancing
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize OurRouting.

        Args:
            config: Optional configuration for LoadAwareRouting.
        """
        self.config = config or {}
        self.load_aware = LoadAwareRouting(self.config)

    def route(self, chain: Any,
              deployment_plan: DeploymentPlan,
              topology: Any) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute routing for a service chain.

        Args:
            chain: ServiceChain with services, arrival_rate, entry_node.
            deployment_plan: DeploymentPlan with placements.
            topology: Topology with nodes and links.

        Returns:
            Dict mapping service_id to list of (node_id, probability).
        """
        return self.load_aware.route(chain, deployment_plan, topology)


class OurAlgorithm(DeploymentAlgorithm):
    """
    Our complete algorithm (deployment + routing) based on globecom.pdf.

    This algorithm dynamically selects model versions based on:
    - Traffic conditions (arrival rate λ)
    - Node resource availability (memory, compute)
    - SLA constraints (max latency)

    Key innovations over baseline:
    - Model search from Excel model library
    - Dynamic compute红线 (F_max) calculation
    - Utility function with dynamic weights (w1, w2, w3)
    - Load-aware routing for request distribution

    Usage:
        algorithm = OurAlgorithm(config={'excel_model_path': 'models.xlsx'})
        result = algorithm.solve(topology, services, chains)
        deployment_plan = result['deployment_plan']
        routing_plan = result['routing_plan']
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OurAlgorithm.

        Args:
            config: Configuration dict with:
                - excel_model_path: Path to Excel model library (required)
                - lambda_th: Traffic threshold for w2 (default: 20.0)
                - alpha: Scaling factor for w2 (default: 1.0)
                - beta: Scaling factor for w3 (default: 1.0)
                - routing_config: Optional config for routing algorithm
        """
        super().__init__(config)
        self.fixed_version = None  # Our algorithm doesn't use fixed version

        # Initialize model searcher
        excel_path = self.config.get('excel_model_path')
        if excel_path:
            self.model_searcher = ModelSearcher(
                excel_path=excel_path,
                lambda_th=self.config.get('lambda_th', 20.0),
                alpha=self.config.get('alpha', 1.0),
                beta=self.config.get('beta', 1.0)
            )
        else:
            self.model_searcher = None

        # Initialize deployment and routing
        self.deployment = None
        self.routing = OurRouting(self.config.get('routing_config', {}))

    def deploy(self, topology: Any,
               services: Dict[str, Any],
               chains: List[Any]) -> DeploymentPlan:
        """
        Execute Our deployment algorithm.

        Args:
            topology: Network topology.
            services: Service configurations.
            chains: Service chain list.

        Returns:
            DeploymentPlan: Deployment plan with model selections.
        """
        if not self.model_searcher:
            raise ValueError(
                "ModelSearcher not initialized. "
                "Set 'excel_model_path' in config."
            )

        # Initialize deployment
        if self.deployment is None:
            self.deployment = OurDeployment(self.model_searcher)

        # Execute deployment
        deployment_plan = self.deployment.deploy(topology, services, chains)
        return deployment_plan

    def route(self, chain: Any,
              deployment_plan: DeploymentPlan,
              topology: Any) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute routing for a service chain.

        Args:
            chain: ServiceChain object.
            deployment_plan: DeploymentPlan with placements.
            topology: Topology object.

        Returns:
            Dict mapping service_id to list of (node_id, probability).
        """
        return self.routing.route(chain, deployment_plan, topology)

    def solve(self, topology: Any,
              services: Dict[str, Any],
              chains: List[Any]) -> Dict[str, Any]:
        """
        Complete solve: deployment + routing.

        Args:
            topology: Network topology.
            services: Service configurations.
            chains: Service chain list.

        Returns:
            Dict with:
                - deployment_plan: DeploymentPlan object
                - routing_plan: Dict mapping chain_id to routing dict
        """
        # Step 1: Deployment
        deployment_plan = self.deploy(topology, services, chains)

        # Step 2: Routing for each chain
        routing_plan = {}
        for chain in chains:
            chain_id = getattr(chain, 'chain_id', chain.get('chain_id', '')) if isinstance(chain, dict) else chain.chain_id
            routing_plan[chain_id] = self.route(chain, deployment_plan, topology)

        return {
            'deployment_plan': deployment_plan,
            'routing_plan': routing_plan
        }
