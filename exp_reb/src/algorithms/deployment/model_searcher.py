"""ModelSearcher class for Our algorithm - searches candidate models from Excel model library."""

import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """
    Model information from Excel model library.

    Excel columns:
    - architecture (model_id): Model architecture identifier
    - proxy_score: S_proxy - accuracy potential proxy score (higher is better)
    - model_params: P_model - parameter count (memory footprint)
    - flops: F_flops - computational complexity (lower is faster)
    """
    model_id: str
    proxy_score: float
    model_params: float
    flops: float

    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelInfo':
        """Create ModelInfo from dictionary (row from Excel)."""
        return cls(
            model_id=d.get('architecture', d.get('model_id', '')),
            proxy_score=float(d.get('proxy_score', 0.0)),
            model_params=float(d.get('model_params', 0.0)),
            flops=float(d.get('flops', 0.0))
        )


class ModelSearcher:
    """
    Model searcher for Our algorithm.

    Based on globecom.pdf, this class:
    1. Loads models from Excel model library by task type
    2. Calculates dynamic compute红线 (F_max) based on arrival rate
    3. Calculates dynamic weights (w2, w3) for utility function
    4. Filters models by hard constraints (memory + compute)
    5. Ranks models by utility function and returns best candidates

    Key formulas from paper:
    - F_max(t) = C_max / (λ(t) + 1/T_SLA)  # Dynamic compute红线
    - U(a_k) = w1·S_proxy - w2·F_flops - w3·P_model  # Utility function
    - w2(t) = α·exp(max(0, λ-λ_th)/λ_th)  # Traffic penalty weight
    - w3(t) = β·(M_used/M_total)  # Memory penalty weight
    - w1(t) = 1/(1+w2+w3)  # Normalized accuracy weight
    """

    def __init__(self, excel_path: str,
                 lambda_th: float = 20.0,
                 alpha: float = 1.0,
                 beta: float = 1.0):
        """
        Initialize ModelSearcher.

        Args:
            excel_path: Path to Excel model library file.
            lambda_th: Traffic threshold for w2 calculation (default: 20.0).
            alpha: Scaling factor for w2 traffic penalty (default: 1.0).
            beta: Scaling factor for w3 memory penalty (default: 1.0).
        """
        self.excel_path = excel_path
        self.lambda_th = lambda_th
        self.alpha = alpha
        self.beta = beta
        self.model_tables: Dict[str, List[Dict]] = {}  # {task_type: list of model dicts}

    def load_models(self, task_type: str) -> None:
        """
        Load models for a task type from Excel.

        Args:
            task_type: Task type identifier (e.g., 'class_scene', 'class_object').
        """
        import pandas as pd

        try:
            df = pd.read_excel(self.excel_path, sheet_name=task_type)
            # Convert DataFrame to list of dicts for easier processing
            self.model_tables[task_type] = df.to_dict('records')
        except Exception as e:
            print(f"Warning: Failed to load models for task type '{task_type}': {e}")
            self.model_tables[task_type] = []

    def calc_F_max(self, C_max: float, lambda_t: float, T_SLA: float) -> float:
        """
        Calculate dynamic compute红线 (F_max).

        F_max(t) = C_max / (λ(t) + 1/T_SLA)

        As traffic λ increases, F_max decreases (can only use lighter models).
        As SLA constraint T_SLA decreases (stricter), F_max also decreases.

        Args:
            C_max: Node peak compute capability (flops).
            lambda_t: Current arrival rate (requests/second).
            T_SLA: SLA latency constraint (milliseconds).

        Returns:
            Maximum compute capacity allowed for selected model.
        """
        if T_SLA <= 0:
            return C_max
        # Convert T_SLA from ms to seconds for formula consistency
        # Formula: F_max = C_max / (λ + 1/T_SLA)
        return C_max / (lambda_t + 1.0 / T_SLA)

    def calc_w2(self, lambda_t: float) -> float:
        """
        Calculate traffic penalty weight w2.

        w2(t) = α·exp(max(0, λ-λ_th)/λ_th)

        When traffic exceeds threshold λ_th, w2 increases exponentially,
        penalizing high-compute models.

        Args:
            lambda_t: Current arrival rate.

        Returns:
            Traffic penalty weight.
        """
        return self.alpha * math.exp(max(0, lambda_t - self.lambda_th) / self.lambda_th)

    def calc_w3(self, M_used: float, M_total: float) -> float:
        """
        Calculate memory penalty weight w3.

        w3(t) = β·(M_used/M_total)

        As memory utilization increases, w3 increases, penalizing
        large parameter count models.

        Args:
            M_used: Currently used memory.
            M_total: Total available memory.

        Returns:
            Memory penalty weight.
        """
        if M_total <= 0:
            return 0.0
        return self.beta * (M_used / M_total)

    def search(self, task_type: str,
               M_max: float,
               C_max: float,
               lambda_t: float,
               T_SLA: float,
               M_used: float = 0.0,
               M_total: float = 1.0) -> List[Dict]:
        """
        Search candidate models based on constraints and utility.

        Args:
            task_type: Task type (e.g., 'class_scene').
            M_max: Maximum available memory for deployment.
            C_max: Node peak compute capability.
            lambda_t: Current arrival rate.
            T_SLA: SLA latency constraint (ms).
            M_used: Currently used memory on node.
            M_total: Total memory on node.

        Returns:
            List of candidate model dicts, sorted by utility (descending).
            Each dict contains: architecture, proxy_score, model_params, flops, utility.
        """
        # Step 1: Calculate dynamic compute红线
        F_max = self.calc_F_max(C_max, lambda_t, T_SLA)

        # Step 2: Load models from Excel if not already loaded
        if task_type not in self.model_tables:
            self.load_models(task_type)

        candidates = self.model_tables.get(task_type, [])
        if not candidates:
            return []

        # Step 3: Hard constraint filtering (memory + compute)
        # Excel columns: architecture=model_id, proxy_score=S_proxy,
        #                model_params=P_model, flops=F_flops
        filtered = [
            m for m in candidates
            if m.get('model_params', float('inf')) <= M_max
            and m.get('flops', float('inf')) <= F_max
        ]

        if not filtered:
            return []

        # Step 4: Calculate dynamic weights
        w2 = self.calc_w2(lambda_t)
        w3 = self.calc_w3(M_used, M_total)
        w1 = 1.0 / (1.0 + w2 + w3)  # Normalized accuracy weight

        # Step 5: Calculate utility for each model and sort
        # U(a_k) = w1·S_proxy - w2·F_flops - w3·P_model
        def calc_utility(model: Dict) -> float:
            S_proxy = model.get('proxy_score', 0.0)
            F_flops = model.get('flops', 0.0)
            P_model = model.get('model_params', 0.0)
            return w1 * S_proxy - w2 * F_flops - w3 * P_model

        for model in filtered:
            model['utility'] = calc_utility(model)

        # Sort by utility (descending)
        filtered.sort(key=lambda m: m['utility'], reverse=True)

        return filtered

    def get_best_model(self, task_type: str,
                       M_max: float,
                       C_max: float,
                       lambda_t: float,
                       T_SLA: float,
                       M_used: float = 0.0,
                       M_total: float = 1.0) -> Optional[Dict]:
        """
        Get the best model for given constraints.

        Args:
            task_type: Task type.
            M_max: Maximum available memory.
            C_max: Node peak compute.
            lambda_t: Current arrival rate.
            T_SLA: SLA latency constraint.
            M_used: Currently used memory.
            M_total: Total memory.

        Returns:
            Best model dict or None if no feasible model.
        """
        candidates = self.search(task_type, M_max, C_max, lambda_t, T_SLA, M_used, M_total)
        return candidates[0] if candidates else None
