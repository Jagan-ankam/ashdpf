"""Configuration for every tunable coefficient in the paper / patent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple

from .models import FailureCategory


@dataclass
class ASHDPFConfig:
    # --- Adaptive Operational Knowledge Graph, Eq. (1): W_e <- W_e + alpha * R
    alpha: float = 0.1
    initial_edge_weight: float = 0.5
    edge_weight_bounds: Tuple[float, float] = (0.0, 1.0)
    graph_max_depth: int = 3
    dependency_window_minutes: float = 180.0
    kg_mode: str = "adaptive"  # "adaptive" | "static" | "none"  (Experiment 3)

    # --- RCA engine, Eq. (3): C_RCA = b1*G + b2*H + b3*ML
    beta: Tuple[float, float, float] = (0.35, 0.25, 0.40)
    use_graph: bool = True
    use_history: bool = True
    use_ml: bool = True
    use_rca: bool = True  # False -> keyword rules (ablation "without RCA engine")
    history_top_k: int = 10
    ml_estimators: int = 60
    retrain_every: int = 200
    min_train_samples: int = 40
    ml_max_train: int = 4000  # sliding window of most recent labelled incidents

    # --- Remediation planner, Eq. (4): RES = g1*S + g2*H + g3*R
    # g3 is negative so that riskier actions rank lower.
    gamma: Tuple[float, float, float] = (0.30, 0.60, -0.30)
    policy_weight: float = 0.25
    use_planner: bool = True  # False -> static first-match playbook

    # --- Safety validation layer
    threshold: float = 0.40  # organisational threshold T
    category_thresholds: Dict[FailureCategory, float] = field(
        default_factory=lambda: {FailureCategory.CREDENTIAL: 0.90}
    )
    max_blast_radius: int = 40
    max_auto_risk: float = 0.55
    irreversible_max_risk: float = 0.30
    forbidden_actions: FrozenSet[str] = frozenset({"drop_and_recreate_table"})

    # --- Continuous Learning Engine, Eq. (5), (6)
    eta: float = 0.5           # learning rate of the policy gradient update
    reward_scale: float = 10.0  # R_t = reward_scale / MTTR
    use_learning: bool = True
    outcome_scores: Dict[str, float] = field(
        default_factory=lambda: {"success": 1.0, "partial": 0.3, "failure": -1.0}
    )

    # --- Execution / reporting
    max_attempts: int = 3
    kri_similarity_threshold: float = 0.5
    seed: int = 7

    def validate(self) -> "ASHDPFConfig":
        if len(self.beta) != 3 or any(b < 0 for b in self.beta):
            raise ValueError("beta must be three non-negative coefficients")
        if abs(sum(self.beta) - 1.0) > 1e-6:
            raise ValueError("beta coefficients must sum to 1 so C_RCA is a probability")
        if len(self.gamma) != 3:
            raise ValueError("gamma must have three coefficients")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold T must be within [0, 1]")
        lo, hi = self.edge_weight_bounds
        if not lo < hi:
            raise ValueError("edge_weight_bounds must be (low, high) with low < high")
        if self.kg_mode not in {"adaptive", "static", "none"}:
            raise ValueError("kg_mode must be adaptive, static or none")
        if self.alpha < 0 or self.eta < 0:
            raise ValueError("alpha and eta must be non-negative")
        return self

    def threshold_for(self, category: FailureCategory) -> float:
        return self.category_thresholds.get(category, self.threshold)
