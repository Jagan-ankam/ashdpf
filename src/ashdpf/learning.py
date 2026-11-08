"""Continuous Learning Engine (CLE) -- paper Sec. III.G, patent [0022], claims 2 & 10.

State  S_t : incident context (here: diagnosed root-cause category)
Action A_t : selected remediation
Reward R_t = kappa / MTTR                       (Eq. 5, kappa = reward_scale)
Policy pi_{t+1} = pi_t + eta * grad J(pi)       (Eq. 6)

The policy is a softmax over per-state action preferences theta. The gradient is
the REINFORCE estimator with a running-average baseline:
    theta_b += eta * (R_t - baseline_s) * (1[b == a] - pi(b | s))
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from .config import ASHDPFConfig
from .knowledge_graph import AdaptiveKnowledgeGraph
from .models import FailureCategory, IncidentRecord, Outcome


def reward(outcome: Outcome, mttr_minutes: float, scale: float = 1.0) -> float:
    """Eq. (5): faster resolutions yield higher rewards; failures earn nothing."""
    if outcome == Outcome.FAILURE:
        return 0.0
    r = scale / max(mttr_minutes, 1e-6)
    return r if outcome == Outcome.SUCCESS else 0.5 * r


class SoftmaxPolicy:
    def __init__(self, eta: float = 0.5) -> None:
        self.eta = eta
        self.theta: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.baseline: Dict[str, float] = defaultdict(float)
        self.visits: Dict[str, int] = defaultdict(int)

    @staticmethod
    def _key(state) -> str:
        return state.value if isinstance(state, FailureCategory) else str(state)

    def probabilities(self, state, actions: Sequence[str]) -> Dict[str, float]:
        if not actions:
            return {}
        th = self.theta[self._key(state)]
        prefs = [th.get(a, 0.0) for a in actions]
        m = max(prefs)
        exps = [math.exp(p - m) for p in prefs]
        z = sum(exps)
        return {a: e / z for a, e in zip(actions, exps)}

    def update(self, state, action: str, actions: Sequence[str], r: float) -> Dict[str, float]:
        """One policy-gradient ascent step (Eq. 6). Returns the gradient applied."""
        key = self._key(state)
        if action not in actions:
            actions = list(actions) + [action]
        probs = self.probabilities(state, actions)
        self.visits[key] += 1
        adv = r - self.baseline[key]
        grad = {}
        for b in actions:
            g = adv * ((1.0 if b == action else 0.0) - probs[b])
            self.theta[key][b] = self.theta[key].get(b, 0.0) + self.eta * g
            grad[b] = g
        # running-average baseline reduces variance
        self.baseline[key] += (r - self.baseline[key]) / self.visits[key]
        return grad


class ContinuousLearningEngine:
    def __init__(self, kg: AdaptiveKnowledgeGraph, cfg: ASHDPFConfig, rca=None,
                 planner=None, policy: Optional[SoftmaxPolicy] = None) -> None:
        self.kg = kg
        self.cfg = cfg
        self.rca = rca
        self.planner = planner
        self.policy = policy or SoftmaxPolicy(cfg.eta)
        self.history: List[IncidentRecord] = []
        self.retrain_count = 0

    def learn(self, record: IncidentRecord, engineer: Optional[str] = None) -> None:
        self.history.append(record)
        cat = record.confirmed_category or (record.hypothesis.category if record.hypothesis else None)

        # 1. Reinforcement-learning policy update.
        if self.cfg.use_learning and record.action is not None and cat is not None:
            actions = ([a.name for a in self.planner.library.for_category(cat)]
                       if self.planner else [record.action.name])
            r = reward(record.outcome, record.resolution_minutes, self.cfg.reward_scale)
            self.policy.update(cat, record.action.name, actions, r)

        # 2. Knowledge-graph evolution (Eq. 1) -- not for a static graph.
        if self.cfg.kg_mode != "static" and record.action is not None and cat is not None:
            score = self.cfg.outcome_scores[record.outcome.value]
            self.kg.record_fix_outcome(cat, record.action.name, record.outcome, score)
        if self.cfg.kg_mode == "adaptive":
            self.kg.ingest_incident(record, engineer=engineer)

        # 3. Remediation repository + confidence-scoring model.
        if self.rca is not None and cat is not None:
            self.rca.history.add(record.event, cat,
                                 record.action.name if record.action else None, record.outcome)
            if self.cfg.use_learning and self.rca.add_labelled(record.event, cat):
                self.retrain_count += 1
