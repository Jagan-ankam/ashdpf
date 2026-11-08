"""Safety Validation Layer -- paper Sec. III.E.2, patent [0020], claim 6.

Checks performed before autonomous execution:
  1. dependency-impact assessment (blast radius of downstream consumers)
  2. rollback-feasibility validation
  3. governance-policy check (forbidden actions, max autonomous risk)
  4. execution-threshold verification: confidence > T (category-specific)

Confidence for gating = C_RCA * W_e, i.e. the probability the diagnosis is right
times the adaptive AOKG confidence that this fix resolves that root cause.
"""

from __future__ import annotations

from typing import List

from .config import ASHDPFConfig
from .knowledge_graph import AdaptiveKnowledgeGraph
from .models import Decision, FailureEvent, RootCauseHypothesis, SafetyVerdict, ScoredAction


class SafetyValidationLayer:
    def __init__(self, kg: AdaptiveKnowledgeGraph, cfg: ASHDPFConfig) -> None:
        self.kg = kg
        self.cfg = cfg

    def action_confidence(self, hyp: RootCauseHypothesis, cand: ScoredAction) -> float:
        if cand.category is not None:
            p_cat = hyp.distribution.get(cand.category, 0.0)
        else:
            p_cat = max(hyp.distribution.get(c, 0.0) for c in cand.action.applicable_to)
        return p_cat * cand.edge_weight

    def validate(self, cand: ScoredAction, hyp: RootCauseHypothesis,
                 event: FailureEvent) -> SafetyVerdict:
        a = cand.action
        reasons: List[str] = []
        conf = self.action_confidence(hyp, cand)
        T = self.cfg.threshold_for(hyp.category)

        blast = len(self.kg.downstream(event.job)) if self.cfg.kg_mode != "none" else 0
        dependency_ok = blast <= self.cfg.max_blast_radius or a.risk <= 0.1
        if not dependency_ok:
            reasons.append(f"blast radius {blast} > {self.cfg.max_blast_radius}")

        rollback_ok = a.reversible or a.risk <= self.cfg.irreversible_max_risk
        if not rollback_ok:
            reasons.append("action is irreversible and above irreversible_max_risk")

        forbidden = a.name in self.cfg.forbidden_actions
        governance_ok = not forbidden and a.risk <= self.cfg.max_auto_risk
        if forbidden:
            reasons.append("action forbidden by governance policy")
        elif not governance_ok:
            reasons.append(f"risk {a.risk:.2f} exceeds max_auto_risk {self.cfg.max_auto_risk}")

        threshold_ok = conf > T
        if not threshold_ok:
            reasons.append(f"confidence {conf:.3f} <= threshold {T:.2f}")

        checks = {"dependency_impact": dependency_ok, "rollback": rollback_ok,
                  "governance": governance_ok, "threshold": threshold_ok}
        if forbidden:
            decision = Decision.REJECTED
        elif all(checks.values()):
            decision = Decision.AUTO_EXECUTE
        else:
            decision = Decision.HUMAN_APPROVAL
        return SafetyVerdict(decision, conf, T, checks, reasons)
