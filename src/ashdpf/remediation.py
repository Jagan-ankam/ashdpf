"""Intelligent Remediation Planner (IRP) -- paper Sec. III.E, patent [0020].

RES(A_i) = gamma1 * S + gamma2 * H + gamma3 * R      (Eq. 4)

S : similarity of the current incident to past incidents fixed by A_i
H : historical success rate of A_i for the diagnosed root cause (from the AOKG)
R : risk score of A_i (gamma3 < 0 so risk lowers the score)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .config import ASHDPFConfig
from .knowledge_graph import AdaptiveKnowledgeGraph
from .models import (FailureCategory as FC, FailureEvent, Outcome, RemediationAction,
                     RootCauseHypothesis, ScoredAction)


def _a(name, cats, risk, reversible=True, connector="job_restart", desc=""):
    return RemediationAction(name, frozenset(cats), risk, reversible, connector, desc)


DEFAULT_ACTIONS: List[RemediationAction] = [
    _a("restart_workflow", [FC.INFRASTRUCTURE, FC.NETWORK_TIMEOUT, FC.RESOURCE_EXHAUSTION],
       0.10, connector="workflow_trigger", desc="Clear and re-trigger the failed workflow run"),
    _a("rerun_failed_partition", [FC.INFRASTRUCTURE, FC.DATA_QUALITY, FC.JOB_DEPENDENCY],
       0.15, connector="job_restart", desc="Re-execute only the failed partition / task"),
    _a("restore_dependency_service", [FC.JOB_DEPENDENCY, FC.INFRASTRUCTURE], 0.20,
       connector="cloud_admin", desc="Restart or fail over an upstream dependency"),
    _a("wait_and_retry_upstream", [FC.JOB_DEPENDENCY, FC.NETWORK_TIMEOUT], 0.05,
       connector="workflow_trigger", desc="Block until upstream succeeds then retry"),
    _a("increase_timeout_and_retry", [FC.NETWORK_TIMEOUT], 0.10, connector="job_restart"),
    _a("scale_up_resources", [FC.RESOURCE_EXHAUSTION], 0.25, connector="cloud_admin",
       desc="Increase executor memory / cluster size and retry"),
    _a("reset_configuration", [FC.CONFIGURATION], 0.30, connector="cloud_admin",
       desc="Roll configuration back to last known good version"),
    _a("quarantine_bad_records", [FC.DATA_QUALITY], 0.30, connector="sql",
       desc="Move rows failing quality rules to a quarantine table and continue"),
    _a("reprocess_missing_data", [FC.DATA_QUALITY], 0.35, connector="sql",
       desc="Backfill the missing / corrupted window from source"),
    _a("evolve_target_schema", [FC.SCHEMA_EVOLUTION], 0.45, connector="sql",
       desc="Apply additive schema migration to the target table"),
    _a("refresh_credentials", [FC.CREDENTIAL], 0.40, connector="cloud_admin",
       desc="Refresh token / rotate secret from vault"),
    _a("drop_and_recreate_table", [FC.SCHEMA_EVOLUTION], 0.90, reversible=False,
       connector="sql", desc="Destructive; forbidden by default governance policy"),
]


class ActionLibrary:
    def __init__(self, actions: Iterable[RemediationAction] = DEFAULT_ACTIONS) -> None:
        self.actions: Dict[str, RemediationAction] = {a.name: a for a in actions}

    def for_category(self, category: FC) -> List[RemediationAction]:
        return [a for a in self.actions.values() if category in a.applicable_to]

    def __getitem__(self, name: str) -> RemediationAction:
        return self.actions[name]

    def __iter__(self):
        return iter(self.actions.values())


def remediation_effectiveness(S: float, H: float, R: float, gamma) -> float:
    """Eq. (4)."""
    g1, g2, g3 = gamma
    return g1 * S + g2 * H + g3 * R


class RemediationPlanner:
    def __init__(self, kg: AdaptiveKnowledgeGraph, cfg: ASHDPFConfig,
                 library: Optional[ActionLibrary] = None, policy=None) -> None:
        self.kg = kg
        self.cfg = cfg
        self.library = library or ActionLibrary()
        self.policy = policy

    def success_rate(self, category: FC, action: str) -> float:
        s, n, _, _ = self.kg.fix_stats(category, action)
        return (s + 1.0) / (n + 2.0)  # Laplace-smoothed historical success rate

    def plan(self, hyp: RootCauseHypothesis, event: FailureEvent,
             neighbours=None, categories: int = 2) -> List[ScoredAction]:
        """Return candidate actions ranked best-first.

        Candidates come from the top ``categories`` hypotheses, so a fallback for
        the runner-up root cause is available if the first attempt fails.
        """
        ranked_cats = sorted(hyp.distribution, key=hyp.distribution.get, reverse=True)[:categories]
        neighbours = neighbours or []
        out: List[ScoredAction] = []
        seen = set()
        for cat in ranked_cats:
            actions = self.library.for_category(cat)
            if not self.cfg.use_planner:
                # Ablation: static playbook = first applicable action, no scoring.
                actions = actions[:1]
            probs = (self.policy.probabilities(cat, [a.name for a in actions])
                     if self.policy is not None else {a.name: 1.0 / max(1, len(actions)) for a in actions})
            for a in actions:
                if a.name in seen:
                    continue
                seen.add(a.name)
                S = max((s for c, s in neighbours
                         if c.action == a.name and c.outcome == Outcome.SUCCESS
                         and c.category == cat), default=0.0)
                H = self.success_rate(cat, a.name)
                _, _, W, real = self.kg.fix_stats(cat, a.name)
                res = remediation_effectiveness(S, H, a.risk, self.cfg.gamma)
                p = probs.get(a.name, 0.0)
                score = res + self.cfg.policy_weight * p if self.cfg.use_planner else 1.0
                # Lower-ranked hypotheses are discounted by their probability gap.
                score *= hyp.distribution[cat] / hyp.distribution[ranked_cats[0]]
                out.append(ScoredAction(a, S, H, a.risk, res, p, W, score,
                                        uses_history=(S >= self.cfg.kri_similarity_threshold
                                                      or real > 0),
                                        category=cat))
        out.sort(key=lambda s: -s.rank_score)
        return out
