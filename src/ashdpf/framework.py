"""ASHDPF closed loop: failure -> context -> RCA -> plan -> safety -> execute ->
validate -> learn (paper Sec. III.A, patent FIG. 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from .config import ASHDPFConfig
from .execution import Connector, ExecutionFramework, ValidationModule
from .knowledge_graph import AdaptiveKnowledgeGraph
from .layers import ContextAcquisitionLayer, MonitoringLayer
from .learning import ContinuousLearningEngine, SoftmaxPolicy, reward
from .metrics import summarize
from .models import (Decision, FailureCategory, FailureEvent, IncidentRecord, Outcome,
                     RemediationAction, RootCauseHypothesis, SafetyVerdict, ScoredAction)
from .rca import MLClassifier, RCAEngine
from .remediation import ActionLibrary, RemediationPlanner
from .safety import SafetyValidationLayer


@dataclass
class HumanReview:
    """A human operator's answer to an approval request."""

    approved: bool
    latency_minutes: float
    action: Optional[RemediationAction] = None   # override / replacement action
    category: Optional[FailureCategory] = None   # confirmed root cause
    approver: str = "human"


@dataclass
class ManualResolution:
    action: Optional[RemediationAction]
    minutes: float
    outcome: Outcome
    category: Optional[FailureCategory]
    engineer: str = "on-call"


class HumanOperator(Protocol):
    def review(self, event: FailureEvent, hyp: RootCauseHypothesis, cand: ScoredAction,
               verdict: SafetyVerdict) -> HumanReview: ...

    def resolve(self, event: FailureEvent, tried: Sequence[str]) -> ManualResolution: ...


class ASHDPF:
    def __init__(
        self,
        cfg: Optional[ASHDPFConfig] = None,
        library: Optional[ActionLibrary] = None,
        human: Optional[HumanOperator] = None,
        classifier: Optional[MLClassifier] = None,
        validation: Optional[ValidationModule] = None,
        analysis_overhead_minutes: float = 2.0,
    ) -> None:
        self.cfg = (cfg or ASHDPFConfig()).validate()
        c = self.cfg
        self.kg = AdaptiveKnowledgeGraph(c.alpha, c.initial_edge_weight, c.edge_weight_bounds,
                                         adaptive=c.kg_mode != "static")
        self.monitoring = MonitoringLayer(self.kg)
        self.context = ContextAcquisitionLayer(self.kg)
        self.rca = RCAEngine(self.kg, c, classifier)
        self.policy = SoftmaxPolicy(c.eta)
        self.planner = RemediationPlanner(self.kg, c, library, self.policy)
        self.safety = SafetyValidationLayer(self.kg, c)
        self.executor = ExecutionFramework()
        self.validation = validation or ValidationModule()
        self.learning = ContinuousLearningEngine(self.kg, c, self.rca, self.planner, self.policy)
        self.human = human
        self.overhead = analysis_overhead_minutes
        self.records: List[IncidentRecord] = []

    # ------------------------------------------------------------- set-up
    def register_connector(self, connector: Connector, name: Optional[str] = None) -> None:
        self.executor.register(connector, name)

    def bootstrap(self, history: Sequence[IncidentRecord]) -> None:
        """Load the historical incident repository (Sec. IV.B) before going live.

        A *static* knowledge graph is built here once and frozen afterwards.
        """
        self.kg.adaptive = True
        for rec in history:
            cat = rec.confirmed_category or rec.event.true_category
            if cat is None:
                continue
            if self.cfg.kg_mode != "none":
                self.kg.ingest_incident(rec)
            if rec.action is not None:
                self.kg.record_fix_outcome(cat, rec.action.name, rec.outcome,
                                           self.cfg.outcome_scores[rec.outcome.value])
                actions = [a.name for a in self.planner.library.for_category(cat)]
                self.policy.update(cat, rec.action.name, actions,
                                   reward(rec.outcome, rec.resolution_minutes, self.cfg.reward_scale))
            self.rca.history.add(rec.event, cat, rec.action.name if rec.action else None, rec.outcome)
            self.rca._train_events.append(rec.event)
            self.rca._train_labels.append(cat)
        if len(self.rca._train_events) >= self.cfg.min_train_samples:
            self.rca.retrain()
        self.kg.adaptive = self.cfg.kg_mode != "static"

    # -------------------------------------------------------------- loop
    def handle_signal(self, signal) -> IncidentRecord:
        return self.handle(self.monitoring.build_event(signal))

    def handle(self, event: FailureEvent) -> IncidentRecord:
        self.kg.mark_failed(event.job, event.timestamp)
        minutes = self.overhead
        hyp = self.rca.analyze(event)
        cands = self.planner.plan(hyp, event, self.rca.last_neighbours)

        human = False
        used_history = False
        confirmed: Optional[FailureCategory] = None
        decision: Optional[Decision] = None
        final_action: Optional[RemediationAction] = None
        outcome = Outcome.FAILURE
        tried: List[str] = []
        audits: List[str] = []
        attempts = 0
        engineer = None

        for cand in cands:
            if attempts >= self.cfg.max_attempts:
                break
            verdict = self.safety.validate(cand, hyp, event)
            if verdict.decision == Decision.REJECTED:
                continue
            decision = verdict.decision if decision is None else decision
            action, approver, target_cat = cand.action, "ashdpf-autonomous", cand.category
            if verdict.decision == Decision.HUMAN_APPROVAL:
                if self.human is None:
                    decision = Decision.HUMAN_APPROVAL
                    break  # escalate: nobody to approve
                rv = self.human.review(event, hyp, cand, verdict)
                human = True
                minutes += rv.latency_minutes
                confirmed = rv.category or confirmed
                if not rv.approved and rv.action is None:
                    break
                if rv.action is not None and rv.action.name != action.name:
                    action = rv.action
                    target_cat = rv.category or target_cat
                approver = rv.approver
            if action.name in tried:
                continue
            used_history = used_history or (cand.uses_history and action.name == cand.action.name)
            result, audit = self.executor.execute(action, event, approver)
            audits.append(audit.audit_id)
            attempts += 1
            tried.append(action.name)
            minutes += result.duration_minutes
            outcome = self.validation.validate(event, result)
            final_action = action
            if outcome == Outcome.SUCCESS:
                if confirmed is None and target_cat is not None:
                    confirmed = target_cat
                break
            # Failed attempt: learn from it immediately, roll back, try the next one.
            if target_cat is not None:
                self._learn_attempt(target_cat, action, outcome)
            if action.reversible:
                self.executor.rollback(audit.audit_id)
            if human:
                break  # once a human is engaged they own the resolution

        if outcome != Outcome.SUCCESS and self.human is not None:
            res = self.human.resolve(event, tried)
            human = True
            minutes += res.minutes
            outcome = res.outcome
            confirmed = res.category or confirmed
            engineer = res.engineer
            # The manual fix (possibly unknown) is what resolved the incident; the
            # automated attempts were already learned from as failures above.
            final_action = res.action
            used_history = False
            if decision is None:
                decision = Decision.HUMAN_APPROVAL

        if outcome == Outcome.SUCCESS:
            self.kg.mark_healthy(event.job)
        rec = IncidentRecord(event, hyp, final_action, decision, outcome, minutes, human,
                             used_history, attempts, confirmed, tried, audits)
        self.learning.learn(rec, engineer=engineer)
        self.records.append(rec)
        return rec

    def _learn_attempt(self, cat: FailureCategory, action: RemediationAction, outcome: Outcome) -> None:
        if self.cfg.kg_mode != "static":
            self.kg.record_fix_outcome(cat, action.name, outcome, self.cfg.outcome_scores[outcome.value])
        if self.cfg.use_learning:
            actions = [a.name for a in self.planner.library.for_category(cat)]
            self.policy.update(cat, action.name, actions, 0.0)

    def run(self, events: Sequence[FailureEvent]) -> List[IncidentRecord]:
        return [self.handle(e) for e in events]

    def report(self, records: Optional[Sequence[IncidentRecord]] = None) -> dict:
        """Governance report incl. the Knowledge Reuse Index (claims 8 & 12)."""
        recs = list(records if records is not None else self.records)
        out = summarize(recs)
        out["audit_entries"] = float(len(self.executor.audit_log))
        out["kg_nodes"] = float(len(self.kg.nodes))
        out["kg_edges"] = float(len(self.kg.edges))
        return out
