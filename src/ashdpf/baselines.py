"""Baselines of paper Sec. IV.C: monitoring-only, rule-based, AIOps recommendations."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .models import Decision, FailureCategory, FailureEvent, IncidentRecord, Outcome, RootCauseHypothesis
from .rca import KeywordRCA, MLClassifier
from .remediation import ActionLibrary
from .simulation import SimulatedOperator, SyntheticEnvironment


class MonitoringOnlyBaseline:
    """Alert generation + fully manual diagnosis and remediation."""

    name = "Manual Operations"

    def __init__(self, env: SyntheticEnvironment, seed: int = 1) -> None:
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.op = SimulatedOperator(env, self.rng)

    def handle(self, ev: FailureEvent) -> IncidentRecord:
        res = self.op.resolve(ev, [], acked=False)
        return IncidentRecord(ev, None, res.action, Decision.HUMAN_APPROVAL, res.outcome,
                              res.minutes, True, False, 1, res.category)

    def run(self, events: Sequence[FailureEvent]) -> List[IncidentRecord]:
        return [self.handle(e) for e in events]


class RuleBasedBaseline:
    """Fixed keyword decision tree + static playbook; humans when rules miss/fail."""

    name = "Rule-Based Automation"

    def __init__(self, env: SyntheticEnvironment, seed: int = 2,
                 library: Optional[ActionLibrary] = None) -> None:
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.op = SimulatedOperator(env, self.rng)
        self.rules = KeywordRCA()
        self.library = library or ActionLibrary()
        self.default_category = FailureCategory.INFRASTRUCTURE

    def handle(self, ev: FailureEvent) -> IncidentRecord:
        matched = self.rules.classify(ev)
        cat = matched or self.default_category  # catch-all rule: "restart it"
        hyp = RootCauseHypothesis(cat, 1.0, {cat: 1.0}, {})
        action = self.library.for_category(cat)[0]  # static playbook entry
        res = self.env.execute(action, ev, self.rng)
        minutes = 3.0 + res.duration_minutes
        if res.outcome_hint == Outcome.SUCCESS:
            return IncidentRecord(ev, hyp, action, Decision.AUTO_EXECUTE, Outcome.SUCCESS,
                                  minutes, False, False, 1, cat, [action.name])
        man = self.op.resolve(ev, [action.name], acked=False)
        return IncidentRecord(ev, hyp, man.action, Decision.HUMAN_APPROVAL, man.outcome,
                              minutes + man.minutes, True, False, 2, man.category)

    def run(self, events):
        return [self.handle(e) for e in events]


class AIOpsRecommendationBaseline:
    """ML root-cause suggestion + recommended fix, always subject to human approval."""

    name = "AIOps Recommendations"

    def __init__(self, env: SyntheticEnvironment, history: Sequence[IncidentRecord],
                 seed: int = 3, retrain_every: int = 500) -> None:
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.op = SimulatedOperator(env, self.rng)
        self.library = ActionLibrary()
        self.clf = MLClassifier(seed=seed)
        self.events = [r.event for r in history]
        self.labels = [r.confirmed_category for r in history]
        self.success: Dict = defaultdict(lambda: [1.0, 2.0])
        for r in history:
            if r.action is not None:
                self._count(r.confirmed_category, r.action.name, r.outcome)
        self.clf.fit(self.events, self.labels)
        self.retrain_every = retrain_every
        self._n = 0

    def _count(self, cat, action, outcome):
        s = self.success[(cat, action)]
        s[1] += 1
        s[0] += 1 if outcome == Outcome.SUCCESS else 0

    def handle(self, ev: FailureEvent) -> IncidentRecord:
        dist = self.clf.predict_proba(ev)
        cat = max(dist, key=dist.get)
        hyp = RootCauseHypothesis(cat, dist[cat], dist, {"ML": dist})
        acts = self.library.for_category(cat)
        action = max(acts, key=lambda a: self.success[(cat, a.name)][0] / self.success[(cat, a.name)][1])
        # A human must acknowledge, review and approve every recommendation.
        minutes = self.op._n(self.op.ack_mean) + self.op._n(self.op.review_mean)
        true = ev.true_category
        if self.env.effectiveness(true, action.name)[0] >= 0.5:
            res = self.env.execute(action, ev, self.rng)
            minutes += res.duration_minutes
            if res.outcome_hint == Outcome.SUCCESS:
                rec = IncidentRecord(ev, hyp, action, Decision.HUMAN_APPROVAL, Outcome.SUCCESS,
                                     minutes, True, True, 1, true, [action.name])
                self._learn(rec)
                return rec
            man = self.op.resolve(ev, [action.name], acked=True)
        else:
            man = self.op.resolve(ev, [], acked=True)
        rec = IncidentRecord(ev, hyp, man.action, Decision.HUMAN_APPROVAL, man.outcome,
                             minutes + man.minutes, True, False, 2, true)
        self._learn(rec)
        return rec

    def _learn(self, rec: IncidentRecord) -> None:
        self.events.append(rec.event)
        self.labels.append(rec.confirmed_category)
        if rec.action is not None:
            self._count(rec.confirmed_category, rec.action.name, rec.outcome)
        self._n += 1
        if self._n % self.retrain_every == 0:
            self.clf.fit(self.events, self.labels)

    def run(self, events):
        return [self.handle(e) for e in events]
