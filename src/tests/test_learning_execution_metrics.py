"""CLE (Eq. 5, 6), execution framework (audit / rollback), validation, metrics."""

import json

import pytest

from ashdpf.config import ASHDPFConfig
from ashdpf.execution import CallableConnector, ExecutionFramework, ValidationModule
from ashdpf.knowledge_graph import AdaptiveKnowledgeGraph
from ashdpf.learning import ContinuousLearningEngine, SoftmaxPolicy, reward
from ashdpf.metrics import (autonomous_resolution_rate, classification_report,
                            knowledge_reuse_index, mttr)
from ashdpf.models import (ExecutionResult, FailureCategory as FC, IncidentRecord, Outcome,
                           RemediationAction)
from ashdpf.rca import RCAEngine
from ashdpf.remediation import RemediationPlanner

ACT = RemediationAction("restart_workflow", frozenset({FC.INFRASTRUCTURE}), 0.1,
                        connector="workflow_trigger")


# ---------------------------------------------------------------- Eq. (5)
def test_reward_is_inverse_mttr():
    assert reward(Outcome.SUCCESS, 20.0) == pytest.approx(1 / 20)
    assert reward(Outcome.SUCCESS, 10.0, scale=10) == pytest.approx(1.0)
    assert reward(Outcome.SUCCESS, 5.0) > reward(Outcome.SUCCESS, 50.0)


def test_partial_and_failure_rewards():
    assert reward(Outcome.PARTIAL, 10.0) == pytest.approx(0.05)
    assert reward(Outcome.FAILURE, 1.0) == 0.0


# ---------------------------------------------------------------- Eq. (6)
def test_policy_starts_uniform():
    p = SoftmaxPolicy().probabilities(FC.INFRASTRUCTURE, ["a", "b", "c", "d"])
    assert all(v == pytest.approx(0.25) for v in p.values())


def test_policy_gradient_step_matches_reinforce():
    pol = SoftmaxPolicy(eta=0.5)
    grad = pol.update("s", "a", ["a", "b"], r=1.0)
    # baseline 0, pi = 0.5 each: grad_a = 1*(1-0.5), grad_b = 1*(0-0.5)
    assert grad == {"a": pytest.approx(0.5), "b": pytest.approx(-0.5)}
    assert pol.theta["s"]["a"] == pytest.approx(0.25)
    assert pol.probabilities("s", ["a", "b"])["a"] > 0.5


def test_policy_converges_to_fastest_action():
    import numpy as np
    rng = np.random.default_rng(0)
    pol = SoftmaxPolicy(eta=0.5)
    acts = ["fast", "slow"]
    for _ in range(300):
        p = pol.probabilities("s", acts)
        a = acts[int(rng.choice(2, p=[p["fast"], p["slow"]]))]
        pol.update("s", a, acts, reward(Outcome.SUCCESS, 5 if a == "fast" else 50, scale=10))
    assert pol.probabilities("s", acts)["fast"] > 0.9


def test_cle_updates_policy_graph_and_repository(event_factory):
    cfg = ASHDPFConfig(min_train_samples=1000)
    kg = AdaptiveKnowledgeGraph(alpha=0.1)
    rca = RCAEngine(kg, cfg)
    planner = RemediationPlanner(kg, cfg)
    cle = ContinuousLearningEngine(kg, cfg, rca, planner)
    ev = event_factory()
    rec = IncidentRecord(ev, None, ACT, None, Outcome.SUCCESS, 10.0, False, False,
                         confirmed_category=FC.INFRASTRUCTURE)
    cle.learn(rec)
    s, n, w, real = kg.fix_stats(FC.INFRASTRUCTURE, "restart_workflow")
    assert (s, n, real) == (1, 1, 1) and w == pytest.approx(0.6)
    assert f"incident:{ev.event_id}" in kg.nodes
    assert len(rca.history) == 1
    assert cle.policy.theta["infrastructure"]["restart_workflow"] > 0


def test_cle_static_graph_is_frozen(event_factory):
    cfg = ASHDPFConfig(kg_mode="static", min_train_samples=1000)
    kg = AdaptiveKnowledgeGraph(adaptive=False)
    cle = ContinuousLearningEngine(kg, cfg, RCAEngine(kg, cfg), RemediationPlanner(kg, cfg))
    cle.learn(IncidentRecord(event_factory(), None, ACT, None, Outcome.SUCCESS, 10.0, False, False,
                             confirmed_category=FC.INFRASTRUCTURE))
    assert kg.fix_stats(FC.INFRASTRUCTURE, "restart_workflow")[1] == 0
    assert not any(n.startswith("incident:") for n in kg.nodes)


# -------------------------------------------------------------- execution
def test_execution_creates_audit_and_checkpoint(event_factory, tmp_path):
    ex = ExecutionFramework()
    ex.register(CallableConnector("workflow_trigger", lambda a, e: ExecutionResult(True, 4.0)))
    res, audit = ex.execute(ACT, event_factory(), approved_by="alice")
    assert res.ok and audit.approved_by == "alice" and audit.checkpoint_id in ex.checkpoints
    p = tmp_path / "audit.jsonl"
    ex.export_audit(str(p))
    assert json.loads(p.read_text().splitlines()[0])["action"] == "restart_workflow"


def test_rollback_invokes_connector(event_factory):
    seen = []
    ex = ExecutionFramework()
    ex.register(CallableConnector("workflow_trigger", lambda a, e: ExecutionResult(False, 1.0),
                                  rollback_fn=lambda snap: seen.append(snap) or True))
    _, audit = ex.execute(ACT, event_factory())
    assert ex.rollback(audit.audit_id) and audit.rolled_back
    assert seen[0]["action"] == "restart_workflow"


def test_wildcard_connector_and_missing_connector(event_factory):
    ex = ExecutionFramework()
    with pytest.raises(KeyError):
        ex.execute(ACT, event_factory())
    ex.register(CallableConnector("*", lambda a, e: ExecutionResult(True, 1.0)))
    assert ex.execute(ACT, event_factory())[0].ok


def test_connector_exception_becomes_failure(event_factory):
    def boom(a, e):
        raise RuntimeError("api down")
    ex = ExecutionFramework()
    ex.register(CallableConnector("*", boom))
    res, audit = ex.execute(ACT, event_factory())
    assert not res.ok and res.outcome_hint == Outcome.FAILURE and "api down" in audit.details["error"]


def test_validation_module(event_factory):
    ev = event_factory()
    assert ValidationModule().validate(ev, ExecutionResult(True, 1)) == Outcome.SUCCESS
    assert ValidationModule().validate(ev, ExecutionResult(False, 1)) == Outcome.FAILURE
    assert ValidationModule().validate(ev, ExecutionResult(True, 1, Outcome.PARTIAL)) == Outcome.PARTIAL
    assert ValidationModule(lambda e, r: "failure").validate(ev, ExecutionResult(True, 1)) == Outcome.FAILURE


# ---------------------------------------------------------------- metrics
def test_classification_report_known_values():
    rep = classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert rep["accuracy"] == 0.75
    # a: P=1, R=0.5, F1=2/3 ; b: P=2/3, R=1, F1=0.8
    assert rep["precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert rep["recall"] == pytest.approx(0.75)
    assert rep["f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_classification_report_validates_length():
    with pytest.raises(ValueError):
        classification_report(["a"], [])


def _rec(ev, minutes, human, used, outcome=Outcome.SUCCESS, action=ACT):
    return IncidentRecord(ev, None, action, None, outcome, minutes, human, used)


def test_mttr_arr_kri(event_factory):
    ev = event_factory()
    recs = [_rec(ev, 10, False, True), _rec(ev, 30, True, False), _rec(ev, 20, False, True),
            _rec(ev, 40, False, False, Outcome.FAILURE), _rec(ev, 0, True, False, action=None)]
    assert mttr(recs) == pytest.approx(20.0)
    assert autonomous_resolution_rate(recs) == pytest.approx(2 / 5)
    assert knowledge_reuse_index(recs) == pytest.approx(2 / 4)  # actionless record excluded
    assert mttr([]) == 0.0 and autonomous_resolution_rate([]) == 0.0 and knowledge_reuse_index([]) == 0.0
