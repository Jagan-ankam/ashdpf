"""Planner (Eq. 4) and Safety Validation Layer (threshold T, governance, rollback)."""

import pytest

from ashdpf.config import ASHDPFConfig
from ashdpf.knowledge_graph import AdaptiveKnowledgeGraph, EdgeType, NodeType
from ashdpf.learning import SoftmaxPolicy
from ashdpf.models import (ALL_CATEGORIES, Decision, FailureCategory as FC, Outcome,
                           RemediationAction, RootCauseHypothesis, ScoredAction)
from ashdpf.remediation import ActionLibrary, RemediationPlanner, remediation_effectiveness
from ashdpf.safety import SafetyValidationLayer


def hyp_for(cat, p=0.9):
    rest = (1 - p) / (len(ALL_CATEGORIES) - 1)
    dist = {c: (p if c == cat else rest) for c in ALL_CATEGORIES}
    return RootCauseHypothesis(cat, p, dist, {})


def test_equation_4_exact():
    assert remediation_effectiveness(0.5, 0.8, 0.2, (0.3, 0.6, -0.3)) == pytest.approx(
        0.3 * 0.5 + 0.6 * 0.8 - 0.3 * 0.2)


def test_risk_lowers_res_with_default_gamma():
    g = ASHDPFConfig().gamma
    assert remediation_effectiveness(0.5, 0.5, 0.9, g) < remediation_effectiveness(0.5, 0.5, 0.1, g)


def test_library_filters_by_category():
    lib = ActionLibrary()
    names = {a.name for a in lib.for_category(FC.CREDENTIAL)}
    assert names == {"refresh_credentials"}
    assert all(FC.DATA_QUALITY in a.applicable_to for a in lib.for_category(FC.DATA_QUALITY))


def test_planner_prefers_historically_successful_action(event_factory):
    kg = AdaptiveKnowledgeGraph()
    for _ in range(10):
        kg.record_fix_outcome(FC.DATA_QUALITY, "reprocess_missing_data", Outcome.SUCCESS, 1.0)
        kg.record_fix_outcome(FC.DATA_QUALITY, "quarantine_bad_records", Outcome.FAILURE, -1.0)
    planner = RemediationPlanner(kg, ASHDPFConfig(), policy=SoftmaxPolicy())
    ranked = planner.plan(hyp_for(FC.DATA_QUALITY), event_factory())
    assert ranked[0].action.name == "reprocess_missing_data"
    assert ranked[0].category == FC.DATA_QUALITY
    assert ranked[0].success_rate == pytest.approx(11 / 12)  # Laplace smoothing
    assert ranked[0].uses_history  # real outcomes exist in the AOKG -> counts for KRI


def test_planner_includes_runner_up_hypothesis(event_factory):
    dist = {c: 0.0 for c in ALL_CATEGORIES}
    dist[FC.NETWORK_TIMEOUT], dist[FC.JOB_DEPENDENCY] = 0.6, 0.4
    hyp = RootCauseHypothesis(FC.NETWORK_TIMEOUT, 0.6, dist, {})
    ranked = RemediationPlanner(AdaptiveKnowledgeGraph(), ASHDPFConfig()).plan(hyp, event_factory())
    cats = {s.category for s in ranked}
    assert cats == {FC.NETWORK_TIMEOUT, FC.JOB_DEPENDENCY}
    assert len({s.action.name for s in ranked}) == len(ranked)  # no duplicates


def test_static_playbook_when_planner_disabled(event_factory):
    ranked = RemediationPlanner(AdaptiveKnowledgeGraph(), ASHDPFConfig(use_planner=False)).plan(
        hyp_for(FC.INFRASTRUCTURE), event_factory(), categories=1)
    assert [s.action.name for s in ranked] == ["restart_workflow"]


# ----------------------------------------------------------------- safety
def scored(action, weight, cat):
    return ScoredAction(action, 0, 0.5, action.risk, 0, 0, weight, 0, category=cat)


SAFE = RemediationAction("restart_workflow", frozenset({FC.INFRASTRUCTURE}), 0.1)


@pytest.fixture
def safety():
    kg = AdaptiveKnowledgeGraph()
    kg.add_node("p1.t1", NodeType.JOB)
    return SafetyValidationLayer(kg, ASHDPFConfig(threshold=0.5))


def test_auto_execute_when_confidence_exceeds_threshold(safety, event_factory):
    v = safety.validate(scored(SAFE, 0.9, FC.INFRASTRUCTURE), hyp_for(FC.INFRASTRUCTURE, 0.8), event_factory())
    assert v.decision == Decision.AUTO_EXECUTE
    assert v.confidence == pytest.approx(0.72) and all(v.checks.values())


def test_human_approval_when_below_threshold(safety, event_factory):
    v = safety.validate(scored(SAFE, 0.5, FC.INFRASTRUCTURE), hyp_for(FC.INFRASTRUCTURE, 0.8), event_factory())
    assert v.decision == Decision.HUMAN_APPROVAL and not v.checks["threshold"]


def test_threshold_is_strict_inequality(safety, event_factory):
    v = safety.validate(scored(SAFE, 1.0, FC.INFRASTRUCTURE), hyp_for(FC.INFRASTRUCTURE, 0.5), event_factory())
    assert v.decision == Decision.HUMAN_APPROVAL


def test_category_specific_threshold_for_credentials(safety, event_factory):
    act = RemediationAction("refresh_credentials", frozenset({FC.CREDENTIAL}), 0.4)
    v = safety.validate(scored(act, 1.0, FC.CREDENTIAL), hyp_for(FC.CREDENTIAL, 0.85), event_factory())
    assert v.threshold == 0.90 and v.decision == Decision.HUMAN_APPROVAL


def test_forbidden_action_is_rejected(safety, event_factory):
    act = RemediationAction("drop_and_recreate_table", frozenset({FC.SCHEMA_EVOLUTION}), 0.9, False)
    v = safety.validate(scored(act, 1.0, FC.SCHEMA_EVOLUTION), hyp_for(FC.SCHEMA_EVOLUTION, 0.99), event_factory())
    assert v.decision == Decision.REJECTED


def test_irreversible_risky_action_needs_human(safety, event_factory):
    act = RemediationAction("purge", frozenset({FC.DATA_QUALITY}), 0.45, reversible=False)
    v = safety.validate(scored(act, 1.0, FC.DATA_QUALITY), hyp_for(FC.DATA_QUALITY, 0.99), event_factory())
    assert v.decision == Decision.HUMAN_APPROVAL and not v.checks["rollback"]


def test_high_risk_action_fails_governance(safety, event_factory):
    act = RemediationAction("big", frozenset({FC.DATA_QUALITY}), 0.8)
    v = safety.validate(scored(act, 1.0, FC.DATA_QUALITY), hyp_for(FC.DATA_QUALITY, 0.99), event_factory())
    assert not v.checks["governance"] and v.decision == Decision.HUMAN_APPROVAL


def test_large_blast_radius_blocks_risky_action(event_factory):
    kg = AdaptiveKnowledgeGraph()
    kg.add_node("root", NodeType.JOB)
    for i in range(5):
        kg.add_node(f"d{i}", NodeType.JOB)
        kg.add_edge(f"d{i}", "root", EdgeType.DEPENDS_ON)
    s = SafetyValidationLayer(kg, ASHDPFConfig(threshold=0.1, max_blast_radius=3))
    act = RemediationAction("reset_configuration", frozenset({FC.CONFIGURATION}), 0.3)
    v = s.validate(scored(act, 1.0, FC.CONFIGURATION), hyp_for(FC.CONFIGURATION), event_factory(job="root"))
    assert not v.checks["dependency_impact"]
    # a near-harmless action is still allowed despite the blast radius
    v2 = s.validate(scored(RemediationAction("wait", frozenset({FC.CONFIGURATION}), 0.05), 1.0,
                           FC.CONFIGURATION), hyp_for(FC.CONFIGURATION), event_factory(job="root"))
    assert v2.checks["dependency_impact"]
