"""AOKG: graph model, Eq. (1) edge-weight learning, traversal, persistence."""

import pytest

from ashdpf.knowledge_graph import AdaptiveKnowledgeGraph, EdgeType, NodeType, fix_id, runbook_id
from ashdpf.models import (FailureCategory as FC, IncidentRecord, Outcome, RemediationAction)


@pytest.fixture
def chain():
    """a <- b <- c <- d   (d DEPENDS_ON c DEPENDS_ON b DEPENDS_ON a)"""
    g = AdaptiveKnowledgeGraph(alpha=0.1)
    for n in "abcd":
        g.add_node(n, NodeType.JOB)
    g.add_edge("b", "a", EdgeType.DEPENDS_ON, weight=1.0)
    g.add_edge("c", "b", EdgeType.DEPENDS_ON, weight=0.5)
    g.add_edge("d", "c", EdgeType.DEPENDS_ON, weight=1.0)
    return g


def test_all_patent_node_and_edge_types_exist():
    assert {t.value for t in NodeType} == {
        "Pipeline", "Job", "Table", "Incident", "Runbook", "Fix", "Engineer", "Application"}
    assert {t.value for t in EdgeType} == {
        "DEPENDS_ON", "FAILED_BY", "FIXED_BY", "REFERENCES", "RELATED_TO"}


def test_add_node_is_idempotent_and_merges_props():
    g = AdaptiveKnowledgeGraph()
    g.add_node("j", NodeType.JOB, owner="x")
    g.add_node("j", NodeType.JOB, sla=3)
    assert len(g.nodes) == 1 and g.nodes["j"].props == {"owner": "x", "sla": 3}


def test_add_node_type_conflict_raises():
    g = AdaptiveKnowledgeGraph()
    g.add_node("j", NodeType.JOB)
    with pytest.raises(ValueError):
        g.add_node("j", NodeType.TABLE)


def test_edge_requires_existing_nodes():
    g = AdaptiveKnowledgeGraph()
    g.add_node("a", NodeType.JOB)
    with pytest.raises(KeyError):
        g.add_edge("a", "missing", EdgeType.DEPENDS_ON)


def test_equation_1_success_strengthens_edge(chain):
    # W_e = W_e + alpha * R  ->  0.5 + 0.1 * 1.0 = 0.6
    assert chain.update_edge_weight("c", "b", EdgeType.DEPENDS_ON, 1.0) == pytest.approx(0.6)


def test_equation_1_failure_weakens_edge(chain):
    assert chain.update_edge_weight("c", "b", EdgeType.DEPENDS_ON, -1.0) == pytest.approx(0.4)


def test_equation_1_custom_alpha(chain):
    assert chain.update_edge_weight("c", "b", EdgeType.DEPENDS_ON, 0.5, alpha=0.2) == pytest.approx(0.6)


def test_edge_weight_is_clipped_to_bounds(chain):
    for _ in range(20):
        chain.update_edge_weight("c", "b", EdgeType.DEPENDS_ON, 1.0)
    assert chain.get_edge("c", "b", EdgeType.DEPENDS_ON).weight == 1.0
    for _ in range(30):
        chain.update_edge_weight("c", "b", EdgeType.DEPENDS_ON, -1.0)
    assert chain.get_edge("c", "b", EdgeType.DEPENDS_ON).weight == 0.0


def test_static_graph_ignores_weight_updates(chain):
    chain.adaptive = False
    assert chain.update_edge_weight("c", "b", EdgeType.DEPENDS_ON, 1.0) == 0.5


def test_upstream_traversal_respects_depth_and_path_weight(chain):
    up = {n: (d, w) for n, d, w in chain.upstream("d", max_depth=3)}
    assert up == {"c": (1, 1.0), "b": (2, 0.5), "a": (3, 0.5)}
    assert {n for n, _, _ in chain.upstream("d", max_depth=1)} == {"c"}


def test_downstream_is_blast_radius(chain):
    assert {n for n, _, _ in chain.downstream("a")} == {"b", "c", "d"}
    assert chain.downstream("d") == []


def test_recently_failed_upstream_uses_time_window(chain):
    chain.mark_failed("a", timestamp=100.0)
    assert [n for n, _, _ in chain.recently_failed_upstream("d", now=150.0, window=60)] == ["a"]
    assert chain.recently_failed_upstream("d", now=500.0, window=60) == []


def test_cycle_detection(chain):
    assert not chain.has_cycle()
    chain.add_edge("a", "d", EdgeType.DEPENDS_ON)
    assert chain.has_cycle()


def test_ingest_incident_populates_graph(event_factory):
    g = AdaptiveKnowledgeGraph()
    ev = event_factory(job="p.t1", pipeline="p", eid="42")
    act = RemediationAction("restart_workflow", frozenset({FC.INFRASTRUCTURE}), 0.1)
    rec = IncidentRecord(ev, None, act, None, Outcome.SUCCESS, 12.0, True, False,
                         confirmed_category=FC.INFRASTRUCTURE)
    inc = g.ingest_incident(rec, engineer="alice")
    assert g.nodes[inc].type == NodeType.INCIDENT
    assert g.get_edge("p.t1", inc, EdgeType.FAILED_BY) is not None
    assert g.get_edge("p", inc, EdgeType.FAILED_BY) is not None
    assert g.get_edge(inc, fix_id("restart_workflow"), EdgeType.FIXED_BY).weight == 1.0
    assert g.get_edge(inc, runbook_id(FC.INFRASTRUCTURE), EdgeType.RELATED_TO) is not None
    assert g.get_edge("alice", inc, EdgeType.RELATED_TO) is not None
    assert g.failure_profile("p.t1")["infrastructure"] == 1


def test_record_fix_outcome_counts_and_applies_eq1():
    g = AdaptiveKnowledgeGraph(alpha=0.1)
    g.ensure_runbook_fix(FC.CONFIGURATION, "reset_configuration")
    g.record_fix_outcome(FC.CONFIGURATION, "reset_configuration", Outcome.SUCCESS, 1.0)
    g.record_fix_outcome(FC.CONFIGURATION, "reset_configuration", Outcome.FAILURE, -1.0)
    g.record_fix_outcome(FC.CONFIGURATION, "reset_configuration", Outcome.SUCCESS, 1.0)
    s, n, w, real = g.fix_stats(FC.CONFIGURATION, "reset_configuration")
    assert (s, n, real) == (2, 3, 3)
    assert w == pytest.approx(0.6)


def test_json_roundtrip(tmp_path, chain):
    chain.failure_profile("a")["credential"] += 2
    p = tmp_path / "kg.json"
    chain.save(str(p))
    g2 = AdaptiveKnowledgeGraph.load(str(p))
    assert set(g2.nodes) == set(chain.nodes)
    assert g2.get_edge("c", "b", EdgeType.DEPENDS_ON).weight == 0.5
    assert g2.failure_profile("a")["credential"] == 2


def test_cypher_export(chain):
    stmts = list(chain.to_cypher())
    assert any(s.startswith("MERGE (n:Job") for s in stmts)
    assert sum("DEPENDS_ON" in s for s in stmts) == 3
