"""RCA engine: Eq. (2) argmax, Eq. (3) weighted fusion, each component."""

import pytest

from ashdpf.config import ASHDPFConfig
from ashdpf.knowledge_graph import AdaptiveKnowledgeGraph, EdgeType, NodeType
from ashdpf.models import ALL_CATEGORIES, FailureCategory as FC, Outcome
from ashdpf.rca import (GraphReasoner, HistoricalMatcher, KeywordRCA, MLClassifier, RCAEngine,
                        combine, normalise, uniform)


def one_hot(cat, p=1.0):
    rest = (1 - p) / (len(ALL_CATEGORIES) - 1)
    return {c: (p if c == cat else rest) for c in ALL_CATEGORIES}


def test_equation_3_weighted_sum_is_exact():
    G, H, ML = one_hot(FC.CONFIGURATION), one_hot(FC.CREDENTIAL), one_hot(FC.CONFIGURATION)
    out = combine([(0.2, G), (0.3, H), (0.5, ML)])
    assert out[FC.CONFIGURATION] == pytest.approx(0.7)
    assert out[FC.CREDENTIAL] == pytest.approx(0.3)
    assert sum(out.values()) == pytest.approx(1.0)


def test_combine_renormalises_when_a_component_is_missing():
    out = combine([(0.25, one_hot(FC.SCHEMA_EVOLUTION)), (0.0, one_hot(FC.CREDENTIAL))])
    assert out[FC.SCHEMA_EVOLUTION] == pytest.approx(1.0)


def test_combine_with_no_components_is_uniform():
    assert combine([]) == uniform()


def test_normalise_handles_negative_and_zero():
    assert normalise({}) == uniform()
    d = normalise({FC.CREDENTIAL: 2.0, FC.CONFIGURATION: -1.0})
    assert d[FC.CREDENTIAL] == 1.0 and d[FC.CONFIGURATION] == 0.0


def _dep_graph():
    g = AdaptiveKnowledgeGraph()
    for n in ("up", "down"):
        g.add_node(n, NodeType.JOB)
    g.add_edge("down", "up", EdgeType.DEPENDS_ON, weight=1.0)
    return g


def test_graph_reasoner_detects_cascading_dependency_failure(event_factory):
    g = _dep_graph()
    g.mark_failed("up", 90.0)
    dist, evidence = GraphReasoner(g, ASHDPFConfig()).score(event_factory(job="down", ts=100.0))
    assert max(dist, key=dist.get) == FC.JOB_DEPENDENCY
    assert evidence and "up" in evidence[0]


def test_graph_reasoner_without_evidence_is_uniform(event_factory):
    dist, _ = GraphReasoner(_dep_graph(), ASHDPFConfig()).score(event_factory(job="down"))
    assert dist == uniform()


def test_graph_reasoner_uses_failure_profile(event_factory):
    g = _dep_graph()
    g.failure_profile("down")["schema_evolution"] = 9
    dist, _ = GraphReasoner(g, ASHDPFConfig()).score(event_factory(job="down"))
    assert max(dist, key=dist.get) == FC.SCHEMA_EVOLUTION


def test_historical_matcher_retrieves_identical_incident(event_factory):
    m = HistoricalMatcher(None, top_k=3, use_graph=False)
    m.add(event_factory(logs=["java.lang.OutOfMemoryError: Java heap space"], eid="a"),
          FC.RESOURCE_EXHAUSTION, "scale_up_resources", Outcome.SUCCESS)
    m.add(event_factory(logs=["Access denied for user etl"], eid="b"),
          FC.CREDENTIAL, "refresh_credentials", Outcome.SUCCESS)
    dist, top, nbrs = m.score(event_factory(logs=["java.lang.OutOfMemoryError: Java heap space"]))
    assert nbrs[0][0].event_id == "a"
    assert top == pytest.approx(1.0, abs=1e-6)
    assert max(dist, key=dist.get) == FC.RESOURCE_EXHAUSTION


def test_empty_history_is_uniform(event_factory):
    dist, top, nbrs = HistoricalMatcher(None).score(event_factory())
    assert dist == uniform() and top == 0.0 and nbrs == []


def test_untrained_classifier_returns_uniform(event_factory):
    assert MLClassifier().predict_proba(event_factory()) == uniform()


@pytest.mark.parametrize("model", ["random_forest", "gradient_boosting"])
def test_classifier_learns_separable_classes(event_factory, model):
    evs, ys = [], []
    for i in range(30):
        evs.append(event_factory(logs=[f"OutOfMemoryError heap {i}"], eid=f"o{i}"))
        ys.append(FC.RESOURCE_EXHAUSTION)
        evs.append(event_factory(logs=[f"Access denied token {i}"], eid=f"c{i}"))
        ys.append(FC.CREDENTIAL)
    clf = MLClassifier(model=model, n_estimators=20).fit(evs, ys)
    p = clf.predict_proba(event_factory(logs=["OutOfMemoryError heap"]))
    assert max(p, key=p.get) == FC.RESOURCE_EXHAUSTION
    assert clf.predict([event_factory(logs=["Access denied token"])]) == [FC.CREDENTIAL]


def test_keyword_rules():
    from ashdpf.models import EventVector, FailureEvent
    ev = lambda s: FailureEvent("x", "p", "j", 0, EventVector([s]))
    k = KeywordRCA()
    assert k.classify(ev("HTTP 403 Forbidden")) == FC.CREDENTIAL
    assert k.classify(ev("java.lang.OutOfMemoryError")) == FC.RESOURCE_EXHAUSTION
    assert k.classify(ev("Task exited with return code 1")) is None


def test_engine_argmax_and_components(event_factory):
    g = _dep_graph()
    g.mark_failed("up", 95.0)
    cfg = ASHDPFConfig(min_train_samples=1000)  # keep ML untrained
    eng = RCAEngine(g, cfg)
    hyp = eng.analyze(event_factory(job="down", ts=100.0))
    assert hyp.category == FC.JOB_DEPENDENCY                       # Eq. (2)
    assert hyp.confidence == pytest.approx(hyp.distribution[hyp.category])
    assert set(hyp.components) == {"G"}


def test_engine_retrains_after_enough_labels(event_factory):
    cfg = ASHDPFConfig(min_train_samples=4, retrain_every=100)
    eng = RCAEngine(AdaptiveKnowledgeGraph(), cfg)
    trained = [eng.add_labelled(event_factory(logs=[l], eid=str(i)), c) for i, (l, c) in enumerate([
        ("Access denied", FC.CREDENTIAL), ("OutOfMemoryError", FC.RESOURCE_EXHAUSTION),
        ("Access denied x", FC.CREDENTIAL), ("OutOfMemoryError y", FC.RESOURCE_EXHAUSTION)])]
    assert trained == [False, False, False, True] and eng.ml.fitted


def test_disabled_rca_falls_back_to_keywords(event_factory):
    eng = RCAEngine(AdaptiveKnowledgeGraph(), ASHDPFConfig(use_rca=False))
    hyp = eng.analyze(event_factory(logs=["Invalid parameter value for foo"]))
    assert hyp.category == FC.CONFIGURATION and "keyword" in hyp.components
