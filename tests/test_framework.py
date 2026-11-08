"""Closed-loop behaviour of ASHDPF (claims 1, 9, 11) with scripted collaborators."""

import pytest

from ashdpf import ASHDPF, ASHDPFConfig, HumanReview, ManualResolution
from ashdpf.execution import CallableConnector
from ashdpf.layers import MonitoringLayer
from ashdpf.models import Decision, ExecutionResult, FailureCategory as FC, Outcome


class ScriptedHuman:
    def __init__(self, approve=True, latency=15.0):
        self.approve, self.latency = approve, latency
        self.reviews, self.resolves = [], []

    def review(self, event, hyp, cand, verdict):
        self.reviews.append((cand.action.name, verdict))
        return HumanReview(self.approve, self.latency, category=event.true_category)

    def resolve(self, event, tried):
        self.resolves.append(list(tried))
        return ManualResolution(None, 60.0, Outcome.SUCCESS, event.true_category)


def make_fw(threshold=0.4, human=None, outcome=Outcome.SUCCESS, minutes=5.0, **kw):
    fw = ASHDPF(ASHDPFConfig(threshold=threshold, min_train_samples=1000, **kw), human=human)
    fw.register_connector(CallableConnector(
        "*", lambda a, e: ExecutionResult(outcome == Outcome.SUCCESS, minutes, outcome)), "*")
    fw.context.load_topology([{"name": "p", "jobs": ["p.up", "p.down"],
                               "depends_on": {"p.down": ["p.up"]}, "owner": "eng"}])
    fw.context.load_runbooks([{"category": "job_dependency", "actions": [
        {"name": "wait_and_retry_upstream", "prior_successes": 20, "prior_attempts": 20}]}])
    return fw


def cascade(fw, event_factory):
    fw.kg.mark_failed("p.up", 95.0)
    return event_factory(job="p.down", pipeline="p", ts=100.0, cat=FC.JOB_DEPENDENCY,
                         logs=["TimeoutError: operation timed out after 3600 seconds"])


def test_autonomous_path_executes_without_human(event_factory):
    human = ScriptedHuman()
    fw = make_fw(human=human)
    rec = fw.handle(cascade(fw, event_factory))
    assert rec.decision == Decision.AUTO_EXECUTE
    assert rec.autonomous and rec.outcome == Outcome.SUCCESS
    assert rec.action.name == "wait_and_retry_upstream"
    assert rec.resolution_minutes == pytest.approx(2.0 + 5.0)  # overhead + execution
    assert human.reviews == [] and len(fw.executor.audit_log) == 1
    assert fw.kg.nodes["p.down"].props["status"] == "healthy"


def test_low_confidence_routes_to_human_approval(event_factory):  # claim 11
    human = ScriptedHuman(latency=15)
    fw = make_fw(threshold=0.99, human=human)
    rec = fw.handle(cascade(fw, event_factory))
    assert rec.decision == Decision.HUMAN_APPROVAL and rec.human_involved and not rec.autonomous
    assert len(human.reviews) == 1
    assert rec.resolution_minutes == pytest.approx(2.0 + 15.0 + 5.0)
    assert fw.executor.audit_log[0].approved_by == "human"


def test_no_human_available_escalates_without_executing(event_factory):
    fw = make_fw(threshold=0.99, human=None)
    rec = fw.handle(cascade(fw, event_factory))
    assert rec.decision == Decision.HUMAN_APPROVAL and rec.outcome == Outcome.FAILURE
    assert fw.executor.audit_log == []


def test_failed_attempts_roll_back_and_retry_then_escalate(event_factory):
    human = ScriptedHuman()
    fw = make_fw(threshold=0.0, human=human, outcome=Outcome.FAILURE, max_attempts=2)
    rec = fw.handle(cascade(fw, event_factory))
    assert rec.attempts == 2 and len(rec.actions_tried) == 2
    assert all(e.rolled_back for e in fw.executor.audit_log)
    assert human.resolves == [rec.actions_tried]           # human gets what was tried
    assert rec.outcome == Outcome.SUCCESS and rec.human_involved


def test_failure_lowers_edge_weight_and_success_raises_it(event_factory):  # claim 9 last step
    fw = make_fw(threshold=0.0, human=ScriptedHuman(), outcome=Outcome.FAILURE, max_attempts=1)
    w0 = fw.kg.fix_stats(FC.JOB_DEPENDENCY, "wait_and_retry_upstream")[2]
    fw.handle(cascade(fw, event_factory))
    assert fw.kg.fix_stats(FC.JOB_DEPENDENCY, "wait_and_retry_upstream")[2] < w0

    fw2 = make_fw(human=ScriptedHuman())
    fw2.kg.get_edge("runbook:job_dependency", "fix:wait_and_retry_upstream",
                    __import__("ashdpf").EdgeType.REFERENCES).weight = 0.7
    fw2.handle(cascade(fw2, event_factory))
    assert fw2.kg.fix_stats(FC.JOB_DEPENDENCY, "wait_and_retry_upstream")[2] == pytest.approx(0.8)


def test_report_contains_governance_metrics(event_factory):
    fw = make_fw(human=ScriptedHuman())
    first = fw.handle(cascade(fw, event_factory))
    second = fw.handle(cascade(fw, lambda **kw: event_factory(eid="e2", **kw)))
    rep = fw.report()
    for k in ("mttr_minutes", "arr", "kri", "audit_entries", "kg_nodes"):
        assert k in rep
    # The first decision relied only on the documented runbook; the second reused
    # an outcome the AOKG had learned from the first -> KRI = 1/2 (claims 8, 12).
    assert not first.used_history and second.used_history
    assert rep["kri"] == pytest.approx(0.5)


def test_monitoring_layer_builds_event_vector():
    fw = make_fw()
    ev = MonitoringLayer(fw.kg).build_event({
        "pipeline": "p", "job": "p.down", "ts": 3, "logs": "line1\n\nline2",
        "quality": {"null_rate": "0.5"}, "true_category": "data_quality"})
    assert ev.vector.logs == ["line1", "line2"]
    assert ev.vector.dependencies == ["p.up"]       # D filled from the AOKG
    assert ev.vector.quality == {"null_rate": 0.5}
    assert ev.true_category == FC.DATA_QUALITY
    with pytest.raises(ValueError):
        MonitoringLayer().build_event({"job": "x"})


def test_handle_signal_end_to_end():
    fw = make_fw(human=ScriptedHuman())
    rec = fw.handle_signal({"pipeline": "p", "job": "p.up", "logs": "Access denied",
                            "true_category": "credential"})
    assert rec.outcome == Outcome.SUCCESS


def test_context_layer_seeds_runbook_weights():
    fw = make_fw()
    s, n, w, real = fw.kg.fix_stats(FC.JOB_DEPENDENCY, "wait_and_retry_upstream")
    assert (s, n, real) == (20, 20, 0) and w == pytest.approx(21 / 22)
    assert fw.kg.get_edge("eng", "p", __import__("ashdpf").EdgeType.RELATED_TO) is not None


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ASHDPF(ASHDPFConfig(beta=(0.5, 0.5, 0.5)))
    with pytest.raises(ValueError):
        ASHDPFConfig(kg_mode="weird").validate()
    with pytest.raises(ValueError):
        ASHDPFConfig(threshold=1.5).validate()
