"""Synthetic environment, baselines, full closed loop and experiment protocol."""

import json
from collections import Counter

import pytest

from ashdpf.baselines import AIOpsRecommendationBaseline, MonitoringOnlyBaseline, RuleBasedBaseline
from ashdpf.cli import main
from ashdpf.config import ASHDPFConfig
from ashdpf.experiments import make_world, run_all
from ashdpf.metrics import summarize
from ashdpf.models import ALL_CATEGORIES, FailureCategory as FC
from ashdpf.simulation import SyntheticEnvironment, build_ashdpf


def test_environment_topology_is_a_dag():
    env = SyntheticEnvironment(n_pipelines=40, n_groups=4, seed=1)
    fw = build_ashdpf(env, [], ASHDPFConfig(min_train_samples=10_000))
    assert len(env.pipelines) == 40
    assert not fw.kg.has_cycle()


def test_events_cover_all_categories_and_are_time_ordered(small_world):
    _, _, events = small_world
    assert len(events) == 500
    assert set(Counter(e.true_category for e in events)) == set(ALL_CATEGORIES)
    ts = [e.timestamp for e in events]
    assert ts == sorted(ts)


def test_dependency_events_follow_an_upstream_failure():
    env = SyntheticEnvironment(n_pipelines=60, n_groups=6, seed=3)
    events = env.generate(400)
    last_fail = {}
    ok = total = 0
    for e in events:
        if e.true_category == FC.JOB_DEPENDENCY:
            total += 1
            ups = env.job_pipeline[e.job].depends_on.get(e.job, [])
            ok += any(u in last_fail and e.timestamp - last_fail[u] <= 60 for u in ups)
        last_fail[e.job] = e.timestamp
    assert total > 0 and ok / total > 0.9


def test_generation_is_deterministic():
    a = [(e.job, e.true_category, e.vector.logs[0]) for e in SyntheticEnvironment(seed=5).generate(50)]
    b = [(e.job, e.true_category, e.vector.logs[0]) for e in SyntheticEnvironment(seed=5).generate(50)]
    assert a == b


@pytest.fixture(scope="module")
def runs():
    env, hist, events = make_world(seed=11, n_history=200, n_events=500, n_pipelines=60, n_groups=6)
    cfg = ASHDPFConfig(retrain_every=100, seed=11)
    proposed = build_ashdpf(env, hist, cfg, seed=5).run(events)
    return {
        "env": env, "events": events,
        "proposed": proposed,
        "manual": MonitoringOnlyBaseline(env, 1).run(events),
        "rules": RuleBasedBaseline(env, 2).run(events),
        "aiops": AIOpsRecommendationBaseline(env, hist, 3, retrain_every=100).run(events),
    }


def test_every_incident_gets_resolved(runs):
    for name in ("proposed", "manual", "rules", "aiops"):
        assert all(r.outcome.value == "success" for r in runs[name]), name


def test_proposed_beats_all_baselines_on_mttr(runs):  # RQ2
    m = {k: summarize(runs[k])["mttr_minutes"] for k in ("proposed", "manual", "rules", "aiops")}
    assert m["proposed"] < m["rules"] < m["manual"]
    assert m["proposed"] < m["aiops"] < m["manual"]


def test_proposed_resolves_a_majority_autonomously(runs):  # RQ5
    s = summarize(runs["proposed"])
    assert s["arr"] > 0.4
    assert summarize(runs["manual"])["arr"] == 0.0
    assert summarize(runs["aiops"])["arr"] == 0.0


def test_hybrid_rca_beats_keyword_rules(runs):  # RQ1
    assert summarize(runs["proposed"])["accuracy"] > summarize(runs["rules"])["accuracy"] + 0.1


def test_credential_incidents_stay_human_gated(runs):  # governance (Table IV)
    cred = [r for r in runs["proposed"] if r.event.true_category == FC.CREDENTIAL]
    infra = [r for r in runs["proposed"] if r.event.true_category == FC.INFRASTRUCTURE]
    arr = lambda rs: sum(r.autonomous for r in rs) / len(rs)
    assert arr(cred) < arr(infra)


def test_learning_improves_over_time(runs):  # RQ4
    recs = runs["proposed"]
    early, late = summarize(recs[:150]), summarize(recs[-150:])
    assert late["kri"] >= early["kri"]
    assert late["accuracy"] >= early["accuracy"] - 0.02


def test_audit_trail_matches_executions():
    env, hist, events = make_world(seed=2, n_history=100, n_events=80, n_pipelines=30, n_groups=3)
    fw = build_ashdpf(env, hist, ASHDPFConfig(seed=2))
    recs = fw.run(events)
    assert len(fw.executor.audit_log) == sum(r.attempts for r in recs)
    assert {a.audit_id for a in fw.executor.audit_log} == {i for r in recs for i in r.audit_ids}


def test_same_seed_same_results():
    def go():
        env, hist, events = make_world(seed=9, n_history=80, n_events=120, n_pipelines=30, n_groups=3)
        recs = build_ashdpf(env, hist, ASHDPFConfig(seed=9), seed=1).run(events)
        return [(r.hypothesis.category, r.action.name if r.action else None,
                 round(r.resolution_minutes, 6)) for r in recs]
    assert go() == go()


@pytest.mark.slow
def test_experiment_protocol_quick(tmp_path):
    res = run_all(n_events=200, n_history=100, seed=3, workers=1, out_dir=str(tmp_path),
                  n_pipelines=30, n_groups=3, quick=True)
    for key in ("exp1_rca", "exp2_mttr", "statistics", "exp3_kg", "exp4_learning",
                "exp5_arr_by_category", "ablation"):
        assert res[key], key
    assert set(res["ablation"]) == {"Full Framework", "Without AOKG", "Without RCA Engine",
                                    "Without Learning Engine", "Without Remediation Planner"}
    assert 0 <= res["statistics"]["Manual Operations"]["wilcoxon_p"] <= 1
    md = (tmp_path / "RESULTS.md").read_text()
    assert "Table I" in md and "Table VI" in md
    assert json.loads((tmp_path / "results.json").read_text())["meta"]["n_events"] == 200


def test_cli_simulate(tmp_path, capsys):
    audit, graph = tmp_path / "a.jsonl", tmp_path / "g.json"
    assert main(["simulate", "--events", "60", "--history", "60", "--pipelines", "20",
                 "--groups", "2", "--audit", str(audit), "--graph", str(graph)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["incidents"] == 60 and audit.exists() and graph.exists()
