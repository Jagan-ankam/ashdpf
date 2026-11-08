"""Experiment protocol of paper Sec. IV (Experiments 1-5, ablation, statistics).

Run:  python -m ashdpf.cli experiments --events 10000 --out results
All figures are produced on the synthetic environment in ``simulation.py``.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

from .baselines import AIOpsRecommendationBaseline, MonitoringOnlyBaseline, RuleBasedBaseline
from .config import ASHDPFConfig
from .metrics import classification_report, summarize
from .models import ALL_CATEGORIES, IncidentRecord
from .rca import MLClassifier
from .simulation import SyntheticEnvironment, build_ashdpf, historical_records


def make_world(seed: int, n_history: int, n_events: int, n_pipelines: int = 500, n_groups: int = 50):
    env = SyntheticEnvironment(n_pipelines=n_pipelines, n_groups=n_groups, seed=seed)
    hist = historical_records(env, env.generate(n_history), np.random.default_rng(seed + 1000))
    events = env.generate(n_events)
    return env, hist, events


def _run_config(args) -> Tuple[str, List[IncidentRecord]]:
    name, cfg, seed, n_history, n_events, world_kw = args
    env, hist, events = make_world(seed, n_history, n_events, **world_kw)
    fw = build_ashdpf(env, hist, cfg, seed=seed + 7)
    return name, fw.run(events)


def _slim(records: Sequence[IncidentRecord]) -> List[dict]:
    return [{
        "true": r.event.true_category.value if r.event.true_category else None,
        "pred": r.hypothesis.category.value if r.hypothesis else None,
        "minutes": r.resolution_minutes, "human": r.human_involved,
        "auto": r.autonomous, "used_history": r.used_history,
        "attempts": r.attempts, "action": r.action.name if r.action else None,
    } for r in records]


def run_all(n_events: int = 10000, n_history: int = 500, seed: int = 42, workers: int = 2,
            out_dir: str = "results", n_pipelines: int = 500, n_groups: int = 50,
            quick: bool = False) -> Dict:
    t0 = time.time()
    world_kw = {"n_pipelines": n_pipelines, "n_groups": n_groups}
    base = ASHDPFConfig(retrain_every=max(200, n_events // 20), seed=seed)
    configs = {
        "full": base,
        "kg_none": replace(base, kg_mode="none"),
        "kg_static": replace(base, kg_mode="static"),
        "no_rca": replace(base, use_rca=False),
        "no_learning": replace(base, use_learning=False, kg_mode="static"),
        "no_planner": replace(base, use_planner=False),
    }
    thresholds = [] if quick else [0.3, 0.5, 0.6, 0.7]
    for T in thresholds:
        configs[f"T={T}"] = replace(base, threshold=T)
    jobs = [(k, v, seed, n_history, n_events, world_kw) for k, v in configs.items()]
    runs: Dict[str, List[IncidentRecord]] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name, recs in pool.map(_run_config, jobs):
                runs[name] = recs
    else:
        for j in jobs:
            name, recs = _run_config(j)
            runs[name] = recs

    env, hist, events = make_world(seed, n_history, n_events, **world_kw)
    baselines = {
        "Manual Operations": MonitoringOnlyBaseline(env, seed + 1).run(events),
        "Rule-Based Automation": RuleBasedBaseline(env, seed + 2).run(events),
        "AIOps Recommendations": AIOpsRecommendationBaseline(
            env, hist, seed + 3, retrain_every=max(200, n_events // 20)).run(events),
    }
    full = runs["full"]
    split = int(0.7 * n_events)
    res: Dict = {"meta": {"n_events": n_events, "n_history": n_history, "seed": seed,
                          "n_pipelines": n_pipelines, "n_groups": n_groups,
                          "config": {k: str(v) for k, v in vars(base).items()}}}

    # ---- Experiment 1: RCA accuracy (train on history + first 70%, test last 30%)
    train_ev = [r.event for r in hist] + list(events[:split])
    train_y = [r.confirmed_category for r in hist] + [e.true_category for e in events[:split]]
    test_ev = events[split:]
    test_y = [e.true_category for e in test_ev]
    exp1 = {}
    for label, model in (("Random Forest", "random_forest"), ("Gradient Boosting (XGBoost-style)", "gradient_boosting")):
        clf = MLClassifier(model=model, n_estimators=100, seed=seed).fit(train_ev, train_y)
        exp1[label] = classification_report(test_y, clf.predict(test_ev))
    exp1["Proposed Hybrid RCA (online)"] = classification_report(
        test_y, [r.hypothesis.category for r in full[split:]])
    res["exp1_rca"] = exp1

    # ---- Experiment 2: MTTR + statistics
    systems = dict(baselines)
    systems["Proposed Framework"] = full
    exp2 = {k: summarize(v) for k, v in systems.items()}
    manual = exp2["Manual Operations"]["mttr_minutes"]
    for k in exp2:
        exp2[k]["mttr_reduction_vs_manual_pct"] = 100 * (manual - exp2[k]["mttr_minutes"]) / manual
    res["exp2_mttr"] = exp2
    prop_t = np.array([r.resolution_minutes for r in full])
    sig = {}
    for k, v in baselines.items():
        b = np.array([r.resolution_minutes for r in v])
        t = stats.ttest_rel(prop_t, b)
        w = stats.wilcoxon(prop_t, b)
        sig[k] = {"paired_t": float(t.statistic), "paired_t_p": float(t.pvalue),
                  "wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue),
                  "mean_diff_minutes": float(np.mean(prop_t - b))}
    f = stats.f_oneway(*[[r.resolution_minutes for r in v] for v in systems.values()])
    sig["ANOVA (all four systems)"] = {"F": float(f.statistic), "p": float(f.pvalue)}
    res["statistics"] = sig

    # ---- Experiment 3: knowledge graph impact
    res["exp3_kg"] = {name: summarize(runs[key]) for name, key in
                      (("No Graph", "kg_none"), ("Static Graph", "kg_static"), ("Adaptive Graph", "full"))}

    # ---- Experiment 4: learning progression (epoch = 1/50 of the stream)
    block = max(1, n_events // 50)
    exp4 = {}
    for ep in (1, 5, 10, 20, 50):
        chunk = full[(ep - 1) * block: ep * block]
        if chunk:
            exp4[str(ep)] = summarize(chunk)
    res["exp4_learning"] = exp4

    # ---- Experiment 5: ARR by category
    exp5 = {}
    for cat in ALL_CATEGORIES:
        sub = [r for r in full if r.event.true_category == cat]
        if sub:
            exp5[cat.value] = {"incidents": len(sub), "arr": sum(r.autonomous for r in sub) / len(sub),
                               "mttr_minutes": float(np.mean([r.resolution_minutes for r in sub]))}
    res["exp5_arr_by_category"] = exp5

    # ---- Ablation
    res["ablation"] = {name: summarize(runs[key]) for name, key in (
        ("Full Framework", "full"), ("Without AOKG", "kg_none"), ("Without RCA Engine", "no_rca"),
        ("Without Learning Engine", "no_learning"), ("Without Remediation Planner", "no_planner"))}

    # ---- Threshold sensitivity
    sens = {}
    for T in [0.3, 0.4, 0.5, 0.6, 0.7]:
        key = "full" if abs(T - base.threshold) < 1e-9 else f"T={T}"
        if key in runs:
            recs = runs[key]
            auto_first = [r for r in recs if r.decision is not None and r.decision.value == "auto_execute"]
            wrong = sum(1 for r in auto_first if r.attempts > 1 or r.human_involved)
            s = summarize(recs)
            s["auto_first_attempt_failure_rate"] = wrong / len(auto_first) if auto_first else 0.0
            sens[str(T)] = s
    res["threshold_sensitivity"] = sens
    res["meta"]["runtime_seconds"] = time.time() - t0

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        with open(os.path.join(out_dir, "RESULTS.md"), "w") as fh:
            fh.write(render_markdown(res))
        with open(os.path.join(out_dir, "incidents_proposed.jsonl"), "w") as fh:
            for row in _slim(full):
                fh.write(json.dumps(row) + "\n")
    return res


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_markdown(res: Dict) -> str:
    m = res["meta"]
    L = ["# ASHDPF experimental results (synthetic environment)\n",
         f"Measured by `python -m ashdpf.cli experiments` on {m['n_events']:,} synthetic failure events "
         f"across {m['n_pipelines']} pipelines / {m['n_groups']} workflow groups, after bootstrapping "
         f"from {m['n_history']} historical incidents (seed {m['seed']}). These are simulation "
         f"results, not production measurements.\n"]

    L += ["## Table I - Root cause analysis (test = last 30% of stream)\n",
          "| Method | Accuracy | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for k, v in res["exp1_rca"].items():
        L.append(f"| {k} | {_pct(v['accuracy'])} | {_pct(v['precision'])} | {_pct(v['recall'])} | {_pct(v['f1'])} |")

    L += ["\n## Table II - Mean time to resolution\n",
          "| Method | MTTR (min) | Reduction vs manual | ARR |", "|---|---|---|---|"]
    for k, v in res["exp2_mttr"].items():
        L.append(f"| {k} | {v['mttr_minutes']:.1f} | {v['mttr_reduction_vs_manual_pct']:.1f}% | {_pct(v['arr'])} |")

    L += ["\n## Statistical significance (per-incident resolution time, paired)\n",
          "| Comparison | paired t | p | Wilcoxon W | p | mean diff (min) |", "|---|---|---|---|---|---|"]
    for k, v in res["statistics"].items():
        if "F" in v:
            continue
        L.append(f"| Proposed vs {k} | {v['paired_t']:.2f} | {v['paired_t_p']:.2e} | "
                 f"{v['wilcoxon_W']:.0f} | {v['wilcoxon_p']:.2e} | {v['mean_diff_minutes']:.1f} |")
    a = res["statistics"]["ANOVA (all four systems)"]
    L.append(f"\nOne-way ANOVA across the four systems: F = {a['F']:.1f}, p = {a['p']:.2e}.")

    L += ["\n## Table III - Knowledge graph impact\n",
          "| Configuration | RCA accuracy | MTTR (min) | ARR |", "|---|---|---|---|"]
    for k, v in res["exp3_kg"].items():
        L.append(f"| {k} | {_pct(v['accuracy'])} | {v['mttr_minutes']:.1f} | {_pct(v['arr'])} |")

    L += ["\n## Table IV - Autonomous resolution rate by failure category\n",
          "| Category | Incidents | ARR | MTTR (min) |", "|---|---|---|---|"]
    for k, v in sorted(res["exp5_arr_by_category"].items(), key=lambda kv: -kv[1]["arr"]):
        L.append(f"| {k} | {v['incidents']} | {_pct(v['arr'])} | {v['mttr_minutes']:.1f} |")

    L += ["\n## Table V - Learning progression (epoch = 1/50 of the stream)\n",
          "| Epoch | Accuracy | MTTR (min) | ARR | KRI |", "|---|---|---|---|---|"]
    for k, v in res["exp4_learning"].items():
        L.append(f"| {k} | {_pct(v['accuracy'])} | {v['mttr_minutes']:.1f} | {_pct(v['arr'])} | {_pct(v['kri'])} |")

    L += ["\n## Table VI - Ablation\n",
          "| Configuration | Accuracy | MTTR (min) | ARR |", "|---|---|---|---|"]
    for k, v in res["ablation"].items():
        L.append(f"| {k} | {_pct(v['accuracy'])} | {v['mttr_minutes']:.1f} | {_pct(v['arr'])} |")

    if res.get("threshold_sensitivity"):
        L += ["\n## Safety threshold T sensitivity\n",
              "| T | ARR | MTTR (min) | auto first-attempt failure rate |", "|---|---|---|---|"]
        for k, v in res["threshold_sensitivity"].items():
            L.append(f"| {k} | {_pct(v['arr'])} | {v['mttr_minutes']:.1f} | "
                     f"{_pct(v['auto_first_attempt_failure_rate'])} |")
    L.append(f"\n_Runtime: {m['runtime_seconds']:.0f} s._\n")
    return "\n".join(L)
