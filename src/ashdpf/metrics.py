"""Evaluation metrics (paper Sec. IV.D): accuracy, P/R/F1, MTTR, ARR, KRI."""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Sequence

from .models import IncidentRecord


def classification_report(y_true: Sequence[Hashable], y_pred: Sequence[Hashable]) -> Dict[str, float]:
    """Accuracy plus macro-averaged precision, recall and F1."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    labels = sorted(set(y_true) | set(y_pred), key=str)
    precs: List[float] = []
    recs: List[float] = []
    f1s: List[float] = []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        if tp + fn == 0:  # label only predicted, never true: skip from macro recall
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        precs.append(p)
        recs.append(r)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    n = len(precs)
    return {"accuracy": acc, "precision": sum(precs) / n, "recall": sum(recs) / n,
            "f1": sum(f1s) / n}


def mttr(records: Iterable[IncidentRecord]) -> float:
    vals = [r.resolution_minutes for r in records]
    return sum(vals) / len(vals) if vals else 0.0


def autonomous_resolution_rate(records: Iterable[IncidentRecord]) -> float:
    recs = list(records)
    return sum(1 for r in recs if r.autonomous) / len(recs) if recs else 0.0


def knowledge_reuse_index(records: Iterable[IncidentRecord]) -> float:
    """KRI = decisions drawing on knowledge already in the AOKG / all decisions."""
    recs = [r for r in records if r.action is not None]
    return sum(1 for r in recs if r.used_history) / len(recs) if recs else 0.0


def summarize(records: Sequence[IncidentRecord]) -> Dict[str, float]:
    y_true = [r.event.true_category for r in records if r.event.true_category and r.hypothesis]
    y_pred = [r.hypothesis.category for r in records if r.event.true_category and r.hypothesis]
    out = classification_report(y_true, y_pred) if y_true else {}
    out.update({
        "incidents": float(len(records)),
        "mttr_minutes": mttr(records),
        "arr": autonomous_resolution_rate(records),
        "kri": knowledge_reuse_index(records),
        "success_rate": (sum(1 for r in records if r.outcome.value == "success") / len(records)
                         if records else 0.0),
    })
    return out
