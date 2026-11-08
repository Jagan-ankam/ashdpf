"""Monitoring layer (110) and Context Acquisition layer (120)."""

from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from .knowledge_graph import AdaptiveKnowledgeGraph, EdgeType, NodeType
from .models import EventVector, FailureCategory, FailureEvent


class MonitoringLayer:
    """Normalises raw signals from orchestrators / warehouses / DQ tools into F.

    A raw signal is a mapping such as an Airflow task-instance callback payload::

        {"pipeline": "sales_daily", "job": "sales_daily.load", "ts": 1712.0,
         "logs": "...", "metadata": {...}, "quality": {...}, "config": {...}}
    """

    REQUIRED = ("pipeline", "job")

    def __init__(self, kg: Optional[AdaptiveKnowledgeGraph] = None) -> None:
        self.kg = kg

    def build_event(self, signal: Mapping[str, Any]) -> FailureEvent:
        missing = [k for k in self.REQUIRED if k not in signal]
        if missing:
            raise ValueError(f"signal missing required fields: {missing}")
        logs = signal.get("logs", [])
        if isinstance(logs, str):
            logs = [ln for ln in logs.splitlines() if ln.strip()]
        deps: List[str] = list(signal.get("dependencies", []))
        if not deps and self.kg is not None and signal["job"] in self.kg.nodes:
            deps = [n for n, _, _ in self.kg.upstream(signal["job"], max_depth=1)]
        true_cat = signal.get("true_category")
        return FailureEvent(
            event_id=str(signal.get("event_id") or uuid.uuid4().hex[:12]),
            pipeline=signal["pipeline"],
            job=signal["job"],
            timestamp=float(signal.get("ts", 0.0)),
            vector=EventVector(
                logs=list(logs),
                metadata=dict(signal.get("metadata", {})),
                quality={k: float(v) for k, v in dict(signal.get("quality", {})).items()},
                dependencies=deps,
                config=dict(signal.get("config", {})),
            ),
            true_category=FailureCategory(true_cat) if true_cat else None,
        )


class ContextAcquisitionLayer:
    """Loads enterprise operational knowledge into the AOKG."""

    def __init__(self, kg: AdaptiveKnowledgeGraph) -> None:
        self.kg = kg

    def load_topology(self, pipelines: Iterable[Mapping[str, Any]]) -> None:
        """``pipelines``: [{"name", "jobs": [..], "depends_on": {job: [upstream jobs]},
        "owner", "application", "tables": {job: table}}]"""
        pipelines = list(pipelines)
        for p in pipelines:
            self.kg.add_node(p["name"], NodeType.PIPELINE, **p.get("props", {}))
            for job in p.get("jobs", []):
                self.kg.add_node(job, NodeType.JOB, pipeline=p["name"])
                self.kg.add_edge(p["name"], job, EdgeType.RELATED_TO)
            for job, table in p.get("tables", {}).items():
                self.kg.add_node(table, NodeType.TABLE)
                self.kg.add_edge(job, table, EdgeType.RELATED_TO)
            if p.get("owner"):
                self.kg.add_node(p["owner"], NodeType.ENGINEER)
                self.kg.add_edge(p["owner"], p["name"], EdgeType.RELATED_TO)
            if p.get("application"):
                self.kg.add_node(p["application"], NodeType.APPLICATION)
                self.kg.add_edge(p["name"], p["application"], EdgeType.RELATED_TO)
        for p in pipelines:  # second pass: cross-pipeline edges need all jobs present
            for job, ups in p.get("depends_on", {}).items():
                for up in ups:
                    if up not in self.kg.nodes:
                        self.kg.add_node(up, NodeType.JOB)
                    self.kg.add_edge(job, up, EdgeType.DEPENDS_ON, weight=1.0)

    def load_runbooks(self, runbooks: Iterable[Mapping[str, Any]]) -> None:
        """``runbooks``: [{"category": "configuration", "actions":
        [{"name": "reset_configuration", "prior_successes": 2, "prior_attempts": 3}]}]

        Documented fixes start with an edge weight equal to their prior success rate.
        """
        for rb in runbooks:
            cat = FailureCategory(rb["category"])
            for a in rb.get("actions", []):
                s = float(a.get("prior_successes", 0))
                n = float(a.get("prior_attempts", 0))
                edge = self.kg.ensure_runbook_fix(cat, a["name"], s, n)
                if n > 0:
                    edge.weight = self.kg._clip((s + 1) / (n + 2))

    def load_incident_history(self, records: Sequence, framework=None) -> None:
        """Replay historical incident records through the learning engine."""
        for rec in records:
            if framework is not None:
                framework.learning.learn(rec)
            else:
                self.kg.ingest_incident(rec)
