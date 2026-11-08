"""Autonomous Execution Framework + Validation Module (patent [0021], claim 7).

Connectors wrap orchestration APIs (job restart, workflow trigger, SQL, cloud admin).
Every execution creates a rollback checkpoint and an audit entry.
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .models import ExecutionResult, FailureEvent, Outcome, RemediationAction


class Connector(ABC):
    name: str = "connector"

    @abstractmethod
    def execute(self, action: RemediationAction, event: FailureEvent) -> ExecutionResult:
        ...

    def checkpoint(self, action: RemediationAction, event: FailureEvent) -> Dict[str, Any]:
        """Capture state needed to undo the action. Override for real systems."""
        return {"job": event.job, "pipeline": event.pipeline, "action": action.name}

    def rollback(self, snapshot: Dict[str, Any]) -> bool:
        return True


class CallableConnector(Connector):
    """Adapter that turns a plain function into a connector (handy for tests/sims)."""

    def __init__(self, name: str, fn: Callable[[RemediationAction, FailureEvent], ExecutionResult],
                 rollback_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> None:
        self.name = name
        self._fn = fn
        self._rollback = rollback_fn

    def execute(self, action, event):
        return self._fn(action, event)

    def rollback(self, snapshot):
        return self._rollback(snapshot) if self._rollback else True


@dataclass
class AuditEntry:
    audit_id: str
    event_id: str
    action: str
    connector: str
    approved_by: str
    checkpoint_id: str
    ok: bool
    duration_minutes: float
    wall_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    rolled_back: bool = False


class ExecutionFramework:
    def __init__(self) -> None:
        self.connectors: Dict[str, Connector] = {}
        self.audit_log: List[AuditEntry] = []
        self.checkpoints: Dict[str, Dict[str, Any]] = {}

    def register(self, connector: Connector, name: Optional[str] = None) -> None:
        self.connectors[name or connector.name] = connector

    def execute(self, action: RemediationAction, event: FailureEvent,
                approved_by: str = "ashdpf-autonomous") -> tuple[ExecutionResult, AuditEntry]:
        conn = self.connectors.get(action.connector) or self.connectors.get("*")
        if conn is None:
            raise KeyError(f"no connector registered for '{action.connector}'")
        cp_id = f"cp-{uuid.uuid4().hex[:12]}"
        self.checkpoints[cp_id] = {"connector": action.connector,
                                   "snapshot": conn.checkpoint(action, event)}
        try:
            result = conn.execute(action, event)
        except Exception as exc:  # connector failures must never crash the loop
            result = ExecutionResult(False, 0.0, Outcome.FAILURE, {"error": repr(exc)})
        entry = AuditEntry(f"au-{uuid.uuid4().hex[:12]}", event.event_id, action.name,
                           action.connector, approved_by, cp_id, result.ok,
                           result.duration_minutes, time.time(), dict(result.details))
        self.audit_log.append(entry)
        return result, entry

    def rollback(self, audit_id: str) -> bool:
        entry = next((e for e in self.audit_log if e.audit_id == audit_id), None)
        if entry is None:
            raise KeyError(audit_id)
        cp = self.checkpoints[entry.checkpoint_id]
        conn = self.connectors.get(cp["connector"]) or self.connectors.get("*")
        ok = bool(conn.rollback(cp["snapshot"]))
        entry.rolled_back = ok
        return ok

    def export_audit(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for e in self.audit_log:
                fh.write(json.dumps(asdict(e), default=str) + "\n")


class ValidationModule:
    """Confirms whether an executed action resolved the failure.

    ``post_check(event, result) -> Outcome`` should re-run the health check
    (e.g. poll task state, row counts). By default it trusts the connector.
    """

    def __init__(self, post_check: Optional[Callable[[FailureEvent, ExecutionResult], Outcome]] = None):
        self.post_check = post_check

    def validate(self, event: FailureEvent, result: ExecutionResult) -> Outcome:
        if self.post_check is not None:
            return Outcome(self.post_check(event, result))
        if result.outcome_hint is not None:
            return result.outcome_hint
        return Outcome.SUCCESS if result.ok else Outcome.FAILURE
