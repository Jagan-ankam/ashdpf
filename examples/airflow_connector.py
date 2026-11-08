"""Template connector for the Apache Airflow 2.x stable REST API.

Not exercised by the test suite (needs a live Airflow). Endpoint paths follow the
Airflow 2 `/api/v1` API; Airflow 3 serves `/api/v2` -- adjust `API` accordingly.

    from examples.airflow_connector import AirflowConnector
    fw.register_connector(AirflowConnector("https://airflow.example.com", token), "workflow_trigger")
"""

from __future__ import annotations

import json
import time
import urllib.request

from ashdpf.execution import Connector
from ashdpf.models import ExecutionResult, Outcome


class AirflowConnector(Connector):
    name = "workflow_trigger"
    API = "/api/v1"

    def __init__(self, base_url: str, token: str, poll_seconds: int = 30, timeout_minutes: int = 60):
        self.base = base_url.rstrip("/") + self.API
        self.token = token
        self.poll = poll_seconds
        self.timeout = timeout_minutes

    def _call(self, method: str, path: str, body=None):
        req = urllib.request.Request(self.base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read() or b"{}")

    def execute(self, action, event):
        dag_id = event.pipeline
        task_id = event.job.split(".", 1)[-1]
        run_id = event.vector.metadata.get("dag_run_id")
        start = time.time()
        self._call("POST", f"/dags/{dag_id}/clearTaskInstances", {
            "dry_run": False, "task_ids": [task_id], "only_failed": True,
            "dag_run_id": run_id, "include_downstream": True, "reset_dag_runs": True})
        state = None
        while time.time() - start < self.timeout * 60:
            ti = self._call("GET", f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}")
            state = ti.get("state")
            if state in ("success", "failed", "upstream_failed"):
                break
            time.sleep(self.poll)
        minutes = (time.time() - start) / 60
        outcome = Outcome.SUCCESS if state == "success" else Outcome.FAILURE
        return ExecutionResult(outcome == Outcome.SUCCESS, minutes, outcome, {"final_state": state})

    def checkpoint(self, action, event):
        return {"dag_id": event.pipeline, "task": event.job, "run": event.vector.metadata.get("dag_run_id")}
