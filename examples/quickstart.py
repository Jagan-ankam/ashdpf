"""Minimal ASHDPF usage outside the simulator.

    python examples/quickstart.py
"""

from ashdpf import ASHDPF, ASHDPFConfig, HumanReview, ManualResolution, Outcome
from ashdpf.execution import CallableConnector
from ashdpf.models import ExecutionResult


class ConsoleOperator:
    """Human-in-the-loop hook. In production: Slack/PagerDuty approval request."""

    def review(self, event, hyp, cand, verdict):
        print(f"  [approval needed] {cand.action.name} for {hyp.category.value} "
              f"(confidence {verdict.confidence:.2f} vs T={verdict.threshold}): {verdict.reasons}")
        return HumanReview(approved=True, latency_minutes=10, category=hyp.category)

    def resolve(self, event, tried):
        print(f"  [manual] engineer takes over after trying {tried}")
        return ManualResolution(None, 45, Outcome.SUCCESS, None)


def restart_job(action, event):
    print(f"  -> executing {action.name} on {event.job}")
    return ExecutionResult(ok=True, duration_minutes=4.0)


fw = ASHDPF(ASHDPFConfig(threshold=0.4), human=ConsoleOperator())
fw.register_connector(CallableConnector("*", restart_job), "*")

fw.context.load_topology([
    {"name": "sales_daily", "jobs": ["sales_daily.extract", "sales_daily.load"],
     "depends_on": {"sales_daily.load": ["sales_daily.extract"]}, "owner": "data-eng"},
])
fw.context.load_runbooks([
    {"category": "job_dependency",
     "actions": [{"name": "wait_and_retry_upstream", "prior_successes": 9, "prior_attempts": 10}]},
    {"category": "credential",
     "actions": [{"name": "refresh_credentials", "prior_successes": 5, "prior_attempts": 5}]},
])

signals = [
    {"pipeline": "sales_daily", "job": "sales_daily.extract", "ts": 0,
     "logs": "HTTP 403 Forbidden: expired token for storage account"},
    {"pipeline": "sales_daily", "job": "sales_daily.load", "ts": 12,
     "logs": "Sensor timed out waiting for upstream task"},
]
for sig in signals:
    print(f"failure in {sig['job']}")
    rec = fw.handle_signal(sig)
    print(f"  root cause={rec.hypothesis.category.value} ({rec.hypothesis.confidence:.2f}) "
          f"action={rec.action.name if rec.action else None} decision={rec.decision.value} "
          f"outcome={rec.outcome.value} minutes={rec.resolution_minutes:.1f}")

print("report:", {k: round(v, 3) for k, v in fw.report().items()})
