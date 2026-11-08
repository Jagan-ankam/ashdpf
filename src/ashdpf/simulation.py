"""Synthetic enterprise environment for reproducible experiments (paper Sec. IV.B).

* ~500 pipelines in 50 workflow groups, 3-6 jobs each, with intra- and
  cross-pipeline DEPENDS_ON edges (a DAG).
* 8 failure categories. Each pipeline has its own failure-mode mix, and ~18% of
  incidents are cascades (an upstream failure followed by downstream
  job-dependency failures) -- the case graph reasoning is meant to catch.
* Logs are deliberately ambiguous: a share of events carry only generic lines,
  and dependency/network timeouts can share identical messages.
* Ground-truth remediation effectiveness and durations per (category, action).

Every number produced from this module is a *simulation* result, not a
measurement on production data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .execution import CallableConnector
from .config import ASHDPFConfig
from .framework import ASHDPF, HumanReview, ManualResolution
from .models import (ALL_CATEGORIES, EventVector, ExecutionResult, FailureCategory as FC,
                     FailureEvent, IncidentRecord, Outcome, RemediationAction)
from .remediation import ActionLibrary

# (p_success, p_partial, mean_minutes) of the *right* actions per true category.
EFFECTIVENESS: Dict[FC, Dict[str, Tuple[float, float, float]]] = {
    FC.INFRASTRUCTURE: {"restart_workflow": (.88, .05, 8), "rerun_failed_partition": (.75, .10, 12),
                        "restore_dependency_service": (.60, .10, 15)},
    FC.NETWORK_TIMEOUT: {"increase_timeout_and_retry": (.90, .05, 7), "restart_workflow": (.70, .10, 8),
                         "wait_and_retry_upstream": (.45, .10, 10)},
    FC.RESOURCE_EXHAUSTION: {"scale_up_resources": (.92, .03, 12), "restart_workflow": (.30, .10, 8)},
    FC.JOB_DEPENDENCY: {"wait_and_retry_upstream": (.90, .05, 10),
                        "restore_dependency_service": (.75, .10, 15),
                        "rerun_failed_partition": (.55, .10, 12)},
    FC.CONFIGURATION: {"reset_configuration": (.90, .05, 10)},
    FC.DATA_QUALITY: {"quarantine_bad_records": (.80, .12, 15), "reprocess_missing_data": (.70, .15, 25),
                      "rerun_failed_partition": (.25, .15, 12)},
    FC.SCHEMA_EVOLUTION: {"evolve_target_schema": (.85, .08, 20)},
    FC.CREDENTIAL: {"refresh_credentials": (.90, .05, 10)},
}
WRONG_ACTION = (0.03, 0.05, 8.0)

SPECIFIC_LOGS: Dict[FC, List[str]] = {
    FC.INFRASTRUCTURE: ["ExecutorLostFailure: executor 7 exited, host down",
                        "Node lost: instance i-0a3f91c2d7 terminated by cloud provider",
                        "Cluster driver became unresponsive, restarting"],
    FC.NETWORK_TIMEOUT: ["java.net.SocketTimeoutException: Read timed out",
                         "Connection reset by peer while fetching shuffle block",
                         "Temporary failure in name resolution: dns lookup failed"],
    FC.RESOURCE_EXHAUSTION: ["java.lang.OutOfMemoryError: Java heap space",
                             "Container killed by YARN for exceeding memory limits",
                             "No space left on device while writing spill file"],
    FC.JOB_DEPENDENCY: ["Sensor timed out waiting for upstream task",
                        "Upstream dependency not satisfied: partition not available",
                        "ExternalTaskSensor poke limit reached"],
    FC.CONFIGURATION: ["Invalid parameter value for spark.sql.shuffle.partitions",
                       "Unknown option 'compressionCodec' in job config",
                       "Misconfigured warehouse name in connection profile"],
    FC.DATA_QUALITY: ["Expectation failed: null rate on customer_id above threshold",
                      "Duplicate primary key detected in staging table",
                      "Row count dropped 64 percent versus 7 day average"],
    FC.SCHEMA_EVOLUTION: ["AnalysisException: column order_channel not found",
                          "Type mismatch: expected INT got STRING for field qty",
                          "Incompatible schema change detected in source"],
    FC.CREDENTIAL: ["Access denied for user etl_svc: invalid credentials",
                    "HTTP 403 Forbidden: expired token for storage account",
                    "Unauthorized: service principal secret expired"],
}
SHARED_TIMEOUT = "TimeoutError: operation timed out after 3600 seconds"
GENERIC_LOGS = ["Task exited with return code 1", "Job failed after 3 retries",
                "ERROR stage failed, marking run as failed", "Process finished with non-zero status"]
ENGINES = ["spark", "airflow", "redshift", "dbt", "databricks"]
STAGES = ["extract", "transform", "load"]
EXIT_CODES = {FC.RESOURCE_EXHAUSTION: "137", FC.CREDENTIAL: "403", FC.CONFIGURATION: "2"}


@dataclass
class PipelineSpec:
    name: str
    group: int
    jobs: List[str]
    engine: str
    profile: np.ndarray  # probabilities over the 7 non-dependency categories
    depends_on: Dict[str, List[str]] = field(default_factory=dict)


ROOT_CATS = [c for c in ALL_CATEGORIES if c != FC.JOB_DEPENDENCY]


class SyntheticEnvironment:
    def __init__(self, n_pipelines: int = 500, n_groups: int = 50, seed: int = 42,
                 generic_log_rate: float = 0.35, cascade_rate: float = 0.18) -> None:
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.generic_log_rate = generic_log_rate
        self.cascade_rate = cascade_rate
        self.library = ActionLibrary()
        self.pipelines: List[PipelineSpec] = []
        self.job_pipeline: Dict[str, PipelineSpec] = {}
        self.downstream: Dict[str, List[str]] = {}
        self._build(n_pipelines, n_groups)
        self._t = 0.0
        self._seq = 0

    # ------------------------------------------------------------ topology
    def _build(self, n_pipelines: int, n_groups: int) -> None:
        r = self.rng
        per_group = max(1, n_pipelines // n_groups)
        for g in range(n_groups):
            group_pipes: List[PipelineSpec] = []
            for k in range(per_group):
                name = f"pl_{g:02d}_{k:02d}"
                jobs = [f"{name}.t{j}" for j in range(int(r.integers(3, 7)))]
                prof = r.dirichlet(np.full(len(ROOT_CATS), 0.6))
                spec = PipelineSpec(name, g, jobs, str(r.choice(ENGINES)), prof)
                for a, b in zip(jobs[1:], jobs[:-1]):
                    spec.depends_on.setdefault(a, []).append(b)
                if group_pipes and r.random() < 0.6:  # cross-pipeline dependency
                    up = group_pipes[int(r.integers(len(group_pipes)))]
                    spec.depends_on.setdefault(jobs[0], []).append(up.jobs[-1])
                group_pipes.append(spec)
                self.pipelines.append(spec)
                for j in jobs:
                    self.job_pipeline[j] = spec
        for p in self.pipelines:
            for job, ups in p.depends_on.items():
                for u in ups:
                    self.downstream.setdefault(u, []).append(job)

    def topology(self) -> List[dict]:
        return [{"name": p.name, "jobs": p.jobs, "depends_on": p.depends_on,
                 "owner": f"eng_group_{p.group:02d}", "application": f"app_{p.group:02d}",
                 "tables": {j: f"tbl:{j}" for j in p.jobs}, "props": {"engine": p.engine}}
                for p in self.pipelines]

    def runbooks(self) -> List[dict]:
        """Documented runbooks: the first listed fix of each category, 2/3 prior success."""
        out = []
        for cat in ALL_CATEGORIES:
            acts = self.library.for_category(cat)
            out.append({"category": cat.value, "actions": [
                {"name": a.name, "prior_successes": 2 if i == 0 else 0,
                 "prior_attempts": 3 if i == 0 else 0} for i, a in enumerate(acts)
                if a.name != "drop_and_recreate_table"]})
        return out

    # -------------------------------------------------------------- events
    def _event(self, job: str, cat: FC, ts: float) -> FailureEvent:
        r = self.rng
        spec = self.job_pipeline[job]
        self._seq += 1
        if r.random() < self.generic_log_rate:
            logs = [str(r.choice(GENERIC_LOGS))]
        elif cat in (FC.JOB_DEPENDENCY, FC.NETWORK_TIMEOUT) and r.random() < 0.45:
            logs = [SHARED_TIMEOUT]
        else:
            logs = [str(r.choice(SPECIFIC_LOGS[cat]))]
        if r.random() < 0.5:
            logs.append(str(r.choice(GENERIC_LOGS)))
        quality = {"null_rate": float(r.uniform(0, 0.05)), "row_count_delta": float(r.uniform(0, 0.08)),
                   "schema_changed": 0.0}
        if cat == FC.DATA_QUALITY and r.random() < 0.7:
            quality["null_rate"] = float(r.uniform(0.3, 0.9))
            quality["row_count_delta"] = float(r.uniform(0.1, 0.7))
        if cat == FC.SCHEMA_EVOLUTION and r.random() < 0.7:
            quality["schema_changed"] = 1.0
        exit_code = EXIT_CODES.get(cat, "1") if r.random() < 0.6 else "1"
        meta = {"engine": spec.engine, "stage": str(r.choice(STAGES)), "exit_code": exit_code,
                "group": spec.group}
        return FailureEvent(f"ev{self._seq:06d}", spec.name, job, ts,
                            EventVector(logs, meta, quality, list(spec.depends_on.get(job, [])),
                                        {"retries": 3}), cat)

    def generate(self, n: int) -> List[FailureEvent]:
        r = self.rng
        events: List[FailureEvent] = []
        while len(events) < n:
            self._t += float(r.exponential(10.0))
            spec = self.pipelines[int(r.integers(len(self.pipelines)))]
            cat = ROOT_CATS[int(r.choice(len(ROOT_CATS), p=spec.profile))]
            job = str(r.choice(spec.jobs))
            if r.random() < self.cascade_rate:
                roots = [j for j in spec.jobs if self.downstream.get(j)]
                if roots:
                    job = str(r.choice(roots))
                    events.append(self._event(job, cat, self._t))
                    frontier = list(self.downstream[job])
                    for d in frontier[: int(r.integers(1, 3))]:
                        events.append(self._event(d, FC.JOB_DEPENDENCY,
                                                  self._t + float(r.uniform(5, 60))))
                    continue
            events.append(self._event(job, cat, self._t))
        events = events[:n]
        events.sort(key=lambda e: e.timestamp)
        return events

    # ---------------------------------------------------------- execution
    def effectiveness(self, cat: FC, action: str) -> Tuple[float, float, float]:
        return EFFECTIVENESS[cat].get(action, WRONG_ACTION)

    def best_actions(self, cat: FC) -> List[str]:
        return sorted(EFFECTIVENESS[cat], key=lambda a: -EFFECTIVENESS[cat][a][0])

    def execute(self, action: RemediationAction, event: FailureEvent,
                rng: np.random.Generator, slowdown: float = 1.0) -> ExecutionResult:
        p_s, p_p, mean = self.effectiveness(event.true_category, action.name)
        u = rng.random()
        outcome = Outcome.SUCCESS if u < p_s else (Outcome.PARTIAL if u < p_s + p_p else Outcome.FAILURE)
        minutes = float(rng.gamma(4.0, mean / 4.0)) * slowdown
        return ExecutionResult(outcome == Outcome.SUCCESS, minutes, outcome,
                               {"simulated": True})

    def connector(self, rng: np.random.Generator) -> CallableConnector:
        return CallableConnector("*", lambda a, e: self.execute(a, e, rng))


# ------------------------------------------------------------------ humans
class SimulatedOperator:
    """On-call engineer model used by ASHDPF's human-approval path and baselines."""

    def __init__(self, env: SyntheticEnvironment, rng: np.random.Generator,
                 skill: float = 0.85, ack_mean: float = 20.0, review_mean: float = 12.0,
                 investigate_mean: float = 50.0) -> None:
        self.env = env
        self.rng = rng
        self.skill = skill
        self.ack_mean = ack_mean
        self.review_mean = review_mean
        self.investigate_mean = investigate_mean

    def _n(self, mean: float, sd_frac: float = 0.3, lo: float = 2.0) -> float:
        return max(lo, float(self.rng.normal(mean, mean * sd_frac)))

    def review(self, event, hyp, cand, verdict) -> HumanReview:
        cat = event.true_category
        p_s = self.env.effectiveness(cat, cand.action.name)[0]
        latency = self._n(self.ack_mean * 0.6) + self._n(self.review_mean)
        if p_s >= 0.5:
            return HumanReview(True, latency, category=cat, approver="sim-oncall")
        # Diagnosis was wrong: investigate and pick a fix manually.
        latency += self._n(self.investigate_mean * 0.6)
        action = self._pick(cat)
        return HumanReview(False, latency, action=action, category=cat, approver="sim-oncall")

    def _pick(self, cat: FC, exclude: Sequence[str] = ()) -> RemediationAction:
        options = [a for a in self.env.best_actions(cat) if a not in exclude] or self.env.best_actions(cat)
        if self.rng.random() < self.skill or len(options) == 1:
            name = options[0]
        else:
            others = [a.name for a in self.env.library
                      if a.name not in exclude and a.name != options[0]
                      and a.name != "drop_and_recreate_table"]
            name = str(self.rng.choice(others))
        return self.env.library[name]

    def resolve(self, event, tried: Sequence[str], acked: bool = True) -> ManualResolution:
        cat = event.true_category
        minutes = 0.0 if acked else self._n(self.ack_mean)
        minutes += self._n(self.investigate_mean)
        excluded = list(tried)
        action = None
        outcome = Outcome.FAILURE
        for _ in range(4):
            action = self._pick(cat, excluded)
            res = self.env.execute(action, event, self.rng, slowdown=1.5)  # manual toil
            minutes += res.duration_minutes
            outcome = res.outcome_hint
            if outcome == Outcome.SUCCESS:
                break
            excluded.append(action.name)
            minutes += self._n(20.0)  # re-investigation
        if outcome != Outcome.SUCCESS:  # escalate until fixed (bounded)
            minutes += self._n(60.0)
            outcome = Outcome.SUCCESS
        return ManualResolution(action, minutes, outcome, cat, engineer=f"eng_group_{event.vector.metadata.get('group', 0):02d}")


def historical_records(env: SyntheticEnvironment, events: Sequence[FailureEvent],
                       rng: np.random.Generator) -> List[IncidentRecord]:
    """Build the historical incident repository (manually resolved incidents)."""
    op = SimulatedOperator(env, rng)
    recs = []
    for ev in events:
        res = op.resolve(ev, [], acked=False)
        recs.append(IncidentRecord(ev, None, res.action, None, res.outcome, res.minutes, True,
                                   False, 1, res.category, [res.action.name] if res.action else []))
    return recs


def build_ashdpf(env: SyntheticEnvironment, history: Sequence[IncidentRecord],
                 cfg: Optional[ASHDPFConfig] = None, seed: int = 4) -> ASHDPF:
    """Wire an ASHDPF instance to the synthetic environment and bootstrap it."""
    rng = np.random.default_rng(seed)
    fw = ASHDPF(cfg or ASHDPFConfig(), library=env.library,
                human=SimulatedOperator(env, rng))
    fw.register_connector(env.connector(rng), "*")
    fw.context.load_topology(env.topology())
    fw.context.load_runbooks(env.runbooks())
    fw.bootstrap(history)
    return fw
