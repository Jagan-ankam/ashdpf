"""Core data structures shared by every ASHDPF component."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FailureCategory(str, Enum):
    """The eight failure categories of the synthetic dataset (paper, Sec. IV.B)."""

    INFRASTRUCTURE = "infrastructure"
    DATA_QUALITY = "data_quality"
    SCHEMA_EVOLUTION = "schema_evolution"
    JOB_DEPENDENCY = "job_dependency"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION = "configuration"
    CREDENTIAL = "credential"
    NETWORK_TIMEOUT = "network_timeout"


ALL_CATEGORIES: List[FailureCategory] = list(FailureCategory)


class Outcome(str, Enum):
    """Outcome classification recorded by the Validation Module (patent [0021])."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"


class Decision(str, Enum):
    """Result of the Safety Validation Layer."""

    AUTO_EXECUTE = "auto_execute"
    HUMAN_APPROVAL = "human_approval"
    REJECTED = "rejected"


@dataclass
class EventVector:
    """Contextual event vector E = {L, M, Q, D, C} (patent [0017])."""

    logs: List[str] = field(default_factory=list)          # L
    metadata: Dict[str, Any] = field(default_factory=dict)  # M
    quality: Dict[str, float] = field(default_factory=dict)  # Q
    dependencies: List[str] = field(default_factory=list)  # D
    config: Dict[str, Any] = field(default_factory=dict)    # C


@dataclass
class FailureEvent:
    """A detected pipeline failure F."""

    event_id: str
    pipeline: str
    job: str
    timestamp: float  # minutes since epoch of the simulation / unix minutes
    vector: EventVector
    # Ground truth, known only in simulation or after post-mortem confirmation.
    true_category: Optional[FailureCategory] = None


@dataclass(frozen=True)
class RemediationAction:
    """An entry in the configurable action library."""

    name: str
    applicable_to: frozenset  # of FailureCategory
    risk: float               # 0 (harmless) .. 1 (dangerous)
    reversible: bool = True
    connector: str = "job_restart"
    description: str = ""


@dataclass
class RootCauseHypothesis:
    """Output of the RCA engine: RC = argmax_i P(RC_i | F) with C_RCA (Eq. 2, 3)."""

    category: FailureCategory
    confidence: float
    distribution: Dict[FailureCategory, float]
    components: Dict[str, Dict[FailureCategory, float]]
    max_history_similarity: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class ScoredAction:
    """A candidate remediation with its RES(A_i) decomposition (Eq. 4)."""

    action: RemediationAction
    similarity: float        # S
    success_rate: float      # H
    risk: float              # R
    res: float               # gamma1*S + gamma2*H + gamma3*R
    policy_prob: float       # pi(a | s) from the Continuous Learning Engine
    edge_weight: float       # adaptive AOKG weight W_e of (runbook -> fix)
    rank_score: float        # final ranking score
    uses_history: bool = False
    category: Optional[FailureCategory] = None  # root cause this action targets


@dataclass
class SafetyVerdict:
    decision: Decision
    confidence: float
    threshold: float
    checks: Dict[str, bool]
    reasons: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    ok: bool
    duration_minutes: float
    outcome_hint: Optional[Outcome] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IncidentRecord:
    """Everything that happened while handling one failure event."""

    event: FailureEvent
    hypothesis: Optional[RootCauseHypothesis]
    action: Optional[RemediationAction]
    decision: Optional[Decision]
    outcome: Outcome
    resolution_minutes: float
    human_involved: bool
    used_history: bool
    attempts: int = 1
    confirmed_category: Optional[FailureCategory] = None
    actions_tried: List[str] = field(default_factory=list)
    audit_ids: List[str] = field(default_factory=list)

    @property
    def autonomous(self) -> bool:
        return self.outcome == Outcome.SUCCESS and not self.human_involved
