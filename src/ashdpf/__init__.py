"""ASHDPF -- Autonomous Self-Healing Data Pipeline Framework.

Reference implementation of the architecture described in the ASHDPF paper and
Indian patent application: Adaptive Operational Knowledge Graph, hybrid Root
Cause Analysis Engine, Intelligent Remediation Planner with a Safety Validation
Layer, Autonomous Execution Framework and a reinforcement-learning Continuous
Learning Engine.
"""

from .config import ASHDPFConfig
from .framework import ASHDPF, HumanReview, ManualResolution
from .knowledge_graph import AdaptiveKnowledgeGraph, EdgeType, NodeType
from .models import (Decision, EventVector, FailureCategory, FailureEvent, IncidentRecord,
                     Outcome, RemediationAction)

__version__ = "0.1.0"

__all__ = [
    "ASHDPF", "ASHDPFConfig", "AdaptiveKnowledgeGraph", "EdgeType", "NodeType",
    "Decision", "EventVector", "FailureCategory", "FailureEvent", "IncidentRecord",
    "Outcome", "RemediationAction", "HumanReview", "ManualResolution", "__version__",
]
