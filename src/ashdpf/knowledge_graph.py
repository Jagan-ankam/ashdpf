"""Adaptive Operational Knowledge Graph (AOKG) -- paper Sec. III.C, patent [0018].

G = (V, E) with typed nodes and typed, weighted edges. Edge weights evolve with
remediation outcomes according to Eq. (1):  W_e <- W_e + alpha * R.

The in-memory implementation has no external dependencies; ``to_cypher`` exports
the graph as Neo4j MERGE statements for the production knowledge layer.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .models import FailureCategory, IncidentRecord, Outcome


class NodeType(str, Enum):
    PIPELINE = "Pipeline"
    JOB = "Job"
    TABLE = "Table"
    INCIDENT = "Incident"
    RUNBOOK = "Runbook"
    FIX = "Fix"
    ENGINEER = "Engineer"
    APPLICATION = "Application"


class EdgeType(str, Enum):
    DEPENDS_ON = "DEPENDS_ON"
    FAILED_BY = "FAILED_BY"
    FIXED_BY = "FIXED_BY"
    REFERENCES = "REFERENCES"
    RELATED_TO = "RELATED_TO"


@dataclass
class Node:
    id: str
    type: NodeType
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    type: EdgeType
    weight: float
    props: Dict[str, Any] = field(default_factory=dict)


EdgeKey = Tuple[str, EdgeType, str]


def runbook_id(category: FailureCategory) -> str:
    return f"runbook:{FailureCategory(category).value}"


def fix_id(action_name: str) -> str:
    return f"fix:{action_name}"


class AdaptiveKnowledgeGraph:
    def __init__(
        self,
        alpha: float = 0.1,
        initial_weight: float = 0.5,
        bounds: Tuple[float, float] = (0.0, 1.0),
        adaptive: bool = True,
    ) -> None:
        self.alpha = alpha
        self.initial_weight = initial_weight
        self.bounds = bounds
        self.adaptive = adaptive
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[EdgeKey, Edge] = {}
        self._out: Dict[str, Set[EdgeKey]] = {}
        self._in: Dict[str, Set[EdgeKey]] = {}
        self._incident_seq = 0

    # ------------------------------------------------------------------ basics
    def add_node(self, node_id: str, node_type: NodeType, **props: Any) -> Node:
        node = self.nodes.get(node_id)
        if node is None:
            node = Node(node_id, NodeType(node_type), dict(props))
            self.nodes[node_id] = node
            self._out.setdefault(node_id, set())
            self._in.setdefault(node_id, set())
        else:
            if node.type != NodeType(node_type):
                raise ValueError(f"node {node_id} already exists with type {node.type}")
            node.props.update(props)
        return node

    def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: EdgeType,
        weight: Optional[float] = None,
        **props: Any,
    ) -> Edge:
        for nid in (src, dst):
            if nid not in self.nodes:
                raise KeyError(f"unknown node {nid}")
        key = (src, EdgeType(edge_type), dst)
        edge = self.edges.get(key)
        if edge is None:
            edge = Edge(src, dst, EdgeType(edge_type),
                        self.initial_weight if weight is None else self._clip(weight), dict(props))
            self.edges[key] = edge
            self._out[src].add(key)
            self._in[dst].add(key)
        else:
            edge.props.update(props)
            if weight is not None:
                edge.weight = self._clip(weight)
        return edge

    def get_edge(self, src: str, dst: str, edge_type: EdgeType) -> Optional[Edge]:
        return self.edges.get((src, EdgeType(edge_type), dst))

    def out_edges(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[Edge]:
        keys = self._out.get(node_id, set())
        return [self.edges[k] for k in keys if edge_type is None or k[1] == edge_type]

    def in_edges(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[Edge]:
        keys = self._in.get(node_id, set())
        return [self.edges[k] for k in keys if edge_type is None or k[1] == edge_type]

    def _clip(self, w: float) -> float:
        lo, hi = self.bounds
        return max(lo, min(hi, w))

    # ------------------------------------------------------------ Equation (1)
    def update_edge_weight(
        self, src: str, dst: str, edge_type: EdgeType, outcome_score: float,
        alpha: Optional[float] = None,
    ) -> float:
        """Apply W_e <- W_e + alpha * R and return the new (clipped) weight.

        ``outcome_score`` R is positive for a successful remediation and negative
        for a failed one, so successful fixes strengthen relationships and failed
        fixes reduce their confidence. Ignored when the graph is static.
        """
        edge = self.get_edge(src, dst, edge_type)
        if edge is None:
            raise KeyError(f"edge {src} -[{edge_type}]-> {dst} not found")
        if not self.adaptive:
            return edge.weight
        a = self.alpha if alpha is None else alpha
        edge.weight = self._clip(edge.weight + a * outcome_score)
        return edge.weight

    # --------------------------------------------------------------- traversal
    def upstream(self, node_id: str, max_depth: int = 3) -> List[Tuple[str, int, float]]:
        """Nodes reachable by following DEPENDS_ON edges (what ``node_id`` depends on).

        Returns (node, depth, path_weight) where path_weight is the product of
        edge weights along the shortest path.
        """
        return self._bfs(node_id, max_depth, forward=True)

    def downstream(self, node_id: str, max_depth: int = 10) -> List[Tuple[str, int, float]]:
        """Nodes that (transitively) depend on ``node_id`` -- the blast radius."""
        return self._bfs(node_id, max_depth, forward=False)

    def _bfs(self, start: str, max_depth: int, forward: bool) -> List[Tuple[str, int, float]]:
        if start not in self.nodes:
            return []
        seen = {start}
        out: List[Tuple[str, int, float]] = []
        queue = deque([(start, 0, 1.0)])
        while queue:
            nid, depth, w = queue.popleft()
            if depth >= max_depth:
                continue
            edges = (self.out_edges(nid, EdgeType.DEPENDS_ON) if forward
                     else self.in_edges(nid, EdgeType.DEPENDS_ON))
            for e in edges:
                nxt = e.dst if forward else e.src
                if nxt in seen:
                    continue
                seen.add(nxt)
                out.append((nxt, depth + 1, w * e.weight))
                queue.append((nxt, depth + 1, w * e.weight))
        return out

    def neighborhood(self, node_id: str, depth: int = 1) -> Set[str]:
        if node_id not in self.nodes:
            return set()
        seen = {node_id}
        frontier = {node_id}
        for _ in range(depth):
            nxt: Set[str] = set()
            for nid in frontier:
                for k in self._out.get(nid, ()):  # structural neighbours only
                    if k[1] in (EdgeType.DEPENDS_ON, EdgeType.RELATED_TO):
                        nxt.add(k[2])
                for k in self._in.get(nid, ()):
                    if k[1] in (EdgeType.DEPENDS_ON, EdgeType.RELATED_TO):
                        nxt.add(k[0])
            nxt -= seen
            seen |= nxt
            frontier = nxt
        seen.discard(node_id)
        return {n for n in seen if self.nodes[n].type != NodeType.INCIDENT}

    def has_cycle(self) -> bool:
        """True if the DEPENDS_ON sub-graph contains a cycle."""
        color: Dict[str, int] = {}
        for root in self.nodes:
            if color.get(root):
                continue
            stack = [(root, iter(self.out_edges(root, EdgeType.DEPENDS_ON)))]
            color[root] = 1
            while stack:
                nid, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    color[nid] = 2
                    stack.pop()
                    continue
                c = color.get(nxt.dst, 0)
                if c == 1:
                    return True
                if c == 0:
                    color[nxt.dst] = 1
                    stack.append((nxt.dst, iter(self.out_edges(nxt.dst, EdgeType.DEPENDS_ON))))
        return False

    # ------------------------------------------------------- runtime state
    def mark_failed(self, job: str, timestamp: float) -> None:
        if job in self.nodes:
            self.nodes[job].props["last_failure_ts"] = timestamp
            self.nodes[job].props["status"] = "failed"

    def mark_healthy(self, job: str) -> None:
        if job in self.nodes:
            self.nodes[job].props["status"] = "healthy"

    def recently_failed_upstream(
        self, job: str, now: float, window: float, max_depth: int = 3
    ) -> List[Tuple[str, int, float]]:
        hits = []
        for nid, depth, w in self.upstream(job, max_depth):
            ts = self.nodes[nid].props.get("last_failure_ts")
            if ts is not None and 0 <= now - ts <= window:
                hits.append((nid, depth, w))
        return hits

    def failure_profile(self, node_id: str) -> Counter:
        if node_id not in self.nodes:
            return Counter()
        return self.nodes[node_id].props.setdefault("failure_profile", Counter())

    # ------------------------------------------------ remediation knowledge
    def ensure_runbook_fix(self, category: FailureCategory, action_name: str,
                           prior_successes: float = 0.0, prior_attempts: float = 0.0) -> Edge:
        rb, fx = runbook_id(category), fix_id(action_name)
        self.add_node(rb, NodeType.RUNBOOK, category=FailureCategory(category).value)
        self.add_node(fx, NodeType.FIX, action=action_name)
        edge = self.get_edge(rb, fx, EdgeType.REFERENCES)
        if edge is None:
            edge = self.add_edge(rb, fx, EdgeType.REFERENCES,
                                 successes=prior_successes, attempts=prior_attempts,
                                 real_attempts=0)
        return edge

    def fix_stats(self, category: FailureCategory, action_name: str) -> Tuple[float, float, float, int]:
        """(successes, attempts, weight, real_attempts) for a runbook->fix edge."""
        edge = self.get_edge(runbook_id(category), fix_id(action_name), EdgeType.REFERENCES)
        if edge is None:
            return 0.0, 0.0, self.initial_weight, 0
        return (edge.props.get("successes", 0.0), edge.props.get("attempts", 0.0),
                edge.weight, edge.props.get("real_attempts", 0))

    def record_fix_outcome(self, category: FailureCategory, action_name: str,
                           outcome: Outcome, outcome_score: float) -> float:
        edge = self.ensure_runbook_fix(category, action_name)
        edge.props["attempts"] = edge.props.get("attempts", 0.0) + 1
        edge.props["real_attempts"] = edge.props.get("real_attempts", 0) + 1
        if outcome == Outcome.SUCCESS:
            edge.props["successes"] = edge.props.get("successes", 0.0) + 1
        elif outcome == Outcome.PARTIAL:
            edge.props["successes"] = edge.props.get("successes", 0.0) + 0.5
        return self.update_edge_weight(edge.src, edge.dst, EdgeType.REFERENCES, outcome_score)

    def ingest_incident(self, record: IncidentRecord, engineer: Optional[str] = None) -> str:
        """Graph population for a resolved incident (paper Sec. III.C.2)."""
        self._incident_seq += 1
        inc_id = f"incident:{record.event.event_id}"
        category = record.confirmed_category or (
            record.hypothesis.category if record.hypothesis else None)
        self.add_node(inc_id, NodeType.INCIDENT,
                      category=category.value if category else None,
                      outcome=record.outcome.value,
                      minutes=record.resolution_minutes)
        ev = record.event
        for nid, ntype in ((ev.pipeline, NodeType.PIPELINE), (ev.job, NodeType.JOB)):
            self.add_node(nid, ntype)
            self.add_edge(nid, inc_id, EdgeType.FAILED_BY)
            if category is not None:
                self.failure_profile(nid)[category.value] += 1
        if category is not None:
            rb = runbook_id(category)
            self.add_node(rb, NodeType.RUNBOOK, category=category.value)
            self.add_edge(inc_id, rb, EdgeType.RELATED_TO)
        if record.action is not None:
            fx = fix_id(record.action.name)
            self.add_node(fx, NodeType.FIX, action=record.action.name)
            w = 1.0 if record.outcome == Outcome.SUCCESS else (
                0.5 if record.outcome == Outcome.PARTIAL else 0.0)
            self.add_edge(inc_id, fx, EdgeType.FIXED_BY, weight=w)
        if engineer:
            self.add_node(engineer, NodeType.ENGINEER)
            self.add_edge(engineer, inc_id, EdgeType.RELATED_TO)
        return inc_id

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> Dict[str, Any]:
        def clean(props: Dict[str, Any]) -> Dict[str, Any]:
            return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in props.items()}

        return {
            "config": {"alpha": self.alpha, "initial_weight": self.initial_weight,
                       "bounds": list(self.bounds), "adaptive": self.adaptive},
            "nodes": [{"id": n.id, "type": n.type.value, "props": clean(n.props)}
                      for n in self.nodes.values()],
            "edges": [{"src": e.src, "dst": e.dst, "type": e.type.value,
                       "weight": e.weight, "props": clean(e.props)} for e in self.edges.values()],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveKnowledgeGraph":
        cfg = data.get("config", {})
        g = cls(alpha=cfg.get("alpha", 0.1), initial_weight=cfg.get("initial_weight", 0.5),
                bounds=tuple(cfg.get("bounds", (0.0, 1.0))), adaptive=cfg.get("adaptive", True))
        for n in data["nodes"]:
            props = dict(n["props"])
            if "failure_profile" in props:
                props["failure_profile"] = Counter(props["failure_profile"])
            g.add_node(n["id"], NodeType(n["type"]), **props)
        for e in data["edges"]:
            g.add_edge(e["src"], e["dst"], EdgeType(e["type"]), weight=e["weight"], **e["props"])
        return g

    @classmethod
    def load(cls, path: str) -> "AdaptiveKnowledgeGraph":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_cypher(self) -> Iterable[str]:
        """Yield Neo4j Cypher statements reproducing the graph."""
        def q(s: Any) -> str:
            return json.dumps(str(s))

        for n in self.nodes.values():
            yield f"MERGE (n:{n.type.value} {{id: {q(n.id)}}});"
        for e in self.edges.values():
            yield (f"MATCH (a {{id: {q(e.src)}}}), (b {{id: {q(e.dst)}}}) "
                   f"MERGE (a)-[r:{e.type.value}]->(b) SET r.weight = {e.weight:.6f};")

    def stats(self) -> Dict[str, int]:
        c = Counter(n.type.value for n in self.nodes.values())
        c.update(f"edge:{e.type.value}" for e in self.edges.values())
        return dict(c)
