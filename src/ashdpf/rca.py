"""Root Cause Analysis Engine (RCAE) -- paper Sec. III.D, patent [0019].

RC = argmax_i P(RC_i | F)                       (Eq. 2)
C_RCA = beta1 * G + beta2 * H + beta3 * ML      (Eq. 3)

G  : graph-based confidence (dependency-chain traversal + failure profiles)
H  : historical similarity (log, metadata and graph-neighbourhood similarity)
ML : supervised classifier confidence (random forest / gradient boosting, pluggable)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from .config import ASHDPFConfig
from .features import event_text, jaccard, make_vectorizer, metadata_tokens
from .knowledge_graph import AdaptiveKnowledgeGraph
from .models import (ALL_CATEGORIES, FailureCategory, FailureEvent, Outcome,
                     RootCauseHypothesis)

Distribution = Dict[FailureCategory, float]


def uniform() -> Distribution:
    p = 1.0 / len(ALL_CATEGORIES)
    return {c: p for c in ALL_CATEGORIES}


def normalise(scores: Dict[FailureCategory, float]) -> Distribution:
    full = {c: max(0.0, float(scores.get(c, 0.0))) for c in ALL_CATEGORIES}
    total = sum(full.values())
    if total <= 0:
        return uniform()
    return {c: v / total for c, v in full.items()}


def combine(components: Sequence[Tuple[float, Distribution]]) -> Distribution:
    """Weighted sum of distributions (Eq. 3). Weights are renormalised over the
    enabled components so the result stays a probability distribution."""
    active = [(w, d) for w, d in components if w > 0]
    if not active:
        return uniform()
    wsum = sum(w for w, _ in active)
    return {c: sum(w * d[c] for w, d in active) / wsum for c in ALL_CATEGORIES}


# --------------------------------------------------------------------- G
class GraphReasoner:
    def __init__(self, kg: AdaptiveKnowledgeGraph, cfg: ASHDPFConfig) -> None:
        self.kg = kg
        self.cfg = cfg

    def score(self, event: FailureEvent) -> Tuple[Distribution, List[str]]:
        evidence: List[str] = []
        scores: Dict[FailureCategory, float] = {}
        hits = self.kg.recently_failed_upstream(
            event.job, event.timestamp, self.cfg.dependency_window_minutes, self.cfg.graph_max_depth)
        if hits:
            signal = sum(w / depth for _, depth, w in hits)
            scores[FailureCategory.JOB_DEPENDENCY] = 3.0 * min(1.0, signal)
            evidence.append("upstream failures: " + ", ".join(h[0] for h in hits[:5]))

        for node, weight in ((event.job, 1.0), (event.pipeline, 0.5)):
            prof = self.kg.failure_profile(node)
            total = sum(prof.values())
            if total:
                for cat_value, n in prof.items():
                    cat = FailureCategory(cat_value)
                    if cat == FailureCategory.JOB_DEPENDENCY and not hits:
                        continue  # dependency failures need a live upstream signal
                    scores[cat] = scores.get(cat, 0.0) + weight * n / total
        if not scores:
            return uniform(), evidence
        # Mix with a small uniform floor so a strong prior never zeroes a category.
        dist = normalise(scores)
        return {c: 0.9 * dist[c] + 0.1 / len(ALL_CATEGORIES) for c in ALL_CATEGORIES}, evidence


# --------------------------------------------------------------------- H
@dataclass
class HistoricalCase:
    event_id: str
    category: FailureCategory
    meta: frozenset
    neighborhood: frozenset
    action: Optional[str]
    outcome: Outcome


class HistoricalMatcher:
    """Retrieves previously resolved incidents and scores their similarity."""

    W_LOG, W_META, W_GRAPH = 0.6, 0.2, 0.2
    PRIOR_MASS = 0.5

    def __init__(self, kg: Optional[AdaptiveKnowledgeGraph], top_k: int = 10,
                 use_graph: bool = True) -> None:
        self.kg = kg
        self.top_k = top_k
        self.use_graph = use_graph and kg is not None
        self.vectorizer = make_vectorizer()
        self.cases: List[HistoricalCase] = []
        self._matrix: Optional[sparse.csr_matrix] = None
        self._pending: List[sparse.csr_matrix] = []

    def _commit(self) -> None:
        if self._pending:
            blocks = ([self._matrix] if self._matrix is not None else []) + self._pending
            self._matrix = sparse.vstack(blocks).tocsr()
            self._pending = []

    def __len__(self) -> int:
        return len(self.cases)

    def _neighborhood(self, event: FailureEvent) -> frozenset:
        if not self.use_graph:
            return frozenset()
        return frozenset(self.kg.neighborhood(event.job, depth=1) | {event.pipeline})

    def add(self, event: FailureEvent, category: FailureCategory,
            action: Optional[str], outcome: Outcome) -> None:
        self._pending.append(self.vectorizer.transform([event_text(event)]))
        if len(self._pending) >= 256:  # amortised O(1) append
            self._commit()
        self.cases.append(HistoricalCase(event.event_id, FailureCategory(category),
                                         frozenset(metadata_tokens(event)),
                                         self._neighborhood(event), action, outcome))

    def neighbours(self, event: FailureEvent) -> List[Tuple[HistoricalCase, float]]:
        if not self.cases:
            return []
        q = self.vectorizer.transform([event_text(event)])
        parts = []
        if self._matrix is not None:
            parts.append((self._matrix @ q.T).toarray().ravel())
        if self._pending:
            parts.append((sparse.vstack(self._pending) @ q.T).toarray().ravel())
        cos = np.concatenate(parts)
        pre = min(len(cos), max(self.top_k * 5, 50))
        idx = np.argpartition(-cos, pre - 1)[:pre]
        meta = metadata_tokens(event)
        nb = self._neighborhood(event)
        scored = []
        for i in idx:
            case = self.cases[i]
            # Each similarity term only counts when there is something to compare.
            terms = [(self.W_LOG, float(cos[i]))]
            if meta or case.meta:
                terms.append((self.W_META, jaccard(meta, case.meta)))
            if self.use_graph and (nb or case.neighborhood):
                terms.append((self.W_GRAPH, jaccard(nb, case.neighborhood)))
            s = sum(w * v for w, v in terms) / sum(w for w, _ in terms)
            scored.append((case, s))
        scored.sort(key=lambda t: -t[1])
        return scored[: self.top_k]

    def score(self, event: FailureEvent) -> Tuple[Distribution, float, List[Tuple[HistoricalCase, float]]]:
        nbrs = self.neighbours(event)
        if not nbrs:
            return uniform(), 0.0, []
        # Similarity-weighted vote plus a uniform pseudo-count, so that one weak
        # match does not become a confident distribution.
        prior = self.PRIOR_MASS / len(ALL_CATEGORIES)
        scores: Dict[FailureCategory, float] = {c: prior for c in ALL_CATEGORIES}
        for case, s in nbrs:
            scores[case.category] += s ** 2
        return normalise(scores), nbrs[0][1], nbrs


# -------------------------------------------------------------------- ML
class MLClassifier:
    """Supervised root-cause classifier over hashed log/metadata features."""

    def __init__(self, model: str = "random_forest", n_estimators: int = 60, seed: int = 7,
                 n_features: Optional[int] = None) -> None:
        self.model_name = model
        self.n_estimators = n_estimators
        self.seed = seed
        if n_features is None:
            n_features = 2 ** 14 if model == "random_forest" else 2 ** 10
        self.vectorizer = make_vectorizer(n_features)
        self.model = None
        self.classes_: List[FailureCategory] = []

    def _new_model(self):
        if self.model_name == "random_forest":
            return RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.seed)
        if self.model_name in ("gradient_boosting", "xgboost"):
            # sklearn's histogram gradient boosting (XGBoost-style) avoids a heavy dependency;
            # swap in xgboost.XGBClassifier here if it is installed.
            return HistGradientBoostingClassifier(max_iter=self.n_estimators, random_state=self.seed,
                                                  early_stopping=False)
        raise ValueError(f"unknown model {self.model_name}")

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def _X(self, events: Sequence[FailureEvent]):
        X = self.vectorizer.transform([event_text(e) for e in events])
        return X if self.model_name == "random_forest" else X.toarray()

    def fit(self, events: Sequence[FailureEvent], labels: Sequence[FailureCategory]) -> "MLClassifier":
        labels = [FailureCategory(lbl).value for lbl in labels]
        if len(set(labels)) < 2:
            return self
        self.model = self._new_model().fit(self._X(events), labels)
        self.classes_ = [FailureCategory(c) for c in self.model.classes_]
        return self

    def predict_proba(self, event: FailureEvent) -> Distribution:
        if not self.fitted:
            return uniform()
        p = self.model.predict_proba(self._X([event]))[0]
        return normalise({c: float(v) for c, v in zip(self.classes_, p)})

    def predict(self, events: Sequence[FailureEvent]) -> List[FailureCategory]:
        if not self.fitted:
            return [ALL_CATEGORIES[0]] * len(events)
        return [FailureCategory(c) for c in self.model.predict(self._X(events))]


# ------------------------------------------------- keyword (rule) baseline
KEYWORD_RULES: List[Tuple[str, FailureCategory]] = [
    (r"access denied|unauthori[sz]ed|expired token|invalid credentials|403", FailureCategory.CREDENTIAL),
    (r"outofmemory|out of memory|disk quota|no space left|oom", FailureCategory.RESOURCE_EXHAUSTION),
    (r"schema|column .* not found|type mismatch|incompatible", FailureCategory.SCHEMA_EVOLUTION),
    (r"null|duplicate|constraint|row count", FailureCategory.DATA_QUALITY),
    (r"sensor|upstream|dependency", FailureCategory.JOB_DEPENDENCY),
    (r"connection reset|read timed out|unreachable|dns", FailureCategory.NETWORK_TIMEOUT),
    (r"invalid parameter|config|unknown option|misconfigur", FailureCategory.CONFIGURATION),
    (r"node lost|executor lost|host down|instance terminated|cluster", FailureCategory.INFRASTRUCTURE),
]


class KeywordRCA:
    """Static rule-based classifier used by the rule-based baseline / ablation."""

    def __init__(self, rules=KEYWORD_RULES) -> None:
        self.rules = [(re.compile(p, re.I), c) for p, c in rules]

    def classify(self, event: FailureEvent) -> Optional[FailureCategory]:
        text = " ".join(event.vector.logs)
        for pat, cat in self.rules:
            if pat.search(text):
                return cat
        return None


# ---------------------------------------------------------------- engine
class RCAEngine:
    def __init__(self, kg: AdaptiveKnowledgeGraph, cfg: ASHDPFConfig,
                 classifier: Optional[MLClassifier] = None) -> None:
        self.cfg = cfg
        self.kg = kg
        graph_on = cfg.use_graph and cfg.kg_mode != "none"
        self.graph = GraphReasoner(kg, cfg) if graph_on else None
        self.history = HistoricalMatcher(kg if graph_on else None, cfg.history_top_k,
                                         use_graph=graph_on)
        self.ml = classifier or MLClassifier(n_estimators=cfg.ml_estimators, seed=cfg.seed)
        self.keywords = KeywordRCA()
        self._train_events: List[FailureEvent] = []
        self._train_labels: List[FailureCategory] = []
        self._since_retrain = 0
        self.last_neighbours: List[Tuple[HistoricalCase, float]] = []

    # -- training data management (used by the Continuous Learning Engine)
    def add_labelled(self, event: FailureEvent, label: FailureCategory) -> bool:
        self._train_events.append(event)
        self._train_labels.append(FailureCategory(label))
        self._since_retrain += 1
        n = len(self._train_events)
        due = (not self.ml.fitted and n >= self.cfg.min_train_samples) or (
            self._since_retrain >= self.cfg.retrain_every)
        if due and n >= self.cfg.min_train_samples:
            self.retrain()
            return True
        return False

    def retrain(self) -> None:
        n = self.cfg.ml_max_train
        self.ml.fit(self._train_events[-n:], self._train_labels[-n:])
        self._since_retrain = 0

    def analyze(self, event: FailureEvent) -> RootCauseHypothesis:
        self.last_neighbours = []
        if not self.cfg.use_rca:
            cat = self.keywords.classify(event)
            dist = uniform() if cat is None else {
                c: (0.65 if c == cat else 0.05) for c in ALL_CATEGORIES}
            best = max(dist, key=dist.get)
            return RootCauseHypothesis(best, dist[best], dist, {"keyword": dist})

        b1, b2, b3 = self.cfg.beta
        evidence: List[str] = []
        comps: Dict[str, Distribution] = {}
        parts: List[Tuple[float, Distribution]] = []
        max_sim = 0.0

        if self.graph is not None:
            g, ev = self.graph.score(event)
            evidence += ev
            comps["G"] = g
            parts.append((b1, g))
        self.last_neighbours = []
        if self.cfg.use_history:
            h, max_sim, nbrs = self.history.score(event)
            self.last_neighbours = nbrs
            if nbrs:
                comps["H"] = h
                parts.append((b2, h))
                evidence.append(f"top historical match {nbrs[0][0].event_id} sim={max_sim:.2f}")
        if self.cfg.use_ml and self.ml.fitted:
            m = self.ml.predict_proba(event)
            comps["ML"] = m
            parts.append((b3, m))
        elif self.cfg.use_ml:
            # Cold start: until enough labelled incidents exist to train the
            # classifier, runbook keyword signatures stand in for the ML term.
            kw = self.keywords.classify(event)
            if kw is not None:
                k = {c: (0.65 if c == kw else 0.05) for c in ALL_CATEGORIES}
                comps["KW"] = k
                parts.append((b3, k))

        dist = combine(parts)
        best = max(dist, key=dist.get)  # Eq. (2)
        return RootCauseHypothesis(best, dist[best], dist, comps, max_sim, evidence)
