"""Turn a FailureEvent into tokens / sparse vectors for similarity and ML."""

from __future__ import annotations

import re
from typing import Iterable, List, Set

from sklearn.feature_extraction.text import HashingVectorizer

from .models import FailureEvent

_NUM = re.compile(r"\b\d+(\.\d+)?\b")
_HEX = re.compile(r"\b[0-9a-f]{8,}\b")


def normalise_log(line: str) -> str:
    line = line.lower()
    line = _HEX.sub("<id>", line)
    return _NUM.sub("<n>", line)


def _bucket(name: str, value: float) -> str:
    if value >= 0.5:
        level = "high"
    elif value >= 0.1:
        level = "mid"
    else:
        level = "low"
    return f"q_{name}_{level}"


def event_text(event: FailureEvent) -> str:
    """Flat textual representation: logs + metadata + bucketed quality metrics."""
    v = event.vector
    parts: List[str] = [normalise_log(line) for line in v.logs]
    parts += [f"m_{k}_{str(val).lower()}" for k, val in sorted(v.metadata.items())]
    parts += [_bucket(k, float(val)) for k, val in sorted(v.quality.items())]
    parts += [f"c_{k}" for k in sorted(v.config)]
    return " ".join(parts)


def metadata_tokens(event: FailureEvent) -> Set[str]:
    v = event.vector
    toks = {f"{k}={val}" for k, val in v.metadata.items()}
    toks |= {_bucket(k, float(val)) for k, val in v.quality.items()}
    return toks


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def make_vectorizer(n_features: int = 2 ** 14) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        ngram_range=(1, 2),
        norm="l2",
        token_pattern=r"(?u)\b[\w<>=.]+\b",
    )
