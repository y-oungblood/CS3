"""Adventurousness dial."""

from __future__ import annotations

import numpy as np

from recsys import config
from recsys.artifacts import Artifacts
from recsys.scoring import Candidates


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min() if len(x) else 0.0
    return (x - x.min()) / span if span > 0 else np.zeros_like(x, dtype=np.float64)


def rerank(candidates: Candidates, lam: float, art: Artifacts) -> Candidates:
    """Reorder the pool by (1 - lam) * relevance_norm + lam * novelty_norm.

    lam is clipped to [0, LAMBDA_MAX]. Ties keep the original (relevance) order,
    so lam = 0 returns the pool unchanged.
    """
    lam = float(np.clip(lam, 0.0, config.LAMBDA_MAX))
    if candidates.size == 0 or lam == 0.0:
        return candidates
    final = (1 - lam) * _minmax(candidates.scores) + lam * _minmax(art.novelty[candidates.positions])
    return candidates.take(np.argsort(-final, kind="stable"))
