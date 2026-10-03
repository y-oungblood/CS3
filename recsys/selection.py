"""Card-selection strategies for the swipe deck."""

from __future__ import annotations

import numpy as np

from recsys import config
from recsys.artifacts import Artifacts
from recsys.scoring import weights
from recsys.session import Session

STRATEGIES = ("random", "popularity", "pop_entropy", "adaptive")
AB_ARMS = ("adaptive", "pop_entropy")
DIVERSITY_WINDOW = 3  # adaptive phase 1: avoid the first genre of the last 3 cards


def assign_strategy(rng: np.random.Generator | None = None) -> str:
    """Card strategy for a new user: a random A/B arm, or adaptive when AB_TEST is off."""
    if not config.AB_TEST:
        return "adaptive"
    rng = rng if rng is not None else np.random.default_rng()
    return str(rng.choice(AB_ARMS))


def _excluded_mask(art: Artifacts, session: Session, extra: set[int] = frozenset()) -> np.ndarray:
    mask = np.zeros(art.n, dtype=bool)
    ids = session.excluded_ids() | set(extra)
    if ids:
        mask[art.index_of(list(ids))] = True
    return mask


def _first_available(order: np.ndarray, excluded: np.ndarray) -> int | None:
    free = order[~excluded[order]]
    return int(free[0]) if len(free) else None


def _random(art, session, excluded, rng) -> int | None:
    free = np.flatnonzero(~excluded)
    return int(rng.choice(free)) if len(free) else None


def _popularity(art, session, excluded, rng) -> int | None:
    return _first_available(art.popularity_order, excluded)


def _pop_entropy(art, session, excluded, rng) -> int | None:
    pick = _first_available(art.elicit_order, excluded)
    return pick if pick is not None else _popularity(art, session, excluded, rng)


def _phase1(art, session, excluded, rng) -> int | None:
    """pop_entropy, skipping movies whose first genre matches one of the last 3 cards."""
    recent = session.recent_swipes(DIVERSITY_WINDOW) + sorted(session.shown_cards)
    recent_genres = {art.first_genre[p] for p in art.index_of(recent)} - {""} if recent else set()
    free = art.elicit_order[~excluded[art.elicit_order]]
    diverse = free[~np.isin(art.first_genre[free], list(recent_genres))] if recent_genres else free
    if len(diverse):
        return int(diverse[0])
    return _pop_entropy(art, session, excluded, rng)  # rule can't be met: drop it


def _phase2(art, session, excluded, rng) -> int | None:
    """Neighbors of positively weighted movies, ranked by sum(sim * w) * informativeness."""
    w = {m: v for m, v in weights(session, art.global_mean).items() if v > 0}
    if not w:
        return None
    pos = art.index_of(list(w))
    owner, nbr, sim = art.forward.gather(pos)
    affinity = np.zeros(art.n)
    np.add.at(affinity, nbr, sim * np.fromiter(w.values(), dtype=np.float64)[owner])
    rank = affinity * art.informativeness
    cand = np.flatnonzero((affinity > 0) & ~excluded)
    if not len(cand):
        return None
    return int(cand[np.lexsort((cand, -rank[cand]))][0])


def _adaptive(art, session, excluded, rng) -> int | None:
    if session.n_informative_answers() < config.PHASE1_ANSWERS or rng.random() < config.WILDCARD_RATE:
        return _phase1(art, session, excluded, rng)
    pick = _phase2(art, session, excluded, rng)
    return pick if pick is not None else _phase1(art, session, excluded, rng)


_PICKERS = {
    "random": _random,
    "popularity": _popularity,
    "pop_entropy": _pop_entropy,
    "adaptive": _adaptive,
}


def next_card(
    session: Session,
    art: Artifacts,
    strategy: str | None = None,
    rng: np.random.Generator | None = None,
    exclude: set[int] = frozenset(),
) -> int | None:
    """movie_id of the next card, never one that is answered, shown, or in `exclude`.

    `exclude` lets the simulation hide test-set movies. Returns None when the
    catalog is exhausted.
    """
    strategy = strategy or session.card_strategy
    if strategy not in _PICKERS:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    rng = rng if rng is not None else np.random.default_rng()
    excluded = _excluded_mask(art, session, exclude)
    pick = _PICKERS[strategy](art, session, excluded, rng)
    return None if pick is None else int(art.movie_ids[pick])
