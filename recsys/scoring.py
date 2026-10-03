"""Signal weights, item-kNN scoring and explanations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recsys import config
from recsys.artifacts import Artifacts
from recsys.session import Answer, Session

FALLBACK_TEXT = "Highly rated by MovieLens users."


# ---------------------------------------------------------------- weights


def user_mean(session: Session, global_mean: float) -> float:
    """Mean of the user's ratings (seeds count as 5), shrunk toward the global mean."""
    ratings = [a.rating for a in session.answers if a.kind == "rated"]
    return (sum(ratings) + config.K_SHRINK * global_mean) / (len(ratings) + config.K_SHRINK)


def answer_weight(answer: Answer, mu: float) -> float:
    if answer.kind == "rated":
        return answer.rating - mu
    if answer.kind == "interested":
        return config.W_INTERESTED
    if answer.kind == "not_interested":
        return -config.W_INTERESTED
    return 0.0


def weights(session: Session, global_mean: float) -> dict[int, float]:
    """movie_id -> w_j for every answer that carries signal.

    unseen_unknown answers (simulation only) are left out entirely, so they add
    nothing to the score's numerator or its similarity denominator.
    """
    mu = user_mean(session, global_mean)
    return {a.movie_id: answer_weight(a, mu) for a in session.answers if a.kind != "unseen_unknown"}


def _weight_arrays(art: Artifacts, session: Session) -> tuple[np.ndarray, np.ndarray]:
    w = weights(session, art.global_mean)
    if not w:
        return np.empty(0, dtype=np.int64), np.empty(0)
    return art.index_of(list(w)), np.fromiter(w.values(), dtype=np.float64, count=len(w))


# ---------------------------------------------------------------- recommend


@dataclass
class Candidates:
    """A scored candidate pool, best first. Positions index into Artifacts arrays."""

    positions: np.ndarray
    scores: np.ndarray
    fallback: np.ndarray  # True where the item was filled in by the popularity fallback

    @property
    def size(self) -> int:
        return len(self.positions)

    def movie_ids(self, art: Artifacts) -> list[int]:
        return art.movie_ids[self.positions].tolist()

    def take(self, order: np.ndarray) -> Candidates:
        return Candidates(self.positions[order], self.scores[order], self.fallback[order])


def score_all(art: Artifacts, session: Session) -> tuple[np.ndarray, np.ndarray]:
    """Score every movie: (scores, support) where support = sum of |sim| to answered movies.

    score(i) = sum_j sim(i,j) w_j / (sum_j |sim(i,j)| + DAMPING), over answered j in i's neighbors.
    Movies with support 0 have no answered neighbor and are not candidates.
    """
    pos, w = _weight_arrays(art, session)
    num = np.zeros(art.n)
    den = np.zeros(art.n)
    if len(pos):
        owner, cand, sim = art.reverse.gather(pos)
        np.add.at(num, cand, sim * w[owner])
        np.add.at(den, cand, np.abs(sim))
    return num / (den + config.DAMPING), den


def recommend(session: Session, art: Artifacts, n: int = config.CANDIDATE_POOL) -> Candidates:
    """Top-n candidate pool by score; filled from top-rated popular movies when sparse."""
    scores, support = score_all(art, session)
    excluded = np.zeros(art.n, dtype=bool)
    if session.answers:
        excluded[art.index_of(list(session.answered_ids()))] = True

    positive = np.flatnonzero((support > 0) & (scores > 0) & ~excluded)
    order = positive[np.lexsort((positive, -scores[positive]))][:n]
    pool_scores = scores[order]
    fallback = np.zeros(len(order), dtype=bool)

    if len(order) < config.MIN_POSITIVE_CANDIDATES:
        # Skip movies the user has negative evidence against, not just answered ones.
        taken = excluded | ((support > 0) & (scores < 0))
        taken[order] = True
        fill = art.fallback_order[~taken[art.fallback_order]][: n - len(order)]
        # Fallback items rank below every scored item, in mean-rating order.
        fill_scores = -np.arange(1, len(fill) + 1) * 1e-6
        order = np.concatenate([order, fill])
        pool_scores = np.concatenate([pool_scores, fill_scores])
        fallback = np.concatenate([fallback, np.ones(len(fill), dtype=bool)])

    return Candidates(order, pool_scores, fallback)


# ---------------------------------------------------------------- explain


@dataclass(frozen=True)
class Explanation:
    text: str
    reason_ids: tuple[int, ...]  # answered movies named in the text
    tags: str | None = None      # e.g. "cerebral · slow-burn", tags shared with the reasons


def contributions(movie_id: int, session: Session, art: Artifacts) -> dict[int, float]:
    """c_j = sim(i, j) * w_j for each answered neighbor j of movie i."""
    w = weights(session, art.global_mean)
    nbr_pos, sims = art.forward.row(int(art.index_of([movie_id])[0]))
    out = {}
    for j, s in zip(art.movie_ids[nbr_pos].tolist(), sims.tolist()):
        if j in w:
            out[j] = s * w[j]
    return out


def _reasons_text(answers: list[Answer], titles: list[str]) -> str:
    """'rated A ★5 and added B to your list'; same-kind reasons share one verb."""

    def kind(a: Answer) -> str:
        return "interested" if a.kind == "interested" else "seed" if a.source == "seed" else "rated"

    def obj(a: Answer, title: str) -> str:
        return f"{title} ★{a.rating}" if kind(a) == "rated" else title

    def phrase(k: str, objects: str) -> str:
        return {"interested": f"added {objects} to your list", "seed": f"love {objects}", "rated": f"rated {objects}"}[k]

    kinds = [kind(a) for a in answers]
    if len(answers) == 2 and kinds[0] == kinds[1]:
        return phrase(kinds[0], " and ".join(obj(a, t) for a, t in zip(answers, titles)))
    return " and ".join(phrase(k, obj(a, t)) for k, a, t in zip(kinds, answers, titles))


def explain(movie_id: int, session: Session, art: Artifacts) -> Explanation:
    """Name the (up to) two answered movies with the largest positive contributions."""
    contrib = contributions(movie_id, session, art)
    top = sorted(((c, j) for j, c in contrib.items() if c > 0), key=lambda t: (-t[0], t[1]))[:2]
    if not top:
        return Explanation(FALLBACK_TEXT, ())

    by_id = {a.movie_id: a for a in session.answers}
    reason_ids = tuple(j for _, j in top)
    titles = [art.movie(j)["title"] for j in reason_ids]
    text = "Because you " + _reasons_text([by_id[j] for j in reason_ids], titles) + "."

    own = list(art.movie(movie_id)["top_tags"])
    shared = {t for j in reason_ids for t in art.movie(j)["top_tags"]}
    tags = [t for t in own if t in shared]
    return Explanation(text, reason_ids, " · ".join(tags) if tags else None)
