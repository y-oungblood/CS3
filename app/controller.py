"""Per-visitor app logic, independent of the UI toolkit.

Each browser visitor gets one Controller. Views only render its state and call
its methods, so every flow here is unit-testable without a browser.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from recsys import config
from recsys.artifacts import Artifacts
from recsys.rerank import rerank
from recsys.scoring import Candidates, explain, recommend
from recsys.selection import assign_strategy, next_card
from recsys.session import Answer, Session
from recsys.storage import RecItem, Storage, User
from recsys.timing import group_watchlist, when_to_watch

TOP_N = 10
MIN_SEEDS, MAX_SEEDS = 3, 5


@dataclass
class RecCard:
    movie_id: int
    rank: int
    title: str
    year: int | None
    poster_url: str | None
    genres: list[str]
    explanation: str
    tags: str | None
    when: str | None
    answer: Answer | None = None  # set once answered on the recommendations page
    timing_vote: str | None = None  # "up" / "down"


class Controller:
    def __init__(self, art: Artifacts, store: Storage, rng: np.random.Generator | None = None, today=date.today):
        self.art = art
        self.store = store
        self.rng = rng if rng is not None else np.random.default_rng()
        self.today = today
        self.user: User | None = None
        self.session: Session | None = None
        # Deck
        self.current_card: int | None = None
        self._card_shown_at: float | None = None
        self._checkins_fired: set[int] = set()
        # Recommendations page
        self.trigger: str = "voluntary"
        self.lam: float = 0.0
        self._pool: Candidates | None = None
        self.list_id: str | None = None
        self.cards: list[RecCard] = []

    # ------------------------------------------------------------ account

    def start_new(self) -> User:
        self.user = self.store.create_user(assign_strategy(self.rng))
        self.session = Session(self.user.user_id, self.user.card_strategy)
        return self.user

    def login(self, handle: str) -> bool:
        user = self.store.get_user_by_handle(handle)
        if user is None:
            return False
        self.user = user
        self.session = self.store.load_session(user)
        # A returning user past a check-in should not be interrupted by it again.
        self._checkins_fired = {c for c in config.CHECKINS if self.n_informative >= c}
        return True

    @property
    def needs_seeds(self) -> bool:
        return not any(a.source == "seed" for a in self.session.answers)

    # ------------------------------------------------------------ seed picks

    def search(self, query: str, limit: int = 8) -> pd.DataFrame:
        hits = self.art.search(query, limit=limit + MAX_SEEDS)
        return hits[~hits["movie_id"].isin(self.session.answered_ids())].head(limit)

    def save_seeds(self, movie_ids: list[int]) -> None:
        if not MIN_SEEDS <= len(movie_ids) <= MAX_SEEDS:
            raise ValueError(f"pick {MIN_SEEDS}-{MAX_SEEDS} movies")
        for mid in movie_ids:
            self._record(Answer.seed(int(mid)))

    # ------------------------------------------------------------ swipe deck

    @property
    def n_informative(self) -> int:
        return self.session.n_informative_answers()

    @property
    def progress(self) -> float:
        return min(1.0, self.n_informative / config.PROGRESS_FULL)

    @property
    def unlocked(self) -> bool:
        return self.n_informative >= config.GATE_ANSWERS

    @property
    def remaining_to_unlock(self) -> int:
        return max(0, config.GATE_ANSWERS - self.n_informative)

    def deal(self) -> int | None:
        """The card on screen, drawing a new one if needed (None if the catalog is exhausted)."""
        if self.current_card is None:
            card = next_card(self.session, self.art, rng=self.rng)
            if card is not None:
                self.session.show(card)
                self._card_shown_at = time.monotonic()
            self.current_card = card
        return self.current_card

    def answer_card(self, kind: str, rating: int | None = None) -> str | None:
        """Record the answer to the current card; returns a check-in trigger if one is due."""
        if self.current_card is None:
            raise RuntimeError("no card on screen")
        response_ms = None
        if self._card_shown_at is not None:
            response_ms = int((time.monotonic() - self._card_shown_at) * 1000)
        self._record(Answer(self.current_card, kind, rating, "swipe"), response_ms, self.session.card_strategy)
        self.current_card = None
        self._card_shown_at = None
        n = self.n_informative
        if n in config.CHECKINS and n not in self._checkins_fired:
            self._checkins_fired.add(n)
            return f"checkin_{n}"
        return None

    @property
    def can_undo(self) -> bool:
        return bool(self.session.answers) and self.session.answers[-1].source == "swipe"

    def undo(self) -> int | None:
        """Reverse the last swipe; its movie becomes the card on screen again."""
        if not self.can_undo:
            return None
        removed = self.store.delete_last_answer(self.user.user_id)
        self.session.undo()
        if self.current_card is not None:
            self.session.shown_cards.discard(self.current_card)  # back into the deck
        self.current_card = removed.movie_id
        self.session.show(removed.movie_id)
        self._card_shown_at = time.monotonic()
        return removed.movie_id

    # ------------------------------------------------------------ recommendations

    def open_recommendations(self, trigger: str = "voluntary") -> list[RecCard]:
        self.trigger = trigger
        self._pool = recommend(self.session, self.art)
        return self._show(trigger)

    def set_lambda(self, lam: float) -> list[RecCard]:
        """Re-rank the cached pool (instant); logs the list the user settled on."""
        lam = float(np.clip(lam, 0.0, config.LAMBDA_MAX))
        if abs(lam - self.lam) < 1e-9 and self.cards:
            return self.cards
        self.lam = lam
        return self._show("dial")

    def refresh(self) -> list[RecCard]:
        self._pool = recommend(self.session, self.art)
        return self._show("refresh")

    def _show(self, trigger: str) -> list[RecCard]:
        answered = self.session.answered_ids()
        ranked = [m for m in rerank(self._pool, self.lam, self.art).movie_ids(self.art) if m not in answered]
        self.cards = [self._card(mid, rank) for rank, mid in enumerate(ranked[:TOP_N], 1)]
        scores = dict(zip(self._pool.movie_ids(self.art), self._pool.scores.tolist()))
        items = [RecItem(c.rank, c.movie_id, float(scores.get(c.movie_id, 0.0)), c.explanation) for c in self.cards]
        self.list_id = self.store.log_rec_list(self.user.user_id, trigger, self.lam, self.n_informative, items)
        return self.cards

    def _card(self, movie_id: int, rank: int) -> RecCard:
        m = self.art.movie(movie_id)
        e = explain(movie_id, self.session, self.art)
        year = m["year"]
        return RecCard(
            movie_id=movie_id,
            rank=rank,
            title=m["title"],
            year=None if pd.isna(year) else int(year),
            poster_url=None if pd.isna(m["poster_url"]) else m["poster_url"],
            genres=list(m["genres"]),
            explanation=e.text,
            tags=e.tags,
            when=when_to_watch(m, self.today()),
        )

    def answer_recommendation(self, movie_id: int, kind: str, rating: int | None = None) -> None:
        """An answer on the recommendations page: updates the session and the list's feedback log."""
        card = next(c for c in self.cards if c.movie_id == movie_id)
        if card.answer is not None:
            return
        answer = Answer(movie_id, kind, rating, "recommendation")
        self._record(answer)
        self.store.log_rec_feedback(self.list_id, self.user.user_id, movie_id, kind, rating)
        card.answer = answer

    def vote_timing(self, movie_id: int, up: bool) -> None:
        card = next(c for c in self.cards if c.movie_id == movie_id)
        vote = "up" if up else "down"
        if card.timing_vote == vote:
            return
        card.timing_vote = vote
        self.store.log_rec_feedback(self.list_id, self.user.user_id, movie_id, f"timing_{vote}")

    def watchlist_groups(self) -> dict[str, list[dict]]:
        movies = []
        for mid in self.session.watchlist():
            m = self.art.movie(mid)
            movies.append(
                {
                    "movie_id": mid,
                    "title": m["title"],
                    "year": None if pd.isna(m["year"]) else int(m["year"]),
                    "poster_url": None if pd.isna(m["poster_url"]) else m["poster_url"],
                    "runtime_min": m["runtime_min"],
                    "genres": list(m["genres"]),
                }
            )
        return group_watchlist(movies)

    # ------------------------------------------------------------ helpers

    def movie(self, movie_id: int) -> pd.Series:
        return self.art.movie(movie_id)

    def _record(self, answer: Answer, response_ms: int | None = None, strategy_used: str | None = None) -> None:
        """Add to the session and save immediately (position = index in the history)."""
        self.session.add(answer)
        self.store.add_answer(self.user.user_id, answer, len(self.session.answers) - 1, response_ms, strategy_used)
