"""Session state: one user's answers, in order."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["rated", "interested", "not_interested", "unseen_unknown"]
Source = Literal["seed", "swipe", "recommendation"]


@dataclass(frozen=True)
class Answer:
    movie_id: int
    kind: Kind
    rating: int | None = None  # 1-5 when kind == "rated"
    source: Source = "swipe"

    def __post_init__(self):
        if self.kind == "rated":
            if self.rating is None or not 1 <= self.rating <= 5:
                raise ValueError(f"rated answers need a 1-5 rating, got {self.rating!r}")
        elif self.rating is not None:
            raise ValueError(f"{self.kind} answers carry no rating")

    @classmethod
    def seed(cls, movie_id: int) -> Answer:
        """A seed pick, recorded as rated 5."""
        return cls(movie_id, "rated", 5, "seed")


@dataclass
class Session:
    user_id: str
    card_strategy: str = "adaptive"
    answers: list[Answer] = field(default_factory=list)
    shown_cards: set[int] = field(default_factory=set)  # shown but not yet answered

    def add(self, answer: Answer) -> None:
        if answer.movie_id in self.answered_ids():
            raise ValueError(f"movie {answer.movie_id} is already answered")
        self.answers.append(answer)
        self.shown_cards.discard(answer.movie_id)

    def show(self, movie_id: int) -> None:
        self.shown_cards.add(movie_id)

    def undo(self) -> Answer | None:
        """Remove and return the most recent answer (None if there is nothing to undo)."""
        return self.answers.pop() if self.answers else None

    def answered_ids(self) -> set[int]:
        return {a.movie_id for a in self.answers}

    def excluded_ids(self) -> set[int]:
        """Movies that must not be offered as cards: answered or currently shown."""
        return self.answered_ids() | self.shown_cards

    def watchlist(self) -> list[int]:
        return [a.movie_id for a in self.answers if a.kind == "interested"]

    def n_informative_answers(self) -> int:
        return sum(a.source != "seed" for a in self.answers)

    def recent_swipes(self, n: int) -> list[int]:
        return [a.movie_id for a in self.answers if a.source == "swipe"][-n:]
