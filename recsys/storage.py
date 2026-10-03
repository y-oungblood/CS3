"""Persistent storage for users, answers and logged recommendation lists.

Two backends share one interface, chosen by the STORAGE_BACKEND environment variable:

- sqlite (default): a local file, data/app.db, for development.
- firestore: Google Cloud Firestore via firebase-admin, for deployment (free
  hosts wipe local disk on restart). Credentials come from FIREBASE_CREDENTIALS
  (the service-account JSON itself, e.g. a host secret) or
  GOOGLE_APPLICATION_CREDENTIALS (a path to that file). Never commit them.

Tables (SQLite) / collections (Firestore):
    users         user_id, handle (unique), card_strategy, created_at
    answers       answer_id, user_id, movie_id, kind, rating, source, position,
                  response_ms, strategy_used, created_at
    rec_lists     list_id, user_id, created_at, trigger, lambda, n_informative_answers
    rec_items     list_id, rank, movie_id, score, explanation
    rec_feedback  feedback_id, list_id, user_id, movie_id, kind, rating, created_at
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from recsys import config
from recsys.handles import generate_handle, normalize_handle
from recsys.session import Answer, Session

MAX_HANDLE_ATTEMPTS = 50
FEEDBACK_KINDS = ("rated", "interested", "not_interested", "timing_up", "timing_down")
TABLES = ("users", "answers", "rec_lists", "rec_items", "rec_feedback")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class User:
    user_id: str
    handle: str
    card_strategy: str
    created_at: str


@dataclass(frozen=True)
class RecItem:
    rank: int
    movie_id: int
    score: float
    explanation: str


class HandleTaken(Exception):
    pass


class Storage(ABC):
    """Backend-independent behavior; subclasses implement the underscore methods."""

    # ------------------------------------------------------------ users

    def create_user(self, card_strategy: str, rng: np.random.Generator | None = None) -> User:
        """Create a user with a fresh generated handle (regenerated on collision)."""
        rng = rng if rng is not None else np.random.default_rng()
        for _ in range(MAX_HANDLE_ATTEMPTS):
            user = User(_new_id(), generate_handle(rng), card_strategy, _now())
            try:
                self._insert_user(user)
                return user
            except HandleTaken:
                continue
        raise RuntimeError("could not find a free handle")

    def get_user_by_handle(self, handle: str) -> User | None:
        return self._get_user_by_handle(normalize_handle(handle))

    def load_session(self, user: User) -> Session:
        """The user's answers in order, ready for recsys."""
        return Session(user.user_id, user.card_strategy, self._load_answers(user.user_id))

    # ------------------------------------------------------------ answers

    def add_answer(
        self,
        user_id: str,
        answer: Answer,
        position: int,
        response_ms: int | None = None,
        strategy_used: str | None = None,
    ) -> str:
        """Save one answer at its position in the user's history; returns answer_id."""
        row = {
            "answer_id": _new_id(),
            "user_id": user_id,
            "movie_id": int(answer.movie_id),
            "kind": answer.kind,
            "rating": answer.rating,
            "source": answer.source,
            "position": int(position),
            "response_ms": None if response_ms is None else int(response_ms),
            "strategy_used": strategy_used,
            "created_at": _now(),
        }
        self._insert_answer(row)
        return row["answer_id"]

    @abstractmethod
    def delete_last_answer(self, user_id: str) -> Answer | None:
        """Undo: delete the user's most recent answer and return it (None if none)."""

    # ------------------------------------------------------------ recommendation logs

    def log_rec_list(
        self, user_id: str, trigger: str, lam: float, n_informative: int, items: list[RecItem]
    ) -> str:
        """Log one list exactly as shown; returns list_id."""
        row = {
            "list_id": _new_id(),
            "user_id": user_id,
            "created_at": _now(),
            "trigger": trigger,
            "lambda": float(lam),
            "n_informative_answers": int(n_informative),
        }
        self._insert_rec_list(row, [asdict(i) for i in items])
        return row["list_id"]

    def log_rec_feedback(
        self, list_id: str, user_id: str, movie_id: int, kind: str, rating: int | None = None
    ) -> str:
        """Log an answer to a shown recommendation (or a 👍/👎 on its when-to-watch line)."""
        if kind not in FEEDBACK_KINDS:
            raise ValueError(f"unknown feedback kind {kind!r}")
        row = {
            "feedback_id": _new_id(),
            "list_id": list_id,
            "user_id": user_id,
            "movie_id": int(movie_id),
            "kind": kind,
            "rating": rating,
            "created_at": _now(),
        }
        self._insert_feedback(row)
        return row["feedback_id"]

    # ------------------------------------------------------------ analysis

    @abstractmethod
    def export(self) -> dict[str, pd.DataFrame]:
        """Every table as a DataFrame, for the real-user analysis."""

    # ------------------------------------------------------------ backend hooks

    @abstractmethod
    def _insert_user(self, user: User) -> None:
        """Insert, raising HandleTaken if the handle exists."""

    @abstractmethod
    def _get_user_by_handle(self, handle: str) -> User | None: ...

    @abstractmethod
    def _load_answers(self, user_id: str) -> list[Answer]: ...

    @abstractmethod
    def _insert_answer(self, row: dict) -> None: ...

    @abstractmethod
    def _insert_rec_list(self, row: dict, items: list[dict]) -> None: ...

    @abstractmethod
    def _insert_feedback(self, row: dict) -> None: ...


def _answer_from_row(row) -> Answer:
    rating = row["rating"]
    return Answer(
        int(row["movie_id"]),
        row["kind"],
        None if rating is None or pd.isna(rating) else int(rating),
        row["source"],
    )


def _frame(rows: list[dict], table: str) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_COLUMNS[table])


_COLUMNS = {
    "users": ["user_id", "handle", "card_strategy", "created_at"],
    "answers": [
        "answer_id", "user_id", "movie_id", "kind", "rating", "source", "position",
        "response_ms", "strategy_used", "created_at",
    ],
    "rec_lists": ["list_id", "user_id", "created_at", "trigger", "lambda", "n_informative_answers"],
    "rec_items": ["list_id", "rank", "movie_id", "score", "explanation"],
    "rec_feedback": ["feedback_id", "list_id", "user_id", "movie_id", "kind", "rating", "created_at"],
}  # fmt: skip


# ================================================================= SQLite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    handle TEXT NOT NULL UNIQUE,
    card_strategy TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS answers (
    answer_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    movie_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    rating INTEGER,
    source TEXT NOT NULL,
    position INTEGER NOT NULL,
    response_ms INTEGER,
    strategy_used TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (user_id, position)
);
CREATE TABLE IF NOT EXISTS rec_lists (
    list_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    created_at TEXT NOT NULL,
    trigger TEXT NOT NULL,
    lambda REAL NOT NULL,
    n_informative_answers INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS rec_items (
    list_id TEXT NOT NULL REFERENCES rec_lists(list_id),
    rank INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    score REAL,
    explanation TEXT,
    PRIMARY KEY (list_id, rank)
);
CREATE TABLE IF NOT EXISTS rec_feedback (
    feedback_id TEXT PRIMARY KEY,
    list_id TEXT NOT NULL REFERENCES rec_lists(list_id),
    user_id TEXT NOT NULL REFERENCES users(user_id),
    movie_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    rating INTEGER,
    created_at TEXT NOT NULL
);
"""


class SQLiteStorage(Storage):
    """One short-lived connection per operation, so web-server threads can share it."""

    def __init__(self, path: Path | str = config.ROOT / "data" / "app.db"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.path == ":memory:" else None
        with self._conn() as c:
            c.executescript(_SCHEMA)
            if not self._memory:
                c.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = self._memory or sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            try:
                with conn:  # commits on success, rolls back on error
                    yield conn
            finally:
                if not self._memory:
                    conn.close()

    def _insert_user(self, user: User) -> None:
        try:
            with self._conn() as c:
                c.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (user.user_id, user.handle, user.card_strategy, user.created_at))
        except sqlite3.IntegrityError as e:
            raise HandleTaken(user.handle) from e

    def _get_user_by_handle(self, handle: str) -> User | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM users WHERE handle = ?", (handle,)).fetchone()
        return User(**dict(row)) if row else None

    def _load_answers(self, user_id: str) -> list[Answer]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM answers WHERE user_id = ? ORDER BY position", (user_id,)).fetchall()
        return [_answer_from_row(r) for r in rows]

    def _insert_answer(self, row: dict) -> None:
        cols = ", ".join(row)
        with self._conn() as c:
            c.execute(f"INSERT INTO answers ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values()))

    def delete_last_answer(self, user_id: str) -> Answer | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM answers WHERE user_id = ? ORDER BY position DESC LIMIT 1", (user_id,)
            ).fetchone()
            if row is None:
                return None
            c.execute("DELETE FROM answers WHERE answer_id = ?", (row["answer_id"],))
        return _answer_from_row(row)

    def _insert_rec_list(self, row: dict, items: list[dict]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO rec_lists VALUES (:list_id, :user_id, :created_at, :trigger, :lambda, :n_informative_answers)",
                row,
            )
            c.executemany(
                "INSERT INTO rec_items VALUES (:list_id, :rank, :movie_id, :score, :explanation)",
                [{**i, "list_id": row["list_id"]} for i in items],
            )

    def _insert_feedback(self, row: dict) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO rec_feedback VALUES (:feedback_id, :list_id, :user_id, :movie_id, :kind, :rating, :created_at)",
                row,
            )

    def export(self) -> dict[str, pd.DataFrame]:
        with self._conn() as c:
            return {t: _frame([dict(r) for r in c.execute(f"SELECT * FROM {t}")], t) for t in TABLES}


# ================================================================= Firestore
#
# Layout:
#   users/{user_id}                       user fields
#   users/{user_id}/answers/{position}    answer fields; doc ID = zero-padded position
#   handles/{handle}                      {"user_id"}; created atomically, so handles are unique
#   rec_lists/{list_id}                   list fields + "items": [rec_items rows]
#   rec_feedback/{feedback_id}            feedback fields


class FirestoreStorage(Storage):
    def __init__(self, client=None):
        self.db = client if client is not None else _firestore_client()

    def _user_doc(self, user_id: str):
        return self.db.collection("users").document(user_id)

    def _answers(self, user_id: str):
        return self._user_doc(user_id).collection("answers")

    def _insert_user(self, user: User) -> None:
        from google.api_core.exceptions import Conflict

        try:
            self.db.collection("handles").document(user.handle).create({"user_id": user.user_id})
        except Conflict as e:  # AlreadyExists is a Conflict
            raise HandleTaken(user.handle) from e
        self._user_doc(user.user_id).set(asdict(user))

    def _get_user_by_handle(self, handle: str) -> User | None:
        if not handle:
            return None
        ref = self.db.collection("handles").document(handle).get()
        if not ref.exists:
            return None
        snap = self._user_doc(ref.to_dict()["user_id"]).get()
        return User(**snap.to_dict()) if snap.exists else None

    def _load_answers(self, user_id: str) -> list[Answer]:
        docs = sorted((d.to_dict() for d in self._answers(user_id).stream()), key=lambda r: r["position"])
        return [_answer_from_row(r) for r in docs]

    def _insert_answer(self, row: dict) -> None:
        self._answers(row["user_id"]).document(f"{row['position']:06d}").set(row)

    def delete_last_answer(self, user_id: str) -> Answer | None:
        # A user's history is at most a few hundred small docs, so reading them
        # all is cheap and keeps this backend to plain get/set/stream calls.
        docs = list(self._answers(user_id).stream())
        if not docs:
            return None
        last = max(docs, key=lambda d: d.to_dict()["position"])
        row = last.to_dict()
        self._answers(user_id).document(last.id).delete()
        return _answer_from_row(row)

    def _insert_rec_list(self, row: dict, items: list[dict]) -> None:
        self.db.collection("rec_lists").document(row["list_id"]).set({**row, "items": items})

    def _insert_feedback(self, row: dict) -> None:
        self.db.collection("rec_feedback").document(row["feedback_id"]).set(row)

    def export(self) -> dict[str, pd.DataFrame]:
        users = [d.to_dict() for d in self.db.collection("users").stream()]
        answers = [a.to_dict() for u in users for a in self._answers(u["user_id"]).stream()]
        lists, items = [], []
        for d in self.db.collection("rec_lists").stream():
            data = d.to_dict()
            for item in data.pop("items", []):
                items.append({**item, "list_id": data["list_id"]})
            lists.append(data)
        feedback = [d.to_dict() for d in self.db.collection("rec_feedback").stream()]
        rows = {"users": users, "answers": answers, "rec_lists": lists, "rec_items": items, "rec_feedback": feedback}
        return {t: _frame(rows[t], t) for t in TABLES}


def _firestore_client():
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        raw = os.environ.get("FIREBASE_CREDENTIALS")
        path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if raw:
            cred = credentials.Certificate(json.loads(raw))
        elif path:
            cred = credentials.Certificate(path)
        else:
            raise RuntimeError("Set FIREBASE_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS for Firestore.")
        firebase_admin.initialize_app(cred)
    return firestore.client()


# ================================================================= factory


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    """The process-wide backend selected by STORAGE_BACKEND (sqlite by default)."""
    backend = os.environ.get("STORAGE_BACKEND", "sqlite").lower()
    if backend == "sqlite":
        return SQLiteStorage(os.environ.get("SQLITE_PATH", config.ROOT / "data" / "app.db"))
    if backend == "firestore":
        return FirestoreStorage()
    raise ValueError(f"unknown STORAGE_BACKEND {backend!r}; expected 'sqlite' or 'firestore'")
