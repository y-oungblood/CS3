"""Storage contract tests, run against every backend."""

import threading

import numpy as np
import pytest
from fake_firestore import FakeFirestore

from recsys import config
from recsys.handles import generate_handle, is_valid_handle, normalize_handle
from recsys.selection import AB_ARMS, assign_strategy
from recsys.session import Answer
from recsys.storage import TABLES, FirestoreStorage, RecItem, SQLiteStorage


@pytest.fixture(params=["sqlite-memory", "sqlite-file", "firestore-fake"])
def store(request, tmp_path):
    if request.param == "sqlite-memory":
        return SQLiteStorage(":memory:")
    if request.param == "sqlite-file":
        return SQLiteStorage(tmp_path / "app.db")
    return FirestoreStorage(FakeFirestore())


ANSWERS = [Answer.seed(1), Answer(2, "rated", 4), Answer(3, "interested"), Answer(4, "not_interested")]


def save_all(store, user, answers=ANSWERS):
    for pos, a in enumerate(answers):
        store.add_answer(user.user_id, a, pos, response_ms=1200 + pos, strategy_used=user.card_strategy)


# ---------------------------------------------------------------- handles


def test_handle_format():
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert is_valid_handle(generate_handle(rng))
    assert normalize_handle("  Brave Otter_42 ") == "brave-otter-42"
    assert not is_valid_handle("brave otter")


def test_assign_strategy(monkeypatch):
    rng = np.random.default_rng(0)
    assert {assign_strategy(rng) for _ in range(40)} == set(AB_ARMS)
    monkeypatch.setattr(config, "AB_TEST", False)
    assert assign_strategy(rng) == "adaptive"


# ---------------------------------------------------------------- users


def test_create_and_find_user(store):
    user = store.create_user("adaptive")
    assert is_valid_handle(user.handle)
    assert store.get_user_by_handle(user.handle) == user
    assert store.get_user_by_handle(user.handle.upper().replace("-", " ")) == user  # forgiving input
    assert store.get_user_by_handle("no-such-user-99") is None
    assert store.get_user_by_handle("") is None


def test_handle_collision_regenerates(store):
    first = store.create_user("adaptive", rng=np.random.default_rng(7))
    second = store.create_user("pop_entropy", rng=np.random.default_rng(7))  # same first draw
    assert second.handle != first.handle
    assert store.get_user_by_handle(first.handle) == first
    assert store.get_user_by_handle(second.handle) == second


# ---------------------------------------------------------------- answers


def test_session_round_trip(store):
    user = store.create_user("pop_entropy")
    save_all(store, user)
    session = store.load_session(user)
    assert session.user_id == user.user_id
    assert session.card_strategy == "pop_entropy"
    assert session.answers == ANSWERS


def test_sessions_are_per_user(store):
    a, b = store.create_user("adaptive"), store.create_user("adaptive")
    save_all(store, a)
    store.add_answer(b.user_id, Answer(99, "rated", 2), 0)
    assert store.load_session(a).answers == ANSWERS
    assert store.load_session(b).answers == [Answer(99, "rated", 2)]


def test_undo_deletes_last_answer(store):
    user = store.create_user("adaptive")
    save_all(store, user)
    assert store.delete_last_answer(user.user_id) == ANSWERS[-1]
    assert store.delete_last_answer(user.user_id) == ANSWERS[-2]
    assert store.load_session(user).answers == ANSWERS[:2]
    # The freed position is reused by the next answer.
    store.add_answer(user.user_id, Answer(5, "rated", 1), 2)
    assert store.load_session(user).answers == [*ANSWERS[:2], Answer(5, "rated", 1)]
    assert len(store.export()["answers"]) == 3


def test_undo_on_empty_history(store):
    user = store.create_user("adaptive")
    assert store.delete_last_answer(user.user_id) is None


# ---------------------------------------------------------------- logs and export


def test_rec_lists_feedback_and_export(store):
    user = store.create_user("adaptive")
    save_all(store, user)
    items = [RecItem(r, 100 + r, 0.5 - r * 0.01, f"Because #{r}") for r in range(1, 11)]
    list_id = store.log_rec_list(user.user_id, "checkin_10", 0.3, 10, items)
    store.log_rec_feedback(list_id, user.user_id, 101, "rated", 5)
    store.log_rec_feedback(list_id, user.user_id, 102, "interested")
    store.log_rec_feedback(list_id, user.user_id, 103, "timing_up")
    with pytest.raises(ValueError):
        store.log_rec_feedback(list_id, user.user_id, 104, "loved_it")

    t = store.export()
    assert set(t) == set(TABLES)
    assert t["users"]["handle"].tolist() == [user.handle]
    assert sorted(t["answers"]["position"]) == [0, 1, 2, 3]
    assert t["answers"]["response_ms"].tolist() == [1200, 1201, 1202, 1203]
    lst = t["rec_lists"].iloc[0]
    assert (lst["trigger"], lst["lambda"], lst["n_informative_answers"]) == ("checkin_10", 0.3, 10)
    items_df = t["rec_items"].sort_values("rank")
    assert items_df["movie_id"].tolist() == list(range(101, 111))
    assert (items_df["list_id"] == list_id).all()
    assert sorted(t["rec_feedback"]["kind"]) == ["interested", "rated", "timing_up"]


def test_export_columns_match_across_backends(tmp_path):
    a = SQLiteStorage(tmp_path / "a.db").export()
    b = FirestoreStorage(FakeFirestore()).export()
    assert {k: list(v.columns) for k, v in a.items()} == {k: list(v.columns) for k, v in b.items()}


# ---------------------------------------------------------------- SQLite specifics


def test_sqlite_persists_across_reopen(tmp_path):
    path = tmp_path / "app.db"
    user = SQLiteStorage(path).create_user("adaptive")
    save_all(SQLiteStorage(path), user)
    reopened = SQLiteStorage(path)
    assert reopened.get_user_by_handle(user.handle) == user
    assert reopened.load_session(user).answers == ANSWERS


def test_sqlite_concurrent_writers(tmp_path):
    store = SQLiteStorage(tmp_path / "app.db")
    users = [store.create_user("adaptive") for _ in range(8)]
    errors = []

    def work(user):
        try:
            for pos in range(25):
                store.add_answer(user.user_id, Answer(pos + 1, "rated", 3), pos)
        except Exception as e:  # pragma: no cover - surfaced by the assert below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(u,)) for u in users]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(store.export()["answers"]) == 8 * 25
