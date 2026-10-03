"""User journeys through the app controller, without a browser."""

from datetime import date

import numpy as np
import pytest
from conftest import OTHER, ROMANCE, SCIFI

from app.controller import Controller
from recsys import config
from recsys.selection import AB_ARMS
from recsys.session import Answer
from recsys.storage import SQLiteStorage


@pytest.fixture
def store():
    return SQLiteStorage(":memory:")


@pytest.fixture
def ctl(art, store):
    c = Controller(art, store, rng=np.random.default_rng(0), today=lambda: date(2026, 3, 1))
    c.start_new()
    return c


def swipe(ctl, n, kind="rated", rating=4):
    triggers = []
    for _ in range(n):
        assert ctl.deal() is not None
        triggers.append(ctl.answer_card(kind, rating if kind == "rated" else None))
    return [t for t in triggers if t]


# ---------------------------------------------------------------- account and seeds


def test_new_user_and_seeds(ctl, store):
    assert ctl.user.card_strategy in AB_ARMS
    assert ctl.needs_seeds
    with pytest.raises(ValueError):
        ctl.save_seeds([1, 11])  # fewer than 3
    ctl.save_seeds([1, 11, 21])
    assert not ctl.needs_seeds
    assert ctl.n_informative == 0  # seeds are not informative answers
    saved = store.export()["answers"]
    assert saved["source"].tolist() == ["seed"] * 3 and saved["rating"].tolist() == [5] * 3


def test_search_hides_answered(ctl):
    ctl.save_seeds([1, 2, 3])
    assert not {1, 2, 3} & set(ctl.search("Sci-Fi")["movie_id"])


def test_login_restores_session(art, store, ctl):
    ctl.save_seeds([1, 11, 21])
    swipe(ctl, 4)
    again = Controller(art, store)
    assert not again.login("nobody-here-11")
    assert again.login(ctl.user.handle.replace("-", " ").upper())
    assert again.session.answers == ctl.session.answers
    assert not again.needs_seeds


# ---------------------------------------------------------------- deck


def test_deal_is_stable_until_answered_and_never_repeats(ctl, store):
    ctl.save_seeds([1, 11, 21])
    seen = set()
    for _ in range(8):
        card = ctl.deal()
        assert ctl.deal() == card  # re-rendering doesn't change the card
        assert card not in seen and card not in {1, 11, 21}
        seen.add(card)
        ctl.answer_card("not_interested")
    swipes = store.export()["answers"].query("source == 'swipe'")
    assert swipes["response_ms"].notna().all()
    assert set(swipes["strategy_used"]) == {ctl.user.card_strategy}


def test_checkins_fire_once_each(ctl, monkeypatch):
    monkeypatch.setattr(config, "CHECKINS", (10, 15))  # the test catalog has only 20 movies
    ctl.save_seeds([1, 11, 21])
    assert swipe(ctl, 9) == []
    assert not ctl.unlocked and ctl.remaining_to_unlock == 1
    assert swipe(ctl, 1) == ["checkin_10"]
    assert ctl.unlocked
    ctl.undo()
    assert swipe(ctl, 1) == []  # re-reaching 10 after an undo doesn't nag again
    assert swipe(ctl, 5, kind="not_interested") == ["checkin_15"]


def test_returning_user_past_checkin_is_not_interrupted(art, store, ctl):
    ctl.save_seeds([1, 11, 21])
    swipe(ctl, 11, kind="interested")
    again = Controller(art, store)
    again.login(ctl.user.handle)
    assert swipe(again, 3, kind="interested") == []


def test_undo(ctl, store):
    ctl.save_seeds([1, 11, 21])
    assert not ctl.can_undo  # seeds can't be undone from the deck
    first = ctl.deal()
    ctl.answer_card("rated", 5)
    second = ctl.deal()
    assert ctl.can_undo
    assert ctl.undo() == first
    assert ctl.deal() == first  # the undone card is back on screen
    assert second not in ctl.session.shown_cards  # the displaced card returns to the deck
    assert len(store.export()["answers"]) == 3
    assert not ctl.can_undo


# ---------------------------------------------------------------- recommendations


def test_recommendations_flow(ctl, store):
    ctl.save_seeds([1, 2, 21])
    swipe(ctl, 10, kind="not_interested")
    cards = ctl.open_recommendations("checkin_10")
    assert 0 < len(cards) <= 10
    assert not {c.movie_id for c in cards} & ctl.session.answered_ids()
    assert cards[0].movie_id in SCIFI and cards[0].explanation.startswith("Because you love")
    assert cards[0].when == "Good for a free evening (1h 50m)"

    target = cards[0].movie_id
    ctl.answer_recommendation(target, "rated", 5)
    ctl.answer_recommendation(target, "interested")  # second answer to the same card is ignored
    assert cards[0].answer == Answer(target, "rated", 5, "recommendation")
    ctl.vote_timing(cards[1].movie_id, up=True)

    ctl.set_lambda(0.0)  # unchanged: nothing new logged
    ctl.set_lambda(0.4)
    refreshed = ctl.refresh()
    assert target not in {c.movie_id for c in refreshed}

    t = store.export()
    assert t["rec_lists"]["trigger"].tolist() == ["checkin_10", "dial", "refresh"]
    assert t["rec_lists"]["lambda"].tolist() == [0.0, 0.4, 0.4]
    assert sorted(t["rec_feedback"]["kind"]) == ["rated", "timing_up"]
    assert t["answers"].query("source == 'recommendation'")["movie_id"].tolist() == [target]
    first_list = t["rec_items"][t["rec_items"]["list_id"] == t["rec_lists"]["list_id"].iloc[0]]
    assert first_list.sort_values("rank")["movie_id"].tolist() == [c.movie_id for c in cards]


def test_watchlist_groups(ctl):
    ctl.save_seeds([1, 11, 21])
    for _ in range(3):
        ctl.deal()
        ctl.answer_card("interested")
    groups = ctl.watchlist_groups()
    assert sum(len(v) for v in groups.values()) == 3


def test_romance_fan_gets_romance(art, store):
    c = Controller(art, store, rng=np.random.default_rng(1))
    c.start_new()
    c.save_seeds(ROMANCE[:3])
    assert {card.movie_id for card in c.open_recommendations()[:3]} <= set(ROMANCE)
    assert not set(OTHER) & {card.movie_id for card in c.cards[:3]}
