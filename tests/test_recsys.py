"""Unit tests for recsys/."""

from datetime import date

import numpy as np
import pytest
from conftest import OTHER, ROMANCE, SCIFI

from recsys import config
from recsys.rerank import rerank
from recsys.scoring import FALLBACK_TEXT, explain, recommend, user_mean, weights
from recsys.selection import STRATEGIES, next_card
from recsys.session import Answer, Session
from recsys.timing import ANYTIME, TONIGHT, WEEKEND, group_watchlist, when_to_watch


def ids(cands, art):
    return cands.movie_ids(art)


# ---------------------------------------------------------------- session


def test_answer_validation():
    with pytest.raises(ValueError):
        Answer(1, "rated", None)
    with pytest.raises(ValueError):
        Answer(1, "rated", 6)
    with pytest.raises(ValueError):
        Answer(1, "interested", 3)


def test_session_helpers():
    s = Session("u")
    s.add(Answer.seed(1))
    s.show(2)
    s.add(Answer(2, "interested"))
    s.add(Answer(3, "rated", 4))
    assert s.answered_ids() == {1, 2, 3}
    assert s.watchlist() == [2]
    assert s.n_informative_answers() == 2
    assert 2 not in s.shown_cards  # answering clears the shown mark
    with pytest.raises(ValueError):
        s.add(Answer(3, "rated", 5))
    assert s.undo() == Answer(3, "rated", 4)
    assert s.answered_ids() == {1, 2}


# ---------------------------------------------------------------- weights


def test_user_mean_shrinkage():
    s = Session("u")
    assert user_mean(s, 3.5) == pytest.approx(3.5)  # no ratings: global mean
    s.add(Answer.seed(1))
    s.add(Answer(2, "rated", 5))
    k = config.K_SHRINK
    expected = (5 + 5 + k * 3.5) / (2 + k)
    assert 3.5 < user_mean(s, 3.5) == pytest.approx(expected)
    s.add(Answer(3, "interested"))  # non-ratings don't move the mean
    assert user_mean(s, 3.5) == pytest.approx(expected)


def test_weights_by_answer_kind():
    s = Session("u")
    s.add(Answer.seed(1))
    s.add(Answer(2, "interested"))
    s.add(Answer(3, "not_interested"))
    s.add(Answer(4, "unseen_unknown"))
    w = weights(s, 3.5)
    mu = (5 + config.K_SHRINK * 3.5) / (1 + config.K_SHRINK)
    assert w == pytest.approx({1: 5 - mu, 2: config.W_INTERESTED, 3: -config.W_INTERESTED})
    assert 4 not in w


# ---------------------------------------------------------------- recommend


def test_liking_a_movie_ranks_its_neighbors_highly(art):
    s = Session("u")
    s.add(Answer.seed(1))
    top = ids(recommend(s, art), art)
    assert top[:5] == sorted(set(SCIFI) - {1}, key=lambda m: abs(m - 1))  # closest similarity first


def test_rating_low_pushes_neighbors_down(art):
    liked = Session("a")
    liked.add(Answer.seed(1))
    liked.add(Answer(11, "rated", 5))
    disliked = Session("b")
    disliked.add(Answer.seed(1))
    disliked.add(Answer(11, "rated", 1))

    assert set(ROMANCE[1:]) <= set(ids(recommend(liked, art), art))
    pool = ids(recommend(disliked, art), art)
    assert not set(ROMANCE) & set(pool)  # negative scores are not candidates
    assert set(pool) >= set(SCIFI) - {1}


def test_answered_movies_never_recommended(art):
    s = Session("u")
    for a in (Answer.seed(1), Answer(2, "interested"), Answer(3, "not_interested"), Answer(12, "rated", 4)):
        s.add(a)
    assert not s.answered_ids() & set(ids(recommend(s, art), art))


def test_fallback_when_no_signal(art):
    pool = recommend(Session("u"), art)
    assert pool.fallback.all()
    # Fallback = top-rated among the most popular, in mean-rating order.
    means = art.mean_rating[pool.positions]
    assert np.all(np.diff(means) <= 0)
    assert explain(int(art.movie_ids[pool.positions[0]]), Session("u"), art).text == FALLBACK_TEXT


def test_fallback_fills_sparse_pool(art):
    s = Session("u")
    s.add(Answer(2, "interested"))  # 5 positive candidates < MIN_POSITIVE_CANDIDATES
    pool = recommend(s, art)
    n_scored = (~pool.fallback).sum()
    assert n_scored == 5
    assert pool.fallback[n_scored:].all()
    assert np.all(pool.scores[:n_scored] > pool.scores[n_scored:].max())


def test_undo_restores_previous_state(art):
    s = Session("u")
    s.add(Answer.seed(1))
    before = ids(recommend(s, art), art)
    s.add(Answer(11, "rated", 5))
    assert ids(recommend(s, art), art) != before
    s.undo()
    assert ids(recommend(s, art), art) == before


# ---------------------------------------------------------------- rerank


def test_lambda_zero_preserves_relevance_order(art):
    s = Session("u")
    s.add(Answer.seed(1))
    s.add(Answer.seed(11))
    pool = recommend(s, art)
    assert ids(rerank(pool, 0.0, art), art) == ids(pool, art)


def test_lambda_is_clipped(art):
    s = Session("u")
    s.add(Answer.seed(1))
    pool = recommend(s, art)
    assert ids(rerank(pool, 5.0, art), art) == ids(rerank(pool, config.LAMBDA_MAX, art), art)


def test_higher_lambda_raises_novelty(art):
    s = Session("u")
    s.add(Answer.seed(1))
    s.add(Answer.seed(11))
    pool = recommend(s, art)

    def mean_novelty(lam, k=4):
        return art.novelty[rerank(pool, lam, art).positions[:k]].mean()

    assert mean_novelty(config.LAMBDA_MAX) > mean_novelty(0.0)


def test_higher_lambda_raises_novelty_real(real_art):
    s = Session("u")
    for title in ("The Dark Knight", "Toy Story", "Pulp Fiction"):
        s.add(Answer.seed(int(real_art.search(title, 1)["movie_id"].iloc[0])))
    pool = recommend(s, real_art)
    novelty = [real_art.novelty[rerank(pool, lam, real_art).positions[:10]].mean() for lam in (0, 0.3, 0.6)]
    assert novelty[0] < novelty[1] < novelty[2]


# ---------------------------------------------------------------- explain


def test_explain_names_only_positive_contributions(art):
    s = Session("u")
    s.add(Answer.seed(1))
    s.add(Answer(2, "rated", 1))  # negative weight: must never be named
    s.add(Answer(4, "interested"))
    e = explain(3, s, art)
    assert set(e.reason_ids) == {1, 4}
    assert 2 not in e.reason_ids
    assert e.text == "Because you love Sci-Fi 1 and added Sci-Fi 4 to your list."
    assert e.tags == "space · cerebral"


def test_explain_rated_phrase_and_single_reason(art):
    s = Session("u")
    s.add(Answer(12, "rated", 5))
    e = explain(13, s, art)
    assert e.text == "Because you rated Romance 12 ★5."
    assert e.reason_ids == (12,)


def test_explain_merges_same_kind_reasons(art):
    s = Session("u")
    s.add(Answer(12, "rated", 5))
    s.add(Answer(14, "rated", 4))
    assert explain(13, s, art).text == "Because you rated Romance 12 ★5 and Romance 14 ★4."
    s2 = Session("v")
    s2.add(Answer.seed(12))
    s2.add(Answer.seed(14))
    assert explain(13, s2, art).text == "Because you love Romance 12 and Romance 14."


# ---------------------------------------------------------------- card selection


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_cards_never_repeat_answered_or_shown(art, strategy):
    rng = np.random.default_rng(0)
    s = Session("u", strategy)
    s.add(Answer.seed(1))
    s.show(OTHER[0])
    seen = set()
    kinds = [Answer(0, "interested"), Answer(0, "not_interested"), Answer(0, "rated", 2), Answer(0, "rated", 5)]
    while (card := next_card(s, art, rng=rng)) is not None:
        assert card not in s.excluded_ids() and card not in seen
        seen.add(card)
        template = kinds[len(seen) % len(kinds)]
        s.add(Answer(card, template.kind, template.rating))
    assert len(seen) == art.n - 2  # every other movie was offered exactly once


def test_exclude_hides_movies(art):
    s = Session("u", "popularity")
    first = next_card(s, art)
    assert next_card(s, art, exclude={first}) != first


def test_popularity_and_pop_entropy(art):
    s = Session("u")
    assert next_card(s, art, "popularity") == int(art.movie_ids[np.argmax(art.popularity)])
    pick = next_card(s, art, "pop_entropy")
    scores = np.log(art.popularity) * art.entropy
    assert pick == int(art.movie_ids[np.argmax(scores)])


def test_adaptive_phase1_diversity(art):
    s = Session("u", "adaptive")
    rng = np.random.default_rng(0)
    genres = []
    for _ in range(4):
        card = next_card(s, art, rng=rng)
        genres.append(art.movie(card)["genres"][0])
        s.add(Answer(card, "rated", 3))
    # With a window of 3, the first three cards all have different first genres.
    assert len(set(genres[:3])) == 3


def test_adaptive_phase2_follows_likes(art, monkeypatch):
    monkeypatch.setattr(config, "WILDCARD_RATE", 0.0)
    s = Session("u", "adaptive")
    s.add(Answer.seed(11))
    for m in (OTHER + SCIFI)[: config.PHASE1_ANSWERS]:
        s.add(Answer(m, "not_interested"))
    assert next_card(s, art, rng=np.random.default_rng(0)) in ROMANCE


# ---------------------------------------------------------------- timing


def movie(runtime=110, genres=("Drama",), tags=()):
    return {"runtime_min": runtime, "genres": list(genres), "top_tags": list(tags)}


@pytest.mark.parametrize(
    "m, today, expected",
    [
        (movie(genres=["Horror"]), date(2026, 10, 15), "Perfect for spooky season"),
        (movie(tags=["halloween night"]), date(2026, 10, 1), "Perfect for spooky season"),
        (movie(genres=["Horror"], runtime=None), date(2026, 10, 15), "Perfect for spooky season"),
        (movie(genres=["Horror"]), date(2026, 3, 1), "Good for a free evening (1h 50m)"),
        (movie(tags=["christmas"]), date(2026, 12, 20), "A holiday-season watch"),
        (movie(tags=["christmas"]), date(2026, 10, 20), "Good for a free evening (1h 50m)"),
        (movie(runtime=config.LONG_RUNTIME), date(2026, 3, 1), "Save it for the weekend (2h 10m)"),
        (movie(runtime=169), date(2026, 3, 1), "Save it for the weekend (2h 49m)"),
        (movie(runtime=config.SHORT_RUNTIME), date(2026, 3, 1), "Easy weeknight pick (1h 40m)"),
        (movie(runtime=45), date(2026, 3, 1), "Easy weeknight pick (45m)"),
        (movie(runtime=None), date(2026, 3, 1), None),
        (movie(runtime=0), date(2026, 3, 1), None),
    ],
)
def test_when_to_watch(m, today, expected):
    assert when_to_watch(m, today) == expected


def test_when_to_watch_accepts_parquet_rows(real_art):
    row = real_art.movies.iloc[0]  # genres/top_tags are numpy arrays here
    when_to_watch(row, date(2026, 10, 1))


def test_group_watchlist():
    groups = group_watchlist([movie(90), movie(140), movie(115), movie(None)])
    assert [len(groups[k]) for k in (TONIGHT, WEEKEND, ANYTIME)] == [1, 1, 2]
