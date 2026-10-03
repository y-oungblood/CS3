import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "offline"))
import build_artifacts as ba  # noqa: E402
import fetch_posters as fp  # noqa: E402


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Toy Story (1995)", ("Toy Story", 1995)),
        ("Matrix, The (1999)", ("The Matrix", 1999)),
        ("Few Good Men, A (1992)", ("A Few Good Men", 1992)),
        ("Babylon 5 (1993-1998)", ("Babylon 5", 1993)),
        ("Black Mirror", ("Black Mirror", None)),
        ("1984 (1984) ", ("1984", 1984)),
    ],
)
def test_parse_title(raw, expected):
    assert ba.parse_title(raw) == expected


def test_parse_genres():
    assert ba.parse_genres("Action|Sci-Fi") == ["Action", "Sci-Fi"]
    assert ba.parse_genres("(no genres listed)") == []


def test_rating_stats_entropy_and_novelty():
    ratings = pd.DataFrame(
        {
            "movie_id": [1, 1, 1, 1, 2, 2],
            # movie 1: all 4.0 -> entropy 0; movie 2: two distinct values -> log(2)/log(10)
            "rating": np.array([4.0, 4.0, 4.0, 4.0, 0.5, 5.0], dtype=np.float32),
        }
    )
    stats = ba.rating_stats(ratings, n_users=8)
    assert stats.loc[1, "popularity"] == 4
    assert stats.loc[1, "entropy"] == pytest.approx(0.0)
    assert stats.loc[2, "entropy"] == pytest.approx(np.log(2) / np.log(10))
    assert stats.loc[1, "novelty"] == pytest.approx(1.0)  # -log2(4/8)
    assert stats.loc[2, "mean_rating"] == pytest.approx(2.75)
    assert stats["entropy"].between(0, 1).all()


def test_split_users_is_deterministic_and_sized():
    users = np.repeat(np.arange(100), 3)
    a = ba.split_users(users, 0.2, seed=1)
    b = ba.split_users(users, 0.2, seed=1)
    assert len(a) == 20 and np.array_equal(a, b)
    assert len(np.unique(a)) == 20


def test_top_tags_counts_users_once_and_drops_singletons():
    tags = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 3, 2, 4],
            "movie_id": [10] * 7,
            "tag": ["Dark", "dark", "dark ", "Dark", "dark", "funny", "funny"],
        }
    )
    tags.loc[len(tags)] = [5, 10, "one-off"]
    result = ba.top_tags(tags, 3)
    assert result[10] == ["dark", "funny"]


def test_apply_cache_builds_urls_and_nulls():
    movies = pd.DataFrame({"movie_id": [1, 2, 3, 4], "tmdb_id": pd.array([11, 22, None, 44], dtype="Int64")})
    cache = {"11": {"poster_path": "/a.jpg", "runtime": 120}, "22": None, "44": {"poster_path": None, "runtime": 0}}
    out = fp.apply_cache(movies, cache)
    assert out.loc[0, "poster_url"] == fp.IMAGE_BASE + "/a.jpg"
    assert out.loc[0, "runtime_min"] == 120
    assert out["poster_url"].isna().tolist() == [False, True, True, True]
    assert out["runtime_min"].isna().tolist() == [False, True, True, True]
