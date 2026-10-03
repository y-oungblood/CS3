"""A small synthetic catalog with known structure, for testing recsys/.

- Sci-fi cluster: movies 1-6, all similar to each other, popularity falling 1 -> 6
- Romance cluster: movies 11-16, same shape
- Unrelated comedies/dramas: movies 21-28 with no neighbors (fallback material)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recsys.artifacts import Artifacts, load_dir  # noqa: E402

N_USERS = 2000
SCIFI = [1, 2, 3, 4, 5, 6]
ROMANCE = [11, 12, 13, 14, 15, 16]
OTHER = [21, 22, 23, 24, 25, 26, 27, 28]


def _movies() -> pd.DataFrame:
    rows = []
    for cluster, genre, tags in ((SCIFI, "Sci-Fi", ["space", "cerebral"]), (ROMANCE, "Romance", ["sweet"])):
        for rank, mid in enumerate(cluster):
            rows.append((mid, f"{genre} {mid}", [genre, "Drama"], 1000 - rank * 180, 3.5 + rank * 0.05, tags))
    for k, mid in enumerate(OTHER):
        genre = "Comedy" if k % 2 else "Drama"
        rows.append((mid, f"Other {mid}", [genre], 1500 - k * 100, 4.4 - k * 0.1, []))
    df = pd.DataFrame(rows, columns=["movie_id", "title", "genres", "popularity", "mean_rating", "top_tags"])
    df["year"] = 2000
    df["entropy"] = np.linspace(0.6, 0.9, len(df))
    df["novelty"] = -np.log2(df["popularity"] / N_USERS)
    df["tmdb_id"] = df["movie_id"] + 1000
    df["poster_url"] = None
    df["runtime_min"] = 110
    return df


def _neighbors() -> pd.DataFrame:
    rows = []
    for cluster in (SCIFI, ROMANCE):
        for i in cluster:
            for j in cluster:
                if i != j:
                    rows.append((i, j, 0.6 - 0.05 * abs(i - j)))
    return pd.DataFrame(rows, columns=["movie_id", "neighbor_id", "sim"])


@pytest.fixture
def art() -> Artifacts:
    return Artifacts(_movies(), _neighbors(), {"global_mean_rating": 3.5, "n_users": N_USERS})


@pytest.fixture(scope="session")
def real_art():
    path = Path(__file__).resolve().parent.parent / "data" / "artifacts" / "deploy"
    if not (path / "movies.parquet").exists():
        pytest.skip("deploy artifacts not built")
    return load_dir(path)
