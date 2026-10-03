"""Load artifacts once and build the indexes scoring and card selection need.

Movies are addressed internally by position 0..N-1 (sorted by movie_id). The
neighbor table is stored twice in CSR form:

- forward: row i lists the neighbors of movie i (used by adaptive card selection)
- reverse: row j lists the movies that have j as a neighbor (used by scoring)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from recsys import config


@dataclass(frozen=True)
class CSR:
    indptr: np.ndarray   # int64, length n_rows + 1
    indices: np.ndarray  # int32 column positions
    data: np.ndarray     # float32 similarities

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.indptr[i], self.indptr[i + 1]
        return self.indices[lo:hi], self.data[lo:hi]

    def gather(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All entries in `rows`, as (row position in `rows`, column, value) arrays."""
        starts, ends = self.indptr[rows], self.indptr[rows + 1]
        lengths = ends - starts
        owner = np.repeat(np.arange(len(rows)), lengths)
        offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        flat = np.repeat(starts, lengths) + offsets
        return owner, self.indices[flat], self.data[flat]


def _csr(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n: int) -> CSR:
    order = np.lexsort((-vals, rows))  # by row, then descending similarity
    rows, cols, vals = rows[order], cols[order], vals[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return CSR(indptr, cols.astype(np.int32), vals.astype(np.float32))


class Artifacts:
    def __init__(self, movies: pd.DataFrame, neighbors: pd.DataFrame, meta: dict):
        movies = movies.sort_values("movie_id").reset_index(drop=True)
        self.movies = movies
        self.meta = meta
        self.global_mean = float(meta["global_mean_rating"])
        self.movie_ids = movies["movie_id"].to_numpy(dtype=np.int64)
        self.n = len(movies)

        self.popularity = movies["popularity"].to_numpy(dtype=np.float64)
        self.entropy = movies["entropy"].to_numpy(dtype=np.float64)
        self.novelty = movies["novelty"].to_numpy(dtype=np.float64)
        self.mean_rating = movies["mean_rating"].to_numpy(dtype=np.float64)
        self.first_genre = np.array([g[0] if len(g) else "" for g in movies["genres"]], dtype=object)
        # How likely a user has seen the movie, times how much their answer would tell us.
        self.informativeness = np.log(self.popularity) * self.entropy

        by_pop = np.argsort(-self.popularity, kind="stable")
        self.popularity_order = by_pop
        elicit = by_pop[: config.ELICIT_POOL]
        self.elicit_order = elicit[np.argsort(-self.informativeness[elicit], kind="stable")]
        top = by_pop[: config.FALLBACK_POPULAR]
        self.fallback_order = top[np.argsort(-self.mean_rating[top], kind="stable")]

        # Title search runs on every keystroke, so its inputs are prepared once.
        self._by_popularity = movies.iloc[by_pop].reset_index(drop=True)
        self._titles_lower = self._by_popularity["title"].str.lower()

        nb = neighbors[neighbors["movie_id"].isin(self.movie_ids) & neighbors["neighbor_id"].isin(self.movie_ids)]
        i = self.index_of(nb["movie_id"].to_numpy())
        j = self.index_of(nb["neighbor_id"].to_numpy())
        sim = nb["sim"].to_numpy(dtype=np.float32)
        self.forward = _csr(i, j, sim, self.n)
        self.reverse = _csr(j, i, sim, self.n)

    def index_of(self, movie_ids) -> np.ndarray:
        """Positions of movie IDs; raises KeyError for unknown IDs."""
        ids = np.asarray(movie_ids, dtype=np.int64)
        pos = np.searchsorted(self.movie_ids, ids)
        pos = np.minimum(pos, self.n - 1)
        if not np.array_equal(self.movie_ids[pos], ids):
            missing = ids[self.movie_ids[pos] != ids]
            raise KeyError(f"unknown movie_id(s): {missing[:5].tolist()}")
        return pos

    def movie(self, movie_id: int) -> pd.Series:
        return self.movies.iloc[int(self.index_of([movie_id])[0])]

    def search(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Title search for seed picks: prefix matches first, then substring, by popularity."""
        q = query.strip().lower()
        if not q:
            return self.movies.iloc[0:0]
        titles = self._titles_lower  # already in popularity order
        contains = titles.str.contains(q, regex=False).to_numpy()
        prefix = titles.str.startswith(q).to_numpy()
        order = np.concatenate([np.flatnonzero(prefix), np.flatnonzero(contains & ~prefix)])[:limit]
        return self._by_popularity.iloc[order]


def load_dir(path: Path) -> Artifacts:
    path = Path(path)
    return Artifacts(
        pd.read_parquet(path / "movies.parquet"),
        pd.read_parquet(path / "neighbors.parquet"),
        json.loads((path / "meta.json").read_text()),
    )


@lru_cache(maxsize=2)
def load(mode: str = "deploy") -> Artifacts:
    """Load data/artifacts/<mode> once per process."""
    return load_dir(config.ARTIFACTS_DIR / mode)
