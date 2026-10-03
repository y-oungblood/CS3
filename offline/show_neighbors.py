"""Print the nearest neighbors of movies by title, to sanity-check artifacts.

Usage:
    python offline/show_neighbors.py "The Dark Knight" "Toy Story" [--artifacts data/artifacts/deploy] [-n 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recsys import config  # noqa: E402


def neighbors_for(title: str, movies: pd.DataFrame, nbrs: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Neighbors of the most popular movie whose title contains `title` (case-insensitive)."""
    matches = movies[movies["title"].str.contains(title, case=False, regex=False)]
    if matches.empty:
        raise KeyError(title)
    movie = matches.sort_values("popularity", ascending=False).iloc[0]
    rows = nbrs[nbrs["movie_id"] == movie["movie_id"]].nlargest(n, "sim")
    out = rows.merge(movies, left_on="neighbor_id", right_on="movie_id", suffixes=("", "_n"))
    out.attrs["movie"] = f"{movie['title']} ({movie['year']})"
    return out[["title", "year", "sim", "popularity"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("titles", nargs="+")
    ap.add_argument("--artifacts", type=Path, default=config.ARTIFACTS_DIR / "deploy")
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()
    movies = pd.read_parquet(args.artifacts / "movies.parquet")
    nbrs = pd.read_parquet(args.artifacts / "neighbors.parquet")
    for title in args.titles:
        try:
            table = neighbors_for(title, movies, nbrs, args.n)
        except KeyError:
            print(f"\nNo movie matching {title!r}")
            continue
        print(f"\n{table.attrs['movie']}")
        print(table.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
