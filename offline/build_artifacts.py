"""Build the artifacts the app and the simulation load.

Usage:
    python offline/build_artifacts.py --data data/raw/ml-32m --mode deploy
    python offline/build_artifacts.py --data data/raw/ml-32m --mode eval [--seed 42]

Writes to data/artifacts/<mode>/:
    movies.parquet      one row per kept movie
    neighbors.parquet   movie_id, neighbor_id, sim (LensKit item-kNN similarities)
    meta.json           dataset statistics and build settings
    test_users.parquet  (eval mode only) held-out user IDs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recsys import config  # noqa: E402

# Half-star rating values 0.5, 1.0, ..., 5.0 map to bins 0..9.
N_RATING_BINS = 10
TITLE_RE = re.compile(r"^(?P<title>.*?)\s*\((?P<year>\d{4})(?:[-–]\d{0,4})?\)\s*$")
LEADING_ARTICLES = ("The", "A", "An")


def log(msg: str, start: float) -> None:
    print(f"[{time.perf_counter() - start:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- loading


def load_raw(data_dir: Path) -> dict[str, pd.DataFrame]:
    ratings = pd.read_csv(
        data_dir / "ratings.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
    ).rename(columns={"userId": "user_id", "movieId": "movie_id"})
    movies = pd.read_csv(data_dir / "movies.csv", dtype={"movieId": np.int32}).rename(
        columns={"movieId": "movie_id"}
    )
    links = pd.read_csv(
        data_dir / "links.csv", dtype={"movieId": np.int32, "tmdbId": "Int64"}
    ).rename(columns={"movieId": "movie_id", "tmdbId": "tmdb_id"})
    tags = pd.read_csv(
        data_dir / "tags.csv",
        usecols=["userId", "movieId", "tag"],
        dtype={"userId": np.int32, "movieId": np.int32, "tag": str},
    ).rename(columns={"userId": "user_id", "movieId": "movie_id"})
    return {"ratings": ratings, "movies": movies, "links": links, "tags": tags}


def split_users(user_ids: np.ndarray, test_fraction: float, seed: int) -> np.ndarray:
    """Return a random `test_fraction` of the distinct user IDs (sorted)."""
    users = np.unique(user_ids)
    rng = np.random.default_rng(seed)
    n_test = int(round(len(users) * test_fraction))
    return np.sort(rng.choice(users, size=n_test, replace=False))


# ---------------------------------------------------------------- movie metadata


def parse_title(raw: str) -> tuple[str, int | None]:
    """Split 'Matrix, The (1999)' into ('The Matrix', 1999)."""
    raw = raw.strip()
    m = TITLE_RE.match(raw)
    title, year = (m["title"], int(m["year"])) if m else (raw, None)
    for article in LEADING_ARTICLES:
        suffix = f", {article}"
        if title.endswith(suffix):
            title = f"{article} {title[: -len(suffix)]}"
            break
    return title, year


def parse_genres(raw: str) -> list[str]:
    return [] if raw == "(no genres listed)" else raw.split("|")


def rating_stats(ratings: pd.DataFrame, n_users: int) -> pd.DataFrame:
    """Per-movie popularity, mean rating, normalized rating entropy and novelty."""
    grouped = ratings.groupby("movie_id")["rating"]
    stats = pd.DataFrame({"popularity": grouped.size(), "mean_rating": grouped.mean()})

    # Rating histogram over the 10 half-star values, computed with one bincount.
    movie_idx = stats.index.get_indexer(ratings["movie_id"])
    bins = np.clip(np.rint(ratings["rating"].to_numpy() * 2).astype(np.int64) - 1, 0, 9)
    hist = np.bincount(
        movie_idx * N_RATING_BINS + bins, minlength=len(stats) * N_RATING_BINS
    ).reshape(len(stats), N_RATING_BINS)
    p = hist / hist.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    stats["entropy"] = ent / np.log(N_RATING_BINS)
    stats["novelty"] = -np.log2(stats["popularity"] / n_users)
    stats["mean_rating"] = stats["mean_rating"].astype(np.float32)
    return stats


def top_tags(tags: pd.DataFrame, n: int) -> pd.Series:
    """Up to `n` most-applied tags per movie, counting each user once per tag.

    Tags applied by only one user are dropped: they are mostly personal notes.
    """
    t = tags.assign(tag=tags["tag"].astype(str).str.strip().str.lower())
    t = t[t["tag"] != ""].drop_duplicates(["movie_id", "user_id", "tag"])
    counts = t.groupby(["movie_id", "tag"]).size().rename("n").reset_index()
    counts = counts[counts["n"] >= 2].sort_values(
        ["movie_id", "n", "tag"], ascending=[True, False, True]
    )
    return counts.groupby("movie_id")["tag"].apply(lambda s: list(s.head(n)))


def build_movies(
    movies: pd.DataFrame,
    links: pd.DataFrame,
    tags: pd.DataFrame,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    kept = movies[movies["movie_id"].isin(stats.index)].copy()
    parsed = kept["title"].map(parse_title)
    kept["title"] = parsed.str[0]
    kept["year"] = pd.array(parsed.str[1], dtype="Int16")
    kept["genres"] = kept["genres"].map(parse_genres)
    kept = kept.merge(links[["movie_id", "tmdb_id"]], on="movie_id", how="left")
    kept = kept.join(stats, on="movie_id")
    tag_lists = top_tags(tags[tags["movie_id"].isin(stats.index)], config.TOP_TAGS)
    kept["top_tags"] = kept["movie_id"].map(tag_lists)
    kept["top_tags"] = kept["top_tags"].map(lambda v: v if isinstance(v, list) else [])
    # Filled in by fetch_posters.py.
    kept["poster_url"] = pd.Series(pd.NA, index=kept.index, dtype="string")
    kept["runtime_min"] = pd.Series(pd.NA, index=kept.index, dtype="Int16")
    cols = [
        "movie_id", "title", "year", "genres", "tmdb_id", "popularity", "mean_rating",
        "entropy", "novelty", "top_tags", "poster_url", "runtime_min",
    ]  # fmt: skip
    return kept[cols].sort_values("movie_id").reset_index(drop=True)


# ---------------------------------------------------------------- similarity


def compute_neighbors(ratings: pd.DataFrame, k: int) -> pd.DataFrame:
    """Train LensKit's item kNN and export its similarity matrix as a long table.

    LensKit computes cosine similarity on item-mean-centered ratings and keeps
    only positive similarities.
    """
    from lenskit.data import from_interactions_df
    from lenskit.knn import ItemKNNScorer

    data = from_interactions_df(
        ratings, user_col="user_id", item_col="movie_id", rating_col="rating"
    )
    scorer = ItemKNNScorer(save_nbrs=k, feedback="explicit")
    scorer.train(data)

    smat = scorer.sim_matrix.to_scipy().tocoo()
    item_ids = np.asarray(scorer.items.ids())
    nbrs = pd.DataFrame(
        {
            "movie_id": item_ids[smat.row].astype(np.int32),
            "neighbor_id": item_ids[smat.col].astype(np.int32),
            "sim": smat.data.astype(np.float32),
        }
    )
    nbrs = nbrs[(nbrs["sim"] > 0) & (nbrs["movie_id"] != nbrs["neighbor_id"])]
    nbrs = nbrs.sort_values(["movie_id", "sim"], ascending=[True, False])
    # LensKit already truncates to save_nbrs; this guards against version changes.
    nbrs = nbrs.groupby("movie_id", sort=False).head(k)
    return nbrs.reset_index(drop=True)


# ---------------------------------------------------------------- main


def build(data_dir: Path, mode: str, seed: int, out_dir: Path) -> None:
    start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(data_dir)
    ratings, tags = raw["ratings"], raw["tags"]
    log(f"loaded {len(ratings):,} ratings from {data_dir.name}", start)

    if mode == "eval":
        test_users = split_users(ratings["user_id"].to_numpy(), config.EVAL_TEST_FRACTION, seed)
        pd.DataFrame({"user_id": test_users}).to_parquet(out_dir / "test_users.parquet", index=False)
        is_test = np.isin(ratings["user_id"].to_numpy(), test_users)
        ratings = ratings[~is_test].reset_index(drop=True)
        tags = tags[~tags["user_id"].isin(test_users)]
        log(f"held out {len(test_users):,} test users; {len(ratings):,} train ratings remain", start)

    n_users = int(ratings["user_id"].nunique())
    global_mean = float(ratings["rating"].mean())

    counts = ratings["movie_id"].value_counts()
    keep_ids = counts.index[counts >= config.MIN_RATINGS]
    ratings = ratings[ratings["movie_id"].isin(keep_ids)].reset_index(drop=True)
    log(f"kept {len(keep_ids):,} movies with >= {config.MIN_RATINGS} ratings", start)

    stats = rating_stats(ratings, n_users)
    movies = build_movies(raw["movies"], raw["links"], tags, stats)
    movies.to_parquet(out_dir / "movies.parquet", index=False)
    log(f"wrote movies.parquet ({len(movies):,} rows)", start)

    nbrs = compute_neighbors(ratings, config.K_NEIGHBORS)
    nbrs.to_parquet(out_dir / "neighbors.parquet", index=False)
    log(f"wrote neighbors.parquet ({len(nbrs):,} rows)", start)

    meta = {
        "dataset": data_dir.name,
        "mode": mode,
        "seed": seed,
        "n_users": n_users,
        "n_movies": int(len(movies)),
        "n_ratings": int(len(ratings)),
        "n_neighbor_pairs": int(len(nbrs)),
        "global_mean_rating": global_mean,
        "min_ratings": config.MIN_RATINGS,
        "k_neighbors": config.K_NEIGHBORS,
        "similarity": "lenskit ItemKNNScorer: cosine on item-mean-centered ratings",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log("wrote meta.json; run offline/fetch_posters.py to fill posters and runtimes", start)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="MovieLens directory, e.g. data/raw/ml-32m")
    ap.add_argument("--mode", choices=["deploy", "eval"], required=True)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, help="output directory (default data/artifacts/<mode>)")
    args = ap.parse_args()
    build(args.data, args.mode, args.seed, args.out or config.ARTIFACTS_DIR / args.mode)


if __name__ == "__main__":
    main()
