"""Fill poster_url and runtime_min in movies.parquet from the TMDB API.

Usage:
    TMDB_API_KEY=... python offline/fetch_posters.py [--artifacts data/artifacts/deploy ...]

TMDB_API_KEY may be either a v3 API key or a v4 read access token. Responses are
cached in data/raw/posters_cache.json, so an interrupted run resumes where it
stopped and later runs (or rebuilt artifacts) only fetch what is missing.

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recsys import config  # noqa: E402

API_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
CACHE_PATH = config.RAW_DIR / "posters_cache.json"
SAVE_EVERY = 200


def load_cache(path: Path) -> dict[str, dict | None]:
    """Map tmdb_id (as str) -> {"poster_path", "runtime"}, or None if TMDB has no such movie."""
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(path)  # atomic, so an interrupt never leaves a half-written cache


def make_session(key: str) -> tuple[requests.Session, dict]:
    session = requests.Session()
    if key.startswith("eyJ"):  # v4 read access token (a JWT)
        session.headers["Authorization"] = f"Bearer {key}"
        return session, {}
    return session, {"api_key": key}


def fetch_one(session: requests.Session, params: dict, tmdb_id: int) -> dict | None | bool:
    """Return the cache entry, or False on a transient failure (not cached, retried next run)."""
    for attempt in range(5):
        try:
            r = session.get(API_URL.format(tmdb_id=tmdb_id), params=params, timeout=15)
        except requests.RequestException:
            time.sleep(2**attempt)
            continue
        if r.status_code == 200:
            d = r.json()
            return {"poster_path": d.get("poster_path"), "runtime": d.get("runtime")}
        if r.status_code == 404:
            return None
        if r.status_code == 401:
            sys.exit("TMDB rejected the credentials (401). Check TMDB_API_KEY.")
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2**attempt)))
            continue
        time.sleep(2**attempt)
    return False


def apply_cache(movies: pd.DataFrame, cache: dict) -> pd.DataFrame:
    def entry(tmdb_id):
        return None if pd.isna(tmdb_id) else cache.get(str(int(tmdb_id)))

    entries = movies["tmdb_id"].map(entry)
    poster = entries.map(lambda e: f"{IMAGE_BASE}{e['poster_path']}" if e and e.get("poster_path") else pd.NA)
    runtime = entries.map(lambda e: e["runtime"] if e and e.get("runtime") else pd.NA)
    return movies.assign(
        poster_url=pd.array(poster, dtype="string"),
        runtime_min=pd.array(runtime, dtype="Int16"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--artifacts", type=Path, nargs="+",
        default=[config.ARTIFACTS_DIR / "deploy", config.ARTIFACTS_DIR / "eval"],
        help="artifact directories whose movies.parquet should be filled",
    )  # fmt: skip
    ap.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    ap.add_argument("--cache-only", action="store_true", help="apply the cache without calling TMDB")
    args = ap.parse_args()

    paths = [d / "movies.parquet" for d in args.artifacts if (d / "movies.parquet").exists()]
    if not paths:
        sys.exit("No movies.parquet found. Run offline/build_artifacts.py first.")
    cache = load_cache(CACHE_PATH)

    if not args.cache_only:
        key = os.environ.get("TMDB_API_KEY")
        if not key:
            sys.exit("Set TMDB_API_KEY (or pass --cache-only).")
        ids = set()
        for p in paths:
            ids |= set(pd.read_parquet(p, columns=["tmdb_id"])["tmdb_id"].dropna().astype(int))
        todo = sorted(i for i in ids if str(i) not in cache)
        print(f"{len(ids):,} TMDB ids, {len(todo):,} not cached", flush=True)

        session, params = make_session(key)
        failed = 0
        try:
            for n, tmdb_id in enumerate(todo, 1):
                result = fetch_one(session, params, tmdb_id)
                if result is False:
                    failed += 1
                else:
                    cache[str(tmdb_id)] = result
                if n % SAVE_EVERY == 0:
                    save_cache(cache, CACHE_PATH)
                    print(f"  {n:,}/{len(todo):,}", flush=True)
                time.sleep(args.delay)
        except KeyboardInterrupt:
            print("Interrupted; saving progress.")
        finally:
            save_cache(cache, CACHE_PATH)
        if failed:
            print(f"{failed:,} requests failed; rerun to retry them.")

    for p in paths:
        movies = apply_cache(pd.read_parquet(p), cache)
        movies.to_parquet(p, index=False)
        print(
            f"{p}: {movies['poster_url'].notna().sum():,}/{len(movies):,} posters, "
            f"{movies['runtime_min'].notna().sum():,} runtimes"
        )


if __name__ == "__main__":
    main()
