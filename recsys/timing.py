"""When-to-watch suggestions.

Rule-based on purpose: MovieLens timestamps record when a movie was rated, not
when it was watched (Harper & Konstan 2015, section 3.2), so there is no data
to learn viewing times from.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import pandas as pd

from recsys import config

TONIGHT, WEEKEND, ANYTIME = "Tonight", "This weekend", "Anytime"


def format_runtime(minutes: int) -> str:
    h, m = divmod(int(minutes), 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def _runtime(movie: Mapping) -> int | None:
    r = movie.get("runtime_min")
    return None if r is None or pd.isna(r) or r <= 0 else int(r)


def _list(movie: Mapping, key: str) -> list:
    """List-valued field; parquet returns numpy arrays, which have no truth value."""
    v = movie.get(key)
    return [] if v is None else list(v)


def _has_tag(movie: Mapping, word: str) -> bool:
    return any(word in t for t in _list(movie, "top_tags"))


def when_to_watch(movie: Mapping, today: date | None = None) -> str | None:
    """One short line for a recommendation card, or None when there is nothing to say."""
    today = today or date.today()
    genres = _list(movie, "genres")
    if today.month == 10 and ("Horror" in genres or _has_tag(movie, "halloween")):
        return "Perfect for spooky season"
    if today.month == 12 and _has_tag(movie, "christmas"):
        return "A holiday-season watch"
    runtime = _runtime(movie)
    if runtime is None:
        return None
    if runtime >= config.LONG_RUNTIME:
        return f"Save it for the weekend ({format_runtime(runtime)})"
    if runtime <= config.SHORT_RUNTIME:
        return f"Easy weeknight pick ({format_runtime(runtime)})"
    return f"Good for a free evening ({format_runtime(runtime)})"


def watch_slot(movie: Mapping) -> str:
    """Watchlist group: Tonight (short), This weekend (long), Anytime (the rest)."""
    runtime = _runtime(movie)
    if runtime is None:
        return ANYTIME
    if runtime <= config.SHORT_RUNTIME:
        return TONIGHT
    if runtime >= config.LONG_RUNTIME:
        return WEEKEND
    return ANYTIME


def group_watchlist(movies: list[Mapping]) -> dict[str, list[Mapping]]:
    groups: dict[str, list[Mapping]] = {TONIGHT: [], WEEKEND: [], ANYTIME: []}
    for m in movies:
        groups[watch_slot(m)].append(m)
    return groups
