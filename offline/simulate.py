"""Offline simulation: how fast does each card-selection strategy learn a user's taste?

Usage:
    python offline/simulate.py --data data/raw/ml-32m --artifacts data/artifacts/eval
    python offline/simulate.py --data data/raw/ml-latest-small --artifacts data/artifacts/eval-small --users 50
    python offline/simulate.py --plots-only --out results/ml-32m

Held-out test users (never seen when the neighbor table was built) replay their
real MovieLens ratings through the same recsys/ code the app uses:

1. A random 20% of the movies they rated >= 4 are hidden as the test set.
2. Three movies they rated >= 4.5 (outside the test set) become seed picks.
3. Each strategy shows 50 cards. A card the user rated is answered with that
   rating (half stars rounded up, e.g. 3.5 -> 4); otherwise "unseen_unknown".
4. Every 5 swipes, the top 10 recommendations are scored against the hidden
   test set with LensKit's nDCG, recall and precision.

Test movies and cards (--test-cards):
- sampled (default): each user gets a fixed evaluation set: the hidden test
  movies plus N_NEGATIVES movies they never rated, sampled in proportion to
  popularity. Cards never show any of them (test or not), and every checkpoint
  ranks that same set. Nothing is conditioned on test membership, so nothing
  leaks, and cards can't use up hits, so the metric isolates how much the
  answers taught the model. This is the main comparison of strategies.
- reveal: cards come from the whole catalog. A card that is a test
  movie is answered honestly and leaves that strategy's test set, as in the app,
  where a movie the user already told us about can't be a recommendation hit.
  Card choice is independent of the test set, so nothing leaks, but strategies
  that ask about likely favorites (adaptive) use up the hits they would have
  recommended.
- exclude: test movies are never shown as cards. This leaks: each swipe removes
  a non-test movie from the candidates, so the remaining popular movies become
  disproportionately test movies, which inflates popularity-like recommenders.
  Kept only to show the effect.

Two LensKit reference lines use the adaptive session's answers: PopScorer
(non-personalized floor) and the stock ItemKNNScorer pipeline (the model whose
similarities the app uses, without our damping and weighting). A dial
experiment sweeps lambda after 25 adaptive swipes.

Outputs in --out (default results/<dataset>): checkpoints.csv, dial.csv,
summary.csv, summary.json and three PNG plots.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recsys import config  # noqa: E402
from recsys.artifacts import Artifacts, load_dir  # noqa: E402
from recsys.rerank import rerank  # noqa: E402
from recsys.scoring import Candidates, recommend  # noqa: E402
from recsys.selection import STRATEGIES, next_card  # noqa: E402
from recsys.session import Answer, Session  # noqa: E402

TOP_N = 10
MIN_USER_RATINGS = 50
TEST_FRACTION = 0.2
N_SEEDS = 3
N_NEGATIVES = 100
CHECKPOINT_EVERY = 5
DIAL_AT = 25
LAMBDAS = np.round(np.arange(0, config.LAMBDA_MAX + 1e-9, 0.1), 1)
REF_POP, REF_KNN = "lenskit_pop", "lenskit_itemknn"


def log(msg: str, start: float) -> None:
    print(f"[{time.perf_counter() - start:7.1f}s] {msg}", flush=True)


def round_rating(r: float) -> int:
    """MovieLens half stars -> app whole stars, rounding halves up (3.5 -> 4)."""
    return int(np.clip(np.floor(r + 0.5), 1, 5))


# ---------------------------------------------------------------- data


def load_ratings(data_dir: Path, art: Artifacts) -> pd.DataFrame:
    ratings = pd.read_csv(
        data_dir / "ratings.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
    ).rename(columns={"userId": "user_id", "movieId": "movie_id"})
    # Only catalog movies: the model can't recommend or show anything else.
    return ratings[ratings["movie_id"].isin(art.movie_ids)].reset_index(drop=True)


def pick_users(test_ratings: pd.DataFrame, art: Artifacts, n_users: int, seed: int) -> list[dict]:
    """Sample test users and build each one's hidden test set, seed picks and negatives."""
    rng = np.random.default_rng(seed)
    pop_p = art.popularity / art.popularity.sum()
    counts = test_ratings["user_id"].value_counts()
    eligible = counts.index[counts >= MIN_USER_RATINGS].to_numpy().copy()
    rng.shuffle(eligible)
    by_user = {u: g for u, g in test_ratings[test_ratings["user_id"].isin(eligible)].groupby("user_id")}

    users = []
    for uid in eligible:
        r = by_user[uid].set_index("movie_id")["rating"]
        relevant = r.index[r >= 4].to_numpy()
        if len(relevant) == 0:
            continue
        n_test = max(1, int(round(len(relevant) * TEST_FRACTION)))
        test = set(rng.choice(relevant, size=n_test, replace=False).tolist())
        seed_pool = np.array([m for m in r.index[r >= 4.5] if m not in test])
        if len(seed_pool) < N_SEEDS:
            continue
        seeds = rng.choice(seed_pool, size=N_SEEDS, replace=False).tolist()
        # Unrated movies, sampled by popularity so they are plausible distractors.
        unrated = ~np.isin(art.movie_ids, r.index.to_numpy())
        p = np.where(unrated, pop_p, 0.0)
        n_neg = min(N_NEGATIVES, int(unrated.sum()))  # tiny dev catalogs may have fewer
        negatives = set(rng.choice(art.movie_ids, size=n_neg, replace=False, p=p / p.sum()).tolist())
        users.append(
            {"user_id": int(uid), "ratings": r.to_dict(), "test": test, "seeds": seeds, "negatives": negatives}
        )
        if len(users) == n_users:
            break
    return users


# ---------------------------------------------------------------- LensKit


class LensKitRefs:
    """LensKit reference recommenders and metrics (training happens offline only)."""

    def __init__(self, train_ratings: pd.DataFrame, catalog: np.ndarray):
        from lenskit.basic import PopScorer
        from lenskit.data import from_interactions_df
        from lenskit.knn import ItemKNNScorer
        from lenskit.metrics import NDCG, Precision, Recall
        from lenskit.pipeline import topn_pipeline

        data = from_interactions_df(train_ratings, user_col="user_id", item_col="movie_id", rating_col="rating")
        self.pipes = {
            REF_POP: topn_pipeline(PopScorer()),
            REF_KNN: topn_pipeline(ItemKNNScorer(save_nbrs=config.K_NEIGHBORS, feedback="explicit")),
        }
        for pipe in self.pipes.values():
            pipe.train(data)
        self.catalog = catalog
        self.metrics = {"ndcg": NDCG(TOP_N), "recall": Recall(TOP_N), "precision": Precision(TOP_N)}

    def recommend(self, name: str, session: Session, candidates: set[int] | None = None) -> list[int]:
        from lenskit import recommend as lk_recommend
        from lenskit.data import ItemList, RecQuery

        rated = [a for a in session.answers if a.kind == "rated"]
        history = ItemList(item_ids=[a.movie_id for a in rated], rating=[float(a.rating) for a in rated])
        pool = self.catalog if candidates is None else np.array(sorted(candidates))
        allowed = ItemList(item_ids=np.setdiff1d(pool, list(session.answered_ids())))
        recs = lk_recommend(self.pipes[name], RecQuery(history_items=history), n=TOP_N, items=allowed)
        return [int(i) for i in recs.ids()]

    def score(self, rec_ids: list[int], test: set[int]) -> dict[str, float]:
        from lenskit.data import ItemList

        recs = ItemList(item_ids=rec_ids, ordered=True)
        truth = ItemList(item_ids=sorted(test))
        return {k: float(m.measure_list(recs, truth)) for k, m in self.metrics.items()}


# ---------------------------------------------------------------- simulation


def answer_for(user: dict, movie_id: int) -> Answer:
    r = user["ratings"].get(movie_id)
    if r is None:
        return Answer(movie_id, "unseen_unknown")
    return Answer(movie_id, "rated", round_rating(r))


def top_within(cands: Candidates, art: Artifacts, items: set[int] | None) -> list[int]:
    """Top-N movie IDs from a ranked pool, optionally restricted to an evaluation set.

    Evaluation-set movies outside the pool (no positive score) follow in popularity order.
    """
    ids = cands.movie_ids(art)
    if items is None:
        return ids[:TOP_N]
    ranked = [m for m in ids if m in items]
    if len(ranked) < TOP_N:
        rest = sorted(items - set(ranked), key=lambda m: -art.popularity[art.index_of([m])[0]])
        ranked += rest
    return ranked[:TOP_N]


def simulate_user(
    user: dict, art: Artifacts, refs: LensKitRefs, seed: int, test_cards: str = "sampled"
) -> tuple[list[dict], list[dict], dict]:
    rows, dial_rows, dial_recs = [], [], {}
    eval_set = user["test"] | user["negatives"] if test_cards == "sampled" else None

    for s_idx, strategy in enumerate(STRATEGIES):
        rng = np.random.default_rng([seed, user["user_id"], s_idx])
        test = set(user["test"])  # shrinks in reveal mode as test movies are shown
        card_exclude = {"sampled": eval_set, "exclude": test, "reveal": set()}[test_cards]
        session = Session(str(user["user_id"]), strategy)
        for m in user["seeds"]:
            session.add(Answer.seed(m))

        def checkpoint(swipes: int) -> None:
            swiped = [a for a in session.answers if a.source == "swipe"]
            unseen = sum(a.kind == "unseen_unknown" for a in swiped)
            if not test:
                return  # every test movie was revealed by a card: nothing left to predict
            stats = {
                "user_id": user["user_id"],
                "n_test": len(test),
                "swipes": swipes,
                "n_rated": len(swiped) - unseen,
                "unseen_frac": unseen / len(swiped) if swiped else np.nan,
            }
            # Sampled mode ranks the whole evaluation set, so the pool must cover it.
            pool = recommend(session, art, n=art.n if eval_set else config.CANDIDATE_POOL)
            top = top_within(pool, art, eval_set)
            rows.append({**stats, "strategy": strategy, **refs.score(top, test)})
            if strategy == "adaptive":
                for ref in (REF_POP, REF_KNN):
                    recs = refs.recommend(ref, session, eval_set)
                    rows.append({**stats, "strategy": ref, **refs.score(recs, test)})
                if swipes == DIAL_AT:
                    # What the user would see: the app's pool of CANDIDATE_POOL, re-ranked.
                    app_pool = recommend(session, art)
                    for lam in LAMBDAS:
                        shown = rerank(app_pool, lam, art)
                        ids = shown.movie_ids(art)[:TOP_N]
                        dial_recs[float(lam)] = ids
                        # Accuracy on the evaluation set (sampled) or the full catalog.
                        scored = ids
                        if eval_set:
                            in_eval = np.flatnonzero(np.isin(pool.movie_ids(art), list(eval_set)))
                            scored = top_within(rerank(pool.take(in_eval), lam, art), art, eval_set)
                        dial_rows.append(
                            {"user_id": user["user_id"], "lam": float(lam), **refs.score(scored, test),
                             "novelty": float(art.novelty[shown.positions[:TOP_N]].mean())}
                        )  # fmt: skip

        checkpoint(0)
        for swipe in range(1, config.SIM_SWIPES + 1):
            card = next_card(session, art, rng=rng, exclude=card_exclude)
            if card is None:
                break
            session.add(answer_for(user, card))
            test.discard(card)
            if swipe % CHECKPOINT_EVERY == 0:
                checkpoint(swipe)
    return rows, dial_rows, dial_recs


def run(args) -> None:
    start = time.perf_counter()
    art = load_dir(args.artifacts)
    test_users = set(pd.read_parquet(args.artifacts / "test_users.parquet")["user_id"].tolist())
    ratings = load_ratings(args.data, art)
    is_test = ratings["user_id"].isin(test_users)
    log(f"loaded {len(ratings):,} catalog ratings; {is_test.sum():,} from {len(test_users):,} test users", start)

    users = pick_users(ratings[is_test], art, args.users, args.seed)
    log(f"selected {len(users)} simulated users", start)

    refs = LensKitRefs(ratings[~is_test], art.movie_ids)
    log("trained LensKit reference pipelines", start)

    rows, dial_rows, coverage = [], [], {float(lam): set() for lam in LAMBDAS}
    for k, user in enumerate(users, 1):
        r, d, recs = simulate_user(user, art, refs, args.seed, args.test_cards)
        rows += r
        dial_rows += d
        for lam, ids in recs.items():
            coverage[lam].update(ids)
        if k % max(1, len(users) // 10) == 0:
            log(f"  {k}/{len(users)} users", start)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "checkpoints.csv", index=False)
    dial = pd.DataFrame(dial_rows)
    dial.to_csv(args.out / "dial.csv", index=False)
    meta = {
        "dataset": art.meta.get("dataset"),
        "artifacts": str(args.artifacts),
        "seed": args.seed,
        "test_cards": args.test_cards,
        "config_overrides": args.overrides,
        "n_users": len(users),
        "catalog_size": art.n,
        "swipes": config.SIM_SWIPES,
        "coverage_by_lambda": {str(lam): len(ids) / art.n for lam, ids in coverage.items()},
        "runtime_s": round(time.perf_counter() - start, 1),
    }
    (args.out / "summary.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote results to {args.out}", start)


# ---------------------------------------------------------------- plots

# Reference palette (light mode). Strategies take categorical slots 1-4 in fixed
# order; LensKit reference lines are neutral ink, dashed, so they read as baselines.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {
    "adaptive": ("Adaptive", "#2a78d6", "-"),
    "pop_entropy": ("Pop × entropy", "#eb6834", "-"),
    "popularity": ("Popularity", "#1baf7a", "-"),
    "random": ("Random", "#eda100", "-"),
    REF_KNN: ("LensKit item kNN", INK_2, (0, (5, 3))),
    REF_POP: ("LensKit popularity", MUTED, (0, (1.5, 2.5))),
}


def mean_ci(df: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    g = df.groupby(by)[col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["ci"] = 1.96 * out["std"] / np.sqrt(out["count"])
    return out


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=30)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def _end_labels(ax, ends: list[tuple[float, float, str]], min_gap: float) -> None:
    """Direct labels at line ends, spread apart symmetrically so they never overlap.

    A thin leader line connects a label to its line end when it had to move.
    """
    ends = sorted(ends, key=lambda e: e[1])
    ys = [e[1] for e in ends]
    for _ in range(200):  # relax: push overlapping neighbors apart by half the deficit each
        moved = False
        for k in range(len(ys) - 1):
            gap = ys[k + 1] - ys[k]
            if gap < min_gap - 1e-12:
                push = (min_gap - gap) / 2
                ys[k] -= push
                ys[k + 1] += push
                moved = True
        if not moved:
            break
    x_text = max(e[0] for e in ends)
    dx = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.035
    for (x, y, text), y_label in zip(ends, ys):
        if abs(y_label - y) > min_gap * 0.2:
            ax.plot([x, x_text + dx * 0.8], [y, y_label], color=MUTED, linewidth=0.6, zorder=1)
        ax.text(x_text + dx, y_label, text, va="center", fontsize=9, color=INK)


def _legend_row(ax, ncol: int) -> None:
    """Legend as one row between the title and the plot, clear of the direct labels."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, frameon=False, fontsize=8.5,
              labelcolor=INK_2, handlelength=2.2, columnspacing=1.4, borderaxespad=0.3)  # fmt: skip


def _lines(ax, stats: pd.DataFrame, keys: list[str], x: str) -> list[tuple[float, float, str]]:
    ends = []
    for key in keys:
        s = stats[stats["strategy"] == key].sort_values(x)
        if s.empty:
            continue
        label, color, dash = SERIES[key]
        ax.fill_between(s[x], s["mean"] - s["ci"], s["mean"] + s["ci"], color=color, alpha=0.08, linewidth=0)
        ax.plot(s[x], s["mean"], color=color, linestyle=dash, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)  # fmt: skip
        ends.append((s[x].iloc[-1], s["mean"].iloc[-1], label))
    return ends


def make_plots(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    cp = pd.read_csv(out / "checkpoints.csv")
    dial = pd.read_csv(out / "dial.csv")
    meta = json.loads((out / "summary.json").read_text())
    n = meta["n_users"]
    protocol = {
        "sampled": "ranking each user's hidden liked movies among 100 popularity-sampled unrated movies",
        "reveal": "full-catalog ranking; cards may reveal test movies",
        "exclude": "full-catalog ranking; test movies hidden from cards (leaky)",
    }[meta.get("test_cards", "reveal")]
    note = f"{meta['dataset']}, {n} held-out users, {protocol}; mean ± 95% CI"

    # 1. nDCG@10 vs swipes
    stats = mean_ci(cp, ["strategy", "swipes"], "ndcg")
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    ends = _lines(ax, stats, list(SERIES), "swipes")
    _style(ax, "Recommendation accuracy vs. number of swipes", "Swipes (after 3 seed picks)", "nDCG@10")
    ax.set_xlim(0, config.SIM_SWIPES + 14)
    ax.set_xticks(range(0, config.SIM_SWIPES + 1, CHECKPOINT_EVERY * 2))
    lo, hi = (stats["mean"] - stats["ci"]).min(), (stats["mean"] + stats["ci"]).max()
    pad = (hi - lo) * 0.08
    ax.set_ylim(lo - pad, hi + pad)  # zoomed: the axis does not start at 0
    _end_labels(ax, ends, min_gap=(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.045)
    _legend_row(ax, ncol=6)
    fig.text(0.01, 0.01, note, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "ndcg_vs_swipes.png", dpi=200)
    plt.close(fig)

    # 2. Fraction of unseen answers vs swipes
    stats = mean_ci(cp[cp["swipes"] > 0], ["strategy", "swipes"], "unseen_frac")
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    ends = _lines(ax, stats, list(STRATEGIES), "swipes")
    _style(ax, 'Share of cards the user hadn\'t seen ("unseen")', "Swipes", "Cumulative unseen fraction")
    ax.set_xlim(0, config.SIM_SWIPES + 12)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _end_labels(ax, ends, min_gap=0.045)
    _legend_row(ax, ncol=4)
    fig.text(0.01, 0.01, note, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "unseen_vs_swipes.png", dpi=200)
    plt.close(fig)

    # 3. Dial tradeoff: nDCG vs novelty across lambda
    nd = mean_ci(dial, ["lam"], "ndcg").set_index("lam")
    nv = mean_ci(dial, ["lam"], "novelty").set_index("lam")
    cov = {float(k): v for k, v in meta["coverage_by_lambda"].items()}
    color = SERIES["adaptive"][1]
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=SURFACE)
    ax.errorbar(nv["mean"], nd["mean"], xerr=nv["ci"], yerr=nd["ci"], color=color, linewidth=2, marker="o",
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, ecolor=GRID, elinewidth=1.5, capsize=0, zorder=3)  # fmt: skip
    # Label selected points; lambda rises left to right in steps of 0.1.
    for lam in nd.index:
        if lam not in (0.0, 0.3, 0.4, 0.5, 0.6):
            continue
        offset = (4, -20) if lam == 0.0 else (10, 8)
        ax.annotate(f"λ={lam:.1f} · {cov.get(lam, 0):.1%} coverage", (nv.loc[lam, "mean"], nd.loc[lam, "mean"]),
                    xytext=offset, textcoords="offset points", fontsize=8.5, color=INK,
                    ha="left")  # fmt: skip
    _style(ax, f"Adventurousness dial: accuracy vs. novelty (after {DIAL_AT} adaptive swipes)",
           "Mean novelty of top 10 (−log₂ popularity share)", "nDCG@10")  # fmt: skip
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + (x1 - x0) * 0.3)
    fig.text(0.01, 0.01, note + "\nPoints: λ = 0, 0.1, …, 0.6 from left to right. "
             "Coverage = distinct movies recommended across users ÷ catalog size.", color=MUTED, fontsize=8)  # fmt: skip
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "dial_tradeoff.png", dpi=200)
    plt.close(fig)

    # Table view of everything plotted.
    table = []
    for metric in ("ndcg", "recall", "precision", "unseen_frac", "n_rated"):
        s = mean_ci(cp, ["strategy", "swipes"], metric).rename(columns={"mean": metric, "ci": f"{metric}_ci"})
        table.append(s.set_index(["strategy", "swipes"])[[metric, f"{metric}_ci"]])
    pd.concat(table, axis=1).round(4).to_csv(out / "summary.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="MovieLens directory the artifacts were built from")
    ap.add_argument("--artifacts", type=Path, default=config.ARTIFACTS_DIR / "eval")
    ap.add_argument("--users", type=int, default=config.N_SIM_USERS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, help="results directory (default results/<dataset>)")
    ap.add_argument("--test-cards", choices=["sampled", "reveal", "exclude"], default="sampled",
                    help="how cards treat hidden test movies (see module docstring)")  # fmt: skip
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a recsys.config value for sensitivity runs, e.g. --set DAMPING=5")  # fmt: skip
    ap.add_argument("--plots-only", action="store_true", help="redraw plots from existing CSVs")
    args = ap.parse_args()
    args.overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if not hasattr(config, key):
            ap.error(f"unknown config key {key!r}")
        setattr(config, key, type(getattr(config, key))(value))
        args.overrides[key] = getattr(config, key)
    if args.out is None:
        if args.data is None:
            ap.error("--out is required with --plots-only")
        args.out = config.RESULTS_DIR / args.data.name
    if not args.plots_only:
        if args.data is None:
            ap.error("--data is required")
        run(args)
    make_plots(args.out)


if __name__ == "__main__":
    main()
