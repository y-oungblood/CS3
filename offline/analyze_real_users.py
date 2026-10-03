"""Real-user analysis: did the recommender work, how much time did it cost, and how
does accuracy grow with time invested?

Usage:
    # 1. Snapshot Firestore (needs credentials); drops handles and excluded accounts.
    GOOGLE_APPLICATION_CREDENTIALS=key.json python offline/analyze_real_users.py --snapshot
    # 2. Analyze the snapshot (no credentials needed).
    python offline/analyze_real_users.py

Outputs in results/real-users/: summary.json, tables (CSV) and three figures.

Outcome coding for an answered movie: a hit is rated >= 4 stars or "Add to my
list"; a miss is rated <= 2 or "Not interested"; 3 stars is neutral. Hit rate =
hits / answered. Confidence intervals resample users (not answers), because one
tester's answers are not independent of each other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recsys import config  # noqa: E402
from simulate import INK, INK_2, MUTED, SERIES, SURFACE, _style  # noqa: E402

OUT = config.RESULTS_DIR / "real-users"
SNAPSHOT = OUT / "snapshot"
SIM = config.RESULTS_DIR / "ml-32m" / "checkpoints.csv"
EXCLUDE_HANDLES = {"silent-wolf-40"}  # the developer's own test account
TABLES = ("users", "answers", "rec_lists", "rec_items", "rec_feedback")
ANSWER_KINDS = ("rated", "interested", "not_interested")
N_BOOT = 5000
SEED = 42
BLUE = SERIES["adaptive"][1]
GRID_TINT = "#f0efec"  # neutral wash marking the seed-picking phase


# ---------------------------------------------------------------- snapshot


def take_snapshot() -> None:
    from recsys.storage import FirestoreStorage

    t = FirestoreStorage().export()
    excluded = set(t["users"].loc[t["users"]["handle"].isin(EXCLUDE_HANDLES), "user_id"])
    keep_lists = set(t["rec_lists"].loc[~t["rec_lists"]["user_id"].isin(excluded), "list_id"])
    t["users"] = t["users"][~t["users"]["user_id"].isin(excluded)].drop(columns="handle")
    for name in ("answers", "rec_lists", "rec_feedback"):
        t[name] = t[name][~t[name]["user_id"].isin(excluded)]
    t["rec_items"] = t["rec_items"][t["rec_items"]["list_id"].isin(keep_lists)]
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        t[name].to_csv(SNAPSHOT / f"{name}.csv", index=False)
    print(f"snapshot: {len(t['users'])} users ({len(excluded)} excluded), {len(t['answers'])} answers -> {SNAPSHOT}")


def load_snapshot() -> dict[str, pd.DataFrame]:
    t = {name: pd.read_csv(SNAPSHOT / f"{name}.csv") for name in TABLES}
    for name in ("users", "answers", "rec_lists", "rec_feedback"):
        t[name]["created_at"] = pd.to_datetime(t[name]["created_at"], format="ISO8601")
    return t


# ---------------------------------------------------------------- helpers


def outcome(kind: str, rating) -> str:
    if kind == "interested":
        return "hit"
    if kind == "not_interested":
        return "miss"
    return "hit" if rating >= 4 else "miss" if rating <= 2 else "neutral"


def with_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["kind"].isin(ANSWER_KINDS)].copy()
    df["outcome"] = [outcome(k, r) for k, r in zip(df["kind"], df["rating"])]
    df["hit"] = (df["outcome"] == "hit").astype(float)
    return df


def user_bootstrap(df: pd.DataFrame, col: str, stat=np.mean, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Point estimate (pooled over answers) and 95% CI resampling users."""
    rng = np.random.default_rng(SEED)
    groups = [g[col].to_numpy() for _, g in df.groupby("user_id")]
    point = float(stat(np.concatenate(groups)))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boots.append(stat(np.concatenate([groups[i] for i in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def rate_summary(df: pd.DataFrame) -> dict:
    hit, lo, hi = user_bootstrap(df, "hit")
    counts = df["outcome"].value_counts()
    return {
        "users": int(df["user_id"].nunique()),
        "answered": int(len(df)),
        "hit_rate": round(hit, 3),
        "hit_rate_ci95": [round(lo, 3), round(hi, 3)],
        "miss_rate": round(counts.get("miss", 0) / len(df), 3),
        "neutral_rate": round(counts.get("neutral", 0) / len(df), 3),
        "seen_share": round(float((df["kind"] == "rated").mean()), 3),
        "mean_stars_when_seen": round(float(df.loc[df["kind"] == "rated", "rating"].mean()), 2),
    }


# ---------------------------------------------------------------- analysis


def analyze(t: dict[str, pd.DataFrame]) -> dict:
    users, answers, lists, feedback = t["users"], t["answers"], t["rec_lists"], t["rec_feedback"]
    active = answers["user_id"].unique()
    swipes = with_outcomes(answers[answers["source"] == "swipe"])
    recs = with_outcomes(feedback).merge(
        lists[["list_id", "trigger", "lambda", "n_informative_answers"]], on="list_id", how="left"
    )
    s: dict = {
        "participants": {
            "accounts": int(len(users)),
            "active": int(len(active)),
            "by_arm": users[users["user_id"].isin(active)]["card_strategy"].value_counts().to_dict(),
            "answers": int(len(answers)),
            "swipes": int(len(swipes)),
            "median_swipes_per_user": float(swipes.groupby("user_id").size().median()),
            "rec_lists_shown": int(len(lists)),
            "excluded_handles": sorted(EXCLUDE_HANDLES),
        }
    }

    # Q1: can it recommend? Recommendations vs the deck's own cards (same testers, same answers).
    s["q1_recommend"] = {"recommendations": rate_summary(recs), "deck_cards": rate_summary(swipes)}
    # Paired: each tester's recommendation hit rate minus their own deck hit rate.
    paired = pd.DataFrame({"recs": recs.groupby("user_id")["hit"].mean(), "deck": swipes.groupby("user_id")["hit"].mean()}).dropna()
    diff = (paired["recs"] - paired["deck"]).to_numpy()
    rng = np.random.default_rng(SEED)
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(N_BOOT)]
    s["q1_recommend"]["paired_by_tester"] = {
        "testers": int(len(paired)),
        "mean_difference": round(float(diff.mean()), 3),
        "difference_ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
        "testers_with_higher_rec_hit_rate": int((diff > 0).sum()),
        "testers_tied": int((diff == 0).sum()),
    }
    per_list = recs.groupby("list_id").size()
    s["q1_recommend"]["answered_per_list_median"] = float(per_list.median())
    s["q1_recommend"]["lists_with_answers"] = int(len(per_list))

    # Q2: time per answer by path, and time to first recommendations.
    timed = swipes[swipes["response_ms"].notna()].copy()
    timed["path"] = np.where(timed["kind"] == "rated", "Seen it → stars", "Haven't seen → list / pass")
    q2 = {}
    for path, g in timed.groupby("path"):
        med, lo, hi = user_bootstrap(g.assign(sec=g["response_ms"] / 1000), "sec", stat=np.median)
        q2[path] = {
            "answers": int(len(g)),
            "median_s": round(med, 2),
            "median_ci95": [round(lo, 2), round(hi, 2)],
            "iqr_s": [round(g["response_ms"].quantile(0.25) / 1000, 2), round(g["response_ms"].quantile(0.75) / 1000, 2)],
        }
    all_med, all_lo, all_hi = user_bootstrap(timed.assign(sec=timed["response_ms"] / 1000), "sec", stat=np.median)
    signup = users.set_index("user_id")["created_at"]
    seeds_at = answers[answers["source"] == "seed"].groupby("user_id")["created_at"].min()
    first_list = lists.sort_values("created_at").groupby("user_id")["created_at"].first()
    journey = pd.DataFrame({"signup": signup, "seeds": seeds_at, "first_recs": first_list}).dropna()
    s["q2_time"] = {
        "per_answer_by_path": q2,
        "per_swipe_median_s": round(all_med, 2),
        "per_swipe_median_ci95": [round(all_lo, 2), round(all_hi, 2)],
        "share_over_60s": round(float((timed["response_ms"] > 60_000).mean()), 3),
        "seed_picking_median_s": round(float((journey["seeds"] - journey["signup"]).dt.total_seconds().median()), 1),
        "signup_to_first_recs_median_s": round(
            float((journey["first_recs"] - journey["signup"]).dt.total_seconds().median()), 1
        ),
        "users_reaching_recs": int(len(journey)),
        "star_distribution_on_deck": swipes.loc[swipes["kind"] == "rated", "rating"].value_counts().sort_index().astype(int).to_dict(),
    }

    # Q3: accuracy vs time invested. Simulated curve mapped onto real users' timing.
    sim = pd.read_csv(SIM)
    sim = sim[sim["strategy"].isin(["adaptive", "lenskit_pop"])]
    curve = sim.groupby(["strategy", "swipes"])["precision"].agg(["mean", "std", "count"]).reset_index()
    curve["ci"] = 1.96 * curve["std"] / np.sqrt(curve["count"])
    seed_s, swipe_s = s["q2_time"]["seed_picking_median_s"], s["q2_time"]["per_swipe_median_s"]
    curve["elapsed_s"] = seed_s + curve["swipes"] * swipe_s
    ad = curve[curve["strategy"] == "adaptive"].set_index("swipes")
    start, end = ad.loc[0], ad.loc[config.SIM_SWIPES]
    n_test = sim.loc[(sim["strategy"] == "adaptive") & (sim["swipes"] == 0), "n_test"].mean()
    checkins = {}
    for trig in ("checkin_10", "checkin_25"):
        g = recs[recs["trigger"] == trig]
        if len(g):
            hit, lo, hi = user_bootstrap(g, "hit")
            checkins[trig] = {"users": int(g["user_id"].nunique()), "answered": int(len(g)),
                              "hit_rate": round(hit, 3), "hit_rate_ci95": [round(lo, 3), round(hi, 3)]}  # fmt: skip
    both = recs[recs["trigger"].isin(["checkin_10", "checkin_25"])].groupby("user_id")["trigger"].nunique()
    s["q3_accuracy_vs_time"] = {
        "sim_precision_at_10": {int(k): round(v, 3) for k, v in ad["mean"].items()},
        "sim_lenskit_popularity": round(float(curve.loc[curve["strategy"] == "lenskit_pop", "mean"].mean()), 3),
        "sim_chance": round(float(n_test / (n_test + 100)), 3),
        "seeds_share_of_50_swipe_accuracy": round(float(start["mean"] / end["mean"]), 3),
        "seed_time_share_of_50_swipe_time": round(float(start["elapsed_s"] / end["elapsed_s"]), 3),
        "elapsed_s_seeds_only": round(float(start["elapsed_s"]), 1),
        "elapsed_s_after_50": round(float(end["elapsed_s"]), 1),
        "real_checkins": checkins,
        "users_answering_both_checkins": int((both == 2).sum()),
    }

    # Extras: A/B arms, the dial, when-to-watch votes.
    arm = recs.merge(users[["user_id", "card_strategy"]], on="user_id")
    s["extras"] = {
        "rec_hit_rate_by_arm": {a: rate_summary(g) for a, g in arm.groupby("card_strategy")},
        "dial": {
            f"{lam:.1f}": {"answered": int(len(g)), "hit_rate": round(float(g["hit"].mean()), 3),
                           "add_to_list_share": round(float((g["kind"] == "interested").mean()), 3)}
            for lam, g in recs.groupby(recs["lambda"].round(1))
        },  # fmt: skip
        "when_to_watch_votes": feedback["kind"].value_counts().reindex(["timing_up", "timing_down"], fill_value=0).astype(int).to_dict(),
    }
    s["_curve"] = curve  # for plotting
    s["_timed"] = timed
    s["_recs"], s["_swipes"] = recs, swipes
    return s


# ---------------------------------------------------------------- figures


def make_figures(s: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    p = s["participants"]
    note = base_note = f"{p['active']} real testers ({p['answers']} answers); 95% CIs resample testers"

    # 1. Hit rate: recommendations vs deck cards.
    q1 = s["q1_recommend"]
    rows = [("Recommended movies", q1["recommendations"]), ("Swipe-deck cards", q1["deck_cards"])]
    fig, ax = plt.subplots(figsize=(8.5, 3.4), facecolor=SURFACE)
    for i, (label, r) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, r["hit_rate"], height=0.5, color=BLUE if i == 0 else MUTED)
        lo, hi = r["hit_rate_ci95"]
        ax.plot([lo, hi], [y, y], color=INK, linewidth=1.5)
        ax.text(hi + 0.015, y, f"{r['hit_rate']:.0%}  ({r['answered']} answers, {r['users']} testers)",
                va="center", fontsize=9, color=INK)  # fmt: skip
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlim(0, 1.32)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    pr = q1["paired_by_tester"]
    note = (note + f"; paired by tester: +{pr['mean_difference']:.0%} (95% CI {pr['difference_ci95'][0]:+.0%} to "
            f"{pr['difference_ci95'][1]:+.0%}), higher for {pr['testers_with_higher_rec_hit_rate']} of {pr['testers']} testers")
    _style(ax, "Share of answers that were positive", "Hit rate (rated ≥4★ or added to list)", "")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    fig.text(0.01, 0.01, note, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "hit_rate.png", dpi=200)
    plt.close(fig)

    # 2. Seconds per answer, by path.
    timed = s["_timed"]
    paths = list(s["q2_time"]["per_answer_by_path"])
    fig, ax = plt.subplots(figsize=(8.5, 3.4), facecolor=SURFACE)
    rng = np.random.default_rng(SEED)
    for i, path in enumerate(paths):
        sec = timed.loc[timed["path"] == path, "response_ms"].to_numpy() / 1000
        ax.scatter(np.clip(sec, 0, 20), i + rng.uniform(-0.18, 0.18, len(sec)), s=6, color=MUTED, alpha=0.35, linewidths=0)
        q = s["q2_time"]["per_answer_by_path"][path]
        ax.plot(q["iqr_s"], [i, i], color=BLUE, linewidth=5, solid_capstyle="round")
        ax.scatter([q["median_s"]], [i], s=70, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.text(q["iqr_s"][1] + 0.4, i + 0.28, f"median {q['median_s']:.1f} s ({q['answers']} answers)", fontsize=9, color=INK)
    ax.set_yticks(range(len(paths)), paths)
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.6, len(paths) - 0.2)
    _style(ax, "Seconds per swipe answer", "Seconds from card shown to final answer (dots above 20 s drawn at 20)", "")
    ax.grid(False, axis="y")
    fig.text(0.01, 0.01, note + "; bar = middle 50% of answers", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "time_per_answer.png", dpi=200)
    plt.close(fig)

    # 3. Accuracy vs time: simulated precision on real timing | real check-in hit rates.
    q3 = s["q3_accuracy_vs_time"]
    curve = s["_curve"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [2.2, 1]})
    ad = curve[curve["strategy"] == "adaptive"]
    ax.fill_between(ad["elapsed_s"], ad["mean"] - ad["ci"], ad["mean"] + ad["ci"], color=BLUE, alpha=0.12, linewidth=0)
    ax.plot(ad["elapsed_s"], ad["mean"], color=BLUE, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5)  # fmt: skip
    x1 = ad["elapsed_s"].max()
    for y, label, dash in ((q3["sim_lenskit_popularity"], "LensKit popularity", (0, (5, 3))), (q3["sim_chance"], "Chance", (0, (1.5, 2.5)))):
        ax.axhline(y, color=MUTED, linestyle=dash, linewidth=1.5)
        ax.text(x1, y + 0.008, label, ha="right", fontsize=9, color=INK_2)
    first = ad.iloc[0]
    ax.annotate(
        f"After 3 seed picks (~{first['elapsed_s']:.0f} s):\n{q3['seeds_share_of_50_swipe_accuracy']:.0%} of the 50-swipe accuracy\n"
        f"in {q3['seed_time_share_of_50_swipe_time']:.0%} of the time",
        (first["elapsed_s"], first["mean"]), xytext=(-40, 34), textcoords="offset points", fontsize=9, color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )  # fmt: skip
    ax.text(x1, ad["mean"].iloc[-1] + 0.012, "Adaptive deck", ha="right", fontsize=9, color=INK)
    ax.set_ylim(0, max(0.66, ad["mean"].max() + 0.12))
    ax.set_xlim(0, x1 * 1.04)
    ax.axvspan(0, first["elapsed_s"], color=GRID_TINT, linewidth=0)
    ax.text(first["elapsed_s"] / 2, 0.02, "picking 3\nseed movies", ha="center", fontsize=8.5, color=INK_2)
    secax = ax.secondary_xaxis("top", functions=(lambda x: (x - q3["elapsed_s_seeds_only"]) / s["q2_time"]["per_swipe_median_s"],
                                                 lambda n: q3["elapsed_s_seeds_only"] + n * s["q2_time"]["per_swipe_median_s"]))  # fmt: skip
    secax.set_xticks(range(0, config.SIM_SWIPES + 1, 10))
    secax.set_xlabel("Swipes after the seed picks", color=INK_2, fontsize=9)
    secax.tick_params(colors=MUTED, labelsize=8)
    _style(ax, "Simulated accuracy vs. time invested", "Seconds since sign-up (real testers' median timing)",
           "Precision@10 (liked movies in top 10)")  # fmt: skip

    labels, vals, cis, ns = [], [], [], []
    for trig, label in (("checkin_10", "After 10\nanswers"), ("checkin_25", "After 25\nanswers")):
        c = q3["real_checkins"].get(trig)
        if c:
            labels.append(label), vals.append(c["hit_rate"]), cis.append(c["hit_rate_ci95"]), ns.append(c)
    for i, (v, (lo, hi), c) in enumerate(zip(vals, cis, ns)):
        ax2.bar(i, v, width=0.55, color=BLUE)
        ax2.plot([i, i], [lo, hi], color=INK, linewidth=1.5)
        ax2.text(i, hi + 0.03, f"{v:.0%}\n{c['users']} testers", ha="center", fontsize=8.5, color=INK)
    ax2.set_xticks(range(len(labels)), labels)
    ax2.set_ylim(0, 1.15)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax2, "Real testers at check-ins", "", "Hit rate")
    ax2.grid(False, axis="x")
    fig.text(0.01, 0.01, f"Left: 500 simulated ML-32M users, mean ± 95% CI, timing from real testers.\nRight: {base_note}; "
             f"only {q3['users_answering_both_checkins']} testers answered at both check-ins, so the bars compare different people.",
             color=MUTED, fontsize=8)  # fmt: skip
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "accuracy_vs_time.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", action="store_true", help="export Firestore to the snapshot first")
    args = ap.parse_args()
    if args.snapshot:
        take_snapshot()
    s = analyze(load_snapshot())
    make_figures(s)
    OUT.mkdir(parents=True, exist_ok=True)
    s["_curve"].to_csv(OUT / "accuracy_vs_time_curve.csv", index=False)
    summary = {k: v for k, v in s.items() if not k.startswith("_")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
