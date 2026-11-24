"""Master-level over-the-board Chess960 games: analysis (Part III).

Input: data/masters960/masters960_games.csv, one row per game from elite
Chess960 events between 2018 and 2026 (sources in config/masters960_sources.csv;
built with code/masters960.py download and parse).

- inclusion: Chess960 game, position identified, position matches the one
  drawn for its round, known result, not Armageddon, at least 10 plies, both
  FIDE ratings (a missing rating is filled from the same player's other games
  in the event) (Table 1);
- White's advantage at equal ratings (fractional logit with the FIDE rating
  gap as an offset, beta fixed at 1) and the decisive rate at equal ratings,
  overall, by time-control class and by rating level, SEs clustered by
  position (Table 2, Figure 11);
- per-position W/D/L, advantage and decisive rate at equal ratings, next to
  the engine self-play values; positions with at least 20 games are the case
  studies used in the heterogeneity and transfer analyses (Figure 13).

    python code/analyse_masters960.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_lichess960 import K, decisive_model, frac_logit, per_position_decisive
from utils import DATA_DIR, RESULTS_DIR, STANDARD_POSITION_ID, setup_logging

log = logging.getLogger("analyse_masters960")

GAMES = DATA_DIR / "masters960" / "masters960_games.csv"
OUT = RESULTS_DIR / "masters960"
SELFPLAY = RESULTS_DIR / "selfplay" / "selfplay_posteriors.csv"

MIN_PLIES = 10
ROUND_POSITION_SHARE = 0.8       # a round where most games share one position drew that position
ROUND_MIN_GAMES = 10
MIN_CASE_GAMES = 20              # positions used as case studies
RATING_LEVELS = [0, 2400, 2600, 4000]
RATING_LABELS = ["<2400", "2400-2599", "2600+"]
TC_ORDER = ["classical", "rapid", "blitz"]


def load_engines(path: Path = SELFPLAY) -> pd.DataFrame:
    post = pd.read_csv(path, usecols=["engine", "position_id", "delta_elo_median", "delta_elo_sd"])
    return post.pivot(index="position_id", columns="engine", values=["delta_elo_median", "delta_elo_sd"])


def fill_ratings(g: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing FIDE rating from the same player's other games in the
    same event (some rounds are published without ratings)."""
    g = g.copy()
    long = pd.concat([g[["event", "white", "white_elo"]].set_axis(["event", "player", "elo"], axis=1),
                      g[["event", "black", "black_elo"]].set_axis(["event", "player", "elo"], axis=1)])
    known = long.dropna().groupby(["event", "player"]).elo.median()
    for side in ("white", "black"):
        miss = g[f"{side}_elo"].isna()
        idx = pd.MultiIndex.from_frame(g.loc[miss, ["event", side]])
        g.loc[miss, f"{side}_elo"] = known.reindex(idx).to_numpy()
    return g


def round_position_ok(g: pd.DataFrame) -> pd.Series:
    """In events that draw one position per round, a game whose position
    differs from the one most of its round played is a broadcast error (a
    board still showing an earlier round's game). Applied to rounds with at
    least ROUND_MIN_GAMES games where one position covers at least
    ROUND_POSITION_SHARE of the identified games."""
    rnd = g["round"].astype(str).str.split(".").str[0]
    key = g.source.astype(str) + "|" + rnd
    ok = pd.Series(True, index=g.index)
    for _, sub in g[g.position_id >= 0].groupby(key[g.position_id >= 0]):
        if len(sub) < ROUND_MIN_GAMES:
            continue
        top = sub.position_id.value_counts()
        if top.iloc[0] / len(sub) >= ROUND_POSITION_SHARE:
            ok.loc[sub.index[sub.position_id != top.index[0]]] = False
    return ok


def include(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_missing = int((g.white_elo.isna() | g.black_elo.isna()).sum())
    g = fill_ratings(g)
    print(f"Ratings: {n_missing} games missing one; "
          f"{n_missing - int((g.white_elo.isna() | g.black_elo.isna()).sum())} filled from the same "
          "player's other games in the event")
    # A Chess960 game: the Variant header says so, or the FEN is a Chess960
    # start other than the standard one (a standard-chess game has no FEN or
    # the standard FEN and no 960 Variant tag).
    if "variant_960" not in g:            # games table from before the Variant check
        g["variant_960"] = False
    chess960 = g.variant_960.fillna(False).astype(bool) | (
        (g.position_id >= 0) & (g.position_id != STANDARD_POSITION_ID))
    steps = [("Chess960 game (Variant tag or Chess960 FEN)", chess960),
             ("starting position identified", g.position_id >= 0),
             ("position matches its round's drawn position", round_position_ok(g)),
             ("known result", g.score_white.notna()),
             ("not Armageddon", ~g.armageddon.astype(bool)),
             (f"at least {MIN_PLIES} plies (drops forfeits)", g.plies >= MIN_PLIES),
             ("both FIDE ratings", g.white_elo.notna() & g.black_elo.notna())]
    keep = pd.Series(True, index=g.index)
    flow = [{"criterion": "parsed", "games": len(g)}]
    for label, m in steps:
        keep &= m.fillna(False).astype(bool)
        flow.append({"criterion": label, "games": int(keep.sum())})
    flow = pd.DataFrame(flow)
    flow["removed"] = -flow.games.diff()
    return g[keep].copy(), flow


def rating_offset(d: pd.DataFrame) -> np.ndarray:
    """FIDE Elo gap on the logit scale, beta fixed at 1."""
    return (d.white_elo.to_numpy(float) - d.black_elo.to_numpy(float)) * np.log(10) / 400


def wdl(s: np.ndarray) -> dict:
    e = float(np.mean(s))
    return {"games": len(s), "white_wins": float(np.mean(s == 1)), "draws": float(np.mean(s == 0.5)),
            "black_wins": float(np.mean(s == 0)), "escore": e,
            "delta_elo_raw": 400 * np.log10(e / (1 - e)) if 0 < e < 1 else np.nan}


def advantage(d: pd.DataFrame, cluster: np.ndarray | None = None) -> dict:
    """White's advantage at equal FIDE ratings (Elo), robust or clustered SE."""
    s = d.score_white.to_numpy(float)
    if len(d) < 2 or s.min() == s.max():
        return {"advantage_elo": np.nan, "advantage_lo": np.nan, "advantage_hi": np.nan}
    b, V = frac_logit(np.ones((len(d), 1)), s, offset=rating_offset(d), cluster=cluster)
    se = float(np.sqrt(V[0, 0]))
    return {"advantage_elo": K * b[0], "advantage_se": K * se,
            "advantage_lo": K * (b[0] - 1.96 * se), "advantage_hi": K * (b[0] + 1.96 * se)}


def level(d: pd.DataFrame) -> pd.Series:
    m = (d.white_elo + d.black_elo) / 2
    return pd.cut(m, RATING_LEVELS, labels=RATING_LABELS, right=False)


def abs_gap(d: pd.DataFrame) -> np.ndarray:
    return np.abs(d.white_elo.to_numpy(float) - d.black_elo.to_numpy(float))


def decisive_at_equal(d: pd.DataFrame) -> dict:
    """Decisive rate between equally rated players (logit on |rating gap| and
    its square, position-clustered)."""
    s = d.score_white.to_numpy(float)
    dec = (s != 0.5).astype(float)
    m = decisive_model(abs_gap(d), dec, cluster=d.position_id.to_numpy())
    return {k: v for k, v in m.items() if not k.startswith("_")}


def events_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ev, yr, st), sub in d.groupby(["event", "year", "stage"], dropna=False):
        rows.append({"event": ev, "year": yr, "stage": st, "positions": sub.position_id.nunique(),
                     "mean_rating": float(((sub.white_elo + sub.black_elo) / 2).mean()),
                     "tc_classes": "; ".join(t for t in TC_ORDER if (sub.tc_class == t).any()),
                     **wdl(sub.score_white.to_numpy(float))})
    return pd.DataFrame(rows).sort_values(["year", "event", "stage"])


def pooled(d: pd.DataFrame) -> pd.DataFrame:
    groups = [("all", "all", d)]
    groups += [("time control", t, d[d.tc_class == t]) for t in TC_ORDER]
    lv = level(d)
    groups += [("rating", r, d[lv == r]) for r in RATING_LABELS]
    rows = []
    for kind, name, sub in groups:
        if len(sub) == 0:
            continue
        rows.append({"group": kind, "level": name, "positions": sub.position_id.nunique(),
                     "events": sub.event.nunique(),
                     "mean_rating": float(((sub.white_elo + sub.black_elo) / 2).mean()),
                     **wdl(sub.score_white.to_numpy(float)),
                     **advantage(sub, sub.position_id.to_numpy()), **decisive_at_equal(sub)})
    return pd.DataFrame(rows)


def positions_table(d: pd.DataFrame, eng: pd.DataFrame) -> pd.DataFrame:
    s = d.score_white.to_numpy(float)
    dec = (s != 0.5).astype(float)
    coef = decisive_model(abs_gap(d), dec, cluster=d.position_id.to_numpy())["_coef"]
    adj = per_position_decisive(d.position_id.to_numpy(), abs_gap(d), dec, coef)
    rows = []
    for p, sub in d.groupby("position_id"):
        r = {"position_id": p, "events": "; ".join(sorted(sub.event.unique())),
             "tc_classes": "; ".join(t for t in TC_ORDER if (sub.tc_class == t).any()),
             "mean_rating": float(((sub.white_elo + sub.black_elo) / 2).mean()),
             **wdl(sub.score_white.to_numpy(float)), **advantage(sub),
             "decisive_raw": float((sub.score_white != 0.5).mean()),
             **adj.loc[int(p)].to_dict()}
        for e in ("stockfish", "caissa"):
            r[f"{e}_selfplay_elo"] = eng.loc[p, ("delta_elo_median", e)]
            r[f"{e}_selfplay_sd"] = eng.loc[p, ("delta_elo_sd", e)]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("games", ascending=False)


def descriptives(d: pd.DataFrame) -> dict:
    """Table 1 counts."""
    mean_r = (d.white_elo + d.black_elo) / 2
    return {"games": len(d), "players": pd.unique(pd.concat([d.white, d.black])).size,
            "positions": d.position_id.nunique(), "events": d.event.nunique(),
            "mean_rating_min": float(mean_r.min()), "mean_rating_max": float(mean_r.max()),
            **{k: v for k, v in wdl(d.score_white.to_numpy(float)).items() if k != "games"}}


def summary_text(flow, desc, ev, pos, pool) -> str:
    L = ["Master-level OTB Chess960: summary", "=" * 34, "", "Inclusion:"]
    L += [f"  {r.criterion}: {r.games:,}" for r in flow.itertuples()]
    L += ["", f"{desc['games']:,} games, {desc['players']} players, {len(ev)} events/stages, "
          f"{desc['positions']} positions (median {pos.games.median():.0f} games, max {pos.games.max()}); "
          f"positions with >= {MIN_CASE_GAMES} games: {(pos.games >= MIN_CASE_GAMES).sum()}",
          f"Mean rating {desc['mean_rating_min']:.0f}-{desc['mean_rating_max']:.0f}",
          "", "White's advantage and decisive rate at equal ratings (FIDE gap as offset, position-clustered):"]
    for r in pool.itertuples():
        L.append(f"  {r.group} / {r.level}: n {r.games:,}, positions {r.positions}; W/D/L "
                 f"{100 * r.white_wins:.1f}/{100 * r.draws:.1f}/{100 * r.black_wins:.1f}%; advantage "
                 f"{r.advantage_elo:+.1f} [{r.advantage_lo:+.1f}, {r.advantage_hi:+.1f}]; decisive "
                 f"{100 * r.decisive_adj:.1f}% [{100 * r.decisive_adj_lo:.1f}, {100 * r.decisive_adj_hi:.1f}]")
    L += ["", f"Case-study positions (>= {MIN_CASE_GAMES} games):"]
    for r in pos[pos.games >= MIN_CASE_GAMES].itertuples():
        L.append(f"  #{r.position_id}: n {r.games}, {r.tc_classes}; W/D/L {100 * r.white_wins:.0f}/"
                 f"{100 * r.draws:.0f}/{100 * r.black_wins:.0f}%; advantage {r.advantage_elo:+.0f} "
                 f"[{r.advantage_lo:+.0f}, {r.advantage_hi:+.0f}]; Stockfish {r.stockfish_selfplay_elo:+.0f}, "
                 f"Caissa {r.caissa_selfplay_elo:+.0f}; decisive at equal ratings {100 * r.decisive_adj:.0f}%")
    L.append(f"#518 played: {'yes' if STANDARD_POSITION_ID in set(pos.position_id) else 'no'}")
    return "\n".join(L) + "\n"


def analyse(games: Path = GAMES, out: Path = OUT) -> None:
    g = pd.read_csv(games)
    d, flow = include(g)
    print(f"Analysing {len(d):,} of {len(g):,} games after the inclusion criteria\n")
    eng = load_engines()
    ev, pos, pool, desc = events_table(d), positions_table(d, eng), pooled(d), descriptives(d)
    out.mkdir(parents=True, exist_ok=True)
    flow.to_csv(out / "masters960_flow.csv", index=False)
    pd.DataFrame([desc]).to_csv(out / "masters960_descriptives.csv", index=False)
    ev.to_csv(out / "masters960_events.csv", index=False)
    pos.to_csv(out / "masters960_positions.csv", index=False)
    pool.to_csv(out / "masters960_pooled.csv", index=False)
    text = summary_text(flow, desc, ev, pos, pool)
    (out / "masters960_summary.txt").write_text(text)
    print(text)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    analyse()
    return 0


if __name__ == "__main__":
    sys.exit(main())
