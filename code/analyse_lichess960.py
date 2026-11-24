"""Human play (Part III): rated Lichess Chess960 games against the engine
self-play results.

Subcommands
-----------
prepare  Combine the monthly tables from lichess960.py, apply the inclusion
         criteria separately for blitz (estimated 300-479 s) and rapid
         (>= 480 s), and write one compact table per set to
         results/lichess960/prepared/ (not committed). Also writes the flow table (games
         removed at each step), a randomisation check (games per position
         against uniform) and the rating-change distribution used for the
         provisional-rating rule.
analyse  For each set (Part III, steps 1, 2 and 4 of the paper's analyses):
           - descriptives overall and by rating band (Table 1);
           - rating model: White's score on the rating gap; White's advantage
             at equal ratings (Delta_Elo) and the rating-gap coefficient beta;
           - change in White's advantage per 100 points of mean rating;
           - White's advantage and decisive rate at equal ratings overall and
             by rating band (Table 2, Figure 11);
           - per-position advantage and decisive rate at equal ratings, and
             rating-matched win/draw/loss counts for the Dirichlet comparison;
           - per-position advantage refitted within each rating band (used
             for the transfer analysis by band).
         Writes tables to results/lichess960/.
all      prepare, then analyse.

Model
-----
White's score s (1, 1/2, 0) is modelled with a fractional logit, so the
fitted mean is the expected score E_score used throughout the paper. Logit
units become Elo by multiplying by K = 400 / ln 10 (about 173.7). With the
rating gap D = R_White - R_Black entered as D * ln10 / 400, its coefficient
beta is 1 when Lichess ratings follow the Elo scale, and the intercept times K
is White's advantage in Elo at equal ratings.

Per-position estimates fix beta at the set's rating-model value and fit one
intercept per position, with SEs clustered on the player pair within each
position (Lichess rematches keep the position).

    python code/analyse_lichess960.py all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

from utils import DATA_DIR, N_POSITIONS, RESULTS_DIR, REPO_ROOT, STANDARD_POSITION_ID, setup_logging

log = logging.getLogger("analyse_lichess960")

RAW = DATA_DIR / "lichess960"
PREPARED = RESULTS_DIR / "lichess960" / "prepared"     # filtered games (not committed)
OUT_TABLES = RESULTS_DIR / "lichess960"

K = 400 / np.log(10)                 # logit -> Elo
SETS = {"blitz": (300, 480), "rapid": (480, np.inf)}   # estimated seconds, [lo, hi)
SET_LABELS = {"blitz": "Blitz", "rapid": "Rapid"}
MIN_BAND_GAMES = 1000                # subgroup estimates of advantage and decisive rate
MIN_BAND_POSITIONS_GAMES = 5000      # per-position estimates within a band (transfer by band)
BAD_TERMINATIONS = {"Abandoned", "Rules infraction"}
MIN_PLIES = 10
PROVISIONAL_CHANGE = 30              # |rating change| above this = provisional (high RD)
RATING_FLOOR = 1200
BAND_EDGES = [1200, 1500, 1800, 2000, np.inf]
BAND_LABELS = ["1200-1499", "1500-1799", "1800-1999", "2000+"]
RATING_CENTRE = 1750

RAW_COLUMNS = ["month", "game_id", "position_id", "estimated_s", "white_elo", "black_elo",
               "white_rating_diff", "black_rating_diff", "white_berserk", "black_berserk",
               "termination", "score_white", "plies", "white_id", "black_id", "tournament",
               "white_title", "black_title"]


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------

def band_of(mean_rating: pd.Series) -> pd.Series:
    return pd.cut(mean_rating, BAND_EDGES, labels=BAND_LABELS, right=False)


def hex_to_int(s: pd.Series) -> np.ndarray:
    """12-hex-character player IDs to int64 (48 bits), to save memory."""
    return np.array([int(x, 16) for x in s.astype(str)], dtype=np.int64)


def filter_steps(g: pd.DataFrame, lo: float, hi: float) -> list[tuple[str, pd.Series]]:
    """Inclusion criteria in order, each as (label, keep mask)."""
    big_change = (g.white_rating_diff.abs() > PROVISIONAL_CHANGE) | \
                 (g.black_rating_diff.abs() > PROVISIONAL_CHANGE)
    bot = (g.white_title == "BOT") | (g.black_title == "BOT")
    return [
        ("time control", (g.estimated_s >= lo) & (g.estimated_s < hi)),
        ("no bot accounts", ~bot),
        ("not abandoned / rules infraction", ~g.termination.isin(BAD_TERMINATIONS)),
        (f"at least {MIN_PLIES} plies", g.plies >= MIN_PLIES),
        ("no berserk", ~(g.white_berserk | g.black_berserk)),
        ("both ratings present", g.white_elo.notna() & g.black_elo.notna()),
        (f"no provisional rating (|change| <= {PROVISIONAL_CHANGE})", ~big_change.fillna(False)),
        (f"mean rating >= {RATING_FLOOR}", (g.white_elo + g.black_elo) / 2 >= RATING_FLOOR),
    ]


def apply_filters(g: pd.DataFrame, lo: float, hi: float) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    flow = [("parsed (rated, blitz and slower)", len(g))]
    keep = pd.Series(True, index=g.index)
    for label, mask in filter_steps(g, lo, hi):
        keep &= mask.fillna(False).astype(bool)
        flow.append((label, int(keep.sum())))
    return g[keep], flow


def compact_set(g: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "month": g.month.astype("category"),
        "position_id": g.position_id.astype(np.int16),
        "estimated_s": g.estimated_s.astype(np.int32),
        "white_elo": g.white_elo.astype(np.int16),
        "black_elo": g.black_elo.astype(np.int16),
        "score_white": g.score_white.astype(np.float32),
        "white_id": hex_to_int(g.white_id),
        "black_id": hex_to_int(g.black_id),
        "tournament": g.tournament.astype(bool),
        "time_forfeit": (g.termination == "Time forfeit").to_numpy(),
    })
    return out.reset_index(drop=True)


def pair_codes(white: np.ndarray, black: np.ndarray) -> np.ndarray:
    """One code per unordered pair of players (a rematch swaps colours)."""
    a, b = np.minimum(white, black), np.maximum(white, black)
    return pd.factorize(pd.MultiIndex.from_arrays([a, b]))[0]


def randomisation_check(pos: pd.Series) -> dict:
    n = pos.value_counts().reindex(range(N_POSITIONS), fill_value=0)
    e = n.sum() / N_POSITIONS
    chi2 = float(((n - e) ** 2 / e).sum())
    z = (n - e) / np.sqrt(e * (1 - 1 / N_POSITIONS))
    return {"games": int(n.sum()), "expected_per_position": e, "min": int(n.min()),
            "max": int(n.max()), "chi2": chi2, "df": N_POSITIONS - 1,
            "p": float(stats.chi2.sf(chi2, N_POSITIONS - 1)),
            "z_518": float(z[STANDARD_POSITION_ID]), "max_abs_z": float(z.abs().max()),
            "position_max_abs_z": int(z.abs().idxmax())}


def prepare(raw_dir: Path = RAW, out_dir: Path = PREPARED, tables: Path = OUT_TABLES) -> None:
    files = sorted(raw_dir.glob("lichess960_*.parquet"))
    if not files:
        raise SystemExit(f"no monthly tables in {raw_dir}; run lichess960.py fetch first")
    parts = {s: [] for s in SETS}
    flows = {s: None for s in SETS}
    change_q = []
    seen: set[str] = set()
    n_dup = 0
    for f in files:
        g = pd.read_parquet(f, columns=RAW_COLUMNS)
        dup = g.game_id.isin(seen) | g.game_id.duplicated()
        n_dup += int(dup.sum())
        g = g[~dup]
        seen.update(g.game_id)
        for s, (lo, hi) in SETS.items():
            kept, flow = apply_filters(g, lo, hi)
            parts[s].append(compact_set(kept))
            counts = np.array([c for _, c in flow])
            flows[s] = (flow if flows[s] is None else
                        [(lab, c0 + c) for (lab, c0), c in zip(flows[s], counts)])
        tc = g[(g.estimated_s >= 300)]
        change_q.append(np.maximum(tc.white_rating_diff.abs(), tc.black_rating_diff.abs()).dropna()
                        .to_numpy(np.float32))
        log.info("%s: %d games", f.stem, len(g))
    out_dir.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    flow_rows, rand_rows = [], []
    for s in SETS:
        d = pd.concat(parts[s], ignore_index=True)
        d["month"] = d.month.astype(str).astype("category")
        d.to_parquet(out_dir / f"lichess960_{s}.parquet", index=False)
        prev = None
        for step, (label, n) in enumerate(flows[s]):
            flow_rows.append({"set": s, "step": step, "criterion": label, "games": n,
                              "removed": None if prev is None else prev - n})
            prev = n
        rand_rows.append({"set": s, "games_counted": "all", **randomisation_check(d.position_id)})
        # Lichess rematches keep the starting position, so count each
        # pair-position series once: positions are drawn per series.
        series = pd.DataFrame({"pair": pair_codes(d.white_id.to_numpy(), d.black_id.to_numpy()),
                               "pos": d.position_id}).drop_duplicates()
        rand_rows.append({"set": s, "games_counted": "first per pair and position",
                          **randomisation_check(series.pos)})
        log.info("%s: %d games kept", s, len(d))
    pd.DataFrame(flow_rows).to_csv(tables / "lichess960_flow.csv", index=False)
    pd.DataFrame(rand_rows).to_csv(tables / "lichess960_randomisation.csv", index=False)
    ch = np.concatenate(change_q)
    q = {f"p{p}": float(np.percentile(ch, p)) for p in (50, 75, 90, 95, 99)}
    q.update({f"share_over_{t}": float((ch > t).mean()) for t in (20, 30, 50)})
    q["duplicates_removed"] = n_dup
    pd.DataFrame([q]).to_csv(tables / "lichess960_rating_change.csv", index=False)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def frac_logit(X: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None,
               cluster: np.ndarray | None = None, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Fractional logit by Newton-Raphson. Returns coefficients and their
    covariance: sandwich (robust) or, with `cluster` codes, cluster-robust."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = special.expit(X @ b + off)
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < tol:
            break
    p = special.expit(X @ b + off)
    H = X.T @ (X * (p * (1 - p))[:, None])
    U = X * (y - p)[:, None]
    if cluster is None:
        meat = U.T @ U
    else:
        codes, inv = np.unique(cluster, return_inverse=True)
        S = np.column_stack([np.bincount(inv, U[:, j], len(codes)) for j in range(U.shape[1])])
        g = len(codes)
        meat = S.T @ S * g / max(g - 1, 1)
    Hi = np.linalg.inv(H)
    return b, Hi @ meat @ Hi


def rating_x(d: pd.DataFrame) -> np.ndarray:
    return (d.white_elo.to_numpy(float) - d.black_elo.to_numpy(float)) * np.log(10) / 400


def rating_model(d: pd.DataFrame) -> dict:
    X = np.column_stack([np.ones(len(d)), rating_x(d)])
    b, V = frac_logit(X, d.score_white, cluster=d.position_id.to_numpy())
    se = np.sqrt(np.diag(V))
    return {"games": len(d), "advantage_elo": K * b[0], "advantage_se": K * se[0],
            "advantage_lo": K * (b[0] - 1.96 * se[0]), "advantage_hi": K * (b[0] + 1.96 * se[0]),
            "beta": b[1], "beta_se": se[1], "beta_lo": b[1] - 1.96 * se[1],
            "beta_hi": b[1] + 1.96 * se[1], "_b": b}


MATCH_GAP = 100          # rating-matched games: |White - Black| <= this (Dirichlet analyses)


def gap_X(abs_gap: np.ndarray) -> np.ndarray:
    """Absolute rating gap terms for the decisive-rate model (per 100 points,
    linear and quadratic)."""
    g = np.asarray(abs_gap, float) / 100
    return np.column_stack([g, g ** 2])


def decisive_model(abs_gap: np.ndarray, decisive: np.ndarray, cluster: np.ndarray | None = None) -> dict:
    """logit P(decisive) = a + b1 |gap| + b2 |gap|^2: the decisive rate between
    equally rated players is expit(a). Returns the coefficients, the adjusted
    rate with 95% CI and the gap coefficients (used as an offset elsewhere)."""
    nan = {"decisive_adj": np.nan, "decisive_adj_lo": np.nan, "decisive_adj_hi": np.nan,
           "decisive_gap_b1": np.nan, "decisive_gap_b2": np.nan, "_coef": np.array([0.0, 0.0])}
    decisive = np.asarray(decisive, float)
    if len(decisive) < 200 or decisive.min() == decisive.max():
        return nan
    X = np.column_stack([np.ones(len(decisive)), gap_X(abs_gap)])
    try:
        b, V = frac_logit(X, decisive, cluster=cluster)
    except np.linalg.LinAlgError:
        return nan
    se = float(np.sqrt(V[0, 0]))
    return {"decisive_adj": float(special.expit(b[0])),
            "decisive_adj_lo": float(special.expit(b[0] - 1.96 * se)),
            "decisive_adj_hi": float(special.expit(b[0] + 1.96 * se)),
            "decisive_gap_b1": float(b[1]), "decisive_gap_b2": float(b[2]), "_coef": b[1:]}


def per_position_decisive(pos: np.ndarray, abs_gap: np.ndarray, decisive: np.ndarray,
                          coef: np.ndarray, cluster: np.ndarray | None = None,
                          n_positions: int = N_POSITIONS) -> pd.DataFrame:
    """Each position's decisive rate between equally rated players: one
    intercept per position with the pooled gap terms as an offset (Newton,
    vectorised); robust or cluster-robust SE on the logit scale."""
    pos = np.asarray(pos, np.int64)
    y = np.asarray(decisive, float)
    off = gap_X(abs_gap) @ np.asarray(coef, float)
    a = np.zeros(n_positions)
    for _ in range(50):
        p = special.expit(a[pos] + off)
        g = np.bincount(pos, y - p, n_positions)
        h = np.bincount(pos, p * (1 - p), n_positions)
        step = np.divide(g, h, out=np.zeros_like(g), where=h > 0)
        a += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = special.expit(a[pos] + off)
    r = y - p
    h = np.bincount(pos, p * (1 - p), n_positions)
    if cluster is None:
        num = np.bincount(pos, r ** 2, n_positions)
    else:
        sums = pd.DataFrame({"pos": pos, "c": cluster, "r": r}).groupby(["pos", "c"], sort=False).r.sum()
        num = (sums ** 2).groupby(level=0).sum().reindex(range(n_positions), fill_value=0).to_numpy()
    se = np.sqrt(num) / np.where(h > 0, h, np.nan)
    n = np.bincount(pos, minlength=n_positions)
    k = np.bincount(pos, y, n_positions)
    bad = (n < 2) | (k == 0) | (k == n) | ~(se > 0)
    out = pd.DataFrame({"decisive_adj": special.expit(a), "decisive_adj_lo": special.expit(a - 1.96 * se),
                        "decisive_adj_hi": special.expit(a + 1.96 * se)})
    out.loc[bad, :] = np.nan
    return out


def stage1(d: pd.DataFrame, beta: float) -> pd.DataFrame:
    """One intercept per position with the rating gap as an offset: White's
    human advantage at equal ratings, in Elo, with SEs clustered on the
    player pair within each position (rematches)."""
    with np.errstate(divide="ignore", invalid="ignore"):   # positions without games -> NaN
        return _stage1(d, beta)


def _stage1(d: pd.DataFrame, beta: float) -> pd.DataFrame:
    pos = d.position_id.to_numpy(np.int64)
    y = d.score_white.to_numpy(float)
    off = beta * rating_x(d)
    a = np.zeros(N_POSITIONS)
    for _ in range(50):
        p = special.expit(a[pos] + off)
        g = np.bincount(pos, y - p, N_POSITIONS)
        h = np.bincount(pos, p * (1 - p), N_POSITIONS)
        step = np.divide(g, h, out=np.zeros_like(g), where=h > 0)
        a += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = special.expit(a[pos] + off)
    r = y - p
    h = np.bincount(pos, p * (1 - p), N_POSITIONS)
    se_game = np.sqrt(np.bincount(pos, r ** 2, N_POSITIONS)) / h
    # Rematches between the same two players keep the position, so cluster
    # on the (unordered) pair within each position; this SE is the one used.
    pair = pair_codes(d.white_id.to_numpy(), d.black_id.to_numpy())
    sums = pd.DataFrame({"pos": pos, "pair": pair, "r": r}).groupby(["pos", "pair"], sort=False).r.sum()
    se = np.sqrt((sums ** 2).groupby(level=0).sum().reindex(range(N_POSITIONS), fill_value=0)
                 .to_numpy()) / h
    n = np.bincount(pos, minlength=N_POSITIONS)
    out = {"position_id": np.arange(N_POSITIONS), "games": n,
           "white_wins": np.bincount(pos, y == 1, N_POSITIONS).astype(int),
           "draws": np.bincount(pos, y == 0.5, N_POSITIONS).astype(int),
           "black_wins": np.bincount(pos, y == 0, N_POSITIONS).astype(int),
           "escore_observed": np.bincount(pos, y, N_POSITIONS) / n,
           "human_elo": K * a, "human_se": K * se,
           "human_lo": K * (a - 1.96 * se), "human_hi": K * (a + 1.96 * se),
           "se_ratio_pair_cluster": se / se_game}
    out = pd.DataFrame(out)
    # Rating-matched counts (|gap| <= MATCH_GAP), for the Dirichlet analyses.
    gap = np.abs(d.white_elo.to_numpy(float) - d.black_elo.to_numpy(float))
    m = gap <= MATCH_GAP
    out["games_matched"] = np.bincount(pos[m], minlength=N_POSITIONS)
    out["white_wins_matched"] = np.bincount(pos[m], y[m] == 1, N_POSITIONS).astype(int)
    out["draws_matched"] = np.bincount(pos[m], y[m] == 0.5, N_POSITIONS).astype(int)
    out["black_wins_matched"] = np.bincount(pos[m], y[m] == 0, N_POSITIONS).astype(int)
    # Decisive rate between equally rated players, per position.
    dec = (y != 0.5).astype(float)
    dm = decisive_model(gap, dec, cluster=pos)
    out = pd.concat([out, per_position_decisive(pos, gap, dec, dm["_coef"], cluster=pair)], axis=1)
    # Too few games, or all wins / all losses: no finite estimate (sparse sets).
    bad = (n < 2) | (out.escore_observed <= 0) | (out.escore_observed >= 1) | ~(out.human_se > 0)
    out.loc[bad, ["human_elo", "human_se", "human_lo", "human_hi"]] = np.nan
    out["rank_raw"] = out.human_elo.rank(ascending=False, method="min").astype("Int64")
    return out


def descriptives(d: pd.DataFrame) -> pd.DataFrame:
    band = band_of((d.white_elo.astype(float) + d.black_elo.astype(float)) / 2)
    rows = []
    for lab, m in [("all", np.ones(len(d), bool))] + [(b, (band == b).to_numpy()) for b in BAND_LABELS]:
        if not m.any():
            continue
        s = d.score_white.to_numpy(float)[m]
        e = s.mean()
        gap = np.abs(d.white_elo.to_numpy(float) - d.black_elo.to_numpy(float))[m]
        dm = decisive_model(gap, (s != 0.5).astype(float), cluster=d.position_id.to_numpy()[m])
        rows.append({"band": lab, "games": int(m.sum()), "white_wins": float((s == 1).mean()),
                     "decisive_raw": float((s != 0.5).mean()),
                     **{k: v for k, v in dm.items() if not k.startswith("_")},
                     "games_matched": int((gap <= MATCH_GAP).sum()),
                     "draws": float((s == 0.5).mean()), "black_wins": float((s == 0).mean()),
                     "escore": e, "delta_elo_raw": 400 * np.log10(e / (1 - e)),
                     "mean_gap": float((d.white_elo.astype(float) - d.black_elo.astype(float))[m].mean()),
                     "time_forfeit": float(d.time_forfeit.to_numpy()[m].mean()),
                     "tournament": float(d.tournament.to_numpy()[m].mean()),
                     "players": int(pd.unique(np.concatenate([d.white_id.to_numpy()[m],
                                                             d.black_id.to_numpy()[m]])).size)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

SET_STYLE = {"blitz": dict(color="#111111", marker="o", mfc="#111111"),
             "rapid": dict(color="#777777", marker="s", mfc="white")}


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------

def group_row(dataset: str, group: str, d: pd.DataFrame) -> dict:
    """White's advantage (fractional logit, beta estimated) and decisive rate
    at equal ratings for one set of games, both with SEs clustered by
    starting position."""
    rm = rating_model(d)
    s = d.score_white.to_numpy(float)
    gap = np.abs(d.white_elo.to_numpy(float) - d.black_elo.to_numpy(float))
    dm = decisive_model(gap, (s != 0.5).astype(float), cluster=d.position_id.to_numpy())
    mean_r = (d.white_elo.astype(float) + d.black_elo.astype(float)) / 2
    return {"dataset": dataset, "group": group, "games": len(d),
            "rating_range": f"{mean_r.min():.0f}-{mean_r.max():.0f}",
            "advantage_elo": rm["advantage_elo"], "advantage_lo": rm["advantage_lo"],
            "advantage_hi": rm["advantage_hi"], "beta": rm["beta"],
            "decisive_raw": float((s != 0.5).mean()), "decisive_adj": dm["decisive_adj"],
            "decisive_adj_lo": dm["decisive_adj_lo"], "decisive_adj_hi": dm["decisive_adj_hi"]}


def mean_rating_trend(d: pd.DataFrame) -> dict:
    """Change in White's advantage (Delta_Elo) per 100 points of mean rating:
    logit E[s] = alpha + beta (ln10/400) D + delta (M - 1750)/100."""
    mc = ((d.white_elo.to_numpy(float) + d.black_elo.to_numpy(float)) / 2 - RATING_CENTRE) / 100
    X = np.column_stack([np.ones(len(d)), rating_x(d), mc])
    b, V = frac_logit(X, d.score_white.to_numpy(float), cluster=d.position_id.to_numpy())
    se = np.sqrt(np.diag(V))
    return {"advantage_at_1750": K * b[0], "delta_elo_per_100": K * b[2],
            "delta_lo": K * (b[2] - 1.96 * se[2]), "delta_hi": K * (b[2] + 1.96 * se[2]),
            "p": float(2 * stats.norm.sf(abs(b[2] / se[2])))}


def per_position(d: pd.DataFrame) -> pd.DataFrame:
    """Per-position advantage refitted within a subset of games (beta from the
    subset's own rating model)."""
    rm = rating_model(d)
    p = stage1(d, rm["_b"][1])
    return p.loc[p.human_elo.notna(), ["position_id", "games", "human_elo", "human_se"]]


def subgroups(d: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All games and each rating band: advantage and decisive rate at equal
    ratings, and per-position estimates within each band."""
    rows, pos = [group_row(dataset, "all", d)], []
    band = band_of((d.white_elo.astype(float) + d.black_elo.astype(float)) / 2)
    for lab in BAND_LABELS:
        sub = d[(band == lab).to_numpy()]
        if len(sub) >= MIN_BAND_GAMES:
            rows.append(group_row(dataset, lab, sub))
        if len(sub) >= MIN_BAND_POSITIONS_GAMES:
            pos.append(per_position(sub).assign(dataset=dataset, group=lab))
    return pd.DataFrame(rows), pd.concat(pos, ignore_index=True)


def summary_text(res: dict) -> str:
    L = ["Lichess Chess960 human play: analysis summary", "=" * 46]
    for s in SETS:
        r = res[s]
        rm, a = r["rating"], r["desc"]
        a = a[a.band == "all"].iloc[0]
        t = r["trend"]
        pos = r["positions"]
        p518 = pos.set_index("position_id").loc[STANDARD_POSITION_ID]
        L += ["", f"{SET_LABELS[s]} ({len(r['data']):,} games, {a.players:,} players)", "-" * 40,
              f"W/D/L {100 * a.white_wins:.1f} / {100 * a.draws:.1f} / {100 * a.black_wins:.1f}%; "
              f"E_score {a.escore:.4f}; raw Delta_Elo {a.delta_elo_raw:+.1f}",
              f"White's advantage at equal ratings {rm['advantage_elo']:+.1f} "
              f"[{rm['advantage_lo']:+.1f}, {rm['advantage_hi']:+.1f}] Elo; rating-gap coefficient "
              f"beta {rm['beta']:.3f} [{rm['beta_lo']:.3f}, {rm['beta_hi']:.3f}]",
              f"Change per 100 rating points: {t['delta_elo_per_100']:+.2f} [{t['delta_lo']:+.2f}, "
              f"{t['delta_hi']:+.2f}] Elo",
              f"Decisive rate at equal ratings {100 * a.decisive_adj:.1f}% "
              f"[{100 * a.decisive_adj_lo:.1f}, {100 * a.decisive_adj_hi:.1f}]; rating-matched games "
              f"(|gap| <= {MATCH_GAP}): {a.games_matched:,}",
              f"Games per position: median {pos.games.median():.0f}; per-position 95% CIs including 0: "
              f"{((pos.human_lo <= 0) & (pos.human_hi >= 0)).sum()} of {pos.human_elo.notna().sum()}",
              f"#518: {p518.games:,.0f} games, {p518.human_elo:+.1f} [{p518.human_lo:+.1f}, "
              f"{p518.human_hi:+.1f}]",
              "By rating band: advantage [95% CI]; decisive rate at equal ratings"]
        for b in r["groups"].itertuples():
            L.append(f"  {b.group:<10} n {b.games:>9,}: {b.advantage_elo:+6.1f} [{b.advantage_lo:+.1f}, "
                     f"{b.advantage_hi:+.1f}]; {100 * b.decisive_adj:.1f}%")
    return "\n".join(L) + "\n"


def analyse(prepared: Path = PREPARED, tables: Path = OUT_TABLES) -> dict:
    tables.mkdir(parents=True, exist_ok=True)
    res = {}
    for s in SETS:
        d = pd.read_parquet(prepared / f"lichess960_{s}.parquet")
        log.info("%s: %d games", s, len(d))
        rm = rating_model(d)
        grp, band_pos = subgroups(d, f"lichess_{s}")
        res[s] = {"data": d, "rating": rm, "desc": descriptives(d), "positions": stage1(d, rm["_b"][1]),
                  "trend": mean_rating_trend(d), "groups": grp, "band_positions": band_pos}

    def stack(key):
        return pd.concat([res[s][key].assign(set=s) for s in SETS], ignore_index=True)
    stack("desc").to_csv(tables / "lichess960_descriptives.csv", index=False)
    pd.DataFrame([{k: v for k, v in res[s]["rating"].items() if not k.startswith("_")} | {"set": s}
                  for s in SETS]).to_csv(tables / "lichess960_rating_model.csv", index=False)
    stack("positions").to_csv(tables / "lichess960_positions.csv", index=False)
    pd.concat([res[s]["groups"] for s in SETS]).to_csv(tables / "lichess960_subgroups.csv", index=False)
    pd.concat([res[s]["band_positions"] for s in SETS]).to_csv(
        tables / "lichess960_positions_by_band.csv", index=False)
    pd.DataFrame([{"dataset": f"lichess_{s}", **res[s]["trend"]} for s in SETS]).to_csv(
        tables / "lichess960_rating_trend.csv", index=False)
    text = summary_text(res)
    (tables / "lichess960_summary.txt").write_text(text)
    print(text)
    return res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    for name, h in (("prepare", "filter the monthly tables into blitz and rapid sets"),
                    ("analyse", "models and tables"), ("all", "prepare then analyse")):
        sub.add_parser(name, parents=[common], help=h)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    if args.command in ("prepare", "all"):
        prepare()
    if args.command in ("analyse", "all"):
        analyse()
    return 0


if __name__ == "__main__":
    sys.exit(main())
