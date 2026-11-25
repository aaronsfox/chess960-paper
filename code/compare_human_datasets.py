"""Comparisons across the human-play datasets and with engine self-play (Part III).

Reads only the outputs of analyse_lichess960.py, analyse_chesscom960.py,
analyse_masters960.py and analyse_selfplay.py (no game-level data), so it runs
from the committed results. Datasets: Lichess blitz, Lichess rapid, Chess.com
Freestyle Friday and master-level over-the-board games (positions with at
least 20 games).

1. White's advantage and decisive rate at equal ratings by dataset and
   subgroup, with engine self-play pooled over all 960 positions (Table 2,
   Figure 11), and the Lichess change in advantage per 100 rating points.
2. Do positions differ within a dataset? Cochran's Q on the per-position
   Delta_Elo (reported as Q/df) and the number of positions whose 95% CI
   excludes the pooled estimate (5% expected by chance), and the
   between-position SD of the true advantage (tau: DerSimonian-Laird, with a
   Q-profile 95% CI).
3. Transfer of engine imbalance: Spearman rho between per-position human
   Delta_Elo and Stockfish / Caissa self-play posterior median Delta_Elo,
   overall and within Lichess rating bands and Freestyle Friday rating thirds,
   with bootstrap 95% CIs (2,000 resamples) and the rho expected under full
   transfer (each position's human value simulated as its engine value plus
   noise with that position's SE; 200 simulations) (Figure 13). Attenuation
   slope: inverse-variance weighted regression of per-position human
   Delta_Elo on self-play Delta_Elo (the fraction of the engine imbalance
   present in human play; 1 = full transfer, 0 = none), with the slope and its
   upper 95% limit divided by the reliability of the self-play estimates
   (1 - mean posterior variance / variance of the posterior medians).
4. Decisive rate by position: Spearman rho between per-position decisive
   rates at equal ratings (Lichess, Freestyle Friday) and self-play posterior
   mean decisive rates, bootstrap 95% CIs.

Outputs in results/compare_human/.

    python code/compare_human_datasets.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from utils import RESULTS_DIR, setup_logging

log = logging.getLogger("compare_human_datasets")

LICHESS = RESULTS_DIR / "lichess960"
CHESSCOM = RESULTS_DIR / "chesscom960"
MASTERS = RESULTS_DIR / "masters960"
SELFPLAY = RESULTS_DIR / "selfplay" / "selfplay_posteriors.csv"
OUT = RESULTS_DIR / "compare_human"

N_BOOT = 2000
N_SIM = 200
MIN_CASE = 20            # master positions used in rank comparisons
ENGINES = ("stockfish", "caissa")
DATASET_LABELS = {"lichess_blitz": "Lichess Blitz", "lichess_rapid": "Lichess Rapid",
                  "freestyle_friday": "Freestyle Friday", "masters": "Masters OTB"}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def position_estimates() -> dict[str, pd.DataFrame]:
    """Per-position Delta_Elo (rating-adjusted) and SE for each dataset."""
    out = {}
    li = pd.read_csv(LICHESS / "lichess960_positions.csv")
    for s in ("blitz", "rapid"):
        p = li[(li.set == s) & li.human_elo.notna()]
        out[f"lichess_{s}"] = p[["position_id", "games", "human_elo", "human_se"]].rename(
            columns={"human_elo": "elo", "human_se": "se"})
    cc = pd.read_csv(CHESSCOM / "chesscom960_positions.csv")
    cc = cc[cc.human_elo.notna()]
    out["freestyle_friday"] = cc[["position_id", "games", "human_elo", "human_se"]].rename(
        columns={"human_elo": "elo", "human_se": "se"})
    ma = pd.read_csv(MASTERS / "masters960_positions.csv")
    ma = ma[(ma.games >= MIN_CASE) & ma.advantage_elo.notna()]
    out["masters"] = ma[["position_id", "games", "advantage_elo", "advantage_se"]].rename(
        columns={"advantage_elo": "elo", "advantage_se": "se"})
    return {k: v.set_index("position_id") for k, v in out.items()}


def engine_values() -> pd.DataFrame:
    sp = pd.read_csv(SELFPLAY)
    return sp.pivot(index="position_id", columns="engine", values="delta_elo_median")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cochran_q(y: np.ndarray, se: np.ndarray) -> dict:
    w = 1 / se ** 2
    mean = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - mean) ** 2))
    df = len(y) - 1
    return {"k": len(y), "pooled": float(mean), "pooled_se": float(np.sqrt(1 / np.sum(w))),
            "Q": q, "df": df, "Q_over_df": q / df if df else np.nan,
            "p": float(stats.chi2.sf(q, df)) if df else np.nan,
            "I2": max(0.0, (q - df) / q) if q > 0 else 0.0}


def spearman_boot(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    rho = float(stats.spearmanr(x, y).statistic)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    boot = np.array([stats.spearmanr(x[i], y[i]).statistic for i in idx])
    return rho, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def rho_full_transfer(x: np.ndarray, se: np.ndarray, rng: np.random.Generator,
                      se2: np.ndarray | None = None) -> float:
    """Mean Spearman rho expected if the human estimates were the engine
    values plus their own sampling noise. With se2, both sides are human
    datasets reflecting the engine values (each with its own noise)."""
    sims = []
    for _ in range(N_SIM):
        a = x + rng.normal(0, se)
        b = x if se2 is None else x + rng.normal(0, se2)
        sims.append(stats.spearmanr(b, a).statistic)
    return float(np.mean(sims))


def seed(*parts) -> np.random.Generator:
    import zlib
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def tau_dl(y: np.ndarray, se: np.ndarray) -> dict:
    """Between-position SD of the true effects (tau): DerSimonian-Laird
    moment estimate from Cochran's Q, with a 95% CI from the Q-profile method
    (Viechtbauer 2007): the tau^2 values at which the generalised Q,
    sum (y - mu(tau^2))^2 / (se^2 + tau^2), equals the 97.5% and 2.5%
    chi-square quantiles on k - 1 df (truncated at zero)."""
    from scipy import optimize
    w = 1 / se ** 2
    df = len(y) - 1
    mu = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - mu) ** 2))
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - df) / c)

    def gen_q(t2: float) -> float:
        ww = 1 / (se ** 2 + t2)
        m = np.sum(ww * y) / np.sum(ww)
        return float(np.sum(ww * (y - m) ** 2))

    def solve(target: float) -> float:
        if gen_q(0.0) <= target:
            return 0.0
        hi = max(1.0, 10 * float(np.var(y)))
        while gen_q(hi) > target:
            hi *= 10
        return float(optimize.brentq(lambda t: gen_q(t) - target, 0.0, hi, xtol=1e-10))

    lo2 = solve(stats.chi2.ppf(0.975, df))
    hi2 = solve(stats.chi2.ppf(0.025, df))
    return {"tau": float(np.sqrt(tau2)), "tau_lo": float(np.sqrt(lo2)), "tau_hi": float(np.sqrt(hi2))}


def engine_reliability(sp: pd.DataFrame) -> pd.DataFrame:
    """Reliability of the per-position self-play estimates: one minus the
    ratio of the mean posterior variance to the variance of the posterior
    medians across positions; the between-position SD net of posterior
    uncertainty is sqrt(var(medians) - mean posterior variance)."""
    rows = []
    for e, d in sp.groupby("engine"):
        v = float(d.delta_elo_median.var(ddof=1))
        noise = float((d.delta_elo_sd ** 2).mean())
        rows.append({"engine": e, "positions": len(d), "sd_posterior_medians": np.sqrt(v),
                     "mean_posterior_sd": float(d.delta_elo_sd.mean()),
                     "sd_between_positions": np.sqrt(max(0.0, v - noise)),
                     "reliability": 1 - noise / v})
    return pd.DataFrame(rows)


def attenuation_slope(y: np.ndarray, se: np.ndarray, x: np.ndarray) -> dict:
    """Inverse-variance weighted regression of the per-position human
    Delta_Elo (y, SE se) on the engine Delta_Elo (x). The slope is the
    fraction of the engine-measured imbalance present in human play
    (1 = full transfer, 0 = none). The covariance is scaled by the residual
    dispersion when this exceeds 1 (fixed-effect meta-regression with
    multiplicative overdispersion)."""
    w = 1 / se ** 2
    X = np.column_stack([np.ones_like(x), x])
    xtwx = X.T @ (w[:, None] * X)
    b = np.linalg.solve(xtwx, X.T @ (w * y))
    resid = y - X @ b
    phi = float(np.sum(w * resid ** 2) / (len(y) - 2))
    cov = np.linalg.inv(xtwx) * max(1.0, phi)
    s, s_se = float(b[1]), float(np.sqrt(cov[1, 1]))
    return {"intercept": float(b[0]), "slope": s, "slope_se": s_se,
            "slope_lo": s - 1.96 * s_se, "slope_hi": s + 1.96 * s_se, "dispersion": phi}


def transfer_slopes(pos: dict[str, pd.DataFrame], sp: pd.DataFrame, eng: pd.DataFrame) -> pd.DataFrame:
    """Attenuation slope per dataset and engine, with the slope and its upper
    95% limit divided by the reliability of the engine estimates (correcting
    regression dilution from uncertainty in the engine values)."""
    rel = engine_reliability(sp).set_index("engine").reliability
    rows = []
    for k, p in pos.items():
        for e in ENGINES:
            a = attenuation_slope(p.elo.to_numpy(), p.se.to_numpy(), eng.loc[p.index, e].to_numpy())
            rows.append({"dataset": k, "engine": e, "positions": len(p), **a,
                         "reliability": float(rel[e]),
                         "slope_corrected": a["slope"] / rel[e],
                         "slope_hi_corrected": a["slope_hi"] / rel[e]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Positions within each dataset
# ---------------------------------------------------------------------------

def heterogeneity(pos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for k, p in pos.items():
        q = cochran_q(p.elo.to_numpy(), p.se.to_numpy())
        lo, hi = p.elo - 1.96 * p.se, p.elo + 1.96 * p.se
        excl = int(((lo > q["pooled"]) | (hi < q["pooled"])).sum())
        rows.append({"dataset": k, **q, **tau_dl(p.elo.to_numpy(), p.se.to_numpy()),
                     "ci_excluding_pooled": excl, "expected_by_chance": 0.05 * len(p)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Transfer of engine imbalance (overall and within rating groups)
# ---------------------------------------------------------------------------

def transfer_row(dataset: str, group: str, p: pd.DataFrame, eng: pd.DataFrame) -> list[dict]:
    rows = []
    for e in ENGINES:
        x = eng.loc[p.index, e].to_numpy()
        rng = seed("transfer", dataset, group, e)
        rho = spearman_boot(x, p.elo.to_numpy(), rng)
        rows.append({"dataset": dataset, "group": group, "engine": e, "positions": len(p),
                     "median_games": float(p.games.median()), "median_se": float(p.se.median()),
                     "rho": rho[0], "rho_lo": rho[1], "rho_hi": rho[2],
                     "rho_if_full_transfer": rho_full_transfer(x, p.se.to_numpy(), rng)})
    return rows


def subgroup_positions() -> dict[tuple[str, str], pd.DataFrame]:
    """Per-position estimates refitted within Lichess rating bands and
    Freestyle Friday rating thirds (from the dataset analyses)."""
    parts = [pd.read_csv(LICHESS / "lichess960_positions_by_band.csv"),
             pd.read_csv(CHESSCOM / "chesscom960_positions_by_third.csv")]
    out = {}
    for d in parts:
        for (k, g), p in d.groupby(["dataset", "group"], sort=False):
            out[(k, g)] = p.set_index("position_id")[["games", "human_elo", "human_se"]].rename(
                columns={"human_elo": "elo", "human_se": "se"})
    return out


def transfer(pos: dict[str, pd.DataFrame], eng: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, p in pos.items():
        rows += transfer_row(k, "all", p, eng)
    for (k, group), p in subgroup_positions().items():
        rows += transfer_row(k, group, p, eng)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Balance and decisiveness by dataset and subgroup
# ---------------------------------------------------------------------------

def groups() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Advantage and decisive rate at equal ratings by dataset and subgroup,
    with engine self-play pooled over all 960 positions; Lichess trend."""
    rows = [pd.read_csv(LICHESS / "lichess960_subgroups.csv"),
            pd.read_csv(CHESSCOM / "chesscom960_subgroups.csv")]
    ma = pd.read_csv(MASTERS / "masters960_pooled.csv")
    ma = ma[ma.group.isin(["all", "time control"]) | (ma.level == "2600+")]
    rows.append(pd.DataFrame({"dataset": "masters", "group": ma.level, "games": ma.games,
                              "advantage_elo": ma.advantage_elo, "advantage_lo": ma.advantage_lo,
                              "advantage_hi": ma.advantage_hi, "beta": 1.0, "decisive_raw": 1 - ma.draws,
                              "decisive_adj": ma.decisive_adj, "decisive_adj_lo": ma.decisive_adj_lo,
                              "decisive_adj_hi": ma.decisive_adj_hi}))
    sp = pd.read_csv(SELFPLAY)
    eng = []
    for e, g in sp.groupby("engine"):
        n = g.games.sum()
        e_score = (g.white_wins.sum() + 0.5 * g.draws.sum()) / n
        eng.append({"dataset": f"{e}_selfplay", "group": "all 960 positions", "games": int(n),
                    "advantage_elo": 400 * np.log10(e_score / (1 - e_score)),
                    "decisive_raw": 1 - g.draws.sum() / n, "decisive_adj": 1 - g.draws.sum() / n})
    rows.append(pd.DataFrame(eng))
    trend = pd.read_csv(LICHESS / "lichess960_rating_trend.csv")
    return pd.concat(rows, ignore_index=True), trend


def decisive_spearman() -> pd.DataFrame:
    sp = pd.read_csv(SELFPLAY).set_index(["engine", "position_id"])
    li = pd.read_csv(LICHESS / "lichess960_positions.csv")
    cc = pd.read_csv(CHESSCOM / "chesscom960_positions.csv")
    sets = {"lichess_blitz": li[li.set == "blitz"], "lichess_rapid": li[li.set == "rapid"],
            "freestyle_friday": cc[cc.games >= MIN_CASE]}
    rows = []
    for k, p in sets.items():
        p = p[p.decisive_adj.notna()]
        for e in ENGINES:
            x = sp.loc[e].reindex(p.position_id).decisive_mean.to_numpy()
            rho = spearman_boot(x, p.decisive_adj.to_numpy(), seed("decisive", k, e))
            rows.append({"dataset": k, "engine": e, "positions": len(p),
                         "rho": rho[0], "rho_lo": rho[1], "rho_hi": rho[2]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def summary_text(grp, trend, het, trans, dec, rel=None, slopes=None) -> str:
    lab = lambda k: DATASET_LABELS.get(k, k)
    L = ["Comparisons across human-play datasets", "=" * 38, "",
         "1. White's advantage and decisive rate at equal ratings (95% CI)"]
    for r in grp.itertuples():
        adv = (f"{r.advantage_elo:+6.1f} [{r.advantage_lo:+.1f}, {r.advantage_hi:+.1f}]"
               if pd.notna(r.advantage_lo) else f"{r.advantage_elo:+6.1f}")
        dec_ = (f"{100 * r.decisive_adj:.1f}%" + (f" [{100 * r.decisive_adj_lo:.1f}, {100 * r.decisive_adj_hi:.1f}]"
                if pd.notna(r.decisive_adj_lo) else "")) if pd.notna(r.decisive_adj) else "n/a"
        L.append(f"  {lab(r.dataset):<20} {r.group:<28} n {r.games:>9,}: advantage {adv}; decisive {dec_}")
    for r in trend.itertuples():
        L.append(f"  {lab(r.dataset)}: change in White's advantage per 100 rating points "
                 f"{r.delta_elo_per_100:+.2f} [{r.delta_lo:+.2f}, {r.delta_hi:+.2f}] Elo (p {r.p:.3g})")
    L += ["", "2. Do positions differ within a dataset? (Cochran's Q on per-position Delta_Elo)"]
    for r in het.itertuples():
        L.append(f"  {lab(r.dataset):<17} k {r.k}: Q/df {r.Q_over_df:.2f}, p {r.p:.2f}; CIs excluding "
                 f"the pooled {r.pooled:+.1f}: {r.ci_excluding_pooled} (expected {r.expected_by_chance:.0f}); "
                 f"tau {r.tau:.1f} [{r.tau_lo:.1f}, {r.tau_hi:.1f}] Elo")
    L += ["", "3. Transfer of engine imbalance (Spearman rho, human vs self-play Delta_Elo)"]
    for r in trans.itertuples():
        L.append(f"  {lab(r.dataset):<17} {r.group:<28} {r.engine:<9} k {r.positions:>3}: rho {r.rho:+.2f} "
                 f"[{r.rho_lo:+.2f}, {r.rho_hi:+.2f}]; expected transfer rho {r.rho_if_full_transfer:.2f}")
    if rel is not None:
        L += ["", "3b. Reliability of the self-play per-position estimates"]
        for r in rel.itertuples():
            L.append(f"  {r.engine:<9} SD of posterior medians {r.sd_posterior_medians:.1f}, between-position SD "
                     f"net of posterior uncertainty {r.sd_between_positions:.1f} Elo; reliability {r.reliability:.2f}")
    if slopes is not None:
        L += ["", "3c. Attenuation slope (human on self-play Delta_Elo; 1 = full transfer)"]
        for r in slopes.itertuples():
            L.append(f"  {lab(r.dataset):<17} {r.engine:<9} k {r.positions:>3}: slope {r.slope:+.2f} "
                     f"[{r.slope_lo:+.2f}, {r.slope_hi:+.2f}]; upper limit corrected for reliability "
                     f"{r.slope_hi_corrected:.2f}")
    L += ["", "4. Decisive rate by position vs engine self-play (Spearman rho)"]
    for r in dec.itertuples():
        L.append(f"  {lab(r.dataset):<17} {r.engine:<9} k {r.positions}: rho {r.rho:+.2f} "
                 f"[{r.rho_lo:+.2f}, {r.rho_hi:+.2f}]")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    OUT.mkdir(parents=True, exist_ok=True)
    pos = position_estimates()
    eng = engine_values()
    grp, trend = groups()
    het = heterogeneity(pos)
    trans = transfer(pos, eng)
    sp = pd.read_csv(SELFPLAY)
    rel = engine_reliability(sp)
    slopes = transfer_slopes(pos, sp, eng)
    dec = decisive_spearman()
    grp.to_csv(OUT / "groups_balance_decisive.csv", index=False)
    trend.to_csv(OUT / "lichess_mean_rating_trend.csv", index=False)
    het.to_csv(OUT / "positions_heterogeneity.csv", index=False)
    trans.to_csv(OUT / "transfer_spearman.csv", index=False)
    dec.to_csv(OUT / "decisive_spearman.csv", index=False)
    rel.to_csv(OUT / "selfplay_reliability.csv", index=False)
    slopes.to_csv(OUT / "transfer_slope.csv", index=False)
    text = summary_text(grp, trend, het, trans, dec, rel, slopes)
    (OUT / "compare_human_summary.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
