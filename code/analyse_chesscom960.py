"""Chess.com Freestyle Friday (titled players, 3+1 and 3+2): analysis.

The same models as the Lichess analysis (analyse_lichess960.py), fitted on
Chess.com ratings, whose scale is estimated from the data and never compared
directly with Lichess or FIDE ratings:

- inclusion: Chess960 position identified, known result, at least 10 plies,
  both ratings present (Table 1 counts);
- rating model: White's score on the rating gap (beta fitted) and White's
  advantage at equal ratings;
- White's advantage and decisive rate at equal ratings, overall and by thirds
  of the players' mean rating (Table 2, Figure 11);
- per-position advantage (SEs clustered on the player pair) and decisive rate
  at equal ratings (Figures 12 and 13);
- per-position advantage refitted within each rating third (transfer by
  third).

Input: data/chesscom960/chesscom960_games.parquet. This table is not
distributed with the repository (Chess.com's data terms); rebuild it from the
Published-Data API with code/chesscom960.py (fetch, then parse). The outputs
of this script, in results/chesscom960/, are committed, so the later steps run
without it.

    python code/analyse_chesscom960.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_lichess960 import (MATCH_GAP, decisive_model, group_row, per_position, rating_model,
                                stage1)
from utils import DATA_DIR, RESULTS_DIR, setup_logging

log = logging.getLogger("analyse_chesscom960")

GAMES = DATA_DIR / "chesscom960" / "chesscom960_games.parquet"   # rebuilt with chesscom960.py
OUT = RESULTS_DIR / "chesscom960"
MIN_PLIES = 10
THIRDS = ["lower third", "middle third", "upper third"]


def include(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = [("Chess960 position identified", g.position_id >= 0),
             ("known result", g.score_white.notna()),
             (f"at least {MIN_PLIES} plies", g.plies >= MIN_PLIES),
             ("both ratings", g.white_elo.notna() & g.black_elo.notna())]
    keep = pd.Series(True, index=g.index)
    flow = [{"criterion": "parsed", "games": len(g)}]
    for label, m in steps:
        keep &= m.fillna(False).astype(bool)
        flow.append({"criterion": label, "games": int(keep.sum())})
    flow = pd.DataFrame(flow)
    flow["removed"] = -flow.games.diff()
    d = g[keep].copy()
    for side in ("white", "black"):          # hashed IDs -> integers (clustering on player pairs)
        d[f"{side}_id"] = [int(x, 16) for x in d[f"{side}_id"]]
    d["position_id"] = d.position_id.astype(int)
    return d.reset_index(drop=True), flow


def descriptives(d: pd.DataFrame) -> dict:
    s = d.score_white.to_numpy(float)
    gap = np.abs(d.white_elo.to_numpy(float) - d.black_elo.to_numpy(float))
    e = s.mean()
    dm = decisive_model(gap, (s != 0.5).astype(float), cluster=d.position_id.to_numpy())
    mean_r = (d.white_elo + d.black_elo) / 2
    return {"games": len(d), "tournaments": d.tournament_id.nunique(),
            "players": pd.unique(np.concatenate([d.white_id, d.black_id])).size,
            "positions": d.position_id.nunique(), "white_wins": float((s == 1).mean()),
            "draws": float((s == 0.5).mean()), "black_wins": float((s == 0).mean()),
            "escore": e, "delta_elo_raw": 400 * np.log10(e / (1 - e)),
            "decisive_raw": float((s != 0.5).mean()),
            **{k: v for k, v in dm.items() if not k.startswith("_")},
            "games_matched": int((gap <= MATCH_GAP).sum()),
            "mean_rating_min": float(mean_r.min()), "mean_rating_max": float(mean_r.max()),
            "mean_rating_median": float(mean_r.median())}


def thirds_of(d: pd.DataFrame) -> pd.Series:
    return pd.qcut((d.white_elo + d.black_elo) / 2, 3, labels=THIRDS)


def subgroups(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All games and each third of mean rating: advantage and decisive rate at
    equal ratings; per-position estimates within each third."""
    rows, pos = [group_row("freestyle_friday", "all", d)], []
    t = thirds_of(d)
    mean_r = (d.white_elo + d.black_elo) / 2
    for lab in THIRDS:
        m = (t == lab).to_numpy()
        rows.append(group_row("freestyle_friday", lab, d[m]))     # rating_range gives the bounds
        pos.append(per_position(d[m]).assign(
            dataset="freestyle_friday", group=f"{lab} ({mean_r[m].min():.0f}-{mean_r[m].max():.0f})"))
    return pd.DataFrame(rows), pd.concat(pos, ignore_index=True)


def summary_text(flow, desc, rm, grp, pos) -> str:
    L = ["Chess.com Freestyle Friday: analysis summary", "=" * 44, "", "Inclusion:"]
    for r in flow.itertuples():
        L.append(f"  {r.criterion:<30} {r.games:>8,}" + (f"  (-{r.removed:,.0f})" if r.removed > 0 else ""))
    L += ["", f"{desc['games']:,} games, {desc['tournaments']} tournaments, {desc['players']:,} players, "
              f"{desc['positions']} positions (median {pos.games[pos.games > 0].median():.0f} games each)",
          f"Mean rating {desc['mean_rating_min']:.0f}-{desc['mean_rating_max']:.0f}",
          f"W/D/L {100 * desc['white_wins']:.1f} / {100 * desc['draws']:.1f} / {100 * desc['black_wins']:.1f}%; "
          f"raw Delta_Elo {desc['delta_elo_raw']:+.1f}",
          f"White's advantage at equal ratings {rm['advantage_elo']:+.1f} [{rm['advantage_lo']:+.1f}, "
          f"{rm['advantage_hi']:+.1f}] Elo; beta {rm['beta']:.3f} [{rm['beta_lo']:.3f}, {rm['beta_hi']:.3f}]",
          "By rating third: advantage [95% CI]; decisive rate at equal ratings [95% CI]"]
    for r in grp.itertuples():
        L.append(f"  {r.group + ' (' + r.rating_range + ')':<28} n {r.games:>6,}: {r.advantage_elo:+6.1f} [{r.advantage_lo:+.1f}, "
                 f"{r.advantage_hi:+.1f}]; {100 * r.decisive_adj:.1f}% [{100 * r.decisive_adj_lo:.1f}, "
                 f"{100 * r.decisive_adj_hi:.1f}]")
    p = pos[pos.human_elo.notna()]
    L.append(f"Per-position 95% CIs including 0: {((p.human_lo <= 0) & (p.human_hi >= 0)).sum()} of {len(p)}; "
             f"decisive rate estimable for {pos.decisive_adj.notna().sum()} positions")
    return "\n".join(L) + "\n"


def analyse(games: Path = GAMES, out: Path = OUT) -> None:
    if not games.exists():
        raise SystemExit(f"{games} not found: rebuild it with code/chesscom960.py fetch and parse "
                         "(the committed results in results/chesscom960/ are used otherwise)")
    d, flow = include(pd.read_parquet(games))
    rm = rating_model(d)
    pos = stage1(d, rm["_b"][1])
    grp, third_pos = subgroups(d)
    desc = descriptives(d)
    out.mkdir(parents=True, exist_ok=True)
    flow.to_csv(out / "chesscom960_flow.csv", index=False)
    pd.DataFrame([desc]).to_csv(out / "chesscom960_descriptives.csv", index=False)
    pd.DataFrame([{k: v for k, v in rm.items() if not k.startswith("_")}]).to_csv(
        out / "chesscom960_rating_model.csv", index=False)
    pos.to_csv(out / "chesscom960_positions.csv", index=False)
    grp.to_csv(out / "chesscom960_subgroups.csv", index=False)
    third_pos.to_csv(out / "chesscom960_positions_by_third.csv", index=False)
    text = summary_text(flow, desc, rm, grp, pos)
    (out / "chesscom960_summary.txt").write_text(text)
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
