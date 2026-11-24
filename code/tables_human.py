"""Tables 1 and 2 of the paper (human play), from the dataset analyses.

Table 1: games, players, Chess960 positions played, range of the players'
mean rating, raw White win / draw / Black win percentages and the
rating-gap coefficient beta, per dataset.
Table 2: White's advantage (Delta_Elo) and decisive rate (%) at equal
ratings with 95% CIs, by dataset and subgroup.

Writes results/tables/table1_datasets.csv and table2_advantage_decisive.csv.

    python code/tables_human.py
"""

from __future__ import annotations

import sys

import pandas as pd

from utils import RESULTS_DIR

OUT = RESULTS_DIR / "tables"
LABELS = {"lichess_blitz": "Lichess Blitz", "lichess_rapid": "Lichess Rapid",
          "freestyle_friday": "Freestyle Friday", "masters": "Masters OTB"}
GROUP_LABELS = {"all": "All", "lower third": "Lower", "middle third": "Middle", "upper third": "Upper",
                "classical": "Classical Time Control", "rapid": "Rapid Time Control",
                "2600+": "2600+ Rating"}


def wdl(w: float, d: float, b: float) -> str:
    return f"{100 * w:.1f} / {100 * d:.1f} / {100 * b:.1f}"


def table1() -> pd.DataFrame:
    rows = []
    li_desc = pd.read_csv(RESULTS_DIR / "lichess960" / "lichess960_descriptives.csv")
    li_rm = pd.read_csv(RESULTS_DIR / "lichess960" / "lichess960_rating_model.csv").set_index("set")
    li_grp = pd.read_csv(RESULTS_DIR / "lichess960" / "lichess960_subgroups.csv")
    li_pos = pd.read_csv(RESULTS_DIR / "lichess960" / "lichess960_positions.csv")
    for s in ("blitz", "rapid"):
        a = li_desc[(li_desc.set == s) & (li_desc.band == "all")].iloc[0]
        g = li_grp[(li_grp.dataset == f"lichess_{s}") & (li_grp.group == "all")].iloc[0]
        p = li_pos[(li_pos.set == s) & (li_pos.games > 0)]
        rows.append({"dataset": LABELS[f"lichess_{s}"], "games": int(a.games), "players": int(a.players),
                     "positions": len(p), "median_games_per_position": float(p.games.median()),
                     "rating_range": g.rating_range.replace("-", " - "),
                     "white_draw_black_pct": wdl(a.white_wins, a.draws, a.black_wins),
                     "beta": round(float(li_rm.loc[s, "beta"]), 2)})
    cc = pd.read_csv(RESULTS_DIR / "chesscom960" / "chesscom960_descriptives.csv").iloc[0]
    cc_rm = pd.read_csv(RESULTS_DIR / "chesscom960" / "chesscom960_rating_model.csv").iloc[0]
    cc_pos = pd.read_csv(RESULTS_DIR / "chesscom960" / "chesscom960_positions.csv")
    rows.append({"dataset": LABELS["freestyle_friday"], "games": int(cc.games), "players": int(cc.players),
                 "positions": int(cc.positions),
                 "median_games_per_position": float(cc_pos.games[cc_pos.games > 0].median()),
                 "rating_range": f"{cc.mean_rating_min:.0f} - {cc.mean_rating_max:.0f}",
                 "white_draw_black_pct": wdl(cc.white_wins, cc.draws, cc.black_wins),
                 "beta": round(float(cc_rm.beta), 2)})
    ma = pd.read_csv(RESULTS_DIR / "masters960" / "masters960_descriptives.csv").iloc[0]
    ma_pos = pd.read_csv(RESULTS_DIR / "masters960" / "masters960_positions.csv")
    rows.append({"dataset": LABELS["masters"], "games": int(ma.games), "players": int(ma.players),
                 "positions": int(ma.positions), "median_games_per_position": float(ma_pos.games.median()),
                 "rating_range": f"{ma.mean_rating_min:.0f} - {ma.mean_rating_max:.0f}",
                 "white_draw_black_pct": wdl(ma.white_wins, ma.draws, ma.black_wins),
                 "beta": "1.00 (fixed)"})
    return pd.DataFrame(rows)


def table2() -> pd.DataFrame:
    g = pd.read_csv(RESULTS_DIR / "compare_human" / "groups_balance_decisive.csv")
    g = g[g.dataset.isin(LABELS) & g.group.ne("blitz") & g.decisive_adj.notna()]
    out = pd.DataFrame({
        "dataset": g.dataset.map(LABELS),
        "category": [f"{GROUP_LABELS[x]} ({r.replace('-', ' - ')})" if x.endswith("third")
                     else GROUP_LABELS.get(x, x.replace("-", " - ")) for x, r in zip(g.group, g.rating_range)],
        "games": g.games.astype(int),
        "white_advantage_elo": [f"{a:+.1f} [{lo:.1f}, {hi:.1f}]"
                                for a, lo, hi in zip(g.advantage_elo, g.advantage_lo, g.advantage_hi)],
        "decisive_pct": [f"{100 * d:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
                         for d, lo, hi in zip(g.decisive_adj, g.decisive_adj_lo, g.decisive_adj_hi)]})
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t1, t2 = table1(), table2()
    t1.to_csv(OUT / "table1_datasets.csv", index=False)
    t2.to_csv(OUT / "table2_advantage_decisive.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("Table 1\n", t1.to_string(index=False), "\n\nTable 2\n", t2.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
