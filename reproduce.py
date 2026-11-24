"""Reproduce every table and figure in the paper from the data in data/.

Runs the analysis scripts in order; each writes to results/. Takes about
10 minutes on a laptop (most of it the Lichess step).

    python reproduce.py                 # everything
    python reproduce.py --list          # show the steps
    python reproduce.py --from human    # start at a step (by name)
    python reproduce.py --only figures  # run one step

The Freestyle Friday step needs data/chesscom960/chesscom960_games.parquet,
which is not distributed (see data/README.md). Without it the step is
skipped and the committed results in results/chesscom960/ are used.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
PY = sys.executable

BOOK_LINES = ("518:1,518:14,518:43,518:34,868:0,868:3,868:4,868:32")
BOOK_LABELS = ["Sicilian Defence: Closed (B23)", "Queen's Gambit Declined (D30)",
               "Caro-Kann Defence (B10)", "English: Anglo-Indian, Slav (A15)", "", "", "", ""]

# name, description, commands (each a list of arguments to python)
STEPS = [
    ("evaluation", "Part I: starting-position evaluations (Figures 2, 4, 5)",
     [["analyse_sp.py"]]),
    ("selfplay", "Part II: engine self-play (Figures 7-10)",
     [["analyse_selfplay.py"]]),
    ("lichess", "Part III: Lichess blitz and rapid",
     [["analyse_lichess960.py", "all"]]),
    ("freestyle", "Part III: Chess.com Freestyle Friday",
     [["analyse_chesscom960.py"]]),
    ("masters", "Part III: master-level over-the-board games",
     [["analyse_masters960.py"]]),
    ("human", "Part III: comparisons across datasets and with self-play",
     [["compare_human_datasets.py"]]),
    ("dirichlet", "Part III: Dirichlet comparison of Lichess and self-play (Figure 14)",
     [["compare_levels.py"]]),
    ("figures", "Part III: Figures 11-13 and Tables 1-2",
     [["figures_human.py"], ["tables_human.py"]]),
    ("boards", "Board diagrams (Figures 1, 3, 6)",
     [["plot_boards.py", "--positions", "0,518,625,822", "--engine", "none", "--no-letters",
       "--cols", "4", "--out", "results/figures/intro_positions"],
      ["plot_boards.py", "--positions", "868,879", "--engine", "none", "--no-letters",
       "--cols", "2", "--out", "results/figures/evaluate_sp_boards_unbalanced"],
      ["plot_boards.py", "--lines", BOOK_LINES, "--cols", "4", "--no-letters",
       "--out", "results/figures/selfplay_book_examples", "--labels", *BOOK_LABELS]]),
]
NAMES = [s[0] for s in STEPS]
CHESSCOM_GAMES = ROOT / "data" / "chesscom960" / "chesscom960_games.parquet"


def run(name: str, desc: str, commands: list[list[str]]) -> None:
    print(f"\n=== {name}: {desc}", flush=True)
    if name == "freestyle" and not CHESSCOM_GAMES.exists():
        print(f"skipped: {CHESSCOM_GAMES.relative_to(ROOT)} not found (see data/README.md); "
              "using the committed results in results/chesscom960/")
        return
    for cmd in commands:
        t0 = time.time()
        subprocess.run([PY, str(CODE / cmd[0]), *cmd[1:]], cwd=ROOT, check=True)
        print(f"--- {cmd[0]} done in {time.time() - t0:.0f} s", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--from", dest="start", choices=NAMES, help="start at this step")
    g.add_argument("--only", choices=NAMES, help="run only this step")
    p.add_argument("--list", action="store_true", help="list the steps and exit")
    args = p.parse_args(argv)
    if args.list:
        for name, desc, _ in STEPS:
            print(f"{name:<11} {desc}")
        return 0
    steps = STEPS
    if args.only:
        steps = [s for s in STEPS if s[0] == args.only]
    elif args.start:
        steps = STEPS[NAMES.index(args.start):]
    t0 = time.time()
    for step in steps:
        run(*step)
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min. Tables and figures are in results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
