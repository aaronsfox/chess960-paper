# A quantitative evaluation of fairness in Chess960: data and code

Data and code for the paper **"A quantitative evaluation of fairness in
Chess960"** (Aaron S. Fox, Deakin University). The study asks whether the 960
starting positions of Chess960 are equally fair, using three sources of
evidence:

1. **Engine evaluation**: all 960 starting positions evaluated by Stockfish,
   Caissa, Lc0 and PlentyChess across a ladder of search budgets.
2. **Engine self-play**: 500 games per position for each of Stockfish and
   Caissa (960,000 games in total), from a balanced opening book.
3. **Human play**: about 1.76 million rated games from Lichess (blitz and
   rapid), Chess.com's titled Freestyle Friday events and master-level
   over-the-board events.

This repository holds everything needed to reproduce every table and figure in
the paper, and the code and engine build recipes that generated the engine
data.

## Contents

| Folder | What it holds |
| --- | --- |
| `data/` | The datasets analysed in the paper. See [`data/README.md`](data/README.md) for every file and column. |
| `code/` | Analysis, figure and data-generation scripts (listed below). |
| `results/` | Every table and figure, as produced by `reproduce.py` (committed, so you can compare your rerun). |
| `engines/` | Build recipes and settings for the four chess engines, and the checksums of the exact builds used (`engines.lock.json`). No binaries. |
| `tools/fastchess/` | Build script for fastchess, which ran the self-play matches. |
| `config/` | Settings for the evaluations, opening book and self-play, and the list of master-level event sources. |
| `tests/` | Unit tests (`pytest`). |
| `reproduce.py` | Runs the whole analysis in order. |

## Quick start: reproduce the paper

Requires Python 3.11 or later. Set up an environment with **one** of the
following.

**conda**

```bash
conda env create -f environment.yml
conda activate chess960
```

**uv** (exact pinned versions from `uv.lock`)

```bash
uv sync --extra figures
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**pip**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install chess numpy pandas pyarrow scipy zstandard matplotlib cairosvg pytest
```

`cairosvg` (board diagrams only) needs the Cairo library. conda-forge installs
it automatically; with pip on Linux use your package manager
(e.g. `apt install libcairo2`), and on macOS `brew install cairo`.

Then, from the repository root:

```bash
python reproduce.py            # all steps, about 10 minutes
python -m pytest               # optional: run the tests
```

`python reproduce.py --list` shows the steps; `--only <step>` runs one and
`--from <step>` starts partway. Tables go to `results/<step>/` and figures to
`results/figures/` (PDF and PNG).

The Freestyle Friday game table is not distributed (see
[Data availability](#data-availability)), so that step is skipped and its
committed results in `results/chesscom960/` are used by the later steps.
To rerun it, rebuild the table first (below).

## Where each table and figure comes from

| Paper | File in `results/` | Script |
| --- | --- | --- |
| Figure 1 | `figures/intro_positions` | `plot_boards.py` |
| Figure 2a, 2b | `figures/evaluate_sp_balance_distribution_escore`, `..._delta_elo` | `analyse_sp.py` |
| Figure 3 | `figures/evaluate_sp_boards_unbalanced` | `plot_boards.py` |
| Figure 4 | `figures/evaluate_sp_stability_step_change_delta_elo` | `analyse_sp.py` |
| Figure 5 | `figures/evaluate_sp_agreement_scatter_matrix` | `analyse_sp.py` |
| Figure 6 | `figures/selfplay_book_examples` | `plot_boards.py` |
| Figure 7 | `figures/selfplay_ternary` | `analyse_selfplay.py` |
| Figure 8 | `figures/selfplay_balance_vs_evaluation` | `analyse_selfplay.py` |
| Figure 9 | `figures/selfplay_balance_ranked` | `analyse_selfplay.py` |
| Figure 10 | `figures/selfplay_agreement` | `analyse_selfplay.py` |
| Figure 11 | `figures/human_levels` | `figures_human.py` |
| Figure 12 | `figures/human_positions_ranked` | `figures_human.py` |
| Figure 13 | `figures/human_vs_selfplay_stockfish` (Caissa version: `human_vs_selfplay_caissa`) | `figures_human.py` |
| Figure 14 | `figures/compare_ternary` | `compare_levels.py` |
| Table 1 | `tables/table1_datasets.csv` | `tables_human.py` |
| Table 2 | `tables/table2_advantage_decisive.csv` | `tables_human.py` |
| Part I results | `evaluate_sp/evaluate_sp_summary.txt` and tables | `analyse_sp.py` |
| Part II results | `selfplay/selfplay_summary.txt` and tables | `analyse_selfplay.py` |
| Part III results | `lichess960/`, `chesscom960/`, `masters960/` summaries; `compare_human/compare_human_summary.txt` (heterogeneity, transfer, decisiveness); `compare_levels/compare_summary.txt` (Dirichlet) | as named |

## Code

All scripts are in `code/`, run from the repository root, and print their
options with `--help`. Each file's docstring describes its methods.

**Analysis (used by `reproduce.py`)**

| Script | Paper section | Does |
| --- | --- | --- |
| `analyse_sp.py` | Part I | Balance, stability across budgets and agreement between engines of the starting-position evaluations |
| `analyse_selfplay.py` | Part II | Dirichlet posteriors with a design effect for each position's self-play results; regression on the evaluations; ranking; agreement between engines; decisiveness |
| `analyse_lichess960.py` | Part III | Lichess inclusion criteria; fractional-logit rating model; White's advantage and decisive rate at equal ratings overall, by rating band and per position; trend with mean rating |
| `analyse_chesscom960.py` | Part III | The same for Freestyle Friday (rating thirds) |
| `analyse_masters960.py` | Part III | The same for master-level games (time controls, 2600+, case-study positions) |
| `compare_human_datasets.py` | Part III | Heterogeneity between positions (Cochran's Q), transfer from self-play (Spearman's rho with the expected-transfer benchmark), decisive-rate correlations |
| `compare_levels.py` | Part III | Dirichlet comparison of rating-matched Lichess games with self-play |
| `figures_human.py`, `tables_human.py` | Part III | Figures 11-13, Tables 1-2 |
| `plot_boards.py` | Figures 1, 3, 6 | Board diagrams |
| `plot_style.py`, `utils.py` | | Shared figure style; positions, metrics and engine helpers |

**Data generation (not needed to reproduce the paper)**

| Script | Produced |
| --- | --- |
| `setup_engines.py` | Checks engine installs, smoke-tests them and records their checksums |
| `evaluate_sp.py` | `data/engine_evaluation/` (node-limited evaluations of every position) |
| `opening_book_generator.py` | `data/opening_book/` (50 balanced 6-ply lines per position) |
| `selfplay.py` | `data/selfplay/` (fastchess self-play; `collect` parses the PGNs into tables) |
| `lichess960.py` | `data/lichess960/` (from the Lichess open database exports) |
| `chesscom960.py` | `data/chesscom960/chesscom960_games.parquet` (from the Chess.com Published-Data API) |
| `masters960.py` | `data/masters960/masters960_games.csv` (from the PGN files in `config/masters960_sources.csv`) |

## Rebuilding the Freestyle Friday games

The 80 tournaments analysed are listed in
`data/chesscom960/freestyle_friday_tournaments.csv`. To rebuild the game table
from Chess.com's public API (no login; requests are rate-limited, so this
takes a while) and rerun that step:

```bash
python code/chesscom960.py fetch      # one PGN per listed tournament into data/chesscom960/pgn/
python code/chesscom960.py parse      # -> data/chesscom960/chesscom960_games.parquet
python reproduce.py --from freestyle
```

Player names are replaced by salted hashes and game URLs are not kept.
Results should match the paper exactly, unless Chess.com has since changed or
removed games.

## Regenerating the engine data (optional)

This is computationally expensive and only needed to check the engine data
themselves. The paper's runs used single-threaded, node-limited searches,
so evaluations are reproducible exactly from the same engine builds; self-play
at 10+0.1 is time-based, so individual games differ between runs but the
results per position should agree within their uncertainty.

1. **Build the engines** (Linux). Each engine folder has a `build_linux.sh`
   that downloads the pinned release's source, builds it and installs it in
   that folder's `bin/`. See [`engines/README.md`](engines/README.md) and each
   engine's own README for versions, build options and Windows installs.

   ```bash
   bash engines/stockfish/build_linux.sh     # Stockfish 19
   bash engines/caissa/build_linux.sh        # Caissa 2.0
   bash engines/plentychess/build_linux.sh   # PlentyChess 8.0.0
   LC0_CUDA_CC=80 bash engines/lc0/build_linux.sh   # Lc0 v0.32.1, CUDA build (needs meson and ninja)
   bash tools/fastchess/build_linux.sh       # fastchess v1.8.2-alpha
   python code/setup_engines.py check        # present and configured
   python code/setup_engines.py smoke        # legal moves from all 960 positions
   ```

   `engines.lock.json` holds the SHA-256 of the builds used in the paper. The
   generation scripts refuse to run if a binary or network differs
   (override with `--allow-unlocked`; record a new build with
   `python code/setup_engines.py record`). Native builds may differ in their
   hash from the paper's while searching identically; compare `bench` node
   counts to check.

2. **Evaluate the starting positions** (settings in `config/evaluate_sp.toml`):

   ```bash
   python code/evaluate_sp.py run --engine stockfish      # likewise caissa, lc0, plentychess
   python code/evaluate_sp.py collect
   ```

   About 640 (Stockfish), 400 (Caissa) and 730 (PlentyChess) core-hours, and
   about 45 GPU-hours for Lc0 (NVIDIA A100).

3. **Build the opening book** (`config/opening_book.toml`; about 270 core-hours):

   ```bash
   python code/opening_book_generator.py build
   python code/opening_book_generator.py collect
   ```

4. **Play the self-play matches** (`config/selfplay.toml`; about 4,000
   core-hours per engine):

   ```bash
   python code/selfplay.py run --engine stockfish        # and caissa
   python code/selfplay.py collect
   ```

All three accept `--positions` (e.g. `--positions 518,868` or a file
`@ids.txt`) to run a subset, and skip positions already finished, so they can
be split across machines. Outputs go to `data/` (raw per-position files and
run records under `raw/` and `runs/`, which are not committed). The paper's
runs were on a SLURM cluster (Intel Xeon Gold 6240R CPUs; A100 GPUs for Lc0).

## Data availability

| Source | In this repository | Licence |
| --- | --- | --- |
| Engine evaluations, opening book, self-play games | Yes | CC BY 4.0 |
| Lichess games (Oct 2024 - Sep 2026) | Yes, with player names and game IDs replaced by salted hashes | CC0 (Lichess open database) |
| Master-level over-the-board games | Yes | Public tournament records (The Week in Chess, Lichess broadcasts) |
| Chess.com Freestyle Friday games | No: tournament list and code to rebuild; all derived results included | Chess.com retains rights to its data |

Details are in [`DATA_LICENSE.md`](DATA_LICENSE.md).

## Citation

Please cite the paper and this archive; see [`CITATION.cff`](CITATION.cff).

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Data: see [`DATA_LICENSE.md`](DATA_LICENSE.md).
The chess engines are not distributed here; each is under its own licence
(Stockfish, Lc0 and PlentyChess GPL-3.0; Caissa MIT; fastchess per its repository).

## Use of AI

Claude (Anthropic, Claude Opus 5.5) was used to assist with code development,
building software on the computing cluster, data analysis and figure
generation, as disclosed in the paper. All code and results were reviewed by
the author.
