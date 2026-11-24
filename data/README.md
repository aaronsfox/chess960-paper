# Data

All inputs to the analyses. Positions use Scharnagl numbering 0-959
(`position_id`), with #518 the standard starting position (RNBQKBNR);
`arrangement` is White's back rank from a to h. Scores are from White's
side: 1 = White win, 0.5 = draw, 0 = Black win. Elo-equivalent values are
`400 * log10(E / (1 - E))` for an expected score E.

## engine_evaluation/tables/ (Part I)

Produced by `code/evaluate_sp.py collect`.

**`evaluate_sp_final.parquet`**: one row per engine, position and node
budget (22,080 rows: Stockfish, Caissa and PlentyChess at 1e4-1e9 nodes; Lc0
at 1e2-1e6 nodes; MultiPV 1).

| Column | Meaning |
| --- | --- |
| `engine`, `engine_version` | Engine key and version |
| `position_id`, `arrangement`, `fen` | Starting position |
| `multipv_setting`, `multipv_index` | MultiPV used for the search, and the line's rank |
| `budget_nodes` | Node limit of the search |
| `depth`, `seldepth`, `nodes`, `time_s`, `bound` | Search statistics of the line used |
| `cp_white`, `mate_white` | Evaluation in centipawns (or mate in n) from White's side |
| `wdl_w`, `wdl_d`, `wdl_l` | Engine's win/draw/loss probabilities, per mille (none for PlentyChess) |
| `expected_score`, `elo_equivalent` | White's expected score (Equation 1) and Delta_Elo (Equation 2) |
| `pv`, `bestmove` | Principal variation and the engine's best move (UCI) |
| `eval_source` | `last_exact`: the last info line with an exact score (normal); `last_line_bound` if none |
| `last_*` | The search's final info line, which may be an aspiration-window bound |
| `search_wall_s` | Wall-clock time of the search |

**`evaluate_sp_convergence.csv`**: per engine and position, the top-budget
values and how they changed across budgets (`tv_*` total variation,
`tail_drift_*` largest change over the three budgets below the top).

## opening_book/book/ (Part II)

Produced by `code/opening_book_generator.py collect`.

| File | Contents |
| --- | --- |
| `opening_book_lines.parquet` | 48,000 lines (50 per position): moves (`moves_uci`, `moves_san`), final position (`end_fen`), evaluations before and after (`cp_root`, `cp_end`, and the deeper `*_check` searches), each side's loss in centipawns, the shift in expected score (`shift_expected_check`), and build metadata (engine, binary hash, run) |
| `opening_book_summary.csv` | Per position: lines, distinct lines, first moves, mean shift and loss per side |
| `opening_book.pgn` | The lines as games, the format fastchess reads |
| `opening_book.epd` | Final position of each line |

## selfplay/tables/ (Part II)

Produced by `code/selfplay.py collect` from the fastchess PGNs (not included;
1.3 GB).

**`selfplay_games.parquet`**: one row per game (960,000).

| Column | Meaning |
| --- | --- |
| `engine`, `engine_id_name` | Stockfish or Caissa (each engine played a copy of itself) |
| `position_id`, `line_index`, `round` | Starting position, opening-book line (0-49) and match round |
| `line_matched_by_round` | The game's opening moves matched the book line for its round |
| `white_player` | Which copy (`<engine>_a` or `_b`) had White |
| `result`, `score_white` | Result and White's score |
| `termination`, `end_reason` | How the game ended (checkmate, adjudication, resignation, ...) |
| `time_overrun_ms` | Largest clock overrun (allowed up to 100 ms) |
| `plies`, `book_plies` | Game length and plies from the book (6) |
| `first_eval_white_cp` | First engine evaluation after the book |
| `mean_depth_*`, `mean_nodes_*`, `time_used_*` | Search statistics per side |
| `duration`, `start_time`, `time_control` | Timing; 10+0.1 throughout |

**`selfplay_lines.parquet`** and **`selfplay_positions.parquet`**: the same
games summed per book line and per position: `games`, `white_wins`, `draws`,
`black_wins`, `expected_score`, `score_sd`, `se`, `elo`, `draw_rate`,
`mean_plies`.

## lichess960/ (Part III)

Rated Chess960 games from the [Lichess open database](https://database.lichess.org/)
(CC0), October 2024 to September 2026, one file per month
(`lichess960_YYYY-MM.parquet`, 5.6 million games before the inclusion
criteria). Produced by `code/lichess960.py fetch`, which keeps rated Chess960
games with a known result and an estimated length of 180 s or more.

| Column | Meaning |
| --- | --- |
| `month`, `utc_datetime` | Month of the export, start time |
| `game_id` | Salted hash of the Lichess game ID (unique key) |
| `event`, `tournament` | Event header; whether the game was in an arena or Swiss tournament |
| `white_id`, `black_id` | Salted hashes of the usernames (stable across months) |
| `white_elo`, `black_elo` | Lichess Chess960 ratings (Glicko-2) before the game |
| `white_rating_diff`, `black_rating_diff` | Rating change from the game (used to exclude provisional ratings) |
| `white_title`, `black_title` | Lichess title (`BOT` accounts are excluded in the analysis) |
| `result`, `score_white` | Result and White's score |
| `time_control`, `base_s`, `increment_s` | Time control |
| `estimated_s`, `category` | Base + 40 x increment, and Lichess's category; blitz = 300-479 s, rapid = 480 s or more in the analysis |
| `termination` | Normal, Time forfeit, Abandoned, Rules infraction |
| `fen`, `position_id` | Starting position |
| `plies` | Game length |
| `white_berserk`, `black_berserk` | Berserk used (arena games) |

The inclusion criteria (in `code/analyse_lichess960.py`) are applied by
`reproduce.py`; the filtered sets are written to `results/lichess960/prepared/`
(not committed).

## masters960/ (Part III)

**`masters960_games.csv`**: 2,240 games from master-level Chess960 events
(2018-2026), produced by `code/masters960.py parse` from the PGN files listed
in `config/masters960_sources.csv` (The Week in Chess and Lichess
broadcasts).

| Column | Meaning |
| --- | --- |
| `source`, `event`, `year`, `stage`, `pgn_event`, `site`, `date`, `round` | Event details |
| `white`, `black`, `white_title`, `black_title` | Players (public tournament records) |
| `white_elo`, `black_elo` | FIDE ratings from the PGN (a few missing ones are filled in the analysis from the same player's other games in the event) |
| `result`, `score_white` | Result and White's score |
| `variant`, `variant_960` | Variant header, and whether it names Chess960 |
| `fen`, `position_id` | Starting position |
| `plies` | Game length |
| `first_clock_white`, `first_clock_black` | First clock readings, used to detect Armageddon games (unequal time) |
| `base_s`, `tc_class` | Base time and class: blitz < 8 min, rapid 8 to < 60 min, classical 60 min or more |
| `armageddon` | Flagged Armageddon game (excluded) |

## chesscom960/ (Part III)

**`freestyle_friday_tournaments.csv`**: the 80 Chess.com Freestyle Friday
tournaments analysed (January 2025 to October 2026), by tournament ID.

The game table itself, `chesscom960_games.parquet`, is **not distributed**:
Chess.com retains the rights to data from its API, and its terms discourage
republishing other players' games. Rebuild it with:

```bash
python code/chesscom960.py fetch
python code/chesscom960.py parse
```

It has one row per game: `tournament_id`, `round`, `date`, `white_id` and
`black_id` (salted hashes), `white_elo` and `black_elo` (Chess.com Chess960
ratings), `result`, `score_white`, `termination`, `time_control`, `base_s`,
`increment_s`, `variant`, `fen`, `position_id`, `plies`, `first_clock_white`,
`first_clock_black`. All results derived from it are in `results/chesscom960/`.
