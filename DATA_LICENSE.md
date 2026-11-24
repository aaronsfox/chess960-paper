# Data licences

The code in this repository is under the MIT licence (`LICENSE`). The data
are covered as follows.

| Data | Licence and terms |
| --- | --- |
| `data/engine_evaluation/`, `data/opening_book/`, `data/selfplay/`, and everything in `results/` | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Please cite the paper and this archive. |
| `data/lichess960/` | Derived from the [Lichess open database](https://database.lichess.org/), released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/). Player names and game IDs are replaced by salted hashes. |
| `data/masters960/` | Compiled from public tournament records published by [The Week in Chess](https://theweekinchess.com/) and [Lichess broadcasts](https://lichess.org/broadcast) (sources in `config/masters960_sources.csv`). Game scores are factual records; please credit the original publishers. |
| `data/chesscom960/` | Only the list of tournament IDs is included. Chess.com retains the rights to data from its Published-Data API, so the game table is not redistributed; `code/chesscom960.py` rebuilds it from the API, subject to Chess.com's terms. |

The chess engines and fastchess are not distributed here. Their build scripts
download the published releases, which remain under their own licences.
