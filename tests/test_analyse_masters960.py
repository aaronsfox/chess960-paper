"""Tests for the master-level Chess960 analysis."""

import numpy as np
import pandas as pd

import analyse_masters960 as M
from test_masters960 import rows


def test_include_flow():
    d, flow = M.include(rows())
    assert len(d) == 1 and d.iloc[0]["round"] == "1.5"
    # game 4 is the standard start with no Variant tag: not counted as Chess960
    assert flow.games.tolist() == [4, 3, 3, 3, 3, 2, 1, 1]


def test_advantage_offset():
    rng = np.random.default_rng(0)
    n = 4000
    we = rng.normal(2600, 100, n)
    be = rng.normal(2600, 100, n)
    p = 1 / (1 + 10 ** (-(we - be + 20) / 400))
    d = pd.DataFrame({"white_elo": we, "black_elo": be, "score_white": (rng.random(n) < p).astype(float)})
    a = M.advantage(d)
    assert a["advantage_lo"] < 20 < a["advantage_hi"]


def test_round_position_ok():
    g = pd.DataFrame({"source": "s", "round": [f"1.{i}" for i in range(12)] + ["2.1", "2.2"],
                      "position_id": [5] * 11 + [7] + [9, 8]})
    ok = M.round_position_ok(g)
    assert ok.tolist() == [True] * 11 + [False] + [True, True]


def test_fill_ratings():
    g = pd.DataFrame({"event": ["E", "E", "E"], "white": ["a", "b", "a"], "black": ["b", "a", "c"],
                      "white_elo": [2500, np.nan, np.nan], "black_elo": [2400, 2500, np.nan]})
    f = M.fill_ratings(g)
    assert f.white_elo.tolist()[:2] == [2500, 2400] and f.white_elo.iloc[2] == 2500
    assert np.isnan(f.black_elo.iloc[2])
