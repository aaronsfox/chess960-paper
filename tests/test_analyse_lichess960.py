"""Tests for the Lichess Chess960 analysis: simulated games with known
position effects and rating scale."""

import numpy as np
import pandas as pd
import pytest
from scipy import special

import analyse_lichess960 as A

K = A.K


def simulate(n=400_000, tau=6.0, gamma=0.5, adv=8.0, beta=0.9, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(18, 12, 960)                      # engine self-play Delta_Elo
    true = adv + gamma * (x - x.mean()) + rng.normal(0, tau, 960)
    pos = rng.integers(0, 960, n)
    mean = rng.normal(1700, 250, n).clip(1200, 2600)
    gap = rng.normal(0, 150, n)
    we, be = (mean + gap / 2).round(), (mean - gap / 2).round()
    eta = (true[pos] + 0 * x[pos]) / K + beta * (we - be) * np.log(10) / 400
    p = special.expit(eta)
    u = rng.random(n)
    draw = rng.random(n) < 0.045
    score = np.where(draw, 0.5, (u < (p - 0.0225) / 0.955).astype(float))
    d = pd.DataFrame({"position_id": pos, "white_elo": we, "black_elo": be, "score_white": score,
                      "white_id": rng.integers(0, 50_000, n), "black_id": rng.integers(0, 50_000, n),
                      "time_forfeit": False, "tournament": False})
    eng = pd.DataFrame({"engine": "stockfish", "position_id": range(960), "delta_elo_median": x,
                        "delta_elo_sd": 0.0})
    eng = pd.concat([eng, eng.assign(engine="caissa")])
    eng = eng.pivot(index="position_id", columns="engine", values=["delta_elo_median", "delta_elo_sd"])
    return d, eng, true


@pytest.fixture(scope="module")
def sim():
    return simulate()


def test_rating_model_recovers_scale_and_advantage(sim):
    d, _, _ = sim
    rm = A.rating_model(d)
    assert rm["beta_lo"] < 0.9 < rm["beta_hi"]
    assert abs(rm["advantage_elo"] - 8.0) < 3


def test_per_position_estimates(sim):
    d, _, true = sim
    rm = A.rating_model(d)
    pos = A.stage1(d, rm["_b"][1])
    assert np.median(pos.se_ratio_pair_cluster) == pytest.approx(1, abs=0.05)
    # per-position estimates scatter around the true values with their SEs
    z = (pos.human_elo - true) / pos.human_se
    assert abs(np.median(z)) < 0.3 and 0.8 < z.std() < 1.25
    assert pos.decisive_adj.between(0, 1).all()


def test_subgroups_and_trend(sim):
    d, _, _ = sim
    grp, band_pos = A.subgroups(d, "lichess_blitz")
    assert grp.group.tolist() == ["all"] + A.BAND_LABELS
    assert set(band_pos.group) <= set(A.BAND_LABELS)
    t = A.mean_rating_trend(d)            # no trend simulated
    assert t["delta_lo"] < 0 < t["delta_hi"]


def test_filters_in_order():
    g = pd.DataFrame({
        "estimated_s": [300, 300, 300, 300, 300, 300, 300, 200, 600],
        "termination": ["Normal", "Abandoned", "Normal", "Normal", "Normal", "Normal", "Normal", "Normal", "Normal"],
        "plies": [40, 40, 4, 40, 40, 40, 40, 40, 40],
        "white_berserk": [False, False, False, True, False, False, False, False, False],
        "black_berserk": False,
        "white_elo": [1600, 1600, 1600, 1600, np.nan, 1600, 1100, 1600, 1600],
        "black_elo": [1600, 1600, 1600, 1600, 1600, 1600, 1100, 1600, 1600],
        "white_rating_diff": [5, 5, 5, 5, 5, 120, 5, 5, 5],
        "black_rating_diff": [-5] * 9,
        "white_title": "", "black_title": "",
    })
    kept, flow = A.apply_filters(g, 300, 480)
    assert list(kept.index) == [0]
    assert [n for _, n in flow] == [9, 7, 7, 6, 5, 4, 3, 2, 1]
