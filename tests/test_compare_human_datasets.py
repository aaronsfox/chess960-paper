"""Tests for the cross-dataset comparisons."""

import numpy as np

import compare_human_datasets as C


def test_cochran_q_null_and_spread():
    rng = np.random.default_rng(0)
    se = np.full(500, 10.0)
    q0 = C.cochran_q(5 + rng.normal(0, 10, 500), se)
    assert 0.85 < q0["Q_over_df"] < 1.15 and q0["p"] > 0.01
    q1 = C.cochran_q(5 + rng.normal(0, 10, 500) + rng.normal(0, 10, 500), se)
    assert q1["Q_over_df"] > 1.6 and q1["p"] < 1e-6


def test_full_transfer_benchmark_falls_with_noise():
    rng = np.random.default_rng(1)
    x = rng.normal(18, 12, 400)
    precise = C.rho_full_transfer(x, np.full(400, 2.0), rng)
    noisy = C.rho_full_transfer(x, np.full(400, 40.0), rng)
    assert precise > 0.95 and noisy < 0.4


def test_tau_zero_when_homogeneous_and_recovered_when_not():
    rng = np.random.default_rng(2)
    se = rng.uniform(8, 12, 960)
    t0 = C.tau_dl(10 + rng.normal(0, se), se)
    assert t0["tau_lo"] == 0.0 and t0["tau"] < 3 and t0["tau_hi"] < 5
    t1 = C.tau_dl(10 + rng.normal(0, 8, 960) + rng.normal(0, se), se)
    assert t1["tau_lo"] < 8 < t1["tau_hi"] and abs(t1["tau"] - 8) < 2


def test_tau_q_profile_brackets_dl_estimate():
    rng = np.random.default_rng(3)
    se = np.full(50, 5.0)
    t = C.tau_dl(rng.normal(0, 7, 50), se)
    assert t["tau_lo"] <= t["tau"] <= t["tau_hi"]


def test_attenuation_slope_recovers_transfer_fraction():
    rng = np.random.default_rng(4)
    x = rng.normal(18, 14, 960)
    se = np.full(960, 3.0)
    full = C.attenuation_slope(10 + x + rng.normal(0, se), se, x)
    half = C.attenuation_slope(10 + 0.5 * x + rng.normal(0, se), se, x)
    none = C.attenuation_slope(10 + rng.normal(0, se), se, x)
    assert full["slope_lo"] < 1 < full["slope_hi"]
    assert half["slope_lo"] < 0.5 < half["slope_hi"]
    assert none["slope_lo"] < 0 < none["slope_hi"]


def test_engine_reliability():
    import pandas as pd
    rng = np.random.default_rng(5)
    true = rng.normal(0, 14, 960)
    sp = pd.DataFrame({"engine": "stockfish", "delta_elo_median": true + rng.normal(0, 6, 960),
                       "delta_elo_sd": 6.0})
    r = C.engine_reliability(sp).iloc[0]
    assert abs(r.reliability - 14 ** 2 / (14 ** 2 + 6 ** 2)) < 0.03
    assert abs(r.sd_between_positions - 14) < 1
