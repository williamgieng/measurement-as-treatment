"""Robustness B (controlled response rate): vary latent-response dependence beta_Y
in Regime B, with the intercept beta_0 adjusted to keep marginal response rate
approximately constant across beta_Y values.

This isolates the effect of non-ignorability strength from the saturation
that occurs when high beta_Y combined with the latent outcome's mean pushes
response probabilities toward 1.
"""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
from scipy.special import expit

import simulate_paper as base

N_REPS = 1_000
N_PER_ARM = 10_000
BETA_YS = [0.10, 0.25, 0.50]
TARGET_RESPONSE_RATE = 0.45
SEED = 20260427

rng_master = np.random.default_rng(SEED)


def _calibrate_intercept(beta_Y: float, target_rate: float, n_calib: int = 200_000):
    rng = np.random.default_rng(99)
    Z = rng.binomial(1, 0.5, size=n_calib)
    X = rng.standard_normal((n_calib, base.N_COVS))
    eps = rng.normal(0.0, base.SIGMA_EPS, size=n_calib)
    Y_star = base.ALPHA + base.TRUE_TAU * Z + X @ base.GAMMA + eps

    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        logits = mid + base.B_BETA_Z * Z + beta_Y * Y_star
        rate = float(expit(logits).mean())
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
        if abs(rate - target_rate) < 1e-4:
            return mid
    return 0.5 * (lo + hi)


def generate_with_betaY(n_per_arm, beta_Y, beta_0, rng):
    n = 2 * n_per_arm
    Z = np.empty(n, dtype=np.int8)
    Z[:n_per_arm] = 1
    Z[n_per_arm:] = 0
    rng.shuffle(Z)
    X = rng.standard_normal((n, base.N_COVS))
    eps = rng.normal(0.0, base.SIGMA_EPS, size=n)
    Y_star = base.ALPHA + base.TRUE_TAU * Z + X @ base.GAMMA + eps
    logits = beta_0 + base.B_BETA_Z * Z + beta_Y * Y_star
    p_response = expit(logits)
    S = (rng.uniform(size=n) < p_response).astype(np.int8)
    Y = np.where(S == 1, Y_star, np.nan)
    return Z, X, Y_star, S, Y


def run_for_betaY(beta_Y, beta_0, n_reps):
    rng = np.random.default_rng(rng_master.integers(0, 2**31))
    naive_pt = np.empty(n_reps)
    ipw_pt = np.empty(n_reps)
    response_rates = np.empty(n_reps)

    for r in range(n_reps):
        Z, X, _Y, S, Y = generate_with_betaY(N_PER_ARM, beta_Y, beta_0, rng)
        np_pt, _ = base.naive_point_se(Z, S, Y)
        ip_pt, _ = base.ipw_point_sandwich_se(Z, X, S, Y)
        naive_pt[r] = np_pt
        ipw_pt[r] = ip_pt
        response_rates[r] = float(S.mean())

    naive_bias = float(np.nanmean(naive_pt) - base.TRUE_TAU)
    ipw_bias = float(np.nanmean(ipw_pt) - base.TRUE_TAU)
    bias_reduction_pct = (1 - abs(ipw_bias) / abs(naive_bias)) * 100 if naive_bias != 0 else 0.0
    return {
        "beta_Y": beta_Y,
        "calibrated_beta_0": beta_0,
        "mean_response_rate": float(np.mean(response_rates)),
        "naive_bias": naive_bias,
        "ipw_bias": ipw_bias,
        "bias_reduction_pct": float(bias_reduction_pct),
    }


def main():
    print(f"Robustness B (response rate held ~ {TARGET_RESPONSE_RATE}):\n")
    results = []
    for b in BETA_YS:
        beta_0 = _calibrate_intercept(b, TARGET_RESPONSE_RATE)
        t0 = time.time()
        r = run_for_betaY(b, beta_0, N_REPS)
        elapsed = time.time() - t0
        print(f"  beta_Y={b:.2f}, beta_0={beta_0:+.3f} (resp={r['mean_response_rate']:.3f}): "
              f"naive={r['naive_bias']:+.4f}, ipw={r['ipw_bias']:+.4f}, "
              f"reduction={r['bias_reduction_pct']:.1f}%  ({elapsed:.0f}s)")
        results.append(r)

    df = pd.DataFrame(results)
    print()
    print(df.to_string(index=False))
    df.to_csv("robustness_B_betaY.csv", index=False)
    with open("robustness_B_betaY.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nWrote robustness_B_betaY.csv and .json")


if __name__ == "__main__":
    main()
