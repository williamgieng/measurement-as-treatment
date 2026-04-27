"""Robustness A: vary sample size in Regime B (non-ignorable response).

Tests whether the partial-repair finding is consistent across sample sizes,
and confirms that high IPW coverage at N=10000 reflects conservative SEs
rather than asymptotic validity (the residual bias persists).

Outputs naive bias, IPW bias, and IPW coverage at N per arm in {2000, 10000, 50000}.
"""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

import simulate_paper as base

N_REPS = 1_000
SAMPLE_SIZES = [2_000, 10_000, 50_000]
SEED = 20260427

rng_master = np.random.default_rng(SEED)


def run_for_n(n_per_arm: int, n_reps: int):
    rng = np.random.default_rng(rng_master.integers(0, 2**31))
    naive_pt = np.empty(n_reps)
    naive_cov = np.empty(n_reps)
    ipw_pt = np.empty(n_reps)
    ipw_cov = np.empty(n_reps)
    z_crit = norm.ppf(0.975)

    for r in range(n_reps):
        Z, X, _Ystar, S, Y = base.generate_arrays(n_per_arm, "B", 0.0, rng)
        np_pt, np_se = base.naive_point_se(Z, S, Y)
        ip_pt, ip_se = base.ipw_point_sandwich_se(Z, X, S, Y)
        naive_pt[r] = np_pt
        ipw_pt[r] = ip_pt
        naive_cov[r] = float(np_pt - z_crit * np_se <= base.TRUE_TAU <= np_pt + z_crit * np_se) if not np.isnan(np_pt) else np.nan
        ipw_cov[r] = float(ip_pt - z_crit * ip_se <= base.TRUE_TAU <= ip_pt + z_crit * ip_se) if not np.isnan(ip_pt) else np.nan

    return {
        "n_per_arm": n_per_arm,
        "naive_bias": float(np.nanmean(naive_pt) - base.TRUE_TAU),
        "naive_rmse": float(np.sqrt(np.nanmean((naive_pt - base.TRUE_TAU) ** 2))),
        "naive_coverage": float(np.nanmean(naive_cov)),
        "ipw_bias": float(np.nanmean(ipw_pt) - base.TRUE_TAU),
        "ipw_rmse": float(np.sqrt(np.nanmean((ipw_pt - base.TRUE_TAU) ** 2))),
        "ipw_coverage": float(np.nanmean(ipw_cov)),
    }


def main():
    print(f"Robustness A: Regime B, vary sample size. {N_REPS} reps per cell.\n")
    results = []
    for n in SAMPLE_SIZES:
        t0 = time.time()
        r = run_for_n(n, N_REPS)
        elapsed = time.time() - t0
        print(f"  N/arm = {n:>6}: naive_bias={r['naive_bias']:+.4f}, "
              f"ipw_bias={r['ipw_bias']:+.4f}, ipw_cov={r['ipw_coverage']:.3f}  ({elapsed:.0f}s)")
        results.append(r)

    df = pd.DataFrame(results)
    print()
    print(df.to_string(index=False))
    df.to_csv("robustness_A_sample_size.csv", index=False)
    with open("robustness_A_sample_size.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nWrote robustness_A_sample_size.csv and .json")


if __name__ == "__main__":
    main()
