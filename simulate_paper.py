"""
Monte Carlo simulation for "Measurement Is Not Passive" (Gieng, 2026).

Three regimes mapped to identification scenarios in Section 4:
    Regime A — ignorable response, passive measurement (Theorem 1 holds)
    Regime B — non-ignorable response (latent-outcome-dependent), passive measurement
    Regime C — ignorable response, active perturbation (h(0) != h(1))

Estimators:
    1. Naive difference-in-means on responding units (parametric SE)
    2. IPW using propensity model on observed covariates only, with sandwich
       (M-estimator) standard errors per Lunceford & Davidian (2004)
    3. Sensitivity-bounded IPW reporting [tau_hat - delta, tau_hat + delta]

Sandwich SEs avoid bootstrap, making the simulation tractable. The reported
sandwich SE uses the influence-function form for the IPW estimating equation
and does NOT include the first-stage propensity correction term. This
conservative variant is used for simplicity; including the first-stage
correction would tighten estimated standard errors modestly without altering
the point estimates or qualitative conclusions of any regime.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

# -------- Configuration ---------------------------------------------------

TRUE_TAU = 0.5
N_PER_ARM = 10_000
N_REPS = 1_000
ALPHA = 7.0
SIGMA_EPS = 1.5
SEED = 20260427

N_COVS = 3
GAMMA = np.array([0.6, -0.4, 0.3])

A_BETA0 = -0.2
A_BETA_X = np.array([0.5, -0.3, 0.4])

B_BETA0 = -0.5
B_BETA_Z = 0.6
B_BETA_Y = 0.25

DELTAS = [0.25, 0.50]

rng_master = np.random.default_rng(SEED)


# -------- DGP -------------------------------------------------------------

def generate_arrays(n_per_arm: int, regime: str, delta: float, rng: np.random.Generator):
    n = 2 * n_per_arm
    Z = np.empty(n, dtype=np.int8)
    Z[:n_per_arm] = 1
    Z[n_per_arm:] = 0
    rng.shuffle(Z)

    X = rng.standard_normal((n, N_COVS))
    eps = rng.normal(0.0, SIGMA_EPS, size=n)
    Y_star = ALPHA + TRUE_TAU * Z + X @ GAMMA + eps

    if regime == "A" or regime == "C":
        logits = A_BETA0 + X @ A_BETA_X
    elif regime == "B":
        logits = B_BETA0 + B_BETA_Z * Z + B_BETA_Y * Y_star
    else:
        raise ValueError(regime)
    p_response = expit(logits)
    S = (rng.uniform(size=n) < p_response).astype(np.int8)

    if regime == "C":
        Y_obs = Y_star + delta * Z
    else:
        Y_obs = Y_star
    Y = np.where(S == 1, Y_obs, np.nan)
    return Z, X, Y_star, S, Y


# -------- Logistic regression --------------------------------------------

def _fit_logit(X_design: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-7):
    beta = np.zeros(X_design.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X_design @ beta, -30.0, 30.0)
        p = expit(eta)
        W = np.maximum(p * (1.0 - p), 1e-10)
        grad = X_design.T @ (y - p)
        H = -(X_design.T * W) @ X_design
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            return beta_new
        beta = beta_new
    return beta


# -------- Estimators with sandwich SEs ------------------------------------

def naive_point_se(Z: np.ndarray, S: np.ndarray, Y: np.ndarray):
    resp = S == 1
    treat = (Z == 1) & resp
    ctrl = (Z == 0) & resp
    if treat.sum() < 2 or ctrl.sum() < 2:
        return np.nan, np.nan
    yt = Y[treat]
    yc = Y[ctrl]
    diff = yt.mean() - yc.mean()
    se = np.sqrt(yt.var(ddof=1) / treat.sum() + yc.var(ddof=1) / ctrl.sum())
    return diff, se


def ipw_point_sandwich_se(Z: np.ndarray, X: np.ndarray, S: np.ndarray, Y: np.ndarray):
    """IPW point estimate with influence-function sandwich SE.

    The estimator uses Horvitz-Thompson within-arm weighted means with a
    logistic response propensity model estimated from observed covariates
    (1, Z, X1, X2, X3). For Bernoulli randomized Z, the within-arm IPW
    means are:

        mu_z = (1 / N_z) sum_i 1{Z_i = z} S_i Y_i / e(z, X_i)

    The reported SE uses the influence-function form for the IPW moment
    and does NOT include the first-stage propensity correction term. This
    is the conservative variant described in the manuscript and README;
    including the first-stage correction would tighten standard errors
    modestly without altering point estimates or qualitative conclusions
    in any of the three regimes. See Lunceford & Davidian (2004) for the
    full first-stage-corrected derivation.
    """
    n = len(Z)
    # Design matrix for propensity model: [1, Z, X1, X2, X3]
    X_design = np.empty((n, 2 + X.shape[1]))
    X_design[:, 0] = 1.0
    X_design[:, 1] = Z
    X_design[:, 2:] = X
    beta = _fit_logit(X_design, S.astype(float))
    eta = np.clip(X_design @ beta, -30.0, 30.0)
    p_hat = np.clip(expit(eta), 0.01, 0.99)

    # Per-arm means
    Y_filled = np.where(S == 1, Y, 0.0)
    contrib = np.where(S == 1, Y_filled / p_hat, 0.0)
    arm1 = Z == 1
    arm0 = Z == 0
    N1 = arm1.sum()
    N0 = arm0.sum()
    mu1 = contrib[arm1].sum() / N1
    mu0 = contrib[arm0].sum() / N0
    tau = mu1 - mu0

    # Influence function for the IPW estimator
    # ifn_i = (1{Z_i=1}/pi_1) * [S_i*Y_i/e_i - mu_1] - (1{Z_i=0}/pi_0) * [S_i*Y_i/e_i - mu_0]
    # plus first-stage correction. For brevity and standard practice, we use
    # the simpler form that ignores first-stage correction; this gives
    # conservative (slightly too wide) intervals in our setting because
    # accounting for the first stage typically reduces variance.
    pi1 = N1 / n
    pi0 = N0 / n
    ifn = (arm1.astype(float) / pi1) * (contrib - mu1) - (arm0.astype(float) / pi0) * (contrib - mu0)
    var_tau = np.var(ifn, ddof=1) / n
    se = float(np.sqrt(var_tau))
    return float(tau), se


# -------- Driver ----------------------------------------------------------

def run_regime(regime: str, delta: float, n_reps: int, n_per_arm: int):
    rng = np.random.default_rng(rng_master.integers(0, 2**31))
    naive_pt = np.empty(n_reps)
    naive_cov = np.empty(n_reps)
    ipw_pt = np.empty(n_reps)
    ipw_cov = np.empty(n_reps)
    bound_cov = np.empty(n_reps)

    z_crit = norm.ppf(0.975)
    delta_used = delta if regime == "C" else 0.0

    t0 = time.time()
    for r in range(n_reps):
        Z, X, _Ystar, S, Y = generate_arrays(n_per_arm, regime, delta, rng)
        np_pt, np_se = naive_point_se(Z, S, Y)
        ip_pt, ip_se = ipw_point_sandwich_se(Z, X, S, Y)
        naive_pt[r] = np_pt
        ipw_pt[r] = ip_pt

        if not np.isnan(np_pt):
            naive_cov[r] = float(np_pt - z_crit * np_se <= TRUE_TAU <= np_pt + z_crit * np_se)
        else:
            naive_cov[r] = np.nan
        if not np.isnan(ip_pt):
            ipw_cov[r] = float(ip_pt - z_crit * ip_se <= TRUE_TAU <= ip_pt + z_crit * ip_se)
            bound_cov[r] = float(ip_pt - delta_used <= TRUE_TAU <= ip_pt + delta_used)
        else:
            ipw_cov[r] = np.nan
            bound_cov[r] = np.nan

        if (r + 1) % 250 == 0:
            elapsed = time.time() - t0
            print(f"  {regime}{' δ='+str(delta) if regime=='C' else ''}: "
                  f"{r+1}/{n_reps} in {elapsed:.0f}s")

    def _summary(pts, cov):
        return {
            "bias": float(np.nanmean(pts) - TRUE_TAU),
            "mc_se": float(np.nanstd(pts, ddof=1)),
            "rmse": float(np.sqrt(np.nanmean((pts - TRUE_TAU) ** 2))),
            "coverage": float(np.nanmean(cov)),
        }

    return {
        "regime": regime,
        "delta": delta,
        "n_reps": n_reps,
        "naive": _summary(naive_pt, naive_cov),
        "ipw": _summary(ipw_pt, ipw_cov),
        "bound_coverage": float(np.nanmean(bound_cov)),
    }


def main():
    print(f"Configuration: tau={TRUE_TAU}, N/arm={N_PER_ARM}, reps={N_REPS}, seed={SEED}\n")

    results = []
    print("Regime A (ignorable response, passive measurement):")
    results.append(run_regime("A", 0.0, N_REPS, N_PER_ARM))
    print("\nRegime B (non-ignorable response, passive measurement):")
    results.append(run_regime("B", 0.0, N_REPS, N_PER_ARM))
    for d in DELTAS:
        print(f"\nRegime C (active perturbation, delta={d}):")
        results.append(run_regime("C", d, N_REPS, N_PER_ARM))

    rows = []
    for r in results:
        label = r["regime"]
        if r["regime"] == "C":
            label += f" (δ={r['delta']})"
        rows.append({
            "Regime": label,
            "Naive bias": f"{r['naive']['bias']:+.4f}",
            "Naive RMSE": f"{r['naive']['rmse']:.4f}",
            "Naive cov": f"{r['naive']['coverage']:.3f}",
            "IPW bias": f"{r['ipw']['bias']:+.4f}",
            "IPW RMSE": f"{r['ipw']['rmse']:.4f}",
            "IPW cov": f"{r['ipw']['coverage']:.3f}",
            "Bound cov": f"{r['bound_coverage']:.3f}",
        })
    summary = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print(f"SUMMARY (true tau = {TRUE_TAU}; coverage of TRUE_TAU)")
    print("=" * 100)
    print(summary.to_string(index=False))

    summary.to_csv("simulation_results.csv", index=False)
    with open("simulation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nWrote simulation_results.csv and simulation_results.json")


if __name__ == "__main__":
    main()
