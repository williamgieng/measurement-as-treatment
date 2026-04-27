[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19827755.svg)](https://doi.org/10.5281/zenodo.19827755)

# Measurement Is Not Passive — Replication Code

This repository contains the simulation code and replication files for:

**Gieng, W. (2026). Measurement Is Not Passive: Identification of Treatment Effects When Outcomes Are Elicited Rather Than Observed.** Manuscript submitted for publication.

## Contents

| File | Purpose |
|------|---------|
| `simulate_paper.py` | Main Monte Carlo for Table 3 (three identification regimes) |
| `robustness_A.py` | Sample size sweep at N ∈ {2,000, 10,000, 50,000} (Appendix B.1) |
| `robustness_B.py` | β_Y sweep at β_Y ∈ {0.10, 0.25, 0.50} (Appendix B.2) |
| `make_figure.py` | Generates the simulation figure |
| `requirements.txt` | Python dependencies |
| `simulation_results.csv` / `.json` | Output from the main Monte Carlo |
| `robustness_A_sample_size.csv` / `.json` | Output from the sample size sweep |
| `robustness_B_betaY.csv` / `.json` | Output from the β_Y sweep |
| `simulation_figure.png` | Figure used in the paper |

## Reproducing the results

Requires Python 3.9 or later.

```bash
pip install -r requirements.txt
python simulate_paper.py       # Table 3 in Section 6
python robustness_A.py          # Table 4 in Appendix B.1
python robustness_B.py          # Table 5 in Appendix B.2
python make_figure.py           # simulation_figure.png
```

All scripts use master seed `20260427` and are deterministic. Output files in this repository should match newly generated output to the displayed precision. Scripts read and write files in the current working directory; run them from the repository root.

## Simulation design

- 1,000 Monte Carlo replications per condition
- N = 10,000 units per arm (main run); varied in robustness checks
- True average treatment effect τ = 0.5
- Three pre-treatment Gaussian covariates X_1, X_2, X_3
- Outcome model: Y* = α + τ·Z + γ'X + ε with α = 7, γ = (0.6, -0.4, 0.3), ε ~ N(0, 1.5²)

The three identification regimes:

- **Regime A** — ignorable response, passive measurement; (A0)–(A4) all hold
- **Regime B** — non-ignorable response (response depends on the latent outcome Y*); (A2) fails
- **Regime C** — active perturbation with |h(1) − h(0)| = δ for δ ∈ {0.25, 0.50}; (A3) fails

See Section 6 of the paper for the full simulation design and regime definitions, and Appendix B for the robustness checks.

## A note on standard errors

The IPW estimator in `simulate_paper.py` uses an influence-function sandwich standard error that omits the first-stage propensity correction term. This is empirically conservative in the simulation design considered here but is not guaranteed to be conservative in general; first-stage propensity estimation can either increase or decrease the asymptotic variance of IPW depending on setup. Applied users seeking nominal coverage should use a fully corrected sandwich (see Lunceford and Davidian 2004) or the bootstrap.

## License

MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the paper:

```
Gieng, W. (2026). Measurement Is Not Passive: Identification of Treatment Effects
When Outcomes Are Elicited Rather Than Observed. [Journal name to be added on
acceptance].
```

## Contact

William Gieng — william.gieng@gmail.com
