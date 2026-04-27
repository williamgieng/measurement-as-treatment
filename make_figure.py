"""Generate Figure 1 for the paper: bias by regime and estimator."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("simulation_results.json") as f:
    results = json.load(f)

fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)

labels = []
naive_biases = []
ipw_biases = []
for r in results:
    if r["regime"] == "C":
        labels.append(f"C (δ={r['delta']})")
    else:
        labels.append(f"Regime {r['regime']}")
    naive_biases.append(r["naive"]["bias"])
    ipw_biases.append(r["ipw"]["bias"])

x = np.arange(len(labels))
width = 0.36

b1 = ax.bar(x - width/2, naive_biases, width, label="Naive", color="#888888", edgecolor="black")
b2 = ax.bar(x + width/2, ipw_biases, width, label="IPW (covariate-only)", color="#4477AA", edgecolor="black")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Bias (estimator − true τ)", fontsize=11)
ax.set_title("Estimator bias by identification regime", fontsize=12)
ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Annotate bars with bias values
for bars, vals in [(b1, naive_biases), (b2, ipw_biases)]:
    for bar, v in zip(bars, vals):
        h = bar.get_height()
        offset = 0.015 if h >= 0 else -0.025
        ax.annotate(f"{v:+.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -10),
                    textcoords="offset points", ha="center", fontsize=8.5)

ax.set_ylim(-0.18, 0.65)
fig.tight_layout()
fig.savefig("simulation_figure.png", dpi=200, bbox_inches="tight")
print("Wrote simulation_figure.png")
