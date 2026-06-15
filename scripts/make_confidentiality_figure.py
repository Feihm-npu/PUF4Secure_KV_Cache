"""Flagship figure: measured IND-MIG advantage vs the proven Gaussian-mechanism
bound, with the fp32 de-mask error and the feasible operating window.

Reads confidentiality_analysis_<tag>.json (run_confidentiality_analysis.py) and
writes paper_latex/figs/confidentiality_sweep.pdf.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tag = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
d = json.load(open(f"experiments/runs/confidentiality_analysis_{tag}.json"))
sig = [s for s in d["sigmas"] if s > 0]
g = lambda k: [d["by_sigma"][str(s)][k] for s in sig]

adv_emp = g("two_cand_adv")     # optimal (basis-equipped) attacker, empirical
bound = g("analytic_bound")     # Theorem B Gaussian-mechanism bound
tv = g("exact_TV")              # exact optimal advantage 2*Phi(D/2s)-1
err = g("fp32_roundtrip_relerr")
N = d["by_sigma"][str(sig[0])]["N"]

fig, ax = plt.subplots(figsize=(5.2, 3.3))
ax.set_xscale("log", base=2)
ax.plot(sig, bound, "k--", lw=1.6, label=r"Thm B bound $\Delta/(\sigma\sqrt{2\pi})$")
ax.plot(sig, tv, color="0.55", lw=1.2, label=r"exact optimal $2\Phi(\Delta/2\sigma){-}1$")
ax.plot(sig, adv_emp, "o-", color="C0", ms=5, lw=1.4, label="measured optimal-attacker adv.")
ax.axhline(1.0 / N, color="C3", ls=":", lw=1.2, label=f"chance $1/{N}$")
ax.set_xlabel(r"mask scale $\sigma$")
ax.set_ylabel("IND-MIG advantage")
ax.set_ylim(-0.02, 0.5)

ax2 = ax.twinx()
ax2.plot(sig, err, "s-", color="C2", ms=4, lw=1.2, alpha=0.8, label="fp32 de-mask error")
ax2.set_yscale("log")
ax2.set_ylabel("fp32 round-trip rel. error", color="C2")
ax2.tick_params(axis="y", labelcolor="C2")

# feasible window: advantage <= 0.05 and error <= 1e-3
tau = 1e-3
win = [s for s, a, e in zip(sig, adv_emp, err) if a <= 0.05 and e <= tau]
if win:
    ax.axvspan(min(win), max(win), color="C0", alpha=0.08)
    ax.text(min(win), 0.46, " feasible window", fontsize=8, color="C0")

l1, lab1 = ax.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lab1 + lab2, fontsize=7, loc="upper right", framealpha=0.9)
fig.tight_layout()

os.makedirs("paper_latex/figs", exist_ok=True)
fig.savefig("paper_latex/figs/confidentiality_sweep.pdf", bbox_inches="tight")
print("wrote paper_latex/figs/confidentiality_sweep.pdf")
print("sigma:", sig)
print("bound:", [round(x, 3) for x in bound])
print("measured:", [round(x, 3) for x in adv_emp])
