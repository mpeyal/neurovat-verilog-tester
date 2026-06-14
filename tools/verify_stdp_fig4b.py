#!/usr/bin/env python
"""Does the model's STDP follow the paper Fig. 4b equation?
   dG = c1 e^-dt/tau1 + c2 e^-dt/tau2 + c3 e^-dt/tau3, tau = 22ms/315ms/19s.

Runs the model's anti-symmetric STDP window, fits the POSITIVE side to that
equation (fixed taus), and overlays the two. Saves results/stdp_fig4b.png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ecfet import EcfetV2, V2Params
from vatester import analysis

TAU = (0.022, 0.315, 19.0)
pos = [10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1800]
dts = sorted([-d * 1e-3 for d in pos] + [d * 1e-3 for d in pos])

ys = np.array(list(analysis.stdp_sweep(
    [EcfetV2(V2Params())], -20e-12, +20e-12, 5e-3, dts, 4.0).values())[0])

pd = np.array([d for d in dts if d > 0])
py = np.array([ys[i] for i, d in enumerate(dts) if d > 0])
M = np.column_stack([np.exp(-pd / t) for t in TAU])
c = np.linalg.lstsq(M, py, rcond=None)[0]
fitfn = lambda t: sum(c[k] * np.exp(-t / TAU[k]) for k in range(3))
r2 = 1 - np.sum((py - M @ c) ** 2) / np.sum((py - py.mean()) ** 2)
print("paper-eq fit: c1=%.2f c2=%.2f c3=%.2f uS  R^2=%.4f" % (c[0], c[1], c[2], r2))
print("peak(10ms)=%.2f  tail(1800ms)=%.2f uS" % (py[0], py[-1]))

tt = np.linspace(1e-3, 1.8, 400)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
# left: positive side vs the paper equation
ax[0].plot(pd * 1e3, py, "o", color="#d62728", label="model")
ax[0].plot(tt * 1e3, fitfn(tt), "--k",
           label="paper eq: c1 e$^{-\\Delta t/\\tau_1}$+c2 e$^{-\\Delta t/\\tau_2}$+c3 e$^{-\\Delta t/\\tau_3}$")
ax[0].set_title("a) +side vs paper Fig.4b   (R$^2$=%.4f)" % r2, fontsize=11)
ax[0].set_xlabel("$\\Delta$t (ms)"); ax[0].set_ylabel("$\\Delta$G ($\\mu$S)")
ax[0].text(500, 0.75 * py[0],
           "$\\tau_1$=22 ms\n$\\tau_2$=315 ms\n$\\tau_3$=19 s\nc1=%.1f c2=%.1f c3=%.1f $\\mu$S"
           % (c[0], c[1], c[2]), fontsize=9)
ax[0].grid(True, alpha=0.3); ax[0].legend(fontsize=8)
# right: full anti-symmetric window
ax[1].plot(np.array(dts) * 1e3, ys, "o-", color="#1f77b4", ms=3)
ax[1].axhline(0, color="0.5", lw=0.8)
ax[1].set_title("b) full anti-symmetric window (+side = eq, -side = mirror)", fontsize=11)
ax[1].set_xlabel("$\\Delta$t = t$_{post}$-t$_{pre}$ (ms)"); ax[1].set_ylabel("$\\Delta$G ($\\mu$S)")
ax[1].grid(True, alpha=0.3)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "stdp_fig4b.png")
fig.savefig(out, dpi=130)
plt.close(fig)
print("plot ->", out)
