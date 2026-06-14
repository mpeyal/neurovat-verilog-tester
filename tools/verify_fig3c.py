#!/usr/bin/env python
"""Fig. 3c LTP/LTD with the new generator defaults (spaced pulses, paper config).
Shows BOTH views are clean: the transient G(t) (no rail saturation) and the
Analysis-tab retained ramp G_nv vs pulse # (the paper's Fig. 3c triangle).
Saves results/fig3c_check.png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ecfet import Waveform, EcfetV2, V2Params, simulate
from vatester import analysis

# new LTP/LTD generator defaults
AMP, N, PER, W = 170e-12, 300, 2.0, 0.01
pul, t = [], 0.5
for _ in range(N):
    pul.append((t, W, +AMP)); t += PER          # LTP (+ potentiates v2)
for _ in range(N):
    pul.append((t, W, -AMP)); t += PER          # LTD
r = simulate(EcfetV2(V2Params()), Waveform(pul), t_stop=t + PER)

gmax = V2Params().Gmax * 1e6
Gt = r.G * 1e6
data = analysis.per_pulse_samples([r], "G", N)[0]
Gnv = np.array(data["vals"])
clip = int(np.sum((Gt > 0.997 * gmax) | (Gt < 1.003 * V2Params().Gmin * 1e6)))
print(f"transient G: {Gt.min():.0f}..{Gt.max():.0f} uS, clipped {clip} pts")
print(f"Analysis G_nv: {Gnv.min():.0f}..{Gnv.max():.0f} uS over {N} states/branch")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
ax[0].plot(r.t, Gt, color="#1f77b4", lw=0.8)
ax[0].axhline(gmax, color="r", ls="--", lw=0.7, alpha=0.6)
ax[0].set_title("a) transient G(t) - no rail saturation", fontsize=11)
ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("G ($\\mu$S)")
ax[0].grid(True, alpha=0.3)
ax[1].plot(np.arange(1, len(Gnv) + 1), Gnv, color="#d62728", lw=1.2)
ax[1].set_title("b) Analysis tab: retained G_nv vs pulse # (Fig. 3c)", fontsize=11)
ax[1].set_xlabel("pulse #"); ax[1].set_ylabel("G ($\\mu$S)")
ax[1].grid(True, alpha=0.3)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "fig3c_check.png")
fig.savefig(out, dpi=130)
plt.close(fig)
print("plot ->", out)
