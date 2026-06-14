#!/usr/bin/env python
"""Single +/-50 pA / 10 ms pulse from 500 ohm, symmetric write-then-relax:
  +50 pA (potentiate): 500 -> 470 (dR=-30) -> 490 (dR'=-10), stable
  -50 pA (depress):    500 -> 530 (dR=+30) -> 510 (dR'=+10), stable
Saves results/spike_500.png.  Uses the committed v2 DEFAULTS (the 500 ohm demo).

Symmetry note: R = 1/G is curved, so a fixed charge step (same dG) would give
an asymmetric dR at a 6%-of-R swing (depression overshooting to ~534).  The
model uses a direction-specific window (nu_d) and volatile fraction (kappa_d)
so both directions land on +/-30 (dip) and +/-10 (settle).  c3=0 drops the
slow 19 s pool so each settles within ~3 s instead of lingering.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ecfet import Waveform, EcfetV2, V2Params, simulate

T0, WIDTH, T_STOP = 1.0, 0.01, 10.0


def run(amp):
    r = simulate(EcfetV2(V2Params.demo_500()), Waveform([(T0, WIDTH, amp)]), t_stop=T_STOP)
    R0 = float(np.interp(T0 - 0.01, r.t, r.R))
    peak = float(r.R.min() if amp > 0 else r.R.max())
    return r, R0, peak, float(r.R[-1])


fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
for ax, amp, title in ((axes[0], +50e-12, "a) +50 pA potentiation"),
                       (axes[1], -50e-12, "b) -50 pA depression")):
    r, R0, peak, settle = run(amp)
    print(f"{title}: R0={R0:.1f}  peak={peak:.1f} (dR={peak-R0:+.1f})  "
          f"settle={settle:.1f} (dR'={settle-R0:+.1f})")
    ax.plot(r.t, r.R, color="#1f77b4", lw=1.8)
    ax.axhline(R0, color="r", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(settle, color="r", ls="--", lw=0.8, alpha=0.6)
    ax.axvspan(T0, T0 + WIDTH, color="0.85")
    ax.annotate(f"dip {peak:.0f} Ω (ΔR={peak-R0:+.0f})",
                (T0 + 0.15, peak), color="r", fontsize=9,
                va="bottom" if amp > 0 else "top")
    ax.annotate(f"settles {settle:.0f} Ω (ΔR'={settle-R0:+.0f})",
                (T0 + 3, settle), color="r", fontsize=9,
                va="top" if amp > 0 else "bottom")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("R_mem (Ω)")
    ax.grid(True, alpha=0.3)
fig.suptitle("v2 @ 500 Ω: symmetric +/-50 pA write-then-relax", fontsize=12)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "spike_500.png")
fig.savefig(out, dpi=130)
plt.close(fig)
print("plot ->", out)
