#!/usr/bin/env python
"""Sweep ECFET v3 STDP time constants and measure the learning-window width.

window metric: on the positive (potentiation) side we evaluate dG(dt) on a fine
log grid, take the peak (dt->0), then report
  - w_1e : |dt| where |dG| falls to 1/e (37%) of peak
  - w_5  : |dt| where |dG| falls to 5%  of peak  (effective window edge)
taus = (w^2/2D, l^2/2D, tao3) * tau_scale, weights (0.39,0.44,0.17).
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecfet.model_v3 import EcfetV3, V3Params
from vatester import analysis

D = 3.6e-10
WIDTH = 2e-3            # 2 ms pulses so the dt grid can start small
TAIL = 2.0             # long settle
# fine positive dt grid 3 ms .. 600 ms
DTS = [3e-3 * (200.0) ** (k / 59.0) for k in range(60)]   # 3ms -> 600ms log


def w_of_tau(tau):  return math.sqrt(2 * D * tau)   # tau1 = w^2/2D
def l_of_tau(tau):  return math.sqrt(2 * D * tau)   # tau2 = l^2/2D


def window(tau1, tau2, tau3):
    p = V3Params(w=w_of_tau(tau1), l=l_of_tau(tau2), tao3=tau3,
                 D=D, tau_scale=1.0)
    m = EcfetV3(p); m.name = "v3"
    curves = analysis.stdp_sweep([m], -50e-12, +50e-12, WIDTH, DTS, TAIL)
    ys = [abs(v) for v in curves["v3"]]
    peak = max(ys)
    def crossing(frac):
        thr = frac * peak
        for i in range(1, len(ys)):
            if ys[i] <= thr <= ys[i - 1] or ys[i - 1] >= thr >= ys[i]:
                # linear interp in log-dt
                x0, x1 = DTS[i - 1], DTS[i]
                y0, y1 = ys[i - 1], ys[i]
                if y0 == y1:
                    return x1
                return x0 + (thr - y0) * (x1 - x0) / (y1 - y0)
        return float('inf') if ys[-1] > thr else DTS[-1]
    return peak, crossing(1 / math.e) * 1e3, crossing(0.05) * 1e3


print(f"{'tau1(ms)':>9} {'tau2(ms)':>9} {'tau3':>8} "
      f"{'peak_uS':>8} {'w_1e(ms)':>9} {'w_5%(ms)':>9}")
print("-" * 60)

# 1) default
for label, t1, t2, t3 in [
    ("default", 0.0222, 0.3125, 19.0),
    # 2) uniform tau_scale compressions (keep paper ratio)
    ("scale0.3", 0.0222 * .3, 0.3125 * .3, 19.0 * .3),
    ("scale0.1", 0.0222 * .1, 0.3125 * .1, 19.0 * .1),
    # 3) independent small triplets
    ("A 5/20/40ms", 0.005, 0.020, 0.040),
    ("B 8/25/60ms", 0.008, 0.025, 0.060),
    ("C 10/30/80ms", 0.010, 0.030, 0.080),
    ("D 15/40/100ms", 0.015, 0.040, 0.100),
    ("E 20/50/150ms", 0.020, 0.050, 0.150),
]:
    peak, w1e, w5 = window(t1, t2, t3)
    print(f"{t1*1e3:9.3g} {t2*1e3:9.3g} {t3:8.4g} "
          f"{peak*1e6:8.3f} {w1e:9.3g} {w5:9.3g}   {label}")
