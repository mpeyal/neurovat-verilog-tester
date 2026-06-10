"""Sanity checks for the ECFET models. Run: python selftest.py"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecfet import Waveform, EcfetV1, V1Params, EcfetV2, V2Params, simulate

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


print("== v1 (Verilog-A port) ==")
# 100 pA x 10 ms -> dM = Rdrift1c * I * t = 20e12 * 1e-10 * 1e-2 = 20 ohm
wf = Waveform([(10e-3, 10e-3, 100e-12)])
r = simulate(EcfetV1(), wf, t_stop=2.0)
R_end_of_pulse, _ = r.at([20e-3])
check("drift magnitude (+20 ohm during pulse)",
      abs(R_end_of_pulse[0] - 520.0) < 0.5, f"got {R_end_of_pulse[0]:.3f}")
# right after pulse: fixed c1+c2+c3 = 20 ohm dip below M_before_diffusion
R_dip, _ = r.at([20.5e-3])
check("post-pulse diffusion dip exists (v1 artifact)",
      R_dip[0] < 510.0, f"got {R_dip[0]:.3f}")
# after the 1 s window state freezes near M_bd - residual
check("state bounded", (r.R.min() >= 100.0) and (r.R.max() <= 10e3))

# below-threshold current does nothing (leak_drift_scale = 0)
wf_sub = Waveform([(10e-3, 10e-3, 0.5e-12)])
r_sub = simulate(EcfetV1(), wf_sub, t_stop=0.1)
check("sub-threshold spike ignored", abs(r_sub.R[-1] - 500.0) < 1e-6,
      f"got {r_sub.R[-1]}")

# negative pulse drives R down
wf_neg = Waveform([(10e-3, 10e-3, -100e-12)])
r_neg = simulate(EcfetV1(), wf_neg, t_stop=15e-3)
check("negative current lowers R", r_neg.R[-1] < 500.0,
      f"got {r_neg.R[-1]:.3f}")

# saturation clamp at Rmax
wf_big = Waveform([(10e-3, 1.0, 10e-9)])  # 10 nA x 1 s -> would be +200 kohm
r_big = simulate(EcfetV1(), wf_big, t_stop=1.05)
check("Rmax clamp", abs(r_big.R.max() - 10e3) < 1e-6, f"got {r_big.R.max()}")

print("== v2 (practical ECFET) ==")
p2 = V2Params()
# one unit pulse (100 pA x 10 ms = Q_ref): |dG| ~ dG_unit * window
m2 = EcfetV2(p2)
wf_p = Waveform([(10e-3, 10e-3, -100e-12)])   # negative I -> G up
r2 = simulate(m2, wf_p, t_stop=120.0)
_, G_after = r2.at([21e-3])
G0 = 1.0 / 500.0
check("unit pulse raises G by ~dG_unit*window",
      0.2 * p2.dG_unit < (G_after[0] - G0) < 1.2 * p2.dG_unit,
      f"dG={G_after[0]-G0:.3e}, unit={p2.dG_unit:.3e}")
# volatile part decays: G(60s) < G(just after pulse), but retains > kappa share
_, G_late = r2.at([60.0])
written = G_after[0] - G0
retained = G_late[0] - G0
check("volatile relaxation (partial decay)",
      0.5 * written < retained < 0.999 * written,
      f"written={written:.3e} retained={retained:.3e}")

# proportionality: half-charge pulse writes ~half (weak state dependence aside)
m2b = EcfetV2(V2Params())
r2b = simulate(m2b, Waveform([(10e-3, 5e-3, -100e-12)]), t_stop=30e-3)
_, G_half = r2b.at([16e-3])
ratio = (G_half[0] - G0) / written
check("charge-proportional write (~0.5x for half charge)",
      0.35 < ratio < 0.65, f"ratio={ratio:.3f}")

# bounds: hammer with potentiation, never exceeds Gmax
m2c = EcfetV2(V2Params())
r2c = simulate(m2c, Waveform.pulse_train(-1e-9, 10e-3, 20e-3, 300), t_stop=7.0)
check("Gmax bound respected", r2c.G.max() <= 1.0 / p2.Rmin + 1e-12,
      f"max G={r2c.G.max():.4e}")
check("R within [Rmin, Rmax]",
      r2c.R.min() >= p2.Rmin - 1e-9 and r2c.R.max() <= p2.Rmax + 1e-9)

# nonlinearity: successive LTP steps shrink (soft saturation)
m2d = EcfetV2(V2Params())
wf_train = Waveform.pulse_train(-100e-12, 10e-3, 50e-3, 10)
r2d = simulate(m2d, wf_train, t_stop=0.6)
import numpy as np
ends = np.array([t1 for _, t1 in wf_train.pulse_windows()])
_, Gs = r2d.at(ends + 1e-4)
dGs = np.diff(np.concatenate([[G0], Gs]))
check("soft-bound: step size shrinks with state", dGs[-1] < dGs[0],
      f"first={dGs[0]:.3e} last={dGs[-1]:.3e}")

# determinism with seed + noise reproducibility
ra = simulate(EcfetV2(V2Params(sigma_c2c=0.1, seed=42)), wf_train, t_stop=0.6)
rb = simulate(EcfetV2(V2Params(sigma_c2c=0.1, seed=42)), wf_train, t_stop=0.6)
check("seeded noise reproducible", float(abs(ra.G[-1] - rb.G[-1])) == 0.0)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
