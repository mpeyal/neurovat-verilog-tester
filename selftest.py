"""Sanity checks for the ECFET models. Run: python selftest.py"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecfet import Waveform, EcfetV1, V1Params, EcfetV2, V2Params, EcfetV3, \
    V3Params, simulate
from vatester import analysis

PASS = 0
FAIL = 0
# v3 checks are tallied SEPARATELY so a v3 regression can never silently mask
# the v2 27/27 result (and an exception in v3 setup cannot abort the v2 report).
V3_PASS = 0
V3_FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def check_v3(name, cond, detail=""):
    """Like check() but rolls into the SEPARATE v3 tally (guarded section)."""
    global V3_PASS, V3_FAIL
    if cond:
        V3_PASS += 1
        print(f"  PASS  {name}")
    else:
        V3_FAIL += 1
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
# Sign convention: POSITIVE (intercalation) current potentiates (raises G /
# lowers R); negative depresses.  DEFAULT params = paper config (Fig.3/4);
# V2Params.demo_500() is the alternate 500-ohm single-spike preset.
import numpy as _np

# --- DEFAULT (paper) Fig.3a: +50 pA/10 ms at 4410 ohm -> dip -30, retained -10
rd = simulate(EcfetV2(V2Params(Rinit=4410.0)),
              Waveform([(5.0, 10e-3, +50e-12)]), t_stop=40.0)
Rd0 = rd.at([4.999])[0][0]
check("default(paper) Fig.3a dip ~ -30 ohm", abs((rd.R.min() - Rd0) + 30.0) < 4.0,
      f"dip={rd.R.min() - Rd0:+.1f}")
check("default(paper) Fig.3a retained ~ -10 ohm", abs((rd.R[-1] - Rd0) + 10.0) < 3.0,
      f"dR'={rd.R[-1] - Rd0:+.1f}")

# --- Fig.3c LTP/LTD must stay TRIANGULAR (not clip at the Gmax rail)
_amp, _per, _W, _N = 240e-12, 2.0, 0.01, 250
_pul, _t = [], 0.5
for _k in range(_N):
    _pul.append((_t, _W, +_amp)); _t += _per
for _k in range(_N):
    _pul.append((_t, _W, -_amp)); _t += _per
rc = simulate(EcfetV2(V2Params(Rinit=10e3)), Waveform(_pul), t_stop=_t + _per)
_Gc = _np.interp(_np.array([p[0] + _W for p in _pul]) + 0.9 * (_per - _W),
                 rc.t, rc.extras["G_nv (S)"]) * 1e6
_clip = int(_np.sum(_Gc[:_N] > 0.98 * V2Params().Gmax * 1e6))
check("Fig.3c LTP/LTD triangular (no rail clipping)", _clip < 0.1 * _N,
      f"clipped {_clip}/{_N}")
check("Fig.3c sweeps a wide G range", (_Gc.max() - _Gc.min()) > 400,
      f"span {_Gc.max() - _Gc.min():.0f} uS")

# --- STDP: ANTI-SYMMETRIC window, +side follows the paper Fig.4b 3-exp
spos = [10, 20, 50, 100, 200, 400, 800, 1200, 1800]
sdts = sorted([-d * 1e-3 for d in spos] + [d * 1e-3 for d in spos])
sy = _np.array(list(analysis.stdp_sweep(
    [EcfetV2(V2Params())], -20e-12, +20e-12, 5e-3, sdts, 4.0).values())[0])
i_m10 = sdts.index(-0.01); i_p10 = sdts.index(0.01)
check("STDP anti-symmetric (dt<0 depress, dt>0 potentiate)",
      sy[i_m10] < 0 < sy[i_p10], f"dG(-10)={sy[i_m10]:+.2f} dG(+10)={sy[i_p10]:+.2f}")
check("STDP +/- window symmetric", abs(sy[i_m10] + sy[i_p10]) < 0.2,
      f"asym={sy[i_m10] + sy[i_p10]:+.3f}")
sxp = _np.array([d for d in sdts if d > 0])
syp = _np.array([sy[i] for i, d in enumerate(sdts) if d > 0])
M = _np.column_stack([_np.exp(-sxp / 0.022), _np.exp(-sxp / 0.315), _np.exp(-sxp / 19.0)])
fit = M @ _np.linalg.lstsq(M, syp, rcond=None)[0]
r2 = 1 - _np.sum((syp - fit) ** 2) / _np.sum((syp - syp.mean()) ** 2)
check("STDP follows paper Fig.4b 3-exp (R^2 > 0.99)", r2 > 0.99, f"R^2={r2:.4f}")
check("STDP has the 19s tail (|dG(1800ms)| > 0.3 uS)", abs(syp[-1]) > 0.3,
      f"dG(1800ms)={syp[-1]:+.2f}")

# --- demo_500 preset: 500-ohm single-spike (alternate config)
dm = V2Params.demo_500()
r5 = simulate(EcfetV2(dm), Waveform([(1.0, 10e-3, +50e-12)]), t_stop=10.0)
R5 = r5.at([0.99])[0][0]
check("demo_500 +pulse 500->470->490",
      abs((r5.R.min() - R5) + 30.0) < 4.0 and abs((r5.R[-1] - R5) + 10.0) < 3.0,
      f"dip={r5.R.min() - R5:+.1f} settle={r5.R[-1] - R5:+.1f}")
r5n = simulate(EcfetV2(dm), Waveform([(1.0, 10e-3, -50e-12)]), t_stop=10.0)
R5n = r5n.at([0.99])[0][0]
check("demo_500 -pulse 500->530->510",
      abs((r5n.R.max() - R5n) - 30.0) < 4.0 and abs((r5n.R[-1] - R5n) - 10.0) < 3.0,
      f"dip={r5n.R.max() - R5n:+.1f} settle={r5n.R[-1] - R5n:+.1f}")

# --- PAPER preset below ---
p2 = V2Params.paper_fig3()
G0 = 1.0 / p2.Rinit

# polarity: +I raises G (potentiation), -I lowers G (depression)
r_pot = simulate(EcfetV2(p2), Waveform([(0.01, 10e-3, +50e-12)]), t_stop=0.05)
check("positive current potentiates (G up)", r_pot.G[-1] > G0,
      f"G end {r_pot.G[-1]:.3e} vs G0 {G0:.3e}")
r_dep = simulate(EcfetV2(p2), Waveform([(0.01, 10e-3, -50e-12)]), t_stop=0.05)
check("negative current depresses (G down)", r_dep.G[-1] < G0,
      f"G end {r_dep.G[-1]:.3e} vs G0 {G0:.3e}")

# one unit pulse (50 pA x 10 ms = Q_ref): instantaneous |dG| ~ dG_unit * window
m2 = EcfetV2(V2Params.paper_fig3())
r2 = simulate(m2, Waveform([(5.0, 10e-3, +50e-12)]), t_stop=120.0)
_, G_inst = r2.at([5.011])
written = G_inst[0] - G0
check("unit pulse writes ~dG_unit*window",
      0.6 * p2.dG_unit < written < 1.1 * p2.dG_unit,
      f"dG={written:.3e}, unit={p2.dG_unit:.3e}")

# volatile part decays; retained settles near (1-kappa_v) of the write
_, G_late = r2.at([90.0])
retained = G_late[0] - G0
exp_ret = (1.0 - p2.kappa_v) * written
check("volatile relaxation -> retained ~ (1-kappa_v)*write",
      0.7 * exp_ret < retained < 1.3 * exp_ret,
      f"retained={retained:.3e} expected~{exp_ret:.3e}")

# retention plateau: the 19 s tail is done by ~60 s, then G_nv is stable
check("retained value stable (60 s vs 90 s)",
      abs(retained - (r2.at([60.0])[1][0] - G0)) < 0.1 * abs(retained),
      f"ret90={retained:.3e}")

# proportionality: half-width pulse writes ~half
m2b = EcfetV2(V2Params.paper_fig3())
r2b = simulate(m2b, Waveform([(0.01, 5e-3, +50e-12)]), t_stop=0.05)
_, G_half = r2b.at([0.0161])
ratio = (G_half[0] - G0) / written
check("charge-proportional write (~0.5x for half charge)",
      0.35 < ratio < 0.65, f"ratio={ratio:.3f}")

# bounds: hammer with potentiation (positive), never exceeds Gmax
m2c = EcfetV2(V2Params.paper_fig3())
r2c = simulate(m2c, Waveform.pulse_train(+1e-9, 10e-3, 5.0, 60), t_stop=320.0)
check("Gmax bound respected", r2c.G.max() <= 1.0 / p2.Rmin + 1e-9,
      f"max G={r2c.G.max():.4e}")
check("R within [Rmin, Rmax]",
      r2c.R.min() >= p2.Rmin - 1e-6 and r2c.R.max() <= p2.Rmax + 1e-6)

# Figure 3a: +50 pA / 10 ms at R~4410 ohm -> instantaneous dR~-30, retained dR'~-10
m2e = EcfetV2(V2Params.paper_fig3(Rinit=4410.0))
r2e = simulate(m2e, Waveform([(5.0, 10e-3, +50e-12)]), t_stop=40.0)
R0 = r2e.at([4.999])[0][0]
R_inst = r2e.at([5.0101])[0][0]
R_set = r2e.R[-1]
check("Fig.3a instantaneous dR ~ -30 ohm", abs((R_inst - R0) + 30.0) < 4.0,
      f"dR={R_inst - R0:+.1f}")
check("Fig.3a retained dR' ~ -10 ohm", abs((R_set - R0) + 10.0) < 3.0,
      f"dR'={R_set - R0:+.1f}")

# determinism with seed + noise reproducibility
wf_train = Waveform.pulse_train(+100e-12, 10e-3, 50e-3, 10)
ra = simulate(EcfetV2(V2Params.paper_fig3(sigma_c2c=0.1, seed=42)), wf_train, t_stop=0.6)
rb = simulate(EcfetV2(V2Params.paper_fig3(sigma_c2c=0.1, seed=42)), wf_train, t_stop=0.6)
check("seeded noise reproducible", float(abs(ra.G[-1] - rb.G[-1])) == 0.0)

# ===========================================================================
# v3 (paper-faithful CONSERVED-x + SATURATING-G(x)) - guarded smoke checks.
#
# The whole v3 section runs in a try/except so that ANY v3 regression (a failed
# assertion OR an outright exception during setup) is recorded in the SEPARATE
# V3_PASS/V3_FAIL tally and printed AFTER the complete v2 report.  v2's 27/27 is
# therefore always evaluated and printed first and can never be masked by v3.
# The full quantitative v3 battery lives in tools/validate_v3.py; these are a
# fast subset (Fig.3a step, Fig.3c span on G_nv, the CONCAVE-DOWN saturating
# curvature signature, STDP anti-symmetry, and the closed-cycle conservation).
# ===========================================================================
print("== v3 (paper-faithful conserved-x saturating) [guarded] ==")
try:
    # --- Fig.3a: +50 pA/10 ms @ 4400 ohm -> instantaneous -30, retained -10 ---
    p3 = V3Params.paper(Rinit=4400.0)
    r3 = simulate(EcfetV3(p3), Waveform([(5.0, 10e-3, +50e-12)]), t_stop=120.0)
    R3_0 = r3.at([4.999])[0][0]
    _mask = (r3.t >= 5.0) & (r3.t <= 5.0 + 10e-3 + 1e-9)
    R3_inst = float(r3.R[_mask][-1])
    _gnv = r3.extras["G_nv (S)"]
    _gnv0 = float(_np.interp(4.999, r3.t, _gnv))
    _gnv1 = float(_np.interp(5.0 + 10e-3, r3.t, _gnv))
    dRp_nv = 1.0 / _gnv1 - 1.0 / _gnv0          # retained step on PURE G_nv
    check_v3("v3 Fig.3a instantaneous dR ~ -30 ohm",
             abs((R3_inst - R3_0) + 30.0) < 4.0, f"dR={R3_inst - R3_0:+.1f}")
    check_v3("v3 Fig.3a retained dR' (G_nv) ~ -10 ohm",
             abs(dRp_nv + 10.0) < 2.0, f"dR'={dRp_nv:+.2f}")

    # --- Fig.3c LTP/LTD on G_nv: spans the window AND SYMMETRIC LTD to Gmin ---
    _amp3, _per3, _W3, _N3 = 50e-12, 0.2, 10e-3, 1100
    _pul3, _t3 = [], 0.5
    for _k in range(_N3):
        _pul3.append((_t3, _W3, +_amp3)); _t3 += _per3
    for _k in range(_N3):
        _pul3.append((_t3, _W3, -_amp3)); _t3 += _per3
    pc = V3Params.paper(Rinit=8333.0)
    rc3 = simulate(EcfetV3(pc), Waveform(_pul3), t_stop=_t3 + _per3)
    _gnv3 = rc3.extras["G_nv (S)"]               # sample RETAINED G_nv, not +stdp
    _samp = _np.array([p[0] + _W3 for p in _pul3]) + 0.9 * (_per3 - _W3)
    _G3 = _np.interp(_samp, rc3.t, _gnv3) * 1e6
    _up = _G3[:_N3]; _dn = _G3[_N3:]
    check_v3("v3 Fig.3c spans ~full window on G_nv (>850 uS)",
             (_up.max() - _up.min()) > 850.0,
             f"span {_up.max() - _up.min():.0f} uS")
    check_v3("v3 Fig.3c SYMMETRIC LTD returns to Gmin on G_nv",
             abs(_dn[-1] - pc.Gmin * 1e6) < 8.0,
             f"return {_dn[-1]:.1f} uS vs Gmin {pc.Gmin * 1e6:.1f}")

    # --- CONCAVE-DOWN (saturating) kill-shot for the old concave-UP bug: the
    # per-pulse retained step in the TOP third of the up-ramp must be SMALLER than
    # in the MIDDLE third (the step SHRINKS as G fills toward Gmax).  The old
    # multiplicative/accelerating law had the step GROW (top > mid).
    _st = _np.abs(_np.diff(_up))
    _t3rd = max(1, len(_st) // 3)
    _mid_step = float(_np.mean(_st[_t3rd:2 * _t3rd]))
    _top_step = float(_np.mean(_st[2 * _t3rd:3 * _t3rd]))
    check_v3("v3 Fig.3c LTP ramp CONCAVE-DOWN (top-third step < mid-third)",
             _top_step < _mid_step,
             f"top={_top_step:.3f} uS  mid={_mid_step:.3f} uS")

    # --- STDP: anti-symmetric, no eligibility-trace runaway on a monotonic ramp
    def _stdp_pair(dt, positive):
        m = EcfetV3(V3Params.paper(Rinit=4400.0))
        first, second = (-50e-12, +50e-12) if positive else (+50e-12, -50e-12)
        t = 1.0
        m.step(t, 10e-3, first); t += 10e-3
        m.step(t, 1e-4, 0.0); t += 1e-4
        gap = max(dt - 10e-3, 1e-6); m.step(t, gap, 0.0); t += gap
        lb = m.stdp_lock; m.step(t, 10e-3, second); t += 10e-3
        return m.stdp_lock - lb
    _dp = _stdp_pair(0.01, True); _dn2 = _stdp_pair(0.01, False)
    check_v3("v3 STDP anti-symmetric (dG+ ~ -dG-)",
             _dp > 0 > _dn2 and abs(_dp + _dn2) < 1e-12,
             f"dG+={_dp * 1e6:+.3f} dG-={_dn2 * 1e6:+.3f} uS")
    # C1: a monotonic LTP ramp must NOT accumulate stdp_lock (trace consumed/cap)
    m_ramp = EcfetV3(V3Params.paper(Rinit=8333.0)); _t = 0.5
    for _k in range(200):
        m_ramp.step(_t, 10e-3, +50e-12); _t += 10e-3
        m_ramp.step(_t, 0.19, 0.0); _t += 0.19
    check_v3("v3 STDP no runaway on monotonic ramp (|stdp_lock| < 1 uS)",
             abs(m_ramp.stdp_lock) < 1e-6, f"stdp_lock={m_ramp.stdp_lock * 1e6:+.3f} uS")

    # --- S6 conservation: a closed +/- cycle must not ratchet G_nv ----------
    mc = EcfetV3(V3Params.paper(Rinit=4400.0)); _t = 0.5
    _lo = []
    for _c in range(500):
        mc.step(_t, 10e-3, +50e-12); _t += 10e-3
        mc.step(_t, 0.19, 0.0); _t += 0.19
        mc.step(_t, 10e-3, -50e-12); _t += 10e-3
        mc.step(_t, 0.19, 0.0); _t += 0.19
        _lo.append(mc.G_nv)
    _drift = abs((_lo[-1] - _lo[0]) / _lo[0] * 100.0)
    check_v3("v3 S6 closed +/- cycle conserves (|drift| < 0.5%/500cy)",
             _drift < 0.5, f"drift={_drift:.3f}%")
except Exception as _exc:           # any v3 exception -> recorded, NOT fatal to v2
    import traceback as _tb
    V3_FAIL += 1
    print(f"  FAIL  v3 section raised an exception (v2 result unaffected): {_exc!r}")
    _tb.print_exc()

print(f"\nv2/v1: {PASS} passed, {FAIL} failed")
print(f"v3   : {V3_PASS} passed, {V3_FAIL} failed")
print(f"TOTAL: {PASS + V3_PASS} passed, {FAIL + V3_FAIL} failed")
sys.exit(1 if (FAIL or V3_FAIL) else 0)
