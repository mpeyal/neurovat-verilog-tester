#!/usr/bin/env python
"""Reproduce Figure 3 (synaptic plasticity of the electrochemical graphene
synapse) with the paper-matched ecfet v2 model.

  python run_fig3.py            # generate all three panels into results/

Panels:
  a) Potentiation: +50 pA / 10 ms intercalation pulse, R ~ 4410 ohm.
     Instantaneous dR = -30 ohm, relaxing to a retained dR' = -10 ohm.
  b) Depression:   -50 pA / 10 ms deintercalation pulse, R ~ 4400 ohm.
     Instantaneous dR = +30 ohm, relaxing to a retained dR' = +10 ohm.
  c) LTP/LTD: stronger programming pulses sweep G over ~100..1150 uS in
     >250 near-linear, symmetric states.

Sign convention (paper / v2 default polarity=-1): positive current potentiates
(raises G, lowers R).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecfet import Waveform, EcfetV2, V2Params, simulate

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# --- Fig. 3 stimulus knobs -------------------------------------------------
PROBE_pA = 50.0          # panel a/b probe amplitude
WIDTH_ms = 10.0          # panel a/b and panel c pulse width
T_PULSE_s = 5.0          # pulse onset in panels a/b (x-axis = 0..40 s)
T_STOP_s = 40.0          # panel a/b observation window

# Panel c: the paper sweeps >250 states/ramp with 50 pA / 10 ms pulses; its
# per-pulse dG grows with G (constant ~0.2% dR/R), while this model writes
# constant charge->dG (paper Fig. S4: dG linear in pulse AMPLITUDE).  We use
# the charge-equivalent amplitude that traverses the full window in the same
# ~250 pulses (390 pA x 10 ms ~ 8 paper probe pulses per state).
PROG_pA = 390.0          # panel c programming amplitude (charge-equivalent)
PROG_PERIOD_s = 3.0      # panel c pulse cadence (>> tau1/tau2; 19 s tail stays small)
N_EACH = 250             # pulses per LTP (and per LTD) ramp, as in Fig. 3c
N_CYCLES = 2             # LTP/LTD cycles drawn


def _single_pulse(amp_A, width_s, t0_s, rinit_ohm, t_stop_s):
    """Run one gate pulse from a given initial resistance; return SimResult."""
    p = V2Params.paper_fig3(Rinit=rinit_ohm)   # paper calibration, not the demo default
    wf = Waveform([(t0_s, width_s, amp_A)])
    return simulate(EcfetV2(p), wf, t_stop=t_stop_s)


def panel_ab():
    width = WIDTH_ms * 1e-3
    amp = PROBE_pA * 1e-12

    # a) potentiation: +50 pA, start near 4410 ohm
    ra = _single_pulse(+amp, width, T_PULSE_s, 4410.0, T_STOP_s)
    # b) depression: -50 pA, start near 4400 ohm
    rb = _single_pulse(-amp, width, T_PULSE_s, 4400.0, T_STOP_s)

    out = {}
    for tag, r in (("a", ra), ("b", rb)):
        R_before = float(np.interp(T_PULSE_s - 1e-3, r.t, r.R))
        # instantaneous extreme just after the 10 ms pulse ends
        mask = (r.t >= T_PULSE_s) & (r.t <= T_PULSE_s + width + 1e-3)
        R_inst = float(r.R[mask][-1])
        R_settle = float(r.R[-1])
        out[tag] = (R_before, R_inst, R_settle,
                    R_inst - R_before, R_settle - R_before)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, tag, r, color, title in (
            (axes[0], "a", ra, "#1f77b4", "a) Potentiation  (+50 pA, 10 ms)"),
            (axes[1], "b", rb, "#d62728", "b) Depression  (-50 pA, 10 ms)")):
        ax.plot(r.t * 1e3, r.R, color=color, lw=1.6)
        Rb, Ri, Rs, dR, dRp = out[tag]
        ax.axhline(Rb, color="r", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(Rs, color="r", ls="--", lw=0.8, alpha=0.7)
        ax.annotate(f"ΔR = {dR:+.0f} Ω", (T_PULSE_s * 1e3, (Rb + Ri) / 2),
                    color="r", fontsize=10, ha="left", va="center")
        ax.annotate(f"ΔR' = {dRp:+.0f} Ω",
                    (T_STOP_s * 1e3 * 0.55, (Rb + Rs) / 2),
                    color="r", fontsize=10, ha="center", va="center")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("R (Ω)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Fig. 3 a/b - single-pulse potentiation / depression", fontsize=12)
    path = os.path.join(RESULTS, "fig3_ab.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, out


def panel_c():
    width = WIDTH_ms * 1e-3
    period = PROG_PERIOD_s                          # seconds: keep volatile small
    amp = PROG_pA * 1e-12

    # build N_CYCLES of [LTP: +amp xN] [LTD: -amp xN]; start at the bottom (G low)
    pulses = []
    t = 0.5
    for _ in range(N_CYCLES):
        for _k in range(N_EACH):                     # LTP (+ potentiates, G up)
            pulses.append((t, width, +amp)); t += period
        for _k in range(N_EACH):                     # LTD (- depresses, G down)
            pulses.append((t, width, -amp)); t += period
    wf = Waveform(pulses)
    t_stop = t + period

    p = V2Params.paper_fig3(Rinit=10e3)              # paper calibration; start at G ~ Gmin
    r = simulate(EcfetV2(p), wf, t_stop=t_stop)

    # Plot the RETAINED synaptic weight G_nv (the long-term state), sampled just
    # before each next pulse.  The fast volatile overshoot is excluded - that is
    # what panels a/b already characterise.
    gnv = r.extras["G_nv (S)"]
    ends = np.array([t0 + width for t0, _, _ in pulses])
    sample = ends + 0.9 * (period - width)
    G = np.interp(sample, r.t, gnv)
    G_uS = G * 1e6
    n = np.arange(1, len(G_uS) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(n, G_uS, color="#1f77b4", lw=1.2)
    ax.set_xlabel("Pulse #")
    ax.set_ylabel("G (µS)")
    ax.set_title("c) LTP / LTD (retained weight) - %d states/ramp, ~%.1f µS/step"
                 % (N_EACH, float(np.median(np.abs(np.diff(G_uS[:N_EACH]))))),
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    path = os.path.join(RESULTS, "fig3_c.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)

    steps = np.abs(np.diff(G_uS[:N_EACH]))
    return path, (float(G_uS.min()), float(G_uS.max()),
                  float(np.median(steps)), len(G_uS))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    path_ab, ab = panel_ab()
    path_c, (gmin, gmax, step, n_states_seen) = panel_c()

    print("Figure 3 reproduction (paper-matched ecfet v2)")
    print("-" * 60)
    for tag, target_dR, target_dRp in (("a", -30, -10), ("b", +30, +10)):
        Rb, Ri, Rs, dR, dRp = ab[tag]
        print(f" panel {tag}: R0={Rb:7.1f}  inst={Ri:7.1f}  settle={Rs:7.1f}"
              f"  | dR={dR:+5.1f} (target {target_dR:+d})"
              f"  dR'={dRp:+5.1f} (target {target_dRp:+d})")
    print(f" panel c: G {gmin:6.1f}..{gmax:6.1f} uS  "
          f"median step ~{step:.2f} uS  over {n_states_seen} samples")
    print("-" * 60)
    print(" plots ->", path_ab)
    print("       ->", path_c)


if __name__ == "__main__":
    main()
