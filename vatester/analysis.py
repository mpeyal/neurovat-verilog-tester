"""Hot-reloadable measurement / analysis layer for the NeuroVAT GUI.

This module holds the logic the GUI runs ON simulation results - the STDP
pairing sweep and the per-pulse (LTP/LTD) sampling.  It is on the GUI's
live-reload list alongside the model twins, so editing the measurement math
here takes effect LIVE, with no app restart (unlike vatester/app.py, which is
the GUI shell and is not hot-reloaded).

Rules for anything that lives here:
  * keep it PURE - take models / SimResults in, return plain data out;
  * NO dearpygui, NO threads, NO self/App state.  app.py owns the plumbing
    (threads, the queue, widgets) and just calls these functions, so a reload
    swaps the math without disturbing the running UI.
"""

import numpy as np

from ecfet import Waveform, simulate


def _state_trace(r, model):
    """The characterization STATE array of a SimResult, per the device profile:
    conductance G for ECFET (default), or a named observable like 'Vth (V)' for
    FeFET. Lets STDP/analysis track dG (uS) for ECFET and dVt (mV) for FeFET
    with the same code."""
    obs = getattr(model, "STDP_OBS", "G")
    if obs == "G":
        return r.G
    if obs == "R":
        return r.R
    extras = getattr(r, "extras", None) or {}
    return extras.get(obs, r.G)


def state_profile(model):
    """(label, unit, SI->display scale) for the model's STDP/analysis state."""
    return (getattr(model, "STDP_LABEL", "dG"),
            getattr(model, "STDP_UNIT", "uS"),
            getattr(model, "STDP_SCALE", 1e6))


def _retained_state(m, pulses, t_stop):
    """Retained change state(end) - state(start) for one stimulus (SI units)."""
    r = simulate(m, Waveform(pulses), t_stop, label=m.name)
    arr = _state_trace(r, m)
    return float(arr[-1] - arr[0])


def stdp_sweep(models, amp_pre, amp_post, width, dts, tail, t0c=10e-3,
               log=None):
    """ANTI-SYMMETRIC STDP window, correlation-isolated (paper Fig. 4b shape).

    For each dt we run the pre/post PAIR (use OPPOSITE polarity, e.g.
    pre = -amp, post = +amp) and ALSO each pulse alone, on an identical time
    grid + settle, then take

        dG_stdp(dt) = dG_pair - dG_pre_alone - dG_post_alone.

    Subtracting the singles removes each pulse's own (timing-independent) write
    + volatile relaxation, leaving ONLY the order-dependent A_stdp lock-in that
    fires when both pulses are present.  That lock-in is the device's 3-exp
    relaxation, so the POSITIVE side follows the paper Fig. 4b equation
        dG = A_stdp*(w1 e^-dt/22ms + w2 e^-dt/315ms + w3 e^-dt/19s)
    (peak ~A_stdp at dt->0, a ~0.7 uS 19 s tail out to ~1800 ms) and the
    negative side is its anti-symmetric mirror.

    No baseline subtraction: the correlation isolation already cancels the
    single-pulse contamination, and the value at large |dt| is the REAL 19 s
    tail (the c3 term), not an artifact - subtracting it would delete the
    paper's slow component.  Needs A_stdp > 0 (else the window is ~0).

    Returns {model.name: [dG_uS per dt]}.  `log` is an optional progress sink.
    """
    curves = {}
    for m in models:
        label, unit, scale = state_profile(m)   # dG/uS (ECFET) or dVt/mV (FeFET)
        ys = []
        for dt in dts:
            pre_t = t0c + max(0.0, -dt)
            post_t = pre_t + dt
            t_stop = max(pre_t, post_t) + width + tail
            pre_pulse = (pre_t, width, amp_pre)
            post_pulse = (post_t, width, amp_post)
            # zero-amplitude companion keeps the time grid identical across the
            # three runs (adds breakpoints but injects no current), so the
            # single-pulse contamination cancels to machine precision.
            pre_zero = (pre_t, width, 0.0)
            post_zero = (post_t, width, 0.0)
            g_pair = _retained_state(m, [pre_pulse, post_pulse], t_stop)
            g_pre = _retained_state(m, [pre_pulse, post_zero], t_stop)
            g_post = _retained_state(m, [pre_zero, post_pulse], t_stop)
            ys.append((g_pair - g_pre - g_post) * scale)
        curves[m.name] = ys
        if log:
            log(f"  [stdp] {m.name}: {label} "
                f"{min(ys):+.4g}..{max(ys):+.4g} {unit}")
    return curves


def polarization_loop(models, v_amp=3.0, period=0.2, n_pts=400, n_cycles=3):
    """Ferroelectric P-V hysteresis loop (FeFET): drive the gate with a
    triangular voltage (-v_amp -> +v_amp -> -v_amp) and record the polarization
    observable vs the applied voltage over the final (settled) cycle.

    Needs a model that exposes polarization - either a `.P` attribute (the
    normalized remnant polarization) or a `POLAR_OBS` observable.  Models without
    one (ECFET) are skipped.  Returns {model.name: {"V": [...], "P": [...],
    "unit": str}} - P is scaled by the model's `Pr` (uC/cm^2) if it has one,
    else left normalized.
    """
    out = {}
    for m in models:
        pr = getattr(getattr(m, "p", None), "Pr", None)
        has_P = hasattr(m, "P") or getattr(m, "POLAR_OBS", None)
        if not has_P:
            continue
        m.reset()
        dt = period / n_pts
        Vs, Ps = [], []
        for c in range(n_cycles):
            last = (c == n_cycles - 1)
            for k in range(n_pts):
                ph = k / n_pts
                # symmetric triangle: -v_amp at ph=0, +v_amp at ph=0.5
                V = (-v_amp + 4.0 * v_amp * ph if ph < 0.5
                     else v_amp - 4.0 * v_amp * (ph - 0.5))
                t = (c * n_pts + k) * dt
                m.step(t, dt, V)
                if last:
                    p_norm = getattr(m, "P", None)
                    if p_norm is None:
                        p_norm = m.observables().get(getattr(m, "POLAR_OBS", ""), 0.0)
                    Vs.append(V)
                    Ps.append(p_norm * pr if pr else p_norm)
        out[m.name] = {"V": Vs, "P": Ps,
                       "unit": "uC/cm^2" if pr else "norm."}
    return out


def per_pulse_samples(results, metric, scale=1.0, n_each=None):
    """Per-pulse retained-value curve - the Analysis tab's LTP/LTD sampling.

    For each SimResult: sample shortly after each pulse, relative to that
    pulse's own gap to the next (capped near the median gap so every point sees
    a comparable relaxation time).  When n_each is set, the run is split into an
    LTP branch (first n_each pulses) and an LTD branch (the rest), and the
    per-pulse deltas of each branch are returned.

    For LTP/LTD this samples the RETAINED nonvolatile weight G_nv (the paper
    Fig. 3c quantity) when the model exposes it (v2's observables include
    "G_nv (S)").  Sampling total G would otherwise CLIP at the rail under a
    rapid pulse train, because the slow (19 s) volatile pool accumulates - the
    G_nv weight is the clean triangular ramp.  Models without G_nv fall back to
    total R/G.

    `metric` selects the observable: "G" (returned in uS) or "R" (ohms) for
    ECFET, or any extras key (e.g. "Vth (V)", "P (uC/cm2)") for other devices;
    `scale` converts SI -> display (1e6 for G->uS, 1e3 for V->mV, etc.).  Returns
    a list (one entry per result, None if that result has < 2 pulses):
        {label, n:[...], vals:[...], n_each, dl:[...], dd:[...]}
    Pure: no plotting - app.py turns this into series.
    """
    out = []
    for r in results:
        wins = r.waveform.pulse_windows()
        if len(wins) < 2:
            out.append(None)
            continue
        ends = [t1 for _, t1 in wins]
        starts = [t0 for t0, _ in wins]
        t_sim = float(r.t[-1])
        inter = [starts[k + 1] - ends[k] for k in range(len(ends) - 1)
                 if starts[k + 1] > ends[k]]
        typ = sorted(inter)[len(inter) // 2] if inter else (t_sim - ends[-1])
        cap = max(1e-4, 0.9 * typ)
        sample_t = []
        for k in range(len(ends)):
            gap = (starts[k + 1] - ends[k]) if k + 1 < len(ends) \
                else (t_sim - ends[k])
            s = min(max(0.9 * gap if gap > 0 else cap, 1e-5), cap)
            sample_t.append(min(ends[k] + s, t_sim))
        extras = getattr(r, "extras", None) or {}
        if metric in ("G", "R") and "G_nv (S)" in extras:  # retained weight (no clip)
            g = np.interp(sample_t, r.t, extras["G_nv (S)"])
            arr = (1.0 / g if metric == "R" else g) * scale
        elif metric == "R":
            R_s, _ = r.at(sample_t)
            arr = R_s * scale
        elif metric == "G":
            _, G_s = r.at(sample_t)
            arr = G_s * scale
        else:                                          # any extras observable
            trace = extras.get(metric)
            if trace is None:
                _, G_s = r.at(sample_t)
                arr = G_s * scale
            else:
                arr = np.interp(sample_t, r.t, trace) * scale
        vals = arr.tolist()
        n = list(range(1, len(vals) + 1))
        d = {"label": r.label, "n": n, "vals": vals, "n_each": None,
             "dl": [], "dd": []}
        if n_each and 0 < n_each < len(vals):
            d["n_each"] = n_each
            d["dl"] = [b - a for a, b in zip(vals[:n_each - 1], vals[1:n_each])]
            d["dd"] = [b - a for a, b in zip(vals[n_each:-1], vals[n_each + 1:])]
        out.append(d)
    return out
