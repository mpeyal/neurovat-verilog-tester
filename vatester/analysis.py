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

from simulation_core import Waveform, simulate


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
    """Legacy rectangular-pair interaction measurement.

    The Fig. 4b interpretation below applies ONLY to the ECFET models with
    their explicit A_stdp timing rule enabled. It is not a derivation of
    intrinsic STDP in a bare FeFET, FeCAP or arbitrary memristor. The desktop
    uses stdp_protocol for waveform shaping and explicit control experiments.

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


def state_trace(r, model):
    """Public alias of the device-profile state array (dG for ECFET, dVt for
    FeFET, dP for FeCAP) - used by the GUI drilldown."""
    return _state_trace(r, model)


# ---------------------------------------------------------------------------
# Overlapping-spike STDP
# ---------------------------------------------------------------------------
# Devices with no internal timing trace (a bare ferroelectric capacitor: nothing
# evolves at V = 0) are TIMING-BLIND to the rectangular pairs above - the pair
# result depends only on pulse order, never on |dt|.  Physical STDP in such
# devices comes from the stimulus: each neuron emits a spike that is
# sub-threshold on its own - a short peak followed by a decaying tail of the
# opposite sign.  Pre spikes drive one electrode and post spikes the other, so
# the device sees their DIFFERENCE.  When post follows pre closely, post's peak
# lands on pre's still-decaying tail and the difference crosses threshold for a
# time that shrinks as |dt| grows; the device's switching nonlinearity turns
# that into a retained, timing-dependent change.  The window width is set by
# the spike tail time constant - a neuron/circuit design choice.
TAIL_SPAN_TAU = 6.0          # tail truncated after 6 tau (e^-6 = 0.25 %)
TAIL_STEPS_PER_TAU = 8       # staircase resolution of the exponential tail


def spike_pulses(t0, amp, width, tail_ratio, tail_tau):
    """One spike as rectangular (t_start, width, amp) pieces.

    Peak: +amp for `width`.  Tail: -tail_ratio*amp*exp(-(t - t0 - width)/tail_tau)
    as a staircase of tail_tau/TAIL_STEPS_PER_TAU steps valued at each step's
    midpoint.  A real Waveform (not an analytic callable) so every GUI consumer
    of a stimulus - plots, snapshots, pulse counts - works unchanged.  amp = 0
    still emits every piece, which keeps companion runs on an identical grid.
    """
    pieces = [(t0, width, amp)]
    if tail_ratio == 0.0 or tail_tau <= 0.0:
        return pieces
    d = tail_tau / TAIL_STEPS_PER_TAU
    n = int(round(TAIL_SPAN_TAU * TAIL_STEPS_PER_TAU))
    for k in range(n):
        v = -tail_ratio * amp * np.exp(-(k + 0.5) * d / tail_tau)
        pieces.append((t0 + width + k * d, d, float(v)))
    return pieces


def spike_pair_waveform(t_pre, t_post, amp_pre, amp_post, width, tail_ratio,
                        tail_tau, use_pre=True, use_post=True):
    """Pre + post spikes on one drive signal (use_* = False zeroes that spike but
    keeps its grid points, for correlation isolation)."""
    return Waveform(
        spike_pulses(t_pre, amp_pre if use_pre else 0.0, width, tail_ratio,
                     tail_tau)
        + spike_pulses(t_post, amp_post if use_post else 0.0, width,
                       tail_ratio, tail_tau))


def spike_pair_times(dt, t0c=10e-3):
    """(t_pre, t_post) for a given dt = t_post - t_pre, both >= t0c."""
    t_pre = t0c + max(0.0, -dt)
    return t_pre, t_pre + dt


def spike_pair_tstop(t_pre, t_post, width, tail_tau, settle):
    """Stop after the later spike's tail has fully decayed, plus `settle`."""
    return max(t_pre, t_post) + width + TAIL_SPAN_TAU * max(tail_tau, 0.0) + settle


def stdp_sweep_overlap(models, amp_pre, amp_post, width, dts, tail_ratio,
                       tail_tau, settle=0.02, t0c=10e-3, log=None):
    """STDP window from OVERLAPPING sub-threshold spikes (see the note above).

    Same correlation isolation as stdp_sweep: state_stdp(dt) = pair - pre alone
    - post alone, all three on an identical time grid, so each spike's own
    (timing-independent) effect cancels and only the overlap interaction is
    left.  Use OPPOSITE-signed amp_pre / amp_post so the device sees the
    post - pre difference.  Returns {model.name: [state change per dt]} in the
    model's STDP display unit, like stdp_sweep.
    """
    curves = {}
    for m in models:
        label, unit, scale = state_profile(m)
        ys = []
        for dt in dts:
            t_pre, t_post = spike_pair_times(dt, t0c)
            t_stop = spike_pair_tstop(t_pre, t_post, width, tail_tau, settle)
            runs = []
            for use_pre, use_post in ((True, True), (True, False),
                                      (False, True)):
                wf = spike_pair_waveform(t_pre, t_post, amp_pre, amp_post,
                                         width, tail_ratio, tail_tau,
                                         use_pre, use_post)
                r = simulate(m, wf, t_stop, label=m.name)
                arr = _state_trace(r, m)
                runs.append(float(arr[-1] - arr[0]))
            pair, pre_alone, post_alone = runs
            ys.append((pair - pre_alone - post_alone) * scale)
        curves[m.name] = ys
        if log:
            log(f"  [stdp] {m.name} (overlapping spikes): {label} "
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
        Vs, Ps, Is, Cs = [], [], [], []
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
                    obs = m.observables()
                    p_norm = getattr(m, "P", None)
                    if p_norm is None:
                        p_norm = obs.get(getattr(m, "POLAR_OBS", ""), 0.0)
                    Vs.append(V)
                    Ps.append(p_norm * pr if pr else p_norm)
                    # capacitor twins also give the switching current (I-V
                    # loop) and the large-signal dQ/dV (C-V butterfly)
                    if "I_dev (A)" in obs:
                        Is.append(obs["I_dev (A)"])
                    if "C_eff (F)" in obs:
                        Cs.append(obs["C_eff (F)"])
        entry = {"V": Vs, "P": Ps, "unit": "uC/cm^2" if pr else "norm."}
        if Is and len(Is) == len(Vs):
            entry["I"] = Is
        if Cs and len(Cs) == len(Vs):
            entry["C"] = Cs
        out[m.name] = entry
    return out


# ---------------------------------------------------------------------------
# Ferroelectric capacitor characterization: PUND and switching kinetics
# ---------------------------------------------------------------------------
PUND_OBS = "Ptot (uC/cm2)"      # total polarization = switched + dielectric
FE_OBS = "P (uC/cm2)"           # switched ferroelectric part only


def pund_metrics(r, windows, obs=PUND_OBS):
    """PUND numbers from one simulated run.

    windows: the (t_start, t_end) of the P, U, N, D pulses, in that order.  The
    polarization change DURING each pulse is read from the total-polarization
    trace; the simulator lands a grid point on every pulse edge, so the value
    at t_start is pre-edge and the value at t_end is end-of-pulse.
      P* = switching + dielectric charge    P^ = dielectric only (U pulse)
      switched (+) = P* - P^ = 2 Pr         switched (-) = N* - N^ = -2 Pr
    Leakage adds the same charge to P and U (and to N and D), so it cancels in
    the differences, as in a real PUND measurement.  Values in uC/cm^2.
    """
    t, y = r.t, r.extras[obs]

    def d(w):
        return float(np.interp(w[1], t, y) - np.interp(w[0], t, y))

    p_star, p_hat, n_star, n_hat = (d(w) for w in windows)
    sw_pos, sw_neg = p_star - p_hat, n_star - n_hat
    return {"P*": p_star, "P^": p_hat, "N*": n_star, "N^": n_hat,
            "switched+": sw_pos, "switched-": sw_neg,
            "Pr": (sw_pos - sw_neg) / 4.0}


def pund_measure(models, amp, width, delay, preset=True):
    """Run a PUND sequence on each capacitor model.

    Returns {model.name: {"t_ms", "I_nA", "P", "windows_ms", "metrics"}}; the
    I(t) trace is the classic PUND current response.  Models without the
    capacitor observables (I_dev, Ptot) are skipped.
    """
    from vatester import signal_factory as sf
    pulses, _ = sf._pund({"amp": amp, "width_ms": width * 1e3,
                          "delay_ms": delay * 1e3, "t0_ms": 10.0,
                          "preset": 1 if preset else 0})
    wf = Waveform(pulses)
    windows = wf.pulse_windows()[-4:]
    t_stop = windows[-1][1] + delay
    out = {}
    for m in models:
        if PUND_OBS not in m.observables() or "I_dev (A)" not in m.observables():
            continue
        r = simulate(m, wf, t_stop, dt_min=min(1e-6, width / 50.0),
                     label=m.name)
        out[m.name] = {
            "t_ms": (r.t * 1e3).tolist(),
            "I_nA": (np.asarray(r.extras["I_dev (A)"]) * 1e9).tolist(),
            "P": list(r.extras[PUND_OBS]),
            "windows_ms": [(a * 1e3, b * 1e3) for a, b in windows],
            "metrics": pund_metrics(r, windows),
        }
    return out


def switching_kinetics(models, amps, widths, preset_amp=3.0,
                       preset_width=1e-3, gap=1e-3):
    """Switched-polarization map vs pulse width, one curve per amplitude.

    The NLS calibration measurement: preset the device fully negative, apply ONE
    positive pulse of width w and amplitude a, and read the retained change of
    the switched polarization at zero bias.  Plotted against log(width) this
    gives the stretched switching curves whose position, spread and slope pin
    tau0 / Ea / alpha, sigma_eta and beta.

    Returns {model.name: {"widths": [...], "curves": {amp: [fraction...]},
    "Pr": Pr}} where fraction = dP / (2 Pr) (1.0 = full reversal).
    """
    out = {}
    for m in models:
        if FE_OBS not in m.observables():
            continue
        pr = getattr(getattr(m, "p", None), "Pr", None)
        curves = {}
        for a in amps:
            fr = []
            for w in widths:
                t_pre = 1e-3
                t_pulse = t_pre + preset_width + gap
                wf = Waveform([(t_pre, preset_width, -abs(preset_amp)),
                               (t_pulse, w, abs(a))])
                r = simulate(m, wf, t_pulse + w + gap,
                             dt_min=min(1e-6, w / 50.0), label=m.name)
                y = r.extras[FE_OBS]
                dp = float(y[-1] - np.interp(t_pulse, r.t, y))
                fr.append(dp / (2.0 * pr) if pr else dp)
            curves[a] = fr
        out[m.name] = {"widths": list(widths), "curves": curves, "Pr": pr}
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
        # Settle delay before the read.  The 10 us / 100 us floors suit the
        # ECFET, whose pulses are seconds apart, but a FeCAP train can have a
        # gap far shorter than that - the read would then land several pulses
        # later and the curve would look scrambled.  Keep the floors, but never
        # let them exceed the gap itself.
        floor = min(1e-5, 0.5 * typ) if typ > 0 else 1e-5
        cap = max(floor, min(1e-4, 0.9 * typ) if typ > 0 else 1e-4, 0.9 * typ)
        sample_t = []
        for k in range(len(ends)):
            gap = (starts[k + 1] - ends[k]) if k + 1 < len(ends) \
                else (t_sim - ends[k])
            # A short gap in an irregular train must override the typical-gap
            # floor. Touching/overlapping pulses have no idle read interval;
            # sample their end instead of advancing into later pulses.
            s = min(0.9 * gap, cap) if gap > 0 else 0.0
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
