"""Extract the Sharbati-2018 observables from a canonical-column run (twin or
scrubbed Spectre).  Every function takes the dict of canonical columns
(validation/common.py COLUMNS) and the matching sharbati-1 Stimulus.

Definitions (also recorded per target in sharbati2018_targets.json):
  single pulse   R0 = R at the pulse onset; dR_peak = extreme R in the 1 s after
                 the onset minus R0; dR_fig = R(onset + 35 s) - R0 (inside the
                 paper's 40 s window); dR_asym = R(onset + 200 s) - R0.
  LTP/LTD        read r_k = G at the onset of pulse k+1 (end of each 50 ms
                 period).  Linearity = R^2 of a straight-line fit of r_k vs k
                 over the reads inside 10-90 % of the LTP ramp's own G range.
                 Distinct states = LTP reads that rise above the previous read by
                 more than 1 % of the median 10-90 % step.  Modulation =
                 (Gmax - Gmin)/Gmin over the whole run.
  paired pulse   peak1 = max G in [t1, t2); peak2 = max G in [t2, t_stop];
                 dG = peak2 - peak1 (primary; the Fig. 4b inset's lower dotted
                 line sits on the first pulse's level).  dG_alt = peak2 - G(t2)
                 (increment of the second pulse alone).
  retention      100 * (G(t_settle + 13 h) - G(t_settle)) / G(t_settle).
"""

import math

import numpy as np

TAU_PAPER_S = (22e-3, 315e-3, 19.0)


def _at(cols, t):
    return float(np.interp(t, cols["t_s"], cols["G_S"]))


def _R_at(cols, t):
    return float(np.interp(t, cols["t_s"], cols["R_ohm"]))


def _window(cols, lo, hi, key="G_S"):
    t = cols["t_s"]
    m = (t >= lo - 1e-12) & (t <= hi + 1e-12)
    return np.asarray(cols[key])[m]


# --------------------------------------------------------------------------
def single_pulse(cols, stim):
    p = stim.params
    t_on = p["t_on_s"]
    pot = p["amp_A"] > 0
    R0 = _R_at(cols, t_on)
    seg = _window(cols, t_on, t_on + 1.0, "R_ohm")
    R_pk = float(seg.min() if pot else seg.max())
    dR_pk = R_pk - R0
    dR_fig = _R_at(cols, t_on + p["read_fig_s"]) - R0
    dR_asym = _R_at(cols, cols["t_s"][-1]) - R0
    return {"R0_ohm": R0, "dR_peak_ohm": dR_pk, "dR_peak_pct": 100 * dR_pk / R0,
            "dR_fig_ohm": dR_fig, "dR_fig_pct": 100 * dR_fig / R0,
            "dR_asym_ohm": dR_asym, "dR_asym_pct": 100 * dR_asym / R0,
            "step_pct_settled": 100 * abs(dR_asym) / R0,
            "step_pct_instant": 100 * abs(dR_pk) / R0,
            "t_asym_s": float(cols["t_s"][-1] - t_on)}


# --------------------------------------------------------------------------
def linear_r2(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) < 3:
        return math.nan
    a, b = np.polyfit(x, y, 1)
    res = y - (a * x + b)
    tot = y - y.mean()
    return float(1.0 - np.dot(res, res) / np.dot(tot, tot))


def window_10_90(r):
    r = np.asarray(r, dtype=float)
    lo, hi = float(r.min()), float(r.max())
    sel = (r >= lo + 0.1 * (hi - lo)) & (r <= lo + 0.9 * (hi - lo))
    return sel


def ltp_reads(cols, stim):
    p = stim.params
    per, n = p["period_s"], p["n_each"]
    t_ltp, t_ltd = p["t_ltp_s"], p["t_ltd_s"]
    ltp_t = t_ltp + per * np.arange(1, n + 1)          # last read = t_ltd
    ltd_t = t_ltd + per * np.arange(1, n + 1)
    ltd_t[-1] = min(ltd_t[-1], cols["t_s"][-1])
    g = lambda ts: np.interp(ts, cols["t_s"], cols["G_S"])  # noqa: E731
    return g(np.concatenate([[t_ltp], ltp_t])), g(np.concatenate([[t_ltd], ltd_t]))


def ltp_ltd(cols, stim):
    ltp, ltd = ltp_reads(cols, stim)
    sel = window_10_90(ltp)
    steps = np.diff(ltp)
    win_steps = steps[sel[1:] & sel[:-1]]
    med = float(np.median(win_steps)) if len(win_steps) else math.nan
    n_states = int(np.sum(steps > 0.01 * med)) if med > 0 else 0
    rel = steps / ltp[:-1]
    sel_d = window_10_90(ltd)
    g = np.asarray(cols["G_S"])
    out = {"r2_10_90": linear_r2(ltp[sel]),
           "ltd_r2_10_90": linear_r2(ltd[sel_d]),
           "n_states": n_states, "n_reads_10_90": int(sel.sum()),
           "step_med_10_90_uS": 1e6 * med,
           "step_pct_max_10_90": 100 * float(np.max(rel[sel[1:] & sel[:-1]]))
           if len(win_steps) else math.nan,
           "G_min_uS": 1e6 * float(g.min()), "G_max_uS": 1e6 * float(g.max()),
           "modulation_pct": 100 * float((g.max() - g.min()) / g.min())}
    x = cols.get("x_Li")
    if x is not None and np.all(np.isfinite(x)):
        # v3 only: the same fit over 10-90 % of the conserved x (lithiation)
        p = stim.params
        t_r = p["t_ltp_s"] + p["period_s"] * np.arange(0, p["n_each"] + 1)
        xr = np.interp(t_r, cols["t_s"], x)
        selx = (xr >= 0.1) & (xr <= 0.9)
        out["r2_x10_90"] = linear_r2(ltp[selx]) if selx.sum() >= 3 else math.nan
    return out


# --------------------------------------------------------------------------
def paired_pulse(cols, stim):
    p = stim.params
    t1, t2 = p["t1_s"], p["t2_s"]
    t = cols["t_s"]
    g = np.asarray(cols["G_S"])
    pk1 = float(g[(t >= t1) & (t < t2)].max())
    pk2 = float(g[t >= t2].max())
    pre2 = _at(cols, t2)
    return {"dt_onset_s": p["dt_onset_s"], "gap_s": p["gap_s"],
            "dG_S": pk2 - pk1, "dG_alt_S": pk2 - pre2,
            "peak1_S": pk1, "peak2_S": pk2, "G_pre2_S": pre2}


def three_exp(dt, c1, t1, c2, t2, c3, t3):
    return c1 * np.exp(-dt / t1) + c2 * np.exp(-dt / t2) + c3 * np.exp(-dt / t3)


def _r2(y, f):
    y = np.asarray(y, dtype=float)
    res = y - f
    tot = y - y.mean()
    return float(1.0 - np.dot(res, res) / np.dot(tot, tot)) if np.dot(tot, tot) > 0 else math.nan


def fit_three_exp(dt_s, dG):
    """Paper Fig. 4b form dG = sum c_i exp(-dt/tau_i), no offset.
    Returns (free fit dict, fixed-tau fit dict).  The free fit has six free
    parameters (c_i >= 0, tau bounds 1 ms-0.15 s, 0.1-3 s, 2-1000 s), started
    from the paper's taus; the fixed fit pins tau to 22 ms / 315 ms / 19 s and
    solves the amplitudes by linear least squares."""
    from scipy.optimize import curve_fit

    dt_s = np.asarray(dt_s, dtype=float)
    y = np.asarray(dG, dtype=float)
    scale = float(np.max(np.abs(y))) or 1.0
    yn = y / scale
    basis = np.column_stack([np.exp(-dt_s / t) for t in TAU_PAPER_S])
    c_fix, *_ = np.linalg.lstsq(basis, yn, rcond=None)
    fixed = {"c_S": [float(c) * scale for c in c_fix],
             "r2": _r2(yn, basis @ c_fix)}
    free = {"ok": False, "tau_s": [math.nan] * 3, "c_S": [math.nan] * 3,
            "r2": math.nan, "note": ""}
    if np.ptp(yn) < 1e-9:
        free["note"] = "flat series: no decay to fit"
        return free, fixed
    p0 = [max(c_fix[0], 1e-3), TAU_PAPER_S[0], max(c_fix[1], 1e-3), TAU_PAPER_S[1],
          max(c_fix[2], 1e-3), TAU_PAPER_S[2]]
    lo = [0, 1e-3, 0, 0.1, 0, 2.0]
    hi = [np.inf, 0.15, np.inf, 3.0, np.inf, 1000.0]
    try:
        popt, _ = curve_fit(three_exp, dt_s, yn, p0=p0, bounds=(lo, hi),
                            maxfev=200000, x_scale="jac")
        free.update(ok=True, tau_s=[float(popt[1]), float(popt[3]), float(popt[5])],
                    c_S=[float(popt[0]) * scale, float(popt[2]) * scale,
                         float(popt[4]) * scale],
                    r2=_r2(yn, three_exp(dt_s, *popt)))
        at_bound = [i for i, k in enumerate((1, 3, 5))
                    if np.isclose(popt[k], lo[k], rtol=1e-3) or np.isclose(popt[k], hi[k], rtol=1e-3)]
        if at_bound:
            free["note"] = "tau at fit bound: " + ",".join(f"tau{i+1}" for i in at_bound)
    except Exception as e:  # pragma: no cover - reported, not raised
        free["note"] = f"fit failed: {e}"
    return free, fixed


def paired_series(runs):
    """runs: list of (cols, stim) for the pair stimuli -> series + fits."""
    pts = sorted((paired_pulse(c, s) for c, s in runs), key=lambda d: d["dt_onset_s"])
    dt = np.array([d["dt_onset_s"] for d in pts])
    dG = np.array([d["dG_S"] for d in pts])
    dGa = np.array([d["dG_alt_S"] for d in pts])
    free, fixed = fit_three_exp(dt, dG)
    free_alt, _ = fit_three_exp(dt, dGa)
    return {"dt_s": dt, "gap_s": np.array([d["gap_s"] for d in pts]),
            "dG_S": dG, "dG_alt_S": dGa,
            "tau1_ms": 1e3 * free["tau_s"][0], "tau2_ms": 1e3 * free["tau_s"][1],
            "tau3_s": free["tau_s"][2], "fit_r2": free["r2"], "fit_note": free["note"],
            "fit_c_uS": [1e6 * c for c in free["c_S"]],
            "fixed_tau_r2": fixed["r2"], "fixed_tau_c_uS": [1e6 * c for c in fixed["c_S"]],
            "dG_first_uS": 1e6 * float(dG[0]), "dG_last_uS": 1e6 * float(dG[-1]),
            "dG_alt_span_pct": 100 * float(np.ptp(dGa) / np.mean(dGa)),
            "alt_fit_note": free_alt["note"] or ("ok" if free_alt["ok"] else "")}


# --------------------------------------------------------------------------
def retention_pct(cols, stim):
    p = stim.params
    g1 = _at(cols, p["t_settle_s"])
    g2 = _at(cols, p["t_settle_s"] + p["hold_s"])
    return 100.0 * (g2 - g1) / g1


def retention(cols, stim):
    p = stim.params
    return {"pct_13h": retention_pct(cols, stim),
            "G_start_uS": 1e6 * _at(cols, p["t_settle_s"]),
            "G_end_uS": 1e6 * _at(cols, p["t_settle_s"] + p["hold_s"])}
