"""Layer-1 waveform agreement between a twin and a Spectre reference.

    m = waveform_metrics(t_twin, y_twin, t_ref, y_ref)
        # NRMSE = rms(y_twin - y_ref) / (max y_ref - min y_ref), on the twin's
        # time points inside the common span, the reference interpolated
        # linearly; e_max = max |y_twin - y_ref| / the same range
    r = read_metrics(twin, ref, read_times)
        # conductance I/V at each read-plateau middle: relative error max / mean
    v = verdict(m, r, rule)

Pass rules (fractions):
    "waveform": NRMSE <= 1 % and e_max <= 3 % of the reference range
    "reads":    response NRMSE <= 1 %, per-read G max <= 3 % and mean <= 1 %
"""

import math

import numpy as np

NRMSE_LIMIT = 0.01
MAX_LIMIT = 0.03
READ_MAX_LIMIT = 0.03
READ_MEAN_LIMIT = 0.01


def waveform_metrics(t_tw, y_tw, t_ref, y_ref):
    t_tw, y_tw = np.asarray(t_tw, float), np.asarray(y_tw, float)
    t_ref, y_ref = np.asarray(t_ref, float), np.asarray(y_ref, float)
    lo, hi = max(t_tw[0], t_ref[0]), min(t_tw[-1], t_ref[-1])
    m = (t_tw >= lo) & (t_tw <= hi) & np.isfinite(y_tw)
    if m.sum() < 2:
        raise ValueError("twin and reference share fewer than two time points")
    yr = np.interp(t_tw[m], t_ref, y_ref)
    d = y_tw[m] - yr
    span = float(np.max(y_ref) - np.min(y_ref))
    if not span > 0:
        return {"n_points": int(m.sum()), "range": span,
                "nrmse": math.nan, "maxerr": math.nan}
    return {"n_points": int(m.sum()), "range": span,
            "nrmse": float(np.sqrt(np.mean(d * d)) / span),
            "maxerr": float(np.max(np.abs(d)) / span)}


def read_metrics(tw, ref, read_times):
    """tw/ref: dicts with time, V, I.  G = I/V at every read time."""
    rt = np.asarray(read_times, float)
    if not len(rt):
        return None

    def g(d):
        i = np.interp(rt, d["time"], d["I"])
        v = np.interp(rt, d["time"], d["V"])
        return i / v

    gt, gr = g(tw), g(ref)
    rel = np.abs(gt - gr) / np.abs(gr)
    return {"n_reads": int(len(rt)), "g_twin": gt, "g_ref": gr,
            "max": float(np.max(rel)), "mean": float(np.mean(rel))}


def verdict(m, r=None, rule="waveform"):
    if rule == "waveform":
        ok = m["nrmse"] <= NRMSE_LIMIT and m["maxerr"] <= MAX_LIMIT
    elif rule == "reads":
        if r is None:
            raise ValueError("rule 'reads' needs read plateaus (--read-v)")
        ok = (m["nrmse"] <= NRMSE_LIMIT and r["max"] <= READ_MAX_LIMIT
              and r["mean"] <= READ_MEAN_LIMIT)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return "PASS" if ok else "FAIL"
