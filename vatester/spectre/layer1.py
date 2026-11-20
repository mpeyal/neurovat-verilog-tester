"""Layer 1: twin against one Spectre result, for a two-terminal device or an ECFET.

    ref = reference(sig, device)          # {"time", <quantity>, ...} from PSF/CSV signals
    tw  = run_twin(spec, stim, dt)        # twinrun
    s   = score(tw, ref, device, stim, read_v, rule)

Compared quantity:
    device "2t":    the terminal current (voltage stimulus) or voltage (current
                    stimulus); with read plateaus also G = I/V at each read
    device "ecfet": the channel conductance G = |I(vread)| / V_read (older
                    results: 1 / V(n_rmem))
"""

import numpy as np

from . import compare as C
from .psf import find_signal

DEFAULT_SIGNALS = {"v": "n_p", "i": "X1:p", "rmem": "n_rmem"}


class SignalError(KeyError):
    pass


def _get(sig, name):
    y = find_signal(sig, name)
    if y is None:
        names = ", ".join(sorted(k for k in sig if k != "time"))
        raise SignalError(f"signal {name!r} not in the result; it holds: {names}")
    return np.asarray(y, float)


def reference(sig, device, signals=None, v_read=0.01):
    s = {**DEFAULT_SIGNALS, **(signals or {})}
    t = np.asarray(sig["time"], float)
    if device == "ecfet":
        i = find_signal(sig, "vread:p")
        if i is not None:
            return {"time": t, "G": np.abs(np.asarray(i, float)) / v_read}
        r = _get(sig, s["rmem"])
        return {"time": t, "G": 1.0 / r}
    return {"time": t, "V": _get(sig, s["v"]), "I": _get(sig, s["i"])}


def quantity(device, stim_kind):
    if device == "ecfet":
        return "G"
    return "I" if stim_kind == "voltage" else "V"


def score(tw, ref, device, stim, read_v=None, rule=None):
    key = quantity(device, stim.kind)
    m = C.waveform_metrics(tw["time"], tw[key], ref["time"], ref[key])
    r = None
    if device != "ecfet" and read_v is not None:
        reads = stim.read_times(read_v)
        r = C.read_metrics(tw, ref, reads) if len(reads) else None
    rule = rule or ("reads" if r is not None else "waveform")
    out = {"compared": {"G": "conductance", "I": "current", "V": "voltage"}[key],
           "twin_points": m["n_points"], "range": m["range"],
           "nrmse_pct": 100 * m["nrmse"], "max_err_pct": 100 * m["maxerr"],
           "rule": rule, "verdict": C.verdict(m, r, rule)}
    if r is not None:
        out.update({"n_reads": r["n_reads"], "read_g_max_pct": 100 * r["max"],
                    "read_g_mean_pct": 100 * r["mean"]})
    return out
