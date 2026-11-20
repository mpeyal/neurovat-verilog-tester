"""Layer 2: is the reference itself converged?

Spectre runs the same stimulus at several maximum time steps; every run is
scored against the finest one with the Layer-1 metric.  The twin gets the same
test by refining its own output step (dt, dt/2, dt/4), and the twin at dt is
scored against the finest Spectre run.

    table = convergence(runs, stim, device, twin_runs)
        runs:      {maxstep: reference dict}   (layer1.reference of each result)
        twin_runs: {dt: twin dict}              (twinrun.run_twin at each dt)

A coarse Spectre step that FAILS against the finest means the reference was
not converged at that step: the twin must then be judged against the finer one.
"""

import numpy as np

from . import compare as C
from .layer1 import quantity


def read_errors(test, ref, key, reads):
    """Error of `test` against `ref` at the read times, as a fraction of the
    reference's range of `key` there; of |ref| when the range is zero (a test
    in which the state should not change).  -> (errors, "range" | "value"),
    or None if the reference is not usable (no reads, non-finite or zero)."""
    reads = np.asarray(reads, float)
    if not len(reads):
        return None
    gt = np.interp(reads, test["time"], test[key])
    gr = np.interp(reads, ref["time"], ref[key])
    if not (np.all(np.isfinite(gr)) and np.all(np.isfinite(gt))):
        return None
    span = float(np.max(gr) - np.min(gr))
    if span > 0 and span > 1e-9 * float(np.max(np.abs(gr))):
        return np.abs(gt - gr) / span, "range"
    if np.min(np.abs(gr)) <= 0:
        return None
    return np.abs(gt - gr) / np.abs(gr), "value"


def _row(kind, step, test, ref, key, reads=None):
    if reads is not None:
        r = read_errors(test, ref, key, reads)
        if r is None:
            return {"what": kind, "step_s": step, "points": int(len(reads)),
                    "nrmse_pct": float("nan"), "max_err_pct": float("nan"),
                    "mean_err_pct": float("nan"), "verdict": "INCONCLUSIVE"}
        e, scale = r
        ok = e.max() <= C.READ_MAX_LIMIT and e.mean() <= C.READ_MEAN_LIMIT
        return {"what": kind, "step_s": step, "points": int(len(e)),
                "nrmse_pct": float("nan"), "max_err_pct": 100 * float(e.max()),
                "mean_err_pct": 100 * float(e.mean()), "scale": scale,
                "verdict": "PASS" if ok else "FAIL"}
    m = C.waveform_metrics(test["time"], test[key], ref["time"], ref[key])
    return {"what": kind, "step_s": step, "points": m["n_points"],
            "nrmse_pct": 100 * m["nrmse"], "max_err_pct": 100 * m["maxerr"],
            "verdict": C.verdict(m)}


def coarsest_converged(runs, key, reads=None):
    """Coarsest Spectre step whose result agrees with the finest one (the
    finest step if none does)."""
    steps = sorted(runs)
    ok = [s for s in steps[1:]
          if _row("", s, runs[s], runs[steps[0]], key, reads)["verdict"] == "PASS"]
    return max(ok) if ok else steps[0]


def convergence(runs, stim, device, twin_runs, quantity_key=None, cross=True,
                spectre_key=None, reads=None):
    """quantity_key: the compared column (default from the device and stimulus);
    spectre_key: a different column for the Spectre-vs-Spectre rows;
    cross=False leaves out the twin-vs-Spectre rows (when the twin's quantity
    is not defined the same way at every instant); reads: compare only at
    these times with the read rule (max <= 3 %, mean <= 1 % of the range)."""
    key = quantity_key or quantity("ecfet" if device == "ecfet" else "2t", stim.kind)
    skey = spectre_key or key
    steps = sorted(runs)
    fine = runs[steps[0]]
    rows = [_row("spectre vs finest spectre", s, runs[s], fine, skey, reads)
            for s in steps[1:]]
    tdts = sorted(twin_runs)
    tfine = twin_runs[tdts[0]]
    rows += [_row("twin vs finest twin", d, twin_runs[d], tfine, key, reads)
             for d in tdts[1:]]
    if cross:
        rows.append(_row("twin vs finest spectre", tdts[-1], twin_runs[tdts[-1]], fine,
                         key, reads))
        for s in steps[1:]:
            rows.append(_row("twin vs spectre at this step", s, twin_runs[tdts[-1]],
                             runs[s], key, reads))
    passing = [s for s in steps[1:]
               if next(r for r in rows if r["what"] == "spectre vs finest spectre"
                       and r["step_s"] == s)["verdict"] == "PASS"]
    summary = {
        "finest_step_s": steps[0],
        "coarsest_converged_step_s": max(passing) if passing else steps[0],
        "unconverged_steps_s": [s for s in steps[1:] if s not in passing],
        "twin_vs_finest": next((r["verdict"] for r in rows
                                if r["what"] == "twin vs finest spectre"), "n/a"),
    }
    return rows, summary
