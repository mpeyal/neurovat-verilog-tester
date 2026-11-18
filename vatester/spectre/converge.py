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

from . import compare as C
from .layer1 import quantity


def _row(kind, step, test, ref, key):
    m = C.waveform_metrics(test["time"], test[key], ref["time"], ref[key])
    return {"what": kind, "step_s": step, "points": m["n_points"],
            "nrmse_pct": 100 * m["nrmse"], "max_err_pct": 100 * m["maxerr"],
            "verdict": C.verdict(m)}


def convergence(runs, stim, device, twin_runs):
    key = quantity(device, stim.kind)
    steps = sorted(runs)
    fine = runs[steps[0]]
    rows = [_row("spectre vs finest spectre", s, runs[s], fine, key) for s in steps[1:]]
    tdts = sorted(twin_runs)
    tfine = twin_runs[tdts[0]]
    rows += [_row("twin vs finest twin", d, twin_runs[d], tfine, key) for d in tdts[1:]]
    rows.append(_row("twin vs finest spectre", tdts[-1], twin_runs[tdts[-1]], fine, key))
    for s in steps[1:]:
        rows.append(_row("twin vs spectre at this step", s, twin_runs[tdts[-1]], runs[s], key))
    passing = [s for s in steps[1:]
               if next(r for r in rows if r["what"] == "spectre vs finest spectre"
                       and r["step_s"] == s)["verdict"] == "PASS"]
    summary = {
        "finest_step_s": steps[0],
        "coarsest_converged_step_s": max(passing) if passing else steps[0],
        "unconverged_steps_s": [s for s in steps[1:] if s not in passing],
        "twin_vs_finest": rows[len(steps) - 1 + len(tdts) - 1]["verdict"],
    }
    return rows, summary
