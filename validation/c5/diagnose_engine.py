#!/usr/bin/env python
"""C5 tier 1 diagnostic: why the engine's full read differs from the
converged twin, measured locally (no Spectre needed).

Two effects:

1. **Splitting.** `EcfetV3Array._step` (= `model_v3.step`) treats a whole
   10 ms pulse as ONE step: it decays the volatile pools over the full dt,
   then injects the new EDL with no decay inside the pulse. Replaying the
   schedule through `EcfetV3Array` with every pulse split into N back-to-back
   sub-pulses should shrink the out-of-pulse error ~N-fold if this is the
   cause.
2. **Read timing.** `peek()` during a pulse returns the post-pulse state
   (the device clock is already at the pulse end). Engine reads inside a
   pulse are therefore scored here against the converged twin at that
   pulse's end time.

Writes validation/data/c5/diagnose.json.

HISTORICAL: this diagnoses the engine of master ac92294, before the
physical-time engine fix (validation/enginefix/). Both effects are fixed
since, and the array semantics it relies on (a pulse applied at once by
``program``) no longer hold, so re-running it on later code does not
reproduce ``diagnose.json``; run it from a checkout of ac92294.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schedule as sch  # noqa: E402

from neurovat.devices import EcfetV3Array  # noqa: E402


def array_replay(meta, pulses, t_read, n_split):
    """EcfetV3Array (physical, full), each pulse as n_split sub-pulses;
    full G at t_read (only reads outside pulses are meaningful)."""
    p = sch.twin_params(meta)
    dev = EcfetV3Array(p, 1, 1, "physical", "full")
    dev.set_conductance(np.full((1, 1), meta["g_start_S"]))
    out = np.empty(len(t_read))
    j = 0
    zero = np.array([0])
    for s, w, a in sorted(pulses):
        t0 = s * sch.TICK
        while j < len(t_read) and t_read[j] < t0 - 1e-12:
            out[j] = dev.peek(t_read[j])[0, 0]
            j += 1
        h = w * sch.TICK / n_split
        for i in range(n_split):
            dev.program(zero, zero, np.array([a]), h, t0 + i * h)
        # reads inside the pulse: state at the pulse end (as the engine)
        while j < len(t_read) and t_read[j] < t0 + w * sch.TICK - 1e-12:
            out[j] = dev.peek(t_read[j])[0, 0]
            j += 1
    while j < len(t_read):
        out[j] = dev.peek(t_read[j])[0, 0]
        j += 1
    return out


def main():
    meta, per = sch.load()
    rd = np.loadtxt(sch.DATA / "engine_reads.csv", delimiter=",", skiprows=1)
    t = rd[:, 0]
    n = len(per)
    res = {"splits": {}, "read_timing": {}, "duty": {}}
    splits = (1, 10, 100)
    acc = {s: [] for s in splits}
    rt = []
    for k in range(n):
        ts, G, _, _ = sch.replay(meta, per[k], meta["t_end_s"])
        rng = G.max() - G.min()
        inp = np.zeros(len(t), bool)
        end_of = np.full(len(t), np.nan)
        for s, w, _a in per[k]:
            m = (t > s * sch.TICK - 1e-9) & (t < (s + w) * sch.TICK - 1e-9)
            inp |= m
            end_of[m] = (s + w) * sch.TICK
        ref = np.interp(t, ts, G)
        for sp in splits:
            g = rd[:, 1 + k] if sp == 1 else array_replay(meta, per[k], t, sp)
            e = (g - ref)[~inp] / rng
            acc[sp].append((100 * float(np.sqrt(np.mean(e * e))),
                            100 * float(np.max(np.abs(e)))))
        # in-pulse engine reads vs the converged twin at the pulse end
        e_end = (rd[inp, 1 + k] - np.interp(end_of[inp], ts, G)) / rng
        e_now = (rd[inp, 1 + k] - ref[inp]) / rng
        rt.append((100 * float(np.max(np.abs(e_now))), 100 * float(np.max(np.abs(e_end)))))
        res["duty"][k] = float(sum(w for _, w, _ in per[k]) * sch.TICK / meta["t_end_s"])
    for sp in splits:
        a = np.array(acc[sp])
        res["splits"][sp] = {"out_pulse_nrmse_pct_max": float(a[:, 0].max()),
                             "out_pulse_max_pct_max": float(a[:, 1].max()),
                             "out_pulse_nrmse_pct_mean": float(a[:, 0].mean())}
    a = np.array(rt)
    res["read_timing"] = {"in_pulse_max_pct_vs_now": float(a[:, 0].max()),
                          "in_pulse_max_pct_vs_pulse_end": float(a[:, 1].max())}
    d = np.array(list(res["duty"].values()))
    res["duty_summary"] = {"mean": float(d.mean()), "min": float(d.min()), "max": float(d.max())}
    (sch.DATA / "diagnose.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "duty"}, indent=1))


if __name__ == "__main__":
    main()
