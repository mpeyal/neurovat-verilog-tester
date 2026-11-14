#!/usr/bin/env python
"""C5 tier 1, step 1: a NeuroVAT stream-mode STDP training with ECFET v3
device synapses in physical time, with every gate write pulse recorded.

The setup is that of the C6 ablation's ``v3_time`` arm
(validation/ablation/run_ablation.py):

- one neuron, ``--afferents`` synapses (half carry the hidden pattern);
- ``neurovat.masquelier.run`` (unchanged), with the device from
  ``vector_twin_factory`` (the vectorised v3 array, self-checked against the
  scalar twin);
- physical time and the full read (G_nv + volatile pools), re-read every 1 ms;
- 10 ms pulses and a 10 ms write window (so pulses on one synapse never
  overlap);
- pulses calibrated on the retained step at the start weight;
- device STDP lock-in off (A_stdp = 0), so the network rule decides.

The device's ``program`` and ``peek`` are wrapped on the instance (the class
and the engine are untouched) to log:

- every pulse: synapse, start time on the device clock, width, amplitude;
- every read the engine made: time, full G and G_nv of every synapse.

Writes, under ``validation/data/c5/``:

- ``schedule.csv`` (synapse, t_start_s, width_s, amp_A);
- ``engine_reads.csv`` (t_s, then G_full and G_nv per synapse, in S);
- ``engine.json`` (parameters, calibration, start and final G per synapse,
  spike and write counts, wall time).

    python validation/c5/run_engine.py --afferents 16 --seconds 4 --seed 1
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402

from simulation_core.model_v3 import EcfetV3, V3Params  # noqa: E402

DATA = ROOT / "validation" / "data" / "c5"
WIDTH = 10e-3
WINDOW_MS = 10.0
I_UNIT = 50e-12


def v3_params():
    return V3Params.paper(A_stdp=0.0)


def make_device(n_pre):
    from neurovat.devices import vector_twin_factory
    fac = vector_twin_factory(lambda: EcfetV3(v3_params()), "physical", "full")
    if fac.kind != "vector":
        raise SystemExit("vector_twin_factory fell back to the scalar adapter")
    return fac(1, n_pre)


def instrument(dev):
    """Log program() and peek() calls on this instance."""
    log = {"pulses": [], "reads": []}
    program, peek = dev.program, dev.peek

    def program_logged(rows, cols, amps, width, t=0.0):
        idx = dev._flat(rows, cols)
        dev._idle(idx, float(t))                  # what program() does first
        start = dev.t_dev[idx].copy()             # = max(t, busy until)
        for c, s, a in zip(np.asarray(cols).ravel(), start,
                           np.asarray(amps, float).ravel()):
            log["pulses"].append((int(c), float(s), float(width), float(a)))
        return program(rows, cols, amps, width, t)

    def peek_logged(t):
        g = np.asarray(peek(t), float).ravel().copy()
        nv = dev.read
        dev.read = "nv"
        try:
            gnv = np.asarray(peek(t), float).ravel().copy()
        finally:
            dev.read = nv
        log["reads"].append((float(t), g, gnv))
        return peek(t)

    dev.program, dev.peek = program_logged, peek_logged
    return log


def params(a):
    from neurovat.masquelier import MasquelierParams
    n = a.afferents
    return MasquelierParams(n_afferents=n, n_pattern=n // 2,
                            threshold=a.threshold_per_aff * n, n_repeats=1,
                            spacing="even", pattern_frac=a.pattern_frac,
                            base_s=a.seconds, seed=a.seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--afferents", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pattern-frac", type=float, default=0.25)
    ap.add_argument("--threshold-per-aff", type=float, default=0.25,
                    help="threshold / afferents (the ablation uses 500/2000)")
    ap.add_argument("--out", default=str(DATA))
    a = ap.parse_args(argv)
    from neurovat.masquelier import calibrate_write, run

    P = params(a)
    dev = make_device(P.n_afferents)
    cal = calibrate_write(dev, P, WIDTH, 30 * I_UNIT)
    lo, hi = dev.g_min, dev.g_max
    g_start = lo + P.w_init * (hi - lo)
    dev.set_conductance(np.full(dev.shape, g_start))
    log = instrument(dev)
    kw = dict(write_window_ms=WINDOW_MS, read_ms=1.0,
              **{q: cal[q] for q in ("write_gain", "dep_gain", "pot_sign",
                                     "pulse_width")})
    t0 = time.perf_counter()
    r = run(P, device=dev, **kw)
    wall = time.perf_counter() - t0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = P.n_afferents
    pulses = sorted(log["pulses"], key=lambda p: (p[1], p[0]))
    with open(out / "schedule.csv", "w", newline="\n") as f:
        f.write("synapse,t_start_s,width_s,amp_A\n")
        for c, s, w, amp in pulses:
            f.write(f"{c},{s:.9f},{w:.6g},{amp:.9g}\n")
    # engine reads: keep the last read of each time (flush + tick may both read)
    reads = {}
    for t, g, gnv in log["reads"]:
        reads[round(t, 9)] = (g, gnv)
    ts = sorted(reads)
    with open(out / "engine_reads.csv", "w", newline="\n") as f:
        f.write("t_s," + ",".join(f"G{k}" for k in range(n)) + ","
                + ",".join(f"Gnv{k}" for k in range(n)) + "\n")
        for t in ts:
            g, gnv = reads[t]
            f.write(f"{t:.9f}," + ",".join(f"{v:.10g}" for v in g) + ","
                    + ",".join(f"{v:.10g}" for v in gnv) + "\n")
    g_end = dev.peek(dev.clock)
    nv_now = dev.read
    dev.read = "nv"
    gnv_end = dev.peek(dev.clock)
    dev.read = nv_now
    amps = np.array([p[3] for p in pulses])
    meta = {
        "what": "C5 tier 1 engine run (masquelier.run, EcfetV3Array physical/full)",
        "masquelier": asdict(P), "v3_params": asdict(v3_params()),
        "pulse_width_s": WIDTH, "write_window_ms": WINDOW_MS, "read_ms": 1.0,
        "calibration": {q: float(v) for q, v in cal.items()},
        "g_start_S": float(g_start), "R_start_ohm": float(1.0 / g_start),
        "t_end_s": float(dev.clock), "post_spikes": int(r["post_spikes"]),
        "writes": int(r["writes"]), "n_pulses": len(pulses),
        "n_pot": int((amps > 0).sum()), "n_dep": int((amps < 0).sum()),
        "amp_abs_pA": [float(np.abs(amps).min() * 1e12) if amps.size else 0.0,
                       float(np.abs(amps).max() * 1e12) if amps.size else 0.0],
        "pulses_per_synapse": np.bincount([p[0] for p in pulses],
                                          minlength=n).tolist(),
        "final_G_full_S": np.asarray(g_end, float).ravel().tolist(),
        "final_G_nv_S": np.asarray(gnv_end, float).ravel().tolist(),
        "final_weights": np.asarray(r["weights"], float).ravel().tolist(),
        "engine_wall_s": round(wall, 3)}
    (out / "engine.json").write_text(json.dumps(meta, indent=1) + "\n",
                                     encoding="utf-8")
    print(f"{len(pulses)} pulses ({meta['n_pot']} pot / {meta['n_dep']} dep, "
          f"|I| {meta['amp_abs_pA'][0]:.1f}..{meta['amp_abs_pA'][1]:.1f} pA), "
          f"{meta['post_spikes']} post spikes, t_end {meta['t_end_s']:.3f} s, "
          f"engine {wall:.2f} s; per synapse {meta['pulses_per_synapse']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
