"""Vectorised ECFET v3 array vs the scalar twin adapter: equivalence and
speed against array size (Gate B).

Workload per event: a write to a random 10 % of the cells (both polarities,
pulse 10 ms), then a read of a random 10 % of the columns 1 ms later (the
network's read of the spiking inputs), gaps drawn from 0.1 ms .. 20 s.
Both arrays run physical timing and the full read. Prints a table; with
``--json PATH`` also writes it.

    python benchmarks/bench_vector_twin.py [--events 200] [--json out.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from neurovat.devices import EcfetV3Array, ScalarTwinArray  # noqa: E402
from simulation_core.model_v3 import EcfetV3, V3Params     # noqa: E402


def workload(shape, n_events, seed=0):
    rng = np.random.default_rng(seed)
    n = shape[0] * shape[1]
    t, ev = 0.0, []
    for _ in range(n_events):
        t += float(rng.choice([1e-4, 2e-3, 0.02, 0.3, 20.0]))
        k = max(n // 10, 1)
        sel = rng.choice(n, k, replace=False)
        amps = rng.choice([-1.0, 1.0], k) * rng.uniform(5e-12, 3e-10, k)
        cols = rng.choice(shape[1], max(shape[1] // 10, 1), replace=False)
        ev.append((t, sel // shape[1], sel % shape[1], amps, cols))
    return ev


def run(arr, ev):
    t0 = time.perf_counter()
    reads = []
    for t, r, c, a, cols in ev:
        arr.program(r, c, a, 10e-3, t)
        reads.append(np.array(arr.read_at(t + 1e-3, cols), float))
    arr.advance(ev[-1][0] + 100.0)
    return time.perf_counter() - t0, reads, np.array(arr.conductance())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    make = lambda: EcfetV3(V3Params.paper(A_stdp=0.0))
    rows = []
    for shape in ((1, 10), (1, 100), (1, 1000), (4, 2000)):
        ev = workload(shape, a.events)
        tv, rv, gv = run(EcfetV3Array(make().p, *shape), ev)
        ts, rs, gs = run(ScalarTwinArray(make, *shape, "physical", "full"), ev)
        err = max(float(np.max(np.abs(x - y) / y)) for x, y in
                  zip(rv + [gv], rs + [gs]))
        rows.append({"shape": list(shape), "devices": shape[0] * shape[1],
                     "events": a.events, "scalar_s": ts, "vector_s": tv,
                     "speedup": ts / tv, "max_rel_err": err})
        print(f"{shape[0]}x{shape[1]:<5d} devices {rows[-1]['devices']:>5d}  "
              f"scalar {ts:8.3f} s  vector {tv:7.4f} s  speed-up "
              f"{ts / tv:7.1f}x  max rel err {err:.2e}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    main()
