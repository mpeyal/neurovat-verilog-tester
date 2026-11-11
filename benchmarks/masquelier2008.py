"""Masquelier, Guyonneau & Thorpe (2008) unsupervised pattern-finding benchmark.

    python benchmarks/masquelier2008.py [--seeds 10] [--jobs 4]
        [--device softbound] [--write-window-ms 0] [--out results/masquelier]

Runs the paper's 450 s single-neuron experiment once per seed and applies its
success test on the last 150 s: hit rate > 98 %, no false alarm, mean latency
after pattern onset < 10 ms. The paper reports 96 % of 100 runs succeeding;
the Brian replication (Hathway & Goodman 2018) > 95 % at a 0.1 ms step.
Exits 1 when the success rate is below ``--min-success`` (default 0.9).

``--device`` replaces the ideal additive weights with a registered device
model (one synapse per afferent), programmed with amplitude
``--write-gain`` x the requested weight change; that is an experiment, not a
reproduction, and is never asserted.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from neurovat.masquelier import MasquelierParams, run


def one(job):
    seed, args = job
    P = replace(MasquelierParams(seed=seed), sim_dt_ms=args["sim_dt_ms"])
    device = None
    if args["device"]:
        from neurovat.devices import device_factory
        device = device_factory(args["device"])(1, P.n_afferents)
    t0 = time.perf_counter()
    r = run(P, device=device, write_window_ms=args["write_window_ms"],
            write_gain=args["write_gain"])
    r["wall_s"] = time.perf_counter() - t0
    pt, lat = r.pop("post_t_ms"), r.pop("post_latency_ms")
    fa = np.nonzero(np.isnan(lat))[0]
    r["last_false_alarm_s"] = float(pt[fa[-1]] / 1000) if fa.size else None
    r["final_weights"] = r.pop("weights").round(4).tolist()
    return seed, r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--sim-dt-ms", type=float, default=0.1)
    ap.add_argument("--device", default="")
    ap.add_argument("--write-window-ms", type=float, default=0.0)
    ap.add_argument("--write-gain", type=float, default=10.0)
    ap.add_argument("--min-success", type=float, default=0.9)
    ap.add_argument("--out", default="results/masquelier")
    a = ap.parse_args()
    args = {"device": a.device, "write_window_ms": a.write_window_ms,
            "write_gain": a.write_gain, "sim_dt_ms": a.sim_dt_ms}
    seeds = range(a.first_seed, a.first_seed + a.seeds)
    rows = {}
    with ProcessPoolExecutor(a.jobs) as pool:
        for seed, r in pool.map(one, [(s, args) for s in seeds]):
            rows[seed] = r
            print(f"seed {seed:3d}  {'PASS' if r['success'] else 'FAIL'}  "
                  f"hit {r['hit_rate_last']:.3f}  FA {r['false_alarms_last']:5d}  "
                  f"latency {r['mean_latency_last_ms']:5.1f} ms  "
                  f"potentiated {r['potentiated']:4d} "
                  f"({r['potentiated_in_pattern']} in pattern)  "
                  f"last FA {r['last_false_alarm_s']} s  "
                  f"writes {r['writes']:.3g}  {r['wall_s']:.0f} s", flush=True)
    rate = float(np.mean([r["success"] for r in rows.values()]))
    label = a.device or "ideal"
    print(f"\n{label}: {rate:.0%} of {len(rows)} runs succeed "
          f"(paper 96 % of 100; replication > 95 %)")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    name = f"{label}_w{a.write_window_ms:g}" if a.device else "ideal"
    path = out / f"{name}.json"
    path.write_text(json.dumps({"args": args, "success_rate": rate,
                                "runs": rows}, indent=1), encoding="utf-8")
    print(path)
    if not a.device and rate < a.min_success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
