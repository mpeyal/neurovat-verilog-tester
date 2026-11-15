#!/usr/bin/env python
"""C3 benchmark 3: the Masquelier stream network, wall time against the
afferent count, ideal weights and physical-time ECFET v3 synapses.

Network = the C6 ablation's configuration (validation/ablation/
run_ablation.py, unchanged): one SRM neuron, half the afferents carry a
50 ms pattern shown every 10th section (k = 10), threshold 0.25 per
afferent, seed 1, 8 presentations = 4 s of stream. Arms ``ideal`` (additive
STDP on ideal weights) and ``v3_time`` (EcfetV3Array, physical time, full
read every 1 ms, 10 ms calibrated write pulses in a 10 ms window). Only
``masquelier.run`` is timed (stream generation included; device
construction and the write calibration are not), with no logging wrappers.

``time``: NeuroVAT timings, per stream-second.

``replay``: what Spectre can do at network level, open loop. One
instrumented engine run (validation/c5/run_engine.py: the same
configuration, wrappers on) records every write pulse; one ``ecfet_v3``
instance per synapse then replays it in Spectre (devices only: no neuron,
no STDP in Spectre). Timed like benchmark 1. The engine's own reads of the
first 16 synapses are scored against Spectre (Gate A rule), so the replay
is at the engine's accuracy. A closed loop (neuron + STDP in Spectre,
validation/c5/TIER2.md) is NOT built; its cost is only an extrapolation.

    python validation/scaling/network.py time --n 64 256 1000 2000 --reps 3
    python validation/scaling/network.py replay --n 64 256 1000 --reps 3
"""
import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(ROOT), str(ROOT / "validation" / "c5"),
                str(ROOT / "validation" / "ablation")]
import machine  # noqa: E402,F401

import numpy as np  # noqa: E402

K = 10
PRESENTATIONS = 8
SEED = 1
STREAM_S = PRESENTATIONS * K * 0.05


def one(arm, n):
    import run_ablation as ra
    from neurovat.masquelier import calibrate_write, run
    P = replace(ra.base_params(argparse.Namespace(afferents=n)), seed=SEED,
                pattern_frac=1.0 / K, base_s=STREAM_S)
    dev, kw = None, {}
    if arm != "ideal":
        dev = ra.make_array(arm, 1, n, None)
        cal = calibrate_write(dev, P, ra.WIDTH, 30 * ra.I_UNIT)
        lo, hi = dev.g_min, dev.g_max
        dev.set_conductance(np.full(dev.shape, lo + P.w_init * (hi - lo)))
        kw = dict(write_window_ms=ra.WINDOW_MS, read_ms=1.0,
                  **{q: cal[q] for q in ("write_gain", "dep_gain", "pot_sign",
                                         "pulse_width")})
    t0 = time.perf_counter()
    r = run(P, device=dev, **kw)
    wall = time.perf_counter() - t0
    return wall, {"post_spikes": int(r["post_spikes"]), "writes": int(r["writes"])}


def bench_time(a):
    from v3_array import summary
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in a.n:
        for arm in a.arms:
            walls, info = [], None
            for _ in range(a.reps):
                mi = machine.info()
                w, info = one(arm, n)
                walls.append(w)
            s = summary(walls)
            rec = {"benchmark": "network_time", "arm": arm, "afferents": n,
                   "stream_s": STREAM_S, "k": K, "seed": SEED, "commit": a.commit,
                   "walls_s": walls, **s,
                   "s_per_stream_s": s["median_s"] / STREAM_S,
                   "s_per_synapse_second": s["median_s"] / (STREAM_S * n),
                   **info, "machine": mi}
            (out / f"net_{arm}_n{n}.json").write_text(json.dumps(rec, indent=1) + "\n",
                                                     encoding="utf-8")
            print(f"{arm} n={n}: {s['median_s']:.2f} s ({s['min_s']:.2f}-{s['max_s']:.2f}), "
                  f"{rec['s_per_stream_s']:.3f} s/stream-s, spikes {info['post_spikes']}, "
                  f"writes {info['writes']}, load {mi['loadavg_1_5_15']}", flush=True)
    return 0


def bench_replay(a):
    import run_engine
    import schedule as sch
    import v3_array as va
    from spectre_run import cleanup, run as sp_run
    out = Path(a.out)
    for n in a.n:
        work = out / f"replay_n{n}"
        if not (work / "schedule.csv").is_file():
            run_engine.main(["--afferents", str(n), "--seconds", str(STREAM_S),
                             "--seed", str(SEED), "--pattern-frac", str(1.0 / K),
                             "--out", str(work)])
        meta, per = sch.load(work)
        t_stop = round(meta["t_end_s"] + 0.05, 6)
        per_dev = [per[k] for k in range(n)]
        deck = va.save_deck(va.deck_text(meta, per_dev, t_stop, va.va_dir_for(work),
                                         f"open-loop replay of a {n}-afferent "
                                         "NeuroVAT training's write pulses"),
                            work / f"net_n{n}.scs")
        runs, probes = [], None
        for r in range(a.reps):
            mi = machine.info()
            res = sp_run(deck, keep_raw=(r == 0), timeout=a.timeout)
            res["machine"] = mi
            if "raw" in res:
                probes = va.read_probes(res.pop("raw"), n)
                cleanup(deck)
            runs.append(res)
            print(f"replay n={n} rep {r}: tran {res['tran_elapsed_s']} s, total "
                  f"{res['total_elapsed_s']} s, steps {res['accepted_steps']}, "
                  f"load {mi['loadavg_1_5_15']}", flush=True)
            if res["timed_out"]:
                break
        acc = None
        if probes is not None:
            rd = np.loadtxt(work / "engine_reads.csv", delimiter=",", skiprows=1,
                            usecols=range(0, 1 + min(n, va.PROBES)))
            t_sp, G_sp = probes
            res_k = []
            for k in range(G_sp.shape[0]):
                e = rd[:, 1 + k] - np.interp(rd[:, 0], t_sp, G_sp[k])
                rng = float(G_sp[k].max() - G_sp[k].min())
                nr = math.sqrt(float(np.mean(e * e))) / rng
                mx = float(np.max(np.abs(e))) / rng
                res_k.append((nr, mx))
            acc = {"probes": len(res_k),
                   "pass": sum(nr <= 0.01 and mx <= 0.03 for nr, mx in res_k),
                   "nrmse_pct_max": 100 * max(x[0] for x in res_k),
                   "max_pct_max": 100 * max(x[1] for x in res_k)}
        rec = {"benchmark": "network_replay", "afferents": n, "commit": a.commit,
               "stream_s": meta["t_end_s"], "pulses": meta["n_pulses"],
               "post_spikes": meta["post_spikes"], "runs": runs,
               "tran": va.summary([x["tran_elapsed_s"] for x in runs]),
               "total": va.summary([x["total_elapsed_s"] for x in runs]),
               "timed_out": any(x["timed_out"] for x in runs),
               "engine_read_vs_spectre": acc}
        (out / f"replay_n{n}.json").write_text(json.dumps(rec, indent=1) + "\n",
                                               encoding="utf-8")
        print(f"replay n={n}: accuracy {acc}", flush=True)
        (work / "engine_reads.csv").unlink(missing_ok=True)     # bulky
        if rec["timed_out"]:
            break
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["time", "replay"])
    ap.add_argument("--n", type=int, nargs="+", default=[64, 256, 1000, 2000])
    ap.add_argument("--arms", nargs="+", default=["ideal", "v3_time"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--commit", default="")
    ap.add_argument("--out", default=str(ROOT / "results" / "scaling" / "net"))
    a = ap.parse_args(argv)
    return bench_time(a) if a.cmd == "time" else bench_replay(a)


if __name__ == "__main__":
    sys.exit(main())
