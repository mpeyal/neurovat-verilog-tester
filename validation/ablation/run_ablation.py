"""C6 ablation: does the synapse's TIME dynamics change what STDP learns?

One Masquelier-style network (one neuron, one hidden 50 ms pattern on half
the afferents, Poisson background, neurovat.masquelier.run), the same streams
(seeded per seed), four synapse arms:

  ideal      the paper's additive STDP on ideal weights in [0, 1]
  neurosim   NeuroSim-style per-pulse nonlinear curve, A_p / A_d fitted to
             the v3 twin's retained LTP / LTD (validation/ablation/
             neurosim_curve.py); no time, read = stored conductance
  v3_static  ECFET v3 twin (vectorised, == the scalar twin), stepped only
             during write pulses, read = G_nv (the pre-Gate-B behaviour;
             control: same write law as v3_time, no time)
  v3_time    ECFET v3 twin in physical time: idle stepping between writes
             (volatile EDL pools tau1 = w^2/2D = 22.2 ms, tau2 = l^2/2D =
             312.5 ms, tau3 = 19 s; retention tau_ret = 1.05e6 s) and the
             FULL read (G_nv + volatile pools), refreshed every 1 ms

Every device arm: pulse width 10 ms, write window 10 ms (each synapse gets
at most one pulse per 10 ms, so its pulses never overlap), pulses
calibrated on the RETAINED step so one pulse at the start weight = one STDP
step (a_plus / a_minus); v3 device STDP lock-in off (A_stdp = 0, as the GUI
does: the network rule decides). The ideal arm writes every pairing at once
(no window), as the paper's rule does.

Sweep exposing time: patterns evenly spaced every k-th 50 ms section
(``spacing="even"``, ``pattern_frac = 1/k``), so the silent-of-pattern gap
between presentations is (k-1) x 50 ms, against the v3 pool constants
22 ms / 312 ms / 19 s. The number of presentations is held fixed
(``--presentations``), so the run length is presentations x k x 50 ms.

Metrics (last third of each run, neurovat.masquelier._score): success (hit
> 98 %, no false alarm, latency < 10 ms), hit rate, false alarms per s,
mean latency, and the weight distribution (read weights, and the stored
weights for devices): pattern afferents with w > 0.9, background
afferents with w < 0.1, bimodal share.

    python validation/ablation/run_ablation.py --seeds 10 --k 2 4 10 \
        --presentations 200 --afferents 1000 --jobs 12 --out validation/ablation/out/first
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).parent)]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ARMS = ("ideal", "neurosim", "v3_static", "v3_time")
WIDTH = 10e-3
WINDOW_MS = 10.0
I_UNIT = 50e-12


def v3_params():
    from simulation_core.model_v3 import V3Params
    return V3Params.paper(A_stdp=0.0)


# Diagnostic arms (attribution only, not part of the four-arm ablation): the
# v3 device evolves exactly as in v3_time (physical time, same state), but
# the neuron reads only part of it - G_nv alone (v3_time_nv: what is left
# of time is retention), or G_nv + ONE volatile pool (v3_time_p1/p2/p3:
# the 22 ms / 312 ms / 19 s pool). The model physics and parameters are
# unchanged; only the read-out differs.
DIAG = {"v3_time_nv": (), "v3_time_p1": (0,), "v3_time_p2": (1,),
        "v3_time_p3": (2,)}
# gain control: v3_static with every pulse scaled by 1 + c3 kappa/(1-kappa)
# (~1.36): the extra step the 19 s pool adds to the read right after a write,
# as if it were stored - "is the 19 s effect just a larger learning rate?"
GAIN_ARMS = ("v3_static_gain",)


def pool3_gain():
    p = v3_params()
    return 1.0 + p.pool_weights[2] * p.kappa_v / (1.0 - p.kappa_v)


def _pool_read_array(pools_read, n_post, n_pre):
    from neurovat.devices import EcfetV3Array

    class PoolRead(EcfetV3Array):
        def peek(self, t):
            p = self.p
            self._pulses_to(float(t))      # in-flight pulses up to t (causal)
            dt = np.maximum(float(t) - self.t_dev, 0.0)
            x = p.x_eq + (self.x - p.x_eq) * np.exp(-dt / p.tau_ret)
            tot = self._G_of_x(x) + self.lock
            for i in pools_read:
                tot = tot + self.pools[i] * np.exp(-dt / self._taus[i])
            return np.minimum(np.maximum(tot, p.Gmin),
                              p.Gmax).reshape(self.shape)

        def nominal(self):
            return _pool_read_array(pools_read, 1, 1)
    return PoolRead(v3_params(), n_post, n_pre, "physical", "full")


def make_array(arm, n_post, n_pre, fit):
    from neurovat.devices import EcfetV3Array
    from neurosim_curve import array_from_fit
    if arm in DIAG:
        return _pool_read_array(DIAG[arm], n_post, n_pre)
    if arm == "neurosim":
        return array_from_fit(fit, n_post, n_pre)
    if arm in ("v3_static", "v3_static_gain"):
        return EcfetV3Array(v3_params(), n_post, n_pre, "serial", "nv")
    if arm == "v3_time":
        return EcfetV3Array(v3_params(), n_post, n_pre, "physical", "full")
    raise ValueError(arm)


def base_params(a):
    from neurovat.masquelier import MasquelierParams
    n = a.afferents
    return MasquelierParams(n_afferents=n, n_pattern=n // 2,
                            threshold=500.0 * n / 2000.0, n_repeats=1,
                            spacing="even")


def one_job(job):
    """(arm, seed, k, presentations, afferents, read_ms, fit) -> metrics."""
    from neurovat.masquelier import calibrate_write, run
    arm, seed, k, n_pres, n_aff, read_ms, fit = job
    ns = argparse.Namespace(afferents=n_aff)
    P = replace(base_params(ns), seed=int(seed), pattern_frac=1.0 / k,
                base_s=n_pres * k * 0.05)
    t0 = time.perf_counter()
    kw = {}
    dev = None
    cal = None
    if arm != "ideal":
        dev = make_array(arm, 1, P.n_afferents, fit)
        cal = calibrate_write(dev, P, WIDTH, 30 * I_UNIT)
        lo, hi = dev.g_min, dev.g_max
        dev.set_conductance(np.full(dev.shape, lo + P.w_init * (hi - lo)))
        kw = dict(write_window_ms=WINDOW_MS, read_ms=read_ms,
                  **{q: cal[q] for q in ("write_gain", "dep_gain", "pot_sign",
                                         "pulse_width")})
        if arm in GAIN_ARMS:
            kw["write_gain"] *= pool3_gain()
            kw["dep_gain"] *= pool3_gain()
    r = run(P, device=dev, **kw)
    w = np.asarray(r["weights"], float)
    ws = np.asarray(r.get("retained_weights", w), float)
    npat = P.n_pattern

    def dist(x):
        return {"pat_hi": float((x[:npat] > 0.9).mean()),
                "bg_lo": float((x[npat:] < 0.1).mean()),
                "bimodal": float(((x > 0.9) | (x < 0.1)).mean()),
                "mean_pat": float(x[:npat].mean()),
                "mean_bg": float(x[npat:].mean()),
                "hist": np.histogram(x, bins=10, range=(0, 1))[0].tolist()}
    pools = None
    if arm == "v3_time" or arm in DIAG:
        # the read - stored difference at the end of the run, split by pool
        # (weight units): which time constant carries it
        span = dev.g_max - dev.g_min
        pools = {name: [float(dev.pools[i][:npat].mean() / span),
                        float(dev.pools[i][npat:].mean() / span)]
                 for i, name in enumerate(("tau1_22ms", "tau2_312ms",
                                           "tau3_19s"))}
    return {"arm": arm, "seed": int(seed), "k": int(k), "pools_end": pools,
            "gap_ms": (k - 1) * P.pattern_ms, "seconds": P.base_s,
            "success": bool(r["success"]), "hit_rate": r["hit_rate_last"],
            "false_alarms": int(r["false_alarms_last"]),
            "fa_per_s": r["false_alarms_per_s"],
            "latency_ms": (None if not math.isfinite(r["mean_latency_last_ms"])
                           else r["mean_latency_last_ms"]),
            "post_spikes": r["post_spikes"], "writes": r["writes"],
            "w_read": dist(w), "w_stored": dist(ws),
            "calibration": None if cal is None else {
                q: float(v) for q, v in cal.items()},
            "wall_s": time.perf_counter() - t0}


def summarise(rows):
    """Per (k, arm): mean / sd / n of each metric."""
    out = {}
    for r in rows:
        out.setdefault((r["k"], r["arm"]), []).append(r)
    table = []
    for (k, arm), rs in sorted(out.items(), key=lambda x: (x[0][0],
                                                          (ARMS + tuple(DIAG) + GAIN_ARMS).index(x[0][1]))):
        def ms(vals):
            v = np.array([x for x in vals if x is not None], float)
            return (float(v.mean()) if v.size else float("nan"),
                    float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size))
        table.append({
            "k": k, "gap_ms": rs[0]["gap_ms"], "arm": arm, "n": len(rs),
            "success": sum(r["success"] for r in rs),
            "hit": ms(r["hit_rate"] for r in rs),
            "fa_per_s": ms(r["fa_per_s"] for r in rs),
            "latency": ms(r["latency_ms"] for r in rs),
            "pat_hi_read": ms(r["w_read"]["pat_hi"] for r in rs),
            "bg_lo_read": ms(r["w_read"]["bg_lo"] for r in rs),
            "pat_hi_stored": ms(r["w_stored"]["pat_hi"] for r in rs),
            "bg_lo_stored": ms(r["w_stored"]["bg_lo"] for r in rs),
            "mean_bg_read": ms(r["w_read"]["mean_bg"] for r in rs),
            "mean_bg_stored": ms(r["w_stored"]["mean_bg"] for r in rs),
            "wall_s": ms(r["wall_s"] for r in rs)})
    return table


def markdown(table, meta):
    def f(m, scale=1.0, d=1):
        mu, sd, n = m
        if not math.isfinite(mu):
            return "-"
        return f"{mu * scale:.{d}f} ± {sd * scale:.{d}f}"
    lines = [
        f"Network: {meta['afferents']} afferents ({meta['afferents'] // 2} in "
        f"the pattern), 1 neuron, Masquelier 2008 rule and neuron; "
        f"{meta['presentations']} evenly spaced presentations per run; "
        f"{meta['seeds']} seeds (streams shared by the arms); scored on the "
        "last third. Device arms: 10 ms pulses, 10 ms write window, "
        "calibrated on the retained step. mean ± sd over seeds; "
        "success = runs meeting hit > 98 %, no false alarm, latency < 10 ms.",
        "",
        "| gap (ms) | arm | success | hit % | false alarms /s | latency ms "
        "| pattern w>0.9 % (read / stored) | background w<0.1 % (read / "
        "stored) | mean background w (read / stored) |",
        "|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        lines.append(
            f"| {t['gap_ms']:g} | {t['arm']} | {t['success']}/{t['n']} | "
            f"{f(t['hit'], 100)} | {f(t['fa_per_s'], 1, 2)} | "
            f"{f(t['latency'])} | {f(t['pat_hi_read'], 100)} / "
            f"{f(t['pat_hi_stored'], 100)} | {f(t['bg_lo_read'], 100)} / "
            f"{f(t['bg_lo_stored'], 100)} | {f(t['mean_bg_read'], 1, 3)} / "
            f"{f(t['mean_bg_stored'], 1, 3)} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 10])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--presentations", type=int, default=200)
    ap.add_argument("--afferents", type=int, default=1000)
    ap.add_argument("--read-ms", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=max((os.cpu_count() or 2) - 2, 1))
    ap.add_argument("--out", default=str(Path(__file__).parent / "out" / "run"))
    a = ap.parse_args(argv)
    from neurosim_curve import fit_v3
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fit = fit_v3(v3_params())
    print(f"NeuroSim fit to v3 retained curves: A_p/Pmax "
          f"{fit['ltp']['A_over_Pmax']:.3f} (rmse "
          f"{100 * fit['ltp']['rmse_frac_range']:.2f} % of range, max "
          f"{100 * fit['ltp']['max_err_frac_range']:.2f} %), A_d/Pmax "
          f"{fit['ltd']['A_over_Pmax']:.3f} (rmse "
          f"{100 * fit['ltd']['rmse_frac_range']:.2f} %, max "
          f"{100 * fit['ltd']['max_err_frac_range']:.2f} %)", flush=True)
    jobs = [(arm, a.seed0 + s, k, a.presentations, a.afferents, a.read_ms,
             fit) for k in a.k for s in range(a.seeds) for arm in a.arms]
    t0 = time.perf_counter()
    rows = []
    path = out / "runs.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        if a.jobs > 1:
            from multiprocessing import Pool
            with Pool(a.jobs) as pool:
                for r in pool.imap_unordered(one_job, jobs):
                    rows.append(r)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    print(f"[{len(rows)}/{len(jobs)}] {r['arm']:9s} k={r['k']} "
                          f"seed {r['seed']}: success {r['success']} hit "
                          f"{r['hit_rate']:.2f} fa/s {r['fa_per_s']:.2f} "
                          f"({r['wall_s']:.0f} s)", flush=True)
        else:
            for j in jobs:
                r = one_job(j)
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                print(f"[{len(rows)}/{len(jobs)}] {r['arm']} k={r['k']} "
                      f"seed {r['seed']}: {r['success']}", flush=True)
    table = summarise(rows)
    meta = {"seeds": a.seeds, "seed0": a.seed0, "k": a.k, "arms": a.arms,
            "presentations": a.presentations, "afferents": a.afferents,
            "read_ms": a.read_ms, "pulse_width_s": WIDTH,
            "write_window_ms": WINDOW_MS, "neurosim_fit": fit,
            "v3_params": asdict(v3_params()),
            "v3_taus_s": list(v3_params().taus),
            "masquelier": asdict(base_params(a)),
            "wall_s": time.perf_counter() - t0}
    (out / "summary.json").write_text(json.dumps(
        {"meta": meta, "table": table}, indent=1), encoding="utf-8")
    md = markdown(table, meta)
    (out / "table.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"wall {meta['wall_s']:.0f} s -> {out}")
    return table


if __name__ == "__main__":
    main()
