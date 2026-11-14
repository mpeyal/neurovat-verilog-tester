#!/usr/bin/env python
"""MemTorch VTEAM vs the NeuroVAT VTEAM twin vs the fine-step Spectre reference.

Runs only where MemTorch is installed (GPL-3, not a NeuroVAT dependency; see
validation/README.md for the venv recipe).  MemTorch is imported lazily and
only called, never copied; without it this script prints a note and exits 0.

    <venv-with-memtorch>/python validation/vteam/memtorch_compare.py
    ... --stimulus s1_vteam s2_vteam --dts 1e-9 1e-10 1e-11 1e-12 --ref ref_1ps/spectre

Same parameters (MemTorch defaults = twin defaults, checked -> param_map.csv)
and the same sampled V(t) arrays (V at each step's midpoint, held over dt):
  * MemTorch VTEAM.simulate() at fixed dt (forward Euler, state clipped to
    [0, d] every step, current from the updated state);
  * "twin exact": the closed-form twin on the SAME held samples (integration
    error 0, so it isolates the sampling of the 50 ps edges);
  * "twin Euler" (integrator=1): one Euler step + clip per dt, in MemTorch's
    order; it must reproduce MemTorch to rounding ("max |I - I_MemTorch|").
Reference: the fine-step Spectre run of verilog/vteam.va (default
validation/data/vteam/ref_1ps/spectre/<stim>.csv), state x(t) interpolated;
if it is missing, the exact twin on the PWL is used and the `reference`
column says so.  Extra rows: the twin on the exact PWL vs the reference
(dt-independent), and the window-edge gap (twin w_eps = 1e-4, the .va
default, vs w_eps = 0, MemTorch's hard clip, both exact on the PWL).

Metrics as validation/memtorch_compare.py: per-read G = I/V at each read
sample vs I_ref = V_k / R(x_ref(t_k_end)) (max, mean, last), current NRMSE
normalised by the reference current range, wall time.

Writes validation/data/vteam/memtorch/{param_map.csv, memtorch_compare.csv}
and validation/figures/vteam/memtorch_<stim>.png.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.vteam import core  # noqa: E402
from validation.vteam import stimuli as S  # noqa: E402

OUT = core.DATA / "memtorch"
NAMES = ("r_off", "r_on", "d", "k_on", "k_off", "alpha_on", "alpha_off", "v_on", "v_off",
         "x_on", "x_off")
NOTES = {"x_on": "MemTorch: initial state only (twin/.va: x_init)",
         "x_off": "MemTorch: unused; bounds are [0, d]",
         "d": "MemTorch normalises by d, not x_off - x_on (paper eq. 7)"}


def param_table(mt_defaults, P):
    rows = []
    for n in NAMES + ("time_series_resolution",):
        mv = mt_defaults.get(n)
        tv = getattr(P, n, None)
        same = (mv is not None and tv is not None and
                abs(float(mv) - float(tv)) <= 1e-12 * max(abs(float(mv)), abs(float(tv)), 1e-300))
        rows.append({"memtorch": n, "memtorch_default": repr(mv), "twin_va": n if tv is not None
                     else "-", "twin_default": repr(tv), "match": "yes" if same else
                     ("n/a" if tv is None else "NO"),
                     "note": NOTES.get(n, "MemTorch default 1e-10 s; the .va has no step"
                                          if n == "time_series_resolution" else "")})
    return rows


def sample(stim, dt):
    n = int(round(stim.t_stop / dt))
    return stim.v((np.arange(n) + 0.5) * dt)


def read_index(stim, dt, n):
    return np.clip(np.round(stim.read_times / dt - 0.5).astype(int), 0, n - 1)


def run_memtorch(memtorch, P, x0, V, dt):
    kw = {n: getattr(P, n) for n in NAMES}
    m = memtorch.bh.memristor.VTEAM(time_series_resolution=dt, **kw)
    m.x = x0
    old = np.geterr()
    try:
        t0 = time.perf_counter()
        i = m.simulate(list(V), return_current=True)
        wall = time.perf_counter() - t0
    finally:
        np.seterr(**old)                     # simulate() sets all="raise"
    return np.asarray(i, dtype=float), wall


def run_twin_samples(stim, V, dt, **over):
    d = core.new_twin(stim, **over)
    i = np.empty(len(V))
    t0 = time.perf_counter()
    for k, v in enumerate(V):
        d.step(k * dt, dt, v)
        i[k] = d.current(v)
    return i, time.perf_counter() - t0


def twin_pwl(stim, out_dt=1e-12, **over):
    """Exact twin on the PWL, x recorded every out_dt."""
    d = core.new_twin(stim, **over)
    n = int(round(stim.t_stop / out_dt))
    ts = np.arange(n + 1) * out_dt
    v = stim.v(ts)
    x = np.empty(n + 1)
    x[0] = d.x_clamped
    t0 = time.perf_counter()
    for k in range(n):
        d.step_linear(ts[k], out_dt, v[k], v[k + 1])
        x[k + 1] = d.x_clamped
    return ts, x, time.perf_counter() - t0


def metrics(stim, dt, I, V, ts_ref, x_ref, P):
    n = len(V)
    xr = np.interp((np.arange(n) + 1) * dt, ts_ref, x_ref)
    i_ref = V / (P.r_off * xr / P.d + P.r_on * (1.0 - xr / P.d))
    rng = float(np.max(i_ref) - np.min(i_ref))
    out = {"i_nrmse_pct": 100 * float(np.sqrt(np.mean((I - i_ref) ** 2)) / rng)}
    g_tool = g_ref = np.array([])
    if len(stim.reads):
        k = read_index(stim, dt, n)
        g_tool, g_ref = I[k] / V[k], i_ref[k] / V[k]
        rel = np.abs(g_tool - g_ref) / g_ref
        out.update(g_max_pct=100 * float(rel.max()), g_mean_pct=100 * float(rel.mean()),
                   g_last_pct=100 * float(rel[-1]))
    else:
        out.update(g_max_pct=math.nan, g_mean_pct=math.nan, g_last_pct=math.nan)
    return out, g_tool, g_ref


def state_metrics(stim, ts, x, ts_ref, x_ref, P):
    """Same metrics for a state trajectory x(ts) (rows without sampling)."""
    reads = stim.read_times
    xs = np.interp(ts, ts_ref, x_ref)
    r = lambda xx: P.r_off * xx / P.d + P.r_on * (1.0 - xx / P.d)  # noqa: E731
    v = stim.v(ts)
    i, i_ref = v / r(x), v / r(xs)
    out = {"i_nrmse_pct": 100 * float(np.sqrt(np.mean((i - i_ref) ** 2)) /
                                      (np.max(i_ref) - np.min(i_ref)))}
    if len(reads):
        g, g0 = 1 / r(np.interp(reads, ts, x)), 1 / r(np.interp(reads, ts_ref, x_ref))
        rel = np.abs(g - g0) / g0
        out.update(g_max_pct=100 * float(rel.max()), g_mean_pct=100 * float(rel.mean()),
                   g_last_pct=100 * float(rel[-1]))
    else:
        out.update(g_max_pct=math.nan, g_mean_pct=math.nan, g_last_pct=math.nan)
    return out


def figure(path, stim, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, g, style in curves:
        ax.plot(np.arange(1, len(g) + 1), g * 1e3, style, label=label, ms=4, lw=1)
    ax.set_xlabel("read # (after each write pulse)")
    ax.set_ylabel("G_read (mS)")
    ax.set_title(f"{stim.name}: per-pulse conductance, MemTorch vs NeuroVAT twin vs Spectre",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stimulus", nargs="*", default=["s1_vteam", "s2_vteam", "sine_memtorch"])
    ap.add_argument("--dts", nargs="*", type=float, default=[1e-9, 1e-10, 1e-11, 1e-12])
    ap.add_argument("--ref", default="ref_1ps/spectre",
                    help="reference set under validation/data/vteam/ (Spectre fine step)")
    args = ap.parse_args(argv)
    try:
        import memtorch  # noqa: F401  (lazy: GPL-3, optional)
        import memtorch.bh.memristor  # noqa: F401
    except Exception as e:                       # noqa: BLE001
        print(f"MemTorch not available ({type(e).__name__}: {e}); comparison skipped.")
        return 0
    import inspect
    sig = inspect.signature(memtorch.bh.memristor.VTEAM.__init__)
    mt_def = {k: v.default for k, v in sig.parameters.items()
              if v.default is not inspect.Parameter.empty}
    P0 = core.params()
    pmap = param_table(mt_def, P0)
    write(OUT / "param_map.csv", pmap)
    print(f"memtorch {getattr(memtorch, '__version__', '?')}: parameter map "
          f"({sum(r['match'] == 'NO' for r in pmap)} mismatches) -> param_map.csv")
    rows = []
    for name in args.stimulus:
        stim = S.BY_NAME[name]
        P = core.params(stim)
        ref = core.load(args.ref, name)
        if ref is not None:
            ts_ref, x_ref, ref_label = ref["t_s"], ref["x_m"], args.ref
        else:
            ts_ref, x_ref, _ = twin_pwl(stim)
            ref_label = "twin exact PWL (no Spectre reference found)"
        curves = []
        g_ref_last = None
        for dt in args.dts:
            V = sample(stim, dt)
            I_mt, w_mt = run_memtorch(memtorch, P, P.x_init, V, dt)
            runs = [("MemTorch", I_mt, w_mt),
                    ("twin exact", *run_twin_samples(stim, V, dt)),
                    ("twin Euler", *run_twin_samples(stim, V, dt, integrator=1))]
            for tool, I, wall in runs:
                m, g_tool, g_ref = metrics(stim, dt, I, V, ts_ref, x_ref, P)
                dev = float(np.max(np.abs(I - I_mt)) / np.max(np.abs(I_mt)))
                rows.append({"stimulus": name, "reference": ref_label, "tool": tool,
                             "dt_s": dt, "n_steps": len(V), **m,
                             "max_rel_dev_vs_memtorch": dev, "wall_s": wall,
                             "us_per_step": 1e6 * wall / len(V)})
                if len(g_tool) and dt in (args.dts[0], args.dts[-1]):
                    curves.append((f"{tool} dt={dt:g}", g_tool,
                                   {"MemTorch": "o-", "twin exact": "s--",
                                    "twin Euler": "x:"}[tool]))
                g_ref_last = g_ref
        # dt-independent rows: the twin on the exact PWL, and the window edge
        ts_t, x_t, w_t = twin_pwl(stim)
        rows.append({"stimulus": name, "reference": ref_label, "tool": "twin exact PWL",
                     "dt_s": 0.0, "n_steps": len(ts_t) - 1,
                     **state_metrics(stim, ts_t, x_t, ts_ref, x_ref, P),
                     "max_rel_dev_vs_memtorch": math.nan, "wall_s": w_t,
                     "us_per_step": math.nan})
        ts_h, x_h, _ = twin_pwl(stim, w_eps=0.0)
        rows.append({"stimulus": name, "reference": "twin exact PWL, w_eps = 1e-4",
                     "tool": "window edge: twin w_eps = 0 (hard clip)", "dt_s": 0.0,
                     "n_steps": len(ts_h) - 1,
                     **state_metrics(stim, ts_h, x_h, ts_t, x_t, P),
                     "max_rel_dev_vs_memtorch": math.nan, "wall_s": math.nan,
                     "us_per_step": math.nan})
        if g_ref_last is not None and len(g_ref_last):
            curves.append((f"reference ({ref_label})", g_ref_last, "k-"))
            figure(core.FIGURES / f"memtorch_{name}.png", stim, curves)
    write(OUT / "memtorch_compare.csv", rows)
    f = lambda v, s: "-" if v != v else format(v, s)  # noqa: E731
    print(f"{'stimulus':14s} {'tool':40s} {'dt':>7s} {'Gmax%':>8s} {'Gmean%':>8s} "
          f"{'I NRMSE%':>9s} {'dev MT':>8s} {'us/step':>8s}")
    for r in rows:
        print(f"{r['stimulus']:14s} {r['tool']:40s} {r['dt_s']:7.0e} {f(r['g_max_pct'], '8.4f')} "
              f"{f(r['g_mean_pct'], '8.4f')} {f(r['i_nrmse_pct'], '9.4f')} "
              f"{f(r['max_rel_dev_vs_memtorch'], '8.1e')} {f(r['us_per_step'], '8.2f')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
