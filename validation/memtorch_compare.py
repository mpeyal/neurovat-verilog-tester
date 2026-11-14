#!/usr/bin/env python
"""MemTorch Stanford_PKU vs the NeuroVAT Stanford-PKU twin (stanford-1 S1).

Runs only where MemTorch is installed (it is GPL-3 and fragile to install, so
it is NOT a NeuroVAT dependency; see validation/README.md for the recipe).
MemTorch is imported lazily and only called, never copied; without it this
script prints a note and exits 0.

    <venv-with-memtorch>/python validation/memtorch_compare.py
    ... --stimulus s1_pm1p3 --dts 1e-9 1e-10 1e-11

What is compared (same parameters, same sampled V(t) arrays):
  * MemTorch Stanford_PKU.simulate() at fixed dt (forward Euler; T from the
    PREVIOUS step's current; gap clamped every step);
  * the twin at the same dt and the same samples (V held over each dt):
      - "twin adaptive" (default integrator: exact ODE inside each dt),
      - "twin Euler"    (integrator=1: one Euler step per dt, same-step T);
  * a fine reference: the twin's adaptive solution on the exact PWL.
The twin runs in MemTorch's reading for this comparison (clamp_state=1 and
q = 1.6e-19, MemTorch's hard-coded charge), so only numerics differ.  The
.va-semantics twin (idt wind-up, q = `P_Q) is reported separately against the
same reference ("semantic gap").

Metrics: per-read conductance G = I/V at each 0.1 V read sample (relative
error vs the reference, max and mean over the reads), current NRMSE
(each tool's current at its applied sample voltage vs I(V_k, gap_ref(t_k)),
normalised by the reference current range), and wall time.

Writes validation/data/stanford/memtorch/{param_map.csv, memtorch_compare.csv}
and validation/figures/stanford/memtorch_<stim>.png.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.stanford import core  # noqa: E402
from validation.stanford import stimuli as S  # noqa: E402

OUT = core.DATA / "memtorch"
Q_MEMTORCH = 1.6e-19          # hard-coded in memtorch/bh/memristor/Stanford_PKU.py


def r_for_gap(P, gap):
    """Resistance at V_read that makes MemTorch derive exactly this gap."""
    return P.V_read / (P.I0 * math.exp(-gap / P.g0) * math.sinh(P.V_read / P.V0))


# MemTorch Stanford_PKU argument -> (.va / twin parameter, note)
PARAM_MAP = [
    ("time_series_resolution", "time_step", "MemTorch default 1e-4 s; .va $bound_step 1e-9 s"),
    ("r_on", "gap_min", "MemTorch DERIVES gap_min = g_0 ln(I_0 sinh(Vr/V_0) r_on / Vr)"),
    ("r_off", "gap_max", "MemTorch DERIVES gap_max the same way from r_off"),
    ("gap_init", "gap_ini", "MemTorch clamps the initial gap; the .va idt starts at gap_ini unclamped"),
    ("g_0", "g0", ""), ("V_0", "V0", ""), ("I_0", "I0", ""),
    ("read_voltage", "V_read", "twin-only read definition of .G"),
    ("T_init", "T_ini", ""), ("R_th", "Rth", ""),
    ("gamma_init", "gamma0", ".va hard-codes gamma_ini = 16 for V < 0; MemTorch uses gamma_init for both"),
    ("beta", "beta", ""), ("t_ox", "tox", ""), ("F_min", "F_min", ""),
    ("vel_0", "Vel0", ""), ("E_a", "Ea", ""), ("a_0", "a0", ""),
    ("delta_g_init", "deltaGap0", ""), ("model_switch", "model_switch", ""),
    ("T_crit", "T_crit", ""), ("T_smth", "T_smth", ""),
    ("(q = 1.6e-19 hard-coded)", "q", ".va: `P_Q (1.602176462e-19 in the Accellera constants.vams)"),
    ("(k_b = 1.3806503e-23 hard-coded)", "kb", ".va: `P_K"),
]


def param_table(mt_defaults, P):
    rows = []
    va = {f: getattr(P, f) for f in P.__dataclass_fields__}
    for mt_name, va_name, note in PARAM_MAP:
        mv = mt_defaults.get(mt_name)
        if mt_name.startswith("(q"):
            mv = Q_MEMTORCH
        elif mt_name.startswith("(k_b"):
            mv = 1.3806503e-23
        vv = va.get(va_name)
        if mt_name in ("r_on", "r_off"):
            g = P.g0 * math.log(P.I0 * math.sinh(P.V_read / P.V0) * mv / P.V_read)
            same = abs(g - vv) / vv < 1e-9
            disp = f"{mv:g} ohm -> {g:.6e} m"
        else:
            same = (mv is not None and vv is not None and
                    abs(float(mv) - float(vv)) <= 1e-9 * max(abs(float(mv)), abs(float(vv)), 1e-300))
            disp = f"{mv!r}"
        rows.append({"memtorch": mt_name, "memtorch_default": disp, "va_param": va_name,
                     "va_default": f"{vv!r}", "match": "yes" if same else "NO", "note": note})
    return rows


# --------------------------------------------------------------------------
def sample(stim, dt):
    n = int(round(stim.t_stop / dt))
    return (np.arange(n) + 0.5) * dt, stim.v((np.arange(n) + 0.5) * dt)


def read_index(stim, dt, n):
    k = np.round(stim.read_times / dt - 0.5).astype(int)
    return np.clip(k, 0, n - 1)


def run_memtorch(memtorch, kw, gap_min, gap_max, V, dt):
    m = memtorch.bh.memristor.Stanford_PKU(time_series_resolution=dt, **kw)
    m.gap_min, m.gap_max = gap_min, gap_max      # exact .va bounds
    m.gap = max(min(kw["gap_init"], gap_max), gap_min)
    old = np.geterr()
    try:
        t0 = time.perf_counter()
        i = m.simulate(list(V), return_current=True)
        wall = time.perf_counter() - t0
    finally:
        np.seterr(**old)                         # simulate() sets all="raise"
    return np.asarray(i, dtype=float), wall


def run_twin_samples(V, dt, **over):
    d = core.new_twin(**over)
    i = np.empty(len(V))
    t0 = time.perf_counter()
    for k, v in enumerate(V):
        d.step(k * dt, dt, v)
        i[k] = d.current(v)
    return i, time.perf_counter() - t0


def reference(stim, out_dt=10e-12, **over):
    """Adaptive twin on the exact PWL, gap recorded every out_dt (all PWL
    corners are multiples of 10 ps)."""
    d = core.new_twin(stim, **over)
    n = int(round(stim.t_stop / out_dt))
    ts = np.arange(n + 1) * out_dt
    v = stim.v(ts)
    gap = np.empty(n + 1)
    gap[0] = d.gap
    t0 = time.perf_counter()
    for k in range(n):
        d.step_linear(ts[k], out_dt, v[k], v[k + 1])
        gap[k + 1] = d.gap
    return ts, gap, time.perf_counter() - t0


def metrics(stim, dt, I, V, ts_ref, gap_ref, P):
    n = len(V)
    t_end = (np.arange(n) + 1) * dt
    g_ref_t = np.interp(t_end, ts_ref, gap_ref)
    i_ref = P.I0 * np.exp(-g_ref_t / P.g0) * np.sinh(V / P.V0)
    rng = float(np.max(i_ref) - np.min(i_ref))
    nrmse = float(np.sqrt(np.mean((I - i_ref) ** 2)) / rng)
    k = read_index(stim, dt, n)
    g_tool = I[k] / V[k]
    g_ref = i_ref[k] / V[k]
    rel = np.abs(g_tool - g_ref) / g_ref
    return {"g_max_pct": 100 * float(rel.max()), "g_mean_pct": 100 * float(rel.mean()),
            "g_last_pct": 100 * float(rel[-1]), "i_nrmse_pct": 100 * nrmse}, g_tool, g_ref


def figure(path, stim, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, g, style in curves:
        ax.semilogy(np.arange(1, len(g) + 1), g * 1e6, style, label=label, ms=4, lw=1)
    ax.set_xlabel("read # (after each write pulse)")
    ax.set_ylabel("G_read (uS)")
    ax.set_title(f"{stim.name}: per-pulse conductance, MemTorch vs NeuroVAT twin", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stimulus", nargs="*", default=["s1_pm1p3", "s1_pm1p4"])
    ap.add_argument("--dts", nargs="*", type=float, default=[1e-9, 1e-10, 1e-11])
    ap.add_argument("--memtorch-ref-dt", type=float, default=1e-12,
                    help="extra MemTorch run checking its own convergence (0 = skip)")
    args = ap.parse_args(argv)
    try:
        import memtorch  # noqa: F401  (lazy: GPL-3, optional)
        import memtorch.bh.memristor  # noqa: F401
    except Exception as e:                           # noqa: BLE001
        print(f"MemTorch not available ({type(e).__name__}: {e}); comparison skipped.")
        return 0
    import inspect
    sig = inspect.signature(memtorch.bh.memristor.Stanford_PKU.__init__)
    mt_def = {k: v.default for k, v in sig.parameters.items()
              if v.default is not inspect.Parameter.empty}
    P = core.twin_module().StanfordPKUParams()
    pmap = param_table(mt_def, P)
    write(OUT / "param_map.csv", pmap)
    print(f"memtorch {getattr(memtorch, '__version__', '?')}: parameter map "
          f"({sum(r['match'] == 'NO' for r in pmap)} mismatches) -> param_map.csv")
    for r in pmap:
        if r["match"] == "NO":
            print(f"  {r['memtorch']:30s} {r['memtorch_default']:28s} vs {r['va_param']:12s} "
                  f"{r['va_default']:18s} {r['note']}")

    kw = dict(r_on=r_for_gap(P, P.gap_min), r_off=r_for_gap(P, P.gap_max),
              gap_init=P.gap_ini, g_0=P.g0, V_0=P.V0, I_0=P.I0, read_voltage=P.V_read,
              T_init=P.T_ini, R_th=P.Rth, gamma_init=P.gamma0, beta=P.beta, t_ox=P.tox,
              F_min=P.F_min, vel_0=P.Vel0, E_a=P.Ea, a_0=P.a0, delta_g_init=P.deltaGap0,
              model_switch=0, T_crit=P.T_crit, T_smth=P.T_smth)
    matched = dict(clamp_state=1, q=Q_MEMTORCH)
    rows = []
    for name in args.stimulus:
        stim = S.BY_NAME[name]
        ts_ref, gap_ref, wall_ref = reference(stim, **matched)
        Pm = core.twin_module().StanfordPKUParams(**matched)
        curves = []
        k_ref = None
        for dt in args.dts:
            t_mid, V = sample(stim, dt)
            runs = [("MemTorch", *run_memtorch(memtorch, kw, P.gap_min, P.gap_max, V, dt)),
                    ("twin adaptive", *run_twin_samples(V, dt, **matched)),
                    ("twin Euler", *run_twin_samples(V, dt, integrator=1, **matched))]
            # the exact ODE on the SAME held samples (twin adaptive) isolates
            # the integration error from the sampling of the 50 ps edges
            i_same = runs[1][1]
            k_rd = read_index(stim, dt, len(V))
            g_same = i_same[k_rd] / V[k_rd]
            rng_same = float(np.max(i_same) - np.min(i_same))
            for tool, I, wall in runs:
                m, g_tool, g_ref = metrics(stim, dt, I, V, ts_ref, gap_ref, Pm)
                rel_s = np.abs(I[k_rd] / V[k_rd] - g_same) / g_same
                rows.append({"stimulus": name, "tool": tool, "dt_s": dt, "n_steps": len(V),
                             **m,
                             "g_max_pct_same_samples": 100 * float(rel_s.max()),
                             "i_nrmse_pct_same_samples":
                                 100 * float(np.sqrt(np.mean((I - i_same) ** 2)) / rng_same),
                             "wall_s": wall, "us_per_step": 1e6 * wall / len(V)})
                curves.append((f"{tool} dt={dt:g}", g_tool,
                               {"MemTorch": "o-", "twin adaptive": "s--", "twin Euler": "x:"}[tool]))
                k_ref = g_ref
        if args.memtorch_ref_dt:
            dt = args.memtorch_ref_dt
            t_mid, V = sample(stim, dt)
            I, wall = run_memtorch(memtorch, kw, P.gap_min, P.gap_max, V, dt)
            m, g_tool, _ = metrics(stim, dt, I, V, ts_ref, gap_ref, Pm)
            rows.append({"stimulus": name, "tool": "MemTorch (convergence check)", "dt_s": dt,
                         "n_steps": len(V), **m, "g_max_pct_same_samples": math.nan,
                         "i_nrmse_pct_same_samples": math.nan, "wall_s": wall,
                         "us_per_step": 1e6 * wall / len(V)})
        # .va semantics (wind-up, q = `P_Q) vs the matched reference
        ts_va, gap_va, _ = reference(stim)
        kr = np.round(stim.read_times / 10e-12).astype(int)
        g_va = Pm.I0 * np.exp(-gap_va[kr] / Pm.g0) * math.sinh(0.1 / Pm.V0) / 0.1
        g_mr = Pm.I0 * np.exp(-gap_ref[kr] / Pm.g0) * math.sinh(0.1 / Pm.V0) / 0.1
        rel = np.abs(g_va - g_mr) / g_mr
        rows.append({"stimulus": name, "tool": "twin .va semantics vs matched ref (semantic gap)",
                     "dt_s": 0.0, "n_steps": len(ts_va) - 1, "g_max_pct": 100 * float(rel.max()),
                     "g_mean_pct": 100 * float(rel.mean()), "g_last_pct": 100 * float(rel[-1]),
                     "i_nrmse_pct": math.nan, "g_max_pct_same_samples": math.nan,
                     "i_nrmse_pct_same_samples": math.nan, "wall_s": wall_ref,
                     "us_per_step": math.nan})
        curves.append(("reference (twin adaptive, exact PWL)", k_ref, "k-"))
        figure(core.FIGURES / f"memtorch_{name}.png", stim, curves)
    write(OUT / "memtorch_compare.csv", rows)
    print(f"{'stimulus':9s} {'tool':48s} {'dt':>7s} {'Gmax%':>8s} {'Gmean%':>8s} {'Glast%':>8s} "
          f"{'I NRMSE%':>9s} {'Gmax%ss':>8s} {'wall s':>8s}")
    for r in rows:
        print(f"{r['stimulus']:9s} {r['tool']:48s} {r['dt_s']:7.0e} {r['g_max_pct']:8.3f} "
              f"{r['g_mean_pct']:8.3f} {r['g_last_pct']:8.3f} {r['i_nrmse_pct']:9.4f} "
              f"{r['g_max_pct_same_samples']:8.3f} {r['wall_s']:8.3f}")
    print("Gmax%ss = per-read G error vs the exact ODE on the same held samples "
          "(integration error only); the other columns are vs the exact-PWL reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
