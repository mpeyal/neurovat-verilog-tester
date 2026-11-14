#!/usr/bin/env python
"""Brian2 hand port of ECFET v3 vs the NeuroVAT twin and Spectre (T3b).

Every gateA-1 stimulus through validation/engine/brian2_v3.py; G compared on
the twin's grid (validation/compare.py's metrics, NRMSE and max error as % of
the reference range). Twin CSVs are regenerated when validation/data/twin is
missing; the Spectre CSVs are the committed validation/data/spectre/v3.

    python validation/engine/v3_compare.py [--stimulus NAME ...] [--dt 1e-4]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation import common, stimuli  # noqa: E402
from validation.compare import metrics, common_grid  # noqa: E402
from validation.engine import common as C  # noqa: E402
from validation.engine import brian2_v3 as V  # noqa: E402


def twin_cols(name):
    p = common.data_path("twin", "v3", name)
    if not p.is_file():
        cols, wall = common.run_twin("v3", stimuli.BY_NAME[name])
        common.write_csv(p, cols, common.provenance_line(
            "Python twin (simulation_core)", "v3"))
    cols, _ = common.read_csv(p)
    return cols


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stimulus", nargs="*")
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    rows, keep = [], {}
    for st in stimuli.get(args.stimulus):
        tw = twin_cols(st.name)
        _c, twall = common.run_twin("v3", st)          # twin wall time, now
        t, G, bwall = V.run(st, dt=args.dt)
        sp_path = common.data_path("spectre", "v3", st.name)
        sp = common.read_csv(sp_path)[0] if sp_path.is_file() else None
        tg = tw["t_s"]
        g_b = np.interp(tg, t, G)
        m_bt = metrics(g_b, tw["G_S"])
        row = {"stimulus": st.name, "dt_s": args.dt, "n_points": len(tg),
               "b2_vs_twin_nrmse_pct": 100 * m_bt["nrmse"],
               "b2_vs_twin_maxerr_pct": 100 * m_bt["maxerr"]}
        if sp is not None:
            tc = common_grid(tg, sp["t_s"])
            g_s = np.interp(tc, sp["t_s"], sp["G_S"])
            m_bs = metrics(np.interp(tc, t, G), g_s)
            m_ts = metrics(np.interp(tc, tg, tw["G_S"]), g_s)
            # a fixed step puts an off-grid onset (and its lock-in jump) up
            # to one dt early: score also without the samples within one
            # dt of a pulse edge
            edges = np.asarray(st.waveform().edges, float)
            far = np.abs(tc[:, None] - edges[None, :]).min(1) > args.dt * 1.0001
            g_bc = np.interp(tc, t, G)
            m_far = {"maxerr": float(np.abs(g_bc - g_s)[far].max()) / m_bs["range"]}
            row.update({"b2_vs_spectre_nrmse_pct": 100 * m_bs["nrmse"],
                        "b2_vs_spectre_maxerr_pct": 100 * m_bs["maxerr"],
                        "b2_vs_spectre_maxerr_off_edge_pct": 100 * m_far["maxerr"],
                        "twin_vs_spectre_nrmse_pct": 100 * m_ts["nrmse"],
                        "twin_vs_spectre_maxerr_pct": 100 * m_ts["maxerr"]})
        row.update({"g_range_uS": 1e6 * m_bt["range"], "sim_s": st.t_stop,
                    "b2_wall_s": bwall, "twin_wall_s": twall,
                    "b2_over_twin": bwall / twall if twall else None})
        rows.append(row)
        keep[st.name] = (tg, tw["G_S"], t, G, sp)
        print(" ", {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in row.items()}, flush=True)
    tag = args.tag or ("" if args.dt == 1e-4 else f"_dt{args.dt:g}")
    C.write_csv(C.DATA / f"brian2_v3{tag}.csv", rows)
    C.write_json(C.DATA / f"brian2_v3_port{tag}.json", {
        "loc_model": V.source_lines(),
        "loc_file": sum(1 for _ in open(V.__file__, encoding="utf-8")),
        "non_ode_constructs": [
            "pulse-onset edge detection (prev flag) in run_regularly",
            "trace harvest + consume + saturate at onset",
            "stdp_lock clamp to [Gmin, Gmax] - G_nv at onset",
            "x rail gating and clip at [0, 1]",
            "1 pA drift threshold / leak scale on I",
            "G output clip"],
        "pitfalls": [
            "sympy cannot differentiate int()/comparisons inside ODE "
            "subexpressions under rk4: the gated write rate and sign had to "
            "move into the per-step event code (held over each step)",
            "the drive as TimedArray inside an ODE also fails (non-real "
            "comparison); it is latched into a parameter each step",
            "fixed dt: the random train's 1 us edges are not on the grid; "
            "the step-averaged current keeps the charge exact"],
        "dt_s": args.dt, "method": "rk4", "codegen": "numpy"})
    if not args.tag and args.dt == 1e-4:
        figure(keep)
    return 0


def figure(keep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    names = [n for n in ("single_pulse", "stdp_pair_sweep", "ltp_ltd_train",
                         "random_spike_train") if n in keep]
    if not names:
        return
    fig, axs = plt.subplots(len(names), 1, figsize=(7, 2.3 * len(names)))
    axs = np.atleast_1d(axs)
    for ax, n in zip(axs, names):
        tg, gt, t, G, sp = keep[n]
        if sp is not None:
            ax.plot(sp["t_s"], sp["G_S"] * 1e6, color="0.6", lw=2.2, label="Spectre")
        ax.plot(tg, gt * 1e6, lw=1.2, label="NeuroVAT twin")
        ax.plot(t, G * 1e6, "--", lw=1, label="Brian2 port")
        ax.set_title(n, fontsize=9)
        ax.set_ylabel("G (uS)")
    axs[0].legend(frameon=False, fontsize=7)
    axs[-1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "v3_brian2_port.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
