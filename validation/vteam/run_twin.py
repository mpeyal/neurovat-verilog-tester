#!/usr/bin/env python
"""Run the vteam-1 stimuli through the VTEAM twin.

    python validation/vteam/run_twin.py                 # data + figures
    python validation/vteam/run_twin.py --self-check    # output grid dt, dt/2, dt/4
    python validation/vteam/run_twin.py --stimulus s1_vteam

Writes validation/data/vteam/twin/<stim>.csv (+ .json wall time / cost),
validation/data/vteam/self_check.csv and figures under validation/figures/vteam/.
The twin integrates in closed form, so the self-check (the same stimulus on a
grid refined 2x and 4x) must agree to rounding.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation import common, compare  # noqa: E402
from validation.vteam import core  # noqa: E402
from validation.vteam import stimuli as S  # noqa: E402


def figure(stim, cols, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = cols["t_s"] * 1e9
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(t, cols["V_V"], color="0.2", lw=0.9)
    ax[0].set_ylabel("V(p,n) (V)")
    ax[0].set_title(f"{stim.name}: {stim.description}", fontsize=8)
    ax[1].plot(t, cols["I_A"] * 1e3, color="tab:blue", lw=0.9)
    ax[1].set_ylabel("I (mA)")
    ax[2].plot(t, cols["x_m"] * 1e9, color="tab:red", lw=1.0)
    ax[2].set_ylabel("x (nm)")
    ax[2].set_xlabel("t (ns)")
    if len(stim.reads):
        ax2 = ax[2].twinx()
        ax2.plot(stim.read_times * 1e9, core.read_conductance(cols, stim) * 1e3, "k.", ms=4)
        ax2.set_ylabel("G_read (mS)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_all(names, figures=True):
    total = 0.0
    for st in S.get(names):
        cols, wall, pieces = core.run_twin(st)
        total += wall
        p = core.data_path("twin", st.name)
        core.write_csv(p, cols, core.provenance_line("Python twin (twins/vteam.py)"))
        common.write_meta(p.with_suffix(".json"), {
            "source": "twin", "model": core.MODEL, "stimulus": st.name,
            "stimulus_set": S.STIMULUS_SET_VERSION, "n_points": int(len(cols["t_s"])),
            "closed_form_pieces": int(pieces), "wall_s": round(wall, 4)})
        g = core.read_conductance(cols, st)
        gtxt = f"G_read {g.min()*1e3:7.3f}..{g.max()*1e3:7.3f} mS" if len(g) else ""
        print(f"  {st.name:20s} {len(cols['t_s']):6d} pts {wall:6.3f} s  "
              f"x {cols['x_m'].min()*1e9:5.3f}..{cols['x_m'].max()*1e9:5.3f} nm  {gtxt}")
        if figures:
            figure(st, cols, core.FIGURES / f"twin_{st.name}.png")
    print(f"total twin wall time {total:.3f} s")


def self_check(names):
    rows = []
    for st in S.get(names):
        runs = {f: core.run_twin(st, refine=f)[0] for f in (1, 2, 4)}
        sub = {f: {k: v[::f] for k, v in runs[f].items()} for f in runs}
        r1 = core.score(sub[1], sub[4], st)
        d14 = float(np.max(np.abs(sub[1]["s_state"] - sub[4]["s_state"])))
        rows.append({"stimulus": st.name, "n_points": r1["n_points"],
                     "g_read_max_pct_dt_vs_dt4": r1["g_read_max_pct"],
                     "i_nrmse_pct_dt_vs_dt4": r1["i_nrmse_pct"],
                     "x_nrmse_pct_dt_vs_dt4": r1["x_nrmse_pct"],
                     "state_maxdiff_dt_dt4": d14, "verdict": core.verdict(r1)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stimulus", nargs="*")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    print(f"stimulus set {S.STIMULUS_SET_VERSION} ({S.fingerprint()[:16]})")
    if not args.self_check:
        run_all(args.stimulus, not args.no_figures)
        return 0
    rows = self_check(args.stimulus)
    compare.write_rows(core.DATA / "self_check.csv", rows)
    for r in rows:
        print(f"  {r['stimulus']:20s} Gmax {r['g_read_max_pct_dt_vs_dt4']:.2e} %  "
              f"I {r['i_nrmse_pct_dt_vs_dt4']:.2e} %  max|ds| {r['state_maxdiff_dt_dt4']:.2e}  "
              f"{r['verdict']}")
    return 1 if any(r["verdict"] != "PASS" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
