#!/usr/bin/env python
"""Score the Stanford-PKU twin against imported Spectre runs (stanford-1).

    python validation/stanford/score.py

Per stimulus (pass rule in validation/stanford/core.py):
  * per-read conductance G = I/V at every 0.1 V read plateau middle:
    max relative error <= 3 %, mean <= 1 %;
  * I(t) NRMSE (range of the Spectre current) <= 1 %;
  * gap NRMSE (where the gap can be inverted from I and V, |V| >= 50 mV) <= 1 %.
Evaluated on the twin's grid clipped to the common span; the reference is
linearly interpolated.  Writes validation/data/stanford/summary.csv and
validation/figures/stanford/spectre_<stim>.png.  Exit 1 if any row FAILs.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation import common, compare  # noqa: E402
from validation.stanford import core  # noqa: E402
from validation.stanford import stimuli as S  # noqa: E402


def overlay(path, stim, tw, sp, row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    ax[0].semilogy(sp["t_s"], np.abs(sp["I_TE_A"]) + 1e-15, color="0.15", lw=1.6, label="Spectre")
    ax[0].semilogy(tw["t_s"], np.abs(tw["I_TE_A"]) + 1e-15, color="tab:blue", lw=0.9, label="twin")
    ax[0].set_ylabel("|I_TE| (A)")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{stim.name}: twin vs Spectre  G max {row['g_read_max_pct']:.3g}%  "
                    f"I NRMSE {row['i_nrmse_pct']:.3g}%  [{row['verdict']}]", fontsize=9)
    ax[1].plot(sp["t_s"], sp["gap_m"] * 1e9, color="0.15", lw=1.6, label="Spectre (from I,V)")
    ax[1].plot(tw["t_s"], tw["gap_m"] * 1e9, color="tab:red", lw=0.9, label="twin")
    ax[1].set_ylabel("gap (nm)")
    ax[1].set_xlabel("t (s)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stimulus", nargs="*")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    rows = []
    for st in S.get(args.stimulus):
        sp = core.load("spectre", st.name)
        if sp is None:
            continue
        tw = core.load("twin", st.name)
        if tw is None:
            tw, _, _ = core.run_twin(st)
        r = {"stimulus": st.name, **core.score(tw, sp, st)}
        tm = common.read_meta(core.data_path("twin", st.name, ".json"))
        sm = common.read_meta(core.data_path("spectre", st.name, ".json"))
        r["twin_wall_s"], r["spectre_wall_s"] = tm.get("wall_s"), sm.get("wall_s")
        r["verdict"] = core.verdict(r)
        rows.append(r)
        if not args.no_figures:
            overlay(core.FIGURES / f"spectre_{st.name}.png", st, tw, sp, r)
    if not rows:
        print("no Spectre data under validation/data/stanford/spectre/ yet - run the decks "
              "in validation/spice/spectre_stanford/ and import them with "
              "validation/stanford/scrub.py (see validation/README.md)")
        return 0
    compare.write_rows(core.DATA / "summary.csv", rows)

    def f(v, s):
        return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, s)
    print(f"{'stimulus':20s} {'Gmax%':>8s} {'Gmean%':>8s} {'I NRMSE%':>9s} {'gap NRMSE%':>10s} verdict")
    for r in rows:
        print(f"{r['stimulus']:20s} {f(r['g_read_max_pct'], '8.3f')} {f(r['g_read_mean_pct'], '8.3f')} "
              f"{f(r['i_nrmse_pct'], '9.3f')} {f(r['gap_nrmse_pct'], '10.3f')} {r['verdict']}")
    return 1 if any(r["verdict"] != "PASS" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
