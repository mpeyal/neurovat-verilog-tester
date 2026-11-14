#!/usr/bin/env python
"""Score the VTEAM twin against imported Spectre runs (vteam-1).

    python validation/vteam/score.py                     # vs the deck-step runs
    python validation/vteam/score.py --ref ref_1ps/spectre

Pass rule (validation/vteam/core.py, the same limits as stanford-1): per-read
G = I/V at every -0.1 V read middle max <= 3 %, mean <= 1 %; I(t) NRMSE <= 1 %
of the reference current range; state x NRMSE <= 1 % of its range.  Evaluated
on the twin's grid clipped to the common span; the reference is linearly
interpolated.  Writes validation/data/vteam/summary[_<ref>].csv and
validation/figures/vteam/spectre[_<ref>]_<stim>.png.  Exit 1 if any row FAILs.
"""

import argparse
import math
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation import common, compare  # noqa: E402
from validation.vteam import core  # noqa: E402
from validation.vteam import stimuli as S  # noqa: E402


def overlay(path, stim, tw, sp, row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    ax[0].plot(sp["t_s"] * 1e9, sp["I_A"] * 1e3, color="0.15", lw=1.6, label="Spectre")
    ax[0].plot(tw["t_s"] * 1e9, tw["I_A"] * 1e3, color="tab:blue", lw=0.9, label="twin")
    ax[0].set_ylabel("I (mA)")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{stim.name}: twin vs Spectre  G max {row['g_read_max_pct']:.3g}%  "
                    f"I NRMSE {row['i_nrmse_pct']:.3g}%  [{row['verdict']}]", fontsize=9)
    ax[1].plot(sp["t_s"] * 1e9, sp["x_m"] * 1e9, color="0.15", lw=1.6, label="Spectre")
    ax[1].plot(tw["t_s"] * 1e9, tw["x_m"] * 1e9, color="tab:red", lw=0.9, label="twin")
    ax[1].set_ylabel("x (nm)")
    ax[1].set_xlabel("t (ns)")
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
    ap.add_argument("--ref", default="spectre",
                    help="reference set under validation/data/vteam/ (default spectre)")
    args = ap.parse_args(argv)
    tag = "" if args.ref == "spectre" else "_" + args.ref.split("/")[0]
    rows = []
    for st in S.get(args.stimulus):
        sp = core.load(args.ref, st.name)
        if sp is None:
            continue
        tw = core.load("twin", st.name)
        if tw is None:
            tw, _, _ = core.run_twin(st)
        r = {"stimulus": st.name, **core.score(tw, sp, st)}
        tm = common.read_meta(core.data_path("twin", st.name, ".json"))
        sm = common.read_meta(core.data_path(args.ref, st.name, ".json"))
        r["twin_wall_s"], r["spectre_wall_s"] = tm.get("wall_s"), sm.get("wall_s")
        r["verdict"] = core.verdict(r)
        rows.append(r)
        if not args.no_figures:
            overlay(core.FIGURES / f"spectre{tag}_{st.name}.png", st, tw, sp, r)
    if not rows:
        print(f"no Spectre data under validation/data/vteam/{args.ref}/ - run the decks in "
              "validation/spice/spectre_vteam/ and import them with validation/vteam/scrub.py")
        return 0
    compare.write_rows(core.DATA / f"summary{tag}.csv", rows)

    def f(v, s):
        return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, s)
    print(f"{'stimulus':20s} {'Gmax%':>8s} {'Gmean%':>8s} {'I NRMSE%':>9s} {'x NRMSE%':>9s} "
          f"{'twin s':>7s} {'Spectre s':>9s} verdict")
    for r in rows:
        print(f"{r['stimulus']:20s} {f(r['g_read_max_pct'], '8.4f')} "
              f"{f(r['g_read_mean_pct'], '8.4f')} {f(r['i_nrmse_pct'], '9.4f')} "
              f"{f(r['x_nrmse_pct'], '9.4f')} {f(r['twin_wall_s'], '7.3f')} "
              f"{f(r['spectre_wall_s'], '9.3f')} {r['verdict']}")
    return 1 if any(r["verdict"] != "PASS" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
