#!/usr/bin/env python
"""Spectre's own step convergence vs the NeuroVAT twin (VTEAM, vteam-1).

Scores, with the same metric and pass rule as score.py (core.score), each
candidate against a fine-step Spectre reference set:
  * Spectre at the deck step (validation/data/vteam/spectre/, 100 ps)
  * Spectre at 10 ps / 1 ps if present (validation/data/vteam/ref_10ps|ref_1ps/spectre/)
  * the twin (exact closed form on the 100 ps output grid)
The candidate equal to the reference is skipped.

    python validation/vteam/step_convergence.py [--ref ref_1ps/spectre]

Writes validation/data/vteam/step_convergence_<ref>.csv.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation import common, compare  # noqa: E402
from validation.vteam import core  # noqa: E402
from validation.vteam import stimuli as S  # noqa: E402

CANDS = [("spectre_100ps", "spectre"), ("spectre_10ps", "ref_10ps/spectre"),
         ("spectre_1ps", "ref_1ps/spectre"), ("twin", "twin")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="ref_1ps/spectre")
    ap.add_argument("--stimulus", nargs="*")
    a = ap.parse_args(argv)
    rows = []
    for st in S.get(a.stimulus):
        ref = core.load(a.ref, st.name)
        if ref is None:
            continue
        for label, src in CANDS:
            if src == a.ref:
                continue
            c = core.load(src, st.name)
            if c is None and src == "twin":
                c, _, _ = core.run_twin(st)
            if c is None:
                continue
            r = {"stimulus": st.name, "reference": a.ref, "candidate": label,
                 **core.score(c, ref, st)}
            r["wall_s"] = common.read_meta(core.data_path(src, st.name, ".json")).get("wall_s")
            r["verdict"] = core.verdict(r)
            rows.append(r)
    if not rows:
        print(f"no reference data under validation/data/vteam/{a.ref}/")
        return 0
    compare.write_rows(core.DATA / f"step_convergence_{a.ref.split('/')[0]}.csv", rows)
    f = lambda v: "-" if v is None or v != v else f"{v:.4f}"  # noqa: E731
    for r in rows:
        print(f"{r['stimulus']:18s} {r['candidate']:14s} Gmax {f(r['g_read_max_pct']):>8s} "
              f"Gmean {f(r['g_read_mean_pct']):>8s} I {f(r['i_nrmse_pct']):>8s} "
              f"x {f(r['x_nrmse_pct']):>8s} {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
