#!/usr/bin/env python
"""Spectre's own step convergence vs the NeuroVAT twin (Stanford-PKU, stanford-1).

Scores, with the same metric and pass rule as score.py (core.score), each
candidate against a fine-step Spectre reference set:
  * Spectre at the deck step (validation/data/stanford/spectre/, 100 ps)
  * Spectre at 10 ps, if present (validation/data/stanford/ref_10ps/spectre/)
  * the twin
and, where a 0.1 ps run exists, the 1 ps reference against it.

    python validation/stanford/step_convergence.py [--ref ref_1ps/spectre]

Writes validation/data/stanford/step_convergence_<ref>.csv.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation import compare  # noqa: E402
from validation.stanford import core  # noqa: E402
from validation.stanford import stimuli as S  # noqa: E402

CANDS = [("spectre_100ps", "spectre"), ("spectre_10ps", "ref_10ps/spectre"),
         ("spectre_1ps", "ref_1ps/spectre"), ("twin", "twin")]
KEYS = ("g_read_max_pct", "g_read_mean_pct", "i_nrmse_pct", "gap_nrmse_pct")


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
            r["verdict"] = core.verdict(r)
            rows.append(r)
    compare.write_rows(core.DATA / f"step_convergence_{a.ref.split('/')[0]}.csv", rows)
    for r in rows:
        f = lambda v: "-" if v is None or v != v else f"{v:.3f}"  # noqa: E731
        print(f"{r['stimulus']:18s} {r['candidate']:14s} Gmax {f(r['g_read_max_pct']):>7s} "
              f"Gmean {f(r['g_read_mean_pct']):>7s} I {f(r['i_nrmse_pct']):>7s} "
              f"gap {f(r['gap_nrmse_pct']):>7s} {r['verdict']}")


if __name__ == "__main__":
    main()
