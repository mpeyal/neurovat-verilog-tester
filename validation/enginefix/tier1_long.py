#!/usr/bin/env python
"""Closed-loop tier-1 check of the fixed engine on the C6 ablation's 450 ms
gap configuration (k = 10: a pattern every 10th 50 ms section), >= 60 s of
stream, against a fresh Spectre run of the schedule this run produced.

    python validation/c5/run_engine.py --afferents 16 --seconds 60 --seed 1 \
        --pattern-frac 0.1 --out results/c5_long
    python validation/c5/make_deck.py --data results/c5_long --strobe 1e-3
    # Spectre on the host; fetch the PSF-ascii tran file
    python validation/enginefix/tier1_long.py scrub <tran.tran.tran> <train.log>
    python validation/enginefix/tier1_long.py score

The bulky inputs (engine reads, the 1 ms Spectre copy) stay in the
git-ignored ``results/c5_long``; the scores go to
``validation/data/c5_long/`` (``acceptance.json``, ``summary.csv``,
``engine.json``, ``spectre.json``).
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "validation" / "c5"),
                str(Path(__file__).parent)]

import numpy as np  # noqa: E402

import schedule as sch  # noqa: E402

WORK = ROOT / "results" / "c5_long"
KEEP = ROOT / "validation" / "data" / "c5_long"


def _point(work):
    import compare_c5 as cc
    sch.load.__defaults__ = (work,)
    cc.DATA = sch.DATA = work
    cc.FULL = work / "spectre_full.npz"
    cc.FIG = ROOT / "validation" / "figures" / "c5_long"
    return cc


def rail_report(work=WORK):
    """Which synapses the engine drove into an x rail (G_nv at Gmin/Gmax)."""
    meta = json.loads((work / "engine.json").read_text(encoding="utf-8"))
    p = sch.twin_params(meta)
    n = meta["masquelier"]["n_afferents"]
    rd = np.loadtxt(work / "engine_reads.csv", delimiter=",", skiprows=1)
    gnv = rd[:, 1 + n:1 + 2 * n]
    tol = 1e-9 * (p.Gmax - p.Gmin)
    hi = (gnv >= p.Gmax - tol).sum(0)
    lo = (gnv <= p.Gmin + tol).sum(0)
    return {"x1_reads": hi.tolist(), "x0_reads": lo.tolist(),
            "synapses_at_x1": [int(k) for k in np.nonzero(hi)[0]],
            "synapses_at_x0": [int(k) for k in np.nonzero(lo)[0]]}


def main(argv):
    cmd = argv[0]
    cc = _point(WORK)
    if cmd == "scrub":
        cc.scrub(argv[1], argv[2] if len(argv) > 2 else None)
        return 0
    import replay_c5 as rc
    rows, summ = rc.score(out=WORK, figures=False)
    summ["rails"] = rail_report(WORK)
    KEEP.mkdir(parents=True, exist_ok=True)
    for f in ("acceptance.json", "summary.csv", "engine.json", "spectre.json",
              "cost.json"):
        shutil.copyfile(WORK / f, KEEP / f)
    acc = json.loads((KEEP / "acceptance.json").read_text(encoding="utf-8"))
    acc["ranges_pct_of_range"] = summ
    (KEEP / "acceptance.json").write_text(json.dumps(acc, indent=1) + "\n",
                                          encoding="utf-8")
    print(json.dumps(summ["rails"]))
    return 0 if summ["engine_pass"] == summ["synapses"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
