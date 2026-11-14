#!/usr/bin/env python
"""Run the sharbati-1 stimuli through the Python twins (v2, v3).

Writes validation/data/sharbati/twin/<model>/<stimulus>.csv (canonical
columns, thinned - see core.SAVE) and a .json sidecar with the wall time.

    python validation/paper/run_twin_paper.py                    # everything
    python validation/paper/run_twin_paper.py --model v3 --stimulus sp_pot
    python validation/paper/run_twin_paper.py --self-check       # dt, dt/2, dt/4
    python validation/paper/run_twin_paper.py --self-check --stimulus sp_pot retention_13h
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation import common, compare  # noqa: E402
from validation.paper import core  # noqa: E402
from validation.paper import sharbati_stimuli as S  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all", help="v2, v3 or all")
    ap.add_argument("--stimulus", nargs="*", help="names (default: all)")
    ap.add_argument("--self-check", action="store_true",
                    help="time-step convergence (default stimuli: retention_13h, "
                         "sp_pot, pp_dt0015ms) -> data/sharbati/self_check.csv")
    args = ap.parse_args(argv)
    keys = common.model_keys(args.model)
    print(f"stimulus set {S.STIMULUS_SET_VERSION} ({S.fingerprint()[:16]})")

    if args.self_check:
        names = args.stimulus or ["retention_13h", "sp_pot", "pp_dt0015ms"]
        S.get(names)
        rows = core.self_check(keys, names)
        compare.print_self_table(rows)
        for r in rows:
            if r.get("retention_pct_dt_dt2_dt4"):
                print(f"  {r['model']} retention %/13h at dt / dt/2 / dt/4: "
                      f"{r['retention_pct_dt_dt2_dt4']}")
        compare.write_rows(core.DATA / "self_check.csv", rows)
        return 1 if any(r["verdict"] != "PASS" for r in rows) else 0

    total = 0.0
    for key in keys:
        for st in S.get(args.stimulus):
            path, cols, kept, wall = core.write_twin(key, st)
            total += wall
            g = cols["G_S"]
            print(f"  {key} {st.name:14s} {len(g):7d} pts run, {len(kept['t_s']):6d} saved "
                  f"{wall:6.2f} s  G {g.min()*1e6:8.2f}..{g.max()*1e6:8.2f} uS  "
                  f"-> {path.relative_to(common.ROOT).as_posix()}")
    print(f"total twin wall time {total:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
