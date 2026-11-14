#!/usr/bin/env python
"""Run every Gate A stimulus through the Python twins (v2, v3).

Writes validation/data/twin/<model>/<stimulus>.csv (canonical columns, see
validation/common.py) and a <stimulus>.json sidecar with the wall time, so the
CSV itself stays byte-identical between runs.

    python validation/run_twin.py                 # all models, all stimuli
    python validation/run_twin.py --model v3 --stimulus single_pulse
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation import common, stimuli  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all", help="v2, v3 or all")
    ap.add_argument("--stimulus", nargs="*", help="names (default: all)")
    args = ap.parse_args(argv)

    total = 0.0
    print(f"stimulus set {stimuli.STIMULUS_SET_VERSION} "
          f"({stimuli.fingerprint()[:16]})")
    for key in common.model_keys(args.model):
        for st in stimuli.get(args.stimulus):
            cols, wall = common.run_twin(key, st)
            total += wall
            path = common.data_path("twin", key, st.name)
            common.write_csv(path, cols, common.provenance_line(
                "Python twin (simulation_core)", key))
            common.write_meta(path.with_suffix(".json"), {
                "source": "twin", "model": key, "stimulus": st.name,
                "stimulus_set": stimuli.STIMULUS_SET_VERSION,
                "n_points": int(len(cols["t_s"])), "wall_s": round(wall, 4)})
            g = cols["G_S"]
            print(f"  {key} {st.name:20s} {len(g):6d} pts  {wall:6.2f} s  "
                  f"G {g.min()*1e6:8.2f}..{g.max()*1e6:8.2f} uS  "
                  f"-> {path.relative_to(common.ROOT).as_posix()}")
    print(f"total twin wall time {total:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
