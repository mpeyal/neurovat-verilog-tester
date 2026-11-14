#!/usr/bin/env python
"""Import user-run Spectre results of the sharbati-1 decks into
validation/data/sharbati/spectre/<model>/<stimulus>.csv.

Same readers, canonical mapping and leak check as validation/scrub_spectre.py
(PSF-ascii or CSV export; host / user / path / date fields are dropped and the
file is refused if any output line could identify a machine or person).  Only
the stimulus set (sharbati-1), its fingerprint and the destination differ, and
the rows are thinned like the twin data (every point within 2 us of a
waveform edge is kept, so the gate-charge check stays exact).

    python validation/paper/scrub_paper.py --dir validation/spice/spectre_sharbati   # batch
    python validation/paper/scrub_paper.py --model v3 --stimulus sp_pot \\
        path/to/sp_pot.raw/tran.tran --log path/to/sp_pot.log
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation import common  # noqa: E402
from validation import scrub_spectre as base  # noqa: E402
from validation.paper import core  # noqa: E402
from validation.paper import sharbati_stimuli as S  # noqa: E402
from validation.stimuli import EDGE_S  # noqa: E402

ScrubError = base.ScrubError


def render(key, cols, info):
    prov = (f"# source: {base.version_tag(info)}; model: {key}; stimulus_set: "
            f"{S.STIMULUS_SET_VERSION} {S.fingerprint()[:16]}")
    data = np.column_stack([np.asarray(cols[c], dtype=float) for c in common.COLUMNS])
    lines = [prov, ",".join(common.COLUMNS)]
    for row in data:
        lines.append(",".join("nan" if math.isnan(v) else f"{v:.10g}" for v in row))
    return lines


def scrub_one(key, stim_name, src, log=None, wall_s=None, out_root=None, extra_ids=()):
    stim = S.BY_NAME[stim_name]
    text = Path(src).read_text(encoding="utf-8", errors="replace")
    if "VALUE" in text and "HEADER" in text:
        sig, info = base.read_psf_ascii(text)
    else:
        sig, info = base.read_csv_export(text)
    cols, measured = base.to_canonical(key, sig, stim)
    if cols["t_s"][-1] < stim.t_stop * (1 - 1e-6):
        raise ScrubError(f"transient stops at {cols['t_s'][-1]:g} s, "
                         f"stimulus needs {stim.t_stop:g} s")
    n_raw = len(cols["t_s"])
    cols = core.thin(cols, stim, keep_edge_halo=2 * EDGE_S)
    lines = render(key, cols, info)
    base.check_clean(lines, extra_ids)
    if wall_s is None and log:
        wall_s = base.wall_from_log(Path(log).read_text(encoding="utf-8", errors="replace"))
    root = Path(out_root) if out_root else core.DATA
    dest = root / "spectre" / key / f"{stim_name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    common.write_meta(dest.with_suffix(".json"), {
        "source": base.version_tag(info), "model": key, "stimulus": stim_name,
        "stimulus_set": S.STIMULUS_SET_VERSION, "n_points_run": int(n_raw),
        "n_points_saved": int(len(cols["t_s"])), "drive_measured": bool(measured),
        "wall_s": None if wall_s is None else round(float(wall_s), 4)})
    return dest


def batch(root, out_root=None):
    done = []
    for key in common.MODELS:
        for st in S.STIMULI:
            d = Path(root) / key
            cands = [d / f"{st.name}.raw" / "tran.tran", d / f"{st.name}.csv"]
            src = next((c for c in cands if c.is_file()), None)
            if src is None:
                continue
            log = d / f"{st.name}.log"
            done.append(scrub_one(key, st.name, src, log=log if log.is_file() else None,
                                  out_root=out_root))
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?")
    ap.add_argument("--model", choices=list(common.MODELS))
    ap.add_argument("--stimulus", choices=list(S.BY_NAME))
    ap.add_argument("--log")
    ap.add_argument("--wall-s", type=float)
    ap.add_argument("--dir", help="batch: <dir>/<model>/<stim>.raw/tran.tran")
    args = ap.parse_args(argv)
    try:
        if args.dir:
            done = batch(args.dir)
            for d in done:
                print("wrote", d.relative_to(common.ROOT).as_posix())
            print(f"{len(done)} Spectre result(s) imported")
            return 0
        if not (args.src and args.model and args.stimulus):
            ap.error("give SRC --model --stimulus, or --dir")
        d = scrub_one(args.model, args.stimulus, args.src, args.log, args.wall_s)
        print("wrote", d.relative_to(common.ROOT).as_posix())
        return 0
    except ScrubError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
