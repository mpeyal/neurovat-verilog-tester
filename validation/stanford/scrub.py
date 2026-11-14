#!/usr/bin/env python
"""Import user-run Spectre results of the stanford-1 decks into
validation/data/stanford/spectre/<stimulus>.csv.

Same readers and leak patterns as validation/scrub_spectre.py (PSF-ascii or a
CSV export; host / user / path / date fields are dropped and the file is
REFUSED if any output line could identify a machine or person).  Only the
signal mapping, the canonical columns (validation/stanford/core.py COLUMNS)
and the neutral provenance form differ:

    V_TE_V   <- n_te
    I_TE_A   <- X1:TE (current into TE); else -vte:p
    gap_m    <- inverted from I and V where |V| >= 50 mV (NaN elsewhere)
    T_K      <- T_ini + |V I| Rth;  G_read_S <- I(0.1 V)/0.1 at that gap
    gap_state_m = NaN (the idt state is not observable from the terminals)

    python validation/stanford/scrub.py --dir validation/spice/spectre_stanford   # batch
    python validation/stanford/scrub.py --stimulus reset_train path/to/reset_train.raw/tran.tran \\
        --log path/to/reset_train.log
"""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation import common  # noqa: E402
from validation import scrub_spectre as base  # noqa: E402
from validation.stanford import core  # noqa: E402
from validation.stanford import stimuli as S  # noqa: E402

ScrubError = base.ScrubError
_PROV_OK = re.compile(r"^# source: Spectre( \d+\.\d+)?; model: stanford_pku; "
                      r"stimulus_set: [\w.-]+ [0-9a-f]{16}$")


def to_canonical(sig, stim):
    t = np.asarray(sig["time"], dtype=float)
    v = base._find(sig, "n_te")
    if v is None:
        raise ScrubError("TE node (n_te) not found; signals found: "
                         + ", ".join(sorted(sig)))
    i = base._find(sig, "x1:te", ":te")
    if i is None:
        isrc = base._find(sig, "vte:p")
        if isrc is None:
            raise ScrubError("no TE current (X1:TE or vte:p) in the Spectre data")
        i = -np.asarray(isrc, dtype=float)
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    # sinh I-V: I and V always share their sign, so a flipped probe shows up
    # as a negative V*I integral
    if np.sum(v * i) < 0:
        raise ScrubError("TE current has the wrong sign (V*I < 0); check the "
                         "X1:TE / vte:p orientation")
    P = core.twin_module().StanfordPKUParams(**stim.instance_dict())
    gap = core.gap_from_iv(v, i, P)
    T = P.T_ini + np.abs(v * i * P.Rth)
    g_read = P.I0 * np.exp(-gap / P.g0) * math.sinh(P.V_read / P.V0) / P.V_read
    nan = np.full(len(t), math.nan)
    return {"t_s": t, "V_TE_V": v, "I_TE_A": i, "gap_m": gap, "gap_state_m": nan,
            "T_K": T, "G_read_S": g_read}


def render(cols, info):
    prov = (f"# source: {base.version_tag(info)}; model: {core.MODEL}; stimulus_set: "
            f"{S.STIMULUS_SET_VERSION} {S.fingerprint()[:16]}")
    data = np.column_stack([np.asarray(cols[c], dtype=float) for c in core.COLUMNS])
    lines = [prov, ",".join(core.COLUMNS)]
    for row in data:
        lines.append(",".join("nan" if math.isnan(x) else f"{x:.10g}" for x in row))
    return lines


def check_clean(lines, extra_ids=()):
    """Raise ScrubError if any output line could identify a host/user/path
    (base.FORBIDDEN patterns, local user/host names, numeric-only rows)."""
    ids = list(extra_ids) + base._local_identifiers()
    if not _PROV_OK.match(lines[0]):
        raise ScrubError(f"provenance line not in the neutral form: {lines[0]!r}")
    if lines[1] != ",".join(core.COLUMNS):
        raise ScrubError("unexpected header line")
    for k, ln in enumerate(lines):
        if k >= 2 and not base._ROW_OK.match(ln):
            raise ScrubError(f"line {k+1} is not purely numeric")
        probe = ln if k else ln.split("stimulus_set:")[0]
        for label, rx in base.FORBIDDEN:
            if k >= 2 and label in ("keyword", "date", "host name", "path"):
                continue
            if rx.search(probe):
                raise ScrubError(f"line {k+1} matches a {label} pattern; refusing to write")
        if k < 2:
            low = ln.lower()
            for ident in ids:
                if ident in low:
                    raise ScrubError(f"line {k+1} contains a local identifier; "
                                     "refusing to write")


def scrub_one(stim_name, src, log=None, wall_s=None, out_root=None, extra_ids=()):
    stim = S.BY_NAME[stim_name]
    text = Path(src).read_text(encoding="utf-8", errors="replace")
    if "VALUE" in text and "HEADER" in text:
        sig, info = base.read_psf_ascii(text)
    else:
        sig, info = base.read_csv_export(text)
    cols = to_canonical(sig, stim)
    if cols["t_s"][-1] < stim.t_stop * (1 - 1e-6):
        raise ScrubError(f"transient stops at {cols['t_s'][-1]:g} s, "
                         f"stimulus needs {stim.t_stop:g} s")
    lines = render(cols, info)
    check_clean(lines, extra_ids)
    if wall_s is None and log:
        wall_s = base.wall_from_log(Path(log).read_text(encoding="utf-8", errors="replace"))
    root = Path(out_root) if out_root else core.DATA
    dest = root / "spectre" / f"{stim_name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    common.write_meta(dest.with_suffix(".json"), {
        "source": base.version_tag(info), "model": core.MODEL, "stimulus": stim_name,
        "stimulus_set": S.STIMULUS_SET_VERSION, "n_points": int(len(cols["t_s"])),
        "wall_s": None if wall_s is None else round(float(wall_s), 4)})
    return dest


def batch(root, out_root=None):
    """Import every stimulus found; returns (written paths, [(name, reason)]).
    A refused or truncated run is reported, not fatal for the others."""
    done, refused = [], []
    for st in S.STIMULI:
        d = Path(root)
        cands = [d / f"{st.name}.raw" / "tran.tran", d / f"{st.name}.csv"]
        src = next((c for c in cands if c.is_file()), None)
        if src is None:
            continue
        log = d / f"{st.name}.log"
        try:
            done.append(scrub_one(st.name, src, log=log if log.is_file() else None,
                                  out_root=out_root))
        except ScrubError as e:
            refused.append((st.name, str(e)))
    return done, refused


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?")
    ap.add_argument("--stimulus", choices=list(S.BY_NAME))
    ap.add_argument("--log")
    ap.add_argument("--wall-s", type=float)
    ap.add_argument("--dir", help="batch: <dir>/<stim>.raw/tran.tran")
    args = ap.parse_args(argv)
    try:
        if args.dir:
            done, refused = batch(args.dir)
            for d in done:
                print("wrote", d.relative_to(common.ROOT).as_posix())
            for name, why in refused:
                print(f"REFUSED {name}: {why}", file=sys.stderr)
            print(f"{len(done)} Spectre result(s) imported, {len(refused)} refused")
            return 2 if refused else 0
        if not (args.src and args.stimulus):
            ap.error("give SRC --stimulus, or --dir")
        d = scrub_one(args.stimulus, args.src, args.log, args.wall_s)
        print("wrote", d.relative_to(common.ROOT).as_posix())
        return 0
    except ScrubError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
