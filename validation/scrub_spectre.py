#!/usr/bin/env python
"""Import user-run Spectre results into validation/data/spectre/<model>/.

Accepts, per (model, stimulus):
  * a PSF-ascii transient file  (<stim>.raw/tran.tran from
    `spectre -format psfascii -raw <stim>.raw ...`), or
  * a CSV export (first column time, then one column per saved signal; ViVA
    "X / Y" column pairs are accepted when all X columns are identical).
Optionally the Spectre log (<stim>.log) to read the transient wall time.

The output is REBUILT, never copied: a fixed provenance line
("# source: Spectre <major.minor>; model: ...; stimulus_set: ..."), the
canonical header (validation/common.py COLUMNS) and numeric rows only.  Host
names, user names, paths, dates and design names in the PSF header or log are
dropped.  As a backstop, every output line is checked against host/user/path/
date/e-mail/IP patterns and the file is NOT written if anything matches.

    python validation/scrub_spectre.py --model v3 --stimulus single_pulse \\
        path/to/single_pulse.raw/tran.tran --log path/to/single_pulse.log
    python validation/scrub_spectre.py --dir validation/spice/spectre   # batch
"""

import argparse
import getpass
import math
import re
import socket
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation_core import V3Params  # noqa: E402
from validation import common, stimuli  # noqa: E402


class ScrubError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------
_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|(\()|(\))|([^\s"()]+)')


def _tokens(text):
    for m in _TOKEN.finditer(text):
        if m.group(1) is not None:
            yield ("str", m.group(1))
        elif m.group(2):
            yield ("(", None)
        elif m.group(3):
            yield (")", None)
        else:
            yield ("word", m.group(4))


def _num(word):
    try:
        return float(word)
    except (TypeError, ValueError):
        return None


def read_psf_ascii(text):
    """-> (signals {name: array}, info {"simulator", "version"})."""
    sections = {}
    cur = None
    for line in text.splitlines():
        s = line.strip()
        if s in ("HEADER", "TYPE", "SWEEP", "TRACE", "VALUE", "END"):
            cur = s
            sections.setdefault(cur, [])
            continue
        if cur:
            sections[cur].append(line)
    if "VALUE" not in sections:
        raise ScrubError("not a PSF-ascii file (no VALUE section)")

    info = {}
    head = list(_tokens("\n".join(sections.get("HEADER", []))))
    for (k1, a), (k2, b) in zip(head, head[1:]):
        if k1 == "str" and k2 == "str" and a in ("simulator", "version"):
            info.setdefault(a, b)

    sweep = [v for k, v in _tokens("\n".join(sections.get("SWEEP", [])))
             if k == "str"]
    sweep_name = sweep[0] if sweep else "time"

    groups = {}
    tr = list(_tokens("\n".join(sections.get("TRACE", []))))
    i = 0
    while i < len(tr):
        k, v = tr[i]
        if (k == "str" and i + 2 < len(tr) and tr[i + 1] == ("word", "GROUP")):
            n = int(tr[i + 2][1])
            members, j = [], i + 3
            while len(members) < n and j < len(tr):
                if tr[j][0] == "str":
                    members.append(tr[j][1])
                    j += 2            # skip the member's type name
                else:
                    j += 1
            groups[v] = members
            i = j
        else:
            i += 1

    known = {sweep_name, *groups}
    for line in sections.get("TRACE", []):         # first string = trace name
        first = next((v for k, v in _tokens(line) if k == "str"), None)
        if first:
            known.add(first)
    known.update(m for ms in groups.values() for m in ms)
    sig = {}
    vals = list(_tokens("\n".join(sections["VALUE"])))
    i = 0
    while i < len(vals):
        k, name = vals[i]
        i += 1
        if k != "str":
            continue
        # optional type string after the name ("n_rmem" "V" 4.4e3): skip it
        if (i + 1 < len(vals) and vals[i][0] == "str" and vals[i][1] not in known
                and _num(vals[i + 1][1]) is not None):
            i += 1
        members = groups.get(name, [name])
        for m in members:
            if i < len(vals) and vals[i][0] == "(":      # complex / tuple: skip
                while i < len(vals) and vals[i][0] != ")":
                    i += 1
                i += 1
                continue
            x = _num(vals[i][1]) if i < len(vals) else None
            if x is None:
                break
            sig.setdefault(m, []).append(x)
            i += 1
    if sweep_name not in sig:
        raise ScrubError(f"PSF file has no sweep variable '{sweep_name}'")
    sig["time"] = sig.pop(sweep_name)
    n = len(sig["time"])
    return ({k: np.asarray(v) for k, v in sig.items() if len(v) == n}, info)


def read_csv_export(text):
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith(("#", ";"))]
    header = [h.strip().strip('"') for h in lines[0].split(",")]
    rows = []
    for ln in lines[1:]:
        vals = [_num(v.strip().strip('"')) for v in ln.split(",")]
        if any(v is None for v in vals):
            continue
        rows.append(vals)
    arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(header):
        raise ScrubError("CSV rows do not match the header")
    xcols = [k for k, h in enumerate(header) if re.search(r"\sX$", h)]
    if xcols:                                   # ViVA "name X, name Y" pairs
        for k in xcols[1:]:
            if not np.allclose(arr[:, k], arr[:, xcols[0]], rtol=0, atol=1e-15):
                raise ScrubError("CSV X columns differ; export on one time base")
        sig = {"time": arr[:, xcols[0]]}
        for k, h in enumerate(header):
            if k not in xcols:
                sig[re.sub(r"\sY$", "", h)] = arr[:, k]
    else:
        sig = {"time": arr[:, 0]}
        for k, h in enumerate(header[1:], start=1):
            sig[h] = arr[:, k]
    return sig, {}


# --------------------------------------------------------------------------
# signal mapping -> canonical columns
# --------------------------------------------------------------------------
def _norm(name):
    s = name.strip().lower()
    s = re.sub(r"^[vi]\((.*)\)$", r"\1", s)      # V(n_rmem) -> n_rmem
    s = re.sub(r"\s+\(.*?\)\s*$", "", s)         # "time (s)" units suffix
    return s.replace("/", ".").lstrip(".")


def _find(sig, *suffixes):
    for name, arr in sig.items():
        n = _norm(name)
        for suf in suffixes:
            if n == suf or n.endswith("." + suf) or n.endswith(suf):
                return arr
    return None


def to_canonical(key, sig, stim):
    t = np.asarray(sig["time"], dtype=float)
    rmem = _find(sig, "n_rmem", "rmem")
    if rmem is None:
        raise ScrubError("Rmem node (n_rmem) not found in the Spectre data; "
                         f"signals found: {', '.join(sorted(sig))}")
    v = [_find(sig, f"n_v{k}") for k in (1, 2, 3)]
    stdp = _find(sig, "n_stdp")
    nan = np.full(len(t), math.nan)
    G = 1.0 / np.asarray(rmem, dtype=float)
    g_vol = sum(v) if all(a is not None for a in v) else nan
    st = stdp if stdp is not None else np.zeros(len(t))
    if key == "v2":
        nv = _find(sig, "n_nv")
        g_ret = (nv + st) if nv is not None else nan
        x = nan
    else:
        xn = _find(sig, "n_x")
        p = V3Params()
        if xn is not None:
            g_ret = np.array([p.G_of_x(float(a)) for a in xn]) + st
            x = np.clip(xn, 0.0, 1.0)
        else:
            g_ret, x = nan, nan
    drive = _find(sig, "x1:igsc", ":igsc")
    measured = drive is not None
    if not measured:              # never substitute the stimulus: compare.py
        drive = nan               # must report "drive not measured"
    iread = _find(sig, "vread:p", "vread")
    iread = np.abs(iread) if iread is not None else common.V_READ * G
    cols = {"t_s": t, "drive_A": np.asarray(drive, dtype=float), "G_S": G,
            "R_ohm": np.asarray(rmem, dtype=float), "i_read_A": iread,
            "G_ret_S": g_ret, "G_vol_S": g_vol,
            "stdp_S": stdp if stdp is not None else nan, "x_Li": x}
    return cols, measured


# --------------------------------------------------------------------------
# provenance + leak check
# --------------------------------------------------------------------------
def version_tag(info):
    m = re.match(r"\s*(\d+)\.(\d+)", info.get("version", "") or "")
    return f"Spectre {m.group(1)}.{m.group(2)}" if m else "Spectre"


_MONTHS = r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
FORBIDDEN = [
    ("e-mail", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("user@host", re.compile(r"\w@\w")),
    ("IPv4", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("path", re.compile(r"(^|[\s\"'=])(/[\w.-]+/|~/|[A-Za-z]:[\\/]|\\\\)")),
    ("host name", re.compile(r"\b[\w-]+\.(edu|com|org|net|local|lan|internal|gov|ac\.\w\w)\b", re.I)),
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}:\d{2}\b|\b(" + _MONTHS + r")\b", re.I)),
    ("keyword", re.compile(r"\b(host|hostname|user|username|login|home|design|cwd|pwd)\b", re.I)),
]
_PROV_OK = re.compile(r"^# source: Spectre( \d+\.\d+)?; model: v[23]; "
                      r"stimulus_set: [\w.-]+ [0-9a-f]{16}$")
_ROW_OK = re.compile(r"^(nan|[-+]?\d+(\.\d+)?(e[-+]?\d+)?)(,(nan|[-+]?\d+(\.\d+)?(e[-+]?\d+)?))*$",
                     re.I)


def _local_identifiers():
    ids = []
    for f in (getpass.getuser, socket.gethostname):
        try:
            v = (f() or "").strip()
        except Exception:
            v = ""
        if len(v) >= 3:
            ids.append(v.lower())
            ids.append(v.split(".")[0].lower())
    return [i for i in set(ids) if len(i) >= 3]


def check_clean(lines, extra_ids=()):
    """Raise ScrubError if any output line could identify a host/user/path."""
    ids = list(extra_ids) + _local_identifiers()
    if not _PROV_OK.match(lines[0]):
        raise ScrubError(f"provenance line not in the neutral form: {lines[0]!r}")
    if lines[1] != ",".join(common.COLUMNS):
        raise ScrubError("unexpected header line")
    for k, ln in enumerate(lines):
        if k >= 2 and not _ROW_OK.match(ln):
            raise ScrubError(f"line {k+1} is not purely numeric")
        probe = ln if k else ln.split("stimulus_set:")[0]
        for label, rx in FORBIDDEN:
            if k >= 2 and label in ("keyword", "date", "host name", "path"):
                continue                 # numeric rows cannot hold these
            if rx.search(probe):
                raise ScrubError(f"line {k+1} matches a {label} pattern; "
                                 "refusing to write")
        low = ln.lower()
        for ident in ids:
            if k < 2 and ident in low:
                raise ScrubError(f"line {k+1} contains a local identifier; "
                                 "refusing to write")


def render(key, cols, info):
    prov = (f"# source: {version_tag(info)}; model: {key}; stimulus_set: "
            f"{stimuli.STIMULUS_SET_VERSION} {stimuli.fingerprint()[:16]}")
    data = np.column_stack([np.asarray(cols[c], dtype=float)
                            for c in common.COLUMNS])
    lines = [prov, ",".join(common.COLUMNS)]
    for row in data:
        lines.append(",".join("nan" if math.isnan(v) else f"{v:.10g}"
                              for v in row))
    return lines


def wall_from_log(text):
    pats = [r"tran\S*\s+analysis.*?elapsed\s*=\s*([\d.]+)\s*(ms|us|s)\b",
            r"Time used:.*?elapsed\s*=\s*([\d.]+)\s*(ms|us|s)\b"]
    for p in pats:
        m = re.search(p, text, re.I | re.S)
        if m:
            return float(m.group(1)) * {"s": 1.0, "ms": 1e-3, "us": 1e-6}[m.group(2).lower()]
    return None


# --------------------------------------------------------------------------
def scrub_one(key, stim_name, src, log=None, wall_s=None, out_root=None,
              extra_ids=()):
    stim = stimuli.BY_NAME[stim_name]
    text = Path(src).read_text(encoding="utf-8", errors="replace")
    if "VALUE" in text and "HEADER" in text:
        sig, info = read_psf_ascii(text)
    else:
        sig, info = read_csv_export(text)
    cols, measured = to_canonical(key, sig, stim)
    if cols["t_s"][-1] < stim.t_stop * (1 - 1e-6):
        raise ScrubError(f"transient stops at {cols['t_s'][-1]:g} s, "
                         f"stimulus needs {stim.t_stop:g} s")
    lines = render(key, cols, info)
    check_clean(lines, extra_ids)
    if wall_s is None and log:
        wall_s = wall_from_log(Path(log).read_text(encoding="utf-8",
                                                   errors="replace"))
    root = Path(out_root) if out_root else common.DATA
    dest = root / "spectre" / key / f"{stim_name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    common.write_meta(dest.with_suffix(".json"), {
        "source": version_tag(info), "model": key, "stimulus": stim_name,
        "stimulus_set": stimuli.STIMULUS_SET_VERSION,
        "n_points": int(len(cols["t_s"])), "drive_measured": bool(measured),
        "wall_s": None if wall_s is None else round(float(wall_s), 4)})
    return dest


def _batch(root):
    done = []
    for key in common.MODELS:
        for st in stimuli.STIMULI:
            d = Path(root) / key
            cands = [d / f"{st.name}.raw" / "tran.tran", d / f"{st.name}.csv"]
            src = next((c for c in cands if c.is_file()), None)
            if src is None:
                continue
            log = d / f"{st.name}.log"
            done.append(scrub_one(key, st.name, src,
                                  log=log if log.is_file() else None))
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", help="PSF-ascii tran file or CSV export")
    ap.add_argument("--model", choices=list(common.MODELS))
    ap.add_argument("--stimulus", choices=list(stimuli.BY_NAME))
    ap.add_argument("--log", help="Spectre log (for the transient wall time)")
    ap.add_argument("--wall-s", type=float, help="wall time if no log")
    ap.add_argument("--dir", help="batch: <dir>/<model>/<stim>.raw/tran.tran")
    args = ap.parse_args(argv)
    try:
        if args.dir:
            done = _batch(args.dir)
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
