"""Read Spectre transient results: PSF-ASCII files and CSV exports.

A PSF-ASCII transient file is what `spectre -format psfascii -raw <dir>` writes
(`<dir>/<analysis>.tran.tran`).  It has HEADER, TYPE, SWEEP, TRACE and VALUE
sections; VALUE holds one block per time point: the sweep value, then each
saved signal (a GROUP name is followed by its members in TRACE order).

    sig, info = read_psf_ascii(text)       # {"time": array, "<name>": array}
    sig, info = read_csv_export(text)      # Virtuoso / ViVA CSV export
    sig, info = load_waveform(path)        # either, by content

Signal names are kept as Spectre writes them ("n_p", "X1:p", "X1.ws").
`find_signal(sig, "X1:p")` looks a name up loosely (V(...)/I(...) wrappers,
hierarchy separators, case).  Pure Python + NumPy, no GUI.
"""

import re
from pathlib import Path

import numpy as np


class PSFError(ValueError):
    """The text is not a readable PSF-ASCII transient file or CSV export."""


_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|([()])|([^\s()"]+)')


def _tokens(text):
    for m in _TOKEN.finditer(text):
        if m.group(1) is not None:
            yield ("str", m.group(1))
        elif m.group(2) is not None:
            yield (m.group(2), m.group(2))
        else:
            yield ("word", m.group(3))


def _num(word):
    try:
        return float(word)
    except (TypeError, ValueError):
        return None


def read_psf_ascii(text):
    """Parse a PSF-ASCII transient file -> (signals, info).

    signals: {"time": ndarray, name: ndarray, ...}, every array the same length.
    info: {"simulator": ..., "version": ...} when the header has them.
    """
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
        raise PSFError("not a PSF-ASCII file (no VALUE section)")

    info = {}
    head = list(_tokens("\n".join(sections.get("HEADER", []))))
    for (k1, a), (k2, b) in zip(head, head[1:]):
        if k1 == "str" and k2 == "str" and a in ("simulator", "version"):
            info.setdefault(a, b)

    sweep = [v for k, v in _tokens("\n".join(sections.get("SWEEP", []))) if k == "str"]
    sweep_name = sweep[0] if sweep else "time"

    groups = {}
    tr = list(_tokens("\n".join(sections.get("TRACE", []))))
    i = 0
    while i < len(tr):
        k, v = tr[i]
        if k == "str" and i + 2 < len(tr) and tr[i + 1] == ("word", "GROUP"):
            n = int(tr[i + 2][1])
            members, j = [], i + 3
            while len(members) < n and j < len(tr):
                if tr[j][0] == "str":
                    members.append(tr[j][1])
                    j += 2                     # skip the member's type name
                else:
                    j += 1
            groups[v] = members
            i = j
        else:
            i += 1

    known = {sweep_name, *groups}
    for line in sections.get("TRACE", []):     # first string = trace name
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
        # optional type string after the name ("n_p" "V" 0.1): skip it
        if (i + 1 < len(vals) and vals[i][0] == "str" and vals[i][1] not in known
                and _num(vals[i + 1][1]) is not None):
            i += 1
        for m in groups.get(name, [name]):
            if i < len(vals) and vals[i][0] == "(":          # complex: skip
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
        raise PSFError(f"PSF file has no sweep variable '{sweep_name}'")
    sig["time"] = sig.pop(sweep_name)
    n = len(sig["time"])
    if n < 2:
        raise PSFError("PSF file holds fewer than two time points")
    return {k: np.asarray(v, dtype=float) for k, v in sig.items() if len(v) == n}, info


def read_csv_export(text):
    """Parse a CSV export: first column time, or ViVA 'name X, name Y' pairs."""
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith(("#", ";"))]
    if len(lines) < 3:
        raise PSFError("CSV holds no data rows")
    header = [h.strip().strip('"') for h in lines[0].split(",")]
    rows = []
    for ln in lines[1:]:
        vals = [_num(v.strip().strip('"')) for v in ln.split(",")]
        if len(vals) == len(header) and all(v is not None for v in vals):
            rows.append(vals)
    arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise PSFError("CSV rows do not match the header")
    xcols = [k for k, h in enumerate(header) if re.search(r"\sX$", h)]
    if xcols:
        for k in xcols[1:]:
            if not np.allclose(arr[:, k], arr[:, xcols[0]], rtol=0, atol=1e-15):
                raise PSFError("CSV X columns differ; export on one time base")
        sig = {"time": arr[:, xcols[0]]}
        for k, h in enumerate(header):
            if k not in xcols:
                sig[re.sub(r"\sY$", "", h)] = arr[:, k]
    else:
        sig = {"time": arr[:, 0]}
        for k, h in enumerate(header[1:], start=1):
            sig[h] = arr[:, k]
    return sig, {}


def parse_waveform(text):
    """PSF-ASCII or CSV, decided by content."""
    if re.search(r"^\s*VALUE\s*$", text, re.M) and re.search(r"^\s*HEADER\s*$", text, re.M):
        return read_psf_ascii(text)
    return read_csv_export(text)


def load_waveform(path):
    """Load a PSF-ASCII file, a PSF raw directory (its *.tran file) or a CSV."""
    p = Path(path)
    if p.is_dir():
        trans = sorted(p.glob("*.tran")) + sorted(p.glob("*.tran.tran"))
        trans = sorted(set(trans))
        if not trans:
            raise PSFError(f"no *.tran file in {p}")
        p = trans[0]
    return parse_waveform(p.read_text(encoding="utf-8", errors="replace"))


def _norm(name):
    s = name.strip().lower()
    s = re.sub(r"^[vi]\((.*)\)$", r"\1", s)        # V(n_p) -> n_p
    s = re.sub(r"\s+\(.*?\)\s*$", "", s)            # "time (s)"
    return s.replace("/", ".").lstrip(".")


def find_signal(sig, name):
    """Look a signal up by exact name, then loosely; None if absent."""
    if name in sig:
        return sig[name]
    want = _norm(name)
    for k, v in sig.items():
        if _norm(k) == want:
            return v
    for k, v in sig.items():
        n = _norm(k)
        if n.endswith("." + want) or n.endswith(":" + want):
            return v
    return None
