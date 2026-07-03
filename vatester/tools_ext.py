"""Extra Tools-menu utilities for NeuroVAT (pure logic, no GUI).

Kept GUI-free so the same functions back both the Tools menu and the control
bridge, and so they can be unit-tested headlessly:

  * check_consistency  - twin (Python) params vs the .va `parameter` declarations
  * openvaf_compile    - compile a .va with OpenVAF (OSDI) if it's installed
  * fit_ltp_ltd        - fit the LTP/LTD nonlinearity of a measured G-vs-pulse CSV
"""

import math
import os
import re
import shutil
import subprocess
import sys

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")        # strip color codes for the log


# --------------------------------------------------------------------------
# 1) twin <-> .va parameter consistency
# --------------------------------------------------------------------------
def check_consistency(va_files, twin_defaults, twin_file, rel_tol=1e-3):
    """Compare each mapped .va's numeric `parameter` values against its Python
    twin's dataclass defaults.

    va_files      : iterable of VaFile (needs .name, .model_key, .params, .raw_params)
    twin_defaults : {model_key: {param_name: default_value}}  (numeric fields only)
    twin_file     : {model_key: "ecfet/model_x.py"}
    Returns {"reports": [ ... ], "n_mismatch": int, "n_unmapped": int}.
    """
    reports = []
    n_mismatch = n_unmapped = 0
    for v in va_files:
        key = v.model_key
        if not key or key not in twin_defaults:
            n_unmapped += 1
            reports.append({"va": v.name, "model_key": key or None,
                            "status": "unmapped",
                            "note": "no Python twin mapped to this .va"})
            continue
        tdef = twin_defaults[key]
        va_num = dict(v.params)                     # name -> float (parsable only)
        shared = set(va_num) & set(tdef)
        mism = []
        for name in sorted(shared):
            tv = tdef[name]
            if not isinstance(tv, (int, float)) or isinstance(tv, bool):
                continue
            vv = va_num[name]
            denom = max(abs(tv), abs(vv), 1e-30)
            if abs(vv - tv) / denom > rel_tol:
                mism.append({"param": name, "va": vv, "twin": tv})
        only_va = sorted(set(v.raw_params) - set(tdef))
        only_twin = sorted(set(tdef) - set(v.raw_params))
        status = "mismatch" if mism else "ok"
        if mism:
            n_mismatch += 1
        reports.append({
            "va": v.name, "model_key": key, "twin": twin_file.get(key),
            "status": status, "mismatched": mism,
            "only_in_va": only_va, "only_in_twin": only_twin,
            "n_shared": len(shared)})
    return {"reports": reports, "n_mismatch": n_mismatch,
            "n_unmapped": n_unmapped}


def format_consistency(result):
    """Human-readable lines for the log console."""
    lines = []
    for r in result["reports"]:
        if r["status"] == "unmapped":
            lines.append(f"  [--] {r['va']}: {r['note']}")
            continue
        tag = "OK" if r["status"] == "ok" else "MISMATCH"
        lines.append(f"  [{tag}] {r['va']} <-> {r['twin']} "
                     f"({r['n_shared']} shared params)")
        for m in r["mismatched"]:
            lines.append(f"why        {m['param']}: .va={m['va']:.6g} "
                         f"twin={m['twin']:.6g}")
        if r["only_in_va"]:
            lines.append(f"why        only in .va: {', '.join(r['only_in_va'])}")
    head = (f"twin/.va check: {result['n_mismatch']} mismatched, "
            f"{result['n_unmapped']} unmapped, "
            f"{len(result['reports'])} file(s)")
    return head, lines


# --------------------------------------------------------------------------
# 2) OpenVAF compile check
# --------------------------------------------------------------------------
def find_openvaf(explicit=None):
    """Locate the OpenVAF executable (env NVAT_OPENVAF, arg, or PATH)."""
    cands = [explicit, os.environ.get("NVAT_OPENVAF"),
             "openvaf", "openvaf-reloaded", "openvaf-r", "openvaf.exe"]
    for c in cands:
        if not c:
            continue
        exe = c if os.path.isabs(c) and os.path.isfile(c) else shutil.which(c)
        if exe:
            return exe
    # fallback: standard per-user install dir (so it's found even if the app was
    # launched before a PATH update propagated to the process env)
    for base in (os.environ.get("LOCALAPPDATA"), os.path.expanduser("~")):
        if base:
            p = os.path.join(base, "Programs", "OpenVAF", "openvaf.exe")
            if os.path.isfile(p):
                return p
    return None


def openvaf_compile(va_path, exe=None, workdir=None, timeout=120):
    """Compile a .va -> .osdi with OpenVAF. Degrades gracefully if not installed.

    Returns {"available", "ok", "returncode", "output", "osdi", "exe"}.
    """
    exe = find_openvaf(exe)
    if not exe:
        return {"available": False, "ok": False, "exe": None,
                "output": ("OpenVAF not found. Install from "
                           "https://openvaf.semimod.de/ and put it on PATH, "
                           "or set NVAT_OPENVAF=/path/to/openvaf.")}
    if not os.path.isfile(va_path):
        return {"available": True, "ok": False, "exe": exe,
                "output": f"no such .va file: {va_path}"}
    cwd = workdir or os.path.dirname(os.path.abspath(va_path))
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    env = {**os.environ, "NO_COLOR": "1", "CLICOLOR": "0"}   # plain-text output
    try:
        p = subprocess.run([exe, va_path], cwd=cwd, capture_output=True,
                           text=True, timeout=timeout, creationflags=flags,
                           env=env)
    except subprocess.TimeoutExpired:
        return {"available": True, "ok": False, "exe": exe,
                "output": f"OpenVAF timed out after {timeout}s"}
    except OSError as e:
        return {"available": True, "ok": False, "exe": exe,
                "output": f"failed to launch OpenVAF: {e}"}
    out = _ANSI_RE.sub("", ((p.stdout or "") + (p.stderr or "")).strip())
    osdi = os.path.splitext(va_path)[0] + ".osdi"
    osdi = osdi if os.path.isfile(osdi) else None
    return {"available": True, "ok": p.returncode == 0, "exe": exe,
            "returncode": p.returncode, "output": out[-4000:], "osdi": osdi}


# --------------------------------------------------------------------------
# 3) LTP/LTD nonlinearity fit from a measured CSV
# --------------------------------------------------------------------------
def read_curve_csv(path):
    """Read a CSV of conductance-per-pulse. Accepts one column (G) or two
    (pulse_index, G); ignores a non-numeric header row. Returns list[float] of
    G in the file's own units (assumed uS unless values look like siemens)."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().replace(",", " ").replace(";", " ")
            if not line:
                continue
            parts = line.split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue                    # header / comment
            rows.append(vals)
    if not rows:
        raise ValueError("no numeric rows found in CSV")
    width = max(len(r) for r in rows)
    g = [r[-1] for r in rows] if width >= 2 else [r[0] for r in rows]
    return g


def _fit_branch(g):
    """Fit g(p) = g0 + span*(1-exp(-p*B/(N-1)))/(1-exp(-B)) over pulses p=0..N-1.

    B is the nonlinearity (B->0 linear, large B = saturating). Solved by a 1-D
    golden-section search on B with (g0, span) from the curve endpoints. Returns
    (B, g0, span, rmse, fitted[list]). Pure numpy-free.
    """
    n = len(g)
    if n < 3:
        return None
    xs = [p / (n - 1) for p in range(n)]        # 0..1

    def model(B):
        if abs(B) < 1e-6:
            shape = xs[:]                        # linear limit
        else:
            denom = 1.0 - math.exp(-B)
            shape = [(1.0 - math.exp(-B * x)) / denom for x in xs]
        g0 = g[0]
        span = g[-1] - g[0]
        fit = [g0 + span * s for s in shape]
        rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(fit, g)) / n)
        return rmse, g0, span, fit

    lo, hi = 1e-3, 12.0                          # golden-section on B
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = model(c)[0], model(d)[0]
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = model(c)[0]
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = model(d)[0]
    B = (a + b) / 2
    rmse, g0, span, fit = model(B)
    return B, g0, span, rmse, fit


def fit_ltp_ltd(g, split=None):
    """Fit LTP (rising) and, if present, LTD (falling) branches of a G-per-pulse
    series. If `split` is None it auto-splits at the peak. Returns a report dict
    with per-branch nonlinearity B, dynamic range, asymmetry, and RMSE."""
    n = len(g)
    if n < 4:
        raise ValueError("need at least 4 points to fit")
    if split is None:
        split = g.index(max(g))                 # peak = LTP->LTD turn
    ltp = g[:split + 1]
    ltd = g[split:]
    out = {"n_points": n, "split_at": split,
           "G_min": min(g), "G_max": max(g),
           "dynamic_range": (max(g) / min(g)) if min(g) > 0 else None}
    fp = _fit_branch(ltp) if len(ltp) >= 3 else None
    # LTD is a decreasing saturating curve = g0 + (negative span)*S(x); fit it
    # directly (reversing would flip the curvature and mis-fit B).
    fd = _fit_branch(ltd) if len(ltd) >= 3 else None
    if fp:
        out["ltp"] = {"nonlinearity_B": fp[0], "rmse": fp[3],
                      "n_points": len(ltp)}
    if fd:
        out["ltd"] = {"nonlinearity_B": fd[0], "rmse": fd[3],
                      "n_points": len(ltd)}
    if fp and fd:
        out["asymmetry_B"] = abs(fp[0] - fd[0])
    return out


def format_fit(result):
    lines = [f"  points={result['n_points']}  split@pulse {result['split_at']}  "
             f"G {result['G_min']:.3g}..{result['G_max']:.3g}"
             + (f"  (dynamic range x{result['dynamic_range']:.1f})"
                if result.get("dynamic_range") else "")]
    for br in ("ltp", "ltd"):
        if br in result:
            b = result[br]
            lines.append(f"  {br.upper()}: nonlinearity B={b['nonlinearity_B']:.2f}"
                         f"  (RMSE {b['rmse']:.3g}, {b['n_points']} pts)")
    if "asymmetry_B" in result:
        lines.append(f"  LTP/LTD asymmetry |dB| = {result['asymmetry_B']:.2f}")
    lines.append("  (B->0 = linear/ideal; larger B = more saturating. Raise the "
                 "twin's nu_p/nu_d to increase B, n_states for dynamic range.)")
    return lines
