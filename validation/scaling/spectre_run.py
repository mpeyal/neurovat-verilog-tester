"""Run one Spectre deck and time it (single job, single process).

Wall time is measured around the process and the log's own figures are
parsed: the transient alone ("Total time required for tran analysis") and
the whole run ("Time used", parse + AHDL compile + DC + transient). The
CPU / elapsed ratio of the transient is kept as the thread check (a
multi-threaded run would show CPU >> elapsed).
"""
import re
import shutil
import subprocess
import time
from pathlib import Path

TIMEOUT_S = 900.0
_UNIT = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def _pair(text, label):
    m = re.search(label + r"[^\n]*?CPU\s*=\s*([\d.]+)\s*(ms|us|ns|s)\b[^\n]*?"
                  r"elapsed\s*=\s*([\d.]+)\s*(ms|us|ns|s)\b", text)
    if not m:
        return None, None
    return (float(m.group(1)) * _UNIT[m.group(2)],
            float(m.group(3)) * _UNIT[m.group(4)])


def parse_log(text):
    cpu_tr, el_tr = _pair(text, r"Total time required for tran analysis")
    cpu_all, el_all = _pair(text, r"Time used:")
    m = re.search(r"Number of accepted tran steps\s*=\s*(\d+)", text)
    thr = re.findall(r"^[^\n]*\bthread[^\n]*$", text, re.I | re.M)
    return {"tran_cpu_s": cpu_tr, "tran_elapsed_s": el_tr,
            "total_cpu_s": cpu_all, "total_elapsed_s": el_all,
            "accepted_steps": int(m.group(1)) if m else None,
            "thread_lines": [ln.strip()[:120] for ln in thr][:4]}


def run(deck, timeout=TIMEOUT_S, keep_raw=False, extra=()):
    """-> dict of timings; ``raw`` = path of the PSF tran file if kept."""
    deck = Path(deck)
    stem = deck.with_suffix("")
    raw, log = Path(str(stem) + ".raw"), Path(str(stem) + ".log")
    if raw.exists():
        shutil.rmtree(raw, ignore_errors=True)
    cmd = ["spectre", "+escchars", "-format", "psfascii", "-raw", raw.name,
           "+log", log.name, *extra, deck.name]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=deck.parent, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=timeout)
        rc, timed_out = p.returncode, False
    except subprocess.TimeoutExpired:
        rc, timed_out = None, True
    wall = time.perf_counter() - t0
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    out = {"rc": rc, "timed_out": timed_out, "wall_s": wall, **parse_log(text)}
    tr = None
    if raw.is_dir():
        c = sorted(raw.glob("*.tran*"))
        tr = c[0] if c else None
    if keep_raw and tr is not None:
        out["raw"] = str(tr)
    else:
        shutil.rmtree(raw, ignore_errors=True)
    return out


def cleanup(deck):
    stem = Path(deck).with_suffix("")
    shutil.rmtree(Path(str(stem) + ".raw"), ignore_errors=True)
