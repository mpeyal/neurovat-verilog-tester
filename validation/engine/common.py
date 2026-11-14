"""Shared pieces of the engine cross-check: input preparation that both
engines can take bit-for-bit, and the comparison metrics."""
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from neurovat.masquelier import Stream  # noqa: E402

DATA = ROOT / "validation" / "data" / "engine"
FIGURES = ROOT / "validation" / "figures" / "engine"


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------
def on_grid(k, dt):
    """Times k*dt (ms) nudged up by at most a few ulps so that
    floor(t / dt) == k holds exactly (NeuroVAT bins by floor(t / dt))."""
    k = np.asarray(k, np.int64)
    t = k * float(dt)
    for _ in range(8):
        bad = np.floor(t / dt).astype(np.int64) < k
        if not bad.any():
            break
        t[bad] = np.nextafter(t[bad], np.inf)
    assert (np.floor(t / dt).astype(np.int64) == k).all()
    return t


def bins(t, dt):
    return np.floor(np.asarray(t, float) / dt).astype(np.int64)


def dedupe(aff, k):
    """Keep the first spike of each (afferent, bin); returns a boolean keep
    mask in the original order (Brian2's SpikeGeneratorGroup refuses two
    spikes of one neuron in one time step)."""
    order = np.lexsort((np.arange(aff.size), aff, k))
    a, b = aff[order], k[order]
    first = np.ones(order.size, bool)
    first[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
    keep = np.zeros(aff.size, bool)
    keep[order[first]] = True
    return keep


def with_spikes(S, aff, t, kind=None):
    """A copy of stream S carrying the given spikes (sorted by time); labels,
    members and section layout are kept."""
    aff = np.asarray(aff, np.int32)
    t = np.asarray(t, float)
    kind = np.zeros(aff.size, np.int8) if kind is None else np.asarray(kind, np.int8)
    o = np.argsort(t, kind="stable")
    aff, t, kind = aff[o], t[o], kind[o]
    n_steps = S.ptr.size - 1
    step = np.minimum((t // 1.0).astype(np.int64), n_steps - 1)
    ptr = np.searchsorted(step, np.arange(n_steps + 1))
    return replace(S, aff=aff, t=t, kind=kind, ptr=ptr)


def grid_stream(S, dt):
    """Stream S with every spike moved to the start of its dt bin and
    same-bin duplicates of an afferent dropped. Both engines then see the
    SAME input. Returns (stream, n_moved_ms_total, n_dropped)."""
    k = bins(S.t, dt)
    keep = dedupe(S.aff, k)
    t = on_grid(k[keep], dt)
    moved = float(np.abs(S.t[keep] - t).sum())
    return with_spikes(S, S.aff[keep], t, S.kind[keep]), moved, int((~keep).sum())


def make_raw_stream(aff, t, duration_ms, pattern_ms=50.0):
    """A minimal Stream (no patterns) around explicit spikes."""
    n_sec = int(math.ceil(duration_ms / pattern_ms))
    n_steps = int(round(n_sec * pattern_ms))
    S = Stream(ptr=np.zeros(n_steps + 1, np.int64), aff=np.zeros(0, np.int32),
               t=np.zeros(0), label=np.full(n_sec, -1),
               section_steps=int(round(pattern_ms)), duration_ms=float(n_steps),
               members=None, kind=np.zeros(0, np.int8), extra=np.full(n_sec, -1))
    return with_spikes(S, aff, t)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def match_spikes(ref, test, tol):
    """Greedy one-to-one matching of two sorted spike-time lists within tol.
    Returns dict: n_ref, n_test, matched, match_frac (matched / max count),
    mean_abs_dt, max_abs_dt (over matched pairs)."""
    ref, test = np.sort(np.asarray(ref, float)), np.sort(np.asarray(test, float))
    i = j = 0
    d = []
    while i < ref.size and j < test.size:
        x = test[j] - ref[i]
        if abs(x) <= tol:
            d.append(abs(x))
            i += 1
            j += 1
        elif x < 0:
            j += 1
        else:
            i += 1
    n = max(ref.size, test.size)
    d = np.array(d)
    return {"n_ref": int(ref.size), "n_test": int(test.size),
            "matched": int(d.size), "match_frac": d.size / n if n else 1.0,
            "mean_abs_dt_ms": float(d.mean()) if d.size else 0.0,
            "max_abs_dt_ms": float(d.max()) if d.size else 0.0}


def first_divergence(ref, test, tol):
    """Time (ms) of the first spike at which the two ordered trains stop
    agreeing within tol (by index), or None when they agree throughout."""
    ref, test = np.sort(np.asarray(ref, float)), np.sort(np.asarray(test, float))
    n = min(ref.size, test.size)
    bad = np.nonzero(np.abs(ref[:n] - test[:n]) > tol)[0]
    if bad.size:
        k = bad[0]
        return float(min(ref[k], test[k]))
    if ref.size != test.size:
        return float((ref if ref.size > n else test)[n])
    return None


def nrmse(test, ref, span=None):
    test, ref = np.asarray(test, float), np.asarray(ref, float)
    ok = np.isfinite(test) & np.isfinite(ref)
    if not ok.any():
        return math.nan
    d = test[ok] - ref[ok]
    rng = span if span is not None else float(np.ptp(ref[ok]))
    return float(np.sqrt(np.mean(d * d)) / rng) if rng > 0 else float(np.abs(d).max())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=conv) + "\n", encoding="utf-8")
    return path


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)

    def f(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)
    lines = [",".join(keys)] + [",".join(f(r.get(k)) for k in keys) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
