"""Shared C5 tier-1 pieces: the recorded write schedule, its per-synapse gate
waveform (twin) and PWL (Spectre), and the twin replay of one synapse.

Times are held as integers in units of TICK (0.1 us); the engine's pulses
start on its 0.1 ms neuron grid, so rounding there is exact, and back-to-back
pulses on one synapse (start = previous end) share one edge exactly.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from simulation_core.model_v3 import EcfetV3, V3Params  # noqa: E402
from simulation_core.signals import Waveform  # noqa: E402
from simulation_core.simulator import time_grid  # noqa: E402

DATA = ROOT / "validation" / "data" / "c5"
TICK = 1e-7                 # s, integer time unit
EDGE_TICKS = 10             # PWL edge = 1 us (as Gate A)
DT_MAX = 1e-3               # twin max step = Spectre maxstep (s)


def load(data=DATA):
    data = Path(data)
    meta = json.loads((data / "engine.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(data / "schedule.csv", encoding="utf-8")))
    n = meta["masquelier"]["n_afferents"]
    per = {k: [] for k in range(n)}
    for r in rows:
        s = int(round(float(r["t_start_s"]) / TICK))
        w = int(round(float(r["width_s"]) / TICK))
        per[int(r["synapse"])].append((s, w, float(r["amp_A"])))
    return meta, per


def edges(pulses):
    """[(tick, value after the edge)], sorted, merged (one synapse)."""
    ev = {}
    for s, w, a in pulses:
        ev[s] = ev.get(s, 0.0) + a
        ev[s + w] = ev.get(s + w, 0.0) - a
    out, acc = [], 0.0
    for e in sorted(ev):
        acc += ev[e]
        if abs(acc) < 1e-24:
            acc = 0.0
        out.append((e, acc))
    # drop edges that do not change the value (a pulse continuing at the same amp)
    clean, prev = [], 0.0
    for e, v in out:
        if v != prev:
            clean.append((e, v))
            prev = v
    return clean


def waveform(pulses):
    """simulation_core Waveform with exact (tick-rounded) edges."""
    wf = Waveform([])
    ed = edges(pulses)
    wf.edges = [e * TICK for e, _ in ed]
    wf.values = [v for _, v in ed]
    return wf


def pwl_points(pulses, t_stop):
    pts, prev = [(0, 0.0)], 0.0
    stop = int(round(t_stop / TICK))
    for e, v in edges(pulses):
        if e <= pts[-1][0]:
            raise ValueError("edges closer than the PWL edge time")
        pts.append((e, prev))
        pts.append((e + EDGE_TICKS, v))
        prev = v
    if pts[-1][0] >= stop:
        raise ValueError("last edge after t_stop")
    pts.append((stop, prev))
    return pts


def spectre_pwl(pulses, t_stop, per_line=4):
    pts = pwl_points(pulses, t_stop)
    chunks = [" ".join(f"{t * TICK:.7f} {i:.9g}" for t, i in pts[k:k + per_line])
              for k in range(0, len(pts), per_line)]
    return "wave=[ " + " \\\n    ".join(chunks) + " ]"


def twin_params(meta):
    return V3Params(**{**meta["v3_params"], "Rinit": meta["R_start_ohm"]})


def grid(pulses, t_stop):
    wf = waveform(pulses)
    return wf, time_grid(wf.breakpoints, t_stop, dt_min=1e-6, dt_max=DT_MAX)


def replay(meta, pulses, t_stop):
    """Scalar twin (EcfetV3) on the Gate A time grid -> (t, G_full, G_nv, I)."""
    from validation.common import simulate_on_grid
    wf, ts = grid(pulses, t_stop)
    raw = simulate_on_grid(EcfetV3(twin_params(meta)), wf, ts)
    return np.asarray(ts), raw["G"], raw["G_nv (S)"], raw["i"]
