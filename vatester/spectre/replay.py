"""Layer 3: replay a training run's write schedule in Spectre.

A short unsupervised STDP training (neurovat.masquelier.run, the stream
network of the Trainer) runs with ECFET v3 device synapses in physical time.
The device's program() and peek() are wrapped on the instance (engine and
class unchanged), which records

- every gate write pulse the engine applied (synapse, start, width, current),
- every conductance read the engine made (time, G of every synapse).

The pulses then drive one ECFET instance per synapse in Spectre, each as its
own PWL gate current, and the conductance the ENGINE READ during training is
compared with Spectre at the same times (Layer-1 rule per synapse, range =
that synapse's Spectre G range).  This checks what device-level stimuli
cannot: that the engine schedules and reads the device correctly.

    rec = record_training(P, params)             # engine run + schedule + reads
    waves, t_stop = gate_waves(rec)              # one PWL body per synapse
    rows = score_replay(rec, sig)                # sig: imported PSF signals
"""

import os
import sys
from dataclasses import asdict, dataclass, field

import numpy as np

from . import compare as C
from .psf import find_signal

TICK = 1e-7          # s: PWL times on a 0.1 us integer grid
EDGE_TICKS = 10      # 1 us PWL edges
WIDTH = 10e-3        # s, write pulse width
WINDOW_MS = 10.0     # write window (pulses on one synapse never overlap)
I_UNIT = 50e-12      # A, the device's usual drive level (calibration start)


def _ensure_engine_path():
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


@dataclass
class Recording:
    n: int
    pulses: list                  # (synapse, t_start, width, amp)
    reads_t: np.ndarray           # engine read times
    reads_g: np.ndarray           # (n_reads, n) full G the engine read
    g_start: float
    t_end: float
    params: dict = field(default_factory=dict)      # V3Params as a dict
    info: dict = field(default_factory=dict)


def _instrument(dev, log):
    program, peek = dev.program, dev.peek

    def program_logged(rows, cols, amps, width, t=0.0):
        idx = dev._flat(rows, cols)
        dev._idle(idx, float(t))                  # what program() does first
        start = np.maximum(np.maximum(float(t), dev.busy[idx]), dev.t_dev[idx])
        for c, s, a in zip(np.asarray(cols).ravel(), start,
                           np.asarray(amps, float).ravel()):
            log["pulses"].append((int(c), float(s), float(width), float(a)))
        return program(rows, cols, amps, width, t)

    def peek_logged(t):
        g = np.asarray(peek(t), float).ravel().copy()
        log["reads"][round(float(t), 9)] = g
        return peek(t)

    dev.program, dev.peek = program_logged, peek_logged


def record_training(afferents=16, seconds=4.0, seed=1, pattern_frac=0.25,
                    threshold_per_aff=0.25, v3_overrides=None):
    """Run the stream training with an ECFET v3 array and record it."""
    _ensure_engine_path()
    from neurovat.devices import vector_twin_factory
    from neurovat.masquelier import MasquelierParams, calibrate_write, run
    from simulation_core.model_v3 import EcfetV3, V3Params

    p3 = V3Params.paper(**{"A_stdp": 0.0, **(v3_overrides or {})})
    fac = vector_twin_factory(lambda: EcfetV3(p3), "physical", "full")
    if fac.kind != "vector":
        raise RuntimeError("the replay needs the vectorised ECFET v3 array "
                           "(an edited or noisy v3 falls back to the scalar adapter)")
    P = MasquelierParams(n_afferents=afferents, n_pattern=afferents // 2,
                         threshold=threshold_per_aff * afferents, n_repeats=1,
                         spacing="even", pattern_frac=pattern_frac,
                         base_s=seconds, seed=seed)
    dev = fac(1, P.n_afferents)
    cal = calibrate_write(dev, P, WIDTH, 30 * I_UNIT)
    g_start = dev.g_min + P.w_init * (dev.g_max - dev.g_min)
    dev.set_conductance(np.full(dev.shape, g_start))
    log = {"pulses": [], "reads": {}}
    _instrument(dev, log)
    r = run(P, device=dev, write_window_ms=WINDOW_MS, read_ms=1.0,
            **{q: cal[q] for q in ("write_gain", "dep_gain", "pot_sign", "pulse_width")})
    ts = np.array(sorted(log["reads"]))
    g = np.array([log["reads"][t] for t in ts])
    pulses = sorted(log["pulses"], key=lambda p: (p[1], p[0]))
    return Recording(n=P.n_afferents, pulses=pulses, reads_t=ts, reads_g=g,
                     g_start=float(g_start), t_end=float(dev.clock),
                     params=asdict(p3),
                     info={"post_spikes": int(r["post_spikes"]),
                           "writes": int(r["writes"]), "n_pulses": len(pulses),
                           "afferents": afferents, "seconds": seconds, "seed": seed})


def _edges(pulses):
    """[(tick, value after the edge)] of one synapse, merged."""
    ev = {}
    for s, w, a in pulses:
        ev[s] = ev.get(s, 0.0) + a
        ev[s + w] = ev.get(s + w, 0.0) - a
    out, acc, prev = [], 0.0, 0.0
    for e in sorted(ev):
        acc += ev[e]
        if abs(acc) < 1e-24:
            acc = 0.0
        if acc != prev:
            out.append((e, acc))
            prev = acc
    return out


def _pwl_wave(pulses, stop_tick, per_line=4):
    pts, prev = [(0, 0.0)], 0.0
    for e, v in _edges(pulses):
        if e <= pts[-1][0]:
            raise ValueError("pulse edges closer than the 1 us PWL edge")
        pts += [(e, prev), (e + EDGE_TICKS, v)]
        prev = v
    if pts[-1][0] >= stop_tick:
        raise ValueError("last edge after t_stop")
    pts.append((stop_tick, prev))
    chunks = [" ".join(f"{t * TICK:.7f} {i:.9g}" for t, i in pts[k:k + per_line])
              for k in range(0, len(pts), per_line)]
    return "[ " + " \\\n    ".join(chunks) + " ]"


def gate_waves(rec):
    """One Spectre PWL body per synapse; returns (waves, t_stop)."""
    t_stop = round(rec.t_end + 0.05, 6)
    stop = int(round(t_stop / TICK))
    per = {k: [] for k in range(rec.n)}
    for c, s, w, a in rec.pulses:
        per[c].append((int(round(s / TICK)), int(round(w / TICK)), a))
    return [_pwl_wave(per[k], stop) for k in range(rec.n)], t_stop


def spectre_g(sig, k, v_read=0.01):
    """Synapse k's conductance in the replay result: |I(vread<k>)| / V_read
    (older decks saved the v3 Rmem node n_rm<k> = R instead)."""
    i = find_signal(sig, f"vread{k}:p")
    if i is not None:
        return np.abs(np.asarray(i, float)) / v_read
    r = find_signal(sig, f"n_rm{k}")
    if r is None:
        raise KeyError(f"neither vread{k}:p nor n_rm{k} in the Spectre result")
    return 1.0 / np.asarray(r, float)


def score_replay(rec, sig, v_read=0.01):
    """Per synapse: engine read G against Spectre G (read current / V_read) at
    the engine's read times, under the Layer-1 waveform rule."""
    t_ref = np.asarray(sig["time"], float)
    rows = []
    for k in range(rec.n):
        g_ref = spectre_g(sig, k, v_read)
        m = C.waveform_metrics(rec.reads_t, rec.reads_g[:, k], t_ref, g_ref)
        rows.append({"synapse": k, "pulses": sum(1 for p in rec.pulses if p[0] == k),
                     "reads": m["n_points"], "g_range_uS": 1e6 * m["range"],
                     "nrmse_pct": 100 * m["nrmse"], "max_err_pct": 100 * m["maxerr"],
                     "verdict": C.verdict(m)})
    return rows
