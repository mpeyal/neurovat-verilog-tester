"""Fixed, versioned VOLTAGE stimulus set "vteam-1" for the VTEAM memristor.

Same construction as stanford-1 (validation/stanford/stimuli.py, whose
VoltageStimulus class is reused): piecewise-linear voltage points in integer
picoseconds, used verbatim by the twin (V linear between points) and by the
Spectre `vsource type=pwl`, trapezoid pulses with 50 ps edges whose
volt-second area is amp * width.

Differences forced by the device (MemTorch defaults = paper Table II "TEAM"):
  * Reads.  v_off = 0.02 V, so a +0.1 V read would move the state.  Reads are
    at V_READ = -0.1 V, inside the dead zone (v_on, v_off) = (-0.2, 0.02) V
    where dx/dt = 0 exactly; the I-V is linear, so G = I/V at the plateau
    middle is the per-pulse read-out.
  * Amplitudes.  The OFF (x up) rate is k_off (V/0.02 - 1): +0.8 V moves
    1.95e-10 m in 10 ns (6.5 % of d).  The ON rate is -10 (V/-0.2 - 1)^3 m/s:
    -0.225 V moves 6.5 % of d in 10 ns, but -1 V crosses the whole 3 nm in
    about 5 ps.  The pulse trains use partial-switching amplitudes; the
    MemTorch hysteresis sine (1 V, 50 MHz) is kept as the fast-switching case.
  * `instance` sets x_init (m), e.g. x_init = d to start in the OFF state.

Change the set only by bumping STIMULUS_SET_VERSION.
"""

import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.stanford.stimuli import (  # noqa: E402
    EDGE_PS, NS, PS, VoltageStimulus, _Builder as _StanfordBuilder, parse_spectre_pwl)

__all__ = ["STIMULI", "BY_NAME", "STIMULUS_SET_VERSION", "V_READ", "PS", "NS",
           "VoltageStimulus", "parse_spectre_pwl", "fingerprint", "get"]

STIMULUS_SET_VERSION = "vteam-1"
V_READ = -0.1                # V, inside the dead zone (v_on, v_off)
READ_PS = 4000
D = 3e-9                     # m, default device length (x_init = D: OFF state)
DT = 0.1e-9                  # s, deck maxstep / twin output grid


class _Builder(_StanfordBuilder):
    def read(self, lead_ps=3 * NS, width_ps=READ_PS, tail_ps=0):
        self.wait(lead_ps)
        t0 = self.t
        self.pulse(V_READ, width_ps)
        self.reads.append(t0 + (width_ps + EDGE_PS) // 2)
        if tail_ps:
            self.wait(tail_ps)


def _train(name, desc, seq, gap_ps=10 * NS, instance=(), dt_max=DT, **params):
    """Write pulses (amp, width_ps), each followed by a read in the gap."""
    b = _Builder()
    b.wait(1 * NS)
    for amp, w in seq:
        t_start = b.t
        b.pulse(amp, w)
        b.read()
        nxt = t_start + w + EDGE_PS + gap_ps
        if nxt <= b.t:
            raise ValueError(f"{name}: gap too short for the read")
        b.wait(nxt - b.t)
    return b.build(name, desc, 5 * NS, dt_max, instance, **params)


OFF_STATE = (("x_init", D),)


def _off_train():
    return _train("off_train", "from ON (x = 0): 20 x +0.8 V / 10 ns OFF pulses "
                  "(6.5 % of d each, reaches the x = d bound at pulse 16), 10 ns gaps, "
                  "-0.1 V read in every gap", [(0.8, 10 * NS)] * 20, amp=0.8, n=20)


def _on_train():
    return _train("on_train", "from OFF (x = d): 20 x -0.225 V / 10 ns ON pulses "
                  "(6.5 % of d each, reaches x = 0 at pulse 16), reads",
                  [(-0.225, 10 * NS)] * 20, instance=OFF_STATE, amp=-0.225, n=20)


def _s1():
    return _train("s1_vteam", "MemTorch-comparison S1: 10 x +0.8 V then 10 x -0.225 V, "
                  "10 ns pulses / 10 ns gaps, from ON, -0.1 V read in each gap",
                  [(0.8, 10 * NS)] * 10 + [(-0.225, 10 * NS)] * 10,
                  v_off_pulse=0.8, v_on_pulse=-0.225, width_ns=10, n_each=10)


def _s2():
    return _train("s2_vteam", "MemTorch-comparison S2 (faster ON): 10 x +1.0 V / 10 ns, "
                  "then 10 x -0.25 V / 2 ns (10.4 % of d each), from ON, reads",
                  [(1.0, 10 * NS)] * 10 + [(-0.25, 2 * NS)] * 10,
                  v_off_pulse=1.0, v_on_pulse=-0.25, width_on_ns=2, n_each=10)


def _amplitude_sweep():
    offs = (0.2, 0.4, 0.6, 0.8, 1.0)
    ons = (-0.21, -0.22, -0.23, -0.24, -0.25)
    return _train("amplitude_sweep", "from ON: 10 ns OFF pulses of +0.2..+1.0 V, then 10 ns "
                  "ON pulses of -0.21..-0.25 V (the last reaches x = 0), each read",
                  [(a, 10 * NS) for a in offs] + [(a, 10 * NS) for a in ons],
                  amps_off_V=list(offs), amps_on_V=list(ons), width_ns=10)


def _width_sweep():
    widths = (1, 2, 5, 10, 20)
    seq = [(0.8, w * NS) for w in widths] + [(-0.23, w * NS) for w in widths]
    return _train("pulse_width_sweep", "from ON: +0.8 V pulses of 1/2/5/10/20 ns, then "
                  "-0.23 V pulses of 1/2/5/10/20 ns (reaches x = 0), each read",
                  seq, widths_ns=list(widths), v_off_pulse=0.8, v_on_pulse=-0.23)


def _read_disturb():
    # program to about s = 0.39 with 6 x +0.8 V / 10 ns, read, hold 100 ns, read;
    # then groups of 5 x 5 ns stress pulses: +0.01, -0.1, -0.19 V are inside
    # the dead zone (no motion at all); +0.03 V and -0.21 V are just outside.
    b = _Builder()
    b.wait(1 * NS)
    for _ in range(6):
        t0 = b.t
        b.pulse(0.8, 10 * NS)
        b.wait(t0 + 20 * NS - b.t)
    b.read(lead_ps=1 * NS)
    b.wait(100 * NS)
    b.read(lead_ps=1 * NS)
    levels = (0.01, -0.1, -0.19, 0.03, -0.21)
    for lv in levels:
        for _ in range(5):
            b.wait(5 * NS)
            b.pulse(lv, 5 * NS)
        b.read()
    return b.build("read_disturb_hold", "6 x +0.8 V / 10 ns from ON, read, 100 ns hold, "
                   "read, then 5 x 5 ns stress at +0.01/-0.1/-0.19 V (dead zone) and "
                   "+0.03/-0.21 V (just past v_off / v_on), each group read",
                   5 * NS, DT, stress_levels_V=list(levels), hold_s=1e-7)


def _saturate_reverse():
    # +1 V for 150 ns drives x into the x = d bound (it gets there at ~123 ns)
    # and holds it there; then -0.22 V / 10 ns pulses (3.3 % of d each) must
    # move the state at once: no integrator wind-up past the bound.
    b = _Builder()
    b.wait(1 * NS)
    b.pulse(1.0, 150 * NS)
    b.read()
    b.wait(5 * NS)
    for _ in range(10):
        t0 = b.t
        b.pulse(-0.22, 10 * NS)
        b.read()
        b.wait(t0 + 20 * NS - b.t)
    return b.build("saturate_reverse", "from ON: +1 V / 150 ns into the x = d bound "
                   "(reached at ~123 ns), read, then 10 x -0.22 V / 10 ns, each read "
                   "(window: no wind-up)", 5 * NS, DT, hold_V=1.0, v_on_pulse=-0.22, n=10)


def _dc_loop():
    # two triangle loops at 2.5e7 V/s: 0 -> +1 V -> 0 moves ~32 % of d up,
    # 0 -> -0.3 V -> 0 moves ~83 % down (reaches x = 0 in loop 1)
    b = _Builder()
    b.wait(100)
    for _ in range(2):
        for v, dur in ((1.0, 40 * NS), (0.0, 40 * NS), (-0.3, 12 * NS), (0.0, 12 * NS)):
            b.ramp_to(v, dur)
    return b.build("dc_loop", "two triangle loops 0 -> +1 -> 0 -> -0.3 -> 0 V at "
                   "2.5e7 V/s from ON", 10 * NS, DT, rate_V_per_s=2.5e7, cycles=2)


def _sine_memtorch():
    # MemTorch VTEAM.plot_hysteresis_loop defaults: 1 V, 50 MHz, 200 ns; here as a
    # PWL with a point every 100 ps (exactly the same waveform for both tools)
    b = _Builder()
    f, amp, step = 50e6, 1.0, 100
    for k in range(1, 2001):
        b._add(k * step, amp * math.sin(2 * math.pi * f * k * step * PS))
    pts = [(t, 0.0 if abs(v) < 1e-12 else round(v, 12)) for t, v in b.pts]
    b.pts = pts
    return b.build("sine_memtorch", "MemTorch hysteresis default: 1 V, 50 MHz sine for "
                   "200 ns (PWL, 100 ps points) from ON; -1 V crosses d in ~5 ps", 0, DT,
                   amp_V=amp, freq_Hz=f)


def build_stimuli():
    return [_off_train(), _on_train(), _s1(), _s2(), _amplitude_sweep(), _width_sweep(),
            _read_disturb(), _saturate_reverse(), _dc_loop(), _sine_memtorch()]


STIMULI = build_stimuli()
BY_NAME = {s.name: s for s in STIMULI}


def fingerprint(stimuli=None):
    blob = json.dumps({"version": STIMULUS_SET_VERSION,
                       "stimuli": [s.as_dict() for s in (stimuli or STIMULI)]},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def get(names=None):
    if not names:
        return list(STIMULI)
    missing = [n for n in names if n not in BY_NAME]
    if missing:
        raise SystemExit(f"unknown stimulus: {', '.join(missing)} "
                         f"(have: {', '.join(BY_NAME)})")
    return [BY_NAME[n] for n in names]


if __name__ == "__main__":
    print(f"stimulus set {STIMULUS_SET_VERSION}  sha256 {fingerprint()[:16]}")
    for s in STIMULI:
        print(f"  {s.name:20s} {len(s.points):5d} pts  {len(s.reads):3d} reads  "
              f"t_stop {s.t_stop:9.3e} s  - {s.description}")
