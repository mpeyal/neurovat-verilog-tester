"""Fixed, versioned VOLTAGE stimulus set "stanford-1" for the Stanford-PKU RRAM.

The Gate A `Stimulus` (validation/stimuli.py) describes rectangular gate
CURRENT pulses on a us grid; it is left untouched (its fingerprint is part of
the gateA-1 data).  A two-terminal RRAM is driven by a voltage across TE-BE,
ramps (DC sweeps) are part of the experiment and switching takes ns, so this
set is defined directly as piecewise-linear (PWL) voltage points:

  * the twin integrates the SAME PWL (V linear between points), and
  * the Spectre deck uses the SAME points in `vsource type=pwl`,

so both sides see an identical waveform and no "ideal vs ramped edge" charge
argument is needed.  All times are integer picoseconds (exact in the PWL
text).  Pulses are trapezoids with EDGE_PS ramps; a pulse of width w spans
[t0, t0 + w + edge] with its full value on [t0 + edge, t0 + w], so its
volt-second area is exactly amp * w.

Read plateaus are at V_READ = 0.1 V.  In this model a 0.1 V read cannot move
the gap at all (gamma * |V| / tox is far below F_min, so the rate is exactly
zero), so reads are non-disturbing in both twin and .va; the read conductance
I/V at the plateau middle is the per-pulse metric.

`instance` holds per-stimulus .va instance-parameter overrides (e.g. the
HSPICE templates start in HRS with gap_ini = 1.9 nm); both twin and deck use
them.  Change the set only by bumping STIMULUS_SET_VERSION.
"""

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STIMULUS_SET_VERSION = "stanford-1"
PS = 1e-12
EDGE_PS = 50                 # trapezoid ramp (ps); HSPICE templates use 10 ps
V_READ = 0.1                 # V
READ_PS = 4000               # read plateau length (ps)
NS = 1000                    # ps per ns


@dataclass(frozen=True)
class VoltageStimulus:
    name: str
    description: str
    points: tuple                 # ((t_ps:int, V:float), ...) strictly increasing t
    t_stop_ps: int
    dt_max: float                 # s; twin max step = Spectre maxstep = time_step
    reads: tuple = ()             # read-plateau mid times (ps), one per read
    instance: tuple = ()          # (("gap_ini", 1.9e-9), ...) .va overrides
    params: dict = field(default_factory=dict, compare=False)

    # ---- shared waveform ---------------------------------------------------
    @property
    def t_stop(self):
        return self.t_stop_ps * PS

    @property
    def t_points(self):
        return np.array([t for t, _ in self.points], dtype=float) * PS

    @property
    def v_points(self):
        return np.array([v for _, v in self.points], dtype=float)

    def v(self, t):
        """V(TE,BE) at times t (s): linear between the PWL points."""
        return np.interp(np.asarray(t, dtype=float), self.t_points, self.v_points)

    @property
    def read_times(self):
        return np.array(self.reads, dtype=float) * PS

    def instance_dict(self):
        return dict(self.instance)

    # ---- circuit-simulator side --------------------------------------------
    def pwl_points(self):
        return [(t * PS, v) for t, v in self.points]

    def spectre_pwl(self, per_line=4):
        pts = self.pwl_points()
        chunks = [" ".join(f"{t:.12g} {v:.9g}" for t, v in pts[k:k + per_line])
                  for k in range(0, len(pts), per_line)]
        return "wave=[ " + " \\\n    ".join(chunks) + " ]"

    def as_dict(self):
        return {"name": self.name, "description": self.description,
                "points": [list(p) for p in self.points],
                "t_stop_ps": self.t_stop_ps, "dt_max": self.dt_max,
                "reads": list(self.reads), "instance": [list(i) for i in self.instance],
                "params": self.params}


def parse_spectre_pwl(text):
    body = text[text.index("[") + 1:text.rindex("]")]
    nums = [float(x) for x in body.replace("\\", " ").split()]
    if len(nums) % 2:
        raise ValueError("odd number of PWL values")
    return list(zip(nums[0::2], nums[1::2]))


# --------------------------------------------------------------------------
class _Builder:
    """Append-only PWL builder in integer ps (returns to 0 V between events)."""

    def __init__(self):
        self.pts = [(0, 0.0)]
        self.reads = []

    @property
    def t(self):
        return self.pts[-1][0]

    def _add(self, t, v):
        if t <= self.pts[-1][0]:
            raise ValueError(f"PWL time {t} ps not after {self.pts[-1][0]} ps")
        self.pts.append((int(t), float(v)))

    def wait(self, dt_ps):
        """Hold the present level (must be 0 V between events) for dt_ps."""
        self._add(self.t + dt_ps, self.pts[-1][1])

    def pulse(self, amp, width_ps, edge_ps=EDGE_PS):
        """Trapezoid from the present time; area amp * width."""
        t0 = self.t
        self._add(t0 + edge_ps, amp)
        self._add(t0 + width_ps, amp)
        self._add(t0 + width_ps + edge_ps, 0.0)

    def read(self, lead_ps=3 * NS, width_ps=READ_PS, tail_ps=0):
        self.wait(lead_ps)
        t0 = self.t
        self.pulse(V_READ, width_ps)
        self.reads.append(t0 + (width_ps + EDGE_PS) // 2)
        if tail_ps:
            self.wait(tail_ps)

    def ramp_to(self, v, t_ps):
        self._add(self.t + t_ps, v)

    def build(self, name, desc, tail_ps, dt_max, instance=(), **params):
        if tail_ps:
            self.wait(tail_ps)
        return VoltageStimulus(name, desc, tuple(self.pts), self.t, dt_max,
                               tuple(self.reads), tuple(instance), params)


def _train(name, desc, seq, gap_ps=10 * NS, instance=(), dt_max=0.1e-9, **params):
    """Write pulses (amp, width_ps) each followed by a 0.1 V read inside the
    gap (read starts 3 ns after the write falls)."""
    b = _Builder()
    b.wait(1 * NS)
    for amp, w in seq:
        t_start = b.t
        b.pulse(amp, w)
        b.read()
        # next write starts gap_ps after this write's falling edge ends
        nxt = t_start + w + EDGE_PS + gap_ps
        if nxt <= b.t:
            raise ValueError(f"{name}: gap too short for the read")
        b.wait(nxt - b.t)
    return b.build(name, desc, 5 * NS, dt_max, instance, **params)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
GAP_HRS_TEMPLATE = 19e-10      # HSPICE templates: "HRS initial ... gap_ini = 19e-10"


def _pulse_reset_ref():
    # pulseSweep.sp: Vin in 0 pulse 0 -1.3 10ps 10ps 10ps 100ns 200ns, LRS
    b = _Builder()
    b.wait(10)
    b.pulse(-1.3, 100 * NS, edge_ps=10)
    b.wait(50 * NS - 10)
    b.reads.append(b.t + (40 * NS + EDGE_PS) // 2)
    b.pulse(V_READ, 40 * NS)
    return b.build("pulse_reset_ref",
                   "HSPICE template pulseSweep.sp RESET: -1.3 V / 100 ns from LRS "
                   "(gap_ini 0.2 nm), 10 ps edges, then a 0.1 V / 40 ns read",
                   15 * NS - 50, 0.1e-9, template="pulseSweep.sp RESET")


def _pulse_set_ref():
    b = _Builder()
    b.wait(10)
    b.pulse(1.8, 100 * NS, edge_ps=10)
    b.wait(50 * NS - 10)
    b.reads.append(b.t + (40 * NS + EDGE_PS) // 2)
    b.pulse(V_READ, 40 * NS)
    return b.build("pulse_set_ref",
                   "HSPICE template pulseSweep.sp SET: +1.8 V / 100 ns from HRS "
                   "(gap_ini 1.9 nm > gap_max), 10 ps edges, then a 0.1 V / 40 ns read",
                   15 * NS - 50, 0.1e-9, instance=(("gap_ini", GAP_HRS_TEMPLATE),),
                   template="pulseSweep.sp SET")


def _dc_ref(name, amp, gap_ini=None, label=""):
    # DCSweep.sp: Vin in 0 pulse 0 +-2 1us 4ms 4ms 1us 10ms ; .tran ... 8.1ms
    b = _Builder()
    b.wait(1_000_000)                       # 1 us delay
    b.ramp_to(amp, 4_000_000_000)           # 4 ms rise
    b.wait(1_000_000)                       # 1 us top
    b.ramp_to(0.0, 4_000_000_000)           # 4 ms fall
    inst = (("gap_ini", gap_ini),) if gap_ini is not None else ()
    tail = 8_100_000_000 - b.t
    return b.build(name, f"HSPICE template DCSweep.sp {label}: 0 -> {amp:+g} V -> 0 "
                   "in 4 ms + 4 ms (0.5 V/ms) after 1 us; one half-sweep per "
                   "transient, as the template does", tail, 1e-6, inst,
                   template=f"DCSweep.sp {label}")


def _dc_loop():
    # Two full bipolar loops in ONE transient at 1 V/us from LRS: RESET to
    # -1.35 V (below the 1.39 V level where RESET stops self-limiting, so the
    # gap stops at g*(V) without integrator wind-up), SET to +1.5 V (the
    # SET-side LRS clamp winds the idt state below gap_min while V > ~1.05 V;
    # the second loop shows what that does).
    b = _Builder()
    b.wait(100)
    for _ in range(2):
        for v, dur in ((-1.35, 1350), (0.0, 1350), (1.5, 1500), (0.0, 1500)):
            b.ramp_to(v, dur * NS)
    return b.build("dc_loop",
                   "two bipolar triangle loops in one transient, 1 V/us, from LRS: "
                   "0 -> -1.35 -> 0 -> +1.5 -> 0 V (x2)", 100 * NS, 1e-9,
                   rate_V_per_s=1e6, v_reset=-1.35, v_set=1.5, cycles=2)


def _reset_train():
    return _train("reset_train",
                  "20 x -1.3 V / 10 ns RESET pulses from LRS, 10 ns gaps, "
                  "0.1 V / 4 ns read in every gap", [(-1.3, 10 * NS)] * 20,
                  amp=-1.3, width_ns=10, n=20)


def _set_train():
    return _train("set_train",
                  "20 x +1.45 V / 10 ns SET pulses from HRS (gap_ini = gap_max "
                  "= 1.7 nm; the HRS SET threshold is ~1.39 V), 10 ns gaps, reads",
                  [(1.45, 10 * NS)] * 20, instance=(("gap_ini", 17e-10),),
                  amp=1.45, width_ns=10, n=20)


def _width_sweep():
    widths = (1, 2, 5, 10, 20)
    seq = [(-1.3, w * NS) for w in widths] + [(1.35, w * NS) for w in widths]
    return _train("pulse_width_sweep",
                  "from LRS: -1.3 V RESET pulses of 1/2/5/10/20 ns, then +1.35 V "
                  "SET pulses of 1/2/5/10/20 ns, each followed by a read",
                  seq, widths_ns=list(widths), v_reset=-1.3, v_set=1.35)


def _amplitude_sweep():
    amps = (1.0, 1.1, 1.2, 1.3, 1.4)
    seq = [(-a, 10 * NS) for a in amps] + [(a, 10 * NS) for a in amps]
    return _train("amplitude_sweep",
                  "from LRS: 10 ns RESET pulses of -1.0/-1.1/-1.2/-1.3/-1.4 V, "
                  "then 10 ns SET pulses of +1.0..+1.4 V, each followed by a read",
                  seq, amps_V=list(amps), width_ns=10)


def _read_disturb():
    # program an intermediate state, hold, then read-stress groups: 10 x 5 ns
    # pulses at each level followed by one 0.1 V read.  At the programmed gap
    # (~0.81 nm) the field threshold is ~1.08 V: 0.1/0.5/0.9 V must not move
    # the state at all; +1.2 V disturbs toward SET, -1.2 V toward RESET.
    b = _Builder()
    b.wait(1 * NS)
    for _ in range(5):
        t0 = b.t
        b.pulse(-1.3, 10 * NS)
        b.wait(t0 + 20 * NS - b.t)
    b.read(lead_ps=1 * NS)
    b.wait(1000 * NS)                        # 1 us hold at 0 V
    b.read(lead_ps=1 * NS)
    levels = (0.1, 0.5, 0.9, 1.2, -1.2)
    for lv in levels:
        for _ in range(10):
            b.wait(5 * NS)
            b.pulse(lv, 5 * NS)
        b.read()
    return b.build("read_disturb_hold",
                   "5 x -1.3 V / 10 ns from LRS, read, 1 us hold at 0 V, read, then "
                   "10 x 5 ns stress pulses at 0.1/0.5/0.9/+1.2/-1.2 V, each group "
                   "followed by a 0.1 V read", 5 * NS, 0.1e-9,
                   stress_levels_V=list(levels), hold_s=1e-6)


def _s1(amp):
    tag = f"{amp:.1f}".replace(".", "p")
    return _train(f"s1_pm{tag}",
                  f"MemTorch-comparison S1: 10 x -{amp:g} V then 10 x +{amp:g} V, "
                  "10 ns pulses / 10 ns gaps, from LRS, 0.1 V read in each gap",
                  [(-amp, 10 * NS)] * 10 + [(amp, 10 * NS)] * 10, amp=amp,
                  width_ns=10, n_each=10)


def build_stimuli():
    return [_pulse_reset_ref(), _pulse_set_ref(),
            _dc_ref("dc_reset_ref", -2.0, None, "RESET (LRS start)"),
            _dc_ref("dc_set_ref", 2.0, GAP_HRS_TEMPLATE, "SET (HRS start, gap_ini 1.9 nm)"),
            _dc_loop(), _reset_train(), _set_train(), _width_sweep(),
            _amplitude_sweep(), _read_disturb(), _s1(1.3), _s1(1.4)]


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
        print(f"  {s.name:20s} {len(s.points):4d} pts  {len(s.reads):3d} reads  "
              f"t_stop {s.t_stop:9.3e} s  dt_max {s.dt_max:.0e} s  - {s.description}")
