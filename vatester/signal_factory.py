"""Neuromorphic stimulus generators.

Each generator is described by a list of (key, label, default, is_int)
parameters so the GUI can auto-build its input form.  build(values) returns
(pulses, meta) where pulses is [(t_start_s, width_s, amplitude_units), ...]
- amplitudes are in *display units* (pA, nA, V, ...); the app scales them
to SI with the unit combo.  meta may carry analysis hints (e.g. n_each for
LTP/LTD splitting).

ECFET sign convention: positive gate CURRENT raises R (depresses);
potentiation = negative current.  FeFET convention: positive gate VOLTAGE
potentiates.
"""

import random
from dataclasses import dataclass, field

UNIT_SCALE = {
    "pA": 1e-12, "nA": 1e-9, "uA": 1e-6, "mA": 1e-3, "A": 1.0,
    "mV": 1e-3, "V": 1.0,
}
CURRENT_UNITS = ["pA", "nA", "uA", "mA", "A"]
VOLTAGE_UNITS = ["mV", "V"]


@dataclass
class GenParam:
    key: str
    label: str
    default: float
    is_int: bool = False


@dataclass
class Generator:
    name: str
    desc: str
    params: list
    build: object = None          # build(values: dict) -> (pulses, meta)
    meta: dict = field(default_factory=dict)


def _p(key, label, default, is_int=False):
    return GenParam(key, label, default, is_int)


# ---------------------------------------------------------------------------

def _single(v):
    return ([(v["t0_ms"] * 1e-3, v["width_ms"] * 1e-3, v["amp"])], {})


def _train(v):
    t0, w, per = v["t0_ms"] * 1e-3, v["width_ms"] * 1e-3, v["period_ms"] * 1e-3
    per = max(per, w)
    n = int(v["n"])
    return ([(t0 + k * per, w, v["amp"]) for k in range(n)], {})


def _ltp_ltd(v):
    t0, w, per = v["t0_ms"] * 1e-3, v["width_ms"] * 1e-3, v["period_ms"] * 1e-3
    per = max(per, w)
    n = int(v["n_each"])
    reps = max(1, int(v.get("reps", 1)))
    gap = v["gap_ms"] * 1e-3
    a = abs(v["amp"])
    pot = -a if v["pot_sign"] < 0 else a          # potentiating polarity
    dep = -pot
    pulses = []
    base = t0
    for _ in range(reps):
        pulses += [(base + k * per, w, pot) for k in range(n)]   # LTP
        t1 = base + n * per + gap
        pulses += [(t1 + k * per, w, dep) for k in range(n)]     # LTD
        base = t1 + n * per + gap                                # next train
    # only tag the LTP/LTD split for a single train (analysis colors branches)
    return (pulses, {"n_each": n} if reps == 1 else {})


def _ppf(v):
    t0, w = v["t0_ms"] * 1e-3, v["width_ms"] * 1e-3
    gap, per = v["gap_ms"] * 1e-3, v["pair_period_ms"] * 1e-3
    pulses = []
    for k in range(int(v["n_pairs"])):
        base = t0 + k * per
        pulses.append((base, w, v["amp"]))
        pulses.append((base + w + gap, w, v["amp"]))
    return (pulses, {})


def _poisson(v):
    rng = random.Random(int(v["seed"]))
    rate = max(v["rate_hz"], 1e-9)
    t, t_end = v["t0_ms"] * 1e-3, v["t0_ms"] * 1e-3 + v["duration_s"]
    w = v["width_ms"] * 1e-3
    pulses = []
    while True:
        t += rng.expovariate(rate)
        if t >= t_end or len(pulses) >= 5000:
            break
        pulses.append((t, w, v["amp"]))
    return (pulses, {})


def _burst(v):
    w = v["width_ms"] * 1e-3
    intra = max(v["intra_ms"] * 1e-3, w)
    inter = v["burst_period_ms"] * 1e-3
    pulses = []
    for b in range(int(v["n_bursts"])):
        base = v["t0_ms"] * 1e-3 + b * inter
        for s in range(int(v["spikes_per_burst"])):
            pulses.append((base + s * intra, w, v["amp"]))
    return (pulses, {})


def _stdp(v):
    w = v["width_ms"] * 1e-3
    dt = v["dt_ms"] * 1e-3
    per = v["period_ms"] * 1e-3
    t0 = v["t0_ms"] * 1e-3 + max(0.0, -dt)
    pulses = []
    for k in range(int(v["n_pairs"])):
        pre = t0 + k * per
        pulses.append((pre, w, v["amp_pre"]))
        pulses.append((pre + dt, w, v["amp_post"]))
    return (pulses, {})


def _stair(v):
    w = v["width_ms"] * 1e-3
    per = max(v["period_ms"] * 1e-3, w)
    pulses = []
    for k in range(int(v["n_steps"])):
        amp = v["amp_start"] + k * v["amp_step"]
        pulses.append((v["t0_ms"] * 1e-3 + k * per, w, amp))
    return (pulses, {})


GENERATORS = [
    Generator("Single spike",
              "One rectangular gate pulse.",
              [_p("t0_ms", "start (ms)", 10), _p("width_ms", "width (ms)", 10),
               _p("amp", "amplitude", 100)], _single),
    Generator("Pulse train",
              "n identical pulses. ECFET: negative amp = potentiate.",
              [_p("t0_ms", "start (ms)", 10), _p("n", "pulses", 20, True),
               _p("amp", "amplitude", -100), _p("width_ms", "width (ms)", 10),
               _p("period_ms", "period (ms)", 50)], _train),
    Generator("LTP / LTD train",
              "n potentiating then n depressing pulses (synaptic curve). "
              "pot. sign: -1 for ECFET (current), +1 for FeFET (voltage). "
              "repeats: how many full LTP+LTD trains to play back-to-back.",
              [_p("n_each", "pulses per branch", 20, True),
               _p("amp", "|amplitude|", 100),
               _p("width_ms", "width (ms)", 10),
               _p("period_ms", "period (ms)", 50),
               _p("gap_ms", "branch gap (ms)", 0),
               _p("t0_ms", "start (ms)", 10),
               _p("pot_sign", "pot. sign (-1/+1)", -1, True),
               _p("reps", "train repeats", 1, True)],
              _ltp_ltd),
    Generator("Paired pulses (PPF)",
              "Pulse pairs for paired-pulse facilitation studies.",
              [_p("n_pairs", "pairs", 5, True), _p("amp", "amplitude", -100),
               _p("width_ms", "width (ms)", 5), _p("gap_ms", "intra gap (ms)", 20),
               _p("pair_period_ms", "pair period (ms)", 500),
               _p("t0_ms", "start (ms)", 10)], _ppf),
    Generator("Poisson spike train",
              "Random spikes, exponential inter-spike intervals (rate coding).",
              [_p("rate_hz", "rate (Hz)", 50), _p("duration_s", "duration (s)", 1.0),
               _p("amp", "amplitude", -100), _p("width_ms", "width (ms)", 2),
               _p("seed", "seed", 1, True), _p("t0_ms", "start (ms)", 10)],
              _poisson),
    Generator("Burst pattern",
              "Bursts of fast spikes separated by quiet periods.",
              [_p("n_bursts", "bursts", 5, True),
               _p("spikes_per_burst", "spikes/burst", 5, True),
               _p("intra_ms", "intra-burst period (ms)", 10),
               _p("width_ms", "width (ms)", 2),
               _p("burst_period_ms", "burst period (ms)", 300),
               _p("amp", "amplitude", -100), _p("t0_ms", "start (ms)", 10)],
              _burst),
    Generator("STDP pair (pre/post)",
              "Pre spike then post spike dt later, repeated. dt<0 = post first.",
              [_p("n_pairs", "pairs", 10, True), _p("dt_ms", "dt post-pre (ms)", 20),
               _p("amp_pre", "pre amplitude", -100),
               _p("amp_post", "post amplitude", 100),
               _p("width_ms", "width (ms)", 5),
               _p("period_ms", "pair period (ms)", 200),
               _p("t0_ms", "start (ms)", 10)], _stdp),
    Generator("Staircase",
              "Increasing-amplitude steps (threshold / linearity sweep).",
              [_p("n_steps", "steps", 8, True), _p("amp_start", "first amplitude", -20),
               _p("amp_step", "amplitude step", -20),
               _p("width_ms", "width (ms)", 20),
               _p("period_ms", "period (ms)", 100),
               _p("t0_ms", "start (ms)", 10)], _stair),
    Generator("Custom pattern",
              "Free-form rows below: t_start_ms  width_ms  amplitude",
              [], None),
]


def parse_custom(text):
    """Parse 't0_ms width_ms amp' rows -> (pulses, errors)."""
    pulses, errors = [], []
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.split("#")[0].strip().replace(",", " ")
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            errors.append(f"line {ln}: need 3 values, got {len(parts)}")
            continue
        try:
            t0, w, a = (float(x) for x in parts)
            pulses.append((t0 * 1e-3, w * 1e-3, a))
        except ValueError:
            errors.append(f"line {ln}: not numeric: {line!r}")
    return pulses, errors
