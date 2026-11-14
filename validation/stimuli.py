"""Fixed, versioned stimulus set for the Gate A twin-vs-Spectre validation.

Every stimulus is a list of explicit rectangular gate-current pulses
(t_start_s, width_s, amplitude_A) plus a stop time and the twin's maximum
time step.  Nothing here depends on the clock, the machine or a global RNG:
the one random train uses its own seeded `random.Random`, whose Mersenne
Twister stream is fixed across Python versions.

Sign convention (both ecfet_v2 and ecfet_v3, polarity = -1): a POSITIVE gate
current potentiates (G up, R down); a negative current depresses.

Each stimulus exports
  * `waveform()`      -> simulation_core.signals.Waveform for the twin,
  * `pwl_points()`    -> [(t, I), ...] piecewise-linear source with finite edges,
  * `spectre_pwl()`   -> the Spectre `isource type=pwl wave=[...]` text.

The PWL ramps each transition over EDGE_S starting at the ideal edge, so the
charge of every pulse equals the rectangular pulse exactly (the ramp-up loses
A*EDGE_S/2 and the ramp-down gives it back).

Change the set only by bumping STIMULUS_SET_VERSION; data in validation/data
is stamped with the version it was produced from.
"""

import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_core.signals import Waveform  # noqa: E402

STIMULUS_SET_VERSION = "gateA-1"
EDGE_S = 1e-6          # PWL rise/fall time (s); far below every pulse width/gap
DT_MIN_S = 1e-6        # twin time-grid minimum step (s)
T0_S = 10e-3           # first pulse onset (s), same as signals.Waveform builders
_TIME_DECIMALS = 6     # all times are rounded to 1 us so the PWL text is exact


@dataclass(frozen=True)
class Stimulus:
    name: str
    description: str
    pulses: tuple               # ((t_start_s, width_s, amplitude_A), ...)
    t_stop: float               # s
    dt_max: float               # s, twin maximum step (Spectre maxstep too)
    params: dict = field(default_factory=dict, compare=False)

    # ---- twin side -----------------------------------------------------
    def waveform(self):
        return Waveform(list(self.pulses))

    def drive(self, t):
        """Ideal (rectangular) gate current at times t (A)."""
        wf = self.waveform()
        return np.array([wf.current(float(x)) for x in np.atleast_1d(t)])

    # ---- circuit-simulator side ----------------------------------------
    def pwl_points(self, edge=EDGE_S):
        """[(t, I)] for a PWL current source.  Built from the compiled edge
        list, so overlapping pulses sum exactly like the twin's Waveform."""
        wf = self.waveform()
        pts = [(0.0, 0.0)]
        prev_v = 0.0
        prev_e = -math.inf
        for e, v in zip(wf.edges, wf.values):
            if e - prev_e <= edge:
                raise ValueError(f"{self.name}: edges {prev_e} and {e} closer "
                                 f"than the PWL edge time {edge}")
            if e <= 0.0:
                raise ValueError(f"{self.name}: edge at t={e} <= 0")
            pts.append((e, prev_v))
            pts.append((e + edge, v))
            prev_v, prev_e = v, e
        if pts[-1][0] >= self.t_stop:
            raise ValueError(f"{self.name}: last edge after t_stop")
        pts.append((self.t_stop, prev_v))
        return pts

    def spectre_pwl(self, edge=EDGE_S, per_line=4):
        """`wave=[ ... ]` body for a Spectre isource type=pwl."""
        pts = self.pwl_points(edge)
        chunks = [" ".join(f"{t:.12g} {i:.9g}" for t, i in pts[k:k + per_line])
                  for k in range(0, len(pts), per_line)]
        return "wave=[ " + " \\\n    ".join(chunks) + " ]"

    def as_dict(self):
        return {"name": self.name, "description": self.description,
                "pulses": [list(p) for p in self.pulses],
                "t_stop": self.t_stop, "dt_max": self.dt_max,
                "params": self.params}


def parse_spectre_pwl(text):
    """Inverse of Stimulus.spectre_pwl(): returns [(t, I), ...]."""
    body = text[text.index("[") + 1:text.rindex("]")]
    nums = [float(x) for x in body.replace("\\", " ").split()]
    if len(nums) % 2:
        raise ValueError("odd number of PWL values")
    return list(zip(nums[0::2], nums[1::2]))


def pwl_eval(points, t):
    tp, ip = zip(*points)
    return np.interp(np.asarray(t, dtype=float), tp, ip)


def _r(t):
    return round(float(t), _TIME_DECIMALS)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def _single_pulse():
    amp, width = 50e-12, 10e-3
    return Stimulus(
        "single_pulse",
        "one +50 pA / 10 ms write (Fig. 3a) then 2 s of relaxation",
        ((T0_S, width, amp),), t_stop=2.0, dt_max=1e-3,
        params=dict(amp_A=amp, width_s=width, t_start_s=T0_S))


def _ltp_ltd_train():
    amp, width, period, n = 500e-12, 10e-3, 50e-3, 20
    pulses = [(_r(T0_S + k * period), width, +amp) for k in range(n)]
    t_dep = T0_S + n * period
    pulses += [(_r(t_dep + k * period), width, -amp) for k in range(n)]
    return Stimulus(
        "ltp_ltd_train",
        f"{n} x +{amp*1e12:.0f} pA (LTP) then {n} x -{amp*1e12:.0f} pA (LTD), "
        f"{width*1e3:.0f} ms @ {period*1e3:.0f} ms",
        tuple(pulses), t_stop=_r(T0_S + 2 * n * period + 1.0), dt_max=1e-3,
        params=dict(amp_A=amp, width_s=width, period_s=period, n_each=n))


def _width_sweep():
    amp, spacing = 200e-12, 0.5
    widths = (2e-3, 5e-3, 10e-3, 20e-3, 40e-3)
    pulses = [(_r(T0_S + k * spacing), w, amp) for k, w in enumerate(widths)]
    return Stimulus(
        "pulse_width_sweep",
        "+200 pA pulses of 2/5/10/20/40 ms, one every 0.5 s",
        tuple(pulses), t_stop=_r(T0_S + len(widths) * spacing + 0.5),
        dt_max=1e-3,
        params=dict(amp_A=amp, widths_s=list(widths), spacing_s=spacing))


def _amplitude_sweep():
    width, spacing = 10e-3, 0.5
    amps = (25e-12, 50e-12, 100e-12, 200e-12, 400e-12)
    pulses = [(_r(T0_S + k * spacing), width, a) for k, a in enumerate(amps)]
    return Stimulus(
        "amplitude_sweep",
        "10 ms pulses of +25/50/100/200/400 pA, one every 0.5 s",
        tuple(pulses), t_stop=_r(T0_S + len(amps) * spacing + 0.5),
        dt_max=1e-3,
        params=dict(amps_A=list(amps), width_s=width, spacing_s=spacing))


def _gap_sweep():
    amp, width, settle = 200e-12, 10e-3, 1.5
    gaps = (5e-3, 20e-3, 50e-3, 200e-3, 1000e-3)   # end of 1st -> start of 2nd
    pulses, t = [], T0_S
    for g in gaps:
        pulses.append((_r(t), width, amp))
        t2 = t + width + g
        pulses.append((_r(t2), width, amp))
        t = t2 + width + settle
    return Stimulus(
        "gap_sweep",
        "pairs of +200 pA / 10 ms pulses with 5/20/50/200/1000 ms gaps "
        "(volatile relaxation between writes), 1.5 s between pairs",
        tuple(pulses), t_stop=_r(t), dt_max=2e-3,
        params=dict(amp_A=amp, width_s=width, gaps_s=list(gaps),
                    settle_s=settle))


def _retention():
    amp, width, period, n, t_stop = 500e-12, 10e-3, 50e-3, 10, 20.0
    pulses = [(_r(T0_S + k * period), width, amp) for k in range(n)]
    return Stimulus(
        "retention_hold",
        f"{n} x +{amp*1e12:.0f} pA / {width*1e3:.0f} ms programming, "
        f"then hold (no drive) to {t_stop:.0f} s",
        tuple(pulses), t_stop=t_stop, dt_max=5e-3,
        params=dict(amp_A=amp, width_s=width, period_s=period, n=n,
                    t_stop_s=t_stop))


def _stdp_pairs():
    # The models have ONE gate pin (IGsc): "pre" and "post" are opposite-
    # polarity pulses on that pin, as in tools/ecfet_v3_tb.scs.  dt is
    # onset-to-onset; dt > 0 = depressing pulse first, then potentiating
    # (lock-in LTP); dt < 0 = potentiating first, then depressing (LTD).
    amp, width, spacing = 50e-12, 10e-3, 2.0
    dts = (20e-3, 50e-3, 200e-3, -20e-3, -50e-3, -200e-3)
    pulses, t = [], T0_S
    for d in dts:
        first = -amp if d > 0 else +amp
        pulses.append((_r(t), width, first))
        pulses.append((_r(t + abs(d)), width, -first))
        t += spacing
    return Stimulus(
        "stdp_pair_sweep",
        "opposite-polarity 50 pA / 10 ms pulse pairs on the single gate pin, "
        "onset-to-onset dt = +20/+50/+200/-20/-50/-200 ms, 2 s apart",
        tuple(pulses), t_stop=_r(t + 0.5), dt_max=2e-3,
        params=dict(amp_A=amp, width_s=width, dts_s=list(dts),
                    spacing_s=spacing,
                    sign_rule="dt>0: -amp then +amp; dt<0: +amp then -amp"))


def _random_train():
    seed, n, amp, width = 20261113, 40, 200e-12, 5e-3
    min_gap, mean_gap, p_pot = 2e-3, 60e-3, 0.5
    rng = random.Random(seed)
    pulses, t = [], T0_S
    for _ in range(n):
        sign = +1.0 if rng.random() < p_pot else -1.0
        pulses.append((_r(t), width, sign * amp))
        t = t + width + min_gap - mean_gap * math.log(1.0 - rng.random())
    end = pulses[-1][0] + width
    return Stimulus(
        "random_spike_train",
        f"{n} seeded random-sign {amp*1e12:.0f} pA / {width*1e3:.0f} ms "
        f"spikes, exponential gaps (mean {mean_gap*1e3:.0f} ms)",
        tuple(pulses), t_stop=_r(end + 0.5), dt_max=1e-3,
        params=dict(seed=seed, n=n, amp_A=amp, width_s=width,
                    min_gap_s=min_gap, mean_gap_s=mean_gap, p_pot=p_pot,
                    rng="python random.Random (Mersenne Twister)"))


def build_stimuli():
    """The full, ordered Gate A stimulus set (fresh objects every call)."""
    return [_single_pulse(), _ltp_ltd_train(), _width_sweep(),
            _amplitude_sweep(), _gap_sweep(), _retention(), _stdp_pairs(),
            _random_train()]


STIMULI = build_stimuli()
BY_NAME = {s.name: s for s in STIMULI}


def fingerprint(stimuli=None):
    """sha256 of the canonical JSON of the set (changes iff the set does)."""
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
        print(f"  {s.name:20s} {len(s.pulses):3d} pulses  t_stop {s.t_stop:7.3f} s"
              f"  dt_max {s.dt_max*1e3:.0f} ms  - {s.description}")
