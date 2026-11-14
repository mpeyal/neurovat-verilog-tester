"""Stimulus set ``sharbati-1``: the measurement protocols of Sharbati et al.,
Adv. Mater. 30, 1802353 (2018), as gate-current stimuli for the twins and the
Verilog-A models.

Built with the same ``Stimulus`` class as the Gate A set (validation/stimuli.py)
so the twin stepping loop, the PWL export (finite 1 us edges, exact charge) and
the Spectre deck code are shared.  The Gate A set itself is NOT touched.

Protocol sources (page = page of the 6-page article PDF):
  * Fig. 3a/3b (p.3 text, p.4 text, Fig. 3 caption p.3): ONE intercalation
    (deintercalation) pulse of 50 pA, 10 ms on an 8 nm flake; dR = -30 ohm
    (+30) relaxing to dR' = -10 ohm (+10).  The figure's time axis runs to
    4e4 ms with the pulse near 5e3 ms and R0 near 4410 ohm (Fig. 3a) / 4400
    ohm (Fig. 3b).  We put the pulse at 5 s and run to 205 s so both the
    figure-window value (35 s after the pulse) and the asymptote are read.
  * Fig. 3c (p.4): LTP/LTD with "a series of intercalation and
    deintercalation pulses (50 pA, 10 ms)", >250 states, R^2 = 0.994 over
    10-90 % lithiation.  The pulse interval and read timing are NOT given in
    the main text: we use 50 ms onset-to-onset (assumption) and read G at the
    end of each interval.  The ramp length is sized from the MODELS so the
    10-90 % window is crossed (v2 needs ~650 pulses per ramp, v3
    Q_full/Q_ref ~ 1607): 400 pre-depressing pulses, 1700 LTP, 1700 LTD.
    (Fig. 3c's axis shows ~250 pulses per ramp - a figure reading, reported,
    not a target.)
  * Fig. 4b (p.4 text + Fig. 4 caption): "a two-pulse programming scheme ...
    the timing difference between the two INTERCALATION pulses" - both pulses
    have the SAME polarity.  Amplitude and width are not given in the main
    text: 50 pA / 10 ms assumed.  Delta-t is onset-to-onset (the inset's
    dashed lines sit on the two rising edges); the gap (end of pulse 1 to
    start of pulse 2) is stored too.  One fresh transient per Delta-t, so the
    19 s pool and the nonvolatile drift of one pair never leak into the next.
  * Retention (p.4, Fig. S5 - SI not available): 0.1 V read, 3.2 %
    conductance decrease after 13 h.  The held state is unknown; we hold the
    model's default operating point (R0 = Rinit) after one +50 pA / 10 ms
    pulse, let the volatile pools settle for 300 s (> 15 tau3), then hold 13 h.
    The twin grid is fine (10 ms) up to the settle point and coarse (60 s)
    afterwards; the Spectre deck uses maxstep = 60 s and relies on the PWL
    breakpoints plus its own LTE control near the pulse.

Sign convention (both models, polarity = -1): positive gate current
potentiates (G up, R down).  The models' only gate input is the current
I(IGsc); there is no voltage-gate path.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_core.simulator import time_grid  # noqa: E402
from validation.stimuli import DT_MIN_S, Stimulus  # noqa: E402

STIMULUS_SET_VERSION = "sharbati-1"
PAPER = "Sharbati et al., Adv. Mater. 30, 1802353 (2018)"

AMP_A = 50e-12          # 50 pA (p.3, p.4)
WIDTH_S = 10e-3         # 10 ms (p.3, p.4)
SP_T_ON = 5.0           # single-pulse onset (Fig. 3a/b axis: pulse near 5e3 ms)
SP_T_STOP = 205.0       # 200 s after the pulse
SP_READ_FIG_S = 35.0    # "settled" read inside the paper's 40 s window
LTP_PERIOD_S = 50e-3    # assumption: not stated in the paper
LTP_N_PRE, LTP_N_EACH = 400, 1700
PP_T1 = 0.5             # first pulse of each pair
# 14 log-spaced onset-to-onset Delta-t from 15 ms to the paper's 1800 ms
PP_DTS_MS = (15, 22, 31, 45, 65, 95, 137, 197, 285, 412, 596, 861, 1245, 1800)
RET_T_ON = 10e-3
RET_T_SETTLE = 300.0    # > 15 tau3 after the programming pulse
RET_HOLD_S = 13 * 3600.0
RET_DT_FINE = 10e-3
RET_DT_COARSE = 60.0

_TIME_DECIMALS = 6


def _r(t):
    return round(float(t), _TIME_DECIMALS)


def _single(name, sign, figure):
    amp = sign * AMP_A
    return Stimulus(
        name,
        f"one {'+' if sign > 0 else '-'}50 pA / 10 ms pulse at {SP_T_ON:g} s "
        f"({figure}), then {SP_T_STOP - SP_T_ON:g} s of relaxation",
        ((SP_T_ON, WIDTH_S, amp),), t_stop=SP_T_STOP, dt_max=5e-3,
        params=dict(kind="single_pulse", figure=figure, amp_A=amp,
                    width_s=WIDTH_S, t_on_s=SP_T_ON, read_fig_s=SP_READ_FIG_S,
                    source="p.3-4 text, Fig. 3 caption"))


def _ltp_ltd():
    pulses, t = [], 0.1
    for _ in range(LTP_N_PRE):
        pulses.append((_r(t), WIDTH_S, -AMP_A))
        t += LTP_PERIOD_S
    t_ltp = t
    for _ in range(LTP_N_EACH):
        pulses.append((_r(t), WIDTH_S, +AMP_A))
        t += LTP_PERIOD_S
    t_ltd = t
    for _ in range(LTP_N_EACH):
        pulses.append((_r(t), WIDTH_S, -AMP_A))
        t += LTP_PERIOD_S
    return Stimulus(
        "ltp_ltd",
        f"{LTP_N_PRE} x -50 pA (to the low rail), {LTP_N_EACH} x +50 pA (LTP), "
        f"{LTP_N_EACH} x -50 pA (LTD); 10 ms every {LTP_PERIOD_S*1e3:.0f} ms (Fig. 3c)",
        tuple(pulses), t_stop=_r(t + 0.5), dt_max=5e-3,
        params=dict(kind="ltp_ltd", figure="Fig. 3c", amp_A=AMP_A,
                    width_s=WIDTH_S, period_s=LTP_PERIOD_S, n_pre=LTP_N_PRE,
                    n_each=LTP_N_EACH, t_ltp_s=_r(t_ltp), t_ltd_s=_r(t_ltd),
                    read="G at the end of each period (just before the next onset)",
                    assumed=["pulse period 50 ms (not in the main text)",
                             "read timing (not in the main text)",
                             "ramp length sized from the models, not the paper"],
                    source="p.4 text, Fig. 3 caption"))


def _pair(dt_ms):
    dt = dt_ms * 1e-3
    t2 = _r(PP_T1 + dt)
    return Stimulus(
        f"pp_dt{dt_ms:04d}ms",
        f"two +50 pA / 10 ms intercalation pulses, onset-to-onset {dt_ms} ms (Fig. 4b)",
        ((PP_T1, WIDTH_S, AMP_A), (t2, WIDTH_S, AMP_A)),
        t_stop=_r(t2 + WIDTH_S + 0.5), dt_max=1e-3,
        params=dict(kind="paired_pulse", figure="Fig. 4b", amp_A=AMP_A,
                    width_s=WIDTH_S, t1_s=PP_T1, t2_s=t2, dt_onset_s=dt,
                    gap_s=_r(dt - WIDTH_S),
                    assumed=["amplitude 50 pA and width 10 ms (not stated for Fig. 4b)"],
                    source="p.4 text, Fig. 4 caption"))


def _retention():
    t_stop = RET_T_SETTLE + RET_HOLD_S
    return Stimulus(
        "retention_13h",
        f"one +50 pA / 10 ms pulse, settle to {RET_T_SETTLE:g} s (dt {RET_DT_FINE*1e3:g} ms), "
        f"then 13 h hold (dt {RET_DT_COARSE:g} s)",
        ((RET_T_ON, WIDTH_S, AMP_A),), t_stop=t_stop, dt_max=RET_DT_COARSE,
        params=dict(kind="retention", figure="p.4 / Fig. S5 (SI missing)",
                    amp_A=AMP_A, width_s=WIDTH_S, t_on_s=RET_T_ON,
                    t_settle_s=RET_T_SETTLE, hold_s=RET_HOLD_S,
                    grid_segments=[[RET_T_SETTLE, RET_DT_FINE],
                                   [t_stop, RET_DT_COARSE]],
                    assumed=["held state unknown (Fig. S5 not available): "
                             "model default R0 plus one pulse"],
                    source="p.4 text"))


def build_stimuli():
    """The full, ordered sharbati-1 set (fresh objects every call)."""
    return ([_single("sp_pot", +1, "Fig. 3a"), _single("sp_dep", -1, "Fig. 3b"),
             _ltp_ltd()] + [_pair(d) for d in PP_DTS_MS] + [_retention()])


STIMULI = build_stimuli()
BY_NAME = {s.name: s for s in STIMULI}
PAIR_NAMES = [s.name for s in STIMULI if s.params["kind"] == "paired_pulse"]


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


def grid(stim):
    """Twin time grid.  Piecewise dt_max when params['grid_segments'] is set
    ([[t_end, dt_max], ...]); otherwise the Gate A rule (time_grid, dt_max)."""
    bps = stim.waveform().breakpoints
    segs = stim.params.get("grid_segments")
    if not segs:
        return time_grid(bps, stim.t_stop, dt_min=DT_MIN_S, dt_max=stim.dt_max)
    out, t0 = [np.array([0.0])], 0.0
    for t_end, dt_max in segs:
        inside = [b - t0 for b in bps if t0 < b < t_end]
        seg = time_grid(inside, t_end - t0, dt_min=DT_MIN_S, dt_max=dt_max) + t0
        out.append(seg[1:])
        t0 = t_end
    ts = np.concatenate(out)
    ts[-1] = stim.t_stop
    return ts


if __name__ == "__main__":
    print(f"stimulus set {STIMULUS_SET_VERSION}  sha256 {fingerprint()[:16]}")
    for s in STIMULI:
        print(f"  {s.name:14s} {len(s.pulses):5d} pulses  t_stop {s.t_stop:9.3f} s  "
              f"grid {len(grid(s)):7d} pts  - {s.description}")
