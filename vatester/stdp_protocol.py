"""Waveform-driven STDP protocols. No imposed STDP window or GUI dependencies.

Drive is the sum of two signed contributions. For a two-terminal voltage
device these represent -Vpre and +Vpost; a FeFET uses an ideal differential
programming driver. RC mode is a buffered, unloaded high-pass shaper, not a
solution of the device-loaded circuit. Every timing starts from reset state.
"""
import copy
import dataclasses
import math

import numpy as np

from simulation_core import Waveform, simulate
from vatester import analysis

SHAPES = ("exponential tail", "linear tail", "RC high-pass")
MEASUREMENTS = ("Pair state change", "Interaction (pair - pre - post + idle)")
TAIL_SPAN = 6.0


def validate(width, spike=None):
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Spike width must be finite and positive.")
    if spike is None:
        return
    shape = spike.get("shape", SHAPES[0])
    if shape not in SHAPES:
        raise ValueError("Unknown spike shape.")
    for key in ("tail_ratio", "tail_tau", "r_ohm", "c_f"):
        value = spike.get(key, {"tail_ratio": 1., "tail_tau": .02,
                                "r_ohm": 1e6, "c_f": 20e-9}[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if tau(spike) <= 0 or not math.isfinite(tau(spike)):
        raise ValueError("Tail time (or R*C) must be finite and positive.")


def tau(spike):
    if spike.get("shape") == SHAPES[2]:
        return spike["r_ohm"] * spike["c_f"]
    return spike["tail_tau"]


def pieces(t0, amp, width, spike=None):
    validate(width, spike)
    if not math.isfinite(amp) or not math.isfinite(t0):
        raise ValueError("Spike amplitude and start time must be finite.")
    if spike is None:
        return [(t0, width, amp)]
    shape = spike.get("shape", SHAPES[0])
    tc = tau(spike)
    # Midpoint staircase; exact edges shared by the pair, singles and idle.
    # RC response to a rectangular input A*[H(t)-H(t-width)]:
    # y=A*exp(-t/RC) during the input, then
    # y=-A*(1-exp(-width/RC))*exp(-(t-width)/RC).
    if shape == SHAPES[2]:
        edges = np.unique(np.r_[np.linspace(0, min(width, TAIL_SPAN*tc), 97), width])
        peak = [(t0+a, b-a, amp*math.exp(-(a+b)/(2*tc)))
                for a, b in zip(edges[:-1], edges[1:]) if b > a]
        ratio = -math.expm1(-width/tc)
    else:
        peak = [(t0, width, amp)]
        ratio = spike["tail_ratio"]
    span = tc if shape == SHAPES[1] else TAIL_SPAN*tc
    n = 96
    step = span/n
    for k in range(n):
        x = (k+.5)*step
        decay = max(0., 1-x/tc) if shape == SHAPES[1] else math.exp(-x/tc)
        peak.append((t0+width+k*step, step, -amp*ratio*decay))
    return peak


def waveforms(dt, amp_pre, amp_post, width, settle, spike=None):
    validate(width, spike)
    if not math.isfinite(dt) or not math.isfinite(settle) or settle < 0:
        raise ValueError("Timing must be finite and read delay nonnegative.")
    pre_t, post_t = analysis.spike_pair_times(dt)
    pre = pieces(pre_t, amp_pre, width, spike)
    post = pieces(post_t, amp_post, width, spike)
    end = max(t+w for t, w, _ in pre+post)
    # Zeroed companions preserve all breakpoints for every control run.
    pre0 = [(t,w,0.) for t,w,_ in pre]
    post0 = [(t,w,0.) for t,w,_ in post]
    return {"pair": Waveform(pre+post), "pre": Waveform(pre+post0),
            "post": Waveform(pre0+post), "idle": Waveform(pre0+post0)}, end+settle


def prepare_model(model, include_rule=False):
    """Copy, never mutate the selected model or the user's parameter values.

    Disable the app's known phenomenological lock-in terms. This does not
    remove genuine retention/switching dynamics or promise that an arbitrary
    third-party twin has no other internal timing mechanism.
    """
    m = copy.deepcopy(model)
    if not include_rule:
        for key in ("A_stdp", "A_stdp_V"):
            if hasattr(m.p, key):
                setattr(m.p, key, 0.)
    if getattr(m, "STDP_OBS", "G") == "Vt_stdp (V)":
        m.STDP_OBS = "Vth (V)"
    m.reset()
    return m


def run_pair(model, dt, amp_pre, amp_post, width, settle, spike=None):
    waves, stop = waveforms(dt, amp_pre, amp_post, width, settle, spike)
    scale = analysis.state_profile(model)[2]
    values = {}
    for key, wave in waves.items():
        result = simulate(model, wave, stop, dt_min=min(1e-6, width/50))
        state = analysis.state_trace(result, model)
        values[key] = float(state[-1]-state[0])*scale
    values["interaction"] = values["pair"]-values["pre"]-values["post"]+values["idle"]
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError(f"{model.name}: non-finite state change")
    return values


def sweep(models, amp_pre, amp_post, width, dts, settle, spike=None,
          measurement=MEASUREMENTS[0], log=None):
    if measurement not in MEASUREMENTS:
        raise ValueError("Unknown STDP measurement.")
    selected = "pair" if measurement == MEASUREMENTS[0] else "interaction"
    curves, details = {}, {}
    for model in models:
        rows = [run_pair(model, dt, amp_pre, amp_post, width, settle, spike)
                for dt in dts]
        curves[model.name] = [row[selected] for row in rows]
        details[model.name] = {key: [row[key] for row in rows]
                               for key in ("pair", "pre", "post", "idle", "interaction")}
        if log:
            single = max(abs(row[key]) for row in rows for key in ("pre", "post"))
            label, unit, _ = analysis.state_profile(model)
            log(f"  [stdp] {model.name}: {measurement}; max single-spike |{label}| = {single:.5g} {unit}")
    return curves, details


def metadata(models, spike, measurement, include_rule):
    return {"measurement": measurement, "shape": spike or {"shape": "rectangular"},
            "include_phenomenological_rule": bool(include_rule),
            "drive_convention": "signed pre + signed post; voltage convention -Vpre + Vpost",
            "rc_assumption": "buffered unloaded high-pass, tau=R*C; device loading not solved",
            "tail_cutoff_tau": TAIL_SPAN, "tail_segments": 96,
            "reset_between_timings": True,
            "models": [{"name": m.name, "parameters": dataclasses.asdict(m.p),
                        "state_observable": getattr(m, "STDP_OBS", "G"),
                        "state_label": analysis.state_profile(m)[0],
                        "state_unit": analysis.state_profile(m)[1],
                        "state_scale": analysis.state_profile(m)[2]}
                       for m in models]}
