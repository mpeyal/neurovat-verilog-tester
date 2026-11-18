"""Automatic test sets: which stimuli a twin is validated on.

The user gives no stimulus.  The device profile is taken from the twin
(built-in ECFET v1/v2/v3, or the twin's device_class / input_kind), and the
stimulus amplitudes are scaled to the device by probing the TWIN once:

    profile     drive            calibration (on the twin)                    compared
    ---------   --------------   ------------------------------------------   ------------------------------
    ecfet       gate current     current that moves G by ~1 % in one pulse    G = 1/V(Rmem), whole waveform
    memristor   voltage          per polarity: the pulse amplitude that       current I(t) + G = I/V at every
                                 moves G by >= 5 % after the opposite pulse   read plateau
    fecap       voltage          amplitude that switches half of P in one     polarization V(pol) + current
                                 pulse from the opposite saturation
    fefet       gate voltage     per polarity: the amplitude that moves G     G = I_d / V_ds at every read
                                 by >= 5 % after the opposite pulse           (gate at the twin's read bias)

Each profile has a fixed list of stimuli (below), the same for every device of
that kind, so the test set is decided before any comparison runs.  The first
stimulus of each set is the short "train" used for the reference-convergence
check (layer 2).
"""

from dataclasses import dataclass, field

import numpy as np

from .stimulus import Stimulus
from .twinrun import make_model

PROFILES = ("ecfet", "memristor", "fecap", "fefet")


def profile_of(spec):
    """Device profile of a twin spec."""
    hint = (spec.get("spectre") or {}).get("profile")
    if hint:
        return hint
    if spec.get("device") == "ecfet" or spec.get("key") in ("v1", "v2", "v3"):
        return "ecfet"
    cls = str(spec.get("device_class", "")).lower()
    key = str(spec.get("key", "")).lower()
    if "fecap" in cls or "fecap" in key or "ferroelectric cap" in cls:
        return "fecap"
    if "fefet" in cls or "fefet" in key:
        return "fefet"
    if spec.get("input_kind") == "current":
        return "ecfet"
    return "memristor"


# ---------------------------------------------------------------- building
class _Seq:
    """Piecewise-linear sequence builder: pulses, read plateaus, waits, ramps."""

    def __init__(self, kind, edge):
        self.kind, self.edge = kind, edge
        self.t, self.v = [0.0], [0.0]

    @property
    def now(self):
        return self.t[-1]

    def _pt(self, t, v):
        if t <= self.t[-1]:
            raise ValueError("time must increase")
        self.t.append(t)
        self.v.append(v)

    def wait(self, d):
        self._pt(self.now + d, self.v[-1])
        return self

    def pulse(self, a, w):
        e = self.edge
        self._pt(self.now + e, a)
        self._pt(self.now + w, a)
        self._pt(self.now + e, 0.0)
        return self

    def to(self, t_rel, v):           # linear ramp to v over t_rel
        self._pt(self.now + t_rel, v)
        return self

    def build(self, name, description):
        st = Stimulus(np.array(self.t), np.array(self.v), kind=self.kind, name=name)
        st.meta["description"] = description
        return st


@dataclass
class Suite:
    profile: str
    stimuli: list
    calibration: dict = field(default_factory=dict)
    read_level: float = None          # read plateaus at this level (None: no reads)
    width: float = 0.0                # the pulse width the set is built on
    base_step: float = 0.0            # Spectre maxstep before the convergence check


# ---------------------------------------------------------------- calibration
def _response(spec, params, kind, pre, pulse, width, edge, quantity):
    """State change of `quantity` from one pulse after an opposite preset."""
    sq = _Seq(kind, edge)
    sq.wait(width)
    if pre:
        for _ in range(3):
            sq.pulse(pre, width).wait(width)
    t_mark = sq.now
    sq.pulse(pulse, width).wait(width)
    st = sq.build("cal", "")
    m = make_model(spec, params)
    m.reset()
    ts = st.grid(width / 20)
    u = st.value(ts)
    before = None
    for k in range(len(ts) - 1):
        if before is None and ts[k] >= t_mark:
            before = quantity(m)
        dt = ts[k + 1] - ts[k]
        if hasattr(m, "step_linear"):
            m.step_linear(ts[k], dt, u[k], u[k + 1])
        else:
            m.step(ts[k], dt, 0.5 * (u[k] + u[k + 1]))
    return before, quantity(m)


def _find_amp(spec, params, kind, sign, width, edge, quantity, target, lo, hi,
              opposite=True):
    """Smallest |amplitude| in [lo, hi] (log bisection) whose single pulse of
    `sign`, after three opposite pulses of the same size, changes the
    quantity by at least `target` (relative); None if even hi does not."""
    def ok(a):
        b, e = _response(spec, params, kind, -sign * a if opposite else 0.0,
                         sign * a, width, edge, quantity)
        if b is None or not np.isfinite(e):
            return False
        return abs(e - b) >= target * max(abs(b), 1e-300)
    if not ok(hi):
        return None
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _G(m):
    return float(m.G)


def _P(m):
    obs = m.observables()
    for k in ("P (norm.)", "Ptot (uC/cm2)", "P (uC/cm2)"):
        if k in obs:
            return float(obs[k])
    raise KeyError("the twin exposes no polarization observable")


# ---------------------------------------------------------------- sets
def _memristor(spec, params, width, read_level, given=None):
    given = given or {}
    edge = width / 200
    amps = {}
    if given.get("amp_pos") is not None:
        amps[+1] = abs(float(given["amp_pos"]))
    if given.get("amp_neg") is not None:
        amps[-1] = abs(float(given["amp_neg"]))
    for sign in (+1, -1):
        if sign in amps:
            continue
        w = width
        while w <= 1e-3:
            a = _find_amp(spec, params, "voltage", sign, w, edge, _G, 0.05, 0.02, 5.0)
            if a is not None:
                amps[sign] = a
                width = w
                break
            w *= 10
    if not amps:
        raise RuntimeError("the twin shows no switching up to 5 V and 1 ms pulses; "
                           "give the pulse amplitudes (--amp-pos / --amp-neg)")
    vp, vn = amps.get(+1), amps.get(-1)
    if vp is None or vn is None:          # unipolar: use the one found for both
        vp = vp or (vn if vn else None)
        vn = vn or vp
    vn = -abs(vn)
    vr = read_level
    if vr is None:
        vr = 0.1 * min(abs(vp), abs(vn))
    edge = width / 200
    gap, rw = 0.3 * width, width

    def seq():
        return _Seq("voltage", edge).wait(gap)

    def rd(s):
        return s.wait(gap).pulse(vr, rw).wait(gap)

    # start with the polarity that moves the device from its initial state
    # (a device that starts at a rail does not move under the other one)
    def moves(a):
        b, e = _response(spec, params, "voltage", 0.0, a, width, edge, _G)
        return abs(e - b) / max(abs(b), 1e-300)
    first, second = (vp, vn) if moves(vp) >= moves(vn) else (vn, vp)
    word = {True: "positive", False: "negative"}
    S = []
    s = seq()
    for a in [first] * 10 + [second] * 10:
        rd(s.pulse(a, width))
    S.append(s.build("train", f"10 {word[first > 0]} then 10 {word[second > 0]} pulses "
                     "at the switching amplitude, a read after each"))
    s = seq()
    for f in (0.5, 0.75, 1.0, 1.25, 1.5):
        rd(s.pulse(f * first, width))
    for f in (0.5, 0.75, 1.0, 1.25, 1.5):
        rd(s.pulse(f * second, width))
    S.append(s.build("amplitude_sweep", "0.5-1.5 x the switching amplitude, each "
                     "polarity, a read after each"))
    s = seq()
    for f in (0.25, 0.5, 1, 2, 4):
        rd(s.pulse(first, f * width))
    for f in (0.25, 0.5, 1, 2, 4):
        rd(s.pulse(second, f * width))
    S.append(s.build("width_sweep", "0.25-4 x the pulse width, each polarity, "
                     "a read after each"))
    s = seq()
    rd(s.pulse(first, width))
    for _ in range(20):
        rd(s)
    S.append(s.build("read_disturb", "one switching pulse, then 20 reads"))
    s = seq()
    rng = np.random.default_rng(7)
    for _ in range(20):
        sign = rng.choice((1, -1))
        f = rng.uniform(0.6, 1.4)
        rd(s.pulse(f * (vp if sign > 0 else vn), width))
    S.append(s.build("random_train", "20 pulses, random polarity and 0.6-1.4 x "
                     "amplitude (seed 7), a read after each"))
    return Suite("memristor", S, {"V_switch_pos": vp, "V_switch_neg": vn,
                                  "width_s": width}, vr, width, width / 100)


def _ecfet(spec, params, given=None):
    given = given or {}
    width = float(given.get("width") or 10e-3)
    edge = 1e-6
    iu = given.get("amp_pos")
    if iu is None:
        iu = _find_amp(spec, params, "current", +1, width, edge, _G, 0.01, 1e-13, 1e-6,
                       opposite=False)
    if iu is None:
        neg = _find_amp(spec, params, "current", -1, width, edge, _G, 0.01, 1e-13,
                        1e-6, opposite=False)
        if neg is None:
            raise RuntimeError("the twin's G does not move by 1 % with 10 ms pulses "
                               "up to 1 uA of either sign")
        iu = -neg
    S = []
    s = _Seq("current", edge).wait(0.01)
    for a in [iu] * 20 + [-iu] * 20:
        s.pulse(a, width).wait(0.04)
    S.append(s.wait(0.5).build("train", "20 potentiation then 20 depression pulses "
                               "(10 ms, 40 ms apart), then relaxation"))
    S.append(_Seq("current", edge).wait(0.01).pulse(iu, width).wait(2.0)
             .build("single_pulse", "one pulse, then 2 s of relaxation"))
    s = _Seq("current", edge).wait(0.01)
    for f in (0.25, 0.5, 1, 2, 4):
        s.pulse(f * iu, width).wait(0.2)
    for f in (0.25, 0.5, 1, 2, 4):
        s.pulse(-f * iu, width).wait(0.2)
    S.append(s.build("amplitude_sweep", "0.25-4 x the unit current, each sign"))
    s = _Seq("current", edge).wait(0.01)
    for w in (1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3):
        s.pulse(iu, w).wait(0.2)
    S.append(s.build("width_sweep", "1-50 ms pulses at the unit current"))
    s = _Seq("current", edge).wait(0.01)
    for g in (5e-3, 20e-3, 100e-3, 300e-3, 1.0):
        s.pulse(iu, width).wait(g).pulse(iu, width).wait(0.5)
    S.append(s.build("gap_sweep", "pulse pairs 5 ms to 1 s apart"))
    s = _Seq("current", edge).wait(0.01)
    for _ in range(10):
        s.pulse(iu, width).wait(0.02)
    S.append(s.wait(10.0).build("retention_hold", "10 pulses, then a 10 s hold"))
    s = _Seq("current", edge).wait(0.01)
    for d in (5e-3, 20e-3, 50e-3, 200e-3):
        s.pulse(iu, width).wait(d).pulse(-iu, width).wait(0.3)
        s.pulse(-iu, width).wait(d).pulse(iu, width).wait(0.3)
    S.append(s.build("stdp_pairs", "opposite-sign pulse pairs 5-200 ms apart, "
                     "both orders"))
    s = _Seq("current", edge).wait(0.01)
    rng = np.random.default_rng(7)
    for _ in range(40):
        s.pulse(rng.choice((1, -1)) * rng.uniform(0.5, 2) * iu, width)
        s.wait(rng.uniform(0.005, 0.15))
    S.append(s.build("random_spike_train", "40 pulses, random sign, 0.5-2 x amplitude "
                     "and 5-150 ms gaps (seed 7)"))
    return Suite("ecfet", S, {"I_unit_A": iu, "width_s": width}, None, width, 1e-3)


def _fecap(spec, params, width, given=None):
    given = given or {}
    edge = width / 50
    vc = given.get("amp_pos")
    vc = abs(float(vc)) if vc is not None else None
    w = width if vc is None else 1e3          # skip the search when given
    while w <= 1e-2:
        vc = _find_amp(spec, params, "voltage", +1, w, edge, _P, 0.5, 0.05, 10.0)
        if vc is not None:
            width = w
            break
        w *= 10
    if vc is None:
        raise RuntimeError("the twin does not switch half its polarization up to 10 V")
    edge = width / 50
    vs = 1.5 * vc
    S = []
    s = _Seq("voltage", edge).wait(width)
    for a in (-vs, vs, vs, -vs, -vs):
        s.pulse(a, width).wait(width)
    S.append(s.build("pund", "PUND: preset, then +, +, -, - pulses at 1.5 x the "
                     "half-switching amplitude"))
    for name, per, amp in (("loop_slow", 100 * width, vs), ("loop_fast", 10 * width, vs),
                           ("minor_loop", 100 * width, 0.8 * vc)):
        s = _Seq("voltage", edge).wait(width)
        for _ in range(2):
            s.to(per / 4, amp).to(per / 2, -amp).to(per / 4, 0.0)
        S.append(s.build(name, f"two triangular cycles to +/-{amp:.3g} V, "
                         f"period {per:.3g} s"))
    s = _Seq("voltage", edge).wait(width)
    for f in (0.5, 0.75, 1.0, 1.25, 1.5):
        s.pulse(-vs, width).wait(width).pulse(f * vc, width).wait(width)
    S.append(s.build("amplitude_sweep", "negative preset, then 0.5-1.5 x the "
                     "half-switching amplitude"))
    s = _Seq("voltage", edge).wait(width)
    for f in (0.1, 0.3, 1, 3, 10):
        s.pulse(-vs, width).wait(width).pulse(vc, f * width).wait(width)
    S.append(s.build("width_sweep", "negative preset, then 0.1-10 x the pulse width "
                     "at the half-switching amplitude"))
    return Suite("fecap", S, {"V_half_switch": vc, "width_s": width}, None, width,
                 width / 100)


def _fefet(spec, params, width, read_level, given=None):
    if read_level is None:
        read_level = float(getattr(make_model(spec, params).p, "Vread", 0.0))
    suite = _memristor(spec, params, width, read_level, given)
    suite.profile = "fefet"
    suite.calibration = {"V_program": suite.calibration["V_switch_pos"],
                         "V_erase": suite.calibration["V_switch_neg"],
                         "width_s": suite.calibration["width_s"]}
    return suite


def build_suite(spec, params=None, profile=None, width=None, read_level=None,
                given=None):
    """The automatic test set of a twin.  `given` holds user answers that
    replace the calibration: amp_pos, amp_neg (V, or A for the ECFET unit
    current), width (s)."""
    given = dict(given or {})
    if given.get("width") and not width:
        width = float(given["width"])
    profile = profile or profile_of(spec)
    hint = spec.get("spectre") or {}
    width = width or hint.get("width")
    read_level = read_level if read_level is not None else hint.get("read_level")
    if read_level is None:
        try:
            p = make_model(spec, params).p
            read_level = getattr(p, "V_read", None)
        except Exception:
            read_level = None
    if profile == "ecfet":
        return _ecfet(spec, params, given)
    if profile == "memristor":
        return _memristor(spec, params, width or 10e-9, read_level, given)
    if profile == "fecap":
        return _fecap(spec, params, width or 1e-6, given)
    if profile == "fefet":
        return _fefet(spec, params, width or 1e-6, read_level, given)
    raise ValueError(f"unknown profile {profile!r}; choose from {PROFILES}")
