"""Python twin of verilog/vteam.va: the VTEAM memristor (Kvatinsky et al.,
IEEE TCAS-II 62(8), 2015) in the variant MemTorch 1.1.6 implements.

Equations (paper numbering; restated, no MemTorch code copied - it is GPL-3):

    (6a) dx/dt = k_off (v/v_off - 1)^alpha_off f_off(x),  0 < v_off < v
    (6b) dx/dt = 0,                                      v_on < v < v_off
    (6c) dx/dt = k_on  (v/v_on  - 1)^alpha_on  f_on(x),   v < v_on < 0
    (7)  i = v / R(x),  R(x) = r_off x/d + r_on (1 - x/d)   (MemTorch form)

MemTorch's VTEAM (memtorch/bh/memristor/VTEAM.py) versus the paper:
  * window: the paper's "ideal rectangular window" (dx/dt = 0 outside the
    state bounds, Sec. III; also the window the paper's own fits use, Sec. IV).
    MemTorch has no f_on/f_off; it clips the state x to [0, d] after every
    forward-Euler step, so the state never winds up past a bound.
  * I-V: eq. (7) with x_on = 0 and (x_off - x_on) replaced by d; the
    exponential I-V eq. (8) is not used (MemTorch computes lambda but never
    uses it).
  * x_on is used only as the initial state and x_off not at all; the bounds
    are [0, d].  Identical to the paper for the defaults x_on = 0, x_off = d.
  * defaults = paper Table II, column "TEAM [9]" (k in m/s, x in m).
  * threshold tests are inclusive (v >= v_off); the rate is 0 there anyway.

Continuous-time model (= the .va).  State s = x/d.  The rectangular window acts
on the rate with a linear edge of width w_eps (fraction of d):
    ds/dt = win(s) * rate(v) / d,
    win = clip((1 - s)/w_eps, 0, 1) for rate > 0, clip(s/w_eps, 0, 1) for rate < 0.
w_eps = 0 is MemTorch's hard clip.  R uses s clipped to [0, 1].

Integration: EXACT, no step-size error.  The rate depends on v(t) only and the
window on s only, so within any interval where the regime (above v_off, dead
zone, below v_on) does not change, s evolves along ds/dQ = win(s) with
Q = (1/d) * integral of rate(v(t)) dt.  For v linear in t that integral is
closed form (u = v/v_th - 1 is linear: integral u^a dt = du^(a+1)/((a+1) u')),
and ds/dQ = win(s) is solved in closed form (linear motion, then an
exponential approach 1 - s ~ w_eps exp(-Q/w_eps) inside the edge).  Each step
is split at the times where the linear v(t) crosses v_off and v_on, so the
kinks of the rate are located exactly.  The Stanford twin needs an adaptive
DOPRI5 integrator because its rate depends on the state; here the
separable form makes it unnecessary.  `integrator=1` instead takes ONE
forward-Euler step per call followed by a hard clip, in MemTorch's order
(update x, clip to [0, d], then the current at the new x).

`.G` is the read conductance 1/R(x) (the I-V is linear, so it holds at any V).
"""

import math
from dataclasses import dataclass


@dataclass
class VTEAMParams:
    # ---- .va / MemTorch parameters (same names and defaults) -------------
    r_off: float = 1000.0            # ohm, at x = d
    r_on: float = 50.0               # ohm, at x = 0
    d: float = 3e-9                  # m
    k_on: float = -10.0              # m/s
    k_off: float = 5e-4              # m/s
    alpha_on: float = 3.0
    alpha_off: float = 1.0
    v_on: float = -0.2               # V
    v_off: float = 0.02              # V
    x_on: float = 0.0                # m (initial state in MemTorch)
    x_off: float = 3e-9              # m (unused by MemTorch)
    x_init: float = 0.0              # m, initial state (MemTorch: x_on)
    w_eps: float = 1e-4              # window edge width (fraction of d); 0 = hard clip
    # ---- twin-only options ---------------------------------------------
    V_read: float = -0.1             # V, a non-disturbing read (v_on < V_read < v_off)
    integrator: int = 0              # 0 exact closed form, 1 MemTorch Euler + clip


def _int_pow(ua, ub, a, dt):
    """Integral over dt of u(t)^a with u linear from ua to ub (both >= 0)."""
    du = ub - ua
    um = 0.5 * (ua + ub)
    if um <= 0.0:
        return 0.0
    if abs(du) <= 1e-4 * um:
        r = du / um
        return dt * um ** a * (1.0 + a * (a - 1.0) / 24.0 * r * r)
    return dt * (ub ** (a + 1.0) - ua ** (a + 1.0)) / ((a + 1.0) * du)


def _advance_up(s, q, eps):
    """Solve ds/dQ = clip((1 - s)/eps, 0, 1) from s over Q = q >= 0."""
    if s >= 1.0:
        return s
    if eps <= 0.0:
        return min(s + q, 1.0)
    a = 1.0 - eps
    if s < a:
        if s + q <= a:
            return s + q
        q -= a - s
        s = a
    return 1.0 - (1.0 - s) * math.exp(-q / eps)


class VTEAM:
    name = "VTEAM memristor (MemTorch variant)"
    input_kind = "voltage"
    TERMINALS = 2

    def __init__(self, params=None):
        self.p = params or VTEAMParams()
        self.reset()

    def reset(self):
        self.x = float(self.p.x_init)            # m, the integrator state
        self.V = 0.0
        self.n_pieces = 0                        # closed-form pieces (cost)

    # ---- algebraic parts ------------------------------------------------
    @property
    def s(self):
        return self.x / self.p.d

    @property
    def x_clamped(self):
        return min(max(self.x, 0.0), self.p.d)

    def resistance(self, x=None):
        p = self.p
        sc = min(max((self.x if x is None else x) / p.d, 0.0), 1.0)
        return p.r_off * sc + p.r_on * (1.0 - sc)

    def current(self, V, x=None):
        return V / self.resistance(x)

    def rate(self, V):
        """dx/dt of eq. (6) without the window (m/s)."""
        p = self.p
        if V > p.v_off:
            return p.k_off * (V / p.v_off - 1.0) ** p.alpha_off
        if V < p.v_on:
            return p.k_on * (V / p.v_on - 1.0) ** p.alpha_on
        return 0.0

    # ---- stepping -------------------------------------------------------
    def step(self, t, dt, V):
        """Advance by dt with a constant voltage V (twin contract)."""
        self.step_linear(t, dt, V, V)

    def step_linear(self, t, dt, v0, v1):
        """Advance by dt with V(t) linear from v0 (at t) to v1 (at t + dt)."""
        p = self.p
        v0, v1 = float(v0), float(v1)
        if dt <= 0.0:
            self.V = v1
            return
        if p.integrator == 1:
            # MemTorch order: x += dxdt(v) dt, clip to [0, d]
            self.x += self.rate(0.5 * (v0 + v1)) * dt
            if self.x >= p.d or self.x <= 0.0:
                self.x = p.d if self.x >= p.d else 0.0
        else:
            self._exact(dt, v0, v1)
        self.V = v1

    def _exact(self, dt, v0, v1):
        p = self.p
        cuts = [0.0, dt]
        if v1 != v0:
            for vt in (p.v_off, p.v_on):
                tc = (vt - v0) / (v1 - v0) * dt
                if 0.0 < tc < dt:
                    cuts.append(tc)
        cuts.sort()
        s = self.x / p.d
        slope = (v1 - v0) / dt
        for ta, tb in zip(cuts[:-1], cuts[1:]):
            h = tb - ta
            if h <= 0.0:
                continue
            va, vb = v0 + slope * ta, v0 + slope * tb
            vm = 0.5 * (va + vb)
            if vm > p.v_off:
                q = p.k_off * _int_pow(max(va / p.v_off - 1.0, 0.0),
                                       max(vb / p.v_off - 1.0, 0.0), p.alpha_off, h) / p.d
                s = _advance_up(s, q, p.w_eps)
            elif vm < p.v_on:
                q = -p.k_on * _int_pow(max(va / p.v_on - 1.0, 0.0),
                                       max(vb / p.v_on - 1.0, 0.0), p.alpha_on, h) / p.d
                s = 1.0 - _advance_up(1.0 - s, q, p.w_eps)   # mirror of the up move
            else:
                continue
            self.n_pieces += 1
        self.x = s * p.d

    # ---- read-out -------------------------------------------------------
    @property
    def R(self):
        return self.resistance()

    @property
    def G(self):
        return 1.0 / self.resistance()

    @G.setter
    def G(self, g):
        """Set the state whose conductance is g (clamped to [1/r_off, 1/r_on])."""
        p = self.p
        r = min(max(1.0 / g, p.r_on), p.r_off)
        self.x = p.d * (r - p.r_on) / (p.r_off - p.r_on)

    def observables(self):
        V = self.V
        return {"x (nm)": self.x_clamped * 1e9, "s_state": self.s,
                "I (A)": self.current(V), "V (V)": V, "R (ohm)": self.R,
                "G_read (S)": self.G}


TWIN_SPEC = {
    "key": "vteam",
    "label": "VTEAM memristor (Kvatinsky 2015, MemTorch variant)",
    "device_class": "Memristor",
    "input_kind": "voltage",
    "va_keywords": ("vteam",),
    "model_class": VTEAM,
    "params_class": VTEAMParams,
    "result_plots": [("G", "G_read", "uS", 1e6), ("x (nm)", "x", "nm", 1.0)],
    "analysis_metrics": [("G", "G_read", "uS", 1e6), ("x (nm)", "x", "nm", 1.0)],
}
