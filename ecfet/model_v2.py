"""Upgraded practical ECFET / ECRAM model (v2).

Fixes the physical shortcomings of the v1 Verilog-A model:

* State update is charge-controlled and works in the conductance domain,
  like measured ECRAM LTP/LTD data: a pulse of charge Q_ref produces a
  nominal conductance step dG_unit, shrunk by an exponential soft-bound
  window (NeuroSim-style nonlinearity parameters nu_p / nu_d) instead of
  a hard clamp.

* Volatile relaxation is PROPORTIONAL to what each pulse wrote (fraction
  kappa_v of every written dG goes into volatile pools that relax with the
  diffusion time constants tau1 = w^2/2D, tau2 = l^2/2D, tau3), instead of
  v1's fixed 20-ohm jump.  After a pulse, R overshoots and settles to the
  retained value - the classic ECRAM "write-then-relax" transient.

* No hard 1 s diffusion cutoff: pools decay exponentially forever.

* Long-term retention drift: G_nv relaxes toward G_eq with tau_retention.

* Optional cycle-to-cycle write noise (sigma_c2c, seedable).

Sign convention matches the v1 Verilog-A: positive gate current RAISES the
resistance (lowers conductance).  Set polarity=-1 to flip.
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class V2Params:
    Rmin: float = 0.1e3            # -> Gmax = 10 mS
    Rmax: float = 10e3             # -> Gmin = 0.1 mS
    Rinit: float = 500.0

    # charge-controlled update
    Q_ref: float = 1e-12           # C; e.g. 100 pA x 10 ms = one unit pulse
    n_states: float = 100.0        # full-range / dG_unit (number of unit steps)
    nu_p: float = 2.0              # potentiation nonlinearity (G rising)
    nu_d: float = 2.0              # depression nonlinearity (G falling)
    polarity: float = +1.0         # +1: positive I raises R (v1 convention)

    # volatile (short-term) component
    kappa_v: float = 0.3           # fraction of each written dG that is volatile
    w: float = 2.828e-6
    l: float = 4e-6
    D: float = 0.4e-9
    tao3: float = 19.0
    c1: float = 10.0               # relative weights of the three pools
    c2: float = 9.9
    c3: float = 0.1
    tau_scale: float = 1.0

    # long-term retention of the nonvolatile part
    tau_retention: float = 1e4     # s; relaxation of G_nv toward G_eq
    G_eq: float = 1.0 / 10e3       # equilibrium conductance (= Gmin default)

    # spike-timing-dependent plasticity (pair-based eligibility trace)
    # Each pulse leaves a trace that decays with tau_stdp.  When a pulse of the
    # OPPOSITE polarity arrives, a portion of the surviving trace is locked into
    # the nonvolatile conductance: a depressing pulse following a recent
    # potentiating one (pre-before-post, dt>0) nets depression; a potentiating
    # pulse following a recent depressing one (post-before-pre, dt<0) nets
    # potentiation.  Magnitude ~ A_stdp * exp(-|dt|/tau_stdp) -> the classic
    # exponential STDP window.  Without this coupling the charge-controlled
    # write is timing-blind and the STDP curve is flat.
    A_stdp: float = 8e-6           # S, locked-in conductance per unit trace
    tau_stdp: float = 0.02         # s, STDP trace time constant

    # gate threshold / leak (same semantics as v1)
    I_drift_th: float = 1e-12
    leak_drift_scale: float = 0.0

    # variability
    sigma_c2c: float = 0.0         # relative std-dev of per-pulse step size
    seed: int = 0

    @property
    def Gmin(self):
        return 1.0 / self.Rmax

    @property
    def Gmax(self):
        return 1.0 / self.Rmin

    @property
    def dG_unit(self):
        return (self.Gmax - self.Gmin) / self.n_states

    @property
    def taus(self):
        return (self.w ** 2 / (2 * self.D) * self.tau_scale,
                self.l ** 2 / (2 * self.D) * self.tau_scale,
                self.tao3 * self.tau_scale)

    @property
    def pool_weights(self):
        s = self.c1 + self.c2 + self.c3
        return (self.c1 / s, self.c2 / s, self.c3 / s)


class EcfetV2:
    name = "v2 (practical ECFET)"

    def __init__(self, params=None):
        self.p = params or V2Params()
        self.reset()

    def reset(self):
        p = self.p
        self.G_nv = min(max(1.0 / p.Rinit, p.Gmin), p.Gmax)
        self.pools = [0.0, 0.0, 0.0]   # volatile conductance components
        self.prev_drift_on = False
        self.cycle_factor = 1.0
        self.tr_pot = 0.0              # eligibility trace: recent potentiation
        self.tr_dep = 0.0              # eligibility trace: recent depression
        self._rng = random.Random(p.seed)

    # ------------------------------------------------------------------

    @property
    def G(self):
        p = self.p
        return min(max(self.G_nv + sum(self.pools), p.Gmin), p.Gmax)

    @property
    def R(self):
        return 1.0 / self.G

    # ------------------------------------------------------------------

    def step(self, t, dt, I_gate):
        p = self.p

        absI = abs(I_gate)
        Ieff = p.leak_drift_scale * I_gate if absI < p.I_drift_th else I_gate
        drift_on = abs(Ieff) >= p.I_drift_th

        # STDP eligibility traces relax every step (exact exponential update)
        if p.tau_stdp > 0:
            tr_decay = math.exp(-dt / p.tau_stdp)
            self.tr_pot *= tr_decay
            self.tr_dep *= tr_decay

        if drift_on and not self.prev_drift_on:
            self.cycle_factor = 1.0
            if p.sigma_c2c > 0:
                self.cycle_factor = max(0.0, 1.0 + self._rng.gauss(0.0, p.sigma_c2c))
            # STDP lock-in: this pulse pairs with the opposite-polarity trace.
            potentiating = (-math.copysign(1.0, Ieff) * p.polarity) > 0.0
            if potentiating:
                self.G_nv += p.A_stdp * self.tr_dep   # post-before-pre -> LTP
                self.tr_pot += 1.0
            else:
                self.G_nv -= p.A_stdp * self.tr_pot   # pre-before-post -> LTD
                self.tr_dep += 1.0
            self.G_nv = min(max(self.G_nv, p.Gmin), p.Gmax)

        # 1) volatile pools always relax (exact exponential update)
        taus = p.taus
        decay = [math.exp(-dt / tau) for tau in taus]
        self.pools = [g * d for g, d in zip(self.pools, decay)]

        # 2) nonvolatile retention drift toward G_eq (exact update)
        self.G_nv = p.G_eq + (self.G_nv - p.G_eq) * math.exp(-dt / p.tau_retention)

        # 3) charge-driven write (sub-stepped explicit integration)
        if Ieff != 0.0:
            span = p.Gmax - p.Gmin
            rate = p.dG_unit * abs(Ieff) / p.Q_ref * self.cycle_factor
            n_sub = min(200, int(rate * dt / (0.005 * span)) + 1)
            h = dt / n_sub
            weights = p.pool_weights
            sign = -math.copysign(1.0, Ieff) * p.polarity  # +I, polarity +1 -> G down
            for _ in range(n_sub):
                x = (self.G - p.Gmin) / span
                x = min(max(x, 0.0), 1.0)
                if sign > 0:    # conductance rising (potentiation)
                    window = math.exp(-p.nu_p * x)
                else:           # conductance falling (depression)
                    window = math.exp(-p.nu_d * (1.0 - x))
                dG = sign * rate * window * h
                # hard stop exactly at the bounds
                head = (p.Gmax - self.G) if sign > 0 else (self.G - p.Gmin)
                if abs(dG) > head:
                    dG = math.copysign(head, dG)
                self.G_nv += (1.0 - p.kappa_v) * dG
                for i in range(3):
                    self.pools[i] += p.kappa_v * weights[i] * dG

        self.prev_drift_on = drift_on

    # ------------------------------------------------------------------

    def observables(self):
        return {
            "G_nv (S)": self.G_nv,
            "G_volatile (S)": sum(self.pools),
            "drift_on": 1.0 if self.prev_drift_on else 0.0,
        }
