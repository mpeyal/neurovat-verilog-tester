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

Defaults are matched to the electrochemical graphene synapse of
M. T. Sharbati et al., "Low-Power, Electrochemically Tunable Graphene
Synapses for Neuromorphic Computing", Adv. Mater. 2018, 30, 1802353.

Sign convention (paper): a POSITIVE gate (intercalation) current POTENTIATES -
it raises the conductance / lowers R (polarity=-1 in this model's internal
mapping).  A negative (deintercalation) current depresses.  This is the
OPPOSITE of the legacy v1 Verilog-A port, which raises R on positive current;
set polarity=+1 to recover the old v1 sign.

DEFAULT parameters reproduce the paper (Sharbati et al. 2018, Fig. 3/4):
  Fig. 3a) +50 pA / 10 ms at R~4410 ohm: dR=-30 (-0.67%) -> dR'=-10 (-0.22%).
  Fig. 3c) LTP/LTD over ~100..1150 uS, >250 near-linear states (R^2=0.994).
  Fig. 4b) STDP anti-symmetric window, +side = 3-exp (tau 22ms/315ms/19s).
  Retention: 3.2% over 13 h -> tau_retention ~ 1.31e6 s.
V2Params.paper_fig3() returns this same config explicitly (used by run_fig3.py
and selftest.py).

V2Params.demo_500() is the alternate 500-ohm single-spike demo: a +50 pA pulse
takes R 500 -> 470 (dR=-30) and settles at 490 (dR'=-10) within ~3 s (c3=0
drops the slow pool; window 120..2500 uS).  Note 500 ohm is OUTSIDE the paper's
100..1150 uS window, so the two configs are mutually exclusive.
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class V2Params:
    Rmin: float = 870.0            # -> Gmax ~ 1150 uS (Fig. 3c top)
    Rmax: float = 10e3             # -> Gmin = 100 uS (Fig. 3c bottom)
    Rinit: float = 4400.0          # Fig. 3a/b operating bias

    # charge-controlled update (Fig. S4: dG linear in pulse amplitude)
    Q_ref: float = 0.5e-12         # C; = 50 pA x 10 ms, one unit pulse
    n_states: float = 650.0        # small step (~1.6 uS) -> -30 dip at 4410, fine LTP/LTD
    nu_p: float = 0.02             # near-linear (paper R^2 = 0.994)
    nu_d: float = 0.02             # symmetric at the small-signal 4400 ohm bias
    polarity: float = -1.0         # -1: positive (intercalation) I potentiates

    # volatile (short-term) component.  Pool taus tau1 = w^2/2D = 22 ms and
    # tau2 = l^2/2D = 315 ms (in-plane diffusion); tau3 = 19 s (LFP exchange).
    # c3 = 2 keeps the slow 19 s tail ON (paper Fig. 3a recovers over ~40 s).
    kappa_v: float = 0.68          # ~2/3 of each write relaxes (dR -30 -> dR' -10)
    kappa_d: float = 0.68          # depression = potentiation at small signal
    w: float = 4e-6                # device width (Fig. 4b)
    l: float = 15e-6               # device length (Fig. 4b)
    D: float = 3.6e-10             # Li diffusion coefficient in graphene
    tao3: float = 19.0             # s (Fig. 4b)
    c1: float = 1.0                # pool weights -> cn = 0.25 / 0.25 / 0.50
    c2: float = 1.0
    c3: float = 2.0
    tau_scale: float = 1.0

    # long-term retention of the nonvolatile part.
    tau_retention: float = 1.31e6   # s; relaxation of G_nv toward G_eq
    G_eq: float = 1.0 / 10e3       # equilibrium conductance (= Gmin, delithiated)

    # spike-timing-dependent plasticity (the classic ANTI-SYMMETRIC window).
    # The volatile ΔR relaxation alone is symmetric in |dt|, so it cannot make
    # one timing side potentiate and the other depress.  A_stdp supplies the
    # order-dependence: each drive polarity leaves a trace (tr_pot/tr_dep) that
    # decays with tau_stdp, and a pulse meeting the OPPOSITE surviving trace
    # LOCKS a portion into the nonvolatile state -> pre-before-post and
    # post-before-pre net OPPOSITE-sign ΔG (the +/- STDP window).  The trace
    # decays with the device's THREE relaxation taus (tau1=22ms, tau2=315ms,
    # tau3=19s - same as the volatile pools) in the weights stdp_c1/2/3, so the
    # window's positive side follows the paper Fig.4b 3-exponential
    #   dG = A_stdp*(w1 e^-dt/tau1 + w2 e^-dt/tau2 + w3 e^-dt/tau3)
    # and the negative side is its anti-symmetric mirror.  A_stdp is the peak
    # (dt->0).  Set A_stdp = 0 to disable STDP.  (This trace is INDEPENDENT of
    # the volatile pools, so it does not affect the single-pulse 30->10.)
    A_stdp: float = 4.5e-6         # S, window peak at dt->0 (paper ~4.5 uS)
    tau_stdp: float = 0.02         # s, legacy single-tau (unused by 3-exp trace)
    stdp_c1: float = 0.39          # trace weight on tau1 (22 ms)
    stdp_c2: float = 0.44          # trace weight on tau2 (315 ms)
    stdp_c3: float = 0.17          # trace weight on tau3 (19 s) -> the slow tail

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

    @property
    def stdp_weights(self):
        s = self.stdp_c1 + self.stdp_c2 + self.stdp_c3
        return (self.stdp_c1 / s, self.stdp_c2 / s, self.stdp_c3 / s)

    @classmethod
    def paper_fig3(cls, **overrides):
        """Paper-matched preset (Sharbati et al. 2018) - same as the DEFAULTS:
        4400 ohm bias, 100..1150 uS window, n_states=650, kappa_v=0.68, slow
        19 s pool ON (c3=2).  Returns the paper config explicitly regardless of
        the dataclass defaults.  Pass overrides (e.g. Rinit=4410)."""
        base = dict(Rmin=870.0, Rmax=10e3, Rinit=4400.0, Q_ref=0.5e-12,
                    n_states=650.0, nu_p=0.02, nu_d=0.02, kappa_v=0.68,
                    kappa_d=0.68, w=4e-6, l=15e-6, D=3.6e-10, tao3=19.0,
                    c1=1.0, c2=1.0, c3=2.0, tau_retention=1.31e6,
                    G_eq=1.0 / 10e3)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def demo_500(cls, **overrides):
        """Alternate 500-ohm single-spike demo: +50 pA -> R 500->470 (dR=-30)
        settling at 490 (dR'=-10) in ~3 s; -50 pA -> 530->510.  Window
        120..2500 uS (Rmin=400), big steps (n_states=17), slow pool off (c3=0),
        direction-specific nu_d/kappa_d for +/- dip symmetry.  NOTE: 500 ohm is
        outside the paper's 100..1150 uS window, so LTP/LTD (Fig. 3c) clips
        here - use the default/paper config for Fig. 3c."""
        base = dict(Rmin=400.0, Rmax=8333.0, Rinit=500.0, Q_ref=0.5e-12,
                    n_states=17.0, nu_p=0.02, nu_d=0.58, kappa_v=0.705,
                    kappa_d=0.680, w=4e-6, l=15e-6, D=3.6e-10, tao3=19.0,
                    c1=1.0, c2=1.0, c3=0.0, tau_retention=1.31e6,
                    G_eq=1.0 / 8333.0)
        base.update(overrides)
        return cls(**base)


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
        self.tr_pot = [0.0, 0.0, 0.0]  # 3-component eligibility trace (recent pot)
        self.tr_dep = [0.0, 0.0, 0.0]  # 3-component eligibility trace (recent dep)
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

        # STDP eligibility traces relax every step with the 3 device taus, so
        # the lock-in window is the paper Fig.4b 3-exponential (exact update).
        taus_s = p.taus
        for i in range(3):
            self.tr_pot[i] *= math.exp(-dt / taus_s[i])
            self.tr_dep[i] *= math.exp(-dt / taus_s[i])

        if drift_on and not self.prev_drift_on:
            self.cycle_factor = 1.0
            if p.sigma_c2c > 0:
                self.cycle_factor = max(0.0, 1.0 + self._rng.gauss(0.0, p.sigma_c2c))
            # STDP lock-in: this pulse pairs with the surviving opposite trace;
            # A_stdp * sum(trace) = A_stdp * 3-exp(dt) is the window value.
            ws = p.stdp_weights
            potentiating = (-math.copysign(1.0, Ieff) * p.polarity) > 0.0
            if potentiating:                               # post-before-pre -> LTP
                self.G_nv += p.A_stdp * sum(self.tr_dep)
                self.tr_dep = [0.0, 0.0, 0.0]              # consume harvested trace
                for i in range(3):                         # saturate at 1-pulse weight
                    self.tr_pot[i] = min(self.tr_pot[i] + ws[i], ws[i])
            else:                                          # pre-before-post -> LTD
                self.G_nv -= p.A_stdp * sum(self.tr_pot)
                self.tr_pot = [0.0, 0.0, 0.0]             # consume harvested trace
                for i in range(3):                         # saturate at 1-pulse weight
                    self.tr_dep[i] = min(self.tr_dep[i] + ws[i], ws[i])
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
                kap = p.kappa_v if sign > 0 else p.kappa_d   # direction-specific
                self.G_nv += (1.0 - kap) * dG
                for i in range(3):
                    self.pools[i] += kap * weights[i] * dG

        self.prev_drift_on = drift_on

    # ------------------------------------------------------------------

    def observables(self):
        return {
            "G_nv (S)": self.G_nv,
            "G_volatile (S)": sum(self.pools),
            "drift_on": 1.0 if self.prev_drift_on else 0.0,
        }
