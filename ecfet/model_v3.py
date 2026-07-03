"""Paper-faithful graphene Li-intercalation synapse model (v3).

CONSERVED-x + SATURATING-G(x) core.  The state is the conserved Li-intercalation
fraction x in [0,1] (x=0 delithiated = Gmin, x=1 = LiC6 stoich = Gmax), and the
retained conductance is a NORMALIZED SYMMETRIC SATURATING S-curve of x:

    state:   dx = sign * |Ieff| * dt / Q_full          (clamped to [0,1])
    map:     G_nv = Gmin + (Gmax-Gmin) * S(x)
             S(x) = (sig(k*(x-0.5)) - sig(-k/2)) / (sig(k/2) - sig(-k/2))
             sig(t) = 1/(1+exp(-t)),   so  S(0)=0, S(1)=1.

WHY THIS LAW.  The earlier v3 used a windowless MULTIPLICATIVE retained law
(dG_nv = f_ret*G_nv*(Q/Q_ref)).  That makes the per-pulse step GROW as G fills
toward Gmax -> the LTP ramp is CONCAVE-UP (accelerating) and slams the Gmax rail,
which is WRONG vs the real device.  He et al. 2025 Fig 3b/3e and Sharbati 2018
Fig 3c show potentiation that SATURATES (CONCAVE-DOWN: the step SHRINKS as G fills
toward Gmax), and depression that saturates toward Gmin.  The logistic S(x) gives
exactly that:
  * potentiation SATURATES near Gmax (concave-down)  -> matches He Fig 3b/3e;
  * depression saturates near Gmin (concave-up);
  * ~LINEAR over the 10-90% middle  -> matches Sharbati Fig 3c (R^2 ~ 0.996);
  * per-pulse step f'(x)*dx is BELL-SHAPED: max mid-range, SMALL near both rails.
    This RECONCILES the Fig 3c inset WITHOUT an accelerating law: at the 227 uS
    bias (near the LOW rail) the step is small (~0.5 uS = -11 ohm, Fig 3a), and
    mid-range it is larger (~1.2 uS).

CHARGE CONSERVATION (replaces the windowless-multiplicative hack).  Because x is
a CONSERVED, REVERSIBLE coordinate, a closed +/- cycle returns x to its start
EXACTLY -> G_nv returns -> charge-conserving BY CONSTRUCTION.  No soft-window or
"scale by G_nv only" rationale is needed any more; the hard rails are the only
bound on x.

KEPT (validated, structurally unchanged):
  * EDL volatile pools (instantaneous -30 ohm relaxing to retained -11 ohm).  The
    EDL injected per sub-step is (kappa_v/(1-kappa_v)) * |dG_nv| spread over the
    FAST-dominant 3 pools (c1=0.80@22ms, c2=0.05@315ms, c3=0.15@19s).  Because it
    is proportional to the ACTUAL retained step dG_nv, the EDL also tapers near
    the rails automatically (the step is small there) - no extra window needed.
  * 3-exp STDP eligibility-trace lock-in (cap + consume; isolated pair = Fig 4b
    3-exp, taus 22/315ms/19s, peak ~5 uS, anti-symmetric).  stdp_lock separate.
  * Self-discharge retention: x relaxes toward x_eq=0 (delithiated = Gmin) with
    tau_ret; calibrated so dR = +3.2% over 13 h from the 227 uS bias.
  * He kinetic eta(on-time), OFF by default (eta=1).

Primary device: M. T. Sharbati et al., "Low-Power, Electrochemically Tunable
Graphene Synapses for Neuromorphic Computing", Adv. Mater. 2018, 30, 1802353
(few-layer graphene, Li intercalation, LiClO4/PEO, LFP reference electrode).
Kinetic dynamic-range extension (eta, OFF by default): He et al., npj Unconv.
Comput. 2025, 2, 28 (bilayer-graphene ECRAM dynamic range).

Sign convention (paper, polarity = -1): a POSITIVE (intercalation) gate current
POTENTIATES - Li IN, x UP, conductance UP, R DOWN.  A negative (deintercalation)
current depresses.  Set polarity = +1 to recover the legacy v1 sign.

The interface mirrors EcfetV2 exactly so v3 plugs into ecfet.simulate():
    EcfetV3(params).step(t, dt, I_gate)  ->  advances state
    .G / .R properties                   ->  observed conductance / resistance
    .observables()                       ->  dict logged by the simulator
V3Params.paper() / .paper_fig3() return the paper config; .demo_500() is an
alternate single-spike preset; .kinetic() enables the He et al. dynamic-range
factor for Paper-3 demonstrations.
"""

import math
import random
from dataclasses import dataclass


def _sig(t):
    """Numerically-safe logistic 1/(1+exp(-t))."""
    if t >= 0.0:
        z = math.exp(-t)
        return 1.0 / (1.0 + z)
    z = math.exp(t)
    return z / (1.0 + z)


@dataclass
class V3Params:
    # ---- conductance window (Fig. 2d / 3c; delithiated .. LiC6 stoich) -----
    Rmin: float = 870.0            # -> Gmax ~ 1150 uS  (Fig. 3c top, LiC6)
    Rmax: float = 8333.0           # -> Gmin = 120 uS   (Fig. 3c bottom, delithiated)
    Rinit: float = 4400.0          # -> G_init = 227 uS (Fig. 3a/b operating bias)

    # ---- conserved-x charge-controlled state (Sec. 2) ----------------------
    Q_ref: float = 0.5e-12         # C; = 50 pA x 10 ms, one unit pulse (reference)
    # Q_full: charge to traverse the WHOLE intercalation window x: 0 -> 1.
    #   A full sweep at 50 pA/10 ms pulses takes Q_full/Q_ref ~ 1461 pulses, so
    #   >250 distinct nonvolatile states across the window.  dx = |I|*dt/Q_full.
    Q_full: float = 8.034e-10
    # k: logistic steepness of the saturating S-curve G_nv = Gmin+(Gmax-Gmin)*S(x).
    #   k=6 gives: concave-down potentiation (saturates near Gmax), concave-up
    #   depression (saturates near Gmin), ~linear 10-90% middle (R^2~0.996), and a
    #   bell-shaped per-pulse step (small at the rails, large mid-range).  At the
    #   227 uS bias (x~0.20, near the low rail) the retained step is ~0.52 uS
    #   (-10 ohm, Fig.3a); mid-range it is ~1.2 uS.
    k: float = 6.0
    polarity: float = -1.0         # -1: positive (intercalation) I potentiates

    # ---- volatile EDL split (Sec. 2.2) ------------------------------------
    # The EDL injected per sub-step is (kappa_v/(1-kappa_v)) * |dG_nv|, spread over
    # the 3 pools.  kappa_v ~ 0.704 makes the OBSERVED instantaneous step -30 ohm
    # @ 227 uS (Fig.3a), relaxing to the retained -10 ohm.  Because it scales by
    # the ACTUAL retained step dG_nv (itself bell-shaped), the EDL tapers at the
    # rails too -> no separate soft window is needed.
    kappa_v: float = 0.704

    # ---- volatile relaxation (3 pools; Sec. 3) ----------------------------
    # tau1 = w^2/2D = 22.2 ms (FAST in-plane / EDL gating), tau2 = l^2/2D =
    # 312.5 ms, tau3 = 19 s (LFP <-> graphene slow tail).  FAST-dominant: c1
    # carries the bulk on tau1=22 ms (present within the 10 ms read pulse, drops
    # fast afterward) with a small slow tail on tau3 (the 2-stage -30 -> -10
    # recovery of Fig.3a).  These are the EDL split ONLY; the STDP trace weights
    # (stdp_c1/2/3) are independent.
    w: float = 4e-6                # device width (Fig. 4b)
    l: float = 15e-6              # device length (Fig. 4b)
    D: float = 3.6e-10            # Li diffusion coefficient in graphene
    tao3: float = 19.0           # s (Fig. 4b slow tail)
    c1: float = 0.80             # EDL pool weight on tau1 (22 ms, fast drop)
    c2: float = 0.05             # EDL pool weight on tau2 (315 ms)
    c3: float = 0.15             # EDL pool weight on tau3 (19 s slow tail)
    tau_scale: float = 1.0

    # ---- long-term retention / self-discharge (Sec. 5; Fig. S5) -----------
    # x relaxes toward x_eq = 0 (delithiated = Gmin).  +3.2% dR over 13 h from
    # the 227 uS bias -> tau_ret ~ 1.05e6 s.
    tau_ret: float = 1.05e6
    x_eq: float = 0.0              # delithiated equilibrium (x=0 -> Gmin)

    # ---- STDP 3-exp eligibility-trace lock-in (Sec. 4; Fig. 4b) -----------
    # dG(dt) = A_stdp*(scn1 e^-dt/tau1 + scn2 e^-dt/tau2 + scn3 e^-dt/tau3).
    # peak ~A_stdp at dt->0; ~1 uS by 1800 ms (19 s tail).  A_stdp=0 disables.
    A_stdp: float = 6e-6           # S, window peak (V2 fix: ~5 uS at dt->0)
    stdp_c1: float = 0.39          # trace weight on tau1 (22 ms)
    stdp_c2: float = 0.44          # trace weight on tau2 (315 ms)
    stdp_c3: float = 0.17          # trace weight on tau3 (19 s) -> slow tail

    # ---- kinetic dynamic-range factor (Paper 3, He et al.); OFF by default -
    kinetic_on: bool = False       # eta = 1 unless enabled
    tau_kin: float = 0.02          # s, kinetic on-time constant (~10-30 ms)
    eta_inf: float = 0.3           # floor efficiency for vanishingly short pulses
    t_on: float = 0.01             # s, pulse high-time used by eta() (waveform info)

    # ---- gate threshold / leak (same semantics as v1/v2) ------------------
    I_drift_th: float = 1e-12
    leak_drift_scale: float = 0.0

    # ---- variability ------------------------------------------------------
    sigma_c2c: float = 0.0         # relative std-dev of per-pulse step size
    seed: int = 0

    # -------------------------------------------------------------------
    @property
    def Gmin(self):
        return 1.0 / self.Rmax

    @property
    def Gmax(self):
        return 1.0 / self.Rmin

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

    # ---- normalized saturating S-curve S(x), S(0)=0, S(1)=1 ---------------
    def S(self, x):
        a = _sig(-0.5 * self.k)
        b = _sig(0.5 * self.k)
        return (_sig(self.k * (x - 0.5)) - a) / (b - a)

    def S_inv(self, frac):
        """Inverse of S: returns x with S(x)=frac (frac in [0,1])."""
        frac = min(max(frac, 1e-12), 1.0 - 1e-12)
        a = _sig(-0.5 * self.k)
        b = _sig(0.5 * self.k)
        y = a + frac * (b - a)
        return 0.5 + math.log(y / (1.0 - y)) / self.k

    def G_of_x(self, x):
        x = min(max(x, 0.0), 1.0)
        return self.Gmin + (self.Gmax - self.Gmin) * self.S(x)

    def x_of_G(self, g):
        """Conserved fraction x for a retained conductance g (clamped to window)."""
        g = min(max(g, self.Gmin), self.Gmax)
        return min(max(self.S_inv((g - self.Gmin) / (self.Gmax - self.Gmin)),
                       0.0), 1.0)

    @classmethod
    def paper(cls, **overrides):
        """Paper-matched preset (Sharbati et al. 2018) - the conserved-x +
        saturating-G(x) law.  120..1150 uS window, 4400 ohm bias, logistic
        steepness k=6, Q_full=8.034e-10 C (~1414 pulses across the window),
        kappa_v=0.704 (observed -30 ohm relaxing to retained -10 ohm), FAST-
        dominant EDL pools (0.80/0.05/0.15 on 22 ms/315 ms/19 s), slow 19 s tail
        ON, tau_ret=1.05e6 s (+3.2%/13 h).  Pass overrides (e.g. Rinit=4410)."""
        base = dict(Rmin=870.0, Rmax=8333.0, Rinit=4400.0, Q_ref=0.5e-12,
                    Q_full=8.034e-10, k=6.0, kappa_v=0.704,
                    w=4e-6, l=15e-6, D=3.6e-10, tao3=19.0,
                    c1=0.80, c2=0.05, c3=0.15, tau_ret=1.05e6, x_eq=0.0,
                    A_stdp=6e-6, kinetic_on=False)
        base.update(overrides)
        return cls(**base)

    # alias matching v2's classmethod name so harnesses can call either
    @classmethod
    def paper_fig3(cls, **overrides):
        return cls.paper(**overrides)

    @classmethod
    def demo_500(cls, **overrides):
        """Alternate 500-ohm single-spike preset (outside the paper window).
        +50 pA -> R 500->470 (dR~-30) settling at 490 (dR'~-10).  Window
        120..2500 uS (Rmin=400), slow pool off (c3=0).  Q_full is scaled so the
        500-ohm bias still gives a -10 ohm retained step; the saturating law is
        the same."""
        base = dict(Rmin=400.0, Rmax=8333.0, Rinit=500.0, Q_ref=0.5e-12,
                    Q_full=2.2e-11, k=6.0, kappa_v=2.0 / 3.0,
                    w=4e-6, l=15e-6, D=3.6e-10, tao3=19.0,
                    c1=1.0, c2=1.0, c3=0.0, tau_ret=1.05e6, x_eq=0.0,
                    A_stdp=5e-6, kinetic_on=False)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def kinetic(cls, **overrides):
        """Paper-3 (He et al.) preset: kinetic efficiency eta(on-time) ENABLED to
        reproduce the dynamic-range trends vs amplitude / duty / frequency.  Built
        on the paper() config; only kinetic_on / tau_kin / eta_inf differ."""
        base = dict(kinetic_on=True, tau_kin=0.02, eta_inf=0.3)
        base.update(overrides)
        return cls.paper(**base)


class EcfetV3:
    name = "v3 (paper-faithful conserved-x saturating)"

    def __init__(self, params=None):
        self.p = params or V3Params()
        self.reset()

    def reset(self):
        p = self.p
        g0 = min(max(1.0 / p.Rinit, p.Gmin), p.Gmax)
        self.x = p.x_of_G(g0)              # conserved Li-intercalation fraction
        self.G_nv = p.G_of_x(self.x)       # nonvolatile (retained) conductance
        self.stdp_lock = 0.0               # STDP-locked contribution (separate)
        self.pools = [0.0, 0.0, 0.0]       # volatile EDL pools (S)
        self.prev_drift_on = False
        self.cycle_factor = 1.0
        self.tr_pot = [0.0, 0.0, 0.0]   # 3-component eligibility trace (recent pot)
        self.tr_dep = [0.0, 0.0, 0.0]   # 3-component eligibility trace (recent dep)
        self.t_evt = 0.0                # time of the previous pulse onset
        self._rng = random.Random(p.seed)

    # ------------------------------------------------------------------
    @property
    def G(self):
        p = self.p
        return min(max(self.G_nv + self.stdp_lock + sum(self.pools), p.Gmin), p.Gmax)

    @property
    def R(self):
        return 1.0 / self.G

    @property
    def x_Li(self):
        """Conserved Li-intercalation fraction x in [0,1] (the retained state)."""
        return min(max(self.x, 0.0), 1.0)

    # ------------------------------------------------------------------
    def _eta(self):
        """Kinetic efficiency eta(on-time) in (0,1].  =1 unless kinetic_on (Sec. 6).
        eta = eta_inf + (1-eta_inf)*(1 - exp(-t_on/tau_kin))."""
        p = self.p
        if not p.kinetic_on:
            return 1.0
        return p.eta_inf + (1.0 - p.eta_inf) * (1.0 - math.exp(-p.t_on / p.tau_kin))

    # ------------------------------------------------------------------
    def step(self, t, dt, I_gate):
        p = self.p

        absI = abs(I_gate)
        Ieff = p.leak_drift_scale * I_gate if absI < p.I_drift_th else I_gate
        drift_on = abs(Ieff) >= p.I_drift_th
        taus = p.taus

        # (A) STDP eligibility traces relax every step (exact, 3 device taus) ----
        for i in range(3):
            decay = math.exp(-dt / taus[i])
            self.tr_pot[i] *= decay
            self.tr_dep[i] *= decay

        # (B) pulse ONSET -> c2c factor + STDP lock-in (edge-triggered) ----------
        # Each eligibility-trace component is SATURATED at its single-pulse weight
        # ws[i] (sum over 3 comps <= 1), and the locked-in pairing CONSUMES (zeros)
        # the opposite trace it harvested, so a genuine pre/post PAIR locks in ONCE
        # (Fig.4b) while a monotonic Fig.3 ramp leaves the opposite trace ~0 -> the
        # ramp is carried by G_nv (the conserved-x state), not by stdp_lock.
        if drift_on and not self.prev_drift_on:
            self.cycle_factor = 1.0
            if p.sigma_c2c > 0:
                self.cycle_factor = max(0.0, 1.0 + self._rng.gauss(0.0, p.sigma_c2c))
            ws = p.stdp_weights
            potentiating = (-math.copysign(1.0, Ieff) * p.polarity) > 0.0
            if potentiating:                          # post-before-pre -> LTP
                self.stdp_lock += p.A_stdp * sum(self.tr_dep)
                self.tr_dep = [0.0, 0.0, 0.0]         # consume harvested trace
                for i in range(3):                    # SATURATE at single-pulse weight
                    self.tr_pot[i] = min(self.tr_pot[i] + ws[i], ws[i])
            else:                                     # pre-before-post -> LTD
                self.stdp_lock -= p.A_stdp * sum(self.tr_pot)
                self.tr_pot = [0.0, 0.0, 0.0]         # consume harvested trace
                for i in range(3):                    # SATURATE at single-pulse weight
                    self.tr_dep[i] = min(self.tr_dep[i] + ws[i], ws[i])
            self._clamp_nv()
            self.t_evt = t

        # (C) volatile pools relax (exact exponential update) -------------------
        decay = [math.exp(-dt / tau) for tau in taus]
        self.pools = [g * d for g, d in zip(self.pools, decay)]

        # (D) nonvolatile retention / self-discharge: x relaxes toward x_eq -----
        #     (conserved-state relaxation; G_nv is recomputed from x below).
        self.x = p.x_eq + (self.x - p.x_eq) * math.exp(-dt / p.tau_ret)
        self.G_nv = p.G_of_x(self.x)

        # (E) conserved-x charge-driven write (sub-stepped) ---------------------
        # The retained step is the EXACT increment of the saturating map,
        #   dG_nv = G_of_x(x+dx) - G_of_x(x),   dx = sign*|I|*h/Q_full,
        # which is BELL-SHAPED in x (small at the rails, large mid-range) -> the
        # LTP ramp is concave-down (saturates toward Gmax), LTD concave-up
        # (saturates toward Gmin), and a closed +/- cycle conserves x exactly.
        # The EDL injected per sub-step is (kappa_v/(1-kappa_v))*|dG_nv| spread
        # over the 3 pools -> the observed instantaneous step is -30 ohm relaxing
        # to the retained -11 ohm, and the EDL tapers at the rails because dG_nv
        # does.
        if Ieff != 0.0:
            eta = self._eta()
            sign = -math.copysign(1.0, Ieff) * p.polarity   # +1 pot, -1 dep
            cf = self.cycle_factor
            # size sub-steps so each moves x by <= 0.002 (fine resolution),
            # INDEPENDENT of how the caller chunks dt.
            dxdt = abs(Ieff) / p.Q_full * eta * cf          # |dx|/dt
            move = dxdt * dt                                # |dx| over this dt
            n_sub = min(500, int(move / 0.002) + 1)
            h = dt / n_sub
            dx_sub = sign * abs(Ieff) * h / p.Q_full * eta * cf
            cw = p.pool_weights
            edl_factor = p.kappa_v / max(1.0 - p.kappa_v, 1e-9)
            for _ in range(n_sub):
                x_new = min(max(self.x + dx_sub, 0.0), 1.0)
                G_new = p.G_of_x(x_new)
                dG_nv = G_new - self.G_nv               # retained step (signed)
                self.x = x_new
                self.G_nv = G_new
                # volatile EDL: proportional to |retained step|, signed by dG_nv
                dG_edl = math.copysign(edl_factor * abs(dG_nv), dG_nv)
                for i in range(3):
                    self.pools[i] += cw[i] * dG_edl

        self.prev_drift_on = drift_on

    # ------------------------------------------------------------------
    def _clamp_nv(self):
        p = self.p
        total = self.G_nv + self.stdp_lock
        if total > p.Gmax:
            self.stdp_lock = p.Gmax - self.G_nv
        elif total < p.Gmin:
            self.stdp_lock = p.Gmin - self.G_nv

    # ------------------------------------------------------------------
    def observables(self):
        return {
            "G_nv (S)": self.G_nv,                       # retained (Li-doping) conductance
            "G_retained (S)": self.G_nv + self.stdp_lock,  # full nonvolatile state
            "G_volatile (S)": sum(self.pools),           # decaying EDL transient
            "stdp_lock (S)": self.stdp_lock,             # STDP-only contribution
            "x_Li": self.x_Li,                           # conserved lithiation fraction
            "drift_on": 1.0 if self.prev_drift_on else 0.0,
        }
