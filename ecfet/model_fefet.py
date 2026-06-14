"""FeFET synaptic device model (Preisach-lite / Merz-law switching).

Voltage-driven ferroelectric FET synapse: gate VOLTAGE pulses partially
switch the normalized remnant polarization P in [-1, +1], P shifts the
channel threshold voltage, and the channel conductance read out at a fixed
read bias follows a logistic transfer curve.

* Switching rate follows a Merz-type field-activation law: above the
  coercive voltage Vc the polarization relaxes toward sign(V) with rate
  (1/tau0) * exp((|V| - Vc)/V0).  Below Vc the rate is exponentially
  suppressed -> quasi-nonvolatile.
* Multi-domain (partial) switching emerges naturally: a finite pulse only
  moves P part of the way, giving accumulative potentiation/depression.
* Retention: P decays toward 0 with tau_ret (depolarization field).

Sign convention: POSITIVE gate voltage pulses raise P and therefore raise
conductance (potentiate) - opposite domain to the ECFET current convention.
The waveform amplitude is interpreted as gate voltage in volts.
"""

import math
from dataclasses import dataclass


@dataclass
class FeFETParams:
    Rmin: float = 0.1e3        # on-resistance bound   -> Gmax
    Rmax: float = 10e3         # off-resistance bound  -> Gmin
    P_init: float = 0.0        # initial polarization in [-1, 1]

    # ferroelectric switching (Merz law)
    Vc: float = 0.6            # coercive voltage (V)
    V0: float = 0.15           # activation steepness (V)
    tau0: float = 5e-3         # switching time constant at |V| = Vc (s)

    # retention / depolarization
    tau_ret: float = 1e4       # s; P relaxes toward 0

    # electrostatics -> readout
    dVth_max: float = 0.8      # total Vth shift from P=-1 to P=+1 (V)
    Vth0: float = 0.0          # threshold at P = 0 (V)
    Vread: float = 0.0         # gate read bias (V)
    SS: float = 0.25           # logistic slope of the transfer curve (V)

    V_th_min: float = 1e-3     # |V| below this is treated as 0 (read noise)

    Pr: float = 25.0           # remnant polarization (uC/cm^2) for the P-V loop

    # spike-timing-dependent plasticity: a per-spike eligibility trace makes the
    # retained Vt shift ORDER-dependent (the bare Merz switch saturates and is
    # timing-blind).  A pre-then-post pair nets a Vt drop (potentiation), the
    # reverse a Vt rise - the classic anti-symmetric window, magnitude
    # ~ A_stdp_V * exp(-|dt|/tau_stdp).  Set A_stdp_V = 0 to disable.
    A_stdp_V: float = 0.04     # V; Vt lock-in per unit trace (~40 mV window)
    tau_stdp: float = 0.02     # s; STDP trace time constant (window width)

    @property
    def Gmin(self):
        return 1.0 / self.Rmax

    @property
    def Gmax(self):
        return 1.0 / self.Rmin


class FeFET:
    name = "FeFET (Merz/Preisach-lite)"
    input_kind = "voltage"     # waveform amplitude = gate voltage (V)

    # ---- device profile: a FeFET's state is its threshold-voltage shift, and
    # its hallmark plot is the polarization hysteresis loop (NOT conductance) ---
    # STDP tracks the TIMING-induced Vt shift (the eligibility lock-in), not the
    # total Vt - the bare polarization switch is a saturating background write
    # that would swamp the timing window.  This is the spike-timing synaptic
    # weight change; it is amplitude-independent (per-spike trace).
    STDP_OBS = "Vt_stdp (V)"
    STDP_LABEL = "dVt"
    STDP_UNIT = "mV"
    STDP_SCALE = 1e3           # V -> mV
    POLAR_OBS = "P (norm.)"    # polarization observable for the P-V loop
    ANALYSES = ("stdp", "polarization")
    # transient Results-tab plots: a FeFET's state is Vth + polarization, NOT
    # resistance/conductance.  (observable, axis label, unit, SI->display scale)
    RESULT_PLOTS = (("Vth (V)", "Vth", "mV", 1e3),
                    ("P (uC/cm2)", "Polarization", "uC/cm^2", 1.0))
    # Analysis-tab per-pulse metrics (LTP/LTD vs pulse #) - track Vt / P, not G/R
    ANALYSIS_METRICS = (("Vth (V)", "Vth", "mV", 1e3),
                        ("P (uC/cm2)", "Polarization", "uC/cm^2", 1.0))

    def __init__(self, params=None):
        self.p = params or FeFETParams()
        self.reset()

    def reset(self):
        self.P = min(max(self.p.P_init, -1.0), 1.0)
        self.prev_on = False
        self.tr_pot = 0.0          # eligibility traces for STDP timing
        self.tr_dep = 0.0
        self.vt_stdp = 0.0         # retained, order-dependent Vt offset (V)

    # ------------------------------------------------------------------

    def step(self, t, dt, V_gate):
        p = self.p
        V = 0.0 if abs(V_gate) < p.V_th_min else V_gate

        # retention: depolarization toward P = 0 (exact exponential)
        self.P *= math.exp(-dt / p.tau_ret)

        # STDP eligibility traces relax every step (exact exponential)
        if p.tau_stdp > 0:
            decay = math.exp(-dt / p.tau_stdp)
            self.tr_pot *= decay
            self.tr_dep *= decay

        on = V != 0.0
        if on and not self.prev_on:        # rising edge of a gate spike
            # positive gate V potentiates (raises P / lowers Vt); pair it with
            # the surviving OPPOSITE trace to lock in an order-dependent dVt
            if V > 0:
                self.vt_stdp -= p.A_stdp_V * self.tr_dep   # post-after-pre -> Vt down
                self.tr_pot += 1.0
            else:
                self.vt_stdp += p.A_stdp_V * self.tr_pot   # pre-after-post -> Vt up
                self.tr_dep += 1.0

        if V != 0.0:
            # Merz-law switching rate; exponentially suppressed below Vc
            rate = math.exp(min((abs(V) - p.Vc) / p.V0, 50.0)) / p.tau0
            target = 1.0 if V > 0 else -1.0
            # exact relaxation toward the saturated state
            self.P = target + (self.P - target) * math.exp(-dt * rate)
            self.P = min(max(self.P, -1.0), 1.0)

        self.prev_on = on

    # ------------------------------------------------------------------

    @property
    def Vth(self):
        return self.p.Vth0 - 0.5 * self.p.dVth_max * self.P + self.vt_stdp

    @property
    def G(self):
        p = self.p
        x = 1.0 / (1.0 + math.exp(-(p.Vread - self.Vth) / p.SS))
        return p.Gmin + (p.Gmax - p.Gmin) * x

    @property
    def R(self):
        return 1.0 / self.G

    def observables(self):
        return {
            "P (norm.)": self.P,
            "P (uC/cm2)": self.p.Pr * self.P,   # polarization in uC/cm^2
            "Vth (V)": self.Vth,
            "Vt_stdp (V)": self.vt_stdp,   # timing-induced Vt shift (STDP)
            "drift_on": 1.0 if self.prev_on else 0.0,
        }
