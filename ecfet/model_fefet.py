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

    @property
    def Gmin(self):
        return 1.0 / self.Rmax

    @property
    def Gmax(self):
        return 1.0 / self.Rmin


class FeFET:
    name = "FeFET (Merz/Preisach-lite)"
    input_kind = "voltage"     # waveform amplitude = gate voltage (V)

    def __init__(self, params=None):
        self.p = params or FeFETParams()
        self.reset()

    def reset(self):
        self.P = min(max(self.p.P_init, -1.0), 1.0)
        self.prev_on = False

    # ------------------------------------------------------------------

    def step(self, t, dt, V_gate):
        p = self.p
        V = 0.0 if abs(V_gate) < p.V_th_min else V_gate

        # retention: depolarization toward P = 0 (exact exponential)
        self.P *= math.exp(-dt / p.tau_ret)

        if V != 0.0:
            # Merz-law switching rate; exponentially suppressed below Vc
            rate = math.exp(min((abs(V) - p.Vc) / p.V0, 50.0)) / p.tau0
            target = 1.0 if V > 0 else -1.0
            # exact relaxation toward the saturated state
            self.P = target + (self.P - target) * math.exp(-dt * rate)
            self.P = min(max(self.P, -1.0), 1.0)

        self.prev_on = V != 0.0

    # ------------------------------------------------------------------

    @property
    def Vth(self):
        return self.p.Vth0 - 0.5 * self.p.dVth_max * self.P

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
            "Vth (V)": self.Vth,
            "drift_on": 1.0 if self.prev_on else 0.0,
        }
