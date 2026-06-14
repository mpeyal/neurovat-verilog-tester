"""Example device twin: a simple voltage-driven RRAM / memristor.

A minimal, self-contained demonstration of the twins/ plugin contract.  Drop a
file like this in twins/ and the GUI registers it as a selectable model - no
edits to the GUI or core engine.  Delete this file if you don't want the demo
device to appear.

Physics (toy filamentary RRAM): a gate/terminal VOLTAGE switches the device
conductance between Gmin (HRS) and Gmax (LRS).  Above +Vset it forms (G rises
toward Gmax); below -Vreset it ruptures (G falls toward Gmin); in between it
holds (nonvolatile).  Read-out is the conductance G / resistance R.
"""

import math
from dataclasses import dataclass


@dataclass
class RRAMParams:
    Rmin: float = 1e3          # LRS -> Gmax
    Rmax: float = 100e3        # HRS -> Gmin
    Ginit: float = 1.0 / 10e3  # start mid-range so both set/reset are visible
    Vset: float = 0.05         # V; above this the filament forms (low for the demo)
    Vreset: float = 0.05       # V; below -Vreset it ruptures
    k_set: float = 300.0       # 1/s switching rate scale (set)
    k_reset: float = 300.0     # 1/s (reset)
    tau_ret: float = 1e6       # s; very slow drift toward Ginit (nonvolatile)
    V_th_min: float = 1e-3

    @property
    def Gmin(self):
        return 1.0 / self.Rmax

    @property
    def Gmax(self):
        return 1.0 / self.Rmin


class RRAM:
    name = "Example RRAM (twins/ demo)"
    input_kind = "voltage"

    def __init__(self, params=None):
        self.p = params or RRAMParams()
        self.reset()

    def reset(self):
        self.G = min(max(self.p.Ginit, self.p.Gmin), self.p.Gmax)

    def step(self, t, dt, V_gate):
        p = self.p
        V = 0.0 if abs(V_gate) < p.V_th_min else V_gate
        # nonvolatile retention drift (negligible over a sim)
        self.G += (p.Ginit - self.G) * (1.0 - math.exp(-dt / p.tau_ret))
        if V > p.Vset:                         # SET: rise toward Gmax
            rate = p.k_set * (V - p.Vset)
            self.G = p.Gmax + (self.G - p.Gmax) * math.exp(-dt * rate)
        elif V < -p.Vreset:                    # RESET: fall toward Gmin
            rate = p.k_reset * (-V - p.Vreset)
            self.G = p.Gmin + (self.G - p.Gmin) * math.exp(-dt * rate)
        self.G = min(max(self.G, p.Gmin), p.Gmax)

    @property
    def R(self):
        return 1.0 / self.G

    def observables(self):
        return {"G (S)": self.G}


TWIN_SPEC = {
    "key": "rram_demo",
    "label": "Example RRAM (twins/ demo)",
    "device_class": "RRAM",
    "input_kind": "voltage",
    "va_keywords": ("rram", "reram"),
    "model_class": RRAM,
    "params_class": RRAMParams,
    # conductance device -> default G/R profile is fine; shown explicitly here
    "result_plots": [("R", "R_mem", "ohm", 1.0), ("G", "G", "uS", 1e6)],
    "analysis_metrics": [("G", "G", "uS", 1e6), ("R", "R_mem", "ohm", 1.0)],
}
