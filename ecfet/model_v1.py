"""Faithful Python port of basic_v1_Diffu_Drft_verilog.va.

This reproduces the *intended* behavior of the Verilog-A analog block so the
model can be exercised without Spectre.  Each `step()` call corresponds to one
accepted transient timestep.

Port notes (deviations from a literal reading of the .va, see README):

1. Integrator sign.  The .va contributes `I(cap) <+ Idrive` together with
   `I(cap) <+ ddt(V(cap))`.  KCL on the floating `capactive` node makes the
   solved equation dV/dt = -Idrive, i.e. positive gate current would drive
   V(cap) DOWN, contradicting every guard in the module
   (`M_cap >= Rmax && Ieff > 0`, source_dir, ...).  The port implements the
   intended dV/dt = +Idrive.  Fix in the .va: `I(cap) <+ -Idrive;`.

2. The G_sat soft clamp is implemented as an ideal hard clamp of V(cap) to
   [Rmin, Rmax]/Rdrift1c (G_sat = 1e6 makes the residual overshoot ~uV-level
   anyway).

3. `V(cap) <+ ...` at pulse start / during diffusion turns the branch into a
   switch branch in Verilog-AMS (voltage contribution wins, the current
   contributions that iteration are discarded).  The port does the same:
   when the voltage assignment path is active it overwrites the integrator
   state directly.
"""

import math
from dataclasses import dataclass


@dataclass
class V1Params:
    Rmin: float = 0.1e3
    Rmax: float = 10e3
    Rinit: float = 500.0
    Rdrift1c: float = 20e12   # ohm per coulomb of gate charge

    w: float = 2.828e-6
    l: float = 4e-6
    D: float = 0.4e-9

    tao3: float = 19.0
    c1: float = 10.0
    c2: float = 9.9
    c3: float = 0.1

    tau_scale: float = 1.0
    I_drift_th: float = 1e-12
    leak_drift_scale: float = 0.0

    diffusion_window: float = 1.0   # hard 1 s window from the .va

    @property
    def tao1(self):
        return self.w * self.w / (2.0 * self.D) * self.tau_scale

    @property
    def tao2(self):
        return self.l * self.l / (2.0 * self.D) * self.tau_scale

    @property
    def tao3_eff(self):
        return self.tao3 * self.tau_scale


class EcfetV1:
    name = "v1 (Verilog-A port)"

    def __init__(self, params=None):
        self.p = params or V1Params()
        self.reset()

    def reset(self):
        p = self.p
        self.M = min(max(p.Rinit, p.Rmin), p.Rmax)
        self.Vcap = self.M / p.Rdrift1c
        self.source_dir = 1.0
        self.sc_pulse_arrived = False
        self.diffusion_start_time = 0.0
        self.M_before_diffusion = self.M
        self.prev_drift_on = False

    # ------------------------------------------------------------------

    def step(self, t, dt, I_gate):
        """Advance one timestep ending at t+dt with constant gate current I_gate."""
        p = self.p

        absI = abs(I_gate)
        Ieff = p.leak_drift_scale * I_gate if absI < p.I_drift_th else I_gate
        drift_on = abs(Ieff) >= p.I_drift_th

        pulse_started = drift_on and not self.prev_drift_on
        pulse_ended = (not drift_on) and self.prev_drift_on

        if pulse_started:
            # switch-branch voltage write: re-seed integrator from present M
            self.Vcap = self.M / p.Rdrift1c
            self.source_dir = 1.0 if Ieff > 0 else -1.0

        if drift_on:
            M_cap = self.Vcap * p.Rdrift1c
            Idrive = Ieff
            if (M_cap >= p.Rmax and Ieff > 0) or (M_cap <= p.Rmin and Ieff < 0):
                Idrive = 0.0
            self.Vcap += Idrive * dt                       # see port note 1
            self.Vcap = min(max(self.Vcap, p.Rmin / p.Rdrift1c),
                            p.Rmax / p.Rdrift1c)           # G_sat clamp
            self.M = min(max(self.Vcap * p.Rdrift1c, p.Rmin), p.Rmax)
            self.source_dir = 1.0 if Ieff > 0 else -1.0

        if pulse_ended:
            self.M_before_diffusion = self.M
            self.diffusion_start_time = t
            self.sc_pulse_arrived = True

        t_end = t + dt
        if (not drift_on) and self.sc_pulse_arrived and \
                (t_end - self.diffusion_start_time) < p.diffusion_window:
            tt = t_end - self.diffusion_start_time
            diffu_M = (p.c1 * math.exp(-tt / p.tao1) +
                       p.c2 * math.exp(-tt / p.tao2) +
                       p.c3 * math.exp(-tt / p.tao3_eff))
            diffu_M = max(diffu_M, 0.0)
            if self.source_dir < 0:
                self.M = self.M_before_diffusion + diffu_M
            else:
                self.M = self.M_before_diffusion - diffu_M
            self.M = min(max(self.M, p.Rmin), p.Rmax)
            self.Vcap = self.M / p.Rdrift1c                # switch-branch write
        elif (not drift_on) and \
                (t_end - self.diffusion_start_time) >= p.diffusion_window:
            self.sc_pulse_arrived = False

        self.prev_drift_on = drift_on

    # ------------------------------------------------------------------

    @property
    def R(self):
        return self.M

    @property
    def G(self):
        return 1.0 / self.M

    def observables(self):
        return {
            "Vcap (V)": self.Vcap,
            "drift_on": 1.0 if self.prev_drift_on else 0.0,
            "diffusing": 1.0 if self.sc_pulse_arrived else 0.0,
        }
