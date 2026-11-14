"""Python twin of the Stanford-PKU metal-oxide RRAM compact model, v1.0.0.

Source model: `rram_v_1_0_0.va` from nanoHUB (doi:10.4231/D37H1DN48), kept
locally as verilog/stanford_pku_rram.va.  This file is a modification of that
software (a Python re-statement of its equations), so the copyright notices of
the original are reproduced here, as its licence requires:

    /*******************************************************************
    Copyright @ 2014 Stanford University (Stanford)

    The terms under which the software and associated documentation (the
    Software) is provided are as the following:

    The Software is provided "as is", without warranty of any kind, express or
    implied, including but not limited to the warranties of merchantability,
    fitness for a particular purpose and noninfringement. In no event shall the
    authors or copyright holders be liable for any claim, damages or other
    liability, whether in an action of contract, tort or otherwise, arising
    from, out of or in connection with the Software or the use or other
    dealings in the Software.

    Stanford grants, free of charge, to any users the right to modify, copy,
    and redistribute the Software, both within the user's organization and
    externally, subject to the following restrictions:

    1. The users agree not to charge for the Stanford code itself but may
       charge for additions, extensions, or support.

    2. In any product based on the Software, the users agree to acknowledge
       the Stanford Nanoelectronics Research Group of Prof. H.-S. Philip Wong
       that developed the software and cite the relevant publications that
       form the basis of the Software. This acknowledgment shall appear in the
       product documentation.

    3. The users agree to obey all U.S. Government restrictions governing
       redistribution or export of the software.

    4. The users agree to reproduce any copyright notice which appears on the
       software on any copy or modification of such made available to others.

    Agreed to by
    H.-S. Philip Wong, Stanford University
    May 6, 2014
    ********************************************************************/
    /*******************************************************************
        Metal Oxide Resistive Random Access Memory
                VerilogA Models
                 Version 1.0.0
             With Dynamic Variations

    Copyright The Board Trustees of the Leland Stanford Junior University 2013
    Jane Zizhen Jiang, Shimeng Yu, Ximeng Guan, Prof. H-S Philip Wong

    12/05/2013 Last Modified by Jane Zizhen Jiang.

    Metal Oxide Resistive Random Access Memory Verilog-A implementation
    based on
       "Verilog-A Compact Model for Oxide-based Resistive Random Access
       Memory" by Jane Zizhen Jiang, Shimeng Yu, Yi Wu, Jesse H. Engel,
       Ximeng Guan, Prof. H.-S Philip Wong

    Patent Pending.

    Model name: rram_random_v_1_0_0.va
    *******************************************************************/
    /*******************************************************************
    * TERMS OF USE
    * Stanford Leland Junior University and the authors ("Stanford")
    * provide these model files to you subject to the Terms of Use,
    * which may be updated by us from time to time without notice to you.
    * The most up-to-date Terms of Use can be found at
    * https://nano.stanford.edu/stanford-rram-model
    *******************************************************************/

Acknowledgement: the model was developed by the Stanford Nanoelectronics
Research Group of Prof. H.-S. Philip Wong.  Cite Z. Jiang et al., "A Compact
Model for Metal-Oxide Resistive Random Access Memory With Experiment
Verification", IEEE Trans. Electron Devices 63(5), 2016, and
Z. Jiang, H.-S. P. Wong, "Stanford University Resistive-Switching Random
Access Memory (RRAM) Verilog-A Model", nanoHUB, 2014, doi:10.4231/D37H1DN48.
Non-commercial: do not charge for this code.

Equations (the .va analog block, two-terminal TE/BE, V = V(TE,BE)):

    T      = T_ini + |V * I * Rth|                         (Joule heating)
    gamma  = gamma_ini - beta * (gap / 1 nm)^3,  gamma_ini = gamma0 (V >= 0),
                                                  16 hard-coded (V < 0)
    gamma  = 0  if gamma * |V| / tox < F_min                (field threshold)
    dy/dt  = -Vel0 exp(-q Ea / kb T) sinh(gamma a0 / tox * q V / kb T)
             + N(0,1) * deltaGap0 * model_switch / (1 + exp((T_crit - T)/T_smth))
    y      = idt(dy/dt, gap_ini)
    gap    = clip(y, gap_min, gap_max)
    I      = I0 exp(-gap / g0) sinh(V / V0)                  (into TE)

Integration semantics (what this twin reproduces, and how):

* `idt` is a continuous-time integral; the clamp is applied to its OUTPUT and
  never fed back.  The integrator state y therefore keeps moving past
  gap_min / gap_max while the field is above threshold ("wind-up"), and a
  later opposite pulse must first bring y back inside before the gap moves.
  The twin keeps y as its state and exposes both y ("gap_state") and the
  clamped gap.  `clamp_state=1` clamps y as well (MemTorch's reading), for a
  like-for-like comparison with MemTorch only.
* gamma reads the variable `gap` before the .va reassigns it, i.e. the
  previous evaluation's clamped gap, and T uses the probed branch current.
  Both are evaluation-order (hidden-state) effects of a step-by-step solver;
  their continuous-time limit is the system above with gap = clip(y) at the
  same instant, which is what the twin integrates.
* The twin integrates that ODE adaptively (Dormand-Prince 5(4), FSAL) inside
  every `step()`, with V(t) linear between the step's end points
  (`step_linear`) or constant (`step`), so the result does not depend on the
  caller's time step.  `integrator=1` instead takes ONE forward-Euler step per
  call (the .va lines evaluated once per time point) for method studies.
* $bound_step(time_step) only limits the simulator's step; the adaptive twin
  needs no such bound.  `time_step` is kept as a parameter for the decks.
* Variability (`model_switch=1`): the .va draws a new $rdist_normal sample on
  every analog evaluation (Newton iterations included), so its result depends
  on the simulator's step and iteration count (the displacement std scales
  with sqrt(step)).  The twin draws one sample per `step()` call from a
  numpy Generator seeded with `rand_seed_ini` at reset (deterministic), held
  over that step.  It is off by default, like the .va.

`.G` is the read conductance I(V_read)/V_read at the present gap (0.1 V, the
read used by MemTorch).  The I-V is sinh-shaped, so I = G*V holds only at the
read voltage; the instantaneous current is in observables()["I_TE (A)"].
"""

import math
from dataclasses import dataclass

import numpy as np

# Dormand-Prince 5(4) tableau
_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_A = ((),
      (1 / 5,),
      (3 / 40, 9 / 40),
      (44 / 45, -56 / 15, 32 / 9),
      (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
      (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
      (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84))
_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200,
       187 / 2100, 1 / 40)
_E = tuple(b5 - b4 for b5, b4 in zip(_B5, _B4))

GAMMA_INI_NEG = 16.0      # hard-coded in the .va for V < 0 (`gamma_ini = 16`)


@dataclass
class StanfordPKUParams:
    # ---- .va parameters (same names and defaults as rram_v_1_0_0.va) ------
    version: float = 1.00
    model_switch: int = 0            # 0 standard, 1 dynamic (variability)
    kb: float = 1.3806503e-23        # `P_K (J/K); pass explicitly in decks
    q: float = 1.602176462e-19       # `P_Q (C);  pass explicitly in decks
    g0: float = 0.25e-9              # m
    V0: float = 0.25                 # V
    Vel0: float = 10.0               # m/s
    I0: float = 1000e-6              # A
    beta: float = 0.8
    gamma0: float = 16.0
    T_crit: float = 450.0            # K
    deltaGap0: float = 0.02          # m/s (variation rate scale)
    T_smth: float = 500.0            # K
    Ea: float = 0.6                  # eV
    a0: float = 0.25e-9              # m
    T_ini: float = 273.0 + 25.0      # K
    F_min: float = 1.4e9             # V/m
    gap_ini: float = 2e-10           # m (idt initial condition)
    gap_min: float = 2e-10           # m
    gap_max: float = 17e-10          # m
    Rth: float = 2.1e3               # K/W
    tox: float = 12e-9               # m
    rand_seed_ini: int = 0
    time_step: float = 1e-9          # s, $bound_step (simulator only)
    pulse_width: float = 20e-9       # s, declared in the .va, unused there
    # ---- twin-only options ---------------------------------------------
    V_read: float = 0.1              # V, read voltage that defines .G
    clamp_state: int = 0             # 1 = clamp y too (MemTorch reading)
    integrator: int = 0              # 0 adaptive DOPRI5, 1 one Euler step/call
    rtol: float = 1e-9
    atol: float = 1e-16              # m


class StanfordPKURRAM:
    name = "Stanford-PKU RRAM v1.0.0"
    input_kind = "voltage"
    TERMINALS = 2

    def __init__(self, params=None):
        self.p = params or StanfordPKUParams()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        p = self.p
        self.y = float(p.gap_ini)                 # idt state (unclamped)
        if p.clamp_state:
            self.y = self._clip(self.y)
        self.V = 0.0
        self._h = None                            # adaptive step memory
        # generator state kept as a plain dict so device checkpoints stay
        # JSON-safe (neurovat capture_state packs vars(device))
        self._rng_state = np.random.default_rng(
            int(p.rand_seed_ini) & 0xFFFFFFFF).bit_generator.state
        self._xi = 0.0                            # held N(0,1) sample
        self.n_rhs = 0                            # rhs evaluations (cost)

    def _clip(self, y):
        p = self.p
        return min(max(y, p.gap_min), p.gap_max)

    @property
    def gap(self):
        return self._clip(self.y)

    # ---- algebraic parts ------------------------------------------------
    def current(self, V, gap=None):
        """I(TE->BE) at voltage V and gap (default: present clamped gap)."""
        p = self.p
        g = self.gap if gap is None else gap
        return p.I0 * math.exp(-g / p.g0) * math.sinh(V / p.V0)

    def temperature(self, V, gap=None):
        return self.p.T_ini + abs(V * self.current(V, gap) * self.p.Rth)

    def _rate(self, V, y):
        """dy/dt of the .va (deterministic part + held noise sample)."""
        p = self.p
        self.n_rhs += 1
        g = min(max(y, p.gap_min), p.gap_max)
        i_tb = p.I0 * math.exp(-g / p.g0) * math.sinh(V / p.V0)
        T = p.T_ini + abs(V * i_tb * p.Rth)
        gamma_ini = GAMMA_INI_NEG if V < 0 else p.gamma0
        gamma = gamma_ini - p.beta * (g / 1e-9) ** 3
        if gamma * abs(V) / p.tox < p.F_min:
            gamma = 0.0
        kT = p.kb * T
        r = -p.Vel0 * math.exp(-p.q * p.Ea / kT) * math.sinh(
            gamma * p.a0 / p.tox * p.q * V / kT)
        if p.model_switch:
            r += self._xi * p.deltaGap0 * p.model_switch / (
                1.0 + math.exp((p.T_crit - T) / p.T_smth))
        return r

    # ---- stepping -------------------------------------------------------
    def step(self, t, dt, V):
        """Advance by dt with a constant TE-BE voltage V (twin contract)."""
        self.step_linear(t, dt, V, V)

    def step_linear(self, t, dt, v0, v1):
        """Advance by dt with V(t) linear from v0 (at t) to v1 (at t+dt)."""
        p = self.p
        v0, v1 = float(v0), float(v1)
        if dt <= 0.0:
            self.V = v1
            return
        if p.model_switch:
            g = np.random.Generator(np.random.PCG64())
            g.bit_generator.state = self._rng_state
            self._xi = float(g.standard_normal())
            self._rng_state = g.bit_generator.state
        if p.integrator == 1:
            self.y += self._rate(0.5 * (v0 + v1), self.y) * dt
        else:
            self._integrate(dt, v0, v1)
        if p.clamp_state:
            self.y = self._clip(self.y)
        self.V = v1

    def _absorbing_zero(self, v0, v1):
        """True when the rate is exactly zero now and stays zero all step
        (constant V below the field threshold, no noise): skip the work."""
        if v0 != v1 or self.p.model_switch:
            return False
        return self._rate(v0, self.y) == 0.0

    def _integrate(self, dt, v0, v1):
        p = self.p
        if self._absorbing_zero(v0, v1):
            return
        y = self.y
        slope = (v1 - v0) / dt
        # outside [gap_min, gap_max] with constant V the rate is constant:
        # exact closed form while y stays on the same side of the bound
        if v0 == v1 and not p.clamp_state and not (p.gap_min <= y <= p.gap_max):
            r = self._rate(v0, y)
            yn = y + r * dt
            if (y < p.gap_min and yn < p.gap_min) or (y > p.gap_max and yn > p.gap_max):
                self.y = yn
                return
        # RESET (V < 0) self-limits at the stop gap g*(V) where gamma*|V|/tox
        # reaches F_min: the rate is zero above it, so the exact solution
        # never crosses g* upward.  On a ramp g* moves and the state slides
        # along it; an explicit error control would chatter on that jump, so
        # a step that provably reaches g* (even the slowest rate on the way
        # gets there within h) is accepted with the state set to g*.
        s_t = 0.0
        h = min(self._h or dt, dt)
        k1 = self._rate(v0, y)
        while s_t < dt:
            h = min(h, dt - s_t)
            if h <= 0.0:
                break
            v_s, v_e = v0 + slope * s_t, v0 + slope * (s_t + h)
            ks = [k1]
            for i in range(1, 7):
                yi = y + h * sum(a * k for a, k in zip(_A[i], ks))
                ks.append(self._rate(v0 + slope * (s_t + _C[i] * h), yi))
            y5 = y + h * sum(b * k for b, k in zip(_B5, ks))
            err = abs(h * sum(e * k for e, k in zip(_E, ks)))
            tol = p.atol + p.rtol * max(abs(y), abs(y5))
            upper = math.inf
            if v_s < 0.0 and v_e < 0.0:          # |V| is monotone in the step
                gs_s, gs_e = self._g_stop(v_s), self._g_stop(v_e)
                if y >= gs_s and y >= gs_e:       # above the stop all step:
                    s_t += h                      # the rate is exactly zero
                    k1 = self._rate(v_e, y)
                    h *= 2.0
                    continue
                upper = max(y, gs_s, gs_e)        # cannot be crossed upward
                if math.isfinite(gs_e) and gs_e > y and y5 > gs_e:
                    # reaching a rising stop: accept g* if even the slowest
                    # rate on the way (just below the stop) gets there
                    f_lo = self._rate(v_e, gs_e - 1e-15)
                    if gs_s > y and math.isfinite(gs_s):
                        f_lo = min(f_lo, self._rate(v_s, gs_s - 1e-15))
                    if y + h * f_lo >= gs_e or err <= tol:
                        s_t += h
                        y = gs_e
                        k1 = self._rate(v_e, y)
                        h *= 2.0
                        continue
                    upper = -math.inf             # force a rejection
            if err <= tol and y5 <= upper:
                s_t += h
                y = y5
                k1 = ks[6]
                fac = 5.0 if err == 0.0 else min(5.0, 0.9 * (tol / err) ** 0.2)
                h *= max(fac, 0.2)
                continue
            h *= max(0.1, 0.9 * (tol / err) ** 0.2) if err > tol else 0.5
            if h < 1e-22:
                raise FloatingPointError("Stanford-PKU twin: step underflow")
        self.y = y
        self._h = h

    def _g_stop(self, V):
        """Stop gap g*(V) for V < 0 (inf: none in range, -inf: always cut)."""
        p = self.p
        if V >= 0.0:
            return math.inf
        s = (GAMMA_INI_NEG - p.F_min * p.tox / abs(V)) / p.beta
        if s <= 0.0:
            return -math.inf
        g = 1e-9 * s ** (1.0 / 3.0)
        return math.inf if g > p.gap_max else g

    # ---- read-out -------------------------------------------------------
    @property
    def G(self):
        p = self.p
        return self.current(p.V_read) / p.V_read

    @G.setter
    def G(self, g):
        """Set the state to the gap whose read conductance is g (clamped to
        [gap_min, gap_max]); used when a crossbar restores weights."""
        p = self.p
        gap = -p.g0 * math.log(g * p.V_read / (p.I0 * math.sinh(p.V_read / p.V0)))
        self.y = self._clip(gap)

    @property
    def R(self):
        return 1.0 / self.G

    def observables(self):
        g = self.gap
        V = self.V
        i = self.current(V, g)
        return {"gap (nm)": g * 1e9, "gap_state (nm)": self.y * 1e9,
                "I_TE (A)": i, "V_TE (V)": V,
                "T (K)": self.p.T_ini + abs(V * i * self.p.Rth),
                "G_read (S)": self.G}


TWIN_SPEC = {
    "key": "stanford_pku",
    "label": "Stanford-PKU RRAM v1.0.0 (Jiang, Wong et al.)",
    "device_class": "RRAM",
    "input_kind": "voltage",
    # most specific keyword wins in va_scan: the nanoHUB file/module name and
    # the local file name both outrank the generic "rram" of the demo twin
    "va_keywords": ("stanford_pku", "stanford_rram", "pku_rram", "rram_v_1_0_0",
                    "rram_random_v_1_0_0"),
    "model_class": StanfordPKURRAM,
    "params_class": StanfordPKUParams,
    "result_plots": [("G", "G_read", "uS", 1e6), ("gap (nm)", "gap", "nm", 1.0)],
    "analysis_metrics": [("G", "G_read", "uS", 1e6), ("gap (nm)", "gap", "nm", 1.0)],
}
