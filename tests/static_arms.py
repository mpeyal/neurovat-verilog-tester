"""Device arms used by test_static_paths_unchanged.py: the ideal / NeuroSim
fitted-curve / v3 static (serial timing, G_nv read) synapses with the stream
parameters, write width and unit current under which the recorded
fingerprints in ``data/static_paths_master.json`` were taken.

NeuroSim-style per-pulse nonlinear synapse (Chen, Peng & Yu, IEEE TCAD 2018),
fitted to the ECFET v3 twin:

    LTP:  G(P) = Gmin + B_p (1 - exp(-P / A_p)),   B_p = (Gmax-Gmin) / (1 - exp(-Pmax / A_p))
    LTD:  G(Q) = Gmax - B_d (1 - exp(-Q / A_d)),   B_d = (Gmax-Gmin) / (1 - exp(-Pmax / A_d))

(Q = depression pulses counted from Gmax). A write moves the device along the
curve of its direction from the pulse number that corresponds to its present
conductance. The device has no time. A pulse of current ``amp`` and width
``w`` counts as ``|amp| w / q_unit`` unit pulses, ``q_unit`` = the v3 unit
pulse charge (50 pA x 10 ms).

The arithmetic here must not change: the test compares results bit for bit.
"""
import math

import numpy as np
from scipy.optimize import minimize_scalar

from neurovat.devices import DeviceArray

WIDTH = 10e-3
WINDOW_MS = 10.0
I_UNIT = 50e-12


def v3_params():
    from simulation_core.model_v3 import V3Params
    return V3Params.paper(A_stdp=0.0)


def make_array(arm, n_post, n_pre, fit):
    from neurovat.devices import EcfetV3Array
    if arm == "neurosim":
        return array_from_fit(fit, n_post, n_pre)
    if arm == "v3_static":
        return EcfetV3Array(v3_params(), n_post, n_pre, "serial", "nv")
    raise ValueError(arm)


def base_params(a):
    from neurovat.masquelier import MasquelierParams
    n = a.afferents
    return MasquelierParams(n_afferents=n, n_pattern=n // 2,
                            threshold=500.0 * n / 2000.0, n_repeats=1,
                            spacing="even")


class NeuroSimArray(DeviceArray):
    name = "neurosim_fit"
    input_kind = "current"

    def __init__(self, n_post, n_pre, g_min, g_max, A_p, A_d, p_max, q_unit):
        super().__init__(n_post, n_pre)
        self.g_min, self.g_max = float(g_min), float(g_max)
        self.A_p, self.A_d, self.p_max = float(A_p), float(A_d), float(p_max)
        self.q_unit = float(q_unit)
        span = self.g_max - self.g_min
        self.B_p = span / (1.0 - math.exp(-self.p_max / self.A_p))
        self.B_d = span / (1.0 - math.exp(-self.p_max / self.A_d))
        self.G = np.full(self.shape, 0.5 * (self.g_min + self.g_max))

    def nominal(self):
        return NeuroSimArray(1, 1, self.g_min, self.g_max, self.A_p, self.A_d,
                             self.p_max, self.q_unit)

    def conductance(self):
        return self.G

    def program(self, rows, cols, amps, width, t=0.0):
        amps = np.asarray(amps, float)
        g = self.G[rows, cols]
        n = np.abs(amps) * width / self.q_unit
        lo, hi = self.g_min, self.g_max
        # LTP: pulse number of g on the LTP curve, then +n
        fp = np.clip((g - lo) / self.B_p, 0.0, 1.0 - 1e-15)
        P = -self.A_p * np.log1p(-fp)
        g_pot = lo + self.B_p * (1.0 - np.exp(-(P + n) / self.A_p))
        # LTD: depression-pulse number of g on the LTD curve, then +n
        fd = np.clip((hi - g) / self.B_d, 0.0, 1.0 - 1e-15)
        Q = -self.A_d * np.log1p(-fd)
        g_dep = hi - self.B_d * (1.0 - np.exp(-(Q + n) / self.A_d))
        new = np.where(amps > 0, g_pot, np.where(amps < 0, g_dep, g))
        self.G[rows, cols] = np.clip(new, lo, hi)
        return t + width * len(amps)

    def set_conductance(self, G):
        self.G = np.clip(np.asarray(G, float), self.g_min, self.g_max).copy()

    def state_dict(self):
        return {"G": self.G.copy()}

    def load_state_dict(self, state):
        self.G = np.asarray(state["G"], float).copy()


def v3_curves(params, gap_s=0.0):
    """Retained G_nv after each unit pulse (Q_ref = 50 pA x 10 ms): LTP from
    Gmin to the window top, then LTD back. ``Pmax`` pulses per branch."""
    from simulation_core.model_v3 import EcfetV3
    dev = EcfetV3(params)
    dev.reset()
    dev.x, dev.G_nv = 0.0, params.Gmin
    p_max = int(round(params.Q_full / params.Q_ref))
    i_unit, w = 50e-12, params.Q_ref / 50e-12
    ltp, ltd = [dev.G_nv], []
    t = 0.0
    for _ in range(p_max):
        dev.step(t, w, i_unit)
        t += w
        if gap_s:
            dev.step(t, gap_s, 0.0)
            t += gap_s
        ltp.append(dev.G_nv)
    ltd.append(dev.G_nv)
    for _ in range(p_max):
        dev.step(t, w, -i_unit)
        t += w
        if gap_s:
            dev.step(t, gap_s, 0.0)
            t += gap_s
        ltd.append(dev.G_nv)
    return np.array(ltp), np.array(ltd), p_max


def _fit(curve, p_max, lo, hi, rising):
    P = np.arange(curve.size, dtype=float)
    span = hi - lo

    def model(logA):
        A = math.exp(logA)
        B = span / (1.0 - math.exp(-p_max / A))
        s = B * (1.0 - np.exp(-P / A))
        return lo + s if rising else hi - s

    def cost(logA):
        return float(np.mean((model(logA) - curve) ** 2))

    r = minimize_scalar(cost, bounds=(math.log(p_max * 1e-3),
                                      math.log(p_max * 1e3)), method="bounded",
                        options={"xatol": 1e-9})
    fit = model(r.x)
    rmse = math.sqrt(cost(r.x))
    return {"A": math.exp(r.x), "A_over_Pmax": math.exp(r.x) / p_max,
            "rmse_S": rmse, "rmse_frac_range": rmse / span,
            "max_err_frac_range": float(np.max(np.abs(fit - curve)) / span)}


def fit_v3(params):
    """Fit A_p / A_d to the v3 twin's retained LTP / LTD curves."""
    ltp, ltd, p_max = v3_curves(params)
    lo, hi = params.Gmin, params.Gmax
    fp = _fit(ltp, p_max, lo, hi, True)
    fd = _fit(ltd, p_max, lo, hi, False)
    return {"p_max": p_max, "q_unit": params.Q_ref, "g_min": lo, "g_max": hi,
            "ltp": fp, "ltd": fd,
            "curve_end_ltp": float(ltp[-1]), "curve_end_ltd": float(ltd[-1])}


def array_from_fit(fit, n_post, n_pre):
    return NeuroSimArray(n_post, n_pre, fit["g_min"], fit["g_max"],
                         fit["ltp"]["A"], fit["ltd"]["A"], fit["p_max"],
                         fit["q_unit"])
