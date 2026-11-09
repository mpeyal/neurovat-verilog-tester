"""Reference synapse: a bounded analog conductance with exponential soft
bounds. Not a physical device - it is the public, dependency-free model the
test suite and examples run on, and the template for a new device."""
from dataclasses import dataclass, asdict

import numpy as np

from .base import DeviceArray, spread


@dataclass
class SoftBoundParams:
    g_min: float = 1e-6        # S
    g_max: float = 1e-4        # S
    w_init: float = 0.5        # initial normalised state in [0, 1]
    a_ref: float = 1.0         # drive amplitude that switches at rate 1/tau_w
    tau_w: float = 10e-3       # s; state time constant at |drive| = a_ref
    a_min: float = 1e-6        # |drive| below this is a no-op


class SoftBoundArray(DeviceArray):
    """``w -> 1`` under positive drive and ``w -> 0`` under negative drive,
    each as an exact exponential relaxation at rate ``|a| / (a_ref tau_w)``.

    variation   {"tau_w": sigma, "g_max": sigma, ...} device-to-device spread
    write_noise cycle-to-cycle sigma, as a fraction of each state change
    """
    name = "softbound"
    input_kind = "voltage"
    Params = SoftBoundParams

    def __init__(self, n_post, n_pre, params=None, variation=None,
                 write_noise=0.0, seed=None):
        super().__init__(n_post, n_pre)
        self.p = params if isinstance(params, SoftBoundParams) else SoftBoundParams(**(params or {}))
        self.variation = dict(variation or {})
        self.write_noise = float(write_noise)
        self.rng = np.random.default_rng(seed)
        unknown = set(self.variation) - set(asdict(self.p))
        if unknown:
            raise ValueError(f"Unknown variation parameter(s): {sorted(unknown)}")
        self.g_min, self.g_max = self.p.g_min, self.p.g_max   # nominal range
        v = lambda k: spread(getattr(self.p, k), self.variation.get(k, 0.0),
                             self.shape, self.rng)
        self._tau, self._gmin, self._gmax = v("tau_w"), v("g_min"), v("g_max")
        self.w = np.full(self.shape, float(np.clip(self.p.w_init, 0.0, 1.0)))

    def nominal(self):
        return SoftBoundArray(1, 1, self.p)

    def conductance(self):
        return self._gmin + (self._gmax - self._gmin) * self.w

    def program(self, rows, cols, amps, width, t=0.0):
        amps = np.asarray(amps, float)
        w = self.w[rows, cols]
        tau = self._tau if np.isscalar(self._tau) else self._tau[rows, cols]
        k = 1.0 - np.exp(-width * np.abs(amps) / (self.p.a_ref * tau))
        dw = np.where(amps > 0, (1.0 - w) * k, -w * k)
        dw = np.where(np.abs(amps) < self.p.a_min, 0.0, dw)
        if self.write_noise > 0:
            dw = dw * (1.0 + self.write_noise * self.rng.standard_normal(dw.shape))
        self.w[rows, cols] = np.clip(w + dw, 0.0, 1.0)
        return t + width * len(amps)

    def set_conductance(self, G):
        self.w = np.clip((np.asarray(G, float) - self._gmin)
                         / (self._gmax - self._gmin), 0.0, 1.0)

    def state_dict(self):
        return {"w": self.w.copy(), "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        w = np.asarray(state["w"], float)
        if w.shape != self.shape:
            raise ValueError("Checkpoint architecture mismatch")
        self.w = w.copy()
        self.rng.bit_generator.state = state["rng"]
