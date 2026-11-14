"""Vectorised ECFET v3 twin: the whole crossbar of
``simulation_core.model_v3.EcfetV3`` devices as NumPy arrays.

The update is a line-by-line transcription of ``EcfetV3.step`` (parts A-E in
the same order, the same sub-step rule, the same floating-point operation
order), applied to any subset of devices at once with a per-device ``dt`` and
drive. It is checked against the scalar twin (``ScalarTwinArray`` over
``EcfetV3``) to machine tolerance in ``tests/test_vector_twin.py``; the only
differences are ``np.exp`` vs ``math.exp`` rounding (~1 ulp).

Timing and read modes are those of :class:`~neurovat.devices.ScalarTwinArray`
(``timing="physical"`` | ``"serial"``, ``read="full"`` | ``"nv"``). An idle
interval is one exact-exponential step (O(1) whatever its length).

Not reproduced: the scalar twin's cycle-to-cycle noise stream
(``sigma_c2c > 0`` draws from ``random.Random`` per device). This array draws
from a NumPy generator instead, so with ``sigma_c2c > 0`` it is statistically,
not sample-for-sample, the scalar twin (``vector_twin_factory`` therefore
keeps the scalar adapter in that case).
"""
import math

import numpy as np

from .base import DeviceArray


def _sig_scalar(t):
    if t >= 0.0:
        return 1.0 / (1.0 + math.exp(-t))
    z = math.exp(t)
    return z / (1.0 + z)


class EcfetV3Array(DeviceArray):
    name = "ecfet_v3"
    input_kind = "current"

    def __init__(self, params, n_post, n_pre, timing="physical", read="full",
                 seed=None):
        super().__init__(n_post, n_pre)
        if timing not in ("serial", "physical"):
            raise ValueError(f"bad timing {timing!r}")
        if read not in ("nv", "full"):
            raise ValueError(f"bad read {read!r}")
        p = self.p = params
        self.timing, self.read = timing, read
        self.time_dynamic = timing == "physical"
        self.g_min, self.g_max = float(p.Gmin), float(p.Gmax)
        self._k = float(p.k)
        self._a = _sig_scalar(-0.5 * p.k)
        self._b = _sig_scalar(0.5 * p.k)
        self._taus = tuple(float(t) for t in p.taus)
        self._cw = tuple(p.pool_weights)
        self._ws = tuple(p.stdp_weights)
        self._edl = p.kappa_v / max(1.0 - p.kappa_v, 1e-9)
        self._eta = (1.0 if not p.kinetic_on else p.eta_inf + (1.0 - p.eta_inf)
                     * (1.0 - math.exp(-p.t_on / p.tau_kin)))
        self.seed = seed
        self.reset()

    # ---- state ----------------------------------------------------------
    def reset(self):
        p, n = self.p, self.shape[0] * self.shape[1]
        g0 = min(max(1.0 / p.Rinit, p.Gmin), p.Gmax)
        x0 = p.x_of_G(g0)
        self.x = np.full(n, x0)
        self.G_nv = np.full(n, p.G_of_x(x0))
        self.lock = np.zeros(n)
        self.pools = np.zeros((3, n))
        self.trp = np.zeros((3, n))
        self.trd = np.zeros((3, n))
        self.prev = np.zeros(n, bool)
        self.cf = np.ones(n)
        self.t_dev = np.zeros(n)
        self.rng = np.random.default_rng(self.seed)

    def nominal(self):
        return EcfetV3Array(self.p, 1, 1, self.timing, self.read)

    def _G_of_x(self, x):
        x = np.minimum(np.maximum(x, 0.0), 1.0)
        t = self._k * (x - 0.5)
        pos = t >= 0.0
        e = np.exp(np.where(pos, -t, t))
        sig = np.where(pos, 1.0 / (1.0 + e), e / (1.0 + e))
        S = (sig - self._a) / (self._b - self._a)
        return self.p.Gmin + (self.p.Gmax - self.p.Gmin) * S

    # ---- the update: EcfetV3.step on devices ``idx`` ----------------------
    def _step(self, idx, dt, I):
        p = self.p
        idx = np.asarray(idx, np.int64)
        if not idx.size:
            return
        dt = np.broadcast_to(np.asarray(dt, float), idx.shape)
        I = np.broadcast_to(np.asarray(I, float), idx.shape)
        absI = np.abs(I)
        Ieff = np.where(absI < p.I_drift_th, p.leak_drift_scale * I, I)
        drift = np.abs(Ieff) >= p.I_drift_th
        taus = self._taus

        # (A) eligibility traces relax
        for i in range(3):
            d = np.exp(-dt / taus[i])
            self.trp[i, idx] *= d
            self.trd[i, idx] *= d

        # (B) pulse onset: c2c factor + STDP lock-in, then clamp
        on = drift & ~self.prev[idx]
        if on.any():
            o = idx[on]
            cf = np.ones(o.size)
            if p.sigma_c2c > 0:
                cf = np.maximum(0.0, 1.0 + self.rng.normal(0.0, p.sigma_c2c,
                                                           o.size))
            self.cf[o] = cf
            sgn = -np.copysign(1.0, Ieff[on]) * p.polarity
            pot, dep = o[sgn > 0.0], o[~(sgn > 0.0)]
            ws = self._ws
            if pot.size:
                s = self.trd[0, pot] + self.trd[1, pot] + self.trd[2, pot]
                self.lock[pot] += p.A_stdp * s
                self.trd[:, pot] = 0.0
                for i in range(3):
                    self.trp[i, pot] = np.minimum(self.trp[i, pot] + ws[i], ws[i])
            if dep.size:
                s = self.trp[0, dep] + self.trp[1, dep] + self.trp[2, dep]
                self.lock[dep] -= p.A_stdp * s
                self.trp[:, dep] = 0.0
                for i in range(3):
                    self.trd[i, dep] = np.minimum(self.trd[i, dep] + ws[i], ws[i])
            total = self.G_nv[o] + self.lock[o]
            self.lock[o] = np.where(total > p.Gmax, p.Gmax - self.G_nv[o],
                                    np.where(total < p.Gmin,
                                             p.Gmin - self.G_nv[o],
                                             self.lock[o]))

        # (C) volatile pools relax
        for i in range(3):
            self.pools[i, idx] *= np.exp(-dt / taus[i])

        # (D) retention: x relaxes toward x_eq
        self.x[idx] = p.x_eq + (self.x[idx] - p.x_eq) * np.exp(-dt / p.tau_ret)
        self.G_nv[idx] = self._G_of_x(self.x[idx])

        # (E) conserved-x write, sub-stepped per device
        wr = Ieff != 0.0
        if wr.any():
            w = idx[wr]
            Iw, dtw = Ieff[wr], dt[wr]
            sign = -np.copysign(1.0, Iw) * p.polarity
            cf = self.cf[w]
            eta = self._eta
            dxdt = np.abs(Iw) / p.Q_full * eta * cf
            move = dxdt * dtw
            n_sub = np.minimum(500, (move / 0.002).astype(np.int64) + 1)
            h = dtw / n_sub
            dx_sub = sign * np.abs(Iw) * h / p.Q_full * eta * cf
            cw, edl = self._cw, self._edl
            for s in range(int(n_sub.max())):
                act = s < n_sub
                a = w[act]
                x_new = np.minimum(np.maximum(self.x[a] + dx_sub[act], 0.0), 1.0)
                G_new = self._G_of_x(x_new)
                dG = G_new - self.G_nv[a]
                self.x[a] = x_new
                self.G_nv[a] = G_new
                dG_edl = np.copysign(edl * np.abs(dG), dG)
                for i in range(3):
                    self.pools[i, a] += cw[i] * dG_edl

        self.prev[idx] = drift

    # ---- reads ----------------------------------------------------------
    def _full(self):
        tot = (self.G_nv + self.lock) + ((self.pools[0] + self.pools[1])
                                         + self.pools[2])
        return np.minimum(np.maximum(tot, self.p.Gmin), self.p.Gmax)

    def conductance(self):
        G = self._full() if self.read == "full" else self.G_nv.copy()
        return G.reshape(self.shape)

    def retained(self):
        G = np.minimum(np.maximum(self.G_nv + self.lock, self.p.Gmin),
                       self.p.Gmax)
        return G.reshape(self.shape)

    @property
    def clock(self):
        return float(self.t_dev.max()) if self.time_dynamic else 0.0

    def _flat(self, rows, cols):
        return (np.asarray(rows, np.int64) * self.shape[1]
                + np.asarray(cols, np.int64))

    def _idle(self, idx, t):
        late = self.t_dev[idx] < t
        if late.any():
            i = idx[late]
            self._step(i, t - self.t_dev[i], 0.0)
            self.t_dev[i] = t

    def advance(self, t, cols=None):
        if not self.time_dynamic:
            return
        if cols is None:
            idx = np.arange(self.t_dev.size)
        else:
            cols = np.asarray(cols, np.int64)
            idx = (np.arange(self.shape[0])[:, None] * self.shape[1]
                   + cols[None, :]).ravel()
        self._idle(idx, float(t))

    def read_at(self, t, cols=None):
        self.advance(t, cols)
        if cols is None:
            return self.conductance()
        cols = np.asarray(cols, np.int64)
        idx = (np.arange(self.shape[0])[:, None] * self.shape[1]
               + cols[None, :])
        if self.read == "nv":
            return self.G_nv[idx]
        tot = (self.G_nv[idx] + self.lock[idx]) + (
            (self.pools[0][idx] + self.pools[1][idx]) + self.pools[2][idx])
        return np.minimum(np.maximum(tot, self.p.Gmin), self.p.Gmax)

    def peek(self, t):
        """Read conductance of every device at time ``t`` WITHOUT changing
        the state: the closed form of one idle step from each device's clock
        (devices still in a pulse are read as they are)."""
        p = self.p
        dt = np.maximum(float(t) - self.t_dev, 0.0)
        x = p.x_eq + (self.x - p.x_eq) * np.exp(-dt / p.tau_ret)
        G_nv = self._G_of_x(x)
        if self.read == "nv":
            return G_nv.reshape(self.shape)
        taus = self._taus
        pools = ((self.pools[0] * np.exp(-dt / taus[0])
                  + self.pools[1] * np.exp(-dt / taus[1]))
                 + self.pools[2] * np.exp(-dt / taus[2]))
        tot = (G_nv + self.lock) + pools
        return np.minimum(np.maximum(tot, p.Gmin), p.Gmax).reshape(self.shape)

    # ---- writes ---------------------------------------------------------
    def program(self, rows, cols, amps, width, t=0.0):
        idx = self._flat(rows, cols)
        amps = np.asarray(amps, float)
        if not self.time_dynamic:
            self._step(idx, width, amps)
            return t + width * len(amps)
        t = float(t)
        self._idle(idx, t)                 # relax up to the pulse
        start = self.t_dev[idx]            # = max(t, busy until)
        self._step(idx, width, amps)
        self.t_dev[idx] = start + width
        return t + width

    def set_conductance(self, G):
        p = self.p
        self.reset()
        g = np.clip(np.asarray(G, float).ravel(), self.g_min, self.g_max)
        self.G_nv = g.copy()
        self.x = np.array([p.x_of_G(float(v)) for v in g])

    def state_dict(self):
        return {k: getattr(self, k).copy() for k in (
            "x", "G_nv", "lock", "pools", "trp", "trd", "prev", "cf", "t_dev")
                } | {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        if np.asarray(state["x"]).shape != (self.shape[0] * self.shape[1],):
            raise ValueError("Checkpoint architecture mismatch")
        for k in ("x", "G_nv", "lock", "pools", "trp", "trd", "cf", "t_dev"):
            setattr(self, k, np.array(state[k], float))
        self.prev = np.array(state["prev"], bool)
        self.rng.bit_generator.state = state["rng"]


def is_canonical_v3(dev):
    """True for an unmodified ``simulation_core.model_v3.EcfetV3`` twin."""
    t = type(dev)
    return t.__name__ == "EcfetV3" and t.__module__.endswith("model_v3")


def self_check(make_device, n_events=24, seed=0, rtol=1e-9):
    """Run the same random pulse / idle sequence on the scalar twin and on
    the array; True when every conductance agrees to ``rtol``. Guards the
    vectorised copy against an edited (hot-reloaded) model_v3."""
    from .scalar import ScalarTwinArray
    probe = make_device()
    vec = EcfetV3Array(probe.p, 2, 3, "physical", "full")
    ref = ScalarTwinArray(make_device, 2, 3, "physical", "full")
    rng = np.random.default_rng(seed)
    t = 0.0
    for _ in range(n_events):
        t += float(rng.choice([1e-4, 3e-3, 0.05, 0.7, 30.0]))
        k = int(rng.integers(1, 7))
        sel = rng.choice(6, k, replace=False)
        rows, cols = sel // 3, sel % 3
        amps = rng.choice([-1, 1], k) * rng.uniform(5e-12, 400e-12, k)
        vec.program(rows, cols, amps, 10e-3, t)
        ref.program(rows, cols, amps, 10e-3, t)
        if not np.allclose(vec.read_at(t + 5e-3), ref.read_at(t + 5e-3),
                           rtol=rtol, atol=0.0):
            return False
    return True
