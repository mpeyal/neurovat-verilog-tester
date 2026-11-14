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
interval is one exact-exponential step (O(1) whatever its length). In
physical time a pulse is scheduled, not applied, by ``program``: it is
integrated when the device is advanced or read past its start, up to that
time (a read inside a pulse sees the state at the read time, the pulse
split there), with the same queue semantics as the scalar adapter.

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
        self.t_dev = np.zeros(n)      # time each device has been integrated to
        # physical time: scheduled pulses not yet (fully) integrated, a FIFO
        # per device in slots 0..q_n-1 (start, end, amplitude), and the time
        # each device is scheduled until (end of its last pulse)
        self.q_s = np.zeros((1, n))
        self.q_e = np.zeros((1, n))
        self.q_a = np.zeros((1, n))
        self.q_n = np.zeros(n, np.int64)
        self.busy = np.zeros(n)
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

        # (C) volatile pools relax (idle devices; a write integrates them in E)
        wr = Ieff != 0.0
        if not wr.all():
            q, dq = idx[~wr], dt[~wr]
            for i in range(3):
                self.pools[i, q] *= np.exp(-dq / taus[i])

        # (D) retention: x relaxes toward x_eq
        self.x[idx] = p.x_eq + (self.x[idx] - p.x_eq) * np.exp(-dt / p.tau_ret)
        self.G_nv[idx] = self._G_of_x(self.x[idx])

        # (E) conserved-x write, sub-stepped per device; the pools decay and
        # take the EDL inflow together (exponential integrator, see model_v3)
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
            z = np.stack([h / tau for tau in taus])           # (3, n_w)
            eh = np.exp(-z)
            with np.errstate(invalid="ignore", divide="ignore"):
                phi = np.where(z > 0.0, -np.expm1(-z) / z, 1.0)
            for s in range(int(n_sub.max())):
                act = s < n_sub
                a = w[act]
                dxa = dx_sub[act]
                x_try = self.x[a] + dxa
                x_new = np.minimum(np.maximum(x_try, 0.0), 1.0)
                G_new = self._G_of_x(x_new)
                dG = G_new - self.G_nv[a]
                wgt = phi[:, act]
                rail = x_new != x_try
                if rail.any():                 # rail hit inside the sub-step
                    wgt = wgt.copy()
                    xo, dr = self.x[a][rail], dxa[rail]
                    f = np.where(dr != 0.0, np.abs(x_new[rail] - xo)
                                 / np.abs(np.where(dr != 0.0, dr, 1.0)), 0.0)
                    zr = z[:, act][:, rail]
                    fz = f * zr
                    with np.errstate(invalid="ignore", divide="ignore"):
                        wr_ = np.where(fz > 0.0, np.exp(-(1.0 - f) * zr)
                                       * -np.expm1(-fz) / fz,
                                       np.where(f > 0.0, 1.0, 0.0))
                    wgt[:, rail] = wr_
                self.x[a] = x_new
                self.G_nv[a] = G_new
                dG_edl = np.copysign(edl * np.abs(dG), dG)
                e_a = eh[:, act]
                for i in range(3):
                    self.pools[i, a] = (self.pools[i, a] * e_a[i]
                                        + cw[i] * dG_edl * wgt[i])

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
        """Device time: the end of the last scheduled pulse or the latest
        time a device was advanced to, whichever is later."""
        if not self.time_dynamic:
            return 0.0
        return float(max(self.t_dev.max(), self.busy.max()))

    def _flat(self, rows, cols):
        return (np.asarray(rows, np.int64) * self.shape[1]
                + np.asarray(cols, np.int64))

    def _pulses_to(self, t, idx=None):
        """Integrate the scheduled pulses of devices ``idx`` (default all)
        that start before ``t``, up to ``t``: the idle gap to each pulse
        start is one idle step, the pulse is driven up to min(end, t). A
        device inside a pulse at ``t`` is left at ``t_dev == t``."""
        while True:
            pend = self.q_n > 0 if idx is None else None
            cand = (np.nonzero(pend & (self.q_s[0] < t))[0] if idx is None
                    else idx[(self.q_n[idx] > 0) & (self.q_s[0, idx] < t)])
            if not cand.size:
                return
            s = self.q_s[0, cand]
            late = self.t_dev[cand] < s
            if late.any():                       # idle up to the pulse start
                i = cand[late]
                self._step(i, s[late] - self.t_dev[i], 0.0)
                self.t_dev[i] = s[late]
            te = np.minimum(self.q_e[0, cand], t)
            go = te > self.t_dev[cand]
            if go.any():                         # drive up to min(end, t)
                i = cand[go]
                self._step(i, te[go] - self.t_dev[i], self.q_a[0, i])
                self.t_dev[i] = te[go]
            done = cand[self.t_dev[cand] >= self.q_e[0, cand]]
            if not done.size:
                return
            for q in (self.q_s, self.q_e, self.q_a):   # pop slot 0
                q[:-1, done] = q[1:, done]
                q[-1, done] = 0.0
            self.q_n[done] -= 1

    def _idle(self, idx, t):
        self._pulses_to(t, idx)
        late = self.t_dev[idx] < t
        if late.any():
            i = idx[late]
            self._step(i, t - self.t_dev[i], 0.0)
            self.t_dev[i] = t

    def _schedule(self, idx, amps, width, t):
        start = np.maximum(np.maximum(t, self.busy[idx]), self.t_dev[idx])
        slot = self.q_n[idx]
        k = int(slot.max()) + 1 if slot.size else 1
        if k > self.q_s.shape[0]:                # grow the queue
            pad = np.zeros((k - self.q_s.shape[0], self.q_s.shape[1]))
            self.q_s, self.q_e, self.q_a = (np.vstack([q, pad]) for q in (
                self.q_s, self.q_e, self.q_a))
        self.q_s[slot, idx] = start
        self.q_e[slot, idx] = start + width
        self.q_a[slot, idx] = amps
        self.q_n[idx] += 1
        self.busy[idx] = start + width

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
        """Read conductance of every device at time ``t``. Scheduled pulses
        that started before ``t`` are integrated up to ``t`` (a device in a
        pulse is read at ``t``, mid-pulse); idle devices are not stepped:
        their read is the closed form of one idle step from their clock."""
        p = self.p
        t = float(t)
        if self.time_dynamic:
            self._pulses_to(t)
        dt = np.maximum(t - self.t_dev, 0.0)
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
        self._idle(idx, t)                 # relax up to the call time
        # scheduled at max(t, busy until); integrated by advance / read / peek
        self._schedule(idx, amps, width, t)
        return t + width

    def set_conductance(self, G):
        p = self.p
        self.reset()
        g = np.clip(np.asarray(G, float).ravel(), self.g_min, self.g_max)
        self.G_nv = g.copy()
        self.x = np.array([p.x_of_G(float(v)) for v in g])

    _STATE = ("x", "G_nv", "lock", "pools", "trp", "trd", "cf", "t_dev",
              "q_s", "q_e", "q_a", "busy")

    def state_dict(self):
        return {k: getattr(self, k).copy() for k in self._STATE + (
            "prev", "q_n")} | {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        n = self.shape[0] * self.shape[1]
        if np.asarray(state["x"]).shape != (n,):
            raise ValueError("Checkpoint architecture mismatch")
        if "q_n" not in state:             # checkpoint from before the queue
            state = dict(state, q_s=np.zeros((1, n)), q_e=np.zeros((1, n)),
                         q_a=np.zeros((1, n)), q_n=np.zeros(n, np.int64),
                         busy=np.array(state["t_dev"], float))
        for k in self._STATE:
            setattr(self, k, np.array(state[k], float))
        self.prev = np.array(state["prev"], bool)
        self.q_n = np.array(state["q_n"], np.int64)
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
