"""Adapter: any scalar device twin (``step(t, dt, drive)`` + ``G``) as a
:class:`DeviceArray`. Each cross-point is its own twin instance, so behaviour
is exactly the twin's - at Python-loop speed. Use it to run a model before it
has a vectorized implementation, and as the reference a vectorized model is
tested against.

Two timing modes:

* ``timing="serial"`` (default, the original contract): a twin is stepped
  ONLY while a write pulse is on it; the pulses of one ``program`` call are
  applied one after another (``t += width``); nothing happens between writes.
* ``timing="physical"``: every twin keeps its own clock. Before a pulse, and
  on ``advance(t)`` / ``read_at(t)``, the twin is stepped with zero drive
  from its clock to ``t`` in ONE ``step`` call (the twins integrate an idle
  interval exactly, so a long gap costs O(1)); volatile relaxation and
  retention therefore act between spikes. The pulses of one call start
  together at ``t`` (a crossbar writes its cells in parallel); a pulse that
  arrives while that device's previous pulse is still on starts when it
  ends.

Two read modes, the conductance ``conductance()`` / ``read_at`` return:

* ``read="nv"``   the non-volatile pool only (``G_nv`` when the twin exposes
  it, else ``G``) - the original read;
* ``read="full"`` the device's full conductance ``G`` (for ECFET v2/v3:
  ``G_nv`` (+ v3 ``stdp_lock``) + the volatile EDL pools, clamped to
  [Gmin, Gmax]) - what a read pulse sees.

``retained()`` is always the stored part (``G_retained`` / ``G_nv``), so
pulse sizing does not depend on the read mode.
"""
import copy

import numpy as np

from .base import DeviceArray

TIMINGS = ("serial", "physical")
READS = ("nv", "full")


def device_weight(dev):
    """Retained synaptic conductance of a device twin (S).  Prefers the
    nonvolatile component G_nv when the model exposes it (the clean stored
    weight), else the total conductance G."""
    obs = dev.observables() if hasattr(dev, "observables") else {}
    for key in ("G_nv (S)", "G_nv"):
        if key in obs:
            return float(obs[key])
    return float(dev.G)


def device_retained(dev):
    """Stored (non-volatile) conductance including any locked-in STDP part
    (v3 ``G_retained``), else :func:`device_weight`."""
    obs = dev.observables() if hasattr(dev, "observables") else {}
    if "G_retained (S)" in obs:
        p = dev.p
        return float(min(max(obs["G_retained (S)"], p.Gmin), p.Gmax))
    return device_weight(dev)


def calibrate_pot_sign(make_device, amp, width):
    """Discover which drive polarity POTENTIATES (raises conductance) for this
    device, by writing a probe pulse to a throwaway copy.  Returns +1 or -1."""
    dev = make_device()
    g0 = float(dev.G)
    dev.step(0.0, width, abs(amp))
    up = float(dev.G) - g0
    return 1.0 if up >= 0.0 else -1.0


class ScalarTwinArray(DeviceArray):
    name = "twin"

    def __init__(self, make_device, n_post, n_pre, timing="serial",
                 read="nv"):
        super().__init__(n_post, n_pre)
        if timing not in TIMINGS:
            raise ValueError(f"timing must be one of {TIMINGS}, not {timing!r}")
        if read not in READS:
            raise ValueError(f"read must be one of {READS}, not {read!r}")
        self.make_device = make_device
        self.timing, self.read = timing, read
        self.time_dynamic = timing == "physical"
        self.devices = [[make_device() for _ in range(self.shape[1])]
                        for _ in range(self.shape[0])]
        probe = make_device()
        self.input_kind = getattr(probe, "input_kind", "current")
        self.g_max = float(getattr(probe.p, "Gmax", None) or (1./probe.p.Rmin if hasattr(probe.p, "Rmin") else max(probe.G, 1e-9)))
        self.g_min = float(getattr(probe.p, "Gmin", None) or (1./probe.p.Rmax if hasattr(probe.p, "Rmax") else 0.0))
        self.t_dev = np.zeros(self.shape)  # per-device clock (physical timing)
        self._G = None                # read cache; twins change only when stepped

    def nominal(self):
        return ScalarTwinArray(self.make_device, 1, 1, self.timing, self.read)

    def potentiation_sign(self, amp, width):
        return calibrate_pot_sign(self.make_device, amp, width)

    def invalidate(self):
        """Drop the conductance cache after mutating ``devices`` directly."""
        self._G = None

    def _read_one(self, dev):
        return float(dev.G) if self.read == "full" else device_weight(dev)

    def conductance(self):
        if self._G is None:
            self._G = np.array([[self._read_one(d) for d in row]
                                for row in self.devices])
        return self._G

    def retained(self):
        return np.array([[device_retained(d) for d in row]
                         for row in self.devices])

    @property
    def clock(self):
        return float(self.t_dev.max()) if self.time_dynamic else 0.0

    def _idle(self, j, i, t):
        t0 = float(self.t_dev[j, i])
        if t > t0:
            self.devices[j][i].step(t0, t - t0, 0.0)
            self.t_dev[j, i] = t
            return True
        return False

    def advance(self, t, cols=None):
        if not self.time_dynamic:
            return
        t = float(t)
        G = self.conductance()
        cols = range(self.shape[1]) if cols is None else np.asarray(cols).tolist()
        for i in cols:
            for j in range(self.shape[0]):
                if self._idle(j, i, t):
                    G[j, i] = self._read_one(self.devices[j][i])

    def program(self, rows, cols, amps, width, t=0.0):
        G = self.conductance()
        if not self.time_dynamic:
            for j, i, a in zip(rows.tolist(), cols.tolist(), amps.tolist()):
                dev = self.devices[j][i]
                dev.step(t, width, a)
                t += width
                G[j, i] = self._read_one(dev)
            return t
        t = float(t)
        for j, i, a in zip(rows.tolist(), cols.tolist(), amps.tolist()):
            self._idle(j, i, t)                      # relax up to the pulse
            start = float(self.t_dev[j, i])          # = max(t, busy until)
            dev = self.devices[j][i]
            dev.step(start, width, a)
            self.t_dev[j, i] = start + width
            G[j, i] = self._read_one(dev)
        return t + width

    def set_conductance(self, G):
        self._G = None
        self.t_dev[:] = 0.0
        for j, row in enumerate(self.devices):
            for i, dev in enumerate(row):
                g = float(np.clip(G[j, i], self.g_min, self.g_max))
                dev.reset()
                if hasattr(dev, "P") and hasattr(dev.p, "dVth_max"):
                    x = np.clip((g-dev.p.Gmin)/(dev.p.Gmax-dev.p.Gmin), 1e-15, 1-1e-15)
                    vth = dev.p.Vread-dev.p.SS*np.log(x/(1-x))
                    dev.P = float(np.clip(2*(dev.p.Vth0-vth)/dev.p.dVth_max, -1., 1.))
                elif hasattr(dev, "G_nv"):
                    dev.G_nv = g
                    if hasattr(dev, "x") and hasattr(dev.p, "x_of_G"):
                        dev.x = dev.p.x_of_G(g)
                elif hasattr(dev, "M") and hasattr(dev.p, "Rdrift1c"):
                    dev.M = dev.M_before_diffusion = 1./g
                    dev.Vcap = dev.M/dev.p.Rdrift1c
                else:
                    try:
                        dev.G = g
                    except AttributeError as exc:
                        raise ValueError(f"Cannot restore {type(dev).__name__} conductance") from exc
                if not np.isclose(device_weight(dev), g, rtol=1e-8, atol=1e-12):
                    raise ValueError("Conductance is not reachable by this device; full state is required")

    def state_dict(self):
        rows = [[copy.deepcopy({k: v for k, v in vars(d).items() if k != "p"})
                 for d in row] for row in self.devices]
        if not self.time_dynamic:
            return rows
        return {"devices": rows, "t_dev": self.t_dev.copy()}

    def load_state_dict(self, state):
        t_dev = None
        if isinstance(state, dict):
            state, t_dev = state["devices"], np.asarray(state["t_dev"], float)
            if t_dev.shape != self.shape:
                raise ValueError("Checkpoint architecture mismatch")
        if len(state) != self.shape[0] or any(len(row) != self.shape[1] for row in state):
            raise ValueError("Checkpoint architecture mismatch")
        for row, states in zip(self.devices, state):
            for device, values in zip(row, states):
                if set(values) != set(vars(device))-{"p"}:
                    raise ValueError("Checkpoint device state mismatch")
        for row, states in zip(self.devices, state):
            for device, values in zip(row, states):
                device.__dict__.update(copy.deepcopy(values))
        if t_dev is not None:
            self.t_dev = t_dev.copy()
        self._G = None
