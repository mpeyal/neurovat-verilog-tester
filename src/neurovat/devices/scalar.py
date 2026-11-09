"""Adapter: any scalar device twin (``step(t, dt, drive)`` + ``G``) as a
:class:`DeviceArray`. Each cross-point is its own twin instance, so behaviour
is exactly the twin's - at Python-loop speed. Use it to run a model before it
has a vectorized implementation, and as the reference a vectorized model is
tested against."""
import numpy as np

from .base import DeviceArray


def device_weight(dev):
    """Retained synaptic conductance of a device twin (S).  Prefers the
    nonvolatile component G_nv when the model exposes it (the clean stored
    weight), else the total conductance G."""
    obs = dev.observables() if hasattr(dev, "observables") else {}
    for key in ("G_nv (S)", "G_nv"):
        if key in obs:
            return float(obs[key])
    return float(dev.G)


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

    def __init__(self, make_device, n_post, n_pre):
        super().__init__(n_post, n_pre)
        self.make_device = make_device
        self.devices = [[make_device() for _ in range(self.shape[1])]
                        for _ in range(self.shape[0])]
        probe = make_device()
        self.input_kind = getattr(probe, "input_kind", "current")
        self.g_max = float(getattr(probe.p, "Gmax", None) or (1./probe.p.Rmin if hasattr(probe.p, "Rmin") else max(probe.G, 1e-9)))
        self.g_min = float(getattr(probe.p, "Gmin", None) or (1./probe.p.Rmax if hasattr(probe.p, "Rmax") else 0.0))
        self._G = None                # conductance cache; twins change only when stepped

    def nominal(self):
        return ScalarTwinArray(self.make_device, 1, 1)

    def potentiation_sign(self, amp, width):
        return calibrate_pot_sign(self.make_device, amp, width)

    def invalidate(self):
        """Drop the conductance cache after mutating ``devices`` directly."""
        self._G = None

    def conductance(self):
        if self._G is None:
            self._G = np.array([[device_weight(d) for d in row]
                                for row in self.devices])
        return self._G

    def program(self, rows, cols, amps, width, t=0.0):
        G = self.conductance()
        for j, i, a in zip(rows.tolist(), cols.tolist(), amps.tolist()):
            dev = self.devices[j][i]
            dev.step(t, width, a)
            t += width
            G[j, i] = device_weight(dev)
        return t

    def set_conductance(self, G):
        self._G = None
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
        return [[{k: v for k, v in vars(d).items() if k != "p"}
                 for d in row] for row in self.devices]

    def load_state_dict(self, state):
        if len(state) != self.shape[0] or any(len(row) != self.shape[1] for row in state):
            raise ValueError("Checkpoint architecture mismatch")
        for row, states in zip(self.devices, state):
            for device, values in zip(row, states):
                if set(values) != set(vars(device))-{"p"}:
                    raise ValueError("Checkpoint device state mismatch")
        for row, states in zip(self.devices, state):
            for device, values in zip(row, states):
                device.__dict__.update(values)
        self._G = None
