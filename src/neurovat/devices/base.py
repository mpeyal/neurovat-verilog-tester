"""The device interface the network programs and reads.

A :class:`DeviceArray` is one crossbar: an ``n_post x n_pre`` grid of
independent synaptic devices held as arrays, so a whole presentation's worth
of writes is one vectorized call. The network only ever

* reads ``conductance()`` (retained weight, Siemens), and
* calls ``program(rows, cols, amps, width, t)`` with signed drive amplitudes.

Everything else (switching kinetics, bounds, retention, variation, write
noise) is the device's business. To add a FeCAP, MFMIS FeFET or FTJ model,
subclass this, implement the abstract methods and register it with
``neurovat.devices.register_device``.
"""
import numpy as np


class DeviceArray:
    name = "device"
    input_kind = "voltage"            # "voltage" (V) | "current" (A) drive

    #: conductance bounds (S) used to normalise weights to [0, 1]
    g_min = 0.0
    g_max = 1.0

    def __init__(self, n_post, n_pre):
        self.shape = (int(n_post), int(n_pre))

    # ---- required -------------------------------------------------------

    def conductance(self):
        """Retained conductance of every device, shape ``self.shape`` (S)."""
        raise NotImplementedError

    def program(self, rows, cols, amps, width, t=0.0):
        """Apply one pulse of signed amplitude ``amps[k]`` and duration
        ``width`` (s) to each device ``(rows[k], cols[k])``. Pairs are unique
        within a call. ``t`` is the device clock at the first pulse; returns
        the clock after the last one."""
        raise NotImplementedError

    def state_dict(self):
        """All dynamic state, as arrays/scalars ``neurovat.state.pack`` accepts."""
        raise NotImplementedError

    def load_state_dict(self, state):
        raise NotImplementedError

    def nominal(self):
        """A fresh 1x1 array with nominal parameters (no variation), used to
        probe the device without touching the crossbar."""
        raise NotImplementedError

    # ---- optional -------------------------------------------------------

    def set_conductance(self, G):
        """Place every device at conductance ``G`` (S). Devices whose state is
        not a function of conductance alone should raise ValueError."""
        raise ValueError(f"Cannot restore {type(self).__name__} conductance")

    def potentiation_sign(self, amp, width):
        """+1 if a positive drive raises conductance, else -1."""
        probe = self.nominal()
        g0 = float(probe.conductance()[0, 0])
        probe.program(np.array([0]), np.array([0]), np.array([abs(amp)]), width)
        return 1.0 if float(probe.conductance()[0, 0]) - g0 >= 0.0 else -1.0


def spread(nominal, sigma, shape, rng, floor=1e-3):
    """Device-to-device variation of one parameter.

    ``sigma`` is relative (fraction of nominal) for a non-zero nominal and
    absolute for a zero nominal. Non-zero parameters keep their sign: they
    are clipped at ``floor`` x nominal. Returns the scalar itself when
    ``sigma`` is 0, so an unvaried array stays cheap."""
    if not sigma:
        return nominal
    z = rng.standard_normal(shape)
    if nominal == 0:
        return sigma * z
    return nominal * np.maximum(1.0 + sigma * z, floor)
