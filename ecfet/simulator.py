"""Transient simulation engine.

Builds a time grid that lands exactly on every waveform edge, with fine
steps right after each edge (where dynamics are fast) growing geometrically
toward dt_max - similar to what Spectre's adaptive stepping gives you, but
deterministic and cheap.
"""

from dataclasses import dataclass, field
import numpy as np


def time_grid(breakpoints, t_stop, dt_min=1e-6, dt_max=None, grow=1.05):
    if dt_max is None:
        dt_max = min(max(t_stop / 2000.0, 1e-4), 0.05)
    bps = sorted({0.0, float(t_stop),
                  *(float(b) for b in breakpoints if 0.0 < b < t_stop)})
    ts = [0.0]
    for a, b in zip(bps[:-1], bps[1:]):
        seg = b - a
        dt = max(dt_min, min(dt_max / 10.0, seg / 100.0))
        t = a
        while t + dt < b - 1e-15:
            t += dt
            ts.append(t)
            dt = min(dt * grow, dt_max)
        ts.append(b)
    return np.asarray(ts)


@dataclass
class SimResult:
    label: str
    t: np.ndarray
    i_gate: np.ndarray
    R: np.ndarray
    G: np.ndarray
    extras: dict = field(default_factory=dict)
    waveform: object = None

    def at(self, times):
        """Interpolated (R, G) sampled at the given times."""
        times = np.asarray(times, dtype=float)
        return (np.interp(times, self.t, self.R),
                np.interp(times, self.t, self.G))

    def save_csv(self, path):
        """Dump t, i_gate, R, G (+extras) for comparison against Spectre."""
        cols = {"t_s": self.t, "i_gate_A": self.i_gate,
                "R_ohm": self.R, "G_S": self.G, **self.extras}
        header = ",".join(cols)
        data = np.column_stack(list(cols.values()))
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        return path

    def summary(self):
        return (f"[{self.label}] R: start {self.R[0]:.4g} ohm, "
                f"end {self.R[-1]:.4g} ohm, min {self.R.min():.4g}, "
                f"max {self.R.max():.4g} | points {len(self.t)}")


def simulate(model, waveform, t_stop, dt_min=1e-6, dt_max=None, label=None):
    """Run a transient: model is EcfetV1/EcfetV2, waveform a signals.Waveform."""
    ts = time_grid(waveform.breakpoints, t_stop, dt_min=dt_min, dt_max=dt_max)
    n = len(ts)

    model.reset()
    i_rec = np.empty(n)
    R_rec = np.empty(n)
    G_rec = np.empty(n)
    extra_keys = list(model.observables().keys())
    extras = {k: np.empty(n) for k in extra_keys}

    def record(k, i_now):
        i_rec[k] = i_now
        R_rec[k] = model.R
        G_rec[k] = model.G
        obs = model.observables()
        for key in extra_keys:
            extras[key][k] = obs[key]

    record(0, waveform.current(0.0))
    for k in range(n - 1):
        t, dt = ts[k], ts[k + 1] - ts[k]
        i_now = waveform.current(t + 0.5 * dt)   # constant within a segment
        model.step(t, dt, i_now)
        record(k + 1, i_now)

    return SimResult(label=label or model.name, t=ts, i_gate=i_rec,
                     R=R_rec, G=G_rec, extras=extras, waveform=waveform)
