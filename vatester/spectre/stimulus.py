"""One piecewise-linear (PWL) stimulus shared by Spectre and the twin.

The same points become the `wave=[...]` of the Spectre source and the drive of
the twin, so both sides see exactly the same input.

    st = Stimulus.from_text("0 0  1e-9 0  1.05e-9 0.5  1.1e-8 0.5 ...", kind="voltage")
    st = Stimulus.from_file("pulses.csv")          # two columns: t, value
    st.value(t)                                    # linear interpolation
    st.grid(dt_max)                                # every corner + steps <= dt_max
    st.read_times(v_read)                          # middles of plateaus at v_read
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stimulus:
    t: np.ndarray
    v: np.ndarray
    kind: str = "voltage"            # "voltage" | "current"
    name: str = "stimulus"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.v = np.asarray(self.v, dtype=float)
        if self.kind not in ("voltage", "current"):
            raise ValueError("kind must be 'voltage' or 'current'")
        if self.t.ndim != 1 or self.t.shape != self.v.shape or len(self.t) < 2:
            raise ValueError("a PWL needs at least two (t, value) points")
        if not np.all(np.isfinite(self.t)) or not np.all(np.isfinite(self.v)):
            raise ValueError("PWL points must be finite numbers")
        if self.t[0] != 0.0:
            raise ValueError("a PWL must start at t = 0")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("PWL times must increase strictly")

    # ------------------------------------------------------------ construction
    @classmethod
    def from_text(cls, text, kind="voltage", name="stimulus"):
        """Whitespace/comma separated 't v t v ...' pairs."""
        nums = [float(x) for x in text.replace(",", " ").split()]
        if len(nums) % 2:
            raise ValueError("PWL text needs an even number of values (t v pairs)")
        a = np.array(nums).reshape(-1, 2)
        return cls(a[:, 0], a[:, 1], kind=kind, name=name)

    @classmethod
    def from_file(cls, path, kind="voltage", name=None):
        """Two-column file (t, value); '#' lines and a text header are skipped."""
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith(("#", ";", "//")):
                    continue
                parts = s.replace(",", " ").split()
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except (ValueError, IndexError):
                    continue                     # header line
        a = np.array(rows)
        if a.ndim != 2 or len(a) < 2:
            raise ValueError(f"{path}: no PWL points")
        import os
        return cls(a[:, 0], a[:, 1], kind=kind,
                   name=name or os.path.splitext(os.path.basename(path))[0])

    @classmethod
    def pulses(cls, amps, width, gap, edge=50e-12, kind="voltage", read=None,
               read_width=None, name="pulses"):
        """Rectangular pulses of the given amplitudes, each followed by an
        optional read plateau `read` (same width as `read_width` or `width`)."""
        t, v = [0.0], [0.0]
        now = gap
        rw = read_width or width

        def plateau(a, w):
            nonlocal now
            t.extend([now, now + edge, now + edge + w, now + 2 * edge + w])
            v.extend([0.0, a, a, 0.0])
            now = now + 2 * edge + w + gap

        for a in amps:
            plateau(a, width)
            if read is not None:
                plateau(read, rw)
        return cls(np.array(t), np.array(v), kind=kind, name=name)

    # ------------------------------------------------------------ use
    @property
    def t_stop(self):
        return float(self.t[-1])

    def value(self, t):
        return np.interp(t, self.t, self.v)

    def grid(self, dt_max):
        """Every PWL corner plus uniform steps of at most dt_max between them."""
        pts = [self.t[:1]]
        for a, b in zip(self.t[:-1], self.t[1:]):
            n = max(1, int(np.ceil((b - a) / dt_max - 1e-9)))
            pts.append(np.linspace(a, b, n + 1)[1:])
        return np.concatenate(pts)

    def read_times(self, level, tol=1e-12):
        """Middles of the plateaus held at `level` (excluding zero-length)."""
        out = []
        for k in range(len(self.t) - 1):
            if (abs(self.v[k] - level) <= tol and abs(self.v[k + 1] - level) <= tol
                    and self.t[k + 1] > self.t[k]):
                out.append(0.5 * (self.t[k] + self.t[k + 1]))
        return np.array(out)

    def spectre_wave(self, per_line=4):
        """'wave=[ t v t v \\ ... ]' body with 12 significant digits."""
        pairs = [f"{t:.12g} {v:.12g}" for t, v in zip(self.t, self.v)]
        lines = [" ".join(pairs[i:i + per_line]) for i in range(0, len(pairs), per_line)]
        return "[ " + " \\\n    ".join(lines) + " ]"
