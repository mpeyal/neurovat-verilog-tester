"""Gate-current waveform construction (piecewise-constant pulse trains).

Currents are in amperes, times in seconds.  A waveform is a sum of
rectangular pulses on a zero baseline; overlapping pulses add.
"""

from bisect import bisect_right


class Waveform:
    def __init__(self, pulses=None):
        """pulses: iterable of (t_start, width, amplitude_A)."""
        self.pulses = list(pulses) if pulses else []
        self._compile()

    def _compile(self):
        events = {}
        for t0, width, amp in self.pulses:
            if width <= 0:
                raise ValueError(f"pulse width must be > 0, got {width}")
            events[t0] = events.get(t0, 0.0) + amp
            events[t0 + width] = events.get(t0 + width, 0.0) - amp
        self.edges = sorted(events)
        self.values = []
        acc = 0.0
        for e in self.edges:
            acc += events[e]
            self.values.append(acc)

    def current(self, t):
        """Gate current at time t (A)."""
        idx = bisect_right(self.edges, t) - 1
        if idx < 0:
            return 0.0
        return self.values[idx]

    @property
    def breakpoints(self):
        """Times where the waveform is discontinuous (simulator must land on these)."""
        return list(self.edges)

    def __add__(self, other):
        return Waveform(self.pulses + other.pulses)

    # ---- builders -------------------------------------------------------

    @classmethod
    def pulse_train(cls, amp, width, period, n, t_start=10e-3):
        """n identical rectangular pulses.  amp in A (signed), width/period in s."""
        if period < width:
            raise ValueError("period must be >= width")
        return cls([(t_start + k * period, width, amp) for k in range(n)])

    @classmethod
    def ltp_ltd(cls, amp, width, period, n_each, t_start=10e-3, gap=0.0):
        """n_each pulses of -|amp| (conductance up / R down) followed by
        n_each pulses of +|amp| (conductance down / R up), matching the
        Verilog-A convention where positive gate current raises R."""
        a = abs(amp)
        up = cls.pulse_train(-a, width, period, n_each, t_start)
        down = cls.pulse_train(+a, width, period, n_each,
                               t_start + n_each * period + gap)
        return up + down

    def pulse_windows(self):
        """[(t_start, t_end), ...] of each constituent pulse, time-sorted."""
        return sorted((t0, t0 + w) for t0, w, _ in self.pulses)
