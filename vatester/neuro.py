"""Neuromorphic network engine - spiking neurons trained on a memristive
crossbar whose synapses ARE the device twins (ECFET / FeFET / ...).

This is the simulation behind the GUI's *Neuromorphic Trainer* canvas.  It is
PURE (numpy + the device twins, no dearpygui, no threads, no GUI state) - the
app owns the worker thread and the widgets and just calls into here, mirroring
vatester/analysis.py.  Because it imports nothing from app.py it is also unit
testable on its own.

Picture
--------
        inputs (a gh x gw pixel grid, one spiking neuron per pixel)
          |   each input pixel i  --[ synapse device g_ij ]-->  output neuron j
          v
        crossbar  W  (n_out x n_in conductances, read from the device twins)
          |
          v
        output LIF neurons (leaky integrate-and-fire), winner-take-all
        lateral inhibition between them

Every cross-point (i, j) is an INDEPENDENT instance of the selected device
twin.  The synaptic weight is that device's retained conductance (G_nv if the
model exposes it, else G).  Learning = spike-timing-dependent plasticity: the
network rule decides WHEN and in WHICH DIRECTION to program a synapse, and the
*device model itself* decides HOW MUCH the conductance actually moves - so the
device's write nonlinearity, soft bounds, retention and cycle-to-cycle noise
are exactly what you watch evolve in the weight heatmap.  That is the whole
point: it tests how a given ECFET / FeFET trains as a synapse.

Two teaching modes:
  * supervised  - each pattern owns an output neuron; a teacher current forces
                  that neuron to fire, so its receptive field cleanly grows
                  into the pattern (reliable, great for a first look).
  * unsupervised- pure winner-take-all competition; neurons self-organize to
                  tile the pattern set (the classic Diehl & Cook demo).
"""

import math
from dataclasses import dataclass, field

import numpy as np


# ============================================================ parameters ====

@dataclass
class NeuronParams:
    """Leaky integrate-and-fire output neuron + post-synaptic potential tuning.
    Membrane potential is in arbitrary units with v_rest = 0 and threshold 1,
    so the gains below are what you actually tune."""
    tau_m: float = 20.0        # membrane time constant (ms)
    v_threshold: float = 1.0   # fire when v >= threshold (+ adaptive theta)
    v_reset: float = 0.0       # reset potential after a spike
    v_rest: float = 0.0        # leak target
    t_refractory: float = 5.0  # refractory period (ms)
    tau_syn: float = 8.0       # synaptic (PSP) trace decay (ms)

    epsp_gain: float = 11.0    # excitatory PSP weight (EPSP height); drive is
    #                            activity-normalised, so this is grid-independent
    ipsp_gain: float = 1.0     # inhibitory PSP weight (IPSP depth)
    inhibition: float = 0.9    # lateral winner-take-all strength (0 = none)

    theta_plus: float = 0.06   # homeostatic threshold bump per output spike
    tau_theta: float = 180.0   # adaptive-threshold decay (ms); 0 disables

    teacher: float = 1.4       # supervised teacher drive onto the target neuron


@dataclass
class STDPParams:
    """Pair-based STDP that PROGRAMS the synapse devices.

    On a post spike the recent-pre trace x_i potentiates synapse (i, j); on a
    pre spike the recent-post trace y_j depresses it.  The programming pulse
    amplitude scales with that trace, and the device twin integrates the pulse
    to produce the real ΔG (so dG is the device's, not an ideal additive step).
    Amplitudes are in SI (A for current-driven ECFET, V for FeFET); the app
    converts the panel's display units before handing them over.
    """
    a_plus: float = 1.0        # potentiation learning rate (scales pulse amp)
    a_minus: float = 1.0       # (heterosynaptic) depression learning rate
    tau_pre: float = 20.0      # pre eligibility-trace decay (ms)
    offset: float = 0.25       # pre-trace split: above -> LTP, below -> LTD
    pot_amp: float = 200e-12   # SI amplitude of a full potentiating pulse
    dep_amp: float = 200e-12   # SI amplitude of a full depressing pulse
    pulse_width: float = 10e-3 # programming pulse width (s)
    pot_sign: float = 0.0      # +1/-1 drive polarity that potentiates;
    #                            0 = auto-calibrate from the device


@dataclass
class NetConfig:
    grid_h: int = 5
    grid_w: int = 5
    n_out: int = 4
    mode: str = "supervised"   # "supervised" | "unsupervised"
    present_ms: float = 120.0  # duration of one pattern presentation
    dt_ms: float = 1.0         # integration step
    max_rate_hz: float = 160.0 # peak input spike rate for a fully-on pixel
    seed: int = 1
    pattern_set: str = "bars"  # which built-in pattern bank to train on


# ================================================================ patterns ==

def make_patterns(name, gh, gw):
    """Built-in input pattern bank rendered onto a gh x gw pixel grid.
    Returns [(label, vec01)] with vec01 a flat (gh*gw,) array in [0, 1]."""
    pats = []

    def blank():
        return np.zeros((gh, gw), dtype=float)

    if name == "bars":
        # vertical, horizontal and the two diagonals - the classic
        # orientation-selectivity stimulus set
        for c in (0, gw // 2, gw - 1):
            g = blank(); g[:, c] = 1.0
            pats.append((f"| col {c}", g))
        for r in (0, gh // 2, gh - 1):
            g = blank(); g[r, :] = 1.0
            pats.append((f"- row {r}", g))
        g = blank()
        for k in range(min(gh, gw)):
            g[k, k] = 1.0
        pats.append(("\\ diag", g))
        g = blank()
        for k in range(min(gh, gw)):
            g[k, gw - 1 - k] = 1.0
        pats.append(("/ diag", g))
    elif name == "letters":
        glyphs = _letter_glyphs()
        for ch, grid in glyphs:
            pats.append((ch, _fit(grid, gh, gw)))
    elif name == "digits":
        for d, grid in _digit_glyphs():
            pats.append((d, _fit(grid, gh, gw)))
    else:  # "random"
        rng = np.random.default_rng(0)
        for k in range(6):
            g = (rng.random((gh, gw)) > 0.6).astype(float)
            pats.append((f"rand {k}", g))

    return [(lbl, g.reshape(-1)) for lbl, g in pats]


def _fit(grid, gh, gw):
    """Nearest-neighbour fit of a bitmap to gh x gw."""
    grid = np.asarray(grid, dtype=float)
    h, w = grid.shape
    rs = (np.linspace(0, h - 1, gh)).round().astype(int)
    cs = (np.linspace(0, w - 1, gw)).round().astype(int)
    return grid[np.ix_(rs, cs)]


def _letter_glyphs():
    L = ["10000", "10000", "10000", "10000", "11111"]
    T = ["11111", "00100", "00100", "00100", "00100"]
    X = ["10001", "01010", "00100", "01010", "10001"]
    O = ["01110", "10001", "10001", "10001", "01110"]
    return [(ch, _bits(rows)) for ch, rows in
            (("L", L), ("T", T), ("X", X), ("O", O))]


def _digit_glyphs():
    one = ["00100", "01100", "00100", "00100", "01110"]
    seven = ["11111", "00010", "00100", "01000", "01000"]
    three = ["11110", "00011", "01110", "00011", "11110"]
    zero = ["01110", "10011", "10101", "11001", "01110"]
    return [(ch, _bits(rows)) for ch, rows in
            (("0", zero), ("1", one), ("3", three), ("7", seven))]


def _bits(rows):
    return np.array([[1.0 if c == "1" else 0.0 for c in r] for r in rows])


# ============================================================ the trainer ===

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


class Trainer:
    """One crossbar + output layer.  Build once, then call present() per
    stimulus; weights_uS() reads the live device conductances."""

    def __init__(self, make_device, input_kind, neuron, stdp, cfg,
                 patterns=None, targets=None):
        self.make_device = make_device
        self.input_kind = input_kind          # "current" | "voltage"
        self.N = neuron
        self.S = stdp
        self.cfg = cfg
        self.n_in = cfg.grid_h * cfg.grid_w
        self.n_out = cfg.n_out
        self.rng = np.random.default_rng(cfg.seed)
        self._t_dev = 0.0                     # shared device wall-clock (s)

        # one independent device per cross-point
        self.syn = [[make_device() for _ in range(self.n_in)]
                    for _ in range(self.n_out)]
        probe = make_device()
        self.g_max = float(getattr(probe.p, "Gmax", None) or max(probe.G, 1e-9))
        self.g_min = float(getattr(probe.p, "Gmin", 0.0))
        self.pot_sign = (stdp.pot_sign if stdp.pot_sign else
                         calibrate_pot_sign(make_device, stdp.pot_amp,
                                            stdp.pulse_width))

        # explicit patterns/targets (custom editor, MNIST/dataset) override the
        # built-in bank; otherwise synthesise the chosen built-in set
        self.patterns = (patterns if patterns is not None else
                         make_patterns(cfg.pattern_set, cfg.grid_h, cfg.grid_w))
        # each pattern's "preferred" output neuron: the supervised teacher
        # target / the label we score against.  For a dataset these are the
        # class indices; otherwise round-robin over the output neurons.
        self.target_of = (list(targets) if targets is not None else
                          [k % self.n_out for k in range(len(self.patterns))])

        self.theta = np.zeros(self.n_out)     # adaptive thresholds (persist)
        self.spike_count = np.zeros(self.n_out)
        self.learn_cap = 12.0                 # max per-presentation write gain

    # ---- weights ----------------------------------------------------------

    def weights_S(self):
        return np.array([[device_weight(d) for d in row] for row in self.syn])

    def weights_uS(self):
        return self.weights_S() * 1e6

    def weights_norm(self):
        """Conductance normalised to [0, 1] over the device range (the synaptic
        efficacy used to drive the neurons - absolute Siemens cancel out)."""
        span = (self.g_max - self.g_min) or 1.0
        return np.clip((self.weights_S() - self.g_min) / span, 0.0, 1.0)

    # ---- device programming ----------------------------------------------

    def _program(self, j, i, potentiate, strength, gain=1.0):
        """Apply one programming pulse to synapse (i->j).  `gain` scales the
        pulse charge to consolidate a whole presentation's worth of writes into
        one device.step - the device sub-steps internally, so a charge-scaled
        pulse honours the same soft-bound nonlinearity as many small ones, but
        is far cheaper (essential for big grids / datasets)."""
        strength = float(min(max(strength, 0.0), 1.0)) * gain
        if strength <= 1e-3:
            return
        amp = (self.S.pot_amp if potentiate else self.S.dep_amp) * strength
        direction = self.pot_sign if potentiate else -self.pot_sign
        self.syn[j][i].step(self._t_dev, self.S.pulse_width, direction * amp)
        self._t_dev += self.S.pulse_width

    def _consolidate(self, acc, npost):
        """Competitive plasticity, applied ONCE per presentation per firing
        neuron.  For neuron j, mean_pre[i] is the average pre-trace seen at j's
        post spikes: inputs that were active just before j fired (mean_pre >=
        offset) are POTENTIATED, quiet ones are heterosynaptically DEPRESSED -
        the rule that carves a clean receptive field.  The write charge scales
        with how often j fired (more coincidence -> more learning).  The ACTUAL
        dG is whatever the device twin yields - its nonlinearity / bounds /
        retention are exactly what you watch evolve."""
        S = self.S
        off = max(S.offset, 1e-6)
        for j in range(self.n_out):
            if npost[j] < 1:
                continue
            mean_pre = acc[j] / npost[j]
            gain = min(self.learn_cap, float(npost[j]))
            for i in range(self.n_in):
                p = mean_pre[i]
                if p >= off:
                    self._program(j, i, True,
                                  S.a_plus * (p - off) / (1.0 - off), gain)
                elif S.a_minus > 0:
                    self._program(j, i, False,
                                  S.a_minus * (off - p) / off, gain)

    # ---- one stimulus presentation ---------------------------------------

    def present(self, vec01, target=None, learn=True):
        """Run the network for one presentation of input intensities vec01.

        Returns a dict with the spike raster, output membrane traces and which
        neuron won - everything the GUI animates.  When learn=True the synapse
        devices are programmed by the STDP events that occur."""
        cfg, N, S = self.cfg, self.N, self.S
        dt = cfg.dt_ms
        n_steps = max(1, int(round(cfg.present_ms / dt)))
        n_in, n_out = self.n_in, self.n_out

        # frozen weights for the duration of this presentation (efficacy in
        # [0,1]); devices are programmed in place and re-read after the loop
        W = self.weights_norm()

        # Poisson input spikes: rate proportional to pixel intensity
        rates = np.clip(np.asarray(vec01, float), 0, 1) * cfg.max_rate_hz
        p_spike = 1.0 - np.exp(-rates * dt * 1e-3)
        S_in = self.rng.random((n_steps, n_in)) < p_spike  # (T, n_in) bool

        # decay factors
        d_syn = math.exp(-dt / N.tau_syn)
        d_pre = math.exp(-dt / S.tau_pre)
        d_theta = math.exp(-dt / N.tau_theta) if N.tau_theta > 0 else 0.0
        leak = dt / max(N.tau_m, 1e-6)

        x = np.zeros(n_in)        # synaptic (PSP) trace
        pre = np.zeros(n_in)      # STDP pre eligibility trace
        v = np.full(n_out, N.v_rest, dtype=float)
        refrac = np.zeros(n_out)  # remaining refractory time (ms)
        acc = np.zeros((n_out, n_in)) if learn else None  # plasticity accumulator
        npost = np.zeros(n_out)                            # post spikes per neuron

        v_trace = np.zeros((n_steps, n_out))
        out_spikes = [[] for _ in range(n_out)]   # spike times (ms) per neuron
        in_spikes = []                              # (t_ms, input_index)

        # split inputs into excitatory / inhibitory channels: EPSP adds,
        # IPSP subtracts (feed-forward inhibition).  Last ~fraction are inhib.
        inh = np.zeros(n_in)
        if N.ipsp_gain > 0:
            n_inh = n_in // 6
            if n_inh:
                inh[self.rng.choice(n_in, n_inh, replace=False)] = 1.0
        exc_mask = 1.0 - inh

        for t in range(n_steps):
            tm = t * dt
            fired_in = np.nonzero(S_in[t])[0]
            if fired_in.size:
                x[fired_in] = 1.0
                pre[fired_in] = 1.0
                for i in fired_in:
                    in_spikes.append((tm, int(i)))

            # synaptic drive: EPSP from excitatory inputs, IPSP from inhibitory.
            # Normalised by the total presynaptic activity so the drive is the
            # weight/input ALIGNMENT (a cosine-like match in [0, gain]) rather
            # than a raw sum - this makes one epsp_gain work across any grid
            # size / pattern density (5x5 bars or 28x28 MNIST alike).
            eff = W * x                                 # (n_out, n_in)
            norm = x.sum() + 1.0
            drive = (N.epsp_gain * (eff * exc_mask).sum(1)
                     - N.ipsp_gain * (eff * inh).sum(1)) / norm
            if cfg.mode == "supervised" and target is not None:
                drive[target] += N.teacher

            active = refrac <= 0
            v[active] += leak * (N.v_rest - v[active] + drive[active])
            v[~active] = N.v_reset
            refrac[~active] -= dt
            v_trace[t] = v

            fired = np.nonzero((v >= (N.v_threshold + self.theta)) & active)[0]
            if fired.size:
                # winner-take-all: the earliest/strongest suppresses the rest
                if N.inhibition > 0 and n_out > 1:
                    winner = fired[int(np.argmax(v[fired]))]
                    others = np.ones(n_out, bool); others[winner] = False
                    v[others] -= N.inhibition
                    fired = np.array([winner])
                for j in fired:
                    out_spikes[j].append(tm)
                    self.spike_count[j] += 1
                    v[j] = N.v_reset
                    refrac[j] = N.t_refractory
                    self.theta[j] += N.theta_plus
                    if learn:                    # accrue, program once at the end
                        acc[j] += pre
                        npost[j] += 1

            # relax traces (exact exponential)
            x *= d_syn
            pre *= d_pre
            if N.tau_theta > 0:
                self.theta *= d_theta

        if learn:
            self._consolidate(acc, npost)
        self._t_dev += cfg.present_ms * 1e-3
        winner = int(np.argmax([len(s) for s in out_spikes])) \
            if any(out_spikes) else -1
        return {
            "v_trace": v_trace, "dt_ms": dt, "n_steps": n_steps,
            "out_spikes": out_spikes, "in_spikes": in_spikes,
            "winner": winner, "n_out_spikes": [len(s) for s in out_spikes],
        }

    # ---- scoring ----------------------------------------------------------

    def classify(self, learn=False, max_n=None):
        """Present patterns once (no learning) and report the winning output
        neuron + spike counts.  Used by the 'Test' button.  The homeostatic
        thresholds built up during training are set aside for the test
        (free-run recognition is about the receptive fields, not the
        accumulated theta) and restored afterwards.  For a big set (a dataset)
        pass max_n to score a random balanced-ish sample for responsiveness."""
        items = list(enumerate(self.patterns))
        if max_n and len(items) > max_n:
            sel = sorted(np.random.default_rng(0).choice(
                len(items), max_n, replace=False).tolist())
            items = [items[i] for i in sel]
        theta_save = self.theta.copy()
        self.theta = np.zeros(self.n_out)
        try:
            rows = []
            for k, (lbl, vec) in items:
                res = self.present(vec, target=None, learn=learn)
                rows.append({"label": lbl, "winner": res["winner"],
                             "counts": res["n_out_spikes"],
                             "target": self.target_of[k]})
            return rows
        finally:
            self.theta = theta_save
