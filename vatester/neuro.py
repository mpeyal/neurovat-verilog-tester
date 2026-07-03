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
    v_spike_peak: float = 1.6  # value drawn into the membrane trace ON the step a
    #                            neuron fires, so the spike shows AT the threshold
    #                            crossing (the reset + refractory floor then follow)
    tau_syn: float = 8.0       # synaptic (PSP) trace decay (ms)

    epsp_gain: float = 11.0    # excitatory PSP weight (EPSP height); drive is
    #                            activity-normalised, so this is grid-independent
    ipsp_gain: float = 1.0     # inhibitory PSP weight (IPSP depth)
    inhibition: float = 0.9    # lateral winner-take-all strength (0 = none)

    theta_plus: float = 0.06   # homeostatic threshold bump per output spike
    tau_theta: float = 180.0   # adaptive-threshold decay (ms); 0 disables

    teacher: float = 1.4       # supervised teacher drive onto the target neuron
    v_noise: float = 0.0       # membrane (thermal/neuronal) noise sigma per step
    hidden_gain: float = 1.7   # extra drive on HIDDEN layers (they have no
    #                            teacher, so they need a boost to fire on their
    #                            own and pass a code to the next crossbar)


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
    pot_amp: float = 50e-12    # SI amplitude of a full potentiating pulse
    dep_amp: float = 50e-12    # SI amplitude of a full depressing pulse
    pulse_width: float = 10e-3 # programming pulse width (s)
    pot_sign: float = 0.0      # +1/-1 drive polarity that potentiates;
    #                            0 = auto-calibrate from the device
    sg_lr: float = 0.04        # surrogate-gradient learning rate (BPTT step)
    sg_beta: float = 5.0       # surrogate steepness (fast-sigmoid derivative)


@dataclass
class NetConfig:
    grid_h: int = 5
    grid_w: int = 5
    n_out: int = 4
    hidden_layers: tuple = ()  # sizes of HIDDEN LIF layers between input and
    #                            output, e.g. (8,) or (8, 4); each consecutive
    #                            pair of layers is joined by its own crossbar
    mode: str = "supervised"   # "supervised" | "unsupervised"
    learn_rule: str = "stdp"   # "stdp" (device-local) | "surrogate" (BPTT,
    #                            supervised, gradient applied through the device)
    present_ms: float = 120.0  # duration of one pattern presentation
    dt_ms: float = 1.0         # integration step
    max_rate_hz: float = 160.0 # peak input spike rate for a fully-on pixel
    seed: int = 1
    pattern_set: str = "bars"  # which built-in pattern bank to train on

    # ---- spike encoding + input noise (front-end "sensor" realism) ----
    encoding: str = "rate"     # "rate" (Poisson) | "latency" (time-to-first
    #                            spike: brighter pixel fires earlier)
    bg_rate_hz: float = 0.0    # spontaneous background rate on EVERY input (Hz)
    input_noise: float = 0.0   # per-presentation pixel noise (Gaussian + salt
    #                            & pepper), as a fraction in [0, 1]
    signal_frac: float = 1.0   # fraction of afferents carrying the REAL pattern;
    #                            the rest fire only background noise (a FIXED
    #                            random subset, the embedded-pattern paradigm)
    jitter_ms: float = 0.0     # Gaussian temporal jitter (sigma) on every spike
    pattern_ms: float = 0.0    # how long the PATTERN is shown within present_ms
    #                            (centred); the rest of the window is background-
    #                            noise only. 0 or >= present_ms = full window.


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

def confusion(y_true, y_pred, n):
    """n x n confusion matrix (rows = true class, cols = predicted)."""
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            cm[int(t), int(p)] += 1
    return cm


def prf1(cm):
    """Per-class precision/recall/F1 + support and the overall accuracy,
    macro-F1 and support-weighted F1 from a confusion matrix."""
    cm = np.asarray(cm, float)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / np.maximum(tp + fn, 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    support = cm.sum(1)
    tot = max(cm.sum(), 1.0)
    return {
        "prec": prec, "rec": rec, "f1": f1, "support": support.astype(int),
        "acc": float(tp.sum() / tot),
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1.0)),
    }


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


def probe_lif(N, drive, present_ms, dt):
    """Simulate ONE leaky integrate-and-fire output neuron driven by a CONSTANT
    `drive` for present_ms (no crossbar, no weights) - exactly the network's
    output-neuron dynamics (leak toward v_rest+drive, fire at threshold + the
    adaptive homeostatic theta, reset, refractory).  Returns
    (t_ms, V, threshold_trace, spike_times_ms) so the GUI can plot the sawtooth.
    """
    n = max(1, int(round(present_ms / dt)))
    leak = dt / max(N.tau_m, 1e-6)
    d_theta = math.exp(-dt / N.tau_theta) if N.tau_theta > 0 else 0.0
    v, theta, refr = N.v_rest, 0.0, 0.0
    t = np.arange(n) * dt
    V = np.empty(n)
    TH = np.empty(n)
    spikes = []
    rng = np.random.default_rng(12345)
    for k in range(n):
        active = refr <= 0
        if active:
            v += leak * (N.v_rest - v + drive)
            if N.v_noise > 0:
                v += rng.normal(0.0, N.v_noise)
        else:
            v = N.v_reset
            refr -= dt
        if N.tau_theta > 0:
            theta *= d_theta
        V[k] = v
        TH[k] = N.v_threshold + theta
        if active and v >= N.v_threshold + theta:
            spikes.append(k * dt)
            v = N.v_reset
            refr = N.t_refractory
            theta += N.theta_plus
    return t, V, TH, spikes


def fi_curve(N, present_ms, dt, drive_max, n_pts=26):
    """Firing rate (Hz) vs constant drive - the neuron's f-I transfer curve."""
    drives = np.linspace(0.0, drive_max, n_pts)
    rates = np.array([len(probe_lif(N, float(d), present_ms, dt)[3])
                      / (present_ms * 1e-3) for d in drives])
    return drives, rates


def probe_lif_psp(N, in_hz, epsp_w, present_ms, dt):
    """Drive ONE LIF neuron with a regular EXCITATORY spike train (one input
    spike every 1000/in_hz ms).  Each input spike adds `epsp_w` to a synaptic
    PSP trace that decays with tau_syn; the membrane leaks toward that PSP - so
    you SEE each EPSP bump, the leak between them, sub-threshold SUMMATION, and
    the threshold crossing that finally triggers a spike + reset.  Returns
    (t_ms, V, threshold_trace, out_spike_times, in_spike_times, psp_trace)."""
    n = max(1, int(round(present_ms / dt)))
    leak = dt / max(N.tau_m, 1e-6)
    d_syn = math.exp(-dt / max(N.tau_syn, 1e-6))
    d_theta = math.exp(-dt / N.tau_theta) if N.tau_theta > 0 else 0.0
    period = max(1, int(round((1000.0 / max(in_hz, 1e-3)) / dt)))
    v, theta, refr, x = N.v_rest, 0.0, 0.0, 0.0
    t = np.arange(n) * dt
    V = np.empty(n)
    TH = np.empty(n)
    P = np.empty(n)
    outs, ins = [], []
    rng = np.random.default_rng(7)
    for k in range(n):
        x *= d_syn
        if k % period == 0:                # an EPSP arrives
            x += epsp_w
            ins.append(k * dt)
        active = refr <= 0
        if active:
            v += leak * (N.v_rest - v + x)
            if N.v_noise > 0:
                v += rng.normal(0.0, N.v_noise)
        else:
            v = N.v_reset
            refr -= dt
        if N.tau_theta > 0:
            theta *= d_theta
        V[k] = v
        TH[k] = N.v_threshold + theta
        P[k] = x
        if active and v >= N.v_threshold + theta:
            outs.append(k * dt)
            v = N.v_reset
            refr = N.t_refractory
            theta += N.theta_plus
    return t, V, TH, outs, ins, P


def fi_curve_psp(N, epsp_w, present_ms, dt, hz_max, n_pts=24):
    """Output firing rate (Hz) vs INPUT spike rate - the spike transfer curve."""
    hz = np.linspace(1.0, max(hz_max, 2.0), n_pts)
    out = np.array([len(probe_lif_psp(N, float(h), epsp_w, present_ms, dt)[3])
                    / (present_ms * 1e-3) for h in hz])
    return hz, out


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

        # layer stack: [n_in, hidden..., n_out].  Consecutive layers are joined
        # by their own crossbar, so a deep net is a STACK of device arrays.
        hidden = [int(h) for h in (cfg.hidden_layers or ()) if int(h) >= 1]
        self.layer_sizes = [self.n_in] + hidden + [self.n_out]
        self.n_layers = len(self.layer_sizes) - 1      # number of crossbars
        # crossbar c is an (size[c+1] x size[c]) grid of independent devices
        self.crossbars = []
        for c in range(self.n_layers):
            n_post, n_pre = self.layer_sizes[c + 1], self.layer_sizes[c]
            self.crossbars.append([[make_device() for _ in range(n_pre)]
                                   for _ in range(n_post)])
        self.syn = self.crossbars[0]          # back-compat alias (first array)
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
        # optional held-out test set (datasets set these) + class names
        self.test_patterns = None
        self.test_targets = None
        self.class_names = [str(j) for j in range(self.n_out)]

        # adaptive thresholds per LIF layer (1..K); spike count = output layer
        self.theta = [np.zeros(s) for s in self.layer_sizes[1:]]
        self.spike_count = np.zeros(self.n_out)
        self.learn_cap = 12.0                 # max per-presentation write gain
        self.lr_scale = 1.0                   # surrogate LR multiplier; the train
        #   loop anneals this 1->~0.15 over epochs so the in-situ device optimiser
        #   settles INTO the solution instead of oscillating past it (the device
        #   write is a coarse, bounded gradient step - large late steps overshoot
        #   the per-pattern margin and knock a solved network back off).
        self.last_write = {}                  # (c,j,i) -> (potentiate, strength)
        #                                       last write event (cell animation)

        # fixed subset of afferents that carry the real pattern; the rest are
        # pure-noise afferents (they fire only the background rate). Chosen once
        # so the pattern lives in the SAME inputs every presentation.
        frac = min(max(cfg.signal_frac, 0.0), 1.0)
        self.signal_mask = np.ones(self.n_in, dtype=float)
        if frac < 1.0:
            n_sig = int(round(frac * self.n_in))
            self.signal_mask = np.zeros(self.n_in, dtype=float)
            if n_sig > 0:
                self.signal_mask[self.rng.choice(self.n_in, n_sig,
                                                  replace=False)] = 1.0
        self.n_signal = int(self.signal_mask.sum())
        # fixed feed-forward inhibitory afferents on the INPUT crossbar (IPSP)
        self.inh0 = np.zeros(self.n_in)
        if neuron.ipsp_gain > 0 and self.n_in // 6:
            self.inh0[self.rng.choice(self.n_in, self.n_in // 6,
                                      replace=False)] = 1.0
        self.exc0 = 1.0 - self.inh0
        # surrogate-gradient training needs ASYMMETRIC initial weights, else all
        # neurons in a layer stay identical (identical gradient) and never
        # specialise.  STDP's WTA + teacher break the symmetry on their own, so
        # only randomise for the surrogate rule.
        if cfg.learn_rule == "surrogate":
            self._randomize_weights(self.rng)

    def _randomize_weights(self, rng, spread=0.55):
        """Spread the crossbar conductances around their reset value by writing
        one random-signed pulse per device (through the device, so they land on
        real reachable states)."""
        for c in range(self.n_layers):
            for j in range(self.layer_sizes[c + 1]):
                for i in range(self.layer_sizes[c]):
                    a = (2.0 * rng.random() - 1.0) * spread
                    self._program(c, j, i, a > 0, abs(a), gain=self.learn_cap)

    # ---- weights ----------------------------------------------------------

    def weights_S(self, c=0):
        return np.array([[device_weight(d) for d in row]
                         for row in self.crossbars[c]])

    def set_weights(self, c, G_S):
        """Restore a saved-model crossbar: write each device's retained
        conductance (S) directly, clamped to the device range.  Sets the
        nonvolatile store `G_nv` when the model has one (ECFET; `G` is a derived
        property there), else the plain `G` (RRAM/memristor)."""
        G = np.asarray(G_S, float)
        for j, row in enumerate(self.crossbars[c]):
            for i, dev in enumerate(row):
                g = float(min(max(G[j][i], self.g_min), self.g_max))
                if hasattr(dev, "G_nv"):
                    dev.G_nv = g
                    if hasattr(dev, "pools"):       # clear fast transients
                        try:
                            dev.pools = [0.0 for _ in dev.pools]
                        except Exception:           # noqa: BLE001
                            pass
                else:
                    try:
                        dev.G = g
                    except AttributeError:
                        pass

    def capture_weights(self):
        """Snapshot every crossbar's retained conductances (S) - cheap arrays the
        train loop keeps as the best-so-far checkpoint."""
        return [self.weights_S(c).copy() for c in range(self.n_layers)]

    def restore_weights(self, snap):
        """Write a capture_weights() snapshot back into the devices."""
        for c, G in enumerate(snap):
            self.set_weights(c, G)

    def weights_uS(self, c=0):
        return self.weights_S(c) * 1e6

    def weights_norm(self, c=0):
        """Conductance of crossbar c normalised to [0, 1] over the device range
        (the synaptic efficacy that drives the neurons - absolute Siemens
        cancel out)."""
        span = (self.g_max - self.g_min) or 1.0
        return np.clip((self.weights_S(c) - self.g_min) / span, 0.0, 1.0)

    # ---- device programming ----------------------------------------------

    def _program(self, c, j, i, potentiate, strength, gain=1.0):
        """Apply one programming pulse to synapse (i->j) of crossbar c.  `gain`
        scales the pulse charge to consolidate a whole presentation's worth of
        writes into one device.step - the device sub-steps internally, so a
        charge-scaled pulse honours the same soft-bound nonlinearity as many
        small ones, but is far cheaper (essential for big grids / datasets)."""
        strength = float(min(max(strength, 0.0), 1.0)) * gain
        if strength <= 1e-3:
            return
        amp = (self.S.pot_amp if potentiate else self.S.dep_amp) * strength
        direction = self.pot_sign if potentiate else -self.pot_sign
        self.crossbars[c][j][i].step(self._t_dev, self.S.pulse_width,
                                     direction * amp)
        self._t_dev += self.S.pulse_width
        self.last_write[(c, j, i)] = (potentiate, strength)

    def _consolidate(self, c, acc, npost):
        """Local competitive plasticity for crossbar c, applied ONCE per
        presentation per firing post-neuron.  mean_pre[i] is the average
        pre-layer trace at that neuron's spikes: pre-neurons active just before
        it fired (mean_pre >= offset) are POTENTIATED, quiet ones are
        heterosynaptically DEPRESSED - the rule that carves clean features.
        Each crossbar learns LOCALLY from its own pre/post layers (no backprop);
        the ACTUAL dG is whatever the device twin yields."""
        S = self.S
        off = max(S.offset, 1e-6)
        n_post, n_pre = self.layer_sizes[c + 1], self.layer_sizes[c]
        for j in range(n_post):
            if npost[j] < 1:
                continue
            mean_pre = acc[j] / npost[j]
            gain = min(self.learn_cap, float(npost[j]))
            for i in range(n_pre):
                p = mean_pre[i]
                if p >= off:
                    self._program(c, j, i, True,
                                  S.a_plus * (p - off) / (1.0 - off), gain)
                elif S.a_minus > 0:
                    self._program(c, j, i, False,
                                  S.a_minus * (off - p) / off, gain)

    # ---- spike encoding (front-end) --------------------------------------

    def encode_spikes(self, vec01, rng, noise=True):
        """Pixel intensities -> (n_steps, n_in) boolean spike matrix using the
        configured encoding.
          noise=False : clean ground-truth pattern - ALL afferents see it, no
                        sensor noise / background / jitter.
          noise=True  : the practical front-end - sensor noise, the signal/noise
                        afferent split, the background rate and spike jitter.
        The base random field is drawn FIRST, so a clean and a noisy call with
        the SAME rng seed share it: the clean spikes nest inside the noisy
        raster (then jitter displaces them) - ideal for a before/after figure.
        """
        cfg = self.cfg
        dt = cfg.dt_ms
        n_steps = max(1, int(round(cfg.present_ms / dt)))
        U = rng.random((n_steps, self.n_in))        # shared base random field
        vec = np.clip(np.asarray(vec01, float), 0.0, 1.0)
        if noise and cfg.input_noise > 0:
            vec = vec + rng.normal(0.0, cfg.input_noise, vec.shape)
            sp = rng.random(vec.shape) < (cfg.input_noise * 0.25)
            vec[sp] = (rng.random(int(sp.sum())) < 0.5).astype(float)
            vec = np.clip(vec, 0.0, 1.0)
        mask = self.signal_mask if noise else np.ones(self.n_in)
        bg = cfg.bg_rate_hz if noise else 0.0
        vec = vec * mask
        rate_pat = vec * cfg.max_rate_hz            # (n_in,) the pattern's rates
        # temporal embedding: show the pattern only for `pattern_ms` (centred)
        # within the window; outside it, only the background fires
        pat_ms = getattr(cfg, "pattern_ms", 0.0)
        if noise and 0.0 < pat_ms < cfg.present_ms:
            pat_steps = max(1, int(round(pat_ms / dt)))
            onset = max(0, (n_steps - pat_steps) // 2)
            tgate = np.zeros(n_steps)
            tgate[onset:onset + pat_steps] = 1.0
            rates = bg + tgate[:, None] * rate_pat[None, :]    # (n_steps, n_in)
        else:
            rates = bg + rate_pat                              # full window
        S = U < (1.0 - np.exp(-rates * dt * 1e-3))
        if cfg.encoding == "latency":
            onset = ((1.0 - vec) * 0.6 * n_steps).astype(int)
            gate = np.arange(n_steps)[:, None] >= onset[None, :]
            gate[:, mask == 0] = True
            S &= gate
        if noise and cfg.jitter_ms > 0:
            ti, ii = np.nonzero(S)
            if ti.size:
                shift = np.round(
                    rng.normal(0.0, cfg.jitter_ms / dt, ti.size)).astype(int)
                nt = np.clip(ti + shift, 0, n_steps - 1)
                S = np.zeros_like(S)
                S[nt, ii] = True
        return S

    def compare_spikes(self, vec01, seed=12345):
        """(clean, noisy) spike matrices for one pattern at a fixed seed, so a
        figure shows exactly what jitter + noise + the afferent split do to the
        SAME underlying realisation."""
        return (self.encode_spikes(vec01, np.random.default_rng(seed), noise=False),
                self.encode_spikes(vec01, np.random.default_rng(seed), noise=True))

    # ---- one stimulus presentation ---------------------------------------

    def present(self, vec01, target=None, learn=True, clean=False):
        """Run the network for one presentation of input intensities vec01.

        Returns a dict with the spike raster, output membrane traces and which
        neuron won - everything the GUI animates.  When learn=True the synapse
        devices are programmed by the STDP events that occur.

        clean=True (used by inference/evaluation) encodes the noise-free
        canonical pattern with a FIXED rng, so a test is a deterministic,
        repeatable measurement of the learned receptive fields - not one random
        noisy Poisson draw - and it does not perturb the training rng stream."""
        cfg, N, S = self.cfg, self.N, self.S
        dt = cfg.dt_ms
        n_steps = max(1, int(round(cfg.present_ms / dt)))
        L = self.layer_sizes                  # [n_in, hidden..., n_out]
        K = self.n_layers                     # crossbars

        # frozen normalised weights per crossbar for this presentation
        Wn = [self.weights_norm(c) for c in range(K)]
        # front-end: pixel intensities -> input spike trains.  A clean eval uses
        # a fixed rng + noise-free pattern (deterministic, repeatable test).
        rng_in = np.random.default_rng(20240517) if clean else self.rng
        S_in = self.encode_spikes(vec01, rng_in, noise=not clean)

        d_syn = math.exp(-dt / N.tau_syn)
        d_pre = math.exp(-dt / S.tau_pre)
        d_theta = math.exp(-dt / N.tau_theta) if N.tau_theta > 0 else 0.0
        leak = dt / max(N.tau_m, 1e-6)

        # per-layer state.  x/pre are traces of EACH layer's own spikes (layer c
        # drives crossbar c); v/refrac exist for the LIF layers 1..K
        x = [np.zeros(s) for s in L]
        pre = [np.zeros(s) for s in L]
        v = [np.full(L[c + 1], N.v_rest) for c in range(K)]
        refrac = [np.zeros(L[c + 1]) for c in range(K)]
        acc = ([np.zeros((L[c + 1], L[c])) for c in range(K)] if learn else None)
        npost = [np.zeros(L[c + 1]) for c in range(K)]

        spikes = [[[] for _ in range(s)] for s in L]   # spike times per neuron
        v_trace = np.zeros((n_steps, self.n_out))      # output layer membrane

        for t in range(n_steps):
            tm = t * dt
            f0 = np.nonzero(S_in[t])[0]                 # input spikes this step
            if f0.size:
                x[0][f0] = 1.0
                pre[0][f0] = 1.0
                for i in f0:
                    spikes[0][int(i)].append(tm)

            # forward pass: layer 0 -> 1 -> ... -> K through the crossbars
            for c in range(K):
                eff = Wn[c] * x[c]                      # (L[c+1], L[c])
                norm = x[c].sum() + 1.0
                # boost drive when the TARGET layer (c+1) is hidden - those
                # neurons have no teacher and must fire on feedforward alone
                g = N.hidden_gain if (c + 1 < K) else 1.0
                if c == 0:                              # input crossbar: EPSP/IPSP
                    drive = g * (N.epsp_gain * (eff * self.exc0).sum(1)
                                 - N.ipsp_gain * (eff * self.inh0).sum(1)) / norm
                else:                                   # hidden->* crossbars
                    drive = g * N.epsp_gain * eff.sum(1) / norm
                if c == K - 1 and cfg.mode == "supervised" and target is not None:
                    drive[target] += N.teacher

                active = refrac[c] <= 0
                v[c][active] += leak * (N.v_rest - v[c][active] + drive[active])
                if N.v_noise > 0:
                    v[c][active] += self.rng.normal(0.0, N.v_noise,
                                                    int(active.sum()))
                v[c][~active] = N.v_reset
                refrac[c][~active] -= dt

                fired = np.nonzero(
                    (v[c] >= (N.v_threshold + self.theta[c])) & active)[0]
                if fired.size:
                    # winner-take-all on the OUTPUT layer (one class wins);
                    # hidden layers stay distributed (boosted, dense) so the
                    # next crossbar has a code to read and learn from
                    if N.inhibition > 0 and L[c + 1] > 1 and c == K - 1:
                        win = fired[int(np.argmax(v[c][fired]))]
                        others = np.ones(L[c + 1], bool); others[win] = False
                        v[c][others] -= N.inhibition
                        fired = np.array([win])
                    for j in fired:
                        spikes[c + 1][int(j)].append(tm)
                        if c == K - 1 and learn:    # don't pollute totals on eval
                            self.spike_count[j] += 1
                        v[c][j] = N.v_reset
                        refrac[c][j] = N.t_refractory
                        if c == K - 1:              # homeostasis: output layer
                            self.theta[c][j] += N.theta_plus
                        x[c + 1][j] = 1.0           # feeds the next crossbar
                        pre[c + 1][j] = 1.0
                        if learn:                   # local STDP accrual
                            acc[c][j] += pre[c]
                            npost[c][j] += 1

                if c == K - 1:
                    # record the OUTPUT membrane AFTER reset, so refractory
                    # neurons sit at the reset floor (a visible flat segment);
                    # a neuron that fired this step is drawn as a spike peak so
                    # the spike appears exactly at the threshold crossing
                    v_trace[t] = v[c]
                    if fired.size:
                        v_trace[t][fired] = N.v_spike_peak

            for c in range(len(L)):                 # relax all layer traces
                x[c] *= d_syn
                pre[c] *= d_pre
            if N.tau_theta > 0:
                for c in range(K):
                    self.theta[c] *= d_theta

        if learn:
            for c in range(K):
                self._consolidate(c, acc[c], npost[c])
        self._t_dev += cfg.present_ms * 1e-3

        out_spikes = spikes[-1]
        in_spikes = [(tm, i) for i, ts in enumerate(spikes[0]) for tm in ts]
        winner = (int(np.argmax([len(s) for s in out_spikes]))
                  if any(out_spikes) else -1)
        return {
            "v_trace": v_trace, "dt_ms": dt, "n_steps": n_steps,
            "out_spikes": out_spikes, "in_spikes": in_spikes,
            "winner": winner, "n_out_spikes": [len(s) for s in out_spikes],
            "layer_spikes": spikes, "layer_sizes": list(L),
        }

    # ---- learning-rule dispatch ------------------------------------------

    def train_step(self, vec01, target):
        """One learning step under the configured rule.  STDP is device-local;
        surrogate is supervised BPTT.  Returns a present()-style result dict."""
        if self.cfg.learn_rule == "surrogate":
            return self._surrogate_forward(
                vec01, 0 if target is None else int(target), learn=True)
        return self.present(vec01, target=target, learn=True)

    def infer(self, vec01):
        """Forward-only pass for testing - uses the SAME forward as training
        under each rule (so the surrogate trains and tests the same network)."""
        if self.cfg.learn_rule == "surrogate":
            return self._surrogate_forward(vec01, None, learn=False, clean=True)
        return self.present(vec01, target=None, learn=False, clean=True)

    # ---- surrogate-gradient (BPTT) training ------------------------------

    def _surrogate_forward(self, vec01, target, learn=True, clean=False):
        """Forward (and, when learn=True, supervised backprop-through-time)
        across ALL crossbars.  A smooth fast-sigmoid surrogate stands in for the
        spike function's derivative; the loss is softmax cross-entropy on the
        mean output membrane; the per-synapse gradient is applied THROUGH the
        device (programming pulses in the gradient direction) - in-situ /
        hardware-aware training, so the device nonlinearity, bounds and write
        noise stay in the loop.  Inference (learn=False) runs the SAME forward,
        so training and testing use the identical network."""
        cfg, N, S = self.cfg, self.N, self.S
        dt = cfg.dt_ms
        T = max(1, int(round(cfg.present_ms / dt)))
        L, K = self.layer_sizes, self.n_layers
        Wn = [self.weights_norm(c) for c in range(K)]
        alpha = math.exp(-dt / max(N.tau_m, 1e-6))
        thr = N.v_threshold
        beta = max(S.sg_beta, 0.5)
        gains = [N.epsp_gain * (N.hidden_gain if (c + 1 < K) else 1.0)
                 for c in range(K)]

        # clean eval: fixed rng + noise-free pattern -> deterministic test
        rng_in = np.random.default_rng(20240517) if clean else self.rng
        S_in = self.encode_spikes(vec01, rng_in, noise=not clean).astype(float)
        # ---- forward, storing activations for the backward pass ----
        spk = [S_in]                              # spk[c]: (T, L[c])  spikes 0/1
        cache = []                                # per crossbar: (u, norm, pre)
        v_out = np.zeros((T, L[-1]))              # output membrane (for the GUI)
        for c in range(K):
            pre = spk[c]
            norm = pre.sum(1) + 1.0
            I = gains[c] * (pre @ Wn[c].T) / norm[:, None]      # (T, L[c+1])
            v = np.zeros(L[c + 1])
            u = np.empty((T, L[c + 1]))
            s = np.empty((T, L[c + 1]))
            for t in range(T):
                uu = alpha * v + I[t]
                sp = (uu >= thr).astype(float)
                v = uu - thr * sp                # soft reset
                u[t] = uu
                s[t] = sp
                if c == K - 1:
                    v_out[t] = v
            spk.append(s)
            cache.append((u, norm, pre))

        # readout: mean output-membrane "evidence" -> softmax.  The membrane
        # (not just spike counts) gives a dense, always-on signal; the winner is
        # the argmax, used for BOTH the loss and inference, so train == test.
        u_out = cache[K - 1][0]                    # (T, n_out) pre-reset membrane
        logit = u_out.mean(0)
        winner = int(np.argmax(logit))
        loss = 0.0

        if learn and target is not None:
            z = logit - logit.max()
            p = np.exp(z); p = p / (p.sum() + 1e-12)
            loss = float(-math.log(p[target] + 1e-12))
            g_logit = p.copy(); g_logit[target] -= 1.0
            g_read = np.tile(g_logit / T, (T, 1))  # dL/du_out[t] (readout)
            # ---- backprop through layers + time, applying the gradient ----
            gS = None                              # spike grad from the layer above
            for c in range(K - 1, -1, -1):
                u, norm, pre = cache[c]
                sg = 1.0 / (1.0 + beta * np.abs(u - thr)) ** 2
                dI = np.empty_like(u)
                dv_next = np.zeros(L[c + 1])
                for t in range(T - 1, -1, -1):
                    if c == K - 1:                 # output: membrane readout
                        du = g_read[t] + dv_next * (1.0 - thr * sg[t])
                    else:                          # hidden: spike path from above
                        du = gS[t] * sg[t] + dv_next * (1.0 - thr * sg[t])
                    dI[t] = du
                    dv_next = du * alpha
                scaled = gains[c] * dI / norm[:, None]
                self._apply_grad(c, scaled.T @ pre)
                if c > 0:
                    gS = scaled @ Wn[c]            # backprop to the pre-layer

        # ---- present()-style result for the GUI ----
        layer_spikes = [[(np.nonzero(spk[c][:, n])[0] * dt).tolist()
                         for n in range(L[c])] for c in range(len(spk))]
        out_spikes = layer_spikes[-1]
        if learn:                                   # don't pollute totals on eval
            for j, ts in enumerate(out_spikes):
                self.spike_count[j] += len(ts)
        in_spikes = [(tm, i) for i, ts in enumerate(layer_spikes[0]) for tm in ts]
        self._t_dev += cfg.present_ms * 1e-3
        return {
            "v_trace": v_out, "dt_ms": dt, "n_steps": T,
            "out_spikes": out_spikes, "in_spikes": in_spikes,
            "winner": winner, "n_out_spikes": [len(s) for s in out_spikes],
            "layer_spikes": layer_spikes, "layer_sizes": list(L),
            "loss": loss,
        }

    def _apply_grad(self, c, dW):
        """Apply a gradient to crossbar c by programming each synapse THROUGH
        the device in the descent direction (-dW), magnitude ~ |gradient|."""
        delta = -self.S.sg_lr * self.lr_scale * dW   # annealed change in Wn
        n_post, n_pre = self.layer_sizes[c + 1], self.layer_sizes[c]
        for j in range(n_post):
            row = delta[j]
            for i in range(n_pre):
                d = float(row[i])
                strength = min(abs(d), 1.0)
                if strength < 4e-3:               # skip negligible updates
                    continue
                self._program(c, j, i, d > 0.0, strength, gain=self.learn_cap)

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
        theta_save = [t.copy() for t in self.theta]
        self.theta = [np.zeros_like(t) for t in self.theta]
        try:
            rows = []
            for k, (lbl, vec) in items:
                res = self.infer(vec)              # same forward as training
                rows.append({"label": lbl, "winner": res["winner"],
                             "counts": res["n_out_spikes"],
                             "target": self.target_of[k]})
            return rows
        finally:
            self.theta = theta_save

    def evaluate(self, use_test=False, max_n=None):
        """Forward-only over the train (or held-out test) set; returns
        (y_true, y_pred) of class indices for the metrics (confusion / F1)."""
        if use_test and self.test_patterns:
            pats, tgts = self.test_patterns, self.test_targets
        else:
            pats, tgts = self.patterns, self.target_of
        items = list(zip(pats, tgts))
        if max_n and len(items) > max_n:
            sel = sorted(np.random.default_rng(1).choice(
                len(items), max_n, replace=False).tolist())
            items = [items[i] for i in sel]
        theta_save = [t.copy() for t in self.theta]
        self.theta = [np.zeros_like(t) for t in self.theta]
        try:
            y_true, y_pred = [], []
            for (lbl, vec), tgt in items:
                y_pred.append(int(self.infer(vec)["winner"]))
                y_true.append(int(tgt))
            return y_true, y_pred
        finally:
            self.theta = theta_save

    def eval_accuracy(self, use_test=False, max_n=None):
        yt, yp = self.evaluate(use_test, max_n)
        if not yt:
            return 0.0
        return sum(1 for a, b in zip(yt, yp) if a == b) / len(yt)
