"""Crossbar spiking network: input spikes -> device crossbar(s) -> LIF layers.

        inputs (a gh x gw pixel grid, one spiking neuron per pixel)
          |   each input i  --[ synapse device g_ij ]-->  neuron j
          v
        crossbar  W  (n_post x n_pre conductances, read from a DeviceArray)
          |
          v
        LIF neurons; winner-take-all lateral inhibition on the output layer

Two learning rules, both applied THROUGH the device (the rule decides when and
in which direction to write; the device model decides how far the conductance
actually moves):

  "stdp"       device-local recent-pre eligibility rule, supervised by a
               teacher current or unsupervised (pure WTA competition)
  "surrogate"  supervised BPTT with a fast-sigmoid surrogate gradient

A presentation is vectorized: the input crossbar's drive for every time step
is one matrix product, plasticity is one matrix product, and all writes of a
presentation are one DeviceArray.program call.
"""
import copy
import math

import numpy as np

from . import stdp as _stdp
from . import surrogate as _sg
from .crossbar import input_drive, normalize, trace_matrix
from .devices import device_factory
from .encoding import encode_spikes as _encode_spikes, input_size, scan_bin_steps
from .patterns import make_patterns
from .power import EnergyMeter
from .state import pack, unpack

CLEAN_SEED = 20240517     # fixed rng of a clean (noise-free) evaluation
MIN_GRAD_STEP = 4e-3      # surrogate weight changes below this are skipped


class CrossbarSNN:
    """Build once, then call train_step()/fit() and evaluate().

    device   a registered model name ("fefet", "softbound", ...) or a factory
             ``f(n_post, n_pre, seed) -> DeviceArray``
    power    :class:`neurovat.power.PowerParams` of the energy model
             (``self.energy``); None runs it on flagged placeholder values
    """

    def __init__(self, device, neuron, stdp, cfg, patterns=None, targets=None,
                 power=None):
        self.N = neuron
        self.S = stdp
        self.cfg = cfg
        self.n_in = input_size(cfg)
        if cfg.encoding == "scan":
            scan_bin_steps(cfg)               # the sweep must fit the window
        self.n_out = cfg.n_out
        self.rng = np.random.default_rng(cfg.seed)
        self._t_dev = 0.0                     # shared device wall-clock (s)

        # layer stack: [n_in, hidden..., n_out].  Consecutive layers are joined
        # by their own crossbar, so a deep net is a STACK of device arrays.
        hidden = [int(h) for h in (cfg.hidden_layers or ()) if int(h) >= 1]
        self.layer_sizes = [self.n_in] + hidden + [self.n_out]
        self.n_layers = len(self.layer_sizes) - 1      # number of crossbars
        factory = device_factory(device) if isinstance(device, str) else device
        # device variation draws from its own stream, never the training rng
        self.arrays = [factory(self.layer_sizes[c + 1], self.layer_sizes[c],
                               seed=[int(cfg.seed), 53, c])
                       for c in range(self.n_layers)]
        # energy accounting by phase: "init" (weight initialisation writes),
        # "train" (learning presentations + writes), "eval" (forward-only)
        self.energy = EnergyMeter([a.shape for a in self.arrays],
                                  [a.input_kind for a in self.arrays],
                                  power, cfg.dt_ms * 1e-3)
        self._phase = "init"
        self.g_max = float(self.arrays[0].g_max)
        self.g_min = float(self.arrays[0].g_min)
        self.pot_sign = (stdp.pot_sign if stdp.pot_sign else
                         self.arrays[0].potentiation_sign(stdp.pot_amp,
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
        self.validation_patterns = None
        self.validation_targets = None
        self.test_patterns = None
        self.test_targets = None
        self.class_names = [str(j) for j in range(self.n_out)]

        # adaptive thresholds per LIF layer (1..K); spike count = output layer
        self.theta = [np.zeros(s) for s in self.layer_sizes[1:]]
        self.spike_count = np.zeros(self.n_out)
        self.learn_cap = 12.0                 # max per-presentation write gain
        self.lr_scale = 1.0                   # surrogate LR multiplier; fit()
        #   anneals this 1->~0.15 over epochs so the in-situ device optimiser
        #   settles INTO the solution instead of oscillating past it
        self.last_write = {}                  # (c,j,i) -> (potentiate, strength)
        # run statistics (not part of the checkpoint): programming effort
        self.write_count = 0                  # pulses applied
        self.write_effort = 0.0               # sum |amplitude| x width (V s or A s)

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
        self._phase = "train"

    def _randomize_weights(self, rng, spread=0.55):
        """Spread the crossbar conductances around their reset value by writing
        one random-signed pulse per device (through the device, so they land on
        real reachable states)."""
        for c in range(self.n_layers):
            a = (2.0 * rng.random(self.arrays[c].shape) - 1.0) * spread
            self._program_many(c, True, a > 0, np.abs(a), self.learn_cap)

    # ---- weights ----------------------------------------------------------

    def weights_S(self, c=0):
        return np.array(self.arrays[c].conductance(), dtype=float)

    def weights_uS(self, c=0):
        return self.weights_S(c) * 1e6

    def weights_norm(self, c=0):
        """Conductance of crossbar c normalised to [0, 1] over the device range
        (the synaptic efficacy that drives the neurons - absolute Siemens
        cancel out)."""
        return normalize(self.arrays[c].conductance(), self.g_min, self.g_max)

    def set_weights(self, c, G_S):
        """Import conductance-only weights; fail on unsupported devices."""
        G = np.asarray(G_S, float)
        if G.shape != (self.layer_sizes[c+1], self.layer_sizes[c]) or not np.isfinite(G).all():
            raise ValueError("Invalid crossbar weights")
        self.arrays[c].set_conductance(G)

    def capture_weights(self):
        """Snapshot every crossbar's retained conductances (S)."""
        return [self.weights_S(c) for c in range(self.n_layers)]

    def restore_weights(self, snap):
        """Write a capture_weights() snapshot back into the devices."""
        for c, G in enumerate(snap):
            self.set_weights(c, G)

    def capture_state(self):
        """Full dynamic state, including device state and RNGs (JSON-safe)."""
        return pack({
            "devices": [a.state_dict() for a in self.arrays],
            "theta": self.theta, "rng": self.rng.bit_generator.state,
            "time": self._t_dev, "spike_count": self.spike_count,
            "lr_scale": self.lr_scale,
        })

    def restore_state(self, state):
        state = unpack(state)
        bars = state["devices"]
        if len(bars) != self.n_layers:
            raise ValueError("Checkpoint architecture mismatch")
        for array, saved in zip(self.arrays, bars):
            array.load_state_dict(saved)
        self.theta = state["theta"]
        self.rng.bit_generator.state = state["rng"]
        self._t_dev = state["time"]
        self.spike_count = state["spike_count"]
        self.lr_scale = state["lr_scale"]
        self.last_write = {}

    def split_validation(self, fraction=.2):
        """Stratify a validation subset from TRAINING only, keeping >=2/class."""
        if self.validation_patterns:
            return
        rng = np.random.default_rng(self.cfg.seed + 913)
        chosen = set()
        targets = np.asarray(self.target_of)
        for label in np.unique(targets):
            indices = np.flatnonzero(targets == label)
            count = min(len(indices)-2, max(1, int(len(indices)*fraction)))
            if count > 0:
                chosen.update(rng.permutation(indices)[:count].tolist())
        if chosen:
            self.validation_patterns = [p for i, p in enumerate(self.patterns) if i in chosen]
            self.validation_targets = [t for i, t in enumerate(self.target_of) if i in chosen]
            self.patterns = [p for i, p in enumerate(self.patterns) if i not in chosen]
            self.target_of = [t for i, t in enumerate(self.target_of) if i not in chosen]

    # ---- device programming ----------------------------------------------

    @property
    def last_write(self):
        """(c, j, i) -> (potentiate, strength) of each synapse's latest write."""
        if self._lw_dict is None:
            self._lw_dict = {
                (c, int(j), int(i)): (bool(pot[j, i]), float(s[j, i]))
                for c, (pot, s) in self._lw.items()
                for j, i in zip(*np.nonzero(~np.isnan(s)))}
        return self._lw_dict

    @last_write.setter
    def last_write(self, value):
        if value:
            raise ValueError("last_write can only be cleared")
        self._lw, self._lw_dict = {}, {}

    def _program_many(self, c, mask, potentiate, strength, gain=1.0):
        """Write every synapse of crossbar c selected by ``mask`` in one device
        call. ``gain`` scales the pulse to consolidate a whole presentation's
        worth of writes into one pulse per device."""
        shape = self.arrays[c].shape
        rows, cols, amps, pot, s = _stdp.pulses(
            np.broadcast_to(mask, shape), potentiate,
            np.broadcast_to(strength, shape), gain, self.S, self.pot_sign)
        if not rows.size:
            return
        array = self.arrays[c]
        g_before = np.asarray(array.conductance(), float)[rows, cols]
        current = getattr(array, "write_current", None)
        if current is not None:
            current = current(rows, cols, amps, self.S.pulse_width)
        self._t_dev = array.program(rows, cols, amps,
                                    self.S.pulse_width, self._t_dev)
        self.energy.record_write(
            self._phase, c, rows, cols, amps, self.S.pulse_width, g_before,
            np.asarray(array.conductance(), float)[rows, cols], current)
        if c not in self._lw:
            self._lw[c] = (np.zeros(shape, bool), np.full(shape, np.nan))
        self._lw[c][0][rows, cols] = pot
        self._lw[c][1][rows, cols] = s
        self._lw_dict = None
        self.write_count += int(rows.size)
        self.write_effort += float(np.abs(amps).sum()) * self.S.pulse_width

    def _program(self, c, j, i, potentiate, strength, gain=1.0):
        """Apply one programming pulse to synapse (i->j) of crossbar c."""
        mask = np.zeros(self.arrays[c].shape, bool)
        mask[j, i] = True
        self._program_many(c, mask, bool(potentiate), float(strength), gain)

    def _consolidate(self, c, acc, npost):
        """Local competitive plasticity for crossbar c, applied ONCE per
        presentation per firing post-neuron: inputs active just before it fired
        are POTENTIATED, quiet ones heterosynaptically DEPRESSED. The ACTUAL dG
        is whatever the device yields."""
        self._program_many(c, *_stdp.consolidation(acc, npost, self.S,
                                                   self.learn_cap))

    def _apply_grad(self, c, dW):
        """Apply a gradient to crossbar c by programming each synapse THROUGH
        the device in the descent direction (-dW), magnitude ~ |gradient|."""
        delta = -self.S.sg_lr * self.lr_scale * dW   # annealed change in Wn
        strength = np.minimum(np.abs(delta), 1.0)
        self._program_many(c, strength >= MIN_GRAD_STEP, delta > 0.0, strength,
                           self.learn_cap)

    # ---- spike encoding (front-end) --------------------------------------

    def encode_spikes(self, vec01, rng, noise=True):
        """Pixel intensities -> (n_steps, n_in) boolean spike matrix; see
        :func:`neurovat.encoding.encode_spikes`."""
        return _encode_spikes(vec01, self.cfg, rng, self.signal_mask, noise)

    def compare_spikes(self, vec01, seed=12345):
        """(clean, noisy) spike matrices for one pattern at a fixed seed."""
        return (self.encode_spikes(vec01, np.random.default_rng(seed), noise=False),
                self.encode_spikes(vec01, np.random.default_rng(seed), noise=True))

    def _input(self, vec01, clean, rng):
        """(rng, spikes) of one presentation. clean: noise-free pattern from a
        FIXED rng (a deterministic, repeatable test that leaves the training
        stream alone). rng: noisy encoding from a caller-owned stream."""
        if rng is not None:
            return rng, self.encode_spikes(vec01, rng, noise=True)
        rng = np.random.default_rng(CLEAN_SEED) if clean else self.rng
        return rng, self.encode_spikes(vec01, rng, noise=not clean)

    # ---- one stimulus presentation ---------------------------------------

    def present(self, vec01, target=None, learn=True, clean=False, rng=None,
                detail=True):
        """Run the network for one presentation of input intensities vec01.

        Returns a dict with the spike raster, output membrane traces and which
        neuron won. When learn=True the synapse devices are programmed by the
        plasticity events that occur. detail=False skips the per-neuron spike
        lists (only winner and counts are returned)."""
        cfg, N, S = self.cfg, self.N, self.S
        dt = cfg.dt_ms
        L, K = self.layer_sizes, self.n_layers

        # frozen normalised weights per crossbar for this presentation
        Wn = [self.weights_norm(c) for c in range(K)]
        rng_in, S_in = self._input(vec01, clean, rng)
        S_in = np.asarray(S_in, bool)
        T = S_in.shape[0]

        d_syn = math.exp(-dt / N.tau_syn)
        d_pre = math.exp(-dt / S.tau_pre)
        d_theta = math.exp(-dt / N.tau_theta) if N.tau_theta > 0 else 0.0
        leak = dt / max(N.tau_m, 1e-6)
        thr = N.v_threshold
        teach = cfg.mode == "supervised" and target is not None

        # input crossbar: the drive of every step in one matrix product
        D0 = input_drive(Wn[0], trace_matrix(S_in, d_syn), self.exc0, self.inh0,
                         N.epsp_gain, N.ipsp_gain,
                         N.hidden_gain if K > 1 else 1.0)
        if teach and K == 1:
            D0[:, target] += N.teacher

        rast = [S_in] + [np.zeros((T, L[c + 1]), bool) for c in range(K)]
        x = [None] + [np.zeros(L[c]) for c in range(1, K)]   # hidden traces
        v = [np.full(L[c + 1], N.v_rest) for c in range(K)]
        refrac = [np.zeros(L[c + 1]) for c in range(K)]
        v_trace = np.zeros((T, self.n_out))      # output layer membrane
        wta = N.inhibition > 0 and self.n_out > 1
        wta_ops = 0

        for t in range(T):
            for c in range(K):
                if c == 0:
                    drive = D0[t]
                else:
                    # boost drive when the TARGET layer is hidden - those
                    # neurons have no teacher and must fire on feedforward alone
                    g = N.hidden_gain if (c + 1 < K) else 1.0
                    drive = g * N.epsp_gain * (Wn[c] @ x[c]) / (x[c].sum() + 1.0)
                    if teach and c == K - 1:
                        drive[target] += N.teacher
                vc, rc = v[c], refrac[c]
                active = rc <= 0
                vc[active] += leak * (N.v_rest - vc[active] + drive[active])
                if N.v_noise > 0:
                    vc[active] += rng_in.normal(0.0, N.v_noise, int(active.sum()))
                resting = ~active
                vc[resting] = N.v_reset
                rc[resting] -= dt

                fired = (vc >= thr + self.theta[c]) & active
                any_fired = bool(fired.any())
                if any_fired:
                    # winner-take-all on the OUTPUT layer (one class wins);
                    # hidden layers stay distributed so the next crossbar has
                    # a code to read and learn from
                    if wta and c == K - 1:
                        idx = np.nonzero(fired)[0]
                        win = idx[int(np.argmax(vc[idx]))]
                        wta_ops += 1
                        vc -= N.inhibition          # the winner is reset below
                        fired = np.zeros(L[c + 1], bool)
                        fired[win] = True
                    vc[fired] = N.v_reset
                    rc[fired] = N.t_refractory
                    if c == K - 1:                  # homeostasis: output layer
                        self.theta[c][fired] += N.theta_plus
                    else:
                        x[c + 1][fired] = 1.0       # feeds the next crossbar
                    rast[c + 1][t] = fired
                if c == K - 1:
                    # record the OUTPUT membrane AFTER reset; a neuron that
                    # fired this step is drawn as a spike peak
                    v_trace[t] = vc
                    if any_fired:
                        v_trace[t][fired] = N.v_spike_peak

            for c in range(1, K):                   # relax hidden traces
                x[c] *= d_syn
            if N.tau_theta > 0:
                for c in range(K):
                    self.theta[c] *= d_theta

        counts = rast[-1].sum(0)
        self.energy.record_forward("train" if learn else "eval",
                                   [self.weights_S(c) for c in range(K)],
                                   rast, wta_ops)
        if learn:
            self.spike_count += counts              # not polluted by eval
            for c in range(K):
                acc, npost = _stdp.eligibility(rast[c], rast[c + 1], d_pre)
                if npost.any():
                    self._consolidate(c, acc, npost)
        self._t_dev += cfg.present_ms * 1e-3

        winner = int(np.argmax(counts)) if counts.any() else -1
        res = {"v_trace": v_trace, "dt_ms": dt, "n_steps": T, "winner": winner,
               "n_out_spikes": counts.tolist(), "layer_sizes": list(L)}
        if detail:
            res.update(self._spike_lists(rast, dt))
        return res

    @staticmethod
    def _spike_lists(rast, dt):
        layer_spikes = []
        for r in rast:                     # per neuron: its spike times (ms)
            neuron, step = np.nonzero(np.asarray(r).T)
            times = (step * dt).tolist()
            ends = np.cumsum(np.bincount(neuron, minlength=r.shape[1])).tolist()
            layer_spikes.append([times[a:b] for a, b in zip([0] + ends, ends)])
        return {"layer_spikes": layer_spikes, "out_spikes": layer_spikes[-1],
                "in_spikes": [(tm, i) for i, ts in enumerate(layer_spikes[0])
                              for tm in ts]}

    # ---- learning-rule dispatch ------------------------------------------

    def train_step(self, vec01, target, detail=True):
        """One learning step under the configured rule.  STDP is device-local;
        surrogate is supervised BPTT.  Returns a present()-style result dict."""
        if self.cfg.learn_rule == "surrogate":
            return self._surrogate_forward(
                vec01, 0 if target is None else int(target), learn=True,
                detail=detail)
        return self.present(vec01, target=target, learn=True, detail=detail)

    def infer(self, vec01, rng=None, detail=True):
        """Forward-only pass - the SAME forward as training under each rule.
        Clean (noise-free, fixed rng) unless ``rng`` is given, in which case
        the input is encoded with the configured noise/jitter from that rng."""
        if self.cfg.learn_rule == "surrogate":
            return self._surrogate_forward(vec01, None, learn=False, clean=True,
                                           rng=rng, detail=detail)
        return self.present(vec01, target=None, learn=False, clean=True,
                            rng=rng, detail=detail)

    # ---- surrogate-gradient (BPTT) training ------------------------------

    def _sg_setup(self):
        N, K = self.N, self.n_layers
        Wn = [self.weights_norm(c) for c in range(K)]
        gains = [N.epsp_gain * (N.hidden_gain if (c + 1 < K) else 1.0)
                 for c in range(K)]
        alpha = math.exp(-self.cfg.dt_ms / max(N.tau_m, 1e-6))
        return Wn, gains, alpha, N.v_threshold

    def _surrogate_forward(self, vec01, target, learn=True, clean=False,
                           rng=None, detail=True):
        """Forward (and, when learn=True, supervised backprop-through-time)
        across ALL crossbars. The per-synapse gradient is applied THROUGH the
        device (programming pulses in the gradient direction) - in-situ /
        hardware-aware training, so the device nonlinearity, bounds and write
        noise stay in the loop. Inference runs the SAME forward."""
        cfg = self.cfg
        dt = cfg.dt_ms
        K = self.n_layers
        Wn, gains, alpha, thr = self._sg_setup()
        _, S_in = self._input(vec01, clean, rng)
        spk, cache = _sg.forward(np.asarray(S_in, float), Wn, gains, alpha, thr)
        T = spk[0].shape[0]
        self.energy.record_forward("train" if learn else "eval",
                                   [self.weights_S(c) for c in range(K)], spk)

        # readout: mean output-membrane "evidence" -> softmax; the winner is the
        # argmax, used for BOTH the loss and inference, so train == test.
        u_out = cache[K - 1][0]                    # (T, n_out) pre-reset membrane
        winner = int(np.argmax(u_out.mean(0)))
        loss = 0.0
        if learn and target is not None:
            loss, grads = _sg.loss_and_grads(cache, Wn, gains, alpha, thr,
                                             max(self.S.sg_beta, 0.5), target)
            for c in range(K - 1, -1, -1):
                self._apply_grad(c, grads[c])

        counts = spk[-1].sum(0)
        if learn:                                   # don't pollute totals on eval
            self.spike_count += counts
        self._t_dev += cfg.present_ms * 1e-3
        res = {"v_trace": u_out - thr * spk[-1], "dt_ms": dt, "n_steps": T,
               "winner": winner, "n_out_spikes": counts.astype(int).tolist(),
               "layer_sizes": list(self.layer_sizes), "loss": loss}
        if detail:
            res.update(self._spike_lists(spk, dt))
        return res

    # ---- scoring ----------------------------------------------------------

    def _frozen(self):
        """Context for forward-only scoring: thresholds set aside (recognition
        is about the receptive fields, not accumulated theta), training rng and
        device clock untouched."""
        net = self

        class _Frozen:
            def __enter__(self):
                self.saved = (copy.deepcopy(net.rng.bit_generator.state),
                              net._t_dev, [t.copy() for t in net.theta])
                net.theta = [np.zeros_like(t) for t in net.theta]

            def __exit__(self, *exc):
                net.rng.bit_generator.state, net._t_dev, net.theta = self.saved
        return _Frozen()

    def classify(self, learn=False, max_n=None):
        """Present patterns once (no learning) and report the winning output
        neuron + spike counts. Pass max_n to score a random sample."""
        items = list(enumerate(self.patterns))
        if max_n and len(items) > max_n:
            sel = sorted(np.random.default_rng(0).choice(
                len(items), max_n, replace=False).tolist())
            items = [items[i] for i in sel]
        with self._frozen():
            rows = []
            for k, (lbl, vec) in items:
                res = self.infer(vec)              # same forward as training
                rows.append({"label": lbl, "winner": res["winner"],
                             "counts": res["n_out_spikes"],
                             "target": self.target_of[k]})
            return rows

    def _split(self, use_test, split):
        if split == "validation":
            if not self.validation_patterns:
                raise ValueError("No validation set")
            return self.validation_patterns, self.validation_targets
        if use_test or split == "test":
            if not self.test_patterns:
                raise ValueError("No held-out test set")
            return self.test_patterns, self.test_targets
        return self.patterns, self.target_of

    def predict(self, vecs, noisy=False, batch=256):
        """Winning class of each input vector (-1 = no output spike).

        noisy=False scores the clean pattern (deterministic). noisy=True
        encodes sample k with the configured noise/jitter from its own rng
        seeded by (seed, k), so the result is reproducible and independent of
        training history. Call inside evaluate() or with thresholds frozen."""
        rng_of = ((lambda k: np.random.default_rng([int(self.cfg.seed), 99991, k]))
                  if noisy else (lambda k: None))
        # the surrogate forward has no state carried between samples, so it is
        # scored in batches - unless a hook replaced part of the forward path
        hooked = any(k in self.__dict__ for k in
                     ("infer", "present", "encode_spikes", "_surrogate_forward"))
        if self.cfg.learn_rule != "surrogate" or hooked:
            return [int(self.infer(v, rng=rng_of(k), detail=False)["winner"])
                    for k, v in enumerate(vecs)]
        Wn, gains, alpha, thr = self._sg_setup()
        out = []
        for a in range(0, len(vecs), batch):
            S = np.array([self._input(v, True, rng_of(a + k))[1]
                          for k, v in enumerate(vecs[a:a + batch])], dtype=float)
            spk, cache = _sg.forward(S, Wn, gains, alpha, thr)
            self.energy.record_forward(
                "eval", [self.weights_S(c) for c in range(self.n_layers)], spk)
            out.extend(np.argmax(cache[-1][0].mean(1), axis=1).tolist())
        return out

    def evaluate(self, use_test=False, max_n=None, split=None, noisy=False):
        """Forward-only over the train / validation / held-out test set;
        returns (y_true, y_pred) of class indices for the metrics."""
        pats, tgts = self._split(use_test, split)
        items = list(zip(pats, tgts))
        if max_n and len(items) > max_n:
            sel = sorted(np.random.default_rng(1).choice(
                len(items), max_n, replace=False).tolist())
            items = [items[i] for i in sel]
        with self._frozen():
            y_pred = self.predict([vec for (_lbl, vec), _t in items], noisy=noisy)
        return [int(t) for _p, t in items], y_pred

    def eval_accuracy(self, use_test=False, max_n=None, split=None, noisy=False):
        yt, yp = self.evaluate(use_test, max_n, split, **({"noisy": True} if noisy else {}))
        if not yt:
            return 0.0
        return sum(1 for a, b in zip(yt, yp) if a == b) / len(yt)

    # ---- training loop ----------------------------------------------------

    def fit(self, epochs, anneal=True, on_epoch=None):
        """Train for ``epochs`` passes over the training set in a seeded random
        order. With a validation split, the best epoch (highest validation
        accuracy, then lowest training loss) is restored at the end; the test
        set is never used for selection. Returns the per-epoch history."""
        import time
        order = np.random.default_rng([int(self.cfg.seed), 777])
        surrogate = self.cfg.learn_rule == "surrogate"
        has_val = bool(self.validation_patterns)
        history, best, best_state = [], None, None
        for epoch in range(1, epochs + 1):
            if surrogate and anneal:
                self.lr_scale = .15 + .85 * .5 * (1 + math.cos(
                    math.pi * (epoch - 1) / max(epochs - 1, 1)))
            t0 = time.perf_counter()
            loss, hits, writes0 = 0.0, 0, self.write_count
            for k in order.permutation(len(self.patterns)):
                target = int(self.target_of[k])
                res = self.train_step(self.patterns[k][1], target, detail=False)
                loss += res.get("loss", 0.0)
                hits += res["winner"] == target
            n = max(len(self.patterns), 1)
            row = {"epoch": epoch, "train_s": time.perf_counter() - t0,
                   "online_acc": hits / n, "writes": self.write_count - writes0}
            if surrogate:
                row["loss"] = loss / n
            if has_val:
                row["val_acc"] = self.eval_accuracy(split="validation")
                score = (row["val_acc"], -loss)
                if best is None or score > best:
                    best, best_state = score, self.capture_state()
                    row["selected"] = True
            history.append(row)
            if on_epoch:
                on_epoch(row)
        if best_state is not None:
            self.restore_state(best_state)
            for row in history:
                row["selected"] = row["epoch"] == max(
                    r["epoch"] for r in history if r.get("selected"))
        return history
