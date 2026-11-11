"""Masquelier-style unsupervised temporal pattern learning.

Reproduces the paradigm of Masquelier, Guyonneau & Thorpe (2008), "Spike
Timing Dependent Plasticity Finds the Start of Repeating Patterns in
Continuous Spike Trains", PLoS ONE 3(1): e1377: one neuron listens to a
continuous stream from 2000 afferents, a 50 ms spike pattern carried by half
of them recurs at random times inside equally dense Poisson activity, and
nearest-spike additive STDP makes the neuron fire only for the pattern, a few
milliseconds after it starts.

Every constant below is the paper's (Methods). The paper prints its five
equations as images; the kernel and STDP forms used here are the standard
ones it cites (Gerstner SRM, exponential additive STDP) and should be
checked against the PDF:

    eps(s) = K (exp(-s/tau_m) - exp(-s/tau_s)),  K so that max eps = 1
    eta(s) = T (K1 exp(-s/tau_m) - K2 (exp(-s/tau_m) - exp(-s/tau_s)))
    dw     = +a_plus  exp(-(t_post - t_pre)/tau_plus)   pre before post
             -a_minus exp(-(t_pre - t_post)/tau_minus)  pre after post

"Nearest spike" is the reduced nearest-neighbour rule of the published
replication (Hathway & Goodman 2018, ReScience 4:6, whose Brian code has the
same kernels written as ODEs): a post spike potentiates only each afferent's
last pre spike SINCE the previous post spike, and a pre spike is depressed
only if it is the afferent's first since the last post spike, so LTP and LTD
strictly alternate per synapse.

Input is generated on the paper's 1 ms grid with exact spike times inside
each bin. The neuron runs at ``sim_dt_ms`` (0.1 ms) with the threshold tested
at the end of each step. The replication found that a 1 ms neuron step makes
every same-bin pre spike count as pre-before-post, so all weights potentiate
and no selectivity emerges.

Beyond the 2008 paper, :func:`make_pattern_stream` hides SEVERAL patterns,
each on its own afferent set, and :func:`run` takes ``n_out`` neurons that
compete through lateral inhibition: when one fires, every other neuron loses
the input it has integrated so far and its potential drops by ``inh_alpha`` x
threshold, decaying with ``tau_inh``; if several cross the threshold in one
step only the highest fires. That is the one-winner-take-all mechanism of
Masquelier, Guyonneau & Thorpe (2009, Neural Computation 21:1259). The
inhibition constants (0.4 x threshold, 20 ms, a 40 % pattern share and
patterns on random halves of the afferents for three neurons) are NOT taken
from that paper, which is not open access; they are free parameters.

Synapses are either ideal weights (``device=None``: the paper's additive rule
clipped to [0, 1]) or a :class:`neurovat.devices.DeviceArray` of shape
(n_out, n_afferents) programmed by the requested weight changes.
``write_window_ms`` accumulates the changes and writes each synapse once per
window; 0 writes every pairing, as the paper's rule does.
"""
from dataclasses import dataclass, field, asdict, replace
import math

import numpy as np


@dataclass
class MasquelierParams:
    # input (Methods, "Poisson spike trains")
    n_afferents: int = 2000
    n_pattern: int = 1000          # the first n_pattern afferents carry it
    base_s: float = 150.0          # generated once, repeated n_repeats times
    n_repeats: int = 3             # 3 x 150 s = the paper's 450 s
    dt_ms: float = 1.0             # input generation grid
    sim_dt_ms: float = 0.1         # neuron / threshold step
    r_max_hz: float = 90.0
    s_max_hz_s: float = 1800.0     # max |speed of rate change|
    ds_max_hz_s: float = 360.0     # per-step change of that speed
    max_silence_ms: float = 50.0   # force a spike after this much silence
    pattern_ms: float = 50.0
    pattern_frac: float = 0.25     # share of 50 ms sections holding a pattern
    jitter_ms: float = 1.0         # Gaussian sigma, applied at each paste
    spike_deletion: float = 0.0    # share of pattern spikes dropped per paste
    noise_hz: float = 10.0         # spontaneous activity on every afferent
    # neuron (SRM)
    tau_m: float = 10.0
    tau_s: float = 2.5
    threshold: float = 500.0
    K1: float = 2.0
    K2: float = 4.0
    t_ref_ms: float = 1.0
    cutoff_tau: float = 7.0        # kernels and STDP windows end at 7 tau
    # STDP
    a_plus: float = 0.03125
    a_minus_ratio: float = 0.85    # a_minus = 0.85 a_plus
    tau_plus: float = 16.8
    tau_minus: float = 33.7
    w_init: float = 0.475
    # population (not in the 2008 paper)
    n_out: int = 1
    inh_alpha: float = 0.4         # lateral inhibition, x threshold (free)
    tau_inh: float = 20.0          # its decay, ms (free)
    inh_reset: bool = True         # a spike clears the others' PSP sums
    seed: int = 1


@dataclass
class Stream:
    """Spikes of the base segment, grouped by 1 ms input step (CSR)."""
    ptr: np.ndarray                # (n_steps + 1,) offsets into aff / t
    aff: np.ndarray                # afferent index of each spike
    t: np.ndarray                  # spike time (ms) within the base segment
    label: np.ndarray              # (n_sections,) pattern id, -1 = none
    section_steps: int
    duration_ms: float
    rate_hz: float = field(default=0.0)
    sources: tuple = ()            # section each pattern was copied from
    members: np.ndarray = None     # (n_patterns, n_afferents) afferent sets
    copies: tuple = ()             # (afferents, times in ms) of each pattern

    @property
    def pattern_on(self):
        return self.label >= 0

    @property
    def source_section(self):
        return self.sources[0] if self.sources else -1

    @property
    def n_patterns(self):
        return 0 if self.members is None else len(self.members)


def _pattern_sections(n, frac, rng):
    """Boolean mask of n sections, about ``frac`` of them set, never two in a
    row (a two-state chain whose stationary share of set sections is frac)."""
    p = frac / (1.0 - frac)
    on = np.zeros(n, bool)
    for k in range(n):
        on[k] = (not (k and on[k - 1])) and rng.random() < p
    return on


def make_stream(P, rng):
    """The paper's input: variable-rate Poisson trains (0-90 Hz, a spike forced
    after 50 ms of silence), a 50 ms section of the first n_pattern trains
    copied with 1 ms jitter into about a quarter of the sections (never two
    consecutive), then 10 Hz spontaneous spikes on every afferent."""
    members = np.zeros((1, P.n_afferents), bool)
    members[0, :P.n_pattern] = True
    return make_pattern_stream(members, P, rng)


def random_members(n_patterns, P, rng):
    """Afferent set of each pattern: an independent random ``n_pattern`` of
    the ``n_afferents`` (sets overlap)."""
    members = np.zeros((n_patterns, P.n_afferents), bool)
    for p in range(n_patterns):
        members[p, rng.choice(P.n_afferents, P.n_pattern, replace=False)] = True
    return members


def make_pattern_stream(members, P, rng, copies=None):
    """The paper's input with one pattern per row of ``members`` (a boolean
    (n_patterns, n_afferents) array: the afferents that carry it). Each
    pattern is a frozen 50 ms section of its afferents' own background
    trains, so it has no rate cue. A pattern section shows one pattern,
    picked uniformly; pattern sections fill ``pattern_frac`` of the time and
    are never consecutive. ``copies`` (the ``copies`` of an earlier stream)
    hides those same patterns in fresh background activity, jitter and noise
    instead of freezing new ones: a test stream."""
    members = np.asarray(members, bool)
    n_pats, n = members.shape
    if n != P.n_afferents:
        raise ValueError(f"members has {n} afferents, params {P.n_afferents}")
    dt = P.dt_ms
    sec = int(round(P.pattern_ms / dt))
    n_sec = int(round(P.base_s * 1000.0 / P.pattern_ms))
    n_steps = n_sec * sec

    r = rng.uniform(0.0, P.r_max_hz, n)
    s = rng.uniform(-P.s_max_hz_s, P.s_max_hz_s, n)
    last = -rng.uniform(0.0, P.max_silence_ms, n)
    steps, affs, times = [], [], []
    for k in range(n_steps):
        t0 = k * dt
        fire = rng.random(n) < r * dt * 1e-3
        fire |= (t0 + dt - last) > P.max_silence_ms
        idx = np.nonzero(fire)[0]
        ts = t0 + rng.random(idx.size) * dt
        last[idx] = ts
        steps.append(np.full(idx.size, k, np.int32))
        affs.append(idx.astype(np.int32))
        times.append(ts)
        r = np.clip(r + s * dt * 1e-3, 0.0, P.r_max_hz)
        s = np.clip(s + rng.uniform(-P.ds_max_hz_s, P.ds_max_hz_s, n),
                    -P.s_max_hz_s, P.s_max_hz_s)
    step = np.concatenate(steps)
    aff = np.concatenate(affs)
    t = np.concatenate(times)
    sec_of = step // sec

    on = _pattern_sections(n_sec, P.pattern_frac, rng)
    label = np.full(n_sec, -1)
    on_idx = np.nonzero(on)[0]
    label[on_idx] = 0 if n_pats == 1 else rng.integers(0, n_pats, on_idx.size)
    # each pattern: one random section of its afferents' trains. It is drawn
    # from that pattern's own sections, so its original spikes are replaced
    # by a jittered copy like every other occurrence; drawn from all
    # sections, the untouched source would be one more, unlabelled and
    # unjittered, occurrence.
    sources, given, copies = [], copies, []
    for p in range(n_pats):
        mine = np.nonzero(label == p)[0]
        if given is not None:
            sources.append(-1)
            copies.append(given[p])
            continue
        if not mine.size:
            sources.append(-1)
            copies.append((aff[:0], t[:0]))
            continue
        src = int(rng.choice(mine))
        sel = (sec_of == src) & members[p][aff]
        sources.append(src)
        copies.append((aff[sel], t[sel] - src * P.pattern_ms))
    lab = label[sec_of]
    covered = lab >= 0
    covered[covered] = members[lab[covered], aff[covered]]
    keep = ~covered
    pa, pt = [aff[keep]], [t[keep]]
    for k in on_idx:
        p_aff, p_t = copies[label[k]]
        if P.spike_deletion > 0.0:
            kept = rng.random(p_t.size) >= P.spike_deletion
            p_aff, p_t = p_aff[kept], p_t[kept]
        jt = p_t + rng.normal(0.0, P.jitter_ms, p_t.size)
        jt = np.clip(jt, 0.0, P.pattern_ms - 1e-6) + k * P.pattern_ms
        pa.append(p_aff)
        pt.append(jt)
    aff, t = np.concatenate(pa), np.concatenate(pt)

    # spontaneous activity, everywhere
    n_noise = rng.poisson(P.noise_hz * n * P.base_s)
    aff = np.concatenate([aff, rng.integers(0, n, n_noise).astype(np.int32)])
    t = np.concatenate([t, rng.random(n_noise) * n_steps * dt])

    order = np.argsort(t, kind="stable")
    aff, t = aff[order], t[order]
    step = np.minimum((t // dt).astype(np.int64), n_steps - 1)
    ptr = np.searchsorted(step, np.arange(n_steps + 1))
    return Stream(ptr=ptr, aff=aff, t=t, label=label, section_steps=sec,
                  duration_ms=n_steps * dt,
                  rate_hz=aff.size / (n * P.base_s), sources=tuple(sources),
                  members=members, copies=tuple(copies))


class Synapses:
    """Ideal additive weights, or a device array driven by weight requests.

    A requested change dw becomes a pulse of amplitude
    ``pot_sign * pot_gain * dw`` (dw > 0) or ``pot_sign * dep_gain * dw``
    (dw < 0) and width ``pulse_width`` seconds; the device decides the actual
    change. ``energy`` (an :class:`neurovat.power.EnergyMeter`) records every
    pulse as crossbar 0 of ``phase``. ``weights`` are the initial ideal
    weights (default: all ``w_init``); a device starts in its own state."""

    def __init__(self, P, device=None, write_window_ms=0.0, write_gain=1.0,
                 dep_gain=None, pot_sign=1.0, pulse_width=None, energy=None,
                 phase="train", weights=None):
        self.P, self.device = P, device
        self.window = float(write_window_ms)
        self.pot_gain = float(write_gain)
        self.dep_gain = float(write_gain if dep_gain is None else dep_gain)
        self.pot_sign = float(pot_sign)
        self.width = float(pulse_width if pulse_width is not None
                           else getattr(device, "pulse_width", 1e-3))
        self.energy, self.phase = energy, phase
        self.writes = 0
        shape = (P.n_out, P.n_afferents)
        if device is None:
            self.w = (np.full(shape, P.w_init) if weights is None else
                      np.array(weights, float).reshape(shape))
        else:
            if tuple(device.shape) != shape:
                raise ValueError(f"device shape {device.shape} != {shape}")
            self.g_lo, self.g_hi = float(device.g_min), float(device.g_max)
            self._read()
        self.pending = np.zeros(shape)
        self._next_flush = self.window

    def conductance(self):
        return None if self.device is None else np.asarray(
            self.device.conductance(), float)

    def _read(self):
        G = self.conductance()
        self.w = np.clip((G - self.g_lo) / (self.g_hi - self.g_lo), 0.0, 1.0)

    def update(self, rows, cols, dw, t_ms):
        if self.device is None:
            self.w[rows, cols] = np.clip(self.w[rows, cols] + dw, 0.0, 1.0)
            self.writes += int(np.size(cols))
            return
        np.add.at(self.pending, (rows, cols), dw)
        if self.window <= 0.0:
            self.flush(t_ms)

    def flush(self, t_ms):
        rows, cols = np.nonzero(self.pending)
        if not rows.size:
            return
        dw = self.pending[rows, cols]
        amps = self.pot_sign * np.where(dw > 0, self.pot_gain, self.dep_gain) * dw
        g0 = self.conductance()[rows, cols] if self.energy is not None else None
        current = None
        if self.energy is not None and hasattr(self.device, "write_current"):
            current = self.device.write_current(rows, cols, amps, self.width)
        self.device.program(rows, cols, amps, self.width, t_ms * 1e-3)
        self.pending[rows, cols] = 0.0
        self.writes += int(rows.size)
        self._read()
        if self.energy is not None:
            self.energy.record_write(self.phase, 0, rows, cols, amps,
                                     self.width, g0,
                                     self.conductance()[rows, cols], current)

    def tick(self, t_ms):
        if self.device is not None and self.window > 0.0 and t_ms >= self._next_flush:
            self.flush(t_ms)
            self._next_flush = t_ms + self.window


def _section_rasters(S, a, b, off_in, n_bins, bin_ms, posts, n_out):
    """Input and output spike counts of one section, binned at ``bin_ms``."""
    pre = np.zeros((n_bins, S.members.shape[1] if S.members is not None
                    else int(S.aff.max()) + 1))
    if b > a:
        k = np.minimum(((S.t[a:b] - off_in) / bin_ms).astype(int), n_bins - 1)
        np.add.at(pre, (k, S.aff[a:b]), 1.0)
    post = np.zeros((n_bins, n_out))
    for t_rel, j in posts:
        post[min(int(t_rel / bin_ms), n_bins - 1), j] += 1.0
    return pre, post


def run(P=None, device=None, write_window_ms=0.0, write_gain=1.0,
        stream=None, progress=None, members=None, dep_gain=None, pot_sign=1.0,
        pulse_width=None, energy=None, phase="train", on_section=None,
        learn=True, weights=None):
    """Simulate one run. Returns a dict with every postsynaptic spike (time,
    neuron, pattern it fell in or -1, latency after that pattern's onset),
    the final weights and the success metrics of the paper.

    ``members`` gives the afferent set of each pattern (see
    :func:`make_pattern_stream`); default: the 2008 paper's single pattern on
    the first ``n_pattern`` afferents. ``on_section(t_ms, info)`` is called
    after every 50 ms section with that section's pattern (``label``, -1 =
    none) and output spikes (``spikes``: (ms after its start, neuron));
    returning False stops the run early (the result then covers the time
    simulated). ``energy`` records reads, neuron spikes and writes per
    section and needs a ``device``. ``learn=False`` freezes the weights
    (evaluation); ``weights`` sets the initial ideal weights."""
    P = P or MasquelierParams()
    rng = np.random.default_rng(P.seed)
    if stream is not None:
        S = stream
    elif members is not None:
        S = make_pattern_stream(members, P, rng)
    else:
        S = make_stream(P, rng)
    syn = Synapses(P, device, write_window_ms, write_gain, dep_gain, pot_sign,
                   pulse_width, energy, phase, weights)
    if device is None:
        energy = None
    dt = P.sim_dt_ms
    n_base = int(round(S.duration_ms / dt))
    k_of = np.minimum((S.t / dt).astype(np.int64), n_base - 1)
    ptr = np.searchsorted(k_of, np.arange(n_base + 1))
    sec_sim = int(round(P.pattern_ms / dt))
    tm, ts = P.tau_m, P.tau_s
    s_pk = tm * ts / (tm - ts) * math.log(tm / ts)
    K = 1.0 / (math.exp(-s_pk / tm) - math.exp(-s_pk / ts))
    a_minus = P.a_minus_ratio * P.a_plus
    ltp_win, ltd_win = P.cutoff_tau * P.tau_plus, P.cutoff_tau * P.tau_minus
    eta_win = P.cutoff_tau * tm
    dm, ds = math.exp(-dt / tm), math.exp(-dt / ts)
    di = math.exp(-dt / P.tau_inh)
    thr = P.threshold
    bin_ms = (energy.dt_s * 1e3) if energy is not None else None
    n_bins = int(round(P.pattern_ms / bin_ms)) if energy is not None else 0

    n, m = P.n_afferents, P.n_out
    rows_all = np.arange(m)
    last_pre = np.full(n, -np.inf)
    ltp_open = np.zeros((m, n), bool)      # a pre spike since the last post
    ltd_open = np.zeros((m, n), bool)      # no pre spike yet since the last post
    Am = np.zeros(m)                       # sum w exp(-(t-tj)/tau) since post
    As = np.zeros(m)
    inh = np.zeros(m)                      # lateral inhibition (potential units)
    t_post = np.full(m, -np.inf)
    post_t, post_j, post_lab, post_lat = [], [], [], []
    sec_posts = []
    stopped = False
    for rep in range(P.n_repeats):
        off = rep * S.duration_ms
        for k in range(n_base):
            t_end = off + (k + 1) * dt
            Am *= dm
            As *= ds
            if m > 1:
                inh *= di
            a, b = ptr[k], ptr[k + 1]
            if b > a:
                idx = S.aff[a:b]
                tj = S.t[a:b] + off
                if learn:
                    lt = ltd_open[:, idx]
                    if lt.any():
                        jj, ii = np.nonzero(lt)
                        d = tj[ii] - t_post[jj]
                        sel = d <= ltd_win
                        if sel.any():
                            syn.update(jj[sel], idx[ii[sel]],
                                       -a_minus * np.exp(-d[sel] / P.tau_minus),
                                       t_end)
                        ltd_open[:, idx] = False
                    ltp_open[:, idx] = True
                last_pre[idx] = tj
                W = syn.w[:, idx]
                Am += W @ np.exp(-(t_end - tj) / tm)
                As += W @ np.exp(-(t_end - tj) / ts)
            syn.tick(t_end)
            sp = t_end - t_post
            ready = sp > P.t_ref_ms
            if ready.any():
                u = K * (Am - As) - inh
                near = sp < eta_win
                if near.any():
                    em, es = np.exp(-sp[near] / tm), np.exp(-sp[near] / ts)
                    u[near] += thr * (P.K1 * em - P.K2 * (em - es))
                cand = ready & (u >= thr)
                if cand.any():
                    j = int(np.argmax(np.where(cand, u, -np.inf)))
                    if learn:
                        ii = np.nonzero(ltp_open[j])[0]
                        d = t_end - last_pre[ii]
                        sel = d <= ltp_win
                        if sel.any():
                            syn.update(np.full(int(sel.sum()), j), ii[sel],
                                       P.a_plus * np.exp(-d[sel] / P.tau_plus),
                                       t_end)
                        ltp_open[j] = False
                        ltd_open[j] = True
                    Am[j] = As[j] = 0.0
                    t_post[j] = t_end
                    if m > 1:
                        others = rows_all != j
                        inh[others] += P.inh_alpha * thr
                        if P.inh_reset:
                            Am[others] = As[others] = 0.0
                    t_in = t_end - off
                    sec_k = min(int(t_in // P.pattern_ms), S.label.size - 1)
                    lab = int(S.label[sec_k])
                    post_t.append(t_end)
                    post_j.append(j)
                    post_lab.append(lab)
                    post_lat.append(t_in - sec_k * P.pattern_ms if lab >= 0
                                    else np.nan)
                    sec_posts.append((t_in - sec_k * P.pattern_ms, j))
            if (k + 1) % sec_sim == 0:
                sk = (k + 1) // sec_sim - 1
                if energy is not None:
                    a0 = S.ptr[sk * S.section_steps]
                    b0 = S.ptr[(sk + 1) * S.section_steps]
                    pre, post = _section_rasters(S, a0, b0, sk * P.pattern_ms,
                                                 n_bins, bin_ms, sec_posts, m)
                    G = syn.conductance()
                    energy.record_forward(phase, [G], [pre, post])
                spikes, sec_posts = sec_posts, []
                if on_section is not None and on_section(t_end, {
                        "rep": rep, "section": sk, "label": int(S.label[sk]),
                        "spikes": spikes, "post_spikes": len(post_t),
                        "weights": syn.w}) is False:
                    stopped = True
                    break
        if progress:
            progress(rep + 1, P.n_repeats)
        if stopped:
            break
    t_done = t_end
    if device is not None:
        syn.flush(t_done)
    res = _score(P, S, np.array(post_t), np.array(post_j, int),
                 np.array(post_lab, int), np.array(post_lat, float), syn,
                 t_done)
    res["stopped"] = stopped
    return res


def pulse_step(device, amp, width, w):
    """Weight change (fraction of the conductance range) that one pulse of
    ``amp`` and ``width`` seconds makes on a fresh nominal device placed at
    weight ``w``."""
    lo, hi = float(device.g_min), float(device.g_max)
    probe = device.nominal()
    probe.set_conductance(np.full((1, 1), lo + w * (hi - lo)))
    g0 = float(probe.conductance()[0, 0])
    probe.program(np.array([0]), np.array([0]), np.array([float(amp)]), width)
    return (float(probe.conductance()[0, 0]) - g0) / (hi - lo)


def calibrate_write(device, P, width, amp0=1.0, tol=0.02):
    """Pulse amplitudes at which one pulse on a device at ``P.w_init`` equals
    one STDP step: ``+a_plus`` (potentiation) and ``-a_minus`` (depression).

    Searched by bisection within 1e-4 .. 1e4 x ``amp0`` (the device's usual
    drive level, in its drive unit). Returns the ``write_gain``, ``dep_gain``,
    ``pot_sign`` and ``pulse_width`` arguments of :func:`run` plus the
    amplitudes and steps found. A larger or smaller request is scaled
    linearly from there; the device decides what it does with it. Raises
    ValueError when a step is out of reach in that range."""
    sign = float(device.potentiation_sign(amp0, width))
    out = {"pot_sign": sign, "pulse_width": float(width)}
    a_minus = P.a_minus_ratio * P.a_plus
    for name, direction, target in (("pot", 1.0, P.a_plus),
                                    ("dep", -1.0, a_minus)):
        if target <= 0.0:                   # that direction is switched off
            out[name + "_amp"], out[name + "_step"] = 0.0, 0.0
            continue

        def step(amp):
            return abs(pulse_step(device, direction * sign * amp, width,
                                  P.w_init))
        lo, hi = abs(amp0) * 1e-4, abs(amp0) * 1e4
        if step(hi) < target or step(lo) > target:
            raise ValueError(
                f"one {name} pulse cannot make a step of {target:.4g}: "
                f"{step(lo):.3g} at {lo:.3g}, {step(hi):.3g} at {hi:.3g}")
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            got = step(mid)
            if abs(got - target) <= tol * target:
                break
            lo, hi = (mid, hi) if got < target else (lo, mid)
        out[name + "_amp"], out[name + "_step"] = mid, got
    out["write_gain"] = out["pot_amp"] / P.a_plus if P.a_plus > 0 else 0.0
    out["dep_gain"] = out["dep_amp"] / a_minus if a_minus > 0 else 0.0
    return out


class Session:
    """One population that is trained and then tested on the same hidden
    patterns. Every :meth:`run` generates a fresh stream (new background
    activity, jitter and noise) around the patterns frozen by the first one
    and carries the weights over, so ``run(learn=False)`` after training is a
    test on unseen input. ``jitter_ms``, ``noise_hz``, ``spike_deletion`` and
    ``pattern_frac`` may differ from run to run.

    With one pattern its afferents are the first ``n_pattern`` (the 2008
    layout), otherwise an independent random set per pattern. A ``device`` is
    placed at ``w_init`` of its conductance range first; that raises
    ValueError for a device that cannot be set by conductance. ``synapse_kw``
    goes to :func:`run` (``write_window_ms``, ``write_gain``, ``dep_gain``,
    ``pot_sign``, ``pulse_width``, ``energy``)."""

    def __init__(self, P, n_patterns=1, device=None, **synapse_kw):
        self.P, self.device, self.kw = P, device, synapse_kw
        if n_patterns == 1:
            self.members = np.zeros((1, P.n_afferents), bool)
            self.members[0, :P.n_pattern] = True
        else:
            self.members = random_members(n_patterns, P,
                                          np.random.default_rng([P.seed, 7]))
        self.copies, self.weights, self.runs = None, None, 0
        if device is not None:
            lo, hi = float(device.g_min), float(device.g_max)
            device.set_conductance(np.full(device.shape,
                                           lo + P.w_init * (hi - lo)))

    def run(self, seconds, learn=True, on_section=None, **changes):
        """Simulate ``seconds`` of stream; ``changes`` override input
        parameters for this run only. Returns the dict of :func:`run`."""
        P = replace(self.P, base_s=float(seconds), n_repeats=1, **changes)
        S = make_pattern_stream(self.members, P, np.random.default_rng(
            [P.seed, 11, self.runs]), self.copies)
        res = run(P, device=self.device, stream=S, learn=learn,
                  weights=self.weights, on_section=on_section,
                  phase="train" if learn else "eval", **self.kw)
        self.copies, self.runs = S.copies, self.runs + 1
        if self.device is None:
            self.weights = np.array(res["weights"], float)
        return res


def _score(P, S, post_t, post_j, post_lab, lat, syn, t_done, last_s=150.0):
    """The paper's success test on the final ``last_s`` seconds (or the last
    third of a shorter run): a hit rate above 98 %, no false alarm and a mean
    latency below 10 ms. Reported for neuron 0 / pattern 0 (the 2008 paper)
    and, per neuron, for the pattern it detects best."""
    window = min(last_s * 1000.0, t_done / 3.0)
    t0 = t_done - window
    starts, slab = [], []
    n_rep = int(math.ceil(t_done / S.duration_ms))
    for r in range(n_rep):
        k = np.nonzero(S.label >= 0)[0]
        st = k * P.pattern_ms + r * S.duration_ms
        ok = (st >= t0) & (st + P.pattern_ms <= t_done)
        starts.append(st[ok])
        slab.append(S.label[k][ok])
    starts = np.concatenate(starts) if starts else np.zeros(0)
    slab = np.concatenate(slab).astype(int) if slab else np.zeros(0, int)
    late = post_t >= t0
    n_pats = max(S.n_patterns, 1)

    per = []
    for j in range(P.n_out):
        mine = late & (post_j == j)
        rows = []
        for p in range(n_pats):
            occ = starts[slab == p]
            sel = mine & (post_lab == p)
            onset = np.round(post_t[sel] - lat[sel], 6)
            hits = np.isin(np.round(occ, 6), onset)
            lp = lat[sel]
            rows.append({"pattern": p,
                         "hit_rate": float(hits.mean()) if occ.size else 0.0,
                         "occurrences": int(occ.size),
                         "mean_latency_ms": float(lp.mean()) if lp.size else float("nan")})
        fa = int((mine & (post_lab < 0)).sum())
        best = max(rows, key=lambda r: r["hit_rate"])
        other = int(sum(((mine & (post_lab == r["pattern"])).sum())
                        for r in rows if r is not best))
        wj = syn.w[j]
        strong = wj > 0.9
        in_set = (S.members[best["pattern"]] if S.members is not None
                  else np.zeros(P.n_afferents, bool))
        ok = (best["hit_rate"] > 0.98 and fa == 0
              and best["mean_latency_ms"] < 10.0)
        per.append({"neuron": j, "best_pattern": best["pattern"],
                    "hit_rate": best["hit_rate"], "false_alarms": fa,
                    "other_pattern_spikes": other,
                    "mean_latency_ms": best["mean_latency_ms"],
                    "potentiated": int(strong.sum()),
                    "potentiated_in_pattern": int((strong & in_set).sum()),
                    "success": bool(ok), "patterns": rows})

    # mean output spikes per 50 ms window: neuron x (each pattern, no pattern)
    n_noise = max(window / P.pattern_ms - starts.size, 1.0)
    table = np.zeros((P.n_out, n_pats + 1))
    for j in range(P.n_out):
        mine = late & (post_j == j)
        for p in range(n_pats):
            table[j, p] = (mine & (post_lab == p)).sum() / max((slab == p).sum(), 1)
        table[j, n_pats] = (mine & (post_lab < 0)).sum() / n_noise
    # one neuron per pattern, best hit rate first; a pattern counts as learnt
    # at a hit rate of 90 % or more by a neuron that is quiet elsewhere (at
    # most 0.05 spikes per window of any other pattern or of background) - a
    # neuron that fires all the time hits every pattern
    hits = np.array([[r["hit_rate"] for r in q["patterns"]] for q in per])
    owner, free = {}, hits.copy()
    for j in range(P.n_out):
        for p in range(n_pats):
            if np.delete(table[j], p).max() > 0.05:
                free[j, p] = -1.0
    for _ in range(min(P.n_out, n_pats)):
        j, p = np.unravel_index(np.argmax(free), free.shape)
        if free[j, p] < 0.9:
            break
        owner[int(p)] = int(j)
        free[j, :] = -1.0
        free[:, p] = -1.0
    fa_per_s = float((late & (post_lab < 0)).sum() / (window / 1000.0))

    first = per[0]
    p0 = first["patterns"][0]
    w = syn.w
    strong, weak = w > 0.9, w < 0.1
    success0 = (p0["hit_rate"] > 0.98 and first["false_alarms"] == 0
                and p0["mean_latency_ms"] < 10.0)
    covered = sorted({r["best_pattern"] for r in per if r["success"]})
    return {
        "params": asdict(P),
        "input_rate_hz": S.rate_hz,
        "pattern_occurrences": int((S.label >= 0).sum() * n_rep),
        "post_spikes": int(post_t.size),
        "simulated_s": t_done / 1000.0,
        "score_window_s": window / 1000.0,
        "hit_rate_last": p0["hit_rate"],
        "false_alarms_last": first["false_alarms"],
        "mean_latency_last_ms": p0["mean_latency_ms"],
        "potentiated": int(strong[0].sum()),
        "potentiated_in_pattern": int(strong[0][:P.n_pattern].sum())
        if S.members is None or S.n_patterns == 1 else first["potentiated_in_pattern"],
        "bimodal_frac": float((strong | weak).mean()),
        "writes": syn.writes,
        "success": bool(success0),
        "per_neuron": per,
        "patterns_learnt": covered,
        "pattern_owner": owner,
        "spikes_per_window": table,
        "false_alarms_per_s": fa_per_s,
        "post_t_ms": post_t,
        "post_neuron": post_j,
        "post_pattern": post_lab,
        "post_latency_ms": lat,
        "weights": w[0].copy() if P.n_out == 1 else w.copy(),
    }
