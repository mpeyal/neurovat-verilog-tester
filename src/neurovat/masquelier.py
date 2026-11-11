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

Synapses are either ideal weights (``device=None``: the paper's additive rule
clipped to [0, 1]) or a :class:`neurovat.devices.DeviceArray` of shape
(1, n_afferents) programmed by the requested weight changes. ``write_window_ms``
accumulates the changes and writes each synapse once per window; 0 writes
every pairing, as the paper's rule does.
"""
from dataclasses import dataclass, field, asdict
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
    pattern_frac: float = 0.25     # share of 50 ms sections holding the pattern
    jitter_ms: float = 1.0         # Gaussian sigma, applied at each paste
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
    seed: int = 1


@dataclass
class Stream:
    """Spikes of the base segment, grouped by time step (CSR)."""
    ptr: np.ndarray                # (n_steps + 1,) offsets into aff / t
    aff: np.ndarray                # afferent index of each spike
    t: np.ndarray                  # spike time (ms) within the base segment
    pattern_on: np.ndarray         # (n_sections,) section holds the pattern
    section_steps: int
    duration_ms: float
    rate_hz: float = field(default=0.0)
    source_section: int = -1       # the section the pattern was copied from


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
    dt = P.dt_ms
    n, n_pat = P.n_afferents, P.n_pattern
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

    # the pattern: one random section of the first n_pat trains. It is drawn
    # from the chosen sections, so its original spikes are themselves
    # replaced by a jittered copy; drawn from all sections, the untouched
    # source would be one more, unlabelled and unjittered, occurrence.
    on = _pattern_sections(n_sec, P.pattern_frac, rng)
    src = int(rng.choice(np.nonzero(on)[0]))
    sel = (step // sec == src) & (aff < n_pat)
    pat_aff, pat_t = aff[sel], t[sel] - src * P.pattern_ms
    keep = ~((aff < n_pat) & on[step // sec])
    step, aff, t = step[keep], aff[keep], t[keep]
    pa, pt = [aff], [t]
    for k in np.nonzero(on)[0]:
        jt = pat_t + rng.normal(0.0, P.jitter_ms, pat_t.size)
        jt = np.clip(jt, 0.0, P.pattern_ms - 1e-6) + k * P.pattern_ms
        pa.append(pat_aff)
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
    return Stream(ptr=ptr, aff=aff, t=t, pattern_on=on, section_steps=sec,
                  duration_ms=n_steps * dt,
                  rate_hz=aff.size / (n * P.base_s), source_section=src)


class Synapses:
    """Ideal additive weights, or a device array driven by weight requests."""

    def __init__(self, P, device=None, write_window_ms=0.0, write_gain=1.0):
        self.P, self.device = P, device
        self.window = float(write_window_ms)
        self.gain = float(write_gain)
        self.writes = 0
        n = P.n_afferents
        if device is None:
            self.w = np.full(n, P.w_init)
        else:
            if device.shape != (1, n):
                raise ValueError(f"device shape {device.shape} != (1, {n})")
            self.g_lo, self.g_hi = float(device.g_min), float(device.g_max)
            self._read()
        self.pending = np.zeros(n)
        self._next_flush = self.window

    def _read(self):
        G = np.asarray(self.device.conductance(), float)[0]
        self.w = np.clip((G - self.g_lo) / (self.g_hi - self.g_lo), 0.0, 1.0)

    def update(self, idx, dw, t_ms):
        if self.device is None:
            self.w[idx] = np.clip(self.w[idx] + dw, 0.0, 1.0)
            self.writes += int(np.size(idx))
            return
        np.add.at(self.pending, idx, dw)
        if self.window <= 0.0:
            self.flush(t_ms)

    def flush(self, t_ms):
        idx = np.nonzero(self.pending)[0]
        if idx.size:
            amps = self.gain * self.pending[idx]
            self.device.program(np.zeros(idx.size, int), idx, amps,
                                getattr(self.device, "pulse_width", 1e-3),
                                t_ms * 1e-3)
            self.pending[idx] = 0.0
            self.writes += int(idx.size)
            self._read()

    def tick(self, t_ms):
        if self.device is not None and self.window > 0.0 and t_ms >= self._next_flush:
            self.flush(t_ms)
            self._next_flush = t_ms + self.window


def run(P=None, device=None, write_window_ms=0.0, write_gain=1.0,
        stream=None, progress=None):
    """Simulate one run. Returns a dict with every postsynaptic spike (time,
    latency after pattern onset or None for a false alarm), the final
    weights, per-occurrence hits and the success metrics of the paper."""
    P = P or MasquelierParams()
    rng = np.random.default_rng(P.seed)
    S = stream or make_stream(P, rng)
    syn = Synapses(P, device, write_window_ms, write_gain)
    dt = P.sim_dt_ms
    n_base = int(round(S.duration_ms / dt))
    k_of = np.minimum((S.t / dt).astype(np.int64), n_base - 1)
    ptr = np.searchsorted(k_of, np.arange(n_base + 1))
    tm, ts = P.tau_m, P.tau_s
    s_pk = tm * ts / (tm - ts) * math.log(tm / ts)
    K = 1.0 / (math.exp(-s_pk / tm) - math.exp(-s_pk / ts))
    a_minus = P.a_minus_ratio * P.a_plus
    ltp_win, ltd_win = P.cutoff_tau * P.tau_plus, P.cutoff_tau * P.tau_minus
    eta_win = P.cutoff_tau * tm
    dm, ds = math.exp(-dt / tm), math.exp(-dt / ts)

    n = P.n_afferents
    last_pre = np.full(n, -np.inf)
    ltp_open = np.zeros(n, bool)           # a pre spike since the last post
    ltd_open = np.zeros(n, bool)           # no pre spike yet since the last post
    Am = As = 0.0                          # sum w exp(-(t-tj)/tau) since post
    t_post = -np.inf
    post_t, post_lat = [], []
    for rep in range(P.n_repeats):
        off = rep * S.duration_ms
        for k in range(n_base):
            t_end = off + (k + 1) * dt
            Am *= dm
            As *= ds
            a, b = ptr[k], ptr[k + 1]
            if b > a:
                idx = S.aff[a:b]
                tj = S.t[a:b] + off
                lt = ltd_open[idx]
                if lt.any():
                    i2, t2 = idx[lt], tj[lt]
                    d = t2 - t_post
                    m = d <= ltd_win
                    if m.any():
                        syn.update(i2[m], -a_minus * np.exp(-d[m] / P.tau_minus),
                                   t_end)
                    ltd_open[i2] = False
                last_pre[idx] = tj
                ltp_open[idx] = True
                w = syn.w[idx]
                Am += float(np.sum(w * np.exp(-(t_end - tj) / tm)))
                As += float(np.sum(w * np.exp(-(t_end - tj) / ts)))
            syn.tick(t_end)
            sp = t_end - t_post
            if sp <= P.t_ref_ms:
                continue
            u = K * (Am - As)
            if sp < eta_win:
                em, es = math.exp(-sp / tm), math.exp(-sp / ts)
                u += P.threshold * (P.K1 * em - P.K2 * (em - es))
            if u < P.threshold:
                continue
            # postsynaptic spike: LTP on each afferent's last pre spike since
            # the previous post spike
            ii = np.nonzero(ltp_open)[0]
            d = t_end - last_pre[ii]
            m = d <= ltp_win
            if m.any():
                syn.update(ii[m], P.a_plus * np.exp(-d[m] / P.tau_plus), t_end)
            ltp_open[:] = False
            ltd_open[:] = True
            Am = As = 0.0
            t_post = t_end
            t_in = t_end - off
            sec_k = min(int(t_in // P.pattern_ms), S.pattern_on.size - 1)
            post_t.append(t_end)
            post_lat.append(t_in - sec_k * P.pattern_ms
                            if S.pattern_on[sec_k] else None)
        if progress:
            progress(rep + 1, P.n_repeats)
    if device is not None:
        syn.flush(P.n_repeats * S.duration_ms)
    return _score(P, S, np.array(post_t), post_lat, syn)


def _score(P, S, post_t, post_lat, syn, last_s=150.0):
    """The paper's success test on the final ``last_s`` seconds: a hit rate
    above 98 %, no false alarm and a mean latency below 10 ms."""
    total = P.n_repeats * S.duration_ms
    t0 = total - last_s * 1000.0
    starts = np.concatenate([np.nonzero(S.pattern_on)[0] * P.pattern_ms
                             + r * S.duration_ms for r in range(P.n_repeats)])
    lat = np.array([np.nan if v is None else v for v in post_lat])
    in_pat = ~np.isnan(lat)
    late = post_t >= t0
    occ = starts[starts >= t0]
    hit_starts = post_t[in_pat & late] - lat[in_pat & late]
    hits = np.isin(np.round(occ, 6), np.round(hit_starts, 6))
    hit_rate = float(hits.mean()) if occ.size else 0.0
    false_alarms = int((~in_pat & late).sum())
    lat_late = lat[in_pat & late]
    mean_lat = float(lat_late.mean()) if lat_late.size else float("nan")
    w = syn.w
    strong = w > 0.9
    weak = w < 0.1
    success = hit_rate > 0.98 and false_alarms == 0 and mean_lat < 10.0
    return {
        "params": asdict(P),
        "input_rate_hz": S.rate_hz,
        "pattern_occurrences": int(starts.size),
        "post_spikes": int(post_t.size),
        "hit_rate_last": hit_rate,
        "false_alarms_last": false_alarms,
        "mean_latency_last_ms": mean_lat,
        "potentiated": int(strong.sum()),
        "potentiated_in_pattern": int(strong[:P.n_pattern].sum()),
        "bimodal_frac": float((strong | weak).mean()),
        "writes": syn.writes,
        "success": bool(success),
        "post_t_ms": post_t,
        "post_latency_ms": lat,
        "weights": w.copy(),
    }
