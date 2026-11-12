"""Spike encoding front-end: rate (Poisson), latency, temporal (phase) and
column-scan (spatio-temporal) codes, with sensor noise, background rate, a
signal/noise afferent split, temporal embedding and spike jitter."""
import numpy as np


def input_size(cfg):
    """Input lines of the network: one per pixel, or one per pixel ROW for
    the column-scan code."""
    return cfg.grid_h if cfg.encoding == "scan" else cfg.grid_h * cfg.grid_w


def scan_bin_steps(cfg):
    """Time steps per column of the column-scan code. Raises ValueError when
    the sweep over all columns does not fit in one presentation."""
    steps = max(1, int(round(cfg.scan_bin_ms / cfg.dt_ms)))
    n_steps = max(1, int(round(cfg.present_ms / cfg.dt_ms)))
    if cfg.grid_w * steps > n_steps:
        raise ValueError(
            f"column scan: {cfg.grid_w} columns x {steps * cfg.dt_ms:g} ms = "
            f"{cfg.grid_w * steps * cfg.dt_ms:g} ms does not fit in the "
            f"{cfg.present_ms:g} ms presentation; lengthen it or shorten the bin")
    return steps


def scan_events(image, bin_ms, threshold=0.5):
    """Column-scan code of a 2-D image as events: (rows, times in ms). Pixel
    (r, c) at or above ``threshold`` is one spike on input line r at the
    start of time bin c. The same code as ``encoding = "scan"``, without the
    presentation window."""
    rows, cols = np.nonzero(np.asarray(image, float) >= threshold)
    return rows.astype(np.int32), cols * float(bin_ms)


def _scan(vec01, cfg, rng, signal_mask, noise):
    """Column-scan code: the image is swept left to right, one pixel column
    per time bin. Input line r is pixel row r; it fires once, at the start of
    bin c, when pixel (r, c) is inked. The spike count of a clean pattern is
    its number of inked pixels. ``max_rate_hz`` and ``pattern_ms`` play no
    part; sensor noise, the signal mask (per row), background rate and jitter
    apply as for the other codes."""
    dt = cfg.dt_ms
    n_steps = max(1, int(round(cfg.present_ms / dt)))
    steps = scan_bin_steps(cfg)
    img = np.clip(np.asarray(vec01, float), 0.0, 1.0).reshape(cfg.grid_h,
                                                              cfg.grid_w)
    U = rng.random((n_steps, cfg.grid_h))       # shared base random field
    if noise and cfg.input_noise > 0:
        img = img + rng.normal(0.0, cfg.input_noise, img.shape)
        sp = rng.random(img.shape) < (cfg.input_noise * 0.25)
        img[sp] = (rng.random(int(sp.sum())) < 0.5).astype(float)
        img = np.clip(img, 0.0, 1.0)
    ink = img >= cfg.scan_threshold
    if noise and signal_mask is not None:
        ink &= np.asarray(signal_mask)[:, None] > 0
    S = np.zeros((n_steps, cfg.grid_h), bool)
    rows, cols = np.nonzero(ink)
    S[cols * steps, rows] = True
    if noise and cfg.bg_rate_hz > 0:
        S |= U < (1.0 - np.exp(-cfg.bg_rate_hz * dt * 1e-3))
    if noise and cfg.jitter_ms > 0:
        ti, ii = np.nonzero(S)
        if ti.size:
            shift = np.round(
                rng.normal(0.0, cfg.jitter_ms / dt, ti.size)).astype(int)
            nt = np.clip(ti + shift, 0, n_steps - 1)
            S = np.zeros_like(S)
            S[nt, ii] = True
    return S


def encode_spikes(vec01, cfg, rng, signal_mask=None, noise=True):
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
    if cfg.encoding == "scan":
        return _scan(vec01, cfg, rng, signal_mask, noise)
    dt = cfg.dt_ms
    n_steps = max(1, int(round(cfg.present_ms / dt)))
    vec = np.clip(np.asarray(vec01, float), 0.0, 1.0)
    n_in = vec.shape[0]
    U = rng.random((n_steps, n_in))             # shared base random field
    if noise and cfg.input_noise > 0:
        vec = vec + rng.normal(0.0, cfg.input_noise, vec.shape)
        sp = rng.random(vec.shape) < (cfg.input_noise * 0.25)
        vec[sp] = (rng.random(int(sp.sum())) < 0.5).astype(float)
        vec = np.clip(vec, 0.0, 1.0)
    mask = signal_mask if (noise and signal_mask is not None) else np.ones(n_in)
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
    if cfg.encoding == "temporal":
        # PHASE code: each active afferent fires one PRECISELY-TIMED spike per
        # cycle, at a phase set by its intensity (bright pixels fire early in
        # the cycle, dim ones late). The repeating, deterministic spike TIMES
        # carry the pattern - a true temporal code, distinct from stochastic
        # rate coding and single-shot TTFS latency.
        period = max(2, int(round((1000.0 / max(cfg.max_rate_hz, 1.0)) / dt)))
        phase = np.rint((1.0 - vec) * (period - 1) * 0.9).astype(int)
        if noise and 0.0 < pat_ms < cfg.present_ms:      # temporal embedding
            ps = max(1, int(round(pat_ms / dt)))
            on0 = max(0, (n_steps - ps) // 2)
            on1 = on0 + ps
        else:
            on0, on1 = 0, n_steps
        S = np.zeros((n_steps, n_in), bool)
        for i in np.nonzero(vec > 0.05)[0]:              # active afferents only
            ts = np.arange(on0 + int(phase[i]), on1, period)
            ts = ts[(ts >= 0) & (ts < n_steps)]
            S[ts, i] = True
        if bg > 0:                          # background dark-count on all inputs
            S |= (U < (1.0 - np.exp(-bg * dt * 1e-3)))
    else:
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
