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


# ---- image codes for the event stream (neurovat.masquelier.Session) --------
COLUMN_SCAN = "column scan"
ROW_SCAN = "row scan"
LATENCY = "latency"
FROZEN_POISSON = "frozen Poisson"
STREAM_CODES = (COLUMN_SCAN, ROW_SCAN, LATENCY, FROZEN_POISSON)
# What one input line is: "row" = pixel row r, "col" = pixel column c,
# "pixel" = one line per pixel, index r * W + c (silent pixels included).
STREAM_LAYOUT = {COLUMN_SCAN: "row", ROW_SCAN: "col", LATENCY: "pixel",
                 FROZEN_POISSON: "pixel"}
# Free parameters of the two pixel codes (not from any paper):
#   window_ms  the span the code uses: 50 ms (the Masquelier pattern length)
#   margin_ms  silence before AND after that span, 15 ms = 3 sigma of the
#              largest jitter of the robustness figure (5 ms). The stream
#              clips jittered spikes into [0, pattern_ms); without a margin
#              a near-binary image's volley (all spikes at t = 0) would lose
#              half its jitter onto the section start.
#   rate_hz    frozen Poisson: rate of a pixel of intensity 1 (100 Hz, so a
#              full pixel fires 5 spikes on average in 50 ms)
#   seed       frozen Poisson: the spikes are drawn ONCE (default seed 0) and
#              the stream pastes that fixed copy at every presentation
STREAM_WINDOW_MS = 50.0
STREAM_MARGIN_MS = 15.0
STREAM_RATE_HZ = 100.0


def stream_code(name):
    """Normalise an encoding name ('column', 'row', 'latency' / 'ttfs',
    'poisson' / 'rate'); ValueError for anything else."""
    s = str(name or COLUMN_SCAN).strip().lower()
    for code, keys in ((COLUMN_SCAN, ("column", "col", "scan")),
                       (ROW_SCAN, ("row",)),
                       (LATENCY, ("latency", "ttfs", "time-to-first")),
                       (FROZEN_POISSON, ("frozen", "poisson", "rate"))):
        if s == code.lower() or s.startswith(keys):
            return code
    raise ValueError(f"unknown stream encoding {name!r}: use one of "
                     f"{', '.join(STREAM_CODES)}")


def stream_events(image, code=COLUMN_SCAN, bin_ms=10.0, threshold=0.5,
                  window_ms=None, rng=None, margin_ms=STREAM_MARGIN_MS,
                  rate_hz=STREAM_RATE_HZ):
    """One 2-D image (intensities 0..1) as the event list of one pattern for
    :func:`neurovat.masquelier.make_event_stream`. Returns a dict:
    ``afferents`` (int32), ``times`` (ms, all in [0, pattern_ms)),
    ``n_afferents`` (input lines of the code), ``pattern_ms`` (section length,
    whole ms), ``layout`` (what one line is, see ``STREAM_LAYOUT``) and
    ``code``.

    * column scan: :func:`scan_events` - line = pixel row, pixel (r, c) at or
      above ``threshold`` spikes once at c x ``bin_ms``; pattern_ms =
      ceil(W x bin_ms).
    * row scan: the same sweep top to bottom - line = pixel column, time =
      r x ``bin_ms``; equal to the column scan of the transpose.
    * latency (time to first spike): one line per pixel; a pixel at or above
      ``threshold`` spikes once at margin + ``window_ms`` x (1 - v) / (1 -
      threshold): the brightest first, one at the threshold last. pattern_ms
      = ceil(2 x margin + window).
    * frozen Poisson: one line per pixel; a pixel at or above ``threshold``
      fires Poisson spikes at ``rate_hz`` x v over ``window_ms`` (after the
      margin), drawn ONCE from ``rng`` (a Generator or a seed; None = seed 0).
      pattern_ms as for latency.
    ``bin_ms`` only matters to the scans, ``window_ms`` (default
    ``STREAM_WINDOW_MS``), ``margin_ms`` and ``rng`` only to the pixel codes."""
    code = stream_code(code)
    img = np.asarray(image, float)
    if img.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {img.shape}")
    H, W = img.shape
    thr = float(threshold)
    if code in (COLUMN_SCAN, ROW_SCAN):
        src = img if code == COLUMN_SCAN else img.T
        aff, t = scan_events(src, bin_ms, thr)
        n_in, n_bins = src.shape
        pattern_ms = float(max(np.ceil(n_bins * float(bin_ms) - 1e-9), 1))
        return {"afferents": aff, "times": np.asarray(t, float),
                "n_afferents": int(n_in), "pattern_ms": pattern_ms,
                "layout": STREAM_LAYOUT[code], "code": code}
    win = float(STREAM_WINDOW_MS if window_ms is None else window_ms)
    margin = float(margin_ms)
    if win <= 0.0 or margin < 0.0:
        raise ValueError(f"window_ms must be > 0 and margin_ms >= 0 (got "
                         f"{win:g}, {margin:g})")
    pattern_ms = float(max(np.ceil(2.0 * margin + win - 1e-9), 1))
    v = np.clip(img.reshape(-1), 0.0, 1.0)
    on = np.nonzero(v >= thr)[0]
    if code == LATENCY:
        span = 1.0 - thr
        frac = ((1.0 - v[on]) / span) if span > 0.0 else np.zeros(on.size)
        t = margin + win * np.clip(frac, 0.0, 1.0)
        aff = on
    else:
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(0 if rng is None else rng)
        counts = rng.poisson(float(rate_hz) * v[on] * win * 1e-3)
        aff = np.repeat(on, counts)
        t = margin + rng.random(aff.size) * win
        order = np.lexsort((aff, t))
        aff, t = aff[order], t[order]
    t = np.minimum(t, np.nextafter(pattern_ms, 0.0))
    return {"afferents": aff.astype(np.int32), "times": t.astype(float),
            "n_afferents": int(H * W), "pattern_ms": pattern_ms,
            "layout": STREAM_LAYOUT[code], "code": code}


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
