"""GUI-free helpers of the Trainer's 'unsupervised stream' views: the
scanned-image input, the auto threshold, and the window of a run that the
Raster tab draws (live from the worker, or replayed afterwards). No
DearPyGui here; tested in tests/test_stream_view.py."""
import math
import zlib

import numpy as np

POISSON = "Poisson patterns (Masquelier)"
IMAGES = "scanned images (spatio-temporal)"
INPUTS = [POISSON, IMAGES]
IMAGE_SETS = ["digits", "letters", "bars", "dataset"]
SPACINGS = ["random", "even"]
# image stream codes, as neurovat.encoding.STREAM_CODES names them (the
# first is the default: one input per pixel row, columns swept in time)
CODES = ["column scan", "row scan", "latency", "frozen Poisson"]
SPEEDS = ["0.25x", "1x", "4x"]
AUTO_THR = 0.85              # image threshold = AUTO_THR x weakest clean peak
RASTER_SHOWN = 60            # Poisson: afferents drawn in the raster
NET_SHOWN = 24               # Poisson: afferents drawn in the architecture
U_BUDGET = 2400.0            # record u only when seconds x neurons <= this
# nt_st_w0 per input. Images start at 1.0. The auto threshold is 0.85 x the
# image's peak AT THE START WEIGHTS, so the untrained network answered the
# same at 0.475 and 1.0 (measured on the MNIST '0', 1 and 3 neurons).
# What differs is what STDP can do: at 1.0 it can only lower weights and
# prunes the answers to noise; at 0.475 it potentiated the digit's rows under
# the fixed threshold and the false answers rose (MNIST '0', 60 s).
W0_DEFAULT = {POISSON: 0.475, IMAGES: 1.0}
# ... except per image code (free parameter, measured on the MNIST '0', 28 x
# 28, 1-10 Hz noise, 2 stream seeds x 2 frozen draws): frozen Poisson fires
# about 5 spikes per line per window, so from 1.0 STDP depressed the image's
# own lines below the threshold and the hits collapsed (13 of 16 cells); from
# 0.5 the hits held in all 16 (0.7 failed, see neurovat/README.md)
W0_CODE = {"frozen Poisson": 0.5}
DETECT_HIT, DETECT_FALSE = 0.9, 0.1   # image verdict: >= 90 % hits, <= 10 % others
# nt_st_noise / nt_st_bg defaults per input layout: 10 Hz for the scans (and
# Poisson input); the pixel codes (latency, frozen Poisson) have 28 x 28 = 784
# lines instead of 28, so the same rate per line is 28 times the noise
# spikes per second on the network.
NOISE_HZ = 10.0
# free parameter, for both pixel codes. Latency, MNIST '0', 2 seeds: learns
# at 3-5 Hz (188/188 false windows -> 0-4/188, hits held); clean untrained
# at 1 Hz; fires in every window at 10 Hz. Frozen Poisson (start weight
# 0.5): the MNIST '1' learns at 3 Hz (117/226 -> 0/228, hits held; 1 seed)
# and fires in every window at 10 Hz; the MNIST '0' is clean untrained at
# 3 Hz and learns at 10 Hz (2 seeds x 2 draws). Noise tolerance depends
# on how much ink the image has.
PIXEL_NOISE_HZ = 3.0
# pixel codes: input lines drawn in the raster (room for every inked pixel
# of one image: the built-in 28 x 28 '0' has 454, an MNIST digit ~100-200)
PIXEL_SHOWN = 480
PIXEL_OTHERS = 24            # ... of which silent (un-inked) lines, sampled
NET_PIXELS = 28              # pixel codes: input lines drawn in the network
# what one input line is, per layout (neurovat.encoding.STREAM_LAYOUT)
LINE_NAME = {"row": "pixel row", "col": "pixel column", "pixel": "pixel (r, c)"}


def input_kind(value):
    """Normalise a value of the input combo (bridge / agent strings too)."""
    s = str(value or "").lower()
    return IMAGES if s.startswith(("scan", "image", "spat", "img")) else POISSON


def code_name(value):
    """Normalise a value of the encoding combo (bridge / agent strings too,
    e.g. 'row', 'ttfs', 'poisson'); ValueError for an unknown one."""
    from neurovat.encoding import stream_code
    return stream_code(value)


def code_layout(value):
    """What one input line of the code is: 'row', 'col' or 'pixel'."""
    from neurovat.encoding import STREAM_LAYOUT
    return STREAM_LAYOUT[code_name(value)]


def noise_default(layout):
    """Default noise Hz / background Hz for an input layout (None = Poisson
    input): PIXEL_NOISE_HZ for the pixel codes, NOISE_HZ otherwise."""
    return PIXEL_NOISE_HZ if layout == "pixel" else NOISE_HZ


def swapped_default(value, old_layout, new_layout):
    """The value a noise control takes when the layout changes: the new
    default if it still holds the old default (the user has not edited
    it), else ``value`` unchanged."""
    v = float(value)
    if abs(v - noise_default(old_layout)) < 1e-6:
        return noise_default(new_layout)
    return v


def w0_default(kind, code=None):
    """Default start weight (nt_st_w0) of an input kind and image code:
    W0_DEFAULT per input, W0_CODE for an image code that has one."""
    if kind != IMAGES:
        return W0_DEFAULT[POISSON]
    try:
        code = code_name(code)
    except ValueError:
        code = None
    return W0_CODE.get(code, W0_DEFAULT[IMAGES])


def line_label(layout, i, size):
    """Short name of input line ``i``: 'row 3', 'col 3' or '(r,c)'."""
    if layout == "pixel":
        r, c = divmod(int(i), int(size))
        return f"({r},{c})"
    return f"{'col' if layout == 'col' else 'row'} {int(i)}"


def pixel_lines(vecs, ink, n, cap=PIXEL_SHOWN, others=PIXEL_OTHERS):
    """Input lines of a pixel code that the raster draws: every pixel inked
    (>= ``ink``) in any of the hidden images ``vecs`` (an even sample of them
    when there are more than ``cap - others``), plus an even sample of
    ``others`` silent pixels so the noise on them shows. Returns (sorted
    line indices, how many of them are inked)."""
    on = np.zeros(int(n), bool)
    for v in vecs:
        on |= np.asarray(v, float).reshape(-1)[:int(n)] >= float(ink)
    inked = np.nonzero(on)[0]
    room = max(int(cap) - int(others), 1)
    if inked.size > room:
        inked = inked[shown_afferents(inked.size, room)]
    rest = np.nonzero(~on)[0]
    k = min(int(others), int(cap) - inked.size, rest.size)
    rest = rest[shown_afferents(rest.size, k)] if k > 0 else rest[:0]
    return np.union1d(inked, rest).astype(int), int(inked.size)


def code_timing(img):
    """The code and its timing in a few words (for the log): 'column scan,
    10 ms / column', 'row scan, 10 ms / row' or 'latency, 50 ms window'."""
    layout = img.get("layout", "row")
    if layout == "row":
        return f"{img.get('code', 'column scan')}, {img['bin_ms']:g} ms / column"
    if layout == "col":
        return f"{img.get('code', 'row scan')}, {img['bin_ms']:g} ms / row"
    return f"{img.get('code')}, {img['window_ms']:g} ms window"


# one line per code: the encoding combo's tooltip
CODE_TIPS = {
    "column scan": "column scan: one input per pixel row; pixel (r, c) fires "
                   "once at c x bin ms (a left-to-right sweep).",
    "row scan": "row scan: one input per pixel column; pixel (r, c) fires once "
                "at r x bin ms (a top-to-bottom sweep).",
    "latency": "latency: one input per pixel (r x W + c); a pixel fires once, "
               "the brighter the earlier, within the window. Learns at 3-5 "
               "Hz noise with 60 s of training (default 3 Hz for latency).",
    "frozen Poisson": "frozen Poisson: one input per pixel; Poisson spikes "
                      "at 100 Hz x brightness over the window, drawn once "
                      "and repeated unchanged (about 5 per line). Start "
                      "weight 0.5 (set when you pick it, unless you "
                      "changed it): from weight 1 STDP depressed the "
                      "image's own lines below the threshold and the hits "
                      "collapsed. Noise 3 Hz, as for latency: the sparse "
                      "MNIST '1' learns at 3 Hz and fires in every window "
                      "at 10 Hz; the dense MNIST '0' is clean untrained at "
                      "3 Hz and learns at 10 Hz.",
}


def code_explainer(code, size, bin_ms, window_ms, pattern_ms, seed=0):
    """One paragraph for the Patterns tab: how the code turns an image into
    spikes, with the numbers of this network."""
    from neurovat.encoding import STREAM_MARGIN_MS, STREAM_RATE_HZ
    code = code_name(code)
    tail = f" Section {pattern_ms:g} ms."
    if code == "column scan":
        return ("Column scan: one afferent per pixel row; pixel (r, c) at or "
                "above the ink threshold fires afferent r at c x "
                f"{bin_ms:g} ms." + tail)
    if code == "row scan":
        return ("Row scan: one afferent per pixel column; pixel (r, c) at or "
                "above the ink threshold fires afferent c at r x "
                f"{bin_ms:g} ms." + tail)
    n = int(size) * int(size)
    head = (f"one afferent per pixel ({n} lines, pixel (r, c) = line r x "
            f"{int(size)} + c)")
    if code == "latency":
        return (f"Latency: {head}; a pixel at or above the ink threshold "
                "fires once, the brighter the earlier, within "
                f"{window_ms:g} ms after a {STREAM_MARGIN_MS:g} ms lead (the "
                "brightest at the lead, one at the threshold at the end)."
                + tail)
    return (f"Frozen Poisson: {head}; a pixel at or above the ink threshold "
            f"fires Poisson spikes at {STREAM_RATE_HZ:g} Hz x its brightness "
            f"over {window_ms:g} ms after a {STREAM_MARGIN_MS:g} ms lead, "
            f"drawn once (seed {int(seed)}) and repeated unchanged at every "
            "presentation." + tail)


def parse_indices(text, n, exclude=(), what="image"):
    """'0, 3, 5-7' -> [0, 3, 5, 6, 7]; '' -> []; 'all' -> every index of
    range(n) not in ``exclude``. ValueError names the bad entry."""
    s = str(text or "").strip().lower()
    if not s:
        return []
    if s == "all":
        ex = set(int(i) for i in exclude)
        return [i for i in range(n) if i not in ex]
    out = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part[1:]:
                a, b = part.split("-", 1)
                idx = list(range(int(a), int(b) + 1))
            else:
                idx = [int(part)]
        except ValueError:
            raise ValueError(f"{what} index '{part}' is not a number "
                             "(use e.g. 0 or 0,3 or 2-4)") from None
        for i in idx:
            if not 0 <= i < n:
                raise ValueError(f"{what} index {i} is out of range: the set "
                                 f"has {n} image(s), 0 to {n - 1}")
            if i not in out:
                out.append(i)
    return out


def _normalise(img, size, invert):
    from .datasets import _resize
    g = _resize(img, size, size).astype(float)
    g = g - g.min()
    g = g / (g.max() or 1.0)
    return 1.0 - g if invert else g


def dataset_images(images, labels, size, invert=False):
    """One image per class, in class order (the first sample of each),
    resized to size x size and normalised as ``datasets.to_patterns`` does.
    Returns [(class name, flat vec in 0..1)]."""
    n = len(images)
    labels = (np.zeros(n, int) if labels is None
              else np.asarray(labels).ravel().astype(int))
    out = []
    for c in sorted(set(labels.tolist())):
        i = int(np.nonzero(labels == c)[0][0])
        out.append((str(c), _normalise(images[i], size, invert).reshape(-1)))
    return out


def image_bank(name, size, ds=None, invert=False):
    """The images an index refers to: a built-in set or the loaded dataset."""
    if name == "dataset":
        if not ds or not len(ds.get("images", ())):
            raise ValueError("image set 'dataset': no dataset loaded - press "
                             "Load MNIST (or load one in the Data tab)")
        return dataset_images(ds["images"], ds.get("labels"), size, invert)
    if name not in IMAGE_SETS:
        raise ValueError(f"unknown image set {name!r}")
    from neurovat.patterns import make_patterns
    return [(str(lab), np.asarray(v, float).reshape(-1))
            for lab, v in make_patterns(name, size, size)]


def image_code(vec, size, bin_ms, ink, code=None, window_ms=None, seed=0):
    """One flat image in the stream code ``code`` (default the column scan):
    the dict of :func:`neurovat.encoding.stream_events` (afferents, times,
    n_afferents, pattern_ms, layout, code). ``seed`` (an int or a list of
    ints) draws the frozen Poisson spikes: fixed, so pressing Train again
    gives the same image; :func:`build_image_input` uses [seed, index]."""
    from neurovat.encoding import stream_events
    return stream_events(np.asarray(vec, float).reshape(size, size), code,
                         bin_ms, ink, window_ms=window_ms, rng=seed)


def image_events(vec, size, bin_ms, ink, code=None, window_ms=None, seed=0):
    """One flat image as events (input lines, times in ms); the column scan
    (one line per pixel row) unless ``code`` names another stream code."""
    if code is None:
        from neurovat.encoding import scan_events
        return scan_events(np.asarray(vec, float).reshape(size, size), bin_ms,
                           ink)
    ev = image_code(vec, size, bin_ms, ink, code, window_ms, seed)
    return ev["afferents"], ev["times"]


def build_image_input(bank, picks_text, distract_text, size, bin_ms, ink,
                      code=None, window_ms=None, seed=0):
    """Hidden images and distractors as event lists for ``Session``, in the
    stream code ``code`` (default: column scan, one input per pixel row).
    ``pattern_ms`` comes from the code (column scan: ceil(columns x bin),
    whole ms, at least the scan); ``n_afferents`` is the code's input count
    (size for the scans, size x size for the pixel codes) and ``layout`` what
    one input is ("row", "col" or "pixel", see
    ``neurovat.encoding.STREAM_LAYOUT``). Frozen Poisson spikes are drawn
    once per image from (``seed``, bank index)."""
    from neurovat.encoding import STREAM_WINDOW_MS, stream_code
    code = stream_code(code)
    picks = parse_indices(picks_text, len(bank))
    if not picks:
        raise ValueError("pick at least one image to hide (comma-separated "
                         "indices, e.g. 0 or 0,3)")
    distract = parse_indices(distract_text, len(bank), exclude=picks,
                             what="distractor")
    both = sorted(set(picks) & set(distract))
    if both:
        raise ValueError(f"image(s) {both} are both hidden and distractors")

    info = {}

    def events(i):
        label, vec = bank[i]
        ev = image_code(vec, size, bin_ms, ink, code, window_ms,
                        [int(seed), int(i)])
        info.update(ev)
        if not ev["afferents"].size:
            raise ValueError(f"image {i} ('{label}') has no pixel at or above "
                             f"the ink threshold {ink:g}" + (
                                 " (or its frozen Poisson draw has no spike)"
                                 if ev["code"] == "frozen Poisson" else ""))
        return ev["afferents"], ev["times"]
    patterns = [events(i) for i in picks]
    distractors = [events(i) for i in distract]
    return {
        "patterns": patterns,
        "distractors": distractors,
        "picks": picks, "distract": distract,
        "labels": [bank[i][0] for i in picks],
        "dlabels": [bank[i][0] for i in distract],
        "vecs": [bank[i][1] for i in picks],
        "dvecs": [bank[i][1] for i in distract],
        "size": size, "bin_ms": float(bin_ms),
        "pattern_ms": float(info["pattern_ms"]),
        "code": code, "layout": info["layout"],
        "n_afferents": int(info["n_afferents"]),
        "window_ms": (None if info["layout"] != "pixel" else float(
            window_ms if window_ms is not None else STREAM_WINDOW_MS)),
        "seed": int(seed),
    }


def events_digest(events, code=None):
    """Stable checksum of a list of (afferents, times) event lists; with
    ``code`` (a stream code name) the code is part of it, so two codes that
    happen to give the same events still differ. ``code=None`` is the
    checksum as before (column-scan networks keep their key)."""
    h = 0 if code is None else zlib.crc32(str(code).encode("utf-8"))
    for a, t in events:
        h = zlib.crc32(np.asarray(a, np.int64).tobytes(), h)
        h = zlib.crc32(np.round(np.asarray(t, float), 6).tobytes(), h)
    return h


def auto_threshold(patterns, P, frac=AUTO_THR):
    """``frac`` x the lowest peak potential that one clean presentation of a
    hidden pattern drives at the starting weights. ``patterns`` is a list of
    (afferents, times) event lists in any stream code, or the dict of
    :func:`build_image_input` (its ``patterns``). Only the spike times and
    ``P.w_init`` matter, so the same rule serves every code."""
    from neurovat.masquelier import pattern_peak
    if isinstance(patterns, dict):
        patterns = patterns["patterns"]
    return frac * min(pattern_peak(p, P) for p in patterns)


def shown_afferents(n, cap):
    """An evenly spaced sample of at most ``cap`` of n afferents."""
    if n <= cap:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, cap)).astype(int))


def hit_rule(images):
    return ("at least one output spike in the window" if images
            else "exactly one spike of some neuron in the window")


def section_hit(spikes, images):
    """Is a pattern section answered: ``spikes`` = [(ms, neuron)]."""
    if not spikes:
        return False
    if images:
        return True
    return 1 in np.bincount([s[1] for s in spikes])


def first_spike_latency(post_t, post_j, post_lab, post_lat, t0_ms, neuron,
                        pattern):
    """Mean ms from window onset to the FIRST spike of ``neuron`` in each
    window of ``pattern`` it answered, over the windows that start at or
    after ``t0_ms`` (the scored part of the run). None if it answered none."""
    t = np.asarray(post_t, float)
    lat = np.asarray(post_lat, float)
    m = ((np.asarray(post_j, int) == int(neuron))
         & (np.asarray(post_lab, int) == int(pattern)) & np.isfinite(lat))
    onset = np.round(t - np.where(np.isfinite(lat), lat, 0.0), 6)
    m &= onset >= float(t0_ms) - 1e-6
    if not m.any():
        return None
    first = {}
    for o, v in zip(onset[m].tolist(), lat[m].tolist()):
        first[o] = min(first.get(o, math.inf), v)
    return float(np.mean(list(first.values())))


def window_rates(w):
    """(share of pattern windows answered, share of other windows answered)."""
    pos, neg = w["tp"] + w["fn"], w["fp"] + w["tn"]
    return (w["tp"] / pos if pos else 0.0, w["fp"] / neg if neg else 0.0)


def detects(w):
    """The image verdict: answers >= 90 % of its windows, <= 10 % of others."""
    if not w:
        return False
    hit, false = window_rates(w)
    return hit >= DETECT_HIT and false <= DETECT_FALSE


def window_text(w):
    if not w:
        return "no neuron"
    ba = w.get("balanced_accuracy")
    return (f"TP {w['tp']} FN {w['fn']} FP {w['fp']} TN {w['tn']}, balanced "
            f"acc {'-' if ba is None else f'{ba:.2f}'}")


def pattern_rows(per_neuron, n_patterns):
    """Per hidden pattern: the neuron that answers it best (among those whose
    best pattern it is), its window counts, and whether it detects it."""
    rows = []
    for p in range(int(n_patterns)):
        cand = [q for q in per_neuron if int(q["best_pattern"]) == p]
        if not cand:
            rows.append({"pattern": p, "neuron": None, "windows": None,
                         "detects": False})
            continue
        q = max(cand, key=lambda q: (detects(q["windows"]),
                                     q["windows"]["balanced_accuracy"] or 0.0))
        rows.append({"pattern": p, "neuron": int(q["neuron"]),
                     "windows": q["windows"], "detects": detects(q["windows"])})
    return rows


COMPARED = ("share", "spacing", "jitter", "noise", "delete", "bg")


def stream_differs(a, b):
    """Stream settings (``result['stream']``) in which two runs differ."""
    return [k for k in COMPARED if (a or {}).get(k) != (b or {}).get(k)]


def training_effect(base, trained):
    """What training changed for one pattern, from the window counts of the
    untrained baseline and of the trained run: (learnt?, text)."""
    if not base or not trained:
        return False, "no neuron answers it in one of the two runs"
    bh, bf = window_rates(base)
    th, tf = window_rates(trained)
    change = (f"other windows answered {100 * bf:.0f} % -> {100 * tf:.0f} %, "
              f"pattern windows {100 * bh:.0f} % -> {100 * th:.0f} %")
    held = th >= min(bh, DETECT_HIT)
    if held and tf < bf:
        line = ("it now detects it" if detects(trained) else
                f"still above the {100 * DETECT_FALSE:.0f} % detection line")
        return True, (f"learnt: training pruned the false answers with the "
                      f"hits held ({change}); {line}")
    if detects(trained) and detects(base):
        return False, ("detected already untrained, so training added "
                       f"nothing measurable ({change})")
    if tf > bf or th < bh:
        return False, f"training made it worse ({change})"
    return False, f"no learning shown ({change})"


def view_window_ms(value, images, pattern_ms, share):
    """Seconds of stream in the raster window: the control, or 0 = auto
    (1.5 s for Poisson input, six presentations' worth for images)."""
    v = float(value or 0.0)
    if v > 0.0:
        return v * 1000.0
    if not images:
        return 1500.0
    period = max(int(round(1.0 / max(share, 1e-6))), 2)
    return 6.0 * period * pattern_ms


class RunView:
    """Extracts what the Raster tab draws for one run: input spikes on the
    shown afferents, pattern windows, output spikes and the potential, for
    the window ending at ``t1_ms`` (ms from the run start). Used by the worker
    (live, the stream is complete but must be cut at "now") and by the
    render thread (replay of the finished run)."""

    def __init__(self, stream, pattern_ms, n_out, threshold, offset_s,
                 shown, n_aff, win_ms, images):
        self.S = stream
        self.pattern_ms = float(pattern_ms)
        self.n_out = int(n_out)
        self.n_pat = max(int(stream.n_patterns), 1)
        self.n_dis = int(stream.extra.max()) + 1 if (
            stream.extra is not None and stream.extra.size) else 0
        self.threshold = float(threshold)
        self.offset_s = float(offset_s)
        self.shown = np.asarray(shown, int)
        self.lut = np.full(int(n_aff), -1, np.int64)
        self.lut[self.shown] = np.arange(self.shown.size)
        self.win_ms = float(win_ms)
        self.images = bool(images)
        self.duration_ms = float(stream.duration_ms)

    def _windows(self, t0, t1, ids, n):
        """Per id: (starts s, ends s) of the sections in [t0, t1]."""
        P = self.pattern_ms
        k0 = max(int(math.floor(t0 / P)), 0)
        k1 = min(int(math.ceil(t1 / P)), ids.size)
        out = [([], []) for _ in range(n)]
        for k in range(k0, k1):
            i = int(ids[k])
            if i < 0 or i >= n:
                continue
            a, b = k * P, min((k + 1) * P, t1)
            if b <= max(a, t0):
                continue
            out[i][0].append(self.offset_s + a / 1000.0)
            out[i][1].append(self.offset_s + b / 1000.0)
        return out

    def snapshot(self, t1_ms, post_t, post_j, u_ms=None, live=False):
        S = self.S
        t1 = float(min(max(t1_ms, 0.0), self.duration_ms))
        t0 = t1 - self.win_ms
        lo, hi = np.searchsorted(S.t, [max(t0, 0.0), t1])
        aff, t = S.aff[lo:hi], S.t[lo:hi]
        kind = (S.kind[lo:hi] if S.kind is not None
                else np.zeros(t.size, np.int8))
        y = self.lut[aff]
        keep = y >= 0
        y, t, kind = y[keep], t[keep], kind[keep]
        x = self.offset_s + t / 1000.0
        sec = np.minimum((t // self.pattern_ms).astype(int), S.label.size - 1)
        lab = S.label[sec] if t.size else np.zeros(0, int)
        pat = []
        for p in range(self.n_pat):
            m = (kind == 1) & (lab == p)
            pat.append((x[m].tolist(), y[m].astype(float).tolist()))
        m = kind == 0
        bg = (x[m].tolist(), y[m].astype(float).tolist())
        m = kind == 2
        dis = (x[m].tolist(), y[m].astype(float).tolist())

        pt = np.asarray(post_t, float)
        pj = np.asarray(post_j, int)
        sel = (pt > t0) & (pt <= t1)
        pt, pj = pt[sel], pj[sel]
        post = [(self.offset_s + pt[pj == j] / 1000.0).tolist()
                for j in range(self.n_out)]

        u = None
        if u_ms is not None:
            r0, r1 = max(int(math.floor(t0)), 0), min(int(math.ceil(t1)),
                                                      len(u_ms))
            if r1 > r0:
                blk = np.array(u_ms[r0:r1], float) / (self.threshold or 1.0)
                blk[~np.isfinite(blk)] = np.nan
                ux = (self.offset_s + (np.arange(r0, r1) + 0.5) / 1000.0)
                u = (ux.tolist(), [blk[:, j].tolist()
                                   for j in range(blk.shape[1])])
            else:
                u = ([], [[] for _ in range(self.n_out)])

        win = self._windows(t0, t1, S.label, self.n_pat)
        dwin = (self._windows(t0, t1, S.extra, self.n_dis)
                if self.n_dis else [])
        present = sorted({int(S.label[k]) for k in range(
            max(int(math.floor(max(t0, 0.0) / self.pattern_ms)), 0),
            min(int(math.ceil(t1 / self.pattern_ms)), S.label.size))
            if S.label[k] >= 0})
        n_win = sum(len(w[0]) for w in win)
        return {
            "live": bool(live),
            "t0": self.offset_s + t0 / 1000.0, "t1": self.offset_s + t1 / 1000.0,
            "t_run_s": t1 / 1000.0, "run_s": self.duration_ms / 1000.0,
            "pat": pat, "bg": bg, "dis": dis, "post": post, "u": u,
            "win": win, "dwin": dwin, "present": present,
            "n_windows": n_win, "n_post": int(pt.size),
            "post_per_neuron": [int((pj == j).sum()) for j in range(self.n_out)],
        }


def frozen_raster(copy, shown):
    """Clean raster of a pattern ((afferents, ms)) on the shown afferents:
    (ms list, row-in-sample list)."""
    a, t = np.asarray(copy[0], int), np.asarray(copy[1], float)
    lut = {int(v): k for k, v in enumerate(np.asarray(shown, int))}
    keep = [i for i, v in enumerate(a.tolist()) if v in lut]
    return ([float(t[i]) for i in keep], [float(lut[int(a[i])]) for i in keep])
