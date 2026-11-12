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
SPEEDS = ["0.25x", "1x", "4x"]
AUTO_THR = 0.85              # image threshold = AUTO_THR x weakest clean peak
RASTER_SHOWN = 60            # Poisson: afferents drawn in the raster
NET_SHOWN = 24               # Poisson: afferents drawn in the architecture
U_BUDGET = 2400.0            # record u only when seconds x neurons <= this
NOISE_DEFAULT = {POISSON: 10.0, IMAGES: 0.0}   # nt_st_noise per input


def input_kind(value):
    """Normalise a value of the input combo (bridge / agent strings too)."""
    s = str(value or "").lower()
    return IMAGES if s.startswith(("scan", "image", "spat", "img")) else POISSON


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


def image_events(vec, size, bin_ms, ink):
    """Column scan of one flat image: (rows, times in ms)."""
    from neurovat.encoding import scan_events
    return scan_events(np.asarray(vec, float).reshape(size, size), bin_ms, ink)


def build_image_input(bank, picks_text, distract_text, size, bin_ms, ink):
    """Hidden images and distractors as event lists for ``Session``.
    ``pattern_ms`` = ceil(columns x bin): whole ms, at least the scan."""
    picks = parse_indices(picks_text, len(bank))
    if not picks:
        raise ValueError("pick at least one image to hide (comma-separated "
                         "indices, e.g. 0 or 0,3)")
    distract = parse_indices(distract_text, len(bank), exclude=picks,
                             what="distractor")
    both = sorted(set(picks) & set(distract))
    if both:
        raise ValueError(f"image(s) {both} are both hidden and distractors")

    def events(i):
        label, vec = bank[i]
        ev = image_events(vec, size, bin_ms, ink)
        if not ev[0].size:
            raise ValueError(f"image {i} ('{label}') has no pixel at or above "
                             f"the ink threshold {ink:g}")
        return ev
    return {
        "patterns": [events(i) for i in picks],
        "distractors": [events(i) for i in distract],
        "picks": picks, "distract": distract,
        "labels": [bank[i][0] for i in picks],
        "dlabels": [bank[i][0] for i in distract],
        "vecs": [bank[i][1] for i in picks],
        "dvecs": [bank[i][1] for i in distract],
        "size": size, "bin_ms": float(bin_ms),
        "pattern_ms": float(max(math.ceil(size * bin_ms - 1e-9), 1)),
    }


def events_digest(events):
    """Stable checksum of a list of (afferents, times) event lists."""
    h = 0
    for a, t in events:
        h = zlib.crc32(np.asarray(a, np.int64).tobytes(), h)
        h = zlib.crc32(np.round(np.asarray(t, float), 6).tobytes(), h)
    return h


def auto_threshold(patterns, P, frac=AUTO_THR):
    """``frac`` x the lowest peak potential that one clean presentation of a
    hidden pattern drives at the starting weights."""
    from neurovat.masquelier import pattern_peak
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
