"""Load external image datasets (MNIST and friends) as input-pattern banks
for the Neuromorphic Trainer.

PURE: returns numpy arrays, no GUI.  Kept dependency-light - only numpy is
required; Pillow is used for image files / high-quality resizing when present.

Auto-detected sources (`load_any`):
  * a folder with the 4 MNIST ubyte files (train/t10k images+labels, .gz ok)
  * a folder of images in per-class subfolders   (needs Pillow)
  * an IDX / IDX.gz file  (the labels sibling is found automatically)
  * a .npz   (keras-style x_train/y_train/..., or generic image+label arrays)
  * a .csv   (one row per image: label, pixel0..pixelN; square side inferred)
  * a single image file   (needs Pillow)

`to_patterns(...)` downsamples to the network's pixel grid and samples a
balanced subset -> [(label, vec01)], targets, class_names - exactly what
neuro.Trainer consumes.
"""

import gzip
import os
import struct

import numpy as np

MNIST_NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".gif", ".tif", ".tiff")


# ----------------------------------------------------------------- IDX ------

def _read_idx_bytes(raw):
    if raw[:2] == b"\x1f\x8b":                       # gzip magic
        raw = gzip.decompress(raw)
    if raw[:2] != b"\x00\x00":
        raise ValueError("not an IDX file (bad magic)")
    ndim = raw[3]
    dims = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    arr = np.frombuffer(raw[4 + 4 * ndim:], dtype=np.uint8)
    return arr.reshape(dims)


def load_idx(path):
    with open(path, "rb") as f:
        return _read_idx_bytes(f.read())


def _labels_sibling(images_path):
    """Guess the label IDX path that pairs with an images IDX path."""
    d, base = os.path.split(images_path)
    for a, b in (("images-idx3-ubyte", "labels-idx1-ubyte"),
                 ("images", "labels"), ("image", "label"),
                 ("idx3", "idx1")):
        if a in base:
            cand = os.path.join(d, base.replace(a, b))
            if os.path.exists(cand):
                return cand
    return None


def _find_mnist_in_dir(path):
    """Return (images_path, labels_path) for MNIST ubyte files in a folder."""
    names = os.listdir(path)
    def pick(must):
        for n in names:
            low = n.lower()
            if all(m in low for m in must) and "idx" in low:
                return os.path.join(path, n)
        return None
    imgs = pick(("train", "images")) or pick(("images",))
    lbls = pick(("train", "labels")) or pick(("labels",))
    return imgs, lbls


# ---------------------------------------------------------------- loaders ---

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    def first(cands):
        for k in cands:
            if k in z:
                return z[k]
        return None
    x = first(["x_train", "X_train", "images", "x", "arr_0"])
    y = first(["y_train", "Y_train", "labels", "y", "arr_1"])
    if x is None:                                    # fall back to first 3-D
        for k in keys:
            if np.asarray(z[k]).ndim == 3:
                x = z[k]
                break
    if x is None:
        raise ValueError(f"no image array found in {os.path.basename(path)}")
    return np.asarray(x), (None if y is None else np.asarray(y).ravel())


def load_csv(path, has_header=None):
    raw = np.genfromtxt(path, delimiter=",", dtype=np.float32,
                        skip_header=0)
    if raw.ndim == 1:
        raw = raw[None, :]
    # detect a header row (non-numeric -> NaN in row 0)
    if has_header is None:
        has_header = bool(np.isnan(raw[0]).any())
    if has_header:
        raw = raw[1:]
    # label column present when the remaining width isn't a perfect square
    side_with = int(round((raw.shape[1]) ** 0.5))
    side_without = int(round((raw.shape[1] - 1) ** 0.5))
    if side_without * side_without == raw.shape[1] - 1:
        labels = raw[:, 0].astype(int)
        pix = raw[:, 1:]
        side = side_without
    elif side_with * side_with == raw.shape[1]:
        labels = None
        pix = raw
        side = side_with
    else:
        raise ValueError("CSV pixel count is not square (+/- a label column)")
    imgs = pix.reshape(-1, side, side)
    return imgs, labels


def load_image_folder(path):
    """Per-class subfolders of images -> (images, labels, class_names)."""
    from PIL import Image
    subs = sorted(d for d in os.listdir(path)
                  if os.path.isdir(os.path.join(path, d)))
    imgs, labels, names = [], [], []
    if subs:                                          # class subfolders
        for ci, sub in enumerate(subs):
            names.append(sub)
            folder = os.path.join(path, sub)
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(IMAGE_EXTS):
                    im = Image.open(os.path.join(folder, fn)).convert("L")
                    imgs.append(np.asarray(im, np.float32))
                    labels.append(ci)
    else:                                             # flat folder, no labels
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(IMAGE_EXTS):
                im = Image.open(os.path.join(path, fn)).convert("L")
                imgs.append(np.asarray(im, np.float32))
        labels = None
    if not imgs:
        raise ValueError("no images found in folder")
    # pad/stack: images may differ in size, so resize later in to_patterns;
    # keep as a list when sizes differ
    same = len({im.shape for im in imgs}) == 1
    out = np.stack(imgs) if same else np.array(imgs, dtype=object)
    return out, (None if labels is None else np.asarray(labels)), names


def load_single_image(path):
    from PIL import Image
    im = Image.open(path).convert("L")
    return np.asarray(im, np.float32)[None, ...], None


def load_any(path):
    """Dispatch on the path -> (images, labels_or_None, class_names_or_None).
    images: (N, H, W) uint8/float, or an object array of varied-size frames."""
    if os.path.isdir(path):
        imgs, lbls = _find_mnist_in_dir(path)
        if imgs:
            X = load_idx(imgs)
            Y = load_idx(lbls) if lbls else None
            return X, Y, None
        return load_image_folder(path)
    low = path.lower()
    if low.endswith(".npz"):
        X, Y = load_npz(path)
        return X, Y, None
    if low.endswith(".csv"):
        X, Y = load_csv(path)
        return X, Y, None
    if low.endswith(IMAGE_EXTS):
        X, Y = load_single_image(path)
        return X, Y, None
    # assume IDX (ubyte / .gz / no extension)
    X = load_idx(path)
    sib = _labels_sibling(path)
    Y = load_idx(sib) if sib else None
    return X, Y, None


# --------------------------------------------------------- resize + sample --

def _resize(img, gh, gw):
    """Downsample one 2-D frame to gh x gw, area-averaged when Pillow is here,
    else strided nearest."""
    img = np.asarray(img, np.float32)
    if img.shape == (gh, gw):
        return img
    try:
        from PIL import Image
        m = float(img.max()) or 1.0
        pim = Image.fromarray(np.clip(img / m * 255.0, 0, 255).astype(np.uint8))
        pim = pim.resize((gw, gh), Image.BILINEAR)
        return np.asarray(pim, np.float32) / 255.0 * m
    except Exception:
        rs = np.linspace(0, img.shape[0] - 1, gh).round().astype(int)
        cs = np.linspace(0, img.shape[1] - 1, gw).round().astype(int)
        return img[np.ix_(rs, cs)]


def to_patterns(images, labels, gh, gw, per_class=12, max_total=240,
                n_classes=None, seed=0, invert=False):
    """Downsample + balance-sample a dataset into Trainer patterns.

    Returns (patterns, targets, class_names):
      patterns = [(label_str, vec01)]   vec01 is (gh*gw,) in [0,1]
      targets  = [class_index, ...]     aligned with patterns
      class_names = [str, ...]
    Sampling is balanced per class (up to per_class each, capped at max_total).
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    if labels is None:
        labels = np.zeros(n, int)
    labels = np.asarray(labels).ravel().astype(int)
    classes = sorted(set(labels.tolist()))
    if n_classes:
        classes = classes[:n_classes]
    idx_by_class = {c: np.where(labels == c)[0] for c in classes}

    chosen = []
    for c in classes:
        ids = idx_by_class[c]
        if len(ids) > per_class:
            ids = rng.choice(ids, per_class, replace=False)
        chosen.extend((int(i), c) for i in ids)
    rng.shuffle(chosen)
    chosen = chosen[:max_total]

    patterns, targets = [], []
    cls_index = {c: k for k, c in enumerate(classes)}
    for i, c in chosen:
        frame = images[i]
        g = _resize(frame, gh, gw)
        g = g - g.min()
        mx = g.max() or 1.0
        g = g / mx
        if invert:
            g = 1.0 - g
        patterns.append((str(c), g.reshape(-1).astype(np.float32)))
        targets.append(cls_index[c])
    names = [str(c) for c in classes]
    return patterns, targets, names


def to_patterns_split(images, labels, gh, gw, train_per_class=12,
                      test_per_class=8, n_classes=None, seed=0, invert=False):
    """Like to_patterns, but returns DISJOINT train and test sets (held-out
    images per class) so the GUI can report real generalisation accuracy.
    Returns (train_pats, train_tgts, test_pats, test_tgts, class_names)."""
    rng = np.random.default_rng(seed)
    n = len(images)
    if labels is None:
        labels = np.zeros(n, int)
    labels = np.asarray(labels).ravel().astype(int)
    classes = sorted(set(labels.tolist()))
    if n_classes:
        classes = classes[:n_classes]
    cls_index = {c: k for k, c in enumerate(classes)}

    def render(i, c):
        g = _resize(images[i], gh, gw)
        g = g - g.min()
        g = g / (g.max() or 1.0)
        if invert:
            g = 1.0 - g
        return (str(c), g.reshape(-1).astype(np.float32)), cls_index[c]

    tr_p, tr_t, te_p, te_t = [], [], [], []
    for c in classes:
        ids = np.where(labels == c)[0]
        rng.shuffle(ids)
        for i in ids[:train_per_class]:
            p, t = render(int(i), c); tr_p.append(p); tr_t.append(t)
        for i in ids[train_per_class:train_per_class + test_per_class]:
            p, t = render(int(i), c); te_p.append(p); te_t.append(t)
    order = rng.permutation(len(tr_p))
    tr_p = [tr_p[i] for i in order]
    tr_t = [tr_t[i] for i in order]
    return tr_p, tr_t, te_p, te_t, [str(c) for c in classes]


def download_mnist(cache_dir):
    """Fetch the keras MNIST .npz into cache_dir; return its path.  Raises on
    failure (offline, etc.) - the caller logs it."""
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, "mnist.npz")
    if not os.path.exists(dest):
        urllib.request.urlretrieve(MNIST_NPZ_URL, dest)
    return dest


def _to_raw_github(url):
    """Rewrite a GitHub 'blob' page URL to its raw-content URL so it downloads
    the file, not the HTML page.  Other URLs pass through unchanged."""
    if "github.com/" in url and "/blob/" in url:
        url = url.replace("https://github.com/",
                          "https://raw.githubusercontent.com/", 1)
        url = url.replace("/blob/", "/", 1)
    return url


def download_url(url, cache_dir):
    """Fetch a dataset from a direct / raw URL (raw GitHub, a mirror, ...) into
    cache_dir and return the local path.  GitHub 'blob' links are auto-rewritten
    to raw; the local name keeps the URL's filename so load_any can dispatch on
    its extension (.npz / .csv / .idx / .gz / image)."""
    import urllib.request
    import urllib.parse
    url = _to_raw_github(url.strip())
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.basename(urllib.parse.urlparse(url).path) or "dataset"
    if "." not in name:                              # let load_any try IDX
        name += ".idx"
    dest = os.path.join(cache_dir, name)
    req = urllib.request.Request(url, headers={"User-Agent": "neurovat/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest
