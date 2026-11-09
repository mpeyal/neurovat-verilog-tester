"""Image datasets as crossbar input patterns.

Any ``.npz`` with keras-style ``x_train / y_train / x_test / y_test`` arrays
(MNIST, Fashion-MNIST, CIFAR-10, ...) or generic ``x / y`` arrays works. Images
are converted to grayscale, area-averaged down to the network's pixel grid and
min/max normalised per image; splits are balanced per class and seeded.
"""
import hashlib
import os
from dataclasses import dataclass, field

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


@dataclass
class Dataset:
    """Balanced, preprocessed splits: each is (x, y) with x (n, grid_h*grid_w)
    float32 intensities in [0, 1] and y integer class indices."""
    train: tuple
    validation: tuple
    test: tuple
    class_names: list
    grid: tuple
    meta: dict = field(default_factory=dict)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_mnist(cache_dir="datasets"):
    """Path of the keras MNIST archive, downloading it on first use. The file
    must match the pinned SHA-256."""
    dest = os.path.join(cache_dir, "mnist.npz")
    if not os.path.exists(dest):
        import urllib.request
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(MNIST_URL, dest + ".part")
        os.replace(dest + ".part", dest)
    digest = sha256(dest)
    if digest != MNIST_SHA256:
        raise ValueError(f"{dest}: SHA-256 {digest} is not the pinned MNIST archive")
    return dest


def load_npz(path):
    """-> {"x_train", "y_train", "x_test", "y_test"}; the test entries are None
    when the archive has a single image/label pair."""
    # allow_pickle=False: an archive may come from an untrusted source and a
    # pickled object array would execute code on load. Image data is numeric.
    with np.load(path, allow_pickle=False) as z:
        def first(names):
            for k in names:
                if k in z.files:
                    return np.asarray(z[k])
            return None
        out = {"x_train": first(["x_train", "X_train", "images", "x", "arr_0"]),
               "y_train": first(["y_train", "Y_train", "labels", "y", "arr_1"]),
               "x_test": first(["x_test", "X_test"]),
               "y_test": first(["y_test", "Y_test"])}
    if out["x_train"] is None or out["y_train"] is None:
        raise ValueError(f"no image/label arrays found in {os.path.basename(path)}")
    for k in ("y_train", "y_test"):
        if out[k] is not None:
            out[k] = out[k].ravel().astype(int)
    return out


def to_gray(images):
    """(n, H, W) float array from (n, H, W), (n, H, W, C) or (n, C, H, W)."""
    x = np.asarray(images, np.float32)
    if x.ndim == 3:
        return x
    if x.ndim != 4:
        raise ValueError(f"expected image stack, got shape {x.shape}")
    if x.shape[1] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.moveaxis(x, 1, -1)                  # channels first -> last
    if x.shape[-1] == 1:
        return x[..., 0]
    return x[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)


def _area_weights(n_src, n_dst):
    """(n_dst, n_src) matrix averaging each destination cell's source span."""
    edges = np.linspace(0.0, n_src, n_dst + 1)
    lo = np.maximum(edges[:-1, None], np.arange(n_src)[None, :])
    hi = np.minimum(edges[1:, None], np.arange(n_src)[None, :] + 1.0)
    w = np.clip(hi - lo, 0.0, None)
    return w / w.sum(1, keepdims=True)


def to_vectors(images, grid_h, grid_w, invert=False):
    """Image stack -> (n, grid_h*grid_w) float32 in [0, 1]: grayscale, area
    average to the grid, per-image min/max normalisation."""
    x = to_gray(images)
    x = np.einsum("ah,nhw,bw->nab", _area_weights(x.shape[1], grid_h), x,
                  _area_weights(x.shape[2], grid_w))
    x = x - x.min(axis=(1, 2), keepdims=True)
    peak = x.max(axis=(1, 2), keepdims=True)
    x = x / np.where(peak > 0, peak, 1.0)
    if invert:
        x = 1.0 - x
    return x.reshape(len(x), grid_h * grid_w).astype(np.float32)


def load_dataset(path, grid_h, grid_w, train_per_class=100, val_per_class=20,
                 test_per_class=50, n_classes=None, seed=0, invert=False):
    """Balanced disjoint train / validation / test subsets of an ``.npz``.

    Train and validation come from the archive's training images; test from
    its test images (or, when it has none, from held-out training images)."""
    raw = load_npz(path)
    rng = np.random.default_rng(seed)
    classes = sorted(set(raw["y_train"].tolist()))[:n_classes]
    ids = {"train": [], "validation": [], "test": []}
    own_test = raw["x_test"] is not None and raw["y_test"] is not None
    for label in classes:
        order = rng.permutation(np.flatnonzero(raw["y_train"] == label))
        a, b = train_per_class, train_per_class + val_per_class
        ids["train"] += order[:a].tolist()
        ids["validation"] += order[a:b].tolist()
        if own_test:
            ids["test"] += rng.permutation(
                np.flatnonzero(raw["y_test"] == label))[:test_per_class].tolist()
        else:
            ids["test"] += order[b:b + test_per_class].tolist()
    index = {c: k for k, c in enumerate(classes)}
    splits = {}
    for name, chosen in ids.items():
        src = "test" if (name == "test" and own_test) else "train"
        chosen = np.asarray(chosen, int)
        y = np.array([index[c] for c in raw[f"y_{src}"][chosen].tolist()], int)
        splits[name] = (to_vectors(raw[f"x_{src}"][chosen], grid_h, grid_w,
                                   invert), y)
    return Dataset(splits["train"], splits["validation"], splits["test"],
                   [str(c) for c in classes], (grid_h, grid_w),
                   {"path": str(path), "sha256": sha256(path), "seed": seed,
                    "counts": {k: len(v) for k, v in ids.items()},
                    "preprocessing": "grayscale; area average to grid; "
                                     "per-image min/max normalisation"})


def as_patterns(split, class_names):
    """(x, y) -> ([(label, vec01)], [target]) as CrossbarSNN takes them."""
    x, y = split
    return [(class_names[int(t)], v) for v, t in zip(x, y)], [int(t) for t in y]


def attach(net, dataset):
    """Give ``net`` the validation and test splits of ``dataset``."""
    names = dataset.class_names
    if len(dataset.validation[0]):
        net.validation_patterns, net.validation_targets = as_patterns(
            dataset.validation, names)
    if len(dataset.test[0]):
        net.test_patterns, net.test_targets = as_patterns(dataset.test, names)
    net.class_names = list(names)
    return net
