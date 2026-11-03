"""Augmented 2-input XOR gate - a LARGER, noisy training set so surrogate BPTT
actually converges instead of collapsing to a constant predictor.

The built-in `xor` source is only the 4 logic rows (repeated to 8).  With so few
samples the trivial "always one class" predictor stays competitive (50%), so most
random inits collapse to it and never form the XOR partition.  This plugin keeps
the SAME complementary [A, ~A, B, ~B] 2x2 coding but emits MANY jittered copies
of each corner (balanced 0/1), which makes the constant solution cost real loss
and gives the gradient a dense, varied signal -> far more reliable convergence.

Still NOT linearly separable: build with a hidden layer (HIDDEN LAYERS e.g. "16")
and the Surrogate-grad (BPTT) rule.
"""

import numpy as np

PER_CORNER = 24        # noisy samples generated per XOR corner (4 corners)
NOISE = 0.12           # Gaussian sigma on the [0,1] complementary coding
SEED = 1234            # deterministic so the set is reproducible across builds


def _base(a, b):
    # complementary [A, ~A, B, ~B] coding on a 2x2 grid (bias-free margin)
    return np.array([float(a), 1.0 - a, float(b), 1.0 - b], np.float32)


def make(cfg):
    """Return (patterns, targets) and set the grid/class count for XOR."""
    cfg.grid_h, cfg.grid_w, cfg.n_out = 2, 2, 2
    rng = np.random.default_rng(SEED)
    corners = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]   # a, b, a^b
    pats, targets = [], []
    for k in range(PER_CORNER):
        for (a, b, y) in corners:
            v = _base(a, b) + rng.normal(0.0, NOISE, size=4).astype(np.float32)
            v = np.clip(v, 0.0, 1.0)
            pats.append((f"A={a} B={b} -> {y} #{k}", v))
            targets.append(y)
    return pats, targets


PATTERN_SPEC = {
    "key": "xor_aug",
    "label": "xor_aug (noisy XOR)",
    "make": make,
}
