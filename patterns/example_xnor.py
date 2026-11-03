"""Example PATTERN plugin: the 2-input XNOR logic gate (output 1 when A == B).

This file is NOT part of the app source - it lives in patterns/ and the trainer
registers it as a selectable PATTERNS source automatically, with no GUI edit and
no restart (the folder is hot-watched).  Copy it to add your own gate/stimulus.

Like XOR, XNOR is NOT linearly separable, so it CANNOT be learned by a single
crossbar - build with a hidden layer (HIDDEN LAYERS e.g. "8") and the
Surrogate-grad (BPTT) rule. Even then these 2-bit gates are a notoriously fiddly
optimisation target and often need tuning / a few re-inits to reach 100%; see
patterns/example_corners.py for a task that trains cleanly out of the box.
"""

import numpy as np


def _v(a, b):
    # complementary [A, ~A, B, ~B] coding on a 2x2 grid (bias-free margin)
    return np.array([float(a), 1.0 - a, float(b), 1.0 - b], np.float32)


def make(cfg):
    """Return (patterns, targets) and set the grid/class count for XNOR."""
    cfg.grid_h, cfg.grid_w, cfg.n_out = 2, 2, 2
    table = [((0, 0), 1), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1),
             ((0, 0), 1), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]
    pats = [(f"A={a} B={b} -> {y}", _v(a, b)) for (a, b), y in table]
    targets = [y for _, y in table]
    return pats, targets


PATTERN_SPEC = {
    "key": "xnor",
    "label": "xnor",
    "make": make,
}
