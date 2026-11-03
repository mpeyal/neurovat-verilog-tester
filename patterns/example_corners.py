"""Example PATTERN plugin: a 4-class "corners" stimulus (one lit quadrant each).

Demonstrates the patterns/ plugin contract with a task that trains out-of-the-box
on the DEFAULT settings (device-local STDP, single crossbar) - linearly separable,
so no hidden layer or surrogate needed. Pick "corners" in the PATTERNS dropdown,
Build, Train.

Copy this file (or edit it live) and a new entry appears in the dropdown with no
restart - the folder is hot-watched.
"""

import numpy as np


def make(cfg):
    """Return (patterns, targets); each class lights one 3x3 quadrant of a 6x6
    grid (top-left, top-right, bottom-left, bottom-right)."""
    cfg.grid_h, cfg.grid_w, cfg.n_out = 6, 6, 4
    quads = [("TL", 0, 0), ("TR", 0, 3), ("BL", 3, 0), ("BR", 3, 3)]
    pats, targets = [], []
    for c, (name, r0, c0) in enumerate(quads):
        g = np.zeros((6, 6), np.float32)
        g[r0:r0 + 3, c0:c0 + 3] = 1.0
        pats.append((name, g.reshape(-1)))
        targets.append(c)
    return pats, targets


PATTERN_SPEC = {
    "key": "corners",
    "label": "corners",
    "make": make,
}
