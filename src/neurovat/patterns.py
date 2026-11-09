"""Built-in input pattern banks (bars, letters, digits, random)."""
import numpy as np


def make_patterns(name, gh, gw):
    """Built-in input pattern bank rendered onto a gh x gw pixel grid.
    Returns [(label, vec01)] with vec01 a flat (gh*gw,) array in [0, 1]."""
    pats = []

    def blank():
        return np.zeros((gh, gw), dtype=float)

    if name == "bars":
        # vertical, horizontal and the two diagonals - the classic
        # orientation-selectivity stimulus set
        for c in (0, gw // 2, gw - 1):
            g = blank(); g[:, c] = 1.0
            pats.append((f"| col {c}", g))
        for r in (0, gh // 2, gh - 1):
            g = blank(); g[r, :] = 1.0
            pats.append((f"- row {r}", g))
        g = blank()
        for k in range(min(gh, gw)):
            g[k, k] = 1.0
        pats.append(("\\ diag", g))
        g = blank()
        for k in range(min(gh, gw)):
            g[k, gw - 1 - k] = 1.0
        pats.append(("/ diag", g))
    elif name == "letters":
        glyphs = _letter_glyphs()
        for ch, grid in glyphs:
            pats.append((ch, _fit(grid, gh, gw)))
    elif name == "digits":
        for d, grid in _digit_glyphs():
            pats.append((d, _fit(grid, gh, gw)))
    else:  # "random"
        rng = np.random.default_rng(0)
        for k in range(6):
            g = (rng.random((gh, gw)) > 0.6).astype(float)
            pats.append((f"rand {k}", g))

    return [(lbl, g.reshape(-1)) for lbl, g in pats]


def _fit(grid, gh, gw):
    """Nearest-neighbour fit of a bitmap to gh x gw."""
    grid = np.asarray(grid, dtype=float)
    h, w = grid.shape
    rs = (np.linspace(0, h - 1, gh)).round().astype(int)
    cs = (np.linspace(0, w - 1, gw)).round().astype(int)
    return grid[np.ix_(rs, cs)]


def _letter_glyphs():
    L = ["10000", "10000", "10000", "10000", "11111"]
    T = ["11111", "00100", "00100", "00100", "00100"]
    X = ["10001", "01010", "00100", "01010", "10001"]
    O = ["01110", "10001", "10001", "10001", "01110"]
    return [(ch, _bits(rows)) for ch, rows in
            (("L", L), ("T", T), ("X", X), ("O", O))]


def _digit_glyphs():
    one = ["00100", "01100", "00100", "00100", "01110"]
    seven = ["11111", "00010", "00100", "01000", "01000"]
    three = ["11110", "00011", "01110", "00011", "11110"]
    zero = ["01110", "10011", "10101", "11001", "01110"]
    return [(ch, _bits(rows)) for ch, rows in
            (("0", zero), ("1", one), ("3", three), ("7", seven))]


def _bits(rows):
    return np.array([[1.0 if c == "1" else 0.0 for c in r] for r in rows])
