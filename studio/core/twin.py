"""Behavioural twin — the default physics engine.

This is a faithful 1:1 Python port of the JavaScript twin baked into the
front-end (the `_simulate` method). Keeping them identical means the plots look
the same whether the page runs standalone (JS twin) or behind this backend
(Python twin) — so you can wire the real simulator underneath one function at a
time without the UI jumping around.

`simulate()` returns exactly the shape the front-end's run_sim contract expects:
    { "stim": [[t, v], ...],       # stimulus staircase
      "gts":  [[t, G_uS], ...],    # conductance trace (volatile + slow pool)
      "ana":  [{"i", "g", "branch"}, ...],  # per-pulse retained conductance
      "Gfinal": float }            # final non-volatile conductance (uS)
"""

import math

GMIN, GMAX, G0, TAU3 = 100.0, 1150.0, 227.0, 19.0  # uS range, init, slow-pool tau (s)


def simulate(pulses, essentials=None, gen="train", device="v2", **_):
    es = essentials or {}
    n_states = float(es.get("n_states", 650))
    kappa = float(es.get("kappa_v", 0.68))     # volatile fraction of each write
    nu = float(es.get("nu", 0.02))             # soft-bound curvature
    sigma = float(es.get("sigma", 0.0))        # cycle-to-cycle write noise
    dGu = (GMAX - GMIN) / max(1.0, n_states)   # conductance step per unit charge

    Gnv, pool, t_prev, seed = G0, 0.0, 0.0, 7

    def rand():                                # deterministic LCG, matches JS
        nonlocal seed
        seed = (seed * 16807) % 2147483647
        return seed / 2147483647 - 0.5

    stim = [[0.0, 0.0]]
    gts = [[0.0, Gnv]]
    ana = []
    is_ltp = gen == "ltpltd"
    half = len(pulses) / 2.0

    for i, p in enumerate(pulses):
        ts, wd, a = float(p[0]), float(p[1]), float(p[2])
        stim += [[ts, 0.0], [ts, a], [ts + wd, a], [ts + wd, 0.0]]
        pool *= math.exp(-(ts - t_prev) / TAU3)
        t_prev = ts
        x = (Gnv - GMIN) / (GMAX - GMIN)
        sgn = 1.0 if a > 0 else -1.0
        win = math.exp(-nu * 20.0 * (x if sgn > 0 else 1.0 - x))
        q = abs(a) * wd * 1000.0 / 500.0
        dG = sgn * dGu * q * win * (1.0 + sigma * rand() * 2.0)
        Gnv = min(GMAX, max(GMIN, Gnv + (1.0 - kappa) * dG))
        pool += kappa * 0.5 * dG
        gts.append([ts + wd, Gnv + pool])
        gap = (pulses[i + 1][0] - ts) if i + 1 < len(pulses) else 1.0
        t_s = ts + gap * 0.95
        p_then = pool * math.exp(-(t_s - ts) / TAU3)
        gts.append([t_s, Gnv + p_then])
        ana.append({"i": i + 1, "g": Gnv + p_then,
                    "branch": 1 if (is_ltp and i >= half) else 0})

    t_end = (pulses[-1][0] if pulses else 0.2) + 2.0
    stim.append([t_end, 0.0])
    gts.append([t_end, Gnv + pool * math.exp(-2.0 / TAU3)])
    return {"stim": stim, "gts": gts, "ana": ana, "Gfinal": Gnv, "engine": "twin"}


def default_va(cell="ecfet_v2"):
    """The bundled sample Verilog-A source, returned when no real file/cell exists."""
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "va", cell + ".va")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    path = os.path.join(here, "va", "ecfet_v2.va")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "// %s.va not found\n" % cell
