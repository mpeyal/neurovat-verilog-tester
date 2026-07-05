"""Real-engine adapter — routes Studio's run_sim to the actual NeuroVAT physics.

The parent repo (one level above studio/) contains the real `ecfet` package:
the EcfetV2 / EcfetV3 / FeFET behavioural twins and the transient simulator
that the Dear PyGui app uses. When that package is importable, Studio's plots
come from the SAME engine as the desktop app; when it isn't (studio/ copied
out on its own), bridge.py falls back to the analytic twin in twin.py.

Conventions (verified against the models):
  * UI pulse rows are [t_start_s, width_s, amplitude].
  * ECFET v2/v3 are current-driven: UI amplitude is in pA (x1e-12 A);
    positive current potentiates (paper polarity = -1).
  * FeFET is voltage-driven: UI amplitude x 0.01 -> gate volts, so the
    UI's +/-300 range spans +/-3 V around Vc = 0.6 V.
"""

import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ecfet = None

# Canonical display-unit -> SI table, shared with the desktop app so the two can
# never drift. amp(display) x UNIT_SCALE[unit] = SI drive (A for current-driven
# ECFET, V for voltage-driven FeFET) — identical to vatester/signal_factory.py.
try:
    from vatester.signal_factory import UNIT_SCALE as _UNIT_SCALE
except Exception:
    _UNIT_SCALE = {"pA": 1e-12, "nA": 1e-9, "uA": 1e-6, "mA": 1e-3, "A": 1.0,
                   "mV": 1e-3, "V": 1.0}


def _drive_scale(unit, device):
    """amp(display) -> SI. Authoritative when the UI sends a known unit; the
    legacy device-based factor is only a fallback for old clients that don't."""
    if unit in _UNIT_SCALE:
        return _UNIT_SCALE[unit]
    return 0.01 if device == "fefet" else 1e-12


def _load():
    """Import the parent repo's ecfet package once; None if unavailable."""
    global _ecfet
    if _ecfet is None:
        try:
            if _REPO_ROOT not in sys.path:
                sys.path.insert(0, _REPO_ROOT)
            import ecfet
            _ecfet = ecfet
        except Exception:
            _ecfet = False
    return _ecfet or None


def available():
    return _load() is not None


def repo_root():
    return _REPO_ROOT


_PARAM_RE = re.compile(
    r"parameter\s+(?:real|integer)\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")


def _apply_va_params(params, va_text):
    """Apply EVERY `parameter real/integer NAME = VALUE;` from the .va text onto
    the model dataclass, wherever the field exists — matching the desktop app's
    full-parameter behaviour (vatester/app.py::_build_models does params_cls(**pv)).
    Names the model doesn't have, or values that aren't plain numbers (e.g. the
    `1.0/10e3` expression defaults), are skipped so the dataclass default stands."""
    if not va_text:
        return
    for m in _PARAM_RE.finditer(va_text):
        name, raw = m.group(1), m.group(2).strip()
        if not hasattr(params, name):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue                      # expression / non-numeric -> keep default
        cur = getattr(params, name)
        if isinstance(cur, int) and not isinstance(cur, bool):
            val = int(round(val))
        try:
            setattr(params, name, val)
        except (TypeError, ValueError):
            pass


def _make_model(ecfet, device, essentials, va_text=None):
    """Build the requested device model. Full .va params are applied first (so
    All-parameters edits take effect), then the live Essentials sliders override
    on top (the slider is authoritative over the text it also patches)."""
    es = essentials or {}
    if device == "fefet":
        params, model_cls = ecfet.FeFETParams(), ecfet.FeFET
    elif device == "v3":
        params, model_cls = ecfet.V3Params(), ecfet.EcfetV3
    else:
        params, model_cls = ecfet.V2Params(), ecfet.EcfetV2

    # 1) full parameter set parsed from the .va editor buffer
    _apply_va_params(params, va_text)

    # 2) live Essentials sliders override (only where the dataclass has the field)
    mapping = {"n_states": ("n_states",), "kappa_v": ("kappa_v",),
               "nu": ("nu_p", "nu_d"), "sigma": ("sigma_c2c",)}
    for key, fields in mapping.items():
        if key in es:
            for f in fields:
                if hasattr(params, f):
                    try:
                        setattr(params, f, float(es[key]))
                    except (TypeError, ValueError):
                        pass
    return model_cls(params)


def _decimate(t, y, max_pts=1600):
    n = len(t)
    if n <= max_pts:
        return [[float(a), float(b)] for a, b in zip(t, y)]
    step = n / float(max_pts)
    idx = sorted({int(k * step) for k in range(max_pts)} | {n - 1})
    return [[float(t[i]), float(y[i])] for i in idx]


def run_sim(pulses, essentials=None, gen="train", device="v2", va=None, unit=None, **_):
    """Same contract as twin.simulate(): {stim, gts, ana, Gfinal, engine}."""
    ecfet = _load()
    if ecfet is None:
        raise RuntimeError("real engine unavailable (ecfet package not importable)")

    amp_to_drive = _drive_scale(unit, device)   # display units -> SI (shared table)
    rows = []
    for p in pulses or []:
        ts, wd, a = float(p[0]), float(p[1]), float(p[2])
        if wd > 0:
            rows.append((ts, wd, a * amp_to_drive))
    if not rows:
        raise RuntimeError("no valid pulses")

    model = _make_model(ecfet, device, essentials, va)
    wf = ecfet.Waveform(rows)
    t_last_end = max(t0 + w for t0, w, _a in rows)
    t_stop = t_last_end + 2.0
    res = ecfet.simulate(model, wf, t_stop, label="studio")

    # stimulus staircase in the UI's own amplitude units (for the top plot)
    stim = [[0.0, 0.0]]
    for p in pulses:
        ts, wd, a = float(p[0]), float(p[1]), float(p[2])
        stim += [[ts, 0.0], [ts, a], [ts + wd, a], [ts + wd, 0.0]]
    stim.append([t_stop, 0.0])

    gts = _decimate(res.t, res.G * 1e6)

    # per-pulse retained conductance: sample just before the next pulse
    is_ltp = gen == "ltpltd"
    half = len(pulses) / 2.0
    ana = []
    starts = [float(p[0]) for p in pulses]
    for i, p in enumerate(pulses):
        ts = float(p[0])
        gap = (starts[i + 1] - ts) if i + 1 < len(starts) else 1.0
        _r, g = res.at([ts + gap * 0.95])
        ana.append({"i": i + 1, "g": float(g[0]) * 1e6,
                    "branch": 1 if (is_ltp and i >= half) else 0})

    g_nv = res.extras.get("G_nv (S)")
    g_final = float(g_nv[-1]) * 1e6 if g_nv is not None else float(res.G[-1]) * 1e6
    return {"stim": stim, "gts": gts, "ana": ana, "Gfinal": g_final,
            "engine": "ecfet-" + device}
