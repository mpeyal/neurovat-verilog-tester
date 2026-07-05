"""Adapter — real device-measured STDP window + FeFET polarization loop.

Reuses vatester.analysis (the SAME correlation-isolated STDP sweep and P-V loop
the desktop app plots) driven by the real ecfet model built from the UI's .va,
so these plots are measured from the device twin, not analytic placeholders.
"""

import math

from . import engine as _engine


def _model(device, va):
    ecfet = _engine._load()
    if ecfet is None:
        raise RuntimeError("real engine unavailable")
    return ecfet, _engine._make_model(ecfet, device or "v2", None, va)


def stdp_window(device=None, va=None, amp=None, width=0.01, dt_max_ms=1800, n=18, **_):
    """Real anti-symmetric STDP window measured from the device: for each Δt run
    the pre/post pair (correlation-isolated) and read the retained ΔG (ECFET) or
    ΔVt (FeFET). Contract: {engine, device, points:[[dt_ms, dY]], label, unit}."""
    from vatester import analysis
    ecfet, model = _model(device, va)
    is_v = getattr(model, "input_kind", "current") == "voltage"
    if amp is None:
        amp = 1.5 if is_v else 170e-12          # SI: V (FeFET) / A (ECFET)
    dt_max = max(float(dt_max_ms) / 1000.0, 0.02)
    n = max(6, min(40, int(n)))
    lo = math.log10(0.01)                       # 10 ms
    hi = math.log10(dt_max)
    mags = [10 ** (lo + (hi - lo) * i / (n - 1)) for i in range(n)]
    dts = [-m for m in reversed(mags)] + [float(m) for m in mags]
    curves = analysis.stdp_sweep([model], -amp, amp, width, dts, tail=1.2)
    label, unit, _scale = analysis.state_profile(model)
    ys = next(iter(curves.values()), [])
    pts = [[dts[i] * 1000.0, float(ys[i])] for i in range(len(ys))]
    return {"engine": "ecfet", "device": device, "points": pts,
            "label": label, "unit": unit}


def polarization(device=None, va=None, v_amp=3.0, **_):
    """Real FeFET P-V hysteresis loop (triangular gate-voltage sweep, polarization
    read from the model). ECFET has no polarization -> {available: False}.
    Contract: {engine, device, points:[[V, P]], unit, available}."""
    from vatester import analysis
    ecfet, model = _model(device, va)
    out = analysis.polarization_loop([model], v_amp=float(v_amp))
    if not out:                                 # ECFET (no P) -> not available
        return {"engine": "ecfet", "device": device, "points": [], "available": False}
    d = next(iter(out.values()))
    pts = [[float(d["V"][i]), float(d["P"][i])] for i in range(len(d["V"]))]
    return {"engine": "ecfet", "device": device, "points": pts,
            "unit": d.get("unit", "norm."), "available": True}
