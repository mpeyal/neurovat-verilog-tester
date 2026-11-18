"""Drive a Python twin with the same PWL stimulus that Spectre receives.

    twin = load_twin("vteam", workspace)            # key from twins/<file>.py TWIN_SPEC
    res = run_twin(twin, stim, dt_max=1e-10, params={"x_init": 0.0})
    res["time"], res["V"], res["I"], res["G"]

The grid holds every PWL corner, so a piecewise-linear drive is reproduced
exactly.  A twin with `step_linear(t, dt, v0, v1)` gets the linear drive;
otherwise `step(t, dt, drive)` gets the mid-interval value.  The terminal
current is `model.current(V)` when the twin has it, else V * G; for a current
stimulus the terminal voltage is I / G.
"""

import os
from dataclasses import fields

import numpy as np

from ..twin_loader import load_twins


def builtin_spec(key):
    """The built-in ECFET models (simulation_core v1/v2/v3) as a TWIN_SPEC-like
    dict, or None."""
    from ..registry import SPEC_BY_KEY
    s = SPEC_BY_KEY.get(key)
    if s is None:
        return None
    return {"key": s.key, "label": s.label, "model_class": s.cls,
            "params_class": s.params_cls, "input_kind": s.input_kind,
            "device": "ecfet"}


def load_twin(key, workspace):
    """Return the TWIN_SPEC dict whose key (or file stem) is `key`: a built-in
    model (v1, v2, v3) or a twin in the workspace's twins/ folder.

    Twins are executed on load (see the security note in twin_loader): only
    the trusted twins/ folder of the workspace is read."""
    b = builtin_spec(key)
    if b is not None:
        return b
    found, errors = [], []
    for path, payload, err in load_twins(os.path.join(workspace, "twins")):
        stem = os.path.splitext(os.path.basename(path))[0]
        if payload is None:
            errors.append(f"{stem}: {err}")
            continue
        mod, spec = payload
        if key in (spec.get("key"), stem):
            found.append(spec)
    if not found:
        extra = ("; load errors: " + "; ".join(errors)) if errors else ""
        raise KeyError(f"no twin with key {key!r} in {workspace}/twins{extra}")
    return found[0]


def make_model(spec, params=None):
    pcls = spec["params_class"]
    names = {f.name for f in fields(pcls)}
    bad = sorted(set(params or {}) - names)
    if bad:
        raise KeyError(f"{spec['key']}: unknown parameter(s) {bad}")
    return spec["model_class"](pcls(**(params or {})))


def run_twin(spec, stim, dt_max, params=None):
    kind = spec.get("input_kind", "voltage")
    if kind != stim.kind:
        raise ValueError(f"twin {spec['key']!r} takes a {kind} drive, "
                         f"the stimulus is a {stim.kind}")
    model = make_model(spec, params)
    model.reset()
    ts = stim.grid(dt_max)
    u = stim.value(ts)
    n = len(ts)
    out = {"time": ts, "V": np.empty(n), "I": np.empty(n), "G": np.empty(n)}
    linear = hasattr(model, "step_linear")
    has_current = hasattr(model, "current")

    def record(k):
        g = float(model.G)
        if kind == "voltage":
            v = u[k]
            i = float(model.current(v)) if has_current else v * g
        else:
            i = u[k]
            v = i / g if g else float("nan")
        out["V"][k], out["I"][k], out["G"][k] = v, i, g

    record(0)
    for k in range(n - 1):
        dt = ts[k + 1] - ts[k]
        if linear:
            model.step_linear(ts[k], dt, u[k], u[k + 1])
        else:
            model.step(ts[k], dt, 0.5 * (u[k] + u[k + 1]))
        record(k + 1)
    return out
