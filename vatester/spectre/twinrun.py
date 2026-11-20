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
import time
from dataclasses import fields

import numpy as np

from ..twin_loader import load_twins


def builtin_spec(key):
    """The built-in ECFET models (simulation_core v1/v2/v3) as a TWIN_SPEC-like
    dict, or None."""
    from ..registry import SPEC_BY_KEY, TWIN_FILE
    s = SPEC_BY_KEY.get(key)
    if s is None:
        return None
    # The GUI also registers the twins/ files it found at startup; those must
    # be read fresh from their file (load_twin), or an edit made while the app
    # runs - e.g. by "Fix twin with AI" - would be scored with the old code.
    if not str(TWIN_FILE.get(key, "")).replace("\\", "/").startswith("simulation_core/"):
        return None
    return {"key": s.key, "label": s.label, "model_class": s.cls,
            "params_class": s.params_cls, "input_kind": s.input_kind,
            **({"device": "ecfet", "device_class": "ECFET"} if s.key in ("v1", "v2", "v3")
               else {"device_class": s.key.upper()})}


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


def seconds_per_step(spec, stim, dt_max, params=None, n=200):
    """Measured wall time of one twin step on the start of `stim` (a short
    probe for the time-remaining estimate)."""
    model = make_model(spec, params)
    model.reset()
    ts = stim.grid(dt_max)[:n + 1]
    u = stim.value(ts)
    linear = hasattr(model, "step_linear")
    t0 = time.perf_counter()
    for k in range(len(ts) - 1):
        if linear:
            model.step_linear(ts[k], ts[k + 1] - ts[k], u[k], u[k + 1])
        else:
            model.step(ts[k], ts[k + 1] - ts[k], 0.5 * (u[k] + u[k + 1]))
        float(model.G)
    return (time.perf_counter() - t0) / max(1, len(ts) - 1)


def run_twin(spec, stim, dt_max, params=None, progress=None):
    """progress(n_steps, seconds) is called every few thousand steps with the
    steps done and the time they took since the previous call."""
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
    obs0 = model.observables() if hasattr(model, "observables") else {}
    obs_keys = [k for k, v in obs0.items() if isinstance(v, (int, float))]
    for k in obs_keys:
        out["obs:" + k] = np.empty(n)

    def record(k):
        if obs_keys:
            ob = model.observables()
            for name in obs_keys:
                out["obs:" + name][k] = float(ob[name])
        g = float(model.G)
        if kind == "voltage":
            v = u[k]
            i = float(model.current(v)) if has_current else v * g
        else:
            i = u[k]
            v = i / g if g else float("nan")
        out["V"][k], out["I"][k], out["G"][k] = v, i, g

    record(0)
    t_last, k_last = time.perf_counter(), 0
    for k in range(n - 1):
        dt = ts[k + 1] - ts[k]
        if linear:
            model.step_linear(ts[k], dt, u[k], u[k + 1])
        else:
            model.step(ts[k], dt, 0.5 * (u[k] + u[k + 1]))
        record(k + 1)
        if progress is not None and k - k_last >= 2000:
            now = time.perf_counter()
            progress(k - k_last, now - t_last)
            t_last, k_last = now, k
    if progress is not None:
        progress(max(0, n - 1 - k_last), time.perf_counter() - t_last)
    return out
