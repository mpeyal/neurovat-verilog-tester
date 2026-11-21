"""Test sets from a file: run a fixed, published stimulus set instead of the
automatic one (e.g. the sets of a paper's reproducibility package).

A test set is a JSON file (data only - nothing in it is executed):

    {
      "format": "neurovat-testset-1",
      "name": "vteam-1",
      "description": "...",
      "profile": "memristor",              # ecfet | memristor | fecap | fefet
      "twin": "vteam",                     # optional: the twin it was made for
      "steps": [1e-10, 1e-11, 1e-12],      # Spectre maxsteps; the finest is the
                                           # reference; one step = no Layer 2
      "v_read": 0.01,                      # optional, ECFET channel read bias (V)
      "state": {                           # optional: compare the internal state
        "label": "x (m)",
        "spectre_save": "X1.ws",           # a saved node ...
        "spectre_scale": 3e-9,             #   ... times this
        "spectre_clip": [0, 1],            #   (optional, applied before scaling)
        "spectre_from_iv": {"I0": .., "g0": .., "V0": .., "v_min": 0.05},
                                           # ... or the Stanford-PKU gap from I, V
        "twin_obs": "x (nm)", "twin_scale": 1e-9
      },
      "stimuli": [
        {"name": "off_train", "description": "...", "kind": "voltage",
         "t": [0, ...], "v": [0, ...],     # piecewise-linear drive (s, V or A)
         "reads": [2.5e-08, ...],          # optional read times (s)
         "instance": {"x_init": 3e-9},     # optional start state / parameters
         "steps": [...],                   # optional per-stimulus Spectre steps
         "twin_dt": 1e-10}                 # optional twin step (then also /2)
      ]
    }

The same pass rule as the automatic sets applies; with a "state" the state
NRMSE (on the twin's grid, where both are defined, relative to the
reference's range) must also be <= 1 %.
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .stimulus import Stimulus

FORMAT = "neurovat-testset-1"
PROFILES = ("ecfet", "memristor", "fecap", "fefet")
MAX_BYTES = 64 * 1024 * 1024
_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,80}$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_SIGNAL = re.compile(r"^[A-Za-z0-9_:.]{1,64}$")


class TestSetError(ValueError):
    """The file is not a valid test set."""


@dataclass
class TestSet:
    name: str
    profile: str
    stimuli: list
    steps: list
    description: str = ""
    twin: str = None
    v_read: float = None
    state: dict = None
    path: str = ""
    meta: dict = field(default_factory=dict)


def _pos_steps(x, where):
    if not isinstance(x, list) or not x or len(x) > 6:
        raise TestSetError(f"{where}: 'steps' must be a list of 1 to 6 numbers")
    out = sorted({float(s) for s in x}, reverse=True)
    if not all(math.isfinite(s) and 1e-15 <= s <= 10.0 for s in out):
        raise TestSetError(f"{where}: steps must lie between 1e-15 and 10 s")
    return out


def _number_dict(d, where, keys=None):
    if not isinstance(d, dict):
        raise TestSetError(f"{where} must be an object")
    out = {}
    for k, v in d.items():
        if not _IDENT.match(str(k)) or (keys and k not in keys):
            raise TestSetError(f"{where}: bad key {k!r}")
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            raise TestSetError(f"{where}: {k} must be a number") from None
        if not math.isfinite(out[k]):
            raise TestSetError(f"{where}: {k} must be finite")
    return out


def _state(d):
    if d is None:
        return None
    if not isinstance(d, dict):
        raise TestSetError("'state' must be an object")
    out = {"label": str(d.get("label", "state"))[:40]}
    obs = d.get("twin_obs")
    if not isinstance(obs, str) or not obs or len(obs) > 64:
        raise TestSetError("state: 'twin_obs' (the twin observable) is required")
    out["twin_obs"] = obs
    out["twin_scale"] = float(d.get("twin_scale", 1.0))
    if "spectre_save" in d:
        if not _SIGNAL.match(str(d["spectre_save"])):
            raise TestSetError("state: 'spectre_save' is not a signal name")
        out["spectre_save"] = str(d["spectre_save"])
        out["spectre_scale"] = float(d.get("spectre_scale", 1.0))
        clip = d.get("spectre_clip")
        if clip is not None:
            if not (isinstance(clip, list) and len(clip) == 2):
                raise TestSetError("state: 'spectre_clip' must be [low, high]")
            out["spectre_clip"] = [float(clip[0]), float(clip[1])]
    elif "spectre_from_iv" in d:
        out["spectre_from_iv"] = _number_dict(d["spectre_from_iv"], "state.spectre_from_iv",
                                              {"I0", "g0", "V0", "v_min"})
        if not {"I0", "g0", "V0"} <= set(out["spectre_from_iv"]):
            raise TestSetError("state.spectre_from_iv needs I0, g0 and V0")
    else:
        raise TestSetError("state: give 'spectre_save' or 'spectre_from_iv'")
    return out


def load_test_set(path):
    """Read and check a test-set file -> TestSet (raises TestSetError)."""
    p = Path(path)
    if not p.is_file():
        raise TestSetError(f"no test set file {p}")
    if p.stat().st_size > MAX_BYTES:
        raise TestSetError(f"{p.name}: larger than {MAX_BYTES // 2**20} MB")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise TestSetError(f"{p.name}: not valid JSON ({e})") from None
    if not isinstance(d, dict) or d.get("format") != FORMAT:
        raise TestSetError(f"{p.name}: 'format' must be {FORMAT!r}")
    name = str(d.get("name", p.stem))
    if not _NAME.match(name):
        raise TestSetError(f"{p.name}: bad set name {name!r}")
    profile = d.get("profile")
    if profile not in PROFILES:
        raise TestSetError(f"{p.name}: 'profile' must be one of {PROFILES}")
    steps = _pos_steps(d.get("steps"), p.name)
    raw = d.get("stimuli")
    if not isinstance(raw, list) or not raw or len(raw) > 200:
        raise TestSetError(f"{p.name}: 'stimuli' must be a list of 1 to 200 stimuli")
    stimuli, seen = [], set()
    for k, s in enumerate(raw):
        where = f"{p.name} stimulus {k + 1}"
        if not isinstance(s, dict):
            raise TestSetError(f"{where}: must be an object")
        sname = str(s.get("name", ""))
        if not _NAME.match(sname) or sname in seen:
            raise TestSetError(f"{where}: missing, bad or repeated name {sname!r}")
        seen.add(sname)
        kind = s.get("kind", "current" if profile == "ecfet" else "voltage")
        try:
            st = Stimulus(s.get("t"), s.get("v"), kind=kind, name=sname,
                          meta={"description": str(s.get("description", ""))[:300]})
        except (TypeError, ValueError) as e:
            raise TestSetError(f"{where}: {e}") from None
        if len(st.t) > 2_000_000:
            raise TestSetError(f"{where}: too many PWL points")
        reads = s.get("reads")
        if reads is not None:
            r = np.asarray(reads, dtype=float)
            if r.ndim != 1 or not np.all(np.isfinite(r)) or \
                    (len(r) and (r.min() < 0 or r.max() > st.t_stop)):
                raise TestSetError(f"{where}: reads must be times within the stimulus")
            st.reads = r
        if s.get("instance"):
            st.meta["instance"] = _number_dict(s["instance"], f"{where} instance")
        if s.get("steps") is not None:
            st.meta["steps"] = _pos_steps(s["steps"], where)
        if s.get("twin_dt") is not None:
            st.meta["twin_dt"] = _pos_steps([s["twin_dt"]], where)[0]
        stimuli.append(st)
    twin = d.get("twin")
    v_read = d.get("v_read")
    return TestSet(name=name, profile=profile, stimuli=stimuli, steps=steps,
                   description=str(d.get("description", ""))[:500],
                   twin=str(twin) if twin else None,
                   v_read=float(v_read) if v_read is not None else None,
                   state=_state(d.get("state")), path=str(p),
                   meta={"version": d.get("version", name)})


def list_test_sets(workspace):
    """The *.json test sets in <workspace>/testsets (paths, sorted)."""
    root = Path(workspace) / "testsets"
    return sorted(str(p) for p in root.glob("*.json")) if root.is_dir() else []


# ---------------------------------------------------------------- the state
def spectre_state(sig_get, state, ref):
    """The reference state series (NaN where it is not defined)."""
    if "spectre_save" in state:
        y = np.asarray(sig_get(state["spectre_save"]), float)
        if "spectre_clip" in state:
            y = np.clip(y, *state["spectre_clip"])
        return y * state.get("spectre_scale", 1.0)
    p = state["spectre_from_iv"]
    v, i = np.asarray(ref["V"], float), np.asarray(ref["I"], float)
    out = np.full(len(v), math.nan)
    ok = (np.abs(v) >= p.get("v_min", 0.05)) & (i * v > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = -p["g0"] * np.log(np.abs(i[ok]) / (p["I0"] * np.abs(np.sinh(v[ok] / p["V0"]))))
    return out


def state_metrics(t_tw, x_tw, t_ref, x_ref):
    """State NRMSE / max error on the twin's grid clipped to the common span,
    where both are defined, relative to the reference's range there; NaN if
    the state does not move (as the paper's validation scripts)."""
    t_tw, x_tw = np.asarray(t_tw, float), np.asarray(x_tw, float)
    t_ref, x_ref = np.asarray(t_ref, float), np.asarray(x_ref, float)
    lo, hi = max(t_tw[0], t_ref[0]), min(t_tw[-1], t_ref[-1])
    m = (t_tw >= lo) & (t_tw <= hi)
    xt, xr = x_tw[m], np.interp(t_tw[m], t_ref, x_ref)
    ok = np.isfinite(xt) & np.isfinite(xr)
    if ok.sum() <= 2:
        return {"nrmse": math.nan, "maxerr": math.nan, "range": math.nan}
    xt, xr = xt[ok], xr[ok]
    rng = float(np.max(xr) - np.min(xr))
    if not rng > 0:
        return {"nrmse": math.nan, "maxerr": math.nan, "range": rng}
    d = xt - xr
    return {"nrmse": float(np.sqrt(np.mean(d * d)) / rng),
            "maxerr": float(np.max(np.abs(d)) / rng), "range": rng}
