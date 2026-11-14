"""Shared plumbing for vteam-1: twin loading, the PWL time grid, twin runs,
the canonical CSV schema, read-out metrics and the pass rule.

Mirrors validation/stanford/core.py (same grid, same metric, same limits);
re-uses validation/common.py and validation/compare.py unchanged.
"""

import importlib.util
import math
import time
from pathlib import Path

import numpy as np

from validation import common, compare
from validation.stanford import core as stanford_core
from validation.vteam import stimuli as S

DATA = common.DATA / "vteam"                   # validation/data/vteam/
FIGURES = common.FIGURES / "vteam"
DECKS = common.VDIR / "spice" / "spectre_vteam"
STEP_DECKS = common.VDIR / "spice" / "spectre_vteam_step"   # make_decks --step <s>
TWIN_PATH = common.ROOT / "twins" / "vteam.py"
VA_NAME = "vteam.va"
VA_PATH = common.ROOT / "verilog" / VA_NAME
MODULE = "vteam"
MODEL = "vteam"

# Canonical columns (twin and Spectre are both written in this order).
#   V_V       V(p,n)
#   I_A       current into p
#   x_m       state clipped to [0, d] (Spectre: d * clip(V(X1.ws)); if the
#             internal node is not saved, inverted from R = V/I where |V| >= V_X_MIN)
#   s_state   the integrator state s = x/d, unclipped (Spectre: V(X1.ws))
#   G_read_S  1/R(x)
COLUMNS = ["t_s", "V_V", "I_A", "x_m", "s_state", "G_read_S"]
V_X_MIN = 5e-3             # V; below this |V| the state is not inverted from I

# pass rule: the same limits as stanford-1 (validation/stanford/core.py)
G_MAX_LIMIT = stanford_core.G_MAX_LIMIT
G_MEAN_LIMIT = stanford_core.G_MEAN_LIMIT
I_NRMSE_LIMIT = stanford_core.I_NRMSE_LIMIT
X_NRMSE_LIMIT = stanford_core.GAP_NRMSE_LIMIT

_twin_mod = None


def twin_module():
    """Import twins/vteam.py by path (trusted repo file)."""
    global _twin_mod
    if _twin_mod is None:
        spec = importlib.util.spec_from_file_location("nvat_validation_vteam_twin", TWIN_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _twin_mod = mod
    return _twin_mod


def params(stim=None, **overrides):
    m = twin_module()
    kw = dict(stim.instance_dict()) if stim is not None else {}
    kw.update(overrides)
    return m.VTEAMParams(**kw)


def new_twin(stim=None, **overrides):
    return twin_module().VTEAM(params(stim, **overrides))


# --------------------------------------------------------------------------
grid = stanford_core.grid      # every PWL corner + uniform steps <= dt_max


def simulate_twin(model, stim, ts):
    """Step the twin over ts with V linear between grid points (exactly the
    PWL, because every PWL corner is a grid point)."""
    n = len(ts)
    model.reset()
    v = stim.v(ts)
    d = model.p.d
    cols = {c: np.empty(n) for c in COLUMNS}

    def record(k):
        cols["t_s"][k] = ts[k]
        cols["V_V"][k] = v[k]
        cols["I_A"][k] = model.current(v[k])
        cols["x_m"][k] = model.x_clamped
        cols["s_state"][k] = model.x / d
        cols["G_read_S"][k] = model.G

    record(0)
    for k in range(n - 1):
        model.step_linear(ts[k], ts[k + 1] - ts[k], v[k], v[k + 1])
        record(k + 1)
    return cols


def run_twin(stim, refine=1, **overrides):
    ts = common.refine_grid(grid(stim), refine)
    model = new_twin(stim, **overrides)
    t0 = time.perf_counter()
    cols = simulate_twin(model, stim, ts)
    wall = time.perf_counter() - t0
    return cols, wall, model.n_pieces


# --------------------------------------------------------------------------
def provenance_line(source):
    return (f"# source: {source}; model: {MODEL}; stimulus_set: "
            f"{S.STIMULUS_SET_VERSION} {S.fingerprint()[:16]}")


def write_csv(path, cols, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([np.asarray(cols[c], dtype=float) for c in COLUMNS])
    lines = [provenance, ",".join(COLUMNS)]
    for row in data:
        lines.append(",".join("nan" if math.isnan(v) else f"{v:.10g}" for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def data_path(source, name, ext=".csv", root=None):
    return Path(root or DATA) / source / f"{name}{ext}"


def load(source, name, root=None):
    p = data_path(source, name, root=root)
    if not p.is_file():
        return None
    cols, _ = common.read_csv(p)
    return cols


# --------------------------------------------------------------------------
def read_conductance(cols, stim):
    """I/V at every read-plateau middle (S), linear interpolation in time."""
    tr = stim.read_times
    if not len(tr):
        return np.array([])
    i = np.interp(tr, cols["t_s"], cols["I_A"])
    v = np.interp(tr, cols["t_s"], cols["V_V"])
    return i / v


def x_from_iv(v, i, P):
    """Invert I = V / (r_off x/d + r_on (1 - x/d)) where |V| >= V_X_MIN."""
    v = np.asarray(v, float)
    i = np.asarray(i, float)
    out = np.full(len(v), math.nan)
    ok = (np.abs(v) >= V_X_MIN) & (i * v > 0)
    r = v[ok] / i[ok]
    out[ok] = P.d * np.clip((r - P.r_on) / (P.r_off - P.r_on), 0.0, 1.0)
    return out


def score(test, ref, stim):
    """Per-read G error, I(t) NRMSE and state NRMSE of `test` against `ref`
    on the test's time grid clipped to the common span (= stanford core.score)."""
    t = test["t_s"]
    lo, hi = max(t[0], ref["t_s"][0]), min(t[-1], ref["t_s"][-1])
    m = (t >= lo) & (t <= hi)
    tg = t[m]
    mi = compare.metrics(test["I_A"][m], np.interp(tg, ref["t_s"], ref["I_A"]))
    x_t = test["x_m"][m]
    x_r = np.interp(tg, ref["t_s"], ref["x_m"])
    ok = np.isfinite(x_t) & np.isfinite(x_r)
    mx = compare.metrics(x_t[ok], x_r[ok]) if ok.sum() > 2 else {
        "nrmse": math.nan, "maxerr": math.nan, "range": math.nan}
    if not (mx["range"] > 0):            # state never moved: NRMSE undefined
        mx = {"nrmse": math.nan, "maxerr": math.nan, "range": mx["range"]}
    gt, gr = read_conductance(test, stim), read_conductance(ref, stim)
    if len(gt):
        rel = np.abs(gt - gr) / np.abs(gr)
        g_max, g_mean = float(np.max(rel)), float(np.mean(rel))
    else:
        g_max = g_mean = math.nan
    return {"n_points": int(len(tg)), "n_reads": int(len(gt)),
            "g_read_max_pct": 100 * g_max, "g_read_mean_pct": 100 * g_mean,
            "i_nrmse_pct": 100 * mi["nrmse"], "i_maxerr_pct": 100 * mi["maxerr"],
            "i_range_mA": 1e3 * mi["range"],
            "x_nrmse_pct": 100 * mx["nrmse"], "x_maxerr_pct": 100 * mx["maxerr"]}


def verdict(row):
    checks = []
    if row["n_reads"]:
        checks += [row["g_read_max_pct"] <= 100 * G_MAX_LIMIT,
                   row["g_read_mean_pct"] <= 100 * G_MEAN_LIMIT]
    checks.append(row["i_nrmse_pct"] <= 100 * I_NRMSE_LIMIT)
    if not math.isnan(row["x_nrmse_pct"]):
        checks.append(row["x_nrmse_pct"] <= 100 * X_NRMSE_LIMIT)
    return "PASS" if all(checks) else "FAIL"
