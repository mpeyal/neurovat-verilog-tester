"""Shared plumbing for stanford-1: twin loading, the PWL time grid, twin runs,
the canonical CSV schema, read-out metrics and the independent brute-force
reference.

Re-uses validation/common.py (CSV meta I/O, refine_grid) and
validation/compare.py (metrics, verdict, write_rows) unchanged.
"""

import importlib.util
import math
import time
from pathlib import Path

import numpy as np

from validation import common, compare
from validation.stanford import stimuli as S

DATA = common.DATA / "stanford"                # validation/data/stanford/
FIGURES = common.FIGURES / "stanford"
DECKS = common.VDIR / "spice" / "spectre_stanford"
TWIN_PATH = common.ROOT / "twins" / "stanford_pku_rram.py"
VA_NAME = "stanford_pku_rram.va"               # local copy, git-ignored
VA_PATH = common.ROOT / "verilog" / VA_NAME
MODULE = "rram_v_1_0_0"
MODEL = "stanford_pku"

# Canonical columns (twin and Spectre are both written in this order).
#   V_TE_V       V(TE,BE)
#   I_TE_A       current into TE
#   gap_m        clamped gap (Spectre: inverted from I and V where |V| >= V_GAP_MIN)
#   gap_state_m  idt state y (twin only; NaN for Spectre)
#   T_K          T_ini + |V I| Rth
#   G_read_S     I(0.1 V)/0.1 at the present gap
COLUMNS = ["t_s", "V_TE_V", "I_TE_A", "gap_m", "gap_state_m", "T_K", "G_read_S"]
V_GAP_MIN = 0.05            # V; below this |V| the gap is not inverted from I

# pass rule (per stimulus, twin vs reference)
G_MAX_LIMIT = 0.03          # max relative error of the per-read conductance
G_MEAN_LIMIT = 0.01         # mean relative error of the per-read conductance
I_NRMSE_LIMIT = 0.01        # rms(I - I_ref) / range(I_ref)
GAP_NRMSE_LIMIT = 0.01      # rms(gap - gap_ref) / range(gap_ref) (where defined)

_twin_mod = None


def twin_module():
    """Import twins/stanford_pku_rram.py by path (trusted repo file)."""
    global _twin_mod
    if _twin_mod is None:
        spec = importlib.util.spec_from_file_location("nvat_validation_stanford_twin",
                                                      TWIN_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _twin_mod = mod
    return _twin_mod


def new_twin(stim=None, **overrides):
    m = twin_module()
    kw = dict(stim.instance_dict()) if stim is not None else {}
    kw.update(overrides)
    return m.StanfordPKURRAM(m.StanfordPKUParams(**kw))


# --------------------------------------------------------------------------
def grid(stim):
    """Every PWL corner plus uniform steps <= dt_max inside each segment."""
    tp = stim.t_points
    tp = np.append(tp, stim.t_stop) if tp[-1] < stim.t_stop else tp
    out = [tp[:1]]
    for a, b in zip(tp[:-1], tp[1:]):
        n = max(1, int(math.ceil((b - a) / stim.dt_max - 1e-9)))
        out.append(a + (b - a) * np.arange(1, n + 1) / n)
    return np.concatenate(out)


def simulate_twin(model, stim, ts):
    """Step the twin over ts with V linear between grid points (exactly the
    PWL, because every PWL corner is a grid point)."""
    n = len(ts)
    model.reset()
    v = stim.v(ts)
    cols = {c: np.empty(n) for c in COLUMNS}

    def record(k):
        o = model.observables()
        cols["t_s"][k] = ts[k]
        cols["V_TE_V"][k] = v[k]
        cols["I_TE_A"][k] = model.current(v[k])
        cols["gap_m"][k] = o["gap (nm)"] * 1e-9
        cols["gap_state_m"][k] = o["gap_state (nm)"] * 1e-9
        cols["T_K"][k] = model.temperature(v[k])
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
    return cols, wall, model.n_rhs


# --------------------------------------------------------------------------
def brute_reference(points_ps, t_stop, dt, params=None):
    """Literal transcription of the .va analog block, ONE evaluation per time
    point and forward-Euler `idt` (independent of the twin's integrator).
    gamma and T use the previous evaluation's clamped gap, as the .va does.
    Returns (t, gap, y) at the end of every step (slow: pure Python)."""
    P = params or twin_module().StanfordPKUParams()
    tp = np.array([t for t, _ in points_ps], float) * S.PS
    vp = np.array([v for _, v in points_ps], float)
    n = int(round(t_stop / dt))
    tm = (np.arange(n) + 0.5) * dt
    vm = np.interp(tm, tp, vp)
    y = gap = float(P.gap_ini)
    gap = min(max(gap, P.gap_min), P.gap_max)
    out_g, out_y = np.empty(n), np.empty(n)
    q, kb = P.q, P.kb
    for k in range(n):
        Vtb = float(vm[k])
        Itb = P.I0 * math.exp(-gap / P.g0) * math.sinh(Vtb / P.V0)
        T = P.T_ini + abs(Vtb * Itb * P.Rth)
        gamma_ini = 16.0 if Vtb < 0 else P.gamma0
        gamma = gamma_ini - P.beta * (gap / 1e-9) ** 3
        if gamma * abs(Vtb) / P.tox < P.F_min:
            gamma = 0.0
        ddt = -P.Vel0 * math.exp(-q * P.Ea / kb / T) * math.sinh(
            gamma * P.a0 / P.tox * q * Vtb / kb / T)
        y += ddt * dt
        gap = min(max(y, P.gap_min), P.gap_max)
        out_g[k], out_y[k] = gap, y
    return (np.arange(1, n + 1) * dt), out_g, out_y


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


def data_path(source, name, ext=".csv"):
    return DATA / source / f"{name}{ext}"


def load(source, name):
    p = data_path(source, name)
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
    i = np.interp(tr, cols["t_s"], cols["I_TE_A"])
    v = np.interp(tr, cols["t_s"], cols["V_TE_V"])
    return i / v


def gap_from_iv(v, i, params=None):
    """Invert I = I0 exp(-gap/g0) sinh(V/V0) where |V| >= V_GAP_MIN."""
    P = params or twin_module().StanfordPKUParams()
    v = np.asarray(v, float)
    i = np.asarray(i, float)
    out = np.full(len(v), math.nan)
    ok = (np.abs(v) >= V_GAP_MIN) & (i * v > 0)
    out[ok] = -P.g0 * np.log(np.abs(i[ok]) / (P.I0 * np.abs(np.sinh(v[ok] / P.V0))))
    return out


def score(test, ref, stim):
    """Per-read G error, I(t) NRMSE and gap NRMSE of `test` against `ref`
    on the test's time grid clipped to the common span."""
    t = test["t_s"]
    lo, hi = max(t[0], ref["t_s"][0]), min(t[-1], ref["t_s"][-1])
    m = (t >= lo) & (t <= hi)
    tg = t[m]
    i_t = test["I_TE_A"][m]
    i_r = np.interp(tg, ref["t_s"], ref["I_TE_A"])
    mi = compare.metrics(i_t, i_r)
    g_t = test["gap_m"][m]
    g_r = np.interp(tg, ref["t_s"], ref["gap_m"])
    ok = np.isfinite(g_t) & np.isfinite(g_r)
    mg = compare.metrics(g_t[ok], g_r[ok]) if ok.sum() > 2 else {
        "nrmse": math.nan, "maxerr": math.nan, "range": math.nan}
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
            "gap_nrmse_pct": 100 * mg["nrmse"], "gap_maxerr_pct": 100 * mg["maxerr"]}


def verdict(row):
    checks = []
    if row["n_reads"]:
        checks += [row["g_read_max_pct"] <= 100 * G_MAX_LIMIT,
                   row["g_read_mean_pct"] <= 100 * G_MEAN_LIMIT]
    checks.append(row["i_nrmse_pct"] <= 100 * I_NRMSE_LIMIT)
    if not math.isnan(row["gap_nrmse_pct"]):
        checks.append(row["gap_nrmse_pct"] <= 100 * GAP_NRMSE_LIMIT)
    return "PASS" if all(checks) else "FAIL"
