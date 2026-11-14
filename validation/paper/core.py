"""Shared plumbing for the sharbati-1 paper overlay: data paths, provenance,
twin runs on the sharbati grid, CSV thinning and the time-step self-check.

Re-uses the Gate A pieces unchanged (validation/common.py: model registry,
canonical columns, stepping loop, CSV I/O; validation/compare.py: metrics).
"""

import math
import time
from pathlib import Path

import numpy as np

from validation import common, compare
from validation.paper import sharbati_stimuli as S

DATA = common.DATA / "sharbati"           # validation/data/sharbati/
FIGURES = common.FIGURES / "sharbati"
DECKS = common.VDIR / "spice" / "spectre_sharbati"
PAPER_DIR = Path(__file__).resolve().parent
TARGETS = PAPER_DIR / "sharbati2018_targets.json"
CURVES = PAPER_DIR / "curves"

# Saved-CSV density.  The twin and Spectre grids are much denser than any
# figure needs (ltp_ltd alone is ~2.8e5 twin points), so the CSVs keep every
# waveform edge plus points no closer than `save_dt` (finer, `near_dt`, during
# `near_s` after each edge).  Every metric is extracted on the kept points.
SAVE = {
    "single_pulse": dict(save_dt=50e-3, near_s=1.0, near_dt=1e-3),
    "ltp_ltd": dict(save_dt=25e-3, near_s=0.0, near_dt=25e-3),
    "paired_pulse": dict(save_dt=2e-3, near_s=0.1, near_dt=0.25e-3),
    "retention": dict(save_dt=60.0, near_s=300.0, near_dt=50e-3),
}


def data_path(source, key, name, ext=".csv"):
    return DATA / source / key / f"{name}{ext}"


def provenance_line(source, model):
    return (f"# source: {source}; model: {model}; stimulus_set: "
            f"{S.STIMULUS_SET_VERSION} {S.fingerprint()[:16]}")


# --------------------------------------------------------------------------
def keep_mask(t, stim, keep_edge_halo=0.0):
    """Boolean mask of the points to save.  Always keeps t[0], t[-1] and every
    waveform edge; `keep_edge_halo` > 0 also keeps every point within that
    distance after an edge (used for Spectre, whose PWL ramps last 1 us)."""
    t = np.asarray(t, dtype=float)
    rule = SAVE[stim.params["kind"]]
    edges = np.asarray(stim.waveform().edges, dtype=float)
    keep = np.zeros(len(t), dtype=bool)
    keep[0] = keep[-1] = True
    if len(edges):
        idx = np.searchsorted(t, edges)
        for j, e in zip(idx, edges):
            for k in (j - 1, j, j + 1):
                if 0 <= k < len(t) and abs(t[k] - e) <= max(1e-12, keep_edge_halo + 1e-12):
                    keep[k] = True
            if keep_edge_halo > 0:
                lo, hi = np.searchsorted(t, [e - keep_edge_halo, e + 2 * keep_edge_halo])
                keep[lo:hi] = True
    last_edge = np.full(len(t), -np.inf)
    if len(edges):
        pos = np.searchsorted(edges, t, side="right") - 1
        ok = pos >= 0
        last_edge[ok] = edges[pos[ok]]
    t_last = -np.inf
    for k in range(len(t)):
        if keep[k]:
            t_last = t[k]
            continue
        near = (t[k] - last_edge[k]) <= rule["near_s"]
        if t[k] - t_last >= (rule["near_dt"] if near else rule["save_dt"]) - 1e-12:
            keep[k] = True
            t_last = t[k]
    return keep


def thin(cols, stim, keep_edge_halo=0.0):
    m = keep_mask(cols["t_s"], stim, keep_edge_halo)
    return {k: np.asarray(v)[m] for k, v in cols.items()}


# --------------------------------------------------------------------------
def run_twin(key, stim, refine=1):
    """Full-resolution twin run on the sharbati grid -> (columns, wall_s)."""
    ts = common.refine_grid(S.grid(stim), refine)
    model = common.new_model(key)
    t0 = time.perf_counter()
    raw = common.simulate_on_grid(model, stim.waveform(), ts)
    wall = time.perf_counter() - t0
    return common.twin_columns(key, ts, raw), wall


def write_twin(key, stim, out_root=None):
    cols, wall = run_twin(key, stim)
    kept = thin(cols, stim)
    root = Path(out_root) if out_root else DATA
    path = root / "twin" / key / f"{stim.name}.csv"
    common.write_csv(path, kept, provenance_line("Python twin (simulation_core)", key))
    common.write_meta(path.with_suffix(".json"), {
        "source": "twin", "model": key, "stimulus": stim.name,
        "stimulus_set": S.STIMULUS_SET_VERSION,
        "n_points_run": int(len(cols["t_s"])), "n_points_saved": int(len(kept["t_s"])),
        "wall_s": round(wall, 4)})
    return path, cols, kept, wall


def load(source, key, name):
    p = data_path(source, key, name)
    if not p.is_file():
        return None
    cols, _ = common.read_csv(p)
    return cols


# --------------------------------------------------------------------------
def self_check(keys, names):
    """Twin at dt, dt/2, dt/4 (every interval split) on the sharbati grid;
    scored with the Gate A pass rule on the base-grid points."""
    rows = []
    for key in keys:
        for name in names:
            st = S.BY_NAME[name]
            runs = {f: run_twin(key, st, refine=f)[0] for f in (1, 2, 4)}
            g = {f: runs[f]["G_S"][::f] for f in runs}
            m1 = compare.metrics(g[1], g[4])
            m2 = compare.metrics(g[2], g[4])
            d12 = float(np.sqrt(np.mean((g[1] - g[2]) ** 2)))
            d24 = float(np.sqrt(np.mean((g[2] - g[4]) ** 2)))
            order = math.log2(d12 / d24) if d12 > 0 and d24 > 0 else math.nan
            row = {"model": key, "stimulus": name, "n_points": len(g[1]),
                   "g_range_uS": 1e6 * m1["range"],
                   "nrmse_dt_vs_dt4_pct": 100 * m1["nrmse"],
                   "maxerr_dt_vs_dt4_pct": 100 * m1["maxerr"],
                   "nrmse_dt2_vs_dt4_pct": 100 * m2["nrmse"],
                   "maxerr_dt2_vs_dt4_pct": 100 * m2["maxerr"],
                   "observed_order": order, "verdict": compare.verdict(m1),
                   "retention_pct_dt_dt2_dt4": ""}
            if st.params["kind"] == "retention":
                # the retention metric itself at each refinement
                from validation.paper import metrics as M
                vals = [M.retention_pct(runs[f], st) for f in (1, 2, 4)]
                row["retention_pct_dt_dt2_dt4"] = " / ".join(f"{v:.6f}" for v in vals)
            rows.append(row)
    return rows
