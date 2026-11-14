"""Shared pieces of the Gate A validation: model registry, the canonical CSV
schema every reference is converted to, and the twin stepping loop."""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_core import EcfetV2, EcfetV3, V2Params, V3Params  # noqa: E402
from simulation_core.simulator import time_grid  # noqa: E402

from validation import stimuli as stim_mod  # noqa: E402

VDIR = Path(__file__).resolve().parent
DATA = VDIR / "data"
FIGURES = VDIR / "figures"
SPECTRE_DECKS = VDIR / "spice" / "spectre"

V_READ = 10e-3        # V, read bias across In-Out (same in every .scs)

# model key -> (twin factory, Verilog-A source relative to the repo root)
MODELS = {
    "v2": (lambda: EcfetV2(V2Params()), "verilog/ecfet_v2.va", "ecfet_v2"),
    "v3": (lambda: EcfetV3(V3Params()), "verilog/ecfet_v3.va", "ecfet_v3"),
}

# Canonical columns.  Every source (twin, Spectre) is written in this order.
#   G_S      observed conductance (twin model.G; Spectre 1/V(Rmem))
#   G_ret_S  full nonvolatile state incl. STDP lock-in
#   G_vol_S  volatile EDL pools (sum of the three)
#   stdp_S   STDP-locked part alone (NaN where the source does not expose it)
#   x_Li     v3 conserved intercalation fraction (NaN for v2)
COLUMNS = ["t_s", "drive_A", "G_S", "R_ohm", "i_read_A",
           "G_ret_S", "G_vol_S", "stdp_S", "x_Li"]


def model_keys(arg=None):
    if not arg or arg == "all":
        return list(MODELS)
    keys = [k.strip() for k in arg.split(",")]
    bad = [k for k in keys if k not in MODELS]
    if bad:
        raise SystemExit(f"unknown model {bad}; have {list(MODELS)}")
    return keys


def new_model(key):
    return MODELS[key][0]()


# --------------------------------------------------------------------------
# twin stepping
# --------------------------------------------------------------------------
def base_grid(stim):
    wf = stim.waveform()
    return time_grid(wf.breakpoints, stim.t_stop, dt_min=stim_mod.DT_MIN_S,
                     dt_max=stim.dt_max)


def refine_grid(ts, factor):
    """Split every interval of ts into `factor` equal sub-intervals."""
    if factor == 1:
        return np.asarray(ts, dtype=float)
    ts = np.asarray(ts, dtype=float)
    frac = np.arange(factor) / factor
    fine = (ts[:-1, None] + np.diff(ts)[:, None] * frac[None, :]).ravel()
    return np.append(fine, ts[-1])


def simulate_on_grid(model, waveform, ts):
    """Exact mirror of simulation_core.simulate()'s loop on a given grid
    (tests check it reproduces simulate() bit-for-bit on the base grid)."""
    n = len(ts)
    model.reset()
    keys = list(model.observables().keys())
    out = {"i": np.empty(n), "G": np.empty(n),
           **{k: np.empty(n) for k in keys}}

    def record(k, i_now):
        out["i"][k] = i_now
        out["G"][k] = model.G
        obs = model.observables()
        for key in keys:
            out[key][k] = obs[key]

    record(0, waveform.current(0.0))
    for k in range(n - 1):
        t, dt = ts[k], ts[k + 1] - ts[k]
        i_now = waveform.current(t + 0.5 * dt)
        model.step(t, dt, i_now)
        record(k + 1, i_now)
    return out


def twin_columns(key, ts, raw):
    n = len(ts)
    nan = np.full(n, math.nan)
    G = raw["G"]
    if key == "v2":
        g_ret, stdp, x = raw["G_nv (S)"], nan, nan
    else:
        g_ret, stdp, x = raw["G_retained (S)"], raw["stdp_lock (S)"], raw["x_Li"]
    return {"t_s": np.asarray(ts), "drive_A": raw["i"], "G_S": G,
            "R_ohm": 1.0 / G, "i_read_A": V_READ * G, "G_ret_S": g_ret,
            "G_vol_S": raw["G_volatile (S)"], "stdp_S": stdp, "x_Li": x}


def run_twin(key, stim, refine=1):
    """Run one stimulus through the twin.  Returns (columns, wall_s)."""
    ts = refine_grid(base_grid(stim), refine)
    model = new_model(key)
    wf = stim.waveform()
    t0 = time.perf_counter()
    raw = simulate_on_grid(model, wf, ts)
    wall = time.perf_counter() - t0
    return twin_columns(key, ts, raw), wall


# --------------------------------------------------------------------------
# canonical CSV I/O
# --------------------------------------------------------------------------
def provenance_line(source, model):
    return (f"# source: {source}; model: {model}; stimulus_set: "
            f"{stim_mod.STIMULUS_SET_VERSION} {stim_mod.fingerprint()[:16]}")


def write_csv(path, cols, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([np.asarray(cols[c], dtype=float) for c in COLUMNS])
    lines = [provenance, ",".join(COLUMNS)]
    for row in data:
        lines.append(",".join("nan" if math.isnan(v) else f"{v:.10g}"
                              for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_csv(path):
    """-> (columns dict, provenance str)."""
    path = Path(path)
    prov, header, rows = "", None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            prov = prov or line[1:].strip()
            continue
        if header is None:
            header = [h.strip() for h in line.split(",")]
            continue
        rows.append([float(v) for v in line.split(",")])
    arr = np.array(rows, dtype=float).reshape(-1, len(header))
    return {h: arr[:, k] for k, h in enumerate(header)}, prov


def write_meta(path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def read_meta(path):
    path = Path(path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def data_path(source, key, stim_name, ext=".csv"):
    return DATA / source / key / f"{stim_name}{ext}"
