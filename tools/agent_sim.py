#!/usr/bin/env python
"""Headless simulation runner for the embedded Claude agent.

The agent edits the Python model twins (ecfet/model_*.py), then calls this to
run a simulation and read back the result, so it can iterate code -> run ->
check -> fix until the behavior is right.  Because it imports the ecfet
package fresh each run, the agent's edits to the twins take effect here exactly
as they will in the GUI.

Usage:
    python tools/agent_sim.py <spec.json> <out.json>

spec.json:
{
  "models": ["v1", "v2", "fefet"],            # which twins to run
  "params": {"v2": {"kappa_v": 0.4, "nu_p": 3}},   # optional param overrides
  "pulses": [[t_start_s, width_s, amplitude_SI], ...],   # amps in A or V
  "t_stop": 2.0,                              # optional (default last edge +1s)
  "analysis": "G",                            # optional: "G" or "R" per-pulse
  "n_each": 20,                               # optional LTP/LTD branch size
  "downsample": 240                           # optional trace points (default 240)
}

out.json gets per-model summary stats, downsampled R(t)/G(t) traces, and (if
requested) the per-pulse retained-value curve with mean delta per branch.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecfet import (Waveform, EcfetV1, V1Params, EcfetV2, V2Params,
                   FeFET, FeFETParams, simulate)

SPECS = {
    "v1": (EcfetV1, V1Params, "v1 (Verilog-A port)"),
    "v2": (EcfetV2, V2Params, "v2 (practical ECFET)"),
    "fefet": (FeFET, FeFETParams, "FeFET"),
}


def _downsample(arr, n):
    m = len(arr)
    if m <= n:
        return [round(float(v), 6) for v in arr]
    step = m / n
    return [round(float(arr[min(int(i * step), m - 1)]), 6) for i in range(n)]


def _per_pulse(r, wf, metric):
    wins = wf.pulse_windows()
    if len(wins) < 2:
        return None
    ends = [t1 for _, t1 in wins]
    starts = [t0 for t0, _ in wins]
    t_sim = float(r.t[-1])
    inter = [starts[k + 1] - ends[k] for k in range(len(ends) - 1)
             if starts[k + 1] > ends[k]]
    typ = sorted(inter)[len(inter) // 2] if inter else (t_sim - ends[-1])
    cap = max(1e-4, 0.9 * typ)
    sample_t = []
    for k in range(len(ends)):
        gap = (starts[k + 1] - ends[k]) if k + 1 < len(ends) \
            else (t_sim - ends[k])
        s = min(max(0.9 * gap if gap > 0 else cap, 1e-5), cap)
        sample_t.append(min(ends[k] + s, t_sim))
    R_s, G_s = r.at(sample_t)
    vals = (R_s if metric == "R" else G_s * 1e6).tolist()
    return [round(float(v), 6) for v in vals]


def main():
    if len(sys.argv) < 3:
        print("usage: agent_sim.py <spec.json> <out.json>", file=sys.stderr)
        return 2
    with open(sys.argv[1], "r", encoding="utf-8-sig") as f:
        spec = json.load(f)

    models = spec.get("models") or ([spec["model"]] if "model" in spec else
                                    ["v2"])
    overrides = spec.get("params", {})
    pulses = [(float(p[0]), float(p[1]), float(p[2]))
              for p in spec.get("pulses", [])]
    if not pulses:
        json.dump({"error": "spec has no pulses"}, open(sys.argv[2], "w"))
        return 1
    wf = Waveform(pulses)
    last_edge = wf.breakpoints[-1] if wf.breakpoints else 0.0
    t_stop = float(spec.get("t_stop", last_edge + 1.0))
    ds = int(spec.get("downsample", 240))
    metric = "R" if spec.get("analysis") == "R" else "G"
    n_each = spec.get("n_each")

    out = {"t_stop": t_stop, "n_pulses": len(pulses), "results": []}
    for key in models:
        if key not in SPECS:
            out["results"].append({"model": key, "error": "unknown model"})
            continue
        cls, pcls, label = SPECS[key]
        try:
            params = pcls(**overrides.get(key, {}))
            r = simulate(cls(params), wf, t_stop, label=label)
        except Exception as e:
            out["results"].append({"model": key,
                                   "error": f"{type(e).__name__}: {e}"})
            continue
        entry = {
            "model": label, "key": key,
            "R_ohm": {"start": float(r.R[0]), "end": float(r.R[-1]),
                      "min": float(r.R.min()), "max": float(r.R.max())},
            "G_uS": {"start": float(r.G[0] * 1e6), "end": float(r.G[-1] * 1e6),
                     "min": float(r.G.min() * 1e6),
                     "max": float(r.G.max() * 1e6)},
            "t_s": _downsample(r.t, ds),
            "R_ohm_trace": _downsample(r.R, ds),
            "G_uS_trace": _downsample(r.G * 1e6, ds),
        }
        curve = _per_pulse(r, wf, metric)
        if curve is not None:
            pp = {"metric": metric, "values": curve}
            if n_each and 0 < n_each < len(curve):
                ltp = curve[:n_each]
                ltd = curve[n_each:]
                d = lambda v: [b - a for a, b in zip(v[:-1], v[1:])]
                mean = lambda v: sum(v) / len(v) if v else 0.0
                pp["mean_delta_LTP"] = round(mean(d(ltp)), 6)
                pp["mean_delta_LTD"] = round(mean(d(ltd)), 6)
            entry["per_pulse"] = pp
        out["results"].append(entry)

    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {sys.argv[2]}: {len(out['results'])} model(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
