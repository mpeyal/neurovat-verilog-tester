#!/usr/bin/env python
"""Acceptance 3a of the physical-time engine fix: what the network reads
from a v3 synapse, against Spectre, on the committed C5 tier-1 schedule.

A closed-loop re-run of ``validation/c5/run_engine.py`` changes the reads
and so the spikes and the write schedule; the committed Spectre run
(``validation/data/c5/spectre_1ms.csv``) replays the OLD schedule. So this
is an open-loop replay of that schedule through the engine's device code,
with the engine's call pattern:

- the device is ``vector_twin_factory(...)`` (the vectorised EcfetV3Array,
  physical time, full read), placed at the recorded start conductance;
- every recorded pulse is ``program``-ed at its start time, all pulses of
  one start time in one call (one flush of the write window);
- the device is ``peek``-ed at every recorded engine read time (1 ms), after
  the writes of that time, as ``Synapses.tick`` / ``flush`` do.

The reads are written as ``engine_reads.csv`` (and ``engine.json`` with the
final G) to ``validation/data/c5_fixed/`` next to copies of the schedule and
the Spectre data, and scored there by ``compare_c5.score`` (Gate A rule:
NRMSE <= 1 %, max <= 3 % of each synapse's Spectre G range), at all reads,
inside and outside the write pulses.

    python validation/enginefix/replay_c5.py
"""
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "validation" / "c5")]

import numpy as np  # noqa: E402

import schedule as sch  # noqa: E402

SRC = ROOT / "validation" / "data" / "c5"
OUT = ROOT / "validation" / "data" / "c5_fixed"


def replay(src=SRC, out=OUT):
    from neurovat.devices import vector_twin_factory
    from simulation_core.model_v3 import EcfetV3, V3Params
    meta = json.loads((src / "engine.json").read_text(encoding="utf-8"))
    rows = np.loadtxt(src / "schedule.csv", delimiter=",", skiprows=1, ndmin=2)
    t_reads = np.loadtxt(src / "engine_reads.csv", delimiter=",", skiprows=1,
                         usecols=0)
    n = meta["masquelier"]["n_afferents"]
    p = V3Params(**meta["v3_params"])
    fac = vector_twin_factory(lambda: EcfetV3(p), "physical", "full")
    if fac.kind != "vector":
        raise SystemExit("vector_twin_factory fell back to the scalar adapter")
    dev = fac(1, n)
    dev.set_conductance(np.full(dev.shape, meta["g_start_S"]))
    # pulses grouped by start tick (one program call per flush)
    tick = np.round(rows[:, 1] / sch.TICK).astype(np.int64)
    order = np.lexsort((rows[:, 0], tick))
    rows, tick = rows[order], tick[order]
    starts, first = np.unique(tick, return_index=True)
    bounds = list(first) + [len(rows)]
    r_tick = np.round(t_reads / sch.TICK).astype(np.int64)
    reads = []
    j = 0
    for t_r, rt in zip(t_reads, r_tick):
        while j < len(starts) and starts[j] <= rt:     # writes of <= this time
            blk = rows[bounds[j]:bounds[j + 1]]
            cols = blk[:, 0].astype(np.int64)
            dev.program(np.zeros_like(cols), cols, blk[:, 3],
                        float(blk[0, 2]), starts[j] * sch.TICK)
            j += 1
        g = np.asarray(dev.peek(t_r), float).ravel().copy()
        dev.read = "nv"
        gnv = np.asarray(dev.peek(t_r), float).ravel().copy()
        dev.read = "full"
        reads.append((t_r, g, gnv))
    if j != len(starts):
        raise SystemExit(f"{len(starts) - j} write times after the last read")
    dev.advance(max(meta["t_end_s"], dev.clock))
    g_end = np.asarray(dev.peek(meta["t_end_s"]), float).ravel()
    dev.read = "nv"
    gnv_end = np.asarray(dev.peek(meta["t_end_s"]), float).ravel()
    dev.read = "full"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("schedule.csv", "spectre.json", "spectre_1ms.csv"):
        shutil.copyfile(src / f, out / f)
    with open(out / "engine_reads.csv", "w", newline="\n") as f:
        f.write("t_s," + ",".join(f"G{k}" for k in range(n)) + ","
                + ",".join(f"Gnv{k}" for k in range(n)) + "\n")
        for t, g, gnv in reads:
            f.write(f"{t:.9f}," + ",".join(f"{v:.10g}" for v in g) + ","
                    + ",".join(f"{v:.10g}" for v in gnv) + "\n")
    m2 = dict(meta, what="C5 tier-1 schedule replayed open loop through the "
              "fixed engine device code (validation/enginefix/replay_c5.py)",
              final_G_full_S=g_end.tolist(), final_G_nv_S=gnv_end.tolist())
    (out / "engine.json").write_text(json.dumps(m2, indent=1) + "\n",
                                     encoding="utf-8")
    return out


def score(out=OUT, figures=True):
    """compare_c5.score on ``out``; adds the in-pulse-only row."""
    import compare_c5 as cc
    sch.load.__defaults__ = (out,)
    cc.DATA = sch.DATA = out
    cc.FULL = out / "_no_full_resolution.npz"     # use the committed 1 ms data
    cc.FIG = ROOT / "validation" / "figures" / "c5_fixed"
    rows = cc.score(figures=figures)
    meta, per = sch.load(out)
    n = len(per)
    t_s, G_s, _, _ = cc.load_spectre(n)
    rd = np.loadtxt(out / "engine_reads.csv", delimiter=",", skiprows=1)
    t_r = rd[:, 0]
    for k, row in enumerate(rows):
        inp = np.zeros(len(t_r), bool)
        for s, w, _a in per[k]:
            inp |= (t_r > s * sch.TICK - 1e-9) & (t_r < (s + w) * sch.TICK - 1e-9)
        e = (rd[:, 1 + k] - np.interp(t_r, t_s, G_s[k]))[inp]
        rng = row["g_range_uS"] * 1e-6
        row["engine_in_nrmse_pct"] = 100 * math.sqrt(float(np.mean(e * e))) / rng
        row["engine_in_max_pct"] = 100 * float(np.max(np.abs(e))) / rng
        row["in_pulse_reads"] = int(inp.sum())
    keys = ("engine_nrmse_pct", "engine_max_pct", "engine_in_nrmse_pct",
            "engine_in_max_pct", "engine_out_nrmse_pct", "engine_out_max_pct",
            "engine_nv_nrmse_pct", "engine_nv_max_pct", "final_G_engine_err_pct",
            "final_Gnv_engine_err_pct", "twin_nrmse_pct", "twin_max_pct")
    summ = {k: [min(r[k] for r in rows), max(r[k] for r in rows)] for k in keys}
    summ["engine_pass"] = sum(r["engine_verdict"] == "PASS" for r in rows)
    summ["twin_pass"] = sum(r["twin_verdict"] == "PASS" for r in rows)
    summ["synapses"] = len(rows)
    (out / "acceptance.json").write_text(json.dumps(
        {"ranges_pct_of_range": summ, "per_synapse": rows}, indent=1) + "\n",
        encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return rows, summ


if __name__ == "__main__":
    replay()
    rows, summ = score()
    # keep the results (acceptance.json, summary.csv, engine.json); the
    # inputs are copies of validation/data/c5 and the reads are bulky
    for f in ("schedule.csv", "spectre.json", "spectre_1ms.csv",
              "engine_reads.csv", "cost.json"):
        (OUT / f).unlink(missing_ok=True)
    sys.exit(0 if summ["engine_pass"] == summ["synapses"] else 1)
