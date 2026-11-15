#!/usr/bin/env python
"""Collect the C3 timing JSONs (fetched from the timing machine) into the
committed CSVs + machine JSON and draw the log-log figures.

    python validation/scaling/collect.py results/scaling/host
        -> validation/data/scaling/*.csv, machine.json
        -> validation/figures/scaling/*.png

Only numbers, the CPU model and core count, software versions and load
averages are kept (no host or user names, no log text).
"""
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "validation" / "data" / "scaling"
FIG = ROOT / "validation" / "figures" / "scaling"

# reference palette, categorical slots in fixed order (dataviz skill)
C_SPECTRE, C_VECTOR, C_SCALAR_ENG, C_SCALAR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TIER2 = (0.1, 0.3)      # s per synapse-second, validation/c5/TIER2.md (extrapolation)


def _f(v, nd=4):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return v


def write_csv(name, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(DATA / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _f(r.get(k)) for k in keys})


def loads(machines):
    one = [m["loadavg_1_5_15"][0] for m in machines if m.get("loadavg_1_5_15")]
    return (min(one), max(one)) if one else (None, None)


def collect(src):
    src = Path(src)
    machines = []
    # ---------------------------------------------------------------- v3
    v3 = []
    for f in sorted(src.glob("v3/v3_n*.json"), key=lambda p: int(p.stem[4:])):
        r = json.loads(f.read_text(encoding="utf-8"))
        sp = r["sides"].get("spectre", {})
        row = {"n": r["n"], "pulses": r["pulses"], "stream_s": r["t_stop_s"]}
        if sp.get("runs"):
            ms = [x["machine"] for x in sp["runs"]]
            machines += ms
            lo, hi = loads(ms)
            row.update({
                "spectre_tran_median_s": sp["tran"]["median_s"],
                "spectre_tran_min_s": sp["tran"]["min_s"],
                "spectre_tran_max_s": sp["tran"]["max_s"],
                "spectre_total_median_s": sp["total"]["median_s"],
                "spectre_total_rep0_s": sp["runs"][0]["total_elapsed_s"],
                "spectre_steps": sp["runs"][0]["accepted_steps"],
                "spectre_cpu_over_elapsed": sp["runs"][0]["tran_cpu_over_elapsed"],
                "spectre_timed_out": sp["timed_out"],
                "spectre_load1_min": lo, "spectre_load1_max": hi})
        elif sp:
            row["spectre_timed_out"] = True
        for side in ("vector", "scalar_engine", "scalar"):
            s = r["sides"].get(side)
            if not s:
                continue
            machines.append(s["machine"])
            row[f"{side}_median_s"] = s["median_s"]
            row[f"{side}_min_s"] = s["min_s"]
            row[f"{side}_max_s"] = s["max_s"]
            if "accuracy" in s:
                a = s["accuracy"]
                row[f"{side}_acc_pass"] = f"{a['pass']}/{a['probes']}"
                row[f"{side}_acc_max_pct"] = a["max_pct_max"]
                row[f"{side}_acc_nrmse_pct"] = a["nrmse_pct_max"]
            t = row.get("spectre_tran_median_s")
            if t:
                row[f"speedup_{side}"] = t / s["median_s"]
        v3.append(row)
    if v3:
        write_csv("v3_array.csv", v3)
    # ---------------------------------------------------------------- network
    net = []
    for f in sorted(src.glob("net/net_*_n*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        machines.append(r["machine"])
        net.append({"arm": r["arm"], "afferents": r["afferents"], "stream_s": r["stream_s"],
                    "median_s": r["median_s"], "min_s": r["min_s"], "max_s": r["max_s"],
                    "s_per_stream_s": r["s_per_stream_s"],
                    "s_per_synapse_second": r["s_per_synapse_second"],
                    "post_spikes": r["post_spikes"], "writes": r["writes"],
                    "load1": r["machine"]["loadavg_1_5_15"][0]})
    net.sort(key=lambda r: (r["arm"], r["afferents"]))
    if net:
        write_csv("network_time.csv", net)
    rep = []
    for f in sorted(src.glob("net/replay_n*.json"), key=lambda p: int(p.stem[8:])):
        r = json.loads(f.read_text(encoding="utf-8"))
        ms = [x["machine"] for x in r["runs"]]
        machines += ms
        lo, hi = loads(ms)
        acc = r.get("engine_read_vs_spectre") or {}
        n, T = r["afferents"], r["stream_s"]
        rep.append({"afferents": n, "stream_s": T, "pulses": r["pulses"],
                    "post_spikes": r["post_spikes"],
                    "spectre_tran_median_s": r["tran"]["median_s"],
                    "spectre_tran_min_s": r["tran"]["min_s"],
                    "spectre_tran_max_s": r["tran"]["max_s"],
                    "spectre_total_median_s": r["total"]["median_s"],
                    "spectre_steps": r["runs"][0]["accepted_steps"],
                    "spectre_cpu_over_elapsed": r["runs"][0]["tran_cpu_over_elapsed"],
                    "spectre_s_per_synapse_second": (r["tran"]["median_s"] / (n * T)
                                                     if r["tran"]["median_s"] else None),
                    "timed_out": r["timed_out"],
                    "engine_read_acc_pass": (f"{acc['pass']}/{acc['probes']}" if acc else ""),
                    "engine_read_acc_max_pct": acc.get("max_pct_max"),
                    "load1_min": lo, "load1_max": hi})
    if rep:
        write_csv("network_replay.csv", rep)
    # ---------------------------------------------------------------- rram
    rr = []
    for f in sorted(src.glob("rram/*_n*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        row = {"model": r["model"], "stimulus": r["stimulus"],
               "spectre_maxstep_s": r["spectre_maxstep_s"],
               "twin_dt_max_s": r["twin_dt_max_s"], "n": r["n"]}
        tw = r["sides"].get("twin_scalar")
        if tw:
            machines.append(tw["machine"])
            row.update({"twin_median_s": tw["median_s"], "twin_min_s": tw["min_s"],
                        "twin_max_s": tw["max_s"]})
        sp = r["sides"].get("spectre", {})
        if sp.get("runs"):
            ms = [x["machine"] for x in sp["runs"]]
            machines += ms
            acc = sp.get("probe0_twin_vs_spectre") or {}
            row.update({"spectre_tran_median_s": sp["tran"]["median_s"],
                        "spectre_tran_min_s": sp["tran"]["min_s"],
                        "spectre_tran_max_s": sp["tran"]["max_s"],
                        "spectre_total_median_s": sp["total"]["median_s"],
                        "spectre_steps": sp["runs"][0]["accepted_steps"],
                        "spectre_rc": sp["runs"][0]["rc"],
                        "spectre_timed_out": sp["timed_out"],
                        "probe0_i_nrmse_pct": acc.get("i_nrmse_pct"),
                        "probe0_g_read_max_pct": acc.get("g_read_max_pct"),
                        "load1_max": loads(ms)[1]})
        rr.append(row)
    rr.sort(key=lambda r: (r["model"], r["spectre_maxstep_s"], r["n"]))
    if rr:
        write_csv("rram_array.csv", rr)
    # ---------------------------------------------------------------- machine
    m0 = machines[0]
    lo, hi = loads(machines)
    mach = {"cpu": m0["cpu"], "logical_cpus": m0["logical_cpus"], "os": m0["os"],
            "python": m0["python"], "numpy": m0["numpy"],
            "spectre": "Spectre 24.1 (classic, +mt=1; transient CPU/elapsed ~1)",
            "threads": "1 (OMP/OPENBLAS/MKL_NUM_THREADS=1; Spectre +mt=1)",
            "reps_per_point": 3, "statistic": "median, min-max spread",
            "load1_range_during_runs": [lo, hi],
            "note": "shared multi-user machine; one timing job at a time"}
    (DATA / "machine.json").write_text(json.dumps(mach, indent=1) + "\n", encoding="utf-8")
    return v3, net, rep, rr, mach


# -------------------------------------------------------------------- plots
def _style(ax, xlabel, ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.grid(False, which="minor")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)


def _line(ax, x, y, lo, hi, color, label, marker="o", ls="-"):
    pts = [(a, b, c, d) for a, b, c, d in zip(x, y, lo, hi) if b]
    if not pts:
        return
    xs, ys, ls_, hs = zip(*pts)
    ax.plot(xs, ys, ls, color=color, lw=2, marker=marker, ms=6,
            markeredgecolor="white", markeredgewidth=1, label=label)
    ax.vlines(xs, ls_, hs, color=color, lw=1.2)
    ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                fontsize=7.5, color=INK, va="center")


def plots(v3, net, rep, rr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    if v3:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        n = [r["n"] for r in v3]
        g = lambda k: [r.get(k) for r in v3]  # noqa: E731
        _line(ax, n, g("spectre_tran_median_s"), g("spectre_tran_min_s"),
              g("spectre_tran_max_s"), C_SPECTRE, "Spectre (transient)")
        _line(ax, n, g("vector_median_s"), g("vector_min_s"), g("vector_max_s"),
              C_VECTOR, "twin, vectorised EcfetV3Array")
        _line(ax, n, g("scalar_engine_median_s"), g("scalar_engine_min_s"),
              g("scalar_engine_max_s"), C_SCALAR_ENG, "twin, scalar adapter (engine path)",
              marker="s")
        _line(ax, n, g("scalar_median_s"), g("scalar_min_s"), g("scalar_max_s"),
              C_SCALAR, "twin, scalar Gate A grid", marker="^", ls="--")
        for r in v3:
            if r.get("speedup_vector") and r["n"] >= 16:
                ax.annotate(f"{r['speedup_vector']:.0f}x", (r["n"], r["vector_median_s"]),
                            xytext=(0, -13), textcoords="offset points", ha="center",
                            fontsize=7.5, color=INK2)
            if r.get("spectre_timed_out") and not r.get("spectre_tran_median_s"):
                ax.annotate(">15 min", (r["n"], 900), ha="center", fontsize=7.5, color=INK2)
        _style(ax, "independent v3 devices N (C5 write schedule, 4.06 s)",
               "wall time (s), median of 3, bars = min-max")
        ax.set_title("ECFET v3 array: twin vs Spectre, same machine, 1 thread "
                     "(labels: Spectre / vectorised twin)", fontsize=8.5, color=INK, loc="left")
        ax.set_xlim(right=max(n) * 9)
        fig.tight_layout()
        fig.savefig(FIG / "v3_array.png", dpi=180)
        plt.close(fig)
    if net:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for arm, c, lab in (("ideal", C_SCALAR, "NeuroVAT network, ideal weights"),
                            ("v3_time", C_VECTOR, "NeuroVAT network, v3 physical time")):
            rows = [r for r in net if r["arm"] == arm]
            x = [r["afferents"] for r in rows]
            T = rows[0]["stream_s"] if rows else 1
            _line(ax, x, [r["s_per_stream_s"] for r in rows],
                  [r["min_s"] / T for r in rows], [r["max_s"] / T for r in rows], c, lab)
        if rep:
            x = [r["afferents"] for r in rep]
            T = [r["stream_s"] for r in rep]
            _line(ax, x, [r["spectre_tran_median_s"] / t if r["spectre_tran_median_s"] else None
                          for r, t in zip(rep, T)],
                  [(r["spectre_tran_min_s"] or 0) / t for r, t in zip(rep, T)],
                  [(r["spectre_tran_max_s"] or 0) / t for r, t in zip(rep, T)],
                  C_SPECTRE, "Spectre, open-loop device replay (measured)")
        xs = [16, 2000]
        ax.fill_between(xs, [TIER2[0] * v for v in xs], [TIER2[1] * v for v in xs],
                        color=C_SPECTRE, alpha=0.12, lw=0)
        ax.annotate("Spectre closed loop, EXTRAPOLATED\n(TIER2.md: 0.1-0.3 s per synapse-s)",
                    (60, TIER2[1] * 60), xytext=(0, 8), textcoords="offset points",
                    fontsize=7.5, color=INK2)
        _style(ax, "afferents (synapses)", "wall time per stream-second (s)")
        ax.set_title("Masquelier stream network (1 neuron, k = 10, 4 s), same machine, 1 thread",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_xlim(right=2000 * 9)
        fig.tight_layout()
        fig.savefig(FIG / "network.png", dpi=180)
        plt.close(fig)
    if rr:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        groups = {}
        for r in rr:
            groups.setdefault((r["model"], r["spectre_maxstep_s"]), []).append(r)
        twin_done = set()
        cols = {("stanford", 1e-11): C_SPECTRE, ("stanford", 1e-12): "#6250d6",
                ("vteam", 1e-10): C_SCALAR_ENG}
        for (m, step), rows in groups.items():
            name = "Stanford-PKU" if m == "stanford" else "VTEAM"
            x = [r["n"] for r in rows]
            c = cols.get((m, step), INK2)
            _line(ax, x, [r.get("spectre_tran_median_s") for r in rows],
                  [r.get("spectre_tran_min_s") for r in rows],
                  [r.get("spectre_tran_max_s") for r in rows], c,
                  f"Spectre {name}, {step * 1e12:g} ps")
            if m not in twin_done and any(r.get("twin_median_s") for r in rows):
                twin_done.add(m)
                _line(ax, x, [r.get("twin_median_s") for r in rows],
                      [r.get("twin_min_s") for r in rows], [r.get("twin_max_s") for r in rows],
                      C_VECTOR if m == "stanford" else C_SCALAR,
                      f"twin {name} (scalar loop)", marker="s", ls="--")
        _style(ax, "independent devices N", "wall time (s), median of 3")
        ax.set_title("Two-terminal arrays: scalar twins vs Spectre, same machine, 1 thread",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_xlim(right=max(r["n"] for r in rr) * 12)
        fig.tight_layout()
        fig.savefig(FIG / "rram_array.png", dpi=180)
        plt.close(fig)


def main(argv):
    v3, net, rep, rr, mach = collect(argv[0])
    plots(v3, net, rep, rr)
    print(json.dumps(mach, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
