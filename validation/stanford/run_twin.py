#!/usr/bin/env python
"""Run the stanford-1 stimuli through the Stanford-PKU RRAM twin.

    python validation/stanford/run_twin.py                # data + figures
    python validation/stanford/run_twin.py --self-check   # dt, dt/2, dt/4 + brute force
    python validation/stanford/run_twin.py --stimulus reset_train set_train

Writes validation/data/stanford/twin/<stim>.csv (+ .json wall time / cost),
validation/data/stanford/self_check.csv, validation/data/stanford/brute_check.csv
and figures under validation/figures/stanford/.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation import common, compare  # noqa: E402
from validation.stanford import core  # noqa: E402
from validation.stanford import stimuli as S  # noqa: E402


def figure(stim, cols, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = cols["t_s"]
    dc = stim.name.startswith("dc_")
    scale, unit = (1e3, "ms") if t[-1] > 1e-4 else ((1e6, "us") if t[-1] > 2e-6 else (1e9, "ns"))
    fig, ax = plt.subplots(3 + dc, 1, figsize=(8, 7.5 + 2.5 * dc))
    ax[0].plot(t * scale, cols["V_TE_V"], color="0.2", lw=0.9)
    ax[0].set_ylabel("V(TE,BE) (V)")
    ax[0].set_title(f"{stim.name}: {stim.description}", fontsize=8)
    ax[1].semilogy(t * scale, np.abs(cols["I_TE_A"]) + 1e-15, color="tab:blue", lw=0.9)
    ax[1].set_ylabel("|I_TE| (A)")
    ax[2].plot(t * scale, cols["gap_m"] * 1e9, color="tab:red", lw=1.0, label="gap (clamped)")
    ax[2].plot(t * scale, cols["gap_state_m"] * 1e9, color="tab:orange", lw=0.8, ls="--",
               label="idt state y")
    lo = min(-0.1, float(np.nanmin(cols["gap_state_m"]) * 1e9))
    hi = max(2.0, float(np.nanmax(cols["gap_state_m"]) * 1e9))
    if hi - lo > 6:
        ax[2].set_ylim(-0.1, 2.0)
        ax[2].text(0.01, 0.05, f"y range {lo:.3g}..{hi:.3g} nm (clipped view)",
                   transform=ax[2].transAxes, fontsize=7)
    ax[2].set_ylabel("gap (nm)")
    ax[2].legend(fontsize=7, loc="best")
    ax[2].set_xlabel(f"t ({unit})")
    if len(stim.reads):
        g = core.read_conductance(cols, stim)
        for tr in stim.read_times:
            ax[1].axvline(tr * scale, color="0.85", lw=0.5, zorder=0)
        ax2 = ax[2].twinx()
        ax2.plot(stim.read_times * scale, g * 1e6, "k.", ms=4)
        ax2.set_ylabel("G_read (uS)")
    if dc:
        ax[3].semilogy(cols["V_TE_V"], np.abs(cols["I_TE_A"]) + 1e-15, color="tab:blue", lw=1.0)
        ax[3].set_xlabel("V(TE,BE) (V)")
        ax[3].set_ylabel("|I_TE| (A)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def hysteresis_figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for name, color in (("dc_reset_ref", "tab:blue"), ("dc_set_ref", "tab:red")):
        c = core.load("twin", name)
        if c is not None:
            axs[0].semilogy(c["V_TE_V"], np.abs(c["I_TE_A"]) + 1e-15, color=color, label=name)
    axs[0].set_title("HSPICE-template half sweeps (0.5 V/ms)", fontsize=9)
    c = core.load("twin", "dc_loop")
    if c is not None:
        n = len(c["t_s"]) // 2
        axs[1].semilogy(c["V_TE_V"][:n], np.abs(c["I_TE_A"][:n]) + 1e-15, label="loop 1")
        axs[1].semilogy(c["V_TE_V"][n:], np.abs(c["I_TE_A"][n:]) + 1e-15, ls="--", label="loop 2")
    axs[1].set_title("dc_loop: two loops in one transient (1 V/us)", fontsize=9)
    for a in axs:
        a.set_xlabel("V(TE,BE) (V)")
        a.set_ylabel("|I_TE| (A)")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_all(names, figures=True):
    total = 0.0
    for st in S.get(names):
        cols, wall, n_rhs = core.run_twin(st)
        total += wall
        p = core.data_path("twin", st.name)
        core.write_csv(p, cols, core.provenance_line("Python twin (twins/stanford_pku_rram.py)"))
        common.write_meta(p.with_suffix(".json"), {
            "source": "twin", "model": core.MODEL, "stimulus": st.name,
            "stimulus_set": S.STIMULUS_SET_VERSION, "n_points": int(len(cols["t_s"])),
            "rhs_evaluations": int(n_rhs), "wall_s": round(wall, 4)})
        g = core.read_conductance(cols, st)
        gtxt = (f"G_read {g.min()*1e6:8.2f}..{g.max()*1e6:8.2f} uS" if len(g) else
                f"|I| max {np.max(np.abs(cols['I_TE_A']))*1e3:8.3f} mA")
        print(f"  {st.name:20s} {len(cols['t_s']):7d} pts {wall:6.2f} s  "
              f"gap {cols['gap_m'].min()*1e9:5.3f}..{cols['gap_m'].max()*1e9:5.3f} nm  "
              f"y {cols['gap_state_m'].min()*1e9:9.3f}..{cols['gap_state_m'].max()*1e9:9.3f} nm  {gtxt}")
        if figures:
            figure(st, cols, core.FIGURES / f"twin_{st.name}.png")
    if figures:
        hysteresis_figure(core.FIGURES / "twin_iv_hysteresis.png")
    print(f"total twin wall time {total:.2f} s")


def self_check(names):
    rows = []
    for st in S.get(names):
        runs = {f: core.run_twin(st, refine=f)[0] for f in (1, 2, 4)}
        sub = {f: {k: v[::f] for k, v in runs[f].items()} for f in runs}
        r1 = core.score(sub[1], sub[4], st)
        r2 = core.score(sub[2], sub[4], st)
        d12 = float(np.sqrt(np.mean((sub[1]["gap_state_m"] - sub[2]["gap_state_m"]) ** 2)))
        d24 = float(np.sqrt(np.mean((sub[2]["gap_state_m"] - sub[4]["gap_state_m"]) ** 2)))
        rows.append({"stimulus": st.name, "n_points": r1["n_points"], "n_reads": r1["n_reads"],
                     "g_read_max_pct_dt_vs_dt4": r1["g_read_max_pct"],
                     "g_read_max_pct_dt2_vs_dt4": r2["g_read_max_pct"],
                     "i_nrmse_pct_dt_vs_dt4": r1["i_nrmse_pct"],
                     "gap_nrmse_pct_dt_vs_dt4": r1["gap_nrmse_pct"],
                     "state_rms_dt_dt2_m": d12, "state_rms_dt2_dt4_m": d24,
                     "verdict": core.verdict(r1)})
    return rows


def brute_cases():
    """(stimulus, [(label, t_ps)], brute dt) for the literal-Euler check:
    a pulse pair (constant-V path, wind-up below gap_min) and a 1e7 V/s
    triangle loop (ramp path: sliding on / freezing at the RESET stop gap)."""
    b = S._Builder()
    b.wait(1000)
    b.pulse(-1.3, 10 * S.NS)
    b.read()
    b.wait(3 * S.NS)
    b.pulse(1.5, 10 * S.NS)
    b.read()
    pair = b.build("brute_pair", "RESET -1.3 V/10 ns + SET +1.5 V/10 ns", 2 * S.NS, 0.1e-9)
    r = S._Builder()
    for v in (-1.5, 0.0, 1.5, 0.0):
        r.ramp_to(v, 150 * S.NS)
    ramp = r.build("brute_ramp", "0 -> -1.5 -> 0 -> +1.5 -> 0 V at 1e7 V/s", 0, 0.1e-9)
    return [(pair, [("pair: after RESET pulse", pair.reads[0]),
                    ("pair: after SET pulse", pair.reads[1])], 0.2e-12),
            (ramp, [("ramp: at -1.5 V", 150 * S.NS), ("ramp: back at 0 V", 300 * S.NS),
                    ("ramp: at +1.5 V", 450 * S.NS), ("ramp: end", 600 * S.NS)], 0.5e-12)]


def brute_check():
    """Twin vs a literal one-evaluation-per-point Euler transcription of the
    .va, Richardson-extrapolated from dt and 2 dt (Euler is first order)."""
    rows = []
    for st, points, dt in brute_cases():
        tw, _, _ = core.run_twin(st)
        t0 = time.perf_counter()
        tb, gb, yb = core.brute_reference(st.points, st.t_stop, dt)
        wall_b = time.perf_counter() - t0
        tb2, gb2, yb2 = core.brute_reference(st.points, st.t_stop, 2 * dt)
        for label, t_ps in points:
            tt = t_ps * S.PS
            k = int(np.argmin(np.abs(tb - tt)))
            k2 = int(np.argmin(np.abs(tb2 - tt)))
            yt = float(np.interp(tt, tw["t_s"], tw["gap_state_m"]))
            y_rich = 2 * yb[k] - yb2[k2]
            rows.append({"point": label, "t_s": tt, "twin_y_m": yt, "brute_y_m": float(yb[k]),
                         "brute_2dt_y_m": float(yb2[k2]), "richardson_y_m": y_rich,
                         "twin_minus_richardson_m": yt - y_rich,
                         "brute_dt_s": dt, "brute_wall_s": round(wall_b, 2)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stimulus", nargs="*")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--no-brute", action="store_true", help="skip the slow brute-force check")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    print(f"stimulus set {S.STIMULUS_SET_VERSION} ({S.fingerprint()[:16]})")
    if not args.self_check:
        run_all(args.stimulus, not args.no_figures)
        return 0
    rows = self_check(args.stimulus)
    compare.write_rows(core.DATA / "self_check.csv", rows)
    print("== self-check: twin at dt, dt/2 vs dt/4 (base-grid points) ==")
    print(f"{'stimulus':20s} {'Gmax% dt':>10s} {'Gmax% dt/2':>11s} {'I NRMSE%':>10s} "
          f"{'gap NRMSE%':>11s} {'rms y dt|dt/2 (m)':>18s} verdict")
    for r in rows:
        def f(v, s):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "-".rjust(int(s.split(".")[0]))
            return format(v, s)
        print(f"{r['stimulus']:20s} {f(r['g_read_max_pct_dt_vs_dt4'], '10.2e')} "
              f"{f(r['g_read_max_pct_dt2_vs_dt4'], '11.2e')} {f(r['i_nrmse_pct_dt_vs_dt4'], '10.2e')} "
              f"{f(r['gap_nrmse_pct_dt_vs_dt4'], '11.2e')} {r['state_rms_dt_dt2_m']:18.2e} {r['verdict']}")
    failed = any(r["verdict"] != "PASS" for r in rows)
    if not args.no_brute:
        b = brute_check()
        compare.write_rows(core.DATA / "brute_check.csv", b)
        print("== twin vs literal .va transcription (forward Euler, one evaluation per point) ==")
        for r in b:
            print(f"  {r['point']:18s} twin y {r['twin_y_m']*1e9:10.6f} nm  brute {r['brute_y_m']*1e9:10.6f} "
                  f"(2dt {r['brute_2dt_y_m']*1e9:10.6f}) Richardson {r['richardson_y_m']*1e9:10.6f} nm  "
                  f"diff {r['twin_minus_richardson_m']:+.2e} m")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
