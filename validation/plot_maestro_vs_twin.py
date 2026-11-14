"""Overlay Cadence Maestro (Spectre, schematic testbench) and the NeuroVAT twin.

Reads validation/data/maestro/<model>_<stimulus>.csv (exported from Maestro with
OCEAN) and the twin CSVs, and writes one figure per device plus a summary table.

    python validation/plot_maestro_vs_twin.py [--out DIR]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "validation" / "data"


def load(path):
    rows = [r for r in csv.reader(l for l in open(path) if not l.startswith("#"))]
    h = rows[0]
    return h, np.array(rows[1:], float)


def col(h, d, name):
    return d[:, h.index(name)]


JOBS = [  # (title, maestro csv, maestro col, twin csv, twin col, scale, unit, time scale, time unit)
    ("ECFET v3 - single +50 pA, 10 ms pulse", "v3_single_pulse", "n_rmem",
     "twin/v3/single_pulse.csv", "R_ohm", 1e-3, "R (kOhm)", 1.0, "s"),
    ("ECFET v3 - LTP/LTD train (20 + 20 pulses)", "v3_ltp_ltd_train", "n_rmem",
     "twin/v3/ltp_ltd_train.csv", "R_ohm", 1e-3, "R (kOhm)", 1.0, "s"),
    ("ECFET v3 - paired pulses, 5 ms to 1 s gaps", "v3_gap_sweep", "n_rmem",
     "twin/v3/gap_sweep.csv", "R_ohm", 1e-3, "R (kOhm)", 1.0, "s"),
    ("Stanford-PKU RRAM - RESET pulse + read", "rram_pulse_reset_ref", "I0:TE",
     "stanford/twin/pulse_reset_ref.csv", "I_TE_A", 1e3, "I(TE) (mA)", 1e9, "ns"),
    ("Stanford-PKU RRAM - +/-1.3 V train", "rram_s1_pm1p3", "I0:TE",
     "stanford/twin/s1_pm1p3.csv", "I_TE_A", 1e3, "I(TE) (mA)", 1e9, "ns"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "validation" / "figures" / "maestro"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    fig, axes = plt.subplots(len(JOBS), 1, figsize=(9, 2.6 * len(JOBS)))
    for ax, (title, mfile, mcol, tfile, tcol, sc, unit, ts, tu) in zip(axes, JOBS):
        hm, m = load(D / "maestro" / f"{mfile}.csv")
        ht, t = load(D / tfile)
        tm, ym = m[:, 0], col(hm, m, mcol)
        yt = np.interp(tm, t[:, 0], col(ht, t, tcol))
        if "n_te" in hm:
            # RRAM: score read conductance only while the 0.1 V read is applied
            # (the validation-set metric); raw current across 50 ps switching
            # edges is dominated by sub-step timing offsets.
            vm = col(hm, m, "n_te")
            sel = np.abs(vm - 0.1) < 1e-6
            gm = ym[sel] / vm[sel]
            gt = yt[sel] / vm[sel]
            # one read level (no range): normalise by the conductance value
            gmean = float(np.mean(np.abs(gm)))
            rng = np.ptp(gm) if np.ptp(gm) > 0.01 * gmean else gmean
            nrmse = 100 * np.sqrt(np.mean((gt - gm) ** 2)) / rng
            mx = 100 * np.max(np.abs(gt - gm)) / rng
            metric = "read G"
        else:
            rng = np.ptp(ym) or 1.0
            nrmse = 100 * np.sqrt(np.mean((yt - ym) ** 2)) / rng
            mx = 100 * np.max(np.abs(yt - ym)) / rng
            metric = "R"
        summary.append((title, len(tm), nrmse, mx))
        ax.plot(tm * ts, ym * sc, lw=2.4, color="#1f77b4", label="Cadence Maestro (Spectre)")
        ax.plot(t[:, 0] * ts, col(ht, t, tcol) * sc, lw=1.0, ls="--", color="#d62728",
                label="NeuroVAT twin")
        ax.set_title(f"{title}   [{metric}: NRMSE {nrmse:.3f} %, max {mx:.2f} % of range]", fontsize=9)
        ax.set_ylabel(unit)
        ax.set_xlabel(f"time ({tu})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out / "maestro_vs_neurovat.png", dpi=150)
    with open(out / "maestro_vs_neurovat.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "points", "nrmse_pct", "max_pct"])
        for r in summary:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.3f}"])
    for r in summary:
        print(f"{r[0]:48s} n={r[1]:5d}  NRMSE {r[2]:.4f} %  max {r[3]:.3f} %")


if __name__ == "__main__":
    main()
