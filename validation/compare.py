#!/usr/bin/env python
"""Score the Python twins against Spectre (and against themselves).

    python validation/compare.py              # twin vs every imported Spectre CSV
    python validation/compare.py --self-check # twin at dt vs dt/2 vs dt/4
    python validation/compare.py --all        # both

Pass rule, per (model, stimulus, reference), on the observed conductance G:
    NRMSE      = rms(G_twin - G_ref) / range(G_ref)      <= 1 %
    max error  = max|G_twin - G_ref| / range(G_ref)      <= 3 %
range(G_ref) = max - min of the reference conductance over the stimulus.
Both are evaluated on a common grid: the twin's own edge-refined time grid,
clipped to the span both sources cover (references are linearly interpolated
onto it).  Before scoring, the reference's measured gate current must agree
with the stimulus in sign and in total charge (1 %), so a reversed or wrong
source fails loudly instead of scoring a different experiment.

Writes validation/data/summary.csv (references), validation/data/self_check.csv
and overlay figures in validation/figures/.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation import common, stimuli  # noqa: E402

NRMSE_LIMIT = 0.01
MAXERR_LIMIT = 0.03
CHARGE_TOL = 0.01
REFERENCES = ["spectre"]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def metrics(g_test, g_ref):
    """NRMSE and max error (fractions of the reference range) on one grid."""
    g_test = np.asarray(g_test, dtype=float)
    g_ref = np.asarray(g_ref, dtype=float)
    rng = float(np.nanmax(g_ref) - np.nanmin(g_ref))
    d = g_test - g_ref
    if not rng > 0.0 or not np.all(np.isfinite(d)):
        return {"nrmse": math.nan, "maxerr": math.nan, "range": rng}
    return {"nrmse": float(np.sqrt(np.mean(d * d)) / rng),
            "maxerr": float(np.max(np.abs(d)) / rng), "range": rng}


def verdict(m):
    ok = (m["nrmse"] <= NRMSE_LIMIT) and (m["maxerr"] <= MAXERR_LIMIT)
    return "PASS" if ok else "FAIL"


def common_grid(t_twin, t_ref):
    lo, hi = max(t_twin[0], t_ref[0]), min(t_twin[-1], t_ref[-1])
    return t_twin[(t_twin >= lo) & (t_twin <= hi)]


def twin_charge(t, i):
    """Twin drive is piecewise constant: i[k+1] holds over (t[k], t[k+1])."""
    return float(np.sum(np.abs(i[1:]) * np.diff(t)))


def pwl_charge(t, i):
    a = np.abs(i)
    return float(np.sum(0.5 * (a[1:] + a[:-1]) * np.diff(t)))


def drive_check(twin, ref):
    """'ok' | 'sign flipped' | 'charge mismatch x%' | 'drive not measured'."""
    if not np.any(np.isfinite(ref["drive_A"])):
        return "drive not measured"
    tg = common_grid(twin["t_s"], ref["t_s"])
    # sample the twin at segment centres to avoid landing on an edge
    mid = 0.5 * (tg[1:] + tg[:-1])
    ref_i = np.interp(mid, ref["t_s"], ref["drive_A"])
    twin_i = np.interp(mid, twin["t_s"][1:], twin["drive_A"][1:], left=0.0)
    if np.dot(ref_i, twin_i) < 0:
        return "sign flipped"
    q_t = twin_charge(twin["t_s"], twin["drive_A"])
    q_r = pwl_charge(ref["t_s"], ref["drive_A"])
    if q_t > 0 and abs(q_r - q_t) / q_t > CHARGE_TOL:
        return f"charge mismatch {100*(q_r-q_t)/q_t:+.1f}%"
    return "ok"


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def overlay(path, title, t, series, ref_label, ref_g):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = float(np.max(ref_g) - np.min(ref_g)) or 1.0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.3]})
    a1.plot(t, ref_g * 1e6, color="0.15", lw=1.8, label=ref_label)
    for label, g, style in series:
        a1.plot(t, g * 1e6, style, lw=1.0, label=label)
        a2.plot(t, 100 * (g - ref_g) / rng, style, lw=0.9, label=label)
    a1.set_ylabel("G (uS)")
    a1.set_title(title, fontsize=10)
    a1.legend(fontsize=8, loc="best")
    a2.axhline(0, color="0.6", lw=0.6)
    for lim in (MAXERR_LIMIT, -MAXERR_LIMIT):
        a2.axhline(100 * lim, color="tab:red", lw=0.6, ls=":")
    a2.set_ylabel("error (% range)")
    a2.set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# twin vs references
# --------------------------------------------------------------------------
def _twin(key, name):
    p = common.data_path("twin", key, name)
    if not p.is_file():
        cols, wall = common.run_twin(key, stimuli.BY_NAME[name])
        common.write_csv(p, cols, common.provenance_line(
            "Python twin (simulation_core)", key))
        common.write_meta(p.with_suffix(".json"), {
            "source": "twin", "model": key, "stimulus": name,
            "stimulus_set": stimuli.STIMULUS_SET_VERSION,
            "n_points": int(len(cols["t_s"])), "wall_s": round(wall, 4)})
    cols, _ = common.read_csv(p)
    return cols, common.read_meta(p.with_suffix(".json"))


def compare_references(keys, names, figures=True):
    rows = []
    for key in keys:
        for name in names:
            for ref_name in REFERENCES:
                rp = common.data_path(ref_name, key, name)
                if not rp.is_file():
                    continue
                twin, tmeta = _twin(key, name)
                ref, _prov = common.read_csv(rp)
                rmeta = common.read_meta(rp.with_suffix(".json"))
                tg = common_grid(twin["t_s"], ref["t_s"])
                g_t = np.interp(tg, twin["t_s"], twin["G_S"])
                g_r = np.interp(tg, ref["t_s"], ref["G_S"])
                m = metrics(g_t, g_r)
                dchk = drive_check(twin, ref)
                v = verdict(m) if dchk in ("ok", "drive not measured") else "FAIL"
                tw, rw = tmeta.get("wall_s"), rmeta.get("wall_s")
                rows.append({
                    "model": key, "stimulus": name, "reference": ref_name,
                    "n_points": len(tg), "nrmse_pct": 100 * m["nrmse"],
                    "maxerr_pct": 100 * m["maxerr"], "g_range_uS": 1e6 * m["range"],
                    "twin_wall_s": tw, "ref_wall_s": rw,
                    "speedup": (rw / tw) if (tw and rw) else None,
                    "drive_check": dchk, "verdict": v})
                if figures:
                    overlay(common.FIGURES / f"{key}_{name}.png",
                            f"{key} {name}: twin vs {ref_name}  NRMSE "
                            f"{100*m['nrmse']:.3f}%  max {100*m['maxerr']:.3f}%  [{v}]",
                            tg, [("twin", g_t, "tab:blue")], ref_name, g_r)
    return rows


# --------------------------------------------------------------------------
# self-check: time-step convergence of the twin
# --------------------------------------------------------------------------
def self_check(keys, names, figures=True):
    rows = []
    for key in keys:
        for name in names:
            st = stimuli.BY_NAME[name]
            runs = {f: common.run_twin(key, st, refine=f) for f in (1, 2, 4)}
            g = {f: runs[f][0]["G_S"][::f] for f in runs}   # base grid points
            t = runs[1][0]["t_s"]
            m1 = metrics(g[1], g[4])
            m2 = metrics(g[2], g[4])
            # observed order p from three refinements (Richardson):
            #   p = log2( |G_dt - G_dt/2| / |G_dt/2 - G_dt/4| )
            d12 = float(np.sqrt(np.mean((g[1] - g[2]) ** 2)))
            d24 = float(np.sqrt(np.mean((g[2] - g[4]) ** 2)))
            order = math.log2(d12 / d24) if d12 > 0 and d24 > 0 else math.nan
            rows.append({
                "model": key, "stimulus": name, "n_points": len(t),
                "g_range_uS": 1e6 * m1["range"],
                "nrmse_dt_vs_dt4_pct": 100 * m1["nrmse"],
                "maxerr_dt_vs_dt4_pct": 100 * m1["maxerr"],
                "nrmse_dt2_vs_dt4_pct": 100 * m2["nrmse"],
                "maxerr_dt2_vs_dt4_pct": 100 * m2["maxerr"],
                "observed_order": order,
                "verdict": verdict(m1)})
            if figures:
                overlay(common.FIGURES / "selfcheck" / f"{key}_{name}.png",
                        f"{key} {name}: twin dt, dt/2 vs dt/4  NRMSE "
                        f"{100*m1['nrmse']:.2e}% / {100*m2['nrmse']:.2e}%",
                        t, [("dt", g[1], "tab:blue"), ("dt/2", g[2], "tab:orange")],
                        "dt/4", g[4])
    return rows


# --------------------------------------------------------------------------
def _fmt(v, spec):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return format(v, spec)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})


def print_ref_table(rows):
    print(f"{'model':5s} {'stimulus':20s} {'ref':8s} {'NRMSE%':>8s} {'max%':>8s} "
          f"{'range uS':>9s} {'speedup':>8s}  drive        verdict")
    for r in rows:
        print(f"{r['model']:5s} {r['stimulus']:20s} {r['reference']:8s} "
              f"{_fmt(r['nrmse_pct'], '8.3f')} {_fmt(r['maxerr_pct'], '8.3f')} "
              f"{_fmt(r['g_range_uS'], '9.2f')} {_fmt(r['speedup'], '8.1f')}  "
              f"{r['drive_check']:12s} {r['verdict']}")


def print_self_table(rows):
    print(f"{'model':5s} {'stimulus':20s} {'NRMSE% dt|dt/4':>15s} "
          f"{'NRMSE% dt/2|dt/4':>17s} {'max% dt|dt/4':>13s} {'order':>6s} verdict")
    for r in rows:
        print(f"{r['model']:5s} {r['stimulus']:20s} "
              f"{_fmt(r['nrmse_dt_vs_dt4_pct'], '15.2e')} "
              f"{_fmt(r['nrmse_dt2_vs_dt4_pct'], '17.2e')} "
              f"{_fmt(r['maxerr_dt_vs_dt4_pct'], '13.2e')} "
              f"{_fmt(r['observed_order'], '6.2f')} {r['verdict']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="self-check plus every available reference")
    ap.add_argument("--model", default="all")
    ap.add_argument("--stimulus", nargs="*")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    keys = common.model_keys(args.model)
    names = [s.name for s in stimuli.get(args.stimulus)]
    figs = not args.no_figures
    failed = False

    if args.self_check or args.all:
        rows = self_check(keys, names, figs)
        print("== self-check: twin time-step convergence (reference = dt/4) ==")
        print_self_table(rows)
        write_rows(common.DATA / "self_check.csv", rows)
        failed |= any(r["verdict"] != "PASS" for r in rows)

    if not args.self_check or args.all:
        rows = compare_references(keys, names, figs)
        print("== twin vs circuit-simulator references ==")
        if rows:
            print_ref_table(rows)
            write_rows(common.DATA / "summary.csv", rows)
            failed |= any(r["verdict"] != "PASS" for r in rows)
        else:
            print("no reference data under validation/data/spectre/ yet - run the "
                  "decks in validation/spice/spectre/ and import them with "
                  "validation/scrub_spectre.py (see validation/README.md)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
