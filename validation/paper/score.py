#!/usr/bin/env python
"""Score the twins and the Spectre runs against Sharbati et al. (2018).

Three scores, reported separately:
  1. twin vs Spectre  - the Gate A rule (NRMSE <= 1 %, max <= 3 %) per stimulus,
                        once scrubbed Spectre data exists (data/sharbati/spectre/);
  2. Spectre vs paper - each text target in sharbati2018_targets.json;
  3. twin vs paper    - the same targets from the twin data.

    python validation/paper/score.py                 # all models, write tables
    python validation/paper/score.py --model v3 --no-figures

Writes validation/data/sharbati/score_table.csv (target rows plus diagnostic
rows), twin_vs_spectre.csv (when Spectre data exists), curve_scores.csv (when
a curve CSV under validation/paper/curves/ has points) and an overview figure
per model in validation/figures/sharbati/.  Missing twin data is generated
first (run_twin_paper.py does the same explicitly).
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation import common, compare  # noqa: E402
from validation.paper import core  # noqa: E402
from validation.paper import metrics as M  # noqa: E402
from validation.paper import sharbati_stimuli as S  # noqa: E402

SOURCES = ("twin", "spectre")

# diagnostic rows (no paper number): (label, metric key, unit)
DIAGNOSTICS = [
    ("R0 (model operating point)", "sp_pot.R0_ohm", "ohm"),
    ("instantaneous step per event", "sp_pot.step_pct_instant", "%"),
    ("dR 35 s after a depression pulse, relative", "sp_dep.dR_fig_pct", "%"),
    ("Fig.4b fit R^2 (free taus)", "pp.fit_r2", ""),
    ("Fig.4b fit R^2 (taus fixed 22/315/19000 ms)", "pp.fixed_tau_r2", ""),
    ("Fig.4b dG at dt = 15 ms", "pp.dG_first_uS", "uS"),
    ("Fig.4b dG at dt = 1800 ms", "pp.dG_last_uS", "uS"),
    ("Fig.4b alt dG (peak2 - G(t2)) span / mean", "pp.dG_alt_span_pct", "%"),
    ("LTD linearity R^2 (10-90 %)", "ltp.ltd_r2_10_90", ""),
    ("LTP linearity R^2 over 10-90 % of x (v3)", "ltp.r2_x10_90", ""),
    ("LTP reads inside 10-90 %", "ltp.n_reads_10_90", "pulses"),
    ("LTP median step (10-90 %)", "ltp.step_med_10_90_uS", "uS"),
    ("LTP max per-event step (10-90 %)", "ltp.step_pct_max_10_90", "%"),
    ("G window min", "ltp.G_min_uS", "uS"),
    ("G window max", "ltp.G_max_uS", "uS"),
]


def load_targets(path=None):
    return json.loads(Path(path or core.TARGETS).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
def _cols(source, key, name, make_twin):
    cols = core.load(source, key, name)
    if cols is None and source == "twin" and make_twin:
        core.write_twin(key, S.BY_NAME[name])
        cols = core.load(source, key, name)
    return cols


def extract_all(source, key, make_twin=True, loader=None):
    """Flat {'sp_pot.dR_peak_ohm': v, ...} for one model and source.  Groups
    whose data is missing are left out (-> 'pending')."""
    get = loader or (lambda n: _cols(source, key, n, make_twin))
    out, series = {}, {}
    for name, prefix, fn in (("sp_pot", "sp_pot", M.single_pulse),
                             ("sp_dep", "sp_dep", M.single_pulse),
                             ("ltp_ltd", "ltp", M.ltp_ltd),
                             ("retention_13h", "ret", M.retention)):
        c = get(name)
        if c is None:
            continue
        series[name] = c
        for k, v in fn(c, S.BY_NAME[name]).items():
            out[f"{prefix}.{k}"] = v
    runs = [(get(n), S.BY_NAME[n]) for n in S.PAIR_NAMES]
    if all(c is not None for c, _ in runs):
        pp = M.paired_series(runs)
        series["pp"] = pp
        for k, v in pp.items():
            if not isinstance(v, (list, np.ndarray)):
                out[f"pp.{k}"] = v
    return out, series


def _rel_err(model, paper):
    if model is None or paper in (None, 0) or (isinstance(model, float) and math.isnan(model)):
        return None
    return 100.0 * (model - paper) / abs(paper)


def _judge(t, model):
    if model is None or (isinstance(model, float) and math.isnan(model)):
        return "pending"
    if t["comparison"] == "gt":
        return "met" if model > t["value"] else "NOT met"
    if t["comparison"] == "lt":
        return "met" if model < t["value"] else "NOT met"
    e = _rel_err(model, t["value"])
    return f"{e:+.1f}%"


def score_rows(targets, values):
    """values: {(key, source): flat metric dict}.  -> list of row dicts."""
    keys = sorted({k for k, _ in values})
    rows = []
    for key in keys:
        for t in targets["targets"]:
            tw = values.get((key, "twin"), {}).get(t["metric"])
            sp = values.get((key, "spectre"), {}).get(t["metric"])
            rows.append({"model": key, "target": t["id"], "unit": t["unit"],
                         "paper": ({"gt": ">", "lt": "<"}.get(t["comparison"], "")
                                   + f"{t['value']:g}"),
                         "twin": tw, "spectre": sp,
                         "twin_vs_paper": _judge(t, tw), "spectre_vs_paper": _judge(t, sp),
                         "role": t["role"], "source_type": t["source_type"],
                         "page_figure": t["page_figure"]})
        for label, mkey, unit in DIAGNOSTICS:
            tw = values.get((key, "twin"), {}).get(mkey)
            sp = values.get((key, "spectre"), {}).get(mkey)
            if tw is None and sp is None:
                continue
            rows.append({"model": key, "target": f"(diag) {label}", "unit": unit,
                         "paper": "-", "twin": tw, "spectre": sp,
                         "twin_vs_paper": "-", "spectre_vs_paper": "-",
                         "role": "diagnostic", "source_type": "-", "page_figure": "-"})
    return rows


def _fmt(v):
    if v is None:
        return "pending"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.4g}" if abs(v) < 1e5 else f"{v:.4e}"
    return str(v)


def print_score(rows):
    print(f"{'model':5s} {'target':44s} {'unit':6s} {'paper':>8s} {'twin':>10s} "
          f"{'Spectre':>10s} {'twin-paper':>10s} {'Spec-paper':>10s}  role")
    for r in rows:
        print(f"{r['model']:5s} {r['target'][:44]:44s} {r['unit']:6s} {r['paper']:>8s} "
              f"{_fmt(r['twin']):>10s} {_fmt(r['spectre']):>10s} "
              f"{r['twin_vs_paper']:>10s} {r['spectre_vs_paper']:>10s}  {r['role']}")


def write_csv_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (_fmt(v) if (v is None or isinstance(v, float)) else v)
                        for k, v in r.items()})


# --------------------------------------------------------------------------
def twin_vs_spectre(keys):
    rows = []
    for key in keys:
        for st in S.STIMULI:
            ref = core.load("spectre", key, st.name)
            if ref is None:
                continue
            twin = _cols("twin", key, st.name, True)
            tg = compare.common_grid(twin["t_s"], ref["t_s"])
            m = compare.metrics(np.interp(tg, twin["t_s"], twin["G_S"]),
                                np.interp(tg, ref["t_s"], ref["G_S"]))
            dchk = compare.drive_check(twin, ref)
            v = compare.verdict(m) if dchk in ("ok", "drive not measured") else "FAIL"
            rows.append({"model": key, "stimulus": st.name, "n_points": len(tg),
                         "nrmse_pct": 100 * m["nrmse"], "maxerr_pct": 100 * m["maxerr"],
                         "g_range_uS": 1e6 * m["range"], "drive_check": dchk, "verdict": v})
    return rows


# --------------------------------------------------------------------------
def read_curve(path):
    """-> (header dict, x array, y array); empty arrays for an empty slot."""
    head, xs, ys, seen_cols = {}, [], [], False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            k, _, v = s[1:].partition(":")
            head.setdefault(k.strip(), v.strip())
            continue
        if not seen_cols:
            seen_cols = True
            continue
        a, b = s.split(",")[:2]
        xs.append(float(a))
        ys.append(float(b))
    return head, np.array(xs), np.array(ys)


def model_curve(figure, series):
    """Model counterpart of a curve slot, in the slot's units, or None."""
    if figure in ("Fig. 3a", "Fig. 3b"):
        c = series.get("sp_pot" if figure == "Fig. 3a" else "sp_dep")
        if c is None:
            return None
        return c["t_s"] - S.SP_T_ON, c["R_ohm"]
    if figure == "Fig. 3c":
        c = series.get("ltp_ltd")
        if c is None:
            return None
        ltp, _ = M.ltp_reads(c, S.BY_NAME["ltp_ltd"])
        return np.arange(len(ltp), dtype=float), 1e6 * ltp
    if figure == "Fig. 4b":
        pp = series.get("pp")
        if pp is None:
            return None
        return 1e3 * pp["dt_s"], 1e6 * pp["dG_S"]
    if figure == "Fig. S5":
        c = series.get("retention_13h")
        if c is None:
            return None
        st = S.BY_NAME["retention_13h"]
        t0 = st.params["t_settle_s"]
        g0 = float(np.interp(t0, c["t_s"], c["G_S"]))
        m = c["t_s"] >= t0
        return (c["t_s"][m] - t0) / 3600.0, 100 * (c["G_S"][m] - g0) / g0
    return None


def curve_score(x, y, mx, my):
    f = np.interp(x, mx, my)
    res = y - f
    tot = y - y.mean()
    r2 = float(1 - np.dot(res, res) / np.dot(tot, tot)) if len(y) > 1 and np.dot(tot, tot) > 0 else math.nan
    return {"n": int(len(y)), "r2": r2, "rmse": float(np.sqrt(np.mean(res * res)))}


def curve_scores(targets, all_series):
    rows = []
    for slot in targets["curve_slots"]:
        path = core.PAPER_DIR / slot["file"]
        if not path.is_file():
            continue
        head, x, y = read_curve(path)
        if len(x) == 0:
            continue
        for (key, source), series in sorted(all_series.items()):
            mc = model_curve(slot["figure"], series)
            if mc is None:
                continue
            s = curve_score(x, y, *mc)
            rows.append({"figure": slot["figure"], "model": key, "source": source,
                         "curve_source_type": head.get("source_type", ""), **s,
                         "unit": slot["y_unit"]})
    return rows


# --------------------------------------------------------------------------
def overview_figure(key, series_by_src, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    styles = {"twin": ("tab:blue", "-"), "spectre": ("0.15", "--")}
    for src, series in series_by_src.items():
        col, ls = styles[src]
        for name, a in (("sp_pot", ax[0, 0]), ("sp_dep", ax[0, 1])):
            c = series.get(name)
            if c is not None:
                a.plot(c["t_s"] - S.SP_T_ON, c["R_ohm"], color=col, ls=ls, lw=1, label=src)
        c = series.get("ltp_ltd")
        if c is not None:
            ltp, ltd = M.ltp_reads(c, S.BY_NAME["ltp_ltd"])
            n = np.arange(len(ltp) + len(ltd))
            ax[1, 0].plot(n, 1e6 * np.concatenate([ltp, ltd]), color=col, ls=ls, lw=1, label=src)
        pp = series.get("pp")
        if pp is not None:
            ax[1, 1].plot(1e3 * pp["dt_s"], 1e6 * pp["dG_S"], "o", color=col, ms=4,
                          label=f"{src} peak2-peak1")
            ax[1, 1].plot(1e3 * pp["dt_s"], 1e6 * pp["dG_alt_S"], "s", color=col, ms=3,
                          mfc="none", label=f"{src} peak2-G(t2)")
            if np.isfinite(pp["tau1_ms"]):
                d = np.linspace(pp["dt_s"][0], pp["dt_s"][-1], 300)
                c_ = [c * 1e-6 for c in pp["fit_c_uS"]]
                f = M.three_exp(d, c_[0], pp["tau1_ms"] / 1e3, c_[1], pp["tau2_ms"] / 1e3,
                                c_[2], pp["tau3_s"])
                ax[1, 1].plot(1e3 * d, 1e6 * f, color=col, lw=0.8, ls=":")
    ax[0, 0].set(title="Fig. 3a protocol: +50 pA / 10 ms", xlabel="t - t_on (s)", ylabel="R (ohm)", xlim=(-2, 40))
    ax[0, 1].set(title="Fig. 3b protocol: -50 pA / 10 ms", xlabel="t - t_on (s)", ylabel="R (ohm)", xlim=(-2, 40))
    ax[1, 0].set(title="Fig. 3c protocol: LTP then LTD reads", xlabel="read #", ylabel="G (uS)")
    ax[1, 1].set(title="Fig. 4b protocol: same-polarity pairs", xlabel="onset-to-onset dt (ms)", ylabel="dG (uS)")
    for a in ax.ravel():
        a.legend(fontsize=7)
    fig.suptitle(f"{key}: sharbati-1 overlay (model only; no paper curve data yet)", fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    keys = common.model_keys(args.model)
    targets = load_targets()
    print(f"{targets['paper']}\nstimulus set {S.STIMULUS_SET_VERSION} ({S.fingerprint()[:16]})"
          f"\nheld-out targets: {targets['held_out_summary']}\n")

    values, all_series = {}, {}
    for key in keys:
        for src in SOURCES:
            v, ser = extract_all(src, key)
            if v:
                values[(key, src)] = v
                all_series[(key, src)] = ser
        values.setdefault((key, "twin"), {})
    rows = score_rows(targets, values)
    print("== paper targets (error = (model - paper) / |paper|) ==")
    print_score(rows)
    write_csv_rows(core.DATA / "score_table.csv", rows)

    tvs = twin_vs_spectre(keys)
    print("\n== twin vs Spectre (Gate A rule: NRMSE <= 1 %, max <= 3 %) ==")
    if tvs:
        compare.print_ref_table([{**r, "reference": "spectre", "speedup": None} for r in tvs])
        write_csv_rows(core.DATA / "twin_vs_spectre.csv", tvs)
    else:
        print("pending: no data under validation/data/sharbati/spectre/ - run the decks in "
              "validation/spice/spectre_sharbati/ and import them with "
              "validation/paper/scrub_paper.py")

    cs = curve_scores(targets, all_series)
    print("\n== curve overlays (R^2 / RMSE vs paper curve CSVs) ==")
    if cs:
        for r in cs:
            print(f"  {r['figure']:8s} {r['model']} {r['source']:8s} n={r['n']:4d} "
                  f"R^2={r['r2']:.4f} RMSE={r['rmse']:.4g} {r['unit']}")
        write_csv_rows(core.DATA / "curve_scores.csv", cs)
    else:
        print("none: the curve slots in validation/paper/curves/ are empty")

    if not args.no_figures:
        for key in keys:
            ser = {src: all_series[(key, src)] for src in SOURCES if (key, src) in all_series}
            p = overview_figure(key, ser, core.FIGURES / f"{key}_overview.png")
            print(f"figure -> {p.relative_to(common.ROOT).as_posix()}")
    return 1 if any(r["verdict"] != "PASS" for r in tvs) else 0


if __name__ == "__main__":
    sys.exit(main())
