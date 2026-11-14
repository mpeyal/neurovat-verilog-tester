#!/usr/bin/env python
"""C5 tier 1, step 3: twin vs Spectre for every synapse of the recorded
training.

    python validation/c5/compare_c5.py scrub <train.raw/tran.tran.tran> <train.log>
    python validation/c5/compare_c5.py score

`scrub` reads the Spectre PSF-ascii run of validation/spice/spectre_c5/v3/train.scs.
It writes:

- `validation/data/c5/spectre_1ms.csv`: t_s, then G_full and G_nv per synapse
  on a 1 ms grid. This file is committed, and the leak-checked text is
  rebuilt from numbers only.
- `results/c5/spectre_full.npz`: the full-resolution copy, git-ignored, used
  by `score` when it is present.

The gate charge of every synapse is checked against the schedule.

`score` compares three references with Spectre, synapse by synapse:

- `twin`: the scalar EcfetV3 twin replaying that synapse's pulses on the
  Gate A time grid (refined at every edge, dt_max 1 ms). This is the Gate A
  comparison, applied to the training's own schedule.
- `engine`: the conductance the network itself read during training
  (`EcfetV3Array.peek`, every 1 ms and after each write). It is scored twice,
  at every read and at reads outside a write pulse.
- `final`: the G at the end of the run.

The pass rule is the Gate A rule on G: NRMSE <= 1 % and max <= 3 % of that
synapse's Spectre G range.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schedule as sch  # noqa: E402

from validation import compare, scrub_spectre as ss  # noqa: E402

DATA = sch.DATA
FULL = sch.ROOT / "results" / "c5" / "spectre_full.npz"
FIG = sch.ROOT / "validation" / "figures" / "c5"
NRMSE_LIMIT, MAXERR_LIMIT = compare.NRMSE_LIMIT, compare.MAXERR_LIMIT


# --------------------------------------------------------------------------
def scrub(src, log):
    meta, per = sch.load()
    n = len(per)
    text = Path(src).read_text(encoding="utf-8", errors="replace")
    sig, info = ss.read_psf_ascii(text)
    t = np.asarray(sig["time"], float)
    p = sch.twin_params(meta)

    def get(name):
        for k, v in sig.items():
            if ss._norm(k) == ss._norm(name):
                return np.asarray(v, float)
        raise SystemExit(f"signal {name} not in the Spectre data")
    G = np.stack([1.0 / get(f"n_rm{k}") for k in range(n)])
    X = np.stack([get(f"X{k}.n_x") for k in range(n)])
    I = np.stack([get(f"X{k}:IGsc") for k in range(n)])
    Gnv = np.array([[p.G_of_x(float(v)) for v in row] for row in X])
    if t[-1] < meta["t_end_s"] - 1e-9:
        raise SystemExit(f"transient stops at {t[-1]:g} s")
    # gate charge per synapse vs the schedule
    q_ref = np.zeros(n)
    for k in range(n):
        q_ref[k] = sum(abs(a) * w * sch.TICK for _, w, a in per[k])
    q_sp = np.array([compare.pwl_charge(t, I[k]) for k in range(n)])
    sign_ok = []
    for k in range(n):
        wf = sch.waveform(per[k])
        mid = 0.5 * (t[1:] + t[:-1])
        ideal = np.array([wf.current(float(x)) for x in mid])
        sign_ok.append(bool(np.dot(ideal, 0.5 * (I[k][1:] + I[k][:-1])) > 0))
    wall = ss.wall_from_log(Path(log).read_text(encoding="utf-8", errors="replace")) \
        if log else None
    steps = None
    if log:
        import re
        m = re.search(r"Number of accepted tran steps\s*=\s*(\d+)",
                      Path(log).read_text(encoding="utf-8", errors="replace"))
        steps = int(m.group(1)) if m else None
    FULL.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FULL, t=t, G=G, Gnv=Gnv, I=I)
    # committed 1 ms copy
    t1 = np.round(np.arange(0.0, meta["t_end_s"] + 1e-9, 1e-3), 6)
    g1 = np.stack([np.interp(t1, t, G[k]) for k in range(n)])
    n1 = np.stack([np.interp(t1, t, Gnv[k]) for k in range(n)])
    prov = f"# source: {ss.version_tag(info)}; model: v3; c5 tier-1 replay"
    lines = [prov, "t_s," + ",".join(f"G{k}" for k in range(n)) + ","
             + ",".join(f"Gnv{k}" for k in range(n))]
    for j in range(len(t1)):
        lines.append(f"{t1[j]:.6f}," + ",".join(f"{v:.10g}" for v in g1[:, j]) + ","
                     + ",".join(f"{v:.10g}" for v in n1[:, j]))
    _leak_check(lines)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "spectre_1ms.csv").write_text("\n".join(lines) + "\n", encoding="utf-8",
                                          newline="\n")
    sp = {"source": ss.version_tag(info), "wall_s": wall, "accepted_steps": steps,
          "n_points": int(len(t)), "t_stop_s": float(t[-1]),
          "charge_rel_err": ((q_sp - q_ref) / q_ref).tolist(),
          "sign_ok": sign_ok,
          "final_G_full_S": G[:, -1].tolist(), "final_G_nv_S": Gnv[:, -1].tolist(),
          "G_full_at_t_end_S": [float(np.interp(meta["t_end_s"], t, G[k])) for k in range(n)],
          "G_nv_at_t_end_S": [float(np.interp(meta["t_end_s"], t, Gnv[k])) for k in range(n)]}
    (DATA / "spectre.json").write_text(json.dumps(sp, indent=1) + "\n", encoding="utf-8")
    print(f"scrubbed {len(t)} points x {n} synapses; transient {wall} s, "
          f"{steps} steps; charge err max {np.max(np.abs(sp['charge_rel_err'])):.2e}; "
          f"sign ok {all(sign_ok)}")


def _leak_check(lines):
    if not lines[0].startswith("# source: Spectre"):
        raise SystemExit("bad provenance line")
    ids = ss._local_identifiers()
    for k, ln in enumerate(lines[:2]):
        for label, rx in ss.FORBIDDEN:
            if label in ("keyword",) and k == 0:
                continue
            if rx.search(ln):
                raise SystemExit(f"line {k+1} matches a {label} pattern; refusing")
        if any(i in ln.lower() for i in ids):
            raise SystemExit(f"line {k+1} contains a local identifier; refusing")
    num = ss._ROW_OK
    for k, ln in enumerate(lines[2:], start=3):
        if not num.match(ln):
            raise SystemExit(f"line {k} is not purely numeric")


# --------------------------------------------------------------------------
def load_spectre(n):
    if FULL.is_file():
        z = np.load(FULL, allow_pickle=False)
        return z["t"], z["G"], z["Gnv"], "full"
    a = np.loadtxt(DATA / "spectre_1ms.csv", delimiter=",", skiprows=2)
    return a[:, 0], a[:, 1:1 + n].T, a[:, 1 + n:1 + 2 * n].T, "1ms"


def score(figures=True):
    meta, per = sch.load()
    n = len(per)
    sp = json.loads((DATA / "spectre.json").read_text(encoding="utf-8"))
    t_s, G_s, Gnv_s, res = load_spectre(n)
    rd = np.loadtxt(DATA / "engine_reads.csv", delimiter=",", skiprows=1)
    t_r = rd[:, 0]
    t_end = meta["t_end_s"]
    rows, curves = [], []
    for k in range(n):
        ts, G, Gnv, _ = sch.replay(meta, per[k], t_end)
        tg = compare.common_grid(ts, t_s)
        gs = np.interp(tg, t_s, G_s[k])
        ns = np.interp(tg, t_s, Gnv_s[k])
        m_full = compare.metrics(np.interp(tg, ts, G), gs)
        m_nv = compare.metrics(np.interp(tg, ts, Gnv), ns)
        rng = m_full["range"]
        # engine reads vs Spectre at the read times
        inp = np.zeros(len(t_r), bool)
        for s, w, _a in per[k]:
            inp |= (t_r > s * sch.TICK - 1e-9) & (t_r < (s + w) * sch.TICK - 1e-9)
        e_full = rd[:, 1 + k] - np.interp(t_r, t_s, G_s[k])
        e_nv = rd[:, 1 + n + k] - np.interp(t_r, t_s, Gnv_s[k])

        def em(e, mask=None):
            e = e if mask is None else e[mask]
            return 100 * math.sqrt(float(np.mean(e * e))) / rng, 100 * float(np.max(np.abs(e))) / rng
        eng_all, eng_out, eng_nv = em(e_full), em(e_full, ~inp), em(e_nv, ~inp)
        eng_nv_all = em(e_nv)
        g_sp_end = sp["G_full_at_t_end_S"][k]
        gnv_sp_end = sp["G_nv_at_t_end_S"][k]
        row = {
            "synapse": k, "pattern": k < meta["masquelier"]["n_pattern"],
            "n_pulses": len(per[k]), "g_range_uS": 1e6 * rng,
            "twin_nrmse_pct": 100 * m_full["nrmse"], "twin_max_pct": 100 * m_full["maxerr"],
            "twin_nv_nrmse_pct": 100 * m_nv["nrmse"], "twin_nv_max_pct": 100 * m_nv["maxerr"],
            "engine_nrmse_pct": eng_all[0], "engine_max_pct": eng_all[1],
            "engine_out_nrmse_pct": eng_out[0], "engine_out_max_pct": eng_out[1],
            "engine_nv_out_max_pct": eng_nv[1],
            "engine_nv_nrmse_pct": eng_nv_all[0], "engine_nv_max_pct": eng_nv_all[1],
            "final_G_spectre_uS": 1e6 * g_sp_end,
            "final_G_twin_err_pct": 100 * (G[-1] - g_sp_end) / rng,
            "final_G_engine_err_pct": 100 * (meta["final_G_full_S"][k] - g_sp_end) / rng,
            "final_Gnv_spectre_uS": 1e6 * gnv_sp_end,
            "final_Gnv_engine_err_pct": 100 * (meta["final_G_nv_S"][k] - gnv_sp_end) / rng,
            "charge_err_pct": 100 * sp["charge_rel_err"][k],
            "twin_verdict": ("PASS" if (m_full["nrmse"] <= NRMSE_LIMIT and m_full["maxerr"]
                                        <= MAXERR_LIMIT and sp["sign_ok"][k]
                                        and abs(sp["charge_rel_err"][k]) <= compare.CHARGE_TOL)
                             else "FAIL"),
            "engine_verdict": ("PASS" if eng_all[0] <= 100 * NRMSE_LIMIT
                               and eng_all[1] <= 100 * MAXERR_LIMIT else "FAIL"),
            "engine_nv_verdict": ("PASS" if eng_nv_all[0] <= 100 * NRMSE_LIMIT
                                  and eng_nv_all[1] <= 100 * MAXERR_LIMIT else "FAIL")}
        rows.append(row)
        curves.append((tg, gs, np.interp(tg, ts, G), ns))
    with open(DATA / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    cost = cost_table(meta, sp)
    (DATA / "cost.json").write_text(json.dumps(cost, indent=1) + "\n", encoding="utf-8")
    hdr = (f"{'syn':>3s} {'pat':>3s} {'pulses':>6s} {'range uS':>8s} | {'twin NRMSE%':>11s} "
           f"{'max%':>6s} | {'engine NRMSE%':>13s} {'max%':>6s} {'out-pulse max%':>14s} | "
           f"{'final G err% twin':>17s} {'engine':>7s} | verdict twin/engine")
    print(f"Spectre data: {res} resolution")
    print(hdr)
    for r in rows:
        print(f"{r['synapse']:3d} {'y' if r['pattern'] else 'n':>3s} {r['n_pulses']:6d} "
              f"{r['g_range_uS']:8.1f} | {r['twin_nrmse_pct']:11.4f} {r['twin_max_pct']:6.3f} | "
              f"{r['engine_nrmse_pct']:13.3f} {r['engine_max_pct']:6.2f} "
              f"{r['engine_out_max_pct']:14.2f} | {r['final_G_twin_err_pct']:17.4f} "
              f"{r['final_G_engine_err_pct']:7.3f} | {r['twin_verdict']}/{r['engine_verdict']}"
              f"  | G_nv engine {r['engine_nv_nrmse_pct']:.3f}/{r['engine_nv_max_pct']:.2f}% "
              f"out-pulse max {r['engine_nv_out_max_pct']:.2e}% final "
              f"{r['final_Gnv_engine_err_pct']:.2e}% [{r['engine_nv_verdict']}]")
    print(json.dumps(cost, indent=1))
    if figures:
        figure(meta, per, rows, curves, t_r, rd, n)
    return rows


def cost_table(meta, sp):
    n = meta["masquelier"]["n_afferents"]
    wall = sp["wall_s"]
    syn_s = n * meta["t_end_s"]
    return {"spectre_tran_wall_s": wall, "accepted_steps": sp["accepted_steps"],
            "synapses": n, "stream_s": meta["t_end_s"], "pulses": meta["n_pulses"],
            "s_per_pulse": wall / meta["n_pulses"] if wall else None,
            "s_per_synapse_second": wall / syn_s if wall else None,
            "steps_per_pulse": (sp["accepted_steps"] / meta["n_pulses"]
                                if sp["accepted_steps"] else None),
            "engine_wall_s": meta["engine_wall_s"]}


def figure(meta, per, rows, curves, t_r, rd, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    cols = 4
    nr = math.ceil(n / cols)
    fig, axs = plt.subplots(nr, cols, figsize=(3.2 * cols, 2.2 * nr), sharex=True,
                            squeeze=False)
    for k in range(n):
        ax = axs[k // cols][k % cols]
        tg, gs, gt, ns = curves[k]
        ax.plot(tg, 1e6 * gs, color="0.2", lw=1.6, label="Spectre G")
        ax.plot(tg, 1e6 * gt, color="tab:blue", lw=0.7, label="twin G")
        ax.plot(tg, 1e6 * ns, color="tab:orange", lw=0.7, label="Spectre G_nv")
        ax.plot(t_r[::20], 1e6 * rd[::20, 1 + k], ".", ms=1.5, color="tab:red",
                label="engine read")
        r = rows[k]
        ax.set_title(f"syn {k}{' (pat)' if r['pattern'] else ''}: engine "
                     f"{r['engine_nrmse_pct']:.1f}/{r['engine_max_pct']:.0f}% "
                     f"[{r['engine_verdict']}], twin {r['twin_nrmse_pct']:.2f}/"
                     f"{r['twin_max_pct']:.1f}% [{r['twin_verdict']}]", fontsize=6)
        ax.tick_params(labelsize=6)
    axs[0][0].legend(fontsize=5, loc="best")
    for ax in axs[-1]:
        ax.set_xlabel("t (s)", fontsize=7)
    for row in axs:
        row[0].set_ylabel("G (uS)", fontsize=7)
    fig.suptitle(f"C5 tier 1: {n} v3 synapses, {meta['n_pulses']} engine write pulses "
                 "replayed in Spectre (titles: NRMSE / max, % of the synapse's G range;\n"
                 "engine = what the network read during training, twin = fine-grid replay)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "tier1_synapses.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["scrub", "score"])
    ap.add_argument("src", nargs="?")
    ap.add_argument("log", nargs="?")
    a = ap.parse_args(argv)
    if a.cmd == "scrub":
        scrub(a.src, a.log)
        return 0
    rows = score()
    # the primary C5 verdict is the ENGINE's read (what the network saw);
    # the fine-grid twin replay is the model-equivalence diagnostic
    return 0 if all(r["engine_verdict"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
