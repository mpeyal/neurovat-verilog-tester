#!/usr/bin/env python
"""NeuroVAT stream engine vs Brian2 on ideal synapses (paper claim C4).

Same network (``neurovat.masquelier.run``, n_out = 1, the 2008 paper's
constants unless stated), same parameters, same input spike times. See
:mod:`validation.engine.brian2_model` for the model and the mapping.

Tests
  a  fixed weights: membrane potential (per-ms max, NeuroVAT's record_u
     reduction) and output spike times
  b  STDP pair protocol: dw vs dt, both engines and the analytic window
  c  full Masquelier stream learning, several seeds: spike times, final
     weights, detection outcome; first divergence
  d  dt convergence of both engines with NeuroVAT's native (off-grid)
     input times

In a-c the input is put on the 0.1 ms grid (and same-bin duplicates of an
afferent dropped) for BOTH engines, so they see identical spikes; in d each
engine takes the raw stream its own way (NeuroVAT exact times, Brian2 binned).

    python validation/engine/brian2_compare.py [--tests abcd] [--seeds 1 2 3]
"""
import argparse
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.engine import common as C  # noqa: E402
from validation.engine import brian2_model as B  # noqa: E402
from neurovat import masquelier as M  # noqa: E402

import brian2  # noqa: E402

DT = 0.1


def nv_run(P, S, **kw):
    t0 = time.perf_counter()
    r = M.run(P, stream=S, **kw)
    r["wall_s"] = time.perf_counter() - t0
    r["weights"] = np.asarray(r["weights"], float).reshape(-1)
    return r


class _Syn:
    def __init__(self, w, writes=0):
        self.w, self.writes = np.asarray(w, float).reshape(1, -1), writes


def score_brian(P, S, b):
    """NeuroVAT's own success test applied to a Brian2 run."""
    post_t = np.asarray(b["post_t_ms"], float)
    lab, lat = [], []
    for t in post_t:
        sec = min(int(t // P.pattern_ms), S.label.size - 1)
        L = int(S.label[sec])
        lab.append(L)
        lat.append(t - sec * P.pattern_ms if L >= 0 else np.nan)
    t_done = S.duration_ms
    return M._score(P, S, post_t, np.zeros(post_t.size, int),
                    np.array(lab, int), np.array(lat, float),
                    _Syn(b["weights"]), t_done)


def outcome(res):
    return {"success": bool(res["success"]), "hit_rate": res["hit_rate_last"],
            "false_alarms": res["false_alarms_last"],
            "latency_ms": res["mean_latency_last_ms"],
            "potentiated": res["potentiated"],
            "potentiated_in_pattern": res["potentiated_in_pattern"]}


# --------------------------------------------------------------------------
def test_a(seed=11, seconds=5.0):
    """Fixed random weights, paper neuron, 2000 afferents."""
    P = M.MasquelierParams(base_s=seconds, n_repeats=1, seed=seed)
    S = M.make_stream(P, np.random.default_rng(seed))
    G, moved, dropped = C.grid_stream(S, DT)
    w = np.random.default_rng(seed + 100).uniform(0.0, 1.0, P.n_afferents)
    r = nv_run(P, G, learn=False, weights=w, record_u=True)
    b = B.run(P, G, weights=w, learn=False, record_u=True)
    um = B.per_ms_max(b["u"], b["ready"], DT)
    nu = r["u_ms"][:, 0].astype(float)
    n = min(um.size, nu.size)
    sp = C.match_spikes(r["post_t_ms"], b["post_t_ms"], DT * 1.0001)
    row = {"test": "a_fixed_weights", "seed": seed, "seconds": seconds,
           "n_afferents": P.n_afferents, "input_spikes": int(G.t.size),
           "dropped_same_bin": dropped, **{f"spk_{k}": v for k, v in sp.items()},
           "first_divergence_ms": C.first_divergence(r["post_t_ms"], b["post_t_ms"], 1e-6),
           "membrane_nrmse": C.nrmse(um[:n], nu[:n]),
           "membrane_max_abs": float(np.nanmax(np.abs(um[:n] - nu[:n]))),
           "nan_mask_mismatch": int((np.isnan(um[:n]) != np.isnan(nu[:n])).sum()),
           "nv_wall_s": r["wall_s"], "b2_wall_s": b["wall_s"]}
    trace = {"t_ms": np.arange(n, dtype=float), "nv_u": nu[:n], "b2_u": um[:n],
             "nv_spk": r["post_t_ms"], "b2_spk": b["post_t_ms"]}
    return row, trace


# --------------------------------------------------------------------------
PAIR_DTS = (-250., -150., -100., -60., -40., -25., -15., -8., -4., -2., -1.,
            1., 2., 4., 8., 15., 25., 40., 60., 100., 150.)


def test_b():
    """Pair protocol. 8 driver afferents (w = 1) fire together every 600 ms and
    make one output spike; test afferent k fires ONCE, dt_k from that
    output spike, so its final weight minus w0 is one pairing's dw."""
    n_drv, w0, period = 8, 0.5, 600.0
    n_test = len(PAIR_DTS)
    P = M.MasquelierParams(n_afferents=n_drv + n_test, n_pattern=0,
                           threshold=4.0, n_repeats=1,
                           base_s=(n_test + 1) * period / 1000.0)
    t_drv = 300.0 + period * np.arange(n_test)
    weights = np.r_[np.ones(n_drv), np.full(n_test, w0)]
    k_drv = np.round(t_drv / DT).astype(np.int64)
    aff_d = np.repeat(np.arange(n_drv), n_test)
    t_d = C.on_grid(np.tile(k_drv, n_drv), DT)
    probe = nv_run(P, C.make_raw_stream(aff_d, t_d, P.base_s * 1000.0),
                   weights=weights, learn=False)
    lat = float(np.median(probe["post_t_ms"] - t_drv[:len(probe["post_t_ms"])]))
    k_pre = np.round((t_drv + lat + np.array(PAIR_DTS) * -1.0) / DT).astype(np.int64)
    # dt = t_post - t_pre (> 0: pre before post)
    aff = np.r_[aff_d, n_drv + np.arange(n_test)]
    t = np.r_[t_d, C.on_grid(k_pre, DT)]
    S = C.make_raw_stream(aff, t, P.base_s * 1000.0)
    r = nv_run(P, S, weights=weights, learn=True)
    b = B.run(P, S, weights=weights, learn=True)
    post = np.asarray(r["post_t_ms"])
    rows = []
    a_minus = P.a_minus_ratio * P.a_plus
    for k, nominal in enumerate(PAIR_DTS):
        tpre = t[n_drv * n_test + k]
        # nearest output spike (the one of this episode)
        tp = post[np.argmin(np.abs(post - tpre))]
        d = tp - tpre
        if d > 0:
            ana = P.a_plus * math.exp(-d / P.tau_plus) if d <= P.cutoff_tau * P.tau_plus else 0.0
        else:
            ana = -a_minus * math.exp(d / P.tau_minus) if -d <= P.cutoff_tau * P.tau_minus else 0.0
        rows.append({"test": "b_pair", "dt_nominal_ms": nominal, "dt_ms": d,
                     "dw_neurovat": r["weights"][n_drv + k] - w0,
                     "dw_brian2": b["weights"][n_drv + k] - w0,
                     "dw_analytic": ana})
    dn = np.array([x["dw_neurovat"] for x in rows])
    db = np.array([x["dw_brian2"] for x in rows])
    da = np.array([x["dw_analytic"] for x in rows])
    summ = {"test": "b_pair", "pairs": n_test,
            "nv_post": int(post.size), "b2_post": int(len(b["post_t_ms"])),
            "post_match": C.match_spikes(post, b["post_t_ms"], DT * 1.0001)["match_frac"],
            "max_abs_dw_nv_b2": float(np.abs(dn - db).max()),
            "max_abs_dw_nv_analytic": float(np.abs(dn - da).max()),
            "driver_latency_ms": lat}
    return summ, rows


# --------------------------------------------------------------------------
def test_c(seeds=(1, 2, 3), seconds=30.0, native=True):
    """Full stream learning from w_init = 0.475, paper constants."""
    rows = []
    for seed in seeds:
        P = M.MasquelierParams(base_s=seconds, n_repeats=1, seed=seed)
        S = M.make_stream(P, np.random.default_rng(seed))
        G, moved, dropped = C.grid_stream(S, DT)
        r = nv_run(P, G, learn=True)
        b = B.run(P, G, learn=True)
        rb = score_brian(P, G, b)
        sp = C.match_spikes(r["post_t_ms"], b["post_t_ms"], DT * 1.0001)
        on, ob = outcome(r), outcome(rb)
        row = {"test": "c_stream", "seed": seed, "seconds": seconds,
               "input_spikes": int(G.t.size), "dropped_same_bin": dropped,
               "mean_shift_ms": moved / max(G.t.size, 1),
               **{f"spk_{k}": v for k, v in sp.items()},
               "first_divergence_ms": C.first_divergence(r["post_t_ms"], b["post_t_ms"], 1e-6),
               "weight_nrmse": C.nrmse(b["weights"], r["weights"], span=1.0),
               "weight_max_abs": float(np.abs(b["weights"] - r["weights"]).max()),
               **{f"nv_{k}": v for k, v in on.items()},
               **{f"b2_{k}": v for k, v in ob.items()},
               "outcome_agree": on["success"] == ob["success"],
               "nv_wall_s": r["wall_s"], "b2_wall_s": b["wall_s"]}
        if native:
            # the same seed with NeuroVAT's own exact (off-grid) input: how far
            # the 0.1 ms grid alone moves a chaotic learning run
            rn = nv_run(P, S, learn=True)
            spn = C.match_spikes(rn["post_t_ms"], r["post_t_ms"], DT * 1.0001)
            row.update({"native_vs_grid_spk_match_frac": spn["match_frac"],
                        "native_vs_grid_first_divergence_ms":
                            C.first_divergence(rn["post_t_ms"], r["post_t_ms"], DT * 1.0001),
                        "native_vs_grid_weight_nrmse":
                            C.nrmse(rn["weights"], r["weights"], span=1.0),
                        "native_success": bool(rn["success"]),
                        "native_hit_rate": rn["hit_rate_last"],
                        "native_false_alarms": rn["false_alarms_last"]})
        rows.append(row)
        print("  c seed", seed, {k: row[k] for k in (
            "spk_match_frac", "first_divergence_ms", "weight_nrmse",
            "nv_success", "b2_success", "nv_hit_rate", "b2_hit_rate",
            "nv_wall_s", "b2_wall_s")}, flush=True)
    return rows


# --------------------------------------------------------------------------
DTS = (0.4, 0.2, 0.1, 0.05, 0.025)
DT_REF = 0.0125


def test_d(seed=21, seconds=3.0, n_aff=500):
    """dt convergence, fixed weights, native input. Reference: NeuroVAT at
    0.0125 ms (exact input times; only the threshold sampling remains)."""
    P0 = M.MasquelierParams(n_afferents=n_aff, n_pattern=n_aff // 2,
                            threshold=125.0, base_s=seconds, n_repeats=1,
                            seed=seed)
    S = M.make_stream(P0, np.random.default_rng(seed))
    w = np.random.default_rng(seed + 1).uniform(0.0, 1.0, n_aff)
    ref = nv_run(replace(P0, sim_dt_ms=DT_REF), S, learn=False, weights=w)
    tref = ref["post_t_ms"]
    rows = []
    for dt in DTS:
        P = replace(P0, sim_dt_ms=dt)
        r = nv_run(P, S, learn=False, weights=w)
        b = B.run(P, S, weights=w, learn=False, dt_ms=dt)
        for eng, t in (("neurovat", r["post_t_ms"]), ("brian2", b["post_t_ms"])):
            m = C.match_spikes(tref, t, 1.0)
            rows.append({"test": "d_dt", "engine": eng, "dt_ms": dt,
                         "n_ref": m["n_ref"], "n": m["n_test"],
                         "match_frac_1ms": m["match_frac"],
                         "mean_abs_dt_ms": m["mean_abs_dt_ms"],
                         "max_abs_dt_ms": m["max_abs_dt_ms"],
                         "dropped_same_bin": b["n_dropped"] if eng == "brian2" else 0})
        m2 = C.match_spikes(r["post_t_ms"], b["post_t_ms"], dt * 1.0001)
        rows.append({"test": "d_dt", "engine": "nv_vs_b2", "dt_ms": dt,
                     "n_ref": m2["n_ref"], "n": m2["n_test"],
                     "match_frac_1ms": m2["match_frac"],
                     "mean_abs_dt_ms": m2["mean_abs_dt_ms"],
                     "max_abs_dt_ms": m2["max_abs_dt_ms"]})
        print("  d dt", dt, rows[-3:], flush=True)
    return rows


def test_d_learning(seed=22, seconds=3.0, n_aff=500):
    """dt convergence WITH learning (final weights vs the 0.0125 ms NeuroVAT
    reference), native input."""
    P0 = M.MasquelierParams(n_afferents=n_aff, n_pattern=n_aff // 2,
                            threshold=125.0, base_s=seconds, n_repeats=1,
                            seed=seed)
    S = M.make_stream(P0, np.random.default_rng(seed))
    ref = nv_run(replace(P0, sim_dt_ms=DT_REF), S, learn=True)
    rows = []
    for dt in DTS:
        P = replace(P0, sim_dt_ms=dt)
        r = nv_run(P, S, learn=True)
        b = B.run(P, S, learn=True, dt_ms=dt)
        for eng, x in (("neurovat", r), ("brian2", b)):
            m = C.match_spikes(ref["post_t_ms"], x["post_t_ms"], 1.0)
            rows.append({"test": "d_dt_learn", "engine": eng, "dt_ms": dt,
                         "n_ref": m["n_ref"], "n": m["n_test"],
                         "match_frac_1ms": m["match_frac"],
                         "mean_abs_dt_ms": m["mean_abs_dt_ms"],
                         "first_divergence_ms": C.first_divergence(
                             ref["post_t_ms"], x["post_t_ms"], 1.0),
                         "weight_nrmse": C.nrmse(x["weights"], ref["weights"], span=1.0)})
        print("  dL dt", dt, rows[-2:], flush=True)
    return rows


# --------------------------------------------------------------------------
def figures(trace_a, rows_b, rows_d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    if trace_a is not None:
        fig, ax = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
        sl = slice(0, 1000)
        ax[0].plot(trace_a["t_ms"][sl], trace_a["nv_u"][sl], lw=1.6, label="NeuroVAT")
        ax[0].plot(trace_a["t_ms"][sl], trace_a["b2_u"][sl], "--", lw=1, label="Brian2")
        ax[0].set_ylabel("u (per-ms max)")
        ax[0].legend(frameon=False, fontsize=8)
        ax[1].plot(trace_a["t_ms"][sl], trace_a["b2_u"][sl] - trace_a["nv_u"][sl], lw=0.8)
        ax[1].set_ylabel("Brian2 - NeuroVAT")
        ax[1].set_xlabel("t (ms)")
        fig.tight_layout()
        fig.savefig(C.FIGURES / "a_membrane.png", dpi=150)
        plt.close(fig)
    if rows_b:
        d = np.array([r["dt_ms"] for r in rows_b])
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.plot(d, [r["dw_analytic"] for r in rows_b], "k-", lw=0.8, label="analytic")
        ax.plot(d, [r["dw_neurovat"] for r in rows_b], "o", ms=6, mfc="none", label="NeuroVAT")
        ax.plot(d, [r["dw_brian2"] for r in rows_b], "x", ms=5, label="Brian2")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_xlabel("t_post - t_pre (ms)")
        ax.set_ylabel("dw")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(C.FIGURES / "b_stdp_window.png", dpi=150)
        plt.close(fig)
    if rows_d:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for eng, mk in (("neurovat", "o-"), ("brian2", "s--")):
            rr = [r for r in rows_d if r["engine"] == eng and r["test"] == "d_dt"]
            ax.loglog([r["dt_ms"] for r in rr],
                      [max(r["mean_abs_dt_ms"], 1e-6) for r in rr], mk, label=eng)
        ax.set_xlabel("dt (ms)")
        ax.set_ylabel(f"mean |spike time error| vs NeuroVAT @ {DT_REF} ms (ms)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(C.FIGURES / "d_dt_convergence.png", dpi=150)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", default="abcd")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args(argv)
    C.DATA.mkdir(parents=True, exist_ok=True)
    meta = {"brian2": brian2.__version__, "numpy": np.__version__,
            "codegen": "numpy", "params": asdict(M.MasquelierParams())}
    trace_a = rows_b = rows_d = None
    if "a" in args.tests:
        row, trace_a = test_a()
        print("a", row, flush=True)
        C.write_csv(C.DATA / "brian2_a_fixed_weights.csv", [row])
    if "b" in args.tests:
        summ, rows_b = test_b()
        print("b", summ, flush=True)
        C.write_csv(C.DATA / "brian2_b_pair_window.csv", rows_b)
        C.write_json(C.DATA / "brian2_b_summary.json", summ)
    if "c" in args.tests:
        rows = test_c(tuple(args.seeds), args.seconds)
        C.write_csv(C.DATA / "brian2_c_stream.csv", rows)
    if "d" in args.tests:
        rows_d = test_d() + test_d_learning()
        C.write_csv(C.DATA / "brian2_d_dt.csv", rows_d)
    C.write_json(C.DATA / "brian2_meta.json", meta)
    figures(trace_a, rows_b, rows_d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
