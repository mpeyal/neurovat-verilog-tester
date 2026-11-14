#!/usr/bin/env python
"""NeuroVAT LIF vs snnTorch (paper claim C4, the LIF-dynamics half).

1. Single neuron. ``neurovat.lif.probe_lif_wave`` (the Trainer's output
   neuron: forward Euler ``v += dt/tau (v_rest - v + I)``, reset to v_reset,
   refractory, adaptive threshold theta) against ``snn.Leaky`` with
   reset_mechanism="none" and NeuroVAT's reset / refractory / theta written
   around it, in two mappings:
     * Euler map:  beta = 1 - dt/tau,  input = dt/tau (v_rest + I)
       -> the same recurrence; must agree to rounding.
     * exact map:  beta = exp(-dt/tau), input = (1 - beta)(v_rest + I)
       -> the exact solution for an input held over each step (what the
       surrogate-gradient mode uses). The residual against NeuroVAT's STDP
       mode is its Euler truncation error, O(dt/tau) per step; shown vs dt.
2. Networks. ``tools/validate_snntorch.compare`` (independent snnTorch
   forward / autograd and Torch eligibility rule, shared device programming)
   at larger sizes than the original 2x2 -> 2 check.

    python validation/engine/snntorch_compare.py [--quick]
"""
import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from validation.engine import common as C  # noqa: E402
from neurovat.config import NeuronParams  # noqa: E402
from neurovat.lif import probe_lif_wave  # noqa: E402

import torch  # noqa: E402
import snntorch as snn  # noqa: E402

torch.set_default_dtype(torch.float64)


def drive_wave(dt, T=400.0):
    """Step 1.6 (0-100 ms), off, a 0 -> 3 ramp (150-300 ms), then 1.2."""
    t = np.arange(int(round(T / dt))) * dt
    d = np.zeros_like(t)
    d[t < 100] = 1.6
    r = (t >= 150) & (t < 300)
    d[r] = 3.0 * (t[r] - 150) / 150
    d[t >= 300] = 1.2
    return t, d


def snn_neuron(N, drive, dt, exact):
    """snn.Leaky + NeuroVAT's reset / refractory / adaptive threshold."""
    snn.Leaky.instances.clear()
    a = dt / N.tau_m
    beta = math.exp(-a) if exact else 1.0 - a
    gain = (1.0 - beta) if exact else a
    cell = snn.Leaky(beta=torch.tensor(beta), threshold=torch.tensor(N.v_threshold),
                     reset_mechanism="none")
    d_theta = math.exp(-dt / N.tau_theta) if N.tau_theta > 0 else 0.0
    v = torch.tensor([N.v_rest])
    theta, refr = 0.0, 0.0
    V, spikes = np.empty(len(drive)), []
    with torch.no_grad():
        for k, I in enumerate(drive):
            active = refr <= 0
            if N.tau_theta > 0:
                theta *= d_theta
            cell.threshold = torch.tensor(N.v_threshold + theta)
            if active:
                spk, v = cell(torch.tensor([gain * (N.v_rest + I)]), v)
            else:
                spk = torch.zeros(1)
                v = torch.tensor([N.v_reset])
                refr -= dt
            V[k] = float(v)
            if active and float(spk) > 0:
                spikes.append(k * dt)
                v = torch.tensor([N.v_reset])
                refr = N.t_refractory
                theta += N.theta_plus
    return V, np.array(spikes)


def single_neuron(dts=(1.0, 0.5, 0.25, 0.1, 0.05)):
    N = NeuronParams()
    rows, traces = [], {}
    # reference for the time-discretisation error: exact map at the finest dt
    t_f, d_f = drive_wave(dts[-1] / 4)
    _, s_fine = snn_neuron(N, d_f, dts[-1] / 4, exact=True)
    for dt in dts:
        t, d = drive_wave(dt)
        t0 = time.perf_counter()
        _, Vn, _, sn = probe_lif_wave(N, d, dt)
        w_nv = time.perf_counter() - t0
        sn = np.array(sn)
        for name, exact in (("euler_map", False), ("exact_map", True)):
            t0 = time.perf_counter()
            Vs, ss = snn_neuron(N, d, dt, exact)
            w_s = time.perf_counter() - t0
            m = C.match_spikes(sn, ss, dt * 1.0001)
            mf = C.match_spikes(s_fine, sn, 1.0)
            ms_ = C.match_spikes(s_fine, ss, 1.0)
            both = (np.isfinite(Vn) & np.isfinite(Vs))
            # subthreshold (no spike, no reset): the pure leak recurrence,
            # where a one-step spike shift cannot dominate the difference
            Nsub = replace(N, v_threshold=1e9)
            _, Vn_sub, _, _ = probe_lif_wave(Nsub, d, dt)
            Vs_sub, _ = snn_neuron(Nsub, d, dt, exact)
            rows.append({"test": "single_neuron", "mapping": name, "dt_ms": dt,
                         "beta": math.exp(-dt / N.tau_m) if exact else 1 - dt / N.tau_m,
                         "nv_spikes": int(sn.size), "snn_spikes": int(ss.size),
                         "spike_match_frac_1dt": m["match_frac"],
                         "mean_abs_dt_ms": m["mean_abs_dt_ms"],
                         "membrane_max_abs": float(np.abs(Vn - Vs)[both].max()),
                         "membrane_nrmse": C.nrmse(Vs, Vn),
                         "subthreshold_max_abs": float(np.abs(Vn_sub - Vs_sub).max()),
                         "subthreshold_rel": float(np.abs(Vn_sub - Vs_sub).max()
                                                   / np.abs(Vs_sub).max()),
                         "nv_vs_fine_exact_mean_abs_dt_ms": mf["mean_abs_dt_ms"],
                         "snn_vs_fine_exact_mean_abs_dt_ms": ms_["mean_abs_dt_ms"],
                         "nv_wall_s": w_nv, "snn_wall_s": w_s})
            if dt == 0.5:
                traces[name] = (t, Vn, Vs)
        print("  single dt", dt, rows[-2:], flush=True)
    return rows, traces


SIZES = (  # (grid, n_out, hidden)
    ((2, 2), 2, ()), ((4, 4), 4, ()), ((8, 8), 10, ()), ((8, 8), 10, (16,)),
    ((16, 16), 10, ()))


def networks(seeds=(1, 7), sizes=SIZES, epochs=8):
    import validate_snntorch as V
    from simulation_core import EcfetV2
    rows = []
    for grid, n_out, hidden in sizes:
        for rule in ("stdp", "surrogate"):
            for seed in seeds:
                t0 = time.perf_counter()
                r = V.compare(EcfetV2, hidden, seed, rule, grid=grid, n_out=n_out,
                              epochs=epochs, n_samples=max(30, 15 * n_out))
                e = r["errors"]
                rows.append({"test": "network", "rule": rule,
                             "layers": "-".join(map(str, r["layers"])),
                             "seed": seed, "epochs": epochs, "passed": r["passed"],
                             "drift_free": r["drift_free"],
                             "spike_mismatches": e["spike_mismatches"],
                             "prediction_mismatches": e["prediction_mismatches"],
                             "membrane_max_abs": e["membrane"],
                             "weight_rel_max": e["weight_rel"],
                             "gradient_max": e["gradient"],
                             "matched_state_gradient": e["matched_state_gradient"],
                             "app_test_acc": r["app_test_accuracy"],
                             "snn_test_acc": r["snntorch_test_accuracy"],
                             "wall_s": time.perf_counter() - t0})
                print("  net", rows[-1], flush=True)
    return rows


def figure(traces, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    t, Vn, Ve = traces["euler_map"]
    _, _, Vx = traces["exact_map"]
    sl = t < 400
    ax[0].plot(t[sl], Vn[sl], lw=1.6, label="NeuroVAT (Euler LIF)")
    ax[0].plot(t[sl], Ve[sl], "--", lw=1, label="snnTorch, beta = 1 - dt/tau")
    ax[0].plot(t[sl], Vx[sl], ":", lw=1, label="snnTorch, beta = exp(-dt/tau)")
    ax[0].set_xlabel("t (ms)")
    ax[0].set_ylabel("v (dt = 0.5 ms)")
    ax[0].legend(frameon=False, fontsize=7)
    for name, mk in (("euler_map", "o-"), ("exact_map", "s--")):
        rr = [r for r in rows if r["mapping"] == name]
        ax[1].loglog([r["dt_ms"] for r in rr],
                     [max(r["subthreshold_max_abs"], 1e-17) for r in rr], mk, label=name)
    ax[1].set_xlabel("dt (ms)")
    ax[1].set_ylabel("subthreshold max |v_snnTorch - v_NeuroVAT|")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "snntorch_single_neuron.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="single neuron + 2 sizes")
    args = ap.parse_args(argv)
    rows, traces = single_neuron()
    C.write_csv(C.DATA / "snntorch_single_neuron.csv", rows)
    figure(traces, rows)
    sizes = SIZES[:2] if args.quick else SIZES
    net = networks(sizes=sizes)
    C.write_csv(C.DATA / "snntorch_networks.csv", net)
    C.write_json(C.DATA / "snntorch_meta.json",
                 {"torch": torch.__version__, "snntorch": snn.__version__,
                  "dtype": "float64 CPU", "device": "EcfetV2 (shared programming)"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
