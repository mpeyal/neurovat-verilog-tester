#!/usr/bin/env python
"""ECFET Verilog-A test bench - transient scenarios with pA gate spikes.

Examples:
  python run_ecfet.py spike                       # single pA spike, v1 vs v2
  python run_ecfet.py potentiate --amp-pA 100 --width-ms 10 --n 20
  python run_ecfet.py depress    --model v1
  python run_ecfet.py ltp-ltd    --n 30           # synaptic curve (v2)
  python run_ecfet.py retention  --hold-s 60
  python run_ecfet.py compare                     # v1 vs v2, same stimulus

Sign convention is model-dependent after the Fig. 3 retune: the paper-matched
v2 .va potentiates (G up / R down) on POSITIVE current; the legacy v1 port is
the opposite.  NOTE: the scenario sign logic in this CLI still follows the v1
convention ('potentiate' = negative current), so v2 curves here are INVERTED
relative to their labels - use run_fig3.py for the paper-faithful v2 panels.
All plots land in results/.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecfet import (Waveform, EcfetV1, V1Params, EcfetV2, V2Params,
                   simulate, plotting)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def make_models(which, v2_kw=None):
    models = []
    if which in ("v1", "both"):
        models.append(EcfetV1(V1Params()))
    if which in ("v2", "both"):
        models.append(EcfetV2(V2Params(**(v2_kw or {}))))
    return models


def run_and_plot(models, wf, t_stop, fname, title, csv=False, **plot_kw):
    results = [simulate(m, wf, t_stop) for m in models]
    for r in results:
        print(" ", r.summary())
        if csv:
            stem = fname.rsplit(".", 1)[0] + "_" + r.label.split()[0] + ".csv"
            print("  csv  ->", r.save_csv(os.path.join(RESULTS, stem)))
    path = os.path.join(RESULTS, fname)
    plotting.plot_transient(results, path, title=title, **plot_kw)
    print(f"  plot -> {path}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", default="compare",
                    choices=["spike", "potentiate", "depress",
                             "ltp-ltd", "retention", "compare"],
                    help="test scenario (default: compare)")
    ap.add_argument("--model", choices=["v1", "v2", "both"], default="both")
    ap.add_argument("--amp-pA", type=float, default=100.0,
                    help="pulse amplitude magnitude in pA (default 100)")
    ap.add_argument("--width-ms", type=float, default=10.0,
                    help="pulse width in ms (default 10)")
    ap.add_argument("--period-ms", type=float, default=50.0,
                    help="pulse period in ms (default 50)")
    ap.add_argument("--n", type=int, default=20, help="pulses per train")
    ap.add_argument("--hold-s", type=float, default=30.0,
                    help="retention observation time (s)")
    ap.add_argument("--kappa-v", type=float, default=0.3,
                    help="v2 volatile fraction of each write")
    ap.add_argument("--nu", type=float, default=2.0,
                    help="v2 nonlinearity (nu_p = nu_d)")
    ap.add_argument("--sigma-c2c", type=float, default=0.0,
                    help="v2 cycle-to-cycle write noise (relative)")
    ap.add_argument("--extras", action="store_true",
                    help="plot internal states panel")
    ap.add_argument("--csv", action="store_true",
                    help="also dump waveforms to CSV (for Spectre comparison)")
    ap.add_argument("--no-show", action="store_true",
                    help="only save PNGs, don't open matplotlib windows")
    args = ap.parse_args()

    if not args.no_show:
        plotting.enable_show()
    os.makedirs(RESULTS, exist_ok=True)
    amp = args.amp_pA * 1e-12
    width = args.width_ms * 1e-3
    period = args.period_ms * 1e-3
    v2_kw = dict(kappa_v=args.kappa_v, nu_p=args.nu, nu_d=args.nu,
                 sigma_c2c=args.sigma_c2c)

    if args.scenario == "spike":
        print(f"Single gate spike: {args.amp_pA:+.3g} pA x {args.width_ms} ms")
        wf = Waveform([(10e-3, width, amp)])
        run_and_plot(make_models(args.model, v2_kw), wf,
                     t_stop=10e-3 + width + 0.5,
                     fname="spike.png",
                     title=f"Single {args.amp_pA:+.0f} pA / {args.width_ms} ms gate spike",
                     show_extras=args.extras, csv=args.csv)

    elif args.scenario in ("potentiate", "depress"):
        sign = -1.0 if args.scenario == "potentiate" else +1.0
        verb = ("conductance UP (R down)" if sign < 0
                else "conductance DOWN (R up)")
        print(f"{args.scenario}: {args.n} x {sign*args.amp_pA:+.3g} pA "
              f"/ {args.width_ms} ms @ {args.period_ms} ms -> {verb}")
        wf = Waveform.pulse_train(sign * amp, width, period, args.n)
        run_and_plot(make_models(args.model, v2_kw), wf,
                     t_stop=10e-3 + args.n * period + 1.0,
                     fname=f"{args.scenario}.png",
                     title=f"{args.scenario}: {args.n} x {sign*args.amp_pA:+.0f} pA"
                           f" / {args.width_ms} ms gate pulses",
                     show_extras=args.extras, csv=args.csv)

    elif args.scenario == "ltp-ltd":
        print(f"LTP/LTD: {args.n}+{args.n} pulses of ±{args.amp_pA} pA")
        wf = Waveform.ltp_ltd(amp, width, period, args.n)
        t_stop = 10e-3 + 2 * args.n * period + 1.0
        models = make_models(args.model, v2_kw)
        results = run_and_plot(models, wf, t_stop,
                               fname="ltp_ltd_transient.png",
                               title=f"LTP/LTD train: ±{args.amp_pA:.0f} pA "
                                     f"/ {args.width_ms} ms",
                               show_extras=args.extras, csv=args.csv)
        pulse_ends = [t1 for _, t1 in wf.pulse_windows()]
        settle = max(1e-4, 0.8 * (period - width))
        for r in results:
            path, info = plotting.plot_ltp_ltd(
                r, pulse_ends, args.n,
                os.path.join(RESULTS, f"ltp_ltd_curve_{r.label.split()[0]}.png"),
                settle=settle,
                title=f"LTP/LTD characteristic - {r.label}")
            print(f"  [{r.label}] G range {info['G_range_uS'][0]:.1f}.."
                  f"{info['G_range_uS'][1]:.1f} uS | "
                  f"mean dG: LTP {info['mean_dG_ltp_uS']:+.2f} uS, "
                  f"LTD {info['mean_dG_ltd_uS']:+.2f} uS")
            print(f"  curve -> {path}")

    elif args.scenario == "retention":
        print(f"Retention: 10 potentiating pulses, then hold {args.hold_s} s")
        wf = Waveform.pulse_train(-amp, width, period, 10)
        run_and_plot(make_models(args.model, v2_kw), wf,
                     t_stop=10e-3 + 10 * period + args.hold_s,
                     fname="retention.png",
                     title=f"Retention after 10 x {-args.amp_pA:.0f} pA pulses",
                     show_extras=args.extras, csv=args.csv)

    elif args.scenario == "compare":
        print("v1 vs v2 under identical stimulus "
              f"({args.n} x +{args.amp_pA} pA then {args.n} x -{args.amp_pA} pA)")
        wf = Waveform.ltp_ltd(amp, width, period, args.n)
        run_and_plot(make_models("both", v2_kw), wf,
                     t_stop=10e-3 + 2 * args.n * period + 2.0,
                     fname="compare.png",
                     title="v1 (Verilog-A port) vs v2 (practical ECFET)",
                     show_extras=args.extras, csv=args.csv)

    plotting.show()


if __name__ == "__main__":
    main()
