"""Command line: validate a Python twin against Spectre.

    python -m vatester.spectre run  --va verilog/vteam.va --twin vteam \\
        --pulses 0.5,-0.25 --width 10e-9 --gap 3e-9 --read-v -0.1 --maxstep 1e-11
    python -m vatester.spectre compare --psf result.raw --va verilog/vteam.va --twin vteam ...
    python -m vatester.spectre import  result.raw/tran.tran.tran
    python -m vatester.spectre import  --skill /path/on/host/tran.tran.tran
    python -m vatester.spectre deck    --va verilog/vteam.va ... > s1.scs

`run` writes the deck, runs Spectre on the Cadence host over SSH (the same
login as the Virtuoso link), imports the PSF-ASCII result, drives the twin with
the same PWL, and reports NRMSE / maximum error under the pass rule.  The
stimulus is one of --pwl "t v t v ...", --pwl-file t,v CSV, or --pulses.
--param name=value goes to both the .va instance and the twin.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from . import compare as C
from .deck import make_deck
from .psf import PSFError, find_signal, load_waveform, parse_waveform
from .stimulus import Stimulus
from .twinrun import load_twin, run_twin

REPO = Path(__file__).resolve().parents[2]


def _module_name(va_path):
    text = Path(va_path).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.M)
    if not m:
        raise SystemExit(f"{va_path}: no 'module <name>' line")
    return m.group(1)


def _params(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not k or not v:
            raise SystemExit(f"--param needs name=value, got {it!r}")
        out[k.strip()] = float(v)
    return out


def _stimulus(a):
    if a.pwl:
        return Stimulus.from_text(a.pwl, kind=a.kind, name=a.name)
    if a.pwl_file:
        return Stimulus.from_file(a.pwl_file, kind=a.kind, name=a.name)
    if a.pulses:
        amps = [float(x) for x in a.pulses.split(",")]
        return Stimulus.pulses(amps, a.width, a.gap, edge=a.edge, kind=a.kind,
                               read=a.read_v, name=a.name)
    raise SystemExit("give a stimulus: --pwl, --pwl-file or --pulses")


def _reference(sig, a):
    t = sig["time"]
    v = find_signal(sig, a.v_signal)
    i = find_signal(sig, a.i_signal)
    if v is None or i is None:
        names = ", ".join(sorted(k for k in sig if k != "time"))
        raise SystemExit(f"signals {a.v_signal!r} / {a.i_signal!r} not in the "
                         f"result; it holds: {names}")
    return {"time": t, "V": v, "I": i}


def compare_twin(sig, info, stim, a, out_dir=None):
    ref = _reference(sig, a)
    spec = load_twin(a.twin, a.workspace)
    params = _params(a.param)
    dt = a.dt_max or a.maxstep
    tw = run_twin(spec, stim, dt, params)
    key = "I" if stim.kind == "voltage" else "V"
    m = C.waveform_metrics(tw["time"], tw[key], ref["time"], ref[key])
    reads = stim.read_times(a.read_v) if a.read_v is not None else []
    r = C.read_metrics(tw, ref, reads) if len(reads) else None
    rule = a.rule or ("reads" if r is not None else "waveform")
    verdict = C.verdict(m, r, rule)
    summary = {
        "twin": spec["key"], "stimulus": stim.name, "kind": stim.kind,
        "simulator": info.get("simulator", ""), "version": info.get("version", ""),
        "reference_points": int(len(ref["time"])), "twin_points": m["n_points"],
        "compared": "current" if key == "I" else "voltage",
        "nrmse_pct": 100 * m["nrmse"], "max_err_pct": 100 * m["maxerr"],
        "range": m["range"], "rule": rule, "verdict": verdict,
    }
    if r is not None:
        summary.update({"n_reads": r["n_reads"], "read_g_max_pct": 100 * r["max"],
                        "read_g_mean_pct": 100 * r["mean"]})
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        yr = np.interp(tw["time"], ref["time"], ref[key])
        rows = ["t_s,twin,spectre"] + [f"{t:.12g},{x:.10g},{y:.10g}"
                                         for t, x, y in zip(tw["time"], tw[key], yr)]
        (out_dir / f"{stim.name}_compare.csv").write_text("\n".join(rows) + "\n",
                                                          encoding="utf-8")
        (out_dir / f"{stim.name}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _print(s):
    print(f"twin {s['twin']} vs {s['simulator']} {s['version']}".rstrip())
    print(f"  stimulus {s['stimulus']} ({s['kind']}); reference {s['reference_points']} "
          f"points, twin {s['twin_points']} points")
    print(f"  {s['compared']}: NRMSE {s['nrmse_pct']:.4g} %, max {s['max_err_pct']:.4g} % "
          f"of the range")
    if "n_reads" in s:
        print(f"  read G ({s['n_reads']} reads): max {s['read_g_max_pct']:.4g} %, "
              f"mean {s['read_g_mean_pct']:.4g} %")
    print(f"  rule {s['rule']}: {s['verdict']}")


def _add_common(p):
    p.add_argument("--va", required=True, help="Verilog-A file")
    p.add_argument("--module", help="module name (default: read from the .va)")
    p.add_argument("--twin", help="twin key (TWIN_SPEC['key'] in twins/)")
    p.add_argument("--workspace", default=str(REPO), help="folder holding twins/")
    p.add_argument("--kind", choices=("voltage", "current"), default="voltage")
    p.add_argument("--name", default="stim", help="run / stimulus name")
    p.add_argument("--pwl", help='"t v t v ..." points')
    p.add_argument("--pwl-file", help="two-column t,value file")
    p.add_argument("--pulses", help="comma-separated pulse amplitudes")
    p.add_argument("--width", type=float, default=10e-9)
    p.add_argument("--gap", type=float, default=3e-9)
    p.add_argument("--edge", type=float, default=50e-12)
    p.add_argument("--read-v", type=float, help="read level: a read plateau after "
                   "each pulse (--pulses) and per-read G at plateaus of this level")
    p.add_argument("--param", action="append", help="name=value (twin and instance)")
    p.add_argument("--maxstep", type=float, default=1e-11, help="Spectre maxstep (s)")
    p.add_argument("--dt-max", type=float, help="twin output step (default maxstep)")
    p.add_argument("--rule", choices=("waveform", "reads"))
    p.add_argument("--v-signal", default="n_p")
    p.add_argument("--i-signal", default="X1:p")
    p.add_argument("--out", help="folder for the comparison CSV and summary JSON")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vatester.spectre", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="deck -> Spectre (SSH) -> PSF import -> twin compare")
    _add_common(p_run)
    p_run.add_argument("--deck", help="use this hand-written deck instead of a generated one")
    p_run.add_argument("--keep", action="store_true", help="keep the remote run folder")
    p_run.add_argument("--timeout", type=int, default=3600)
    p_cmp = sub.add_parser("compare", help="compare a local PSF / CSV result with the twin")
    _add_common(p_cmp)
    p_cmp.add_argument("--psf", required=True, help="PSF-ASCII file, .raw folder or CSV")
    p_deck = sub.add_parser("deck", help="print the generated Spectre deck")
    _add_common(p_deck)
    p_imp = sub.add_parser("import", help="list the signals of a PSF-ASCII / CSV result")
    p_imp.add_argument("path", nargs="?")
    p_imp.add_argument("--skill", help="read this file on the host through skillbridge")
    a = ap.parse_args(argv)

    if a.cmd == "import":
        try:
            if a.skill:
                from ..virtuoso import VirtuosoLink
                link = VirtuosoLink()
                try:                       # disconnect also stops a tunnel we started
                    link.connect()
                    sig, info = parse_waveform(link.read_file(a.skill))
                finally:
                    link.disconnect()
            elif a.path:
                sig, info = load_waveform(a.path)
            else:
                raise SystemExit("give a path or --skill <remote path>")
        except PSFError as e:
            raise SystemExit(f"import failed: {e}")
        t = sig["time"]
        print(f"{info.get('simulator', '')} {info.get('version', '')}".strip() or "(no header)")
        print(f"{len(t)} time points, {t[0]:.6g} .. {t[-1]:.6g} s")
        for k in sorted(k for k in sig if k != "time"):
            y = sig[k]
            print(f"  {k:24s} {np.min(y):.6g} .. {np.max(y):.6g}")
        return 0

    stim = _stimulus(a)
    module = a.module or _module_name(a.va)
    deck = make_deck(Path(a.va).name, module, stim, a.maxstep, _params(a.param),
                     pin=a.i_signal.split(":", 1)[1] if ":" in a.i_signal else "p")
    if a.cmd == "deck":
        sys.stdout.write(deck)
        return 0
    if not a.twin:
        raise SystemExit("--twin is required for run / compare")

    if a.cmd == "run":
        from .remote import run_remote
        if a.deck:
            deck = Path(a.deck).read_text(encoding="utf-8")
        run = run_remote(deck, a.va, a.name, timeout=a.timeout, keep=a.keep,
                         out_dir=a.out)
        print(f"Spectre run {run.wall_s:.1f} s -> {run.raw_dir}")
        sig, info = load_waveform(run.raw_dir)
    else:
        sig, info = load_waveform(a.psf)
    s = compare_twin(sig, info, stim, a, a.out)
    _print(s)
    return 0 if s["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
