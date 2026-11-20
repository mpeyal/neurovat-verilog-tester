"""Command line: validate a Python twin against Spectre in three layers.

Layer 1, waveform agreement (one stimulus):
    python -m vatester.spectre run  --va verilog/vteam.va --twin vteam \\
        --pulses 0.5,-0.25 --width 10e-9 --gap 3e-9 --read-v -0.1 --maxstep 1e-11
    python -m vatester.spectre run  --va verilog/ecfet_v3.va --twin v3 --kind current \\
        --pulses 5e-11 --width 10e-3 --gap 0.2 --maxstep 1e-3
    python -m vatester.spectre compare --psf result.raw|result.csv ...   (no Spectre run)
Layer 2, reference convergence (Spectre at several steps, twin refined):
    python -m vatester.spectre converge ... --steps 1e-10,1e-11,1e-12
Layer 3, training-schedule replay (ECFET v3):
    python -m vatester.spectre replay --va verilog/ecfet_v3.va --afferents 16 --seconds 4
Utilities:
    python -m vatester.spectre import result.raw | result.csv | --skill /path/on/host
    python -m vatester.spectre deck ... > my.scs

Spectre runs on the Cadence host over SSH (the Virtuoso login); results come
back as PSF-ASCII.  A result you already have (PSF-ASCII or a CSV export from
Virtuoso/ViVA) is compared with `compare --psf`.  The stimulus is one of
--pwl "t v t v ...", --pwl-file (t,value CSV) or --pulses; --param name=value
goes to both the .va instance and the twin.  --device picks the deck:
"2t" (X1 (n_p 0)) or "ecfet" (pins by name, --pins to set them; G = read
current / V_read); the
default is "ecfet" for the built-in ECFET twins v1/v2/v3, else "2t".
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import layer1
from .converge import convergence
from .deck import make_deck, make_ecfet_deck
from .psf import PSFError, load_waveform, parse_waveform
from .stimulus import Stimulus
from .twinrun import builtin_spec, load_twin, run_twin

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


def _device(a):
    if a.device:
        return a.device
    return "ecfet" if a.twin and builtin_spec(a.twin) else "2t"


def _va_params(items, maxstep):
    """Instance-only parameters; the value 'maxstep' becomes the deck's maxstep
    (e.g. a model's $bound_step parameter)."""
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not k or not v:
            raise SystemExit(f"--va-param needs name=value, got {it!r}")
        out[k.strip()] = maxstep if v.strip() == "maxstep" else float(v)
    return out


def _deck(a, stim, maxstep, name=None):
    module = a.module or _module_name(a.va)
    params = {**_params(a.param), **_va_params(a.va_param, maxstep)}
    va = Path(a.va).name
    if _device(a) == "ecfet":
        st = stim if name is None else Stimulus(stim.t, stim.v, stim.kind, name)
        return make_ecfet_deck(va, module, st, maxstep, params, v_read=a.v_read_bias,
                               strobe=a.strobe, ports=_ports(a.va), pins=_pins(a))
    return make_deck(va, module, stim, maxstep, params,
                     pin=a.i_signal.split(":", 1)[1] if ":" in a.i_signal else "p")


def _ports(va_path):
    from .deck import va_interface
    return va_interface(Path(va_path).read_text(encoding="utf-8", errors="replace"))[1]


def _pins(a):
    """--pins in=In,out=Out,gate=IGsc -> dict (ECFET pin roles)."""
    out = {}
    for part in (getattr(a, "pins", None) or "").replace(";", ",").split(","):
        if part.strip():
            k, _, v = part.partition("=")
            if not v:
                raise SystemExit(f"--pins needs role=port pairs, got {part!r}")
            out[k.strip()] = v.strip()
    return out


def _signals(a):
    return {"v": a.v_signal, "i": a.i_signal, "rmem": a.rmem_signal}


def _twin(a, stim, dt):
    spec = load_twin(a.twin, a.workspace)
    return spec, run_twin(spec, stim, dt, _params(a.param))


def _save(out, name, obj):
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / name).write_text(json.dumps(obj, indent=2, default=float),
                                      encoding="utf-8")


def _layer1(a, sig, info, stim):
    dev = _device(a)
    ref = layer1.reference(sig, dev, _signals(a), a.v_read_bias)
    spec, tw = _twin(a, stim, a.dt_max or a.maxstep)
    s = {"twin": spec["key"], "device": dev, "stimulus": stim.name, "kind": stim.kind,
         "simulator": info.get("simulator", ""), "version": info.get("version", ""),
         "reference_points": int(len(ref["time"]))}
    s.update(layer1.score(tw, ref, dev, stim, a.read_v, a.rule))
    if a.out:
        key = layer1.quantity(dev, stim.kind)
        yr = np.interp(tw["time"], ref["time"], ref[key])
        rows = ["t_s,twin,spectre"] + [f"{t:.12g},{x:.10g},{y:.10g}"
                                         for t, x, y in zip(tw["time"], tw[key], yr)]
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f"{stim.name}_compare.csv").write_text("\n".join(rows) + "\n",
                                                              encoding="utf-8")
        _save(a.out, f"{stim.name}_summary.json", s)
    print(f"Layer 1: twin {s['twin']} vs {s['simulator']} {s['version']}".rstrip())
    print(f"  stimulus {s['stimulus']} ({s['kind']}, {s['device']} deck); reference "
          f"{s['reference_points']} points, twin {s['twin_points']} points")
    print(f"  {s['compared']}: NRMSE {s['nrmse_pct']:.4g} %, max {s['max_err_pct']:.4g} % "
          f"of the range")
    if "n_reads" in s:
        print(f"  read G ({s['n_reads']} reads): max {s['read_g_max_pct']:.4g} %, "
              f"mean {s['read_g_mean_pct']:.4g} %")
    print(f"  rule {s['rule']}: {s['verdict']}")
    return 0 if s["verdict"] == "PASS" else 1


def _converge(a, stim):
    from .remote import run_remote_batch
    steps = sorted(float(x) for x in a.steps.split(","))
    if len(steps) < 2:
        raise SystemExit("--steps needs at least two maximum steps")
    names = {s: f"{a.name}_s{k}" for k, s in enumerate(steps)}
    decks = {names[s]: _deck(a, stim, s, names[s]) for s in steps}
    runs = run_remote_batch(decks, a.va, timeout=a.timeout, keep=a.keep, out_dir=a.out)
    dev = _device(a)
    refs = {}
    for s in steps:
        sig, info = load_waveform(runs[names[s]].raw_dir)
        refs[s] = layer1.reference(sig, dev, _signals(a), a.v_read_bias)
    dt = a.dt_max or max(steps)
    spec = load_twin(a.twin, a.workspace)
    twins = {dt / f: run_twin(spec, stim, dt / f, _params(a.param)) for f in (1, 2, 4)}
    rows, summ = convergence(refs, stim, dev, twins)
    print(f"Layer 2: {spec['key']} under {stim.name}; Spectre steps "
          f"{', '.join(f'{s:g}' for s in steps)} s, twin dt {dt:g} s")
    for r in rows:
        print(f"  {r['what']:30s} step {r['step_s']:<8.3g} NRMSE {r['nrmse_pct']:<10.4g} % "
              f"max {r['max_err_pct']:<10.4g} %  {r['verdict']}")
    print(f"  coarsest Spectre step that agrees with the finest: "
          f"{summ['coarsest_converged_step_s']:g} s")
    if summ["unconverged_steps_s"]:
        print("  NOT converged at: " + ", ".join(f"{s:g} s" for s in
                                               summ["unconverged_steps_s"]))
    print(f"  twin vs finest Spectre: {summ['twin_vs_finest']}")
    _save(a.out, f"{a.name}_convergence.json", {"rows": rows, "summary": summ})
    return 0 if summ["twin_vs_finest"] == "PASS" else 1


def _replay(a):
    from .deck import make_replay_deck
    from .remote import run_remote_batch
    from .replay import gate_waves, record_training, score_replay
    over = _params(a.param)
    rec = record_training(a.afferents, a.seconds, a.seed, a.pattern_frac,
                          a.threshold_per_aff, over)
    print(f"engine: {rec.info['n_pulses']} write pulses on {rec.n} synapses, "
          f"{rec.info['post_spikes']} output spikes, {len(rec.reads_t)} reads, "
          f"t_end {rec.t_end:.3f} s")
    waves, t_stop = gate_waves(rec)
    module = a.module or _module_name(a.va)
    deck = make_replay_deck(Path(a.va).name, module, waves, t_stop, a.maxstep,
                            {"Rinit": 1.0 / rec.g_start, "A_stdp": 0.0, **over,
                             **_va_params(a.va_param, a.maxstep)},
                            strobe=a.strobe or 2e-4, ports=_ports(a.va), pins=_pins(a))
    run = run_remote_batch({a.name: deck}, a.va, timeout=a.timeout, keep=a.keep,
                           out_dir=a.out)[a.name]
    sig, info = load_waveform(run.raw_dir)
    rows = score_replay(rec, sig)
    npass = sum(r["verdict"] == "PASS" for r in rows)
    print(f"Layer 3: engine reads vs {info.get('simulator', '')} {info.get('version', '')}"
          f" ({len(sig['time'])} points)")
    for r in rows:
        print(f"  synapse {r['synapse']:2d}: {r['pulses']:4d} pulses, G range "
              f"{r['g_range_uS']:7.2f} uS, NRMSE {r['nrmse_pct']:.4g} %, "
              f"max {r['max_err_pct']:.4g} %  {r['verdict']}")
    worst = max(r["max_err_pct"] for r in rows)
    print(f"  {npass}/{len(rows)} synapses pass; worst max error {worst:.4g} %")
    _save(a.out, f"{a.name}_replay.json", {"info": rec.info, "rows": rows})
    return 0 if npass == len(rows) else 1


def _add_common(p, twin_required=False):
    p.add_argument("--va", required=True, help="Verilog-A file")
    p.add_argument("--module", help="module name (default: read from the .va)")
    p.add_argument("--twin", required=twin_required,
                   help="twin key: v1/v2/v3 (built-in ECFET) or TWIN_SPEC['key'] in twins/")
    p.add_argument("--workspace", default=str(REPO), help="folder holding twins/")
    p.add_argument("--device", choices=("2t", "ecfet"))
    p.add_argument("--kind", choices=("voltage", "current"), default="voltage")
    p.add_argument("--name", default="stim", help="run / stimulus name")
    p.add_argument("--pwl", help='"t v t v ..." points')
    p.add_argument("--pwl-file", help="two-column t,value file (CSV)")
    p.add_argument("--pulses", help="comma-separated pulse amplitudes")
    p.add_argument("--width", type=float, default=10e-9)
    p.add_argument("--gap", type=float, default=3e-9)
    p.add_argument("--edge", type=float, default=50e-12)
    p.add_argument("--read-v", type=float, help="2t: a read plateau after each pulse "
                   "(--pulses) and per-read G at plateaus of this level")
    p.add_argument("--v-read-bias", type=float, default=0.01,
                   help="ecfet: read bias across In-Out (V)")
    p.add_argument("--param", action="append", help="name=value (twin and instance)")
    p.add_argument("--va-param", action="append",
                   help="name=value on the .va instance only; value 'maxstep' = the deck's step")
    p.add_argument("--maxstep", type=float, default=1e-11, help="Spectre maxstep (s)")
    p.add_argument("--strobe", type=float, help="Spectre strobe period (s)")
    p.add_argument("--dt-max", type=float, help="twin output step (default maxstep)")
    p.add_argument("--rule", choices=("waveform", "reads"))
    p.add_argument("--v-signal", default="n_p")
    p.add_argument("--i-signal", default="X1:p")
    p.add_argument("--rmem-signal", default="n_rmem")
    p.add_argument("--pins", help="ECFET pin roles, e.g. in=In,out=Out,gate=IGsc "
                   "(default: by name)")
    p.add_argument("--out", help="folder for results, CSV and JSON")
    p.add_argument("--keep", action="store_true", help="keep the remote run folder")
    p.add_argument("--timeout", type=int, default=3600)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vatester.spectre", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate", help="ONE command: automatic test set, layers 1-3, "
                       "plots in report.html")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--va", help="Verilog-A file")
    src.add_argument("--cell", help="LIB/CELL/VIEW in Virtuoso (read through skillbridge)")
    p.add_argument("--twin", help="twin key (default: matched from the .va)")
    p.add_argument("--workspace", default=str(REPO), help="folder holding twins/")
    p.add_argument("--profile", choices=("ecfet", "memristor", "fecap", "fefet"),
                   help="device profile (default: from the twin)")
    p.add_argument("--width", type=float, help="base pulse width (default per profile)")
    p.add_argument("--param", action="append", help="name=value for the twin and the .va")
    p.add_argument("--v-ds", type=float, default=0.1, help="FeFET drain read bias (V)")
    p.add_argument("--read-level", type=float,
                   help="read bias of the read plateaus (memristor: V; FeFET: gate V)")
    p.add_argument("--amp-pos", type=float, help="positive / program amplitude (V; ECFET: A)")
    p.add_argument("--amp-neg", type=float, help="negative / erase amplitude (V)")
    p.add_argument("--ask", action="store_true",
                   help="ask the setup questions even when saved answers exist")
    p.add_argument("--yes", action="store_true",
                   help="ask nothing: use saved answers, else the automatic values")
    p.add_argument("--no-replay", action="store_true", help="skip layer 3")
    p.add_argument("--no-open", action="store_true", help="do not open the report")
    p.add_argument("--out", help="report folder (default results/validation/<module>_<time>)")
    p.add_argument("--keep", action="store_true", help="keep the remote run folders")
    p.add_argument("--timeout", type=int, default=3600)
    p = sub.add_parser("run", help="Layer 1: deck -> Spectre (SSH) -> PSF import -> twin")
    _add_common(p, True)
    p.add_argument("--deck", help="run this hand-written deck instead")
    p = sub.add_parser("compare", help="Layer 1 on a local PSF-ASCII or CSV result")
    _add_common(p, True)
    p.add_argument("--psf", required=True, help="PSF-ASCII file, .raw folder or CSV")
    p = sub.add_parser("converge", help="Layer 2: Spectre at several steps, twin refined")
    _add_common(p, True)
    p.add_argument("--steps", required=True, help="comma-separated Spectre maxsteps (s)")
    p = sub.add_parser("replay", help="Layer 3: training-schedule replay (ECFET v3)")
    _add_common(p)
    p.add_argument("--afferents", type=int, default=16)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pattern-frac", type=float, default=0.25)
    p.add_argument("--threshold-per-aff", type=float, default=0.25)
    p = sub.add_parser("deck", help="print the generated Spectre deck")
    _add_common(p)
    p = sub.add_parser("import", help="list the signals of a PSF-ASCII / CSV result")
    p.add_argument("path", nargs="?")
    p.add_argument("--skill", help="read this file on the host through skillbridge")
    a = ap.parse_args(argv)

    if a.cmd == "validate":
        from .validate import ssh_runner, va_from_virtuoso, validate
        va = a.va
        if a.cell:
            import tempfile
            va = va_from_virtuoso(a.cell, a.out or tempfile.mkdtemp(prefix="nvat_va_"))
            print(f"read {a.cell} from Virtuoso -> {va}")
        from .setup import interactive_setup, load_answers
        runner = ssh_runner(a.timeout, a.keep)
        saved = load_answers(va)
        answers = saved
        if not a.yes and sys.stdin.isatty():
            if saved and not a.ask:
                print(f"saved setup for {Path(va).name}: "
                      + ", ".join(f"{k}={v}" for k, v in saved.items()))
                if input("use it? [Y/n]: ").strip().lower() in ("n", "no"):
                    answers = interactive_setup(va, a.workspace, runner, saved=saved)
            else:
                answers = interactive_setup(va, a.workspace, runner, saved=saved)
        answers = dict(answers or {})
        for k in ("amp_pos", "amp_neg"):
            if getattr(a, k) is not None:
                answers[k] = getattr(a, k)
        report, out = validate(va, a.twin, a.workspace, a.out, _params(a.param) or None,
                               a.profile, a.width, runner,
                               not a.no_replay, not a.no_open, a.v_ds,
                               read_level=a.read_level, answers=answers)
        return 0 if report["verdict"] == "PASS" else 1

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

    if a.cmd == "replay":
        if a.maxstep == 1e-11:            # the 2t default; the replay runs in ms
            a.maxstep = 1e-3
        a.name = a.name if a.name != "stim" else "replay"
        return _replay(a)

    stim = _stimulus(a)
    if a.cmd == "deck":
        sys.stdout.write(_deck(a, stim, a.maxstep))
        return 0
    if a.cmd == "converge":
        return _converge(a, stim)
    if a.cmd == "run":
        from .remote import run_remote
        deck = Path(a.deck).read_text(encoding="utf-8") if a.deck else _deck(a, stim, a.maxstep)
        run = run_remote(deck, a.va, a.name, timeout=a.timeout, keep=a.keep, out_dir=a.out)
        print(f"Spectre run {run.wall_s:.1f} s -> {run.raw_dir}")
        sig, info = load_waveform(run.raw_dir)
    else:
        sig, info = load_waveform(a.psf)
    try:
        return _layer1(a, sig, info, stim)
    except layer1.SignalError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    sys.exit(main())
