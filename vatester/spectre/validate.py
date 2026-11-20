"""One-command validation of a twin against Spectre.

    python -m vatester.spectre validate --va my_device.va
    python -m vatester.spectre validate --cell MYLIB/my_device/veriloga   # from Virtuoso

What it does, with no further input:

1. Reads the .va (a file, or a Virtuoso cellview through skillbridge): module,
   ports, parameters, $bound_step parameter.
2. Finds the twin: the built-in ECFET v1/v2/v3 by module name, else the twins/
   file whose TWIN_SPEC va_keywords match the file or module name.
3. Picks the device profile (ecfet, memristor, fecap, fefet) and builds its
   fixed test set, scaled by probing the twin (suites.py).
4. Passes every parameter the twin and the .va share to the Spectre instance
   with the twin's value, so both sides start from the same model card.
5. Layer 2: runs EVERY test in Spectre at three maximum steps; the finest run is
   the reference, accepted when the next-finest run agrees with it.
6. Layer 1: scores the twin on every test against that reference (a FAIL
   against an unconverged reference becomes INCONCLUSIVE).
7. Layer 3 (ECFET v3): replays a short training run's write schedule.
8. Writes report.html with every plot, summary.json and the CSV data, and
   opens the report.

Spectre runs on the Cadence host over SSH (remote.py).  `runner` can be
replaced (tests use the twin itself as the reference).
"""

import json
import math
import os
import re
import tempfile
import time
import webbrowser
from dataclasses import fields
from pathlib import Path

import numpy as np

from . import compare as C
from .converge import coarsest_converged, convergence, read_errors
from .deck import (ecfet_roles, make_deck, make_ecfet_deck, make_fecap_deck,
                   make_fefet_deck, transistor_roles, va_interface)
from .psf import find_signal, load_waveform
from .suites import build_suite
from .twinrun import builtin_spec, load_twin, make_model, run_twin, seconds_per_step
from ..twin_loader import load_twins

REPO = Path(__file__).resolve().parents[2]
MAX_POINTS = 200_000          # finest convergence step >= t_stop / MAX_POINTS
POL_KEYS = ("P (uC/cm2)", "Ptot (uC/cm2)", "P (norm.)")   # FE part first


# ------------------------------------------------------------------ twin
def match_twin(va_name, module, workspace):
    """Twin key for a .va: built-in ECFET by module name, else the longest
    va_keyword of a twins/ TWIN_SPEC found in the file or module name."""
    m = re.match(r"ecfet_v([123])\b", module.lower())
    if m:
        return f"v{m.group(1)}"
    if module.lower() == "fefet" and builtin_spec("fefet"):
        return "fefet"
    hay = (Path(va_name).stem + " " + module).lower()
    best, key = 0, None
    for path, payload, err in load_twins(os.path.join(workspace, "twins")):
        if payload is None:
            continue
        spec = payload[1]
        for kw in spec.get("va_keywords", ()):
            if kw.lower() in hay and len(kw) > best:
                best, key = len(kw), spec["key"]
    if key is None:
        raise LookupError(f"no twin matches {va_name} (module {module}); "
                          "write one in twins/ or pass --twin")
    return key


def shared_params(spec, va_params, overrides=None):
    """Twin parameter values the .va also declares (numbers only)."""
    model = make_model(spec, overrides)
    out = {}
    names = {f.name for f in fields(model.p)}
    for n in sorted(names & set(va_params)):
        v = getattr(model.p, n)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        out[n] = float(v)
    return out


# ------------------------------------------------------------------ decks / refs
class Device:
    def __init__(self, profile, va_path, module, ports, inst, bound, v_ds=0.1, spec=None,
                 pins=None, v_read=0.01):
        self.spec = spec or {}
        self.profile, self.va_path, self.module, self.ports = profile, va_path, module, ports
        self.inst, self.bound, self.v_ds = inst, bound, v_ds
        self.pins, self.v_read = pins or {}, v_read
        if profile == "fefet":
            self.roles, self.monitors = transistor_roles(ports)
        elif profile == "ecfet":
            self.roles, self.monitors = ecfet_roles(ports, self.pins)
        else:
            self.roles, self.monitors = {}, ports[2:] if profile == "fecap" else []

    def deck(self, stim, step):
        p = dict(self.inst)
        if self.bound:
            p[self.bound] = step
        va = Path(self.va_path).name
        if self.profile == "ecfet":
            return make_ecfet_deck(va, self.module, stim, step, p, self.v_read,
                                   ports=self.ports, pins=self.pins)
        if self.profile == "fecap":
            return make_fecap_deck(va, self.module, self.ports, stim, step, p)
        if self.profile == "fefet":
            return make_fefet_deck(va, self.module, self.ports, stim, step, p, self.v_ds)
        return make_deck(va, self.module, stim, step, p, pin=self.ports[0])

    def reference(self, sig):
        t = np.asarray(sig["time"], float)

        def get(name):
            y = find_signal(sig, name)
            if y is None:
                raise KeyError(f"{name} not in the Spectre result "
                               f"({', '.join(k for k in sig if k != 'time')})")
            return np.asarray(y, float)
        if self.profile == "ecfet":
            return {"time": t, "G": np.abs(get("vread:p")) / self.v_read}
        if self.profile == "fefet":
            idd = get(f"X1:{self.roles['drain']}")
            return {"time": t, "V": get("n_g"), "I": idd, "G": idd / self.v_ds}
        ref = {"time": t, "V": get("n_p"), "I": get(f"X1:{self.ports[0]}")}
        for p in self.monitors:
            ref["mon:" + p] = get(f"n_{p}")
        return ref


def twin_view(dev, tw):
    """Twin record in the reference's terms (current of a FeCAP from I_dev)."""
    if dev.profile == "fecap" and "obs:I_dev (A)" in tw:
        tw = dict(tw, I=tw["obs:I_dev (A)"])
    return tw


def _pol_key(tw, spec=None, monitor=None):
    """Twin observable compared with a monitor pin: TWIN_SPEC["spectre"]["monitors"]
    [pin] if given, else the first of POL_KEYS the twin has."""
    hint = ((spec or {}).get("spectre") or {}).get("monitors", {}).get(monitor)
    if hint and "obs:" + hint in tw:
        return hint
    return next((k for k in POL_KEYS if "obs:" + k in tw), None)


def score(dev, suite, stim, tw, ref):
    """Profile-specific Layer-1 score -> dict with 'verdict' and the series to plot."""
    out = {"stimulus": stim.name, "description": stim.meta.get("description", "")}
    if dev.profile == "ecfet":
        m = C.waveform_metrics(tw["time"], tw["G"], ref["time"], ref["G"])
        out.update(quantity="G", unit="S", nrmse_pct=100 * m["nrmse"],
                   max_err_pct=100 * m["maxerr"], rule="waveform", verdict=C.verdict(m))
        return out
    if dev.profile in ("memristor", "fefet"):
        reads = stim.read_times(suite.read_level)
        if dev.profile == "fefet":
            gt = np.interp(reads, tw["time"], tw["G"])
            gr = np.interp(reads, ref["time"], ref["G"])
        else:
            r = C.read_metrics(tw, ref, reads)
            gt, gr = r["g_twin"], r["g_ref"]
        if not len(reads):
            out.update(quantity="G", unit="S", rule="reads", verdict="INCONCLUSIVE",
                       note="the test has no read plateaus at the read level")
            return out
        if dev.profile == "fefet":
            r = read_errors(tw, ref, "G", reads)
            if r is None:
                out.update(quantity="G", unit="S", rule="reads", verdict="INCONCLUSIVE",
                           read_times=reads, g_twin=gt, g_ref=gr, n_reads=int(len(reads)),
                           note=f"Spectre conducts no usable read current (read gate "
                                f"{suite.read_level:g} V); check the read bias")
                return out
            err, scale = r
            out.update(read_times=reads, g_twin=gt, g_ref=gr, n_reads=int(len(reads)),
                       read_g_max_pct=100 * float(np.max(err)),
                       read_g_mean_pct=100 * float(np.mean(err)),
                       quantity="G", unit="S",
                       rule=("read G error / G range: max <= 3 %, mean <= 1 %"
                             if scale == "range" else
                             "read G error / G (G does not change in this test): "
                             "max <= 3 %, mean <= 1 %"))
            ok = err.max() <= C.READ_MAX_LIMIT and err.mean() <= C.READ_MEAN_LIMIT
            out["verdict"] = "PASS" if ok else "FAIL"
            return out
        g_floor = 1e-6 * float(np.max(np.abs(gr))) if np.any(gr) else 0.0
        if not np.all(np.isfinite(gr)) or np.min(np.abs(gr)) <= max(g_floor, 1e-30):
            out.update(quantity="G", unit="S", rule="reads", verdict="INCONCLUSIVE",
                       read_times=reads, g_twin=gt, g_ref=gr, n_reads=int(len(reads)),
                       note=f"Spectre conducts (almost) no current at the read level "
                            f"{suite.read_level:g} V; set --read-level")
            return out
        rel = np.abs(gt - gr) / np.abs(gr)
        out.update(read_times=reads, g_twin=gt, g_ref=gr, n_reads=int(len(reads)),
                   read_g_max_pct=100 * float(np.max(rel)),
                   read_g_mean_pct=100 * float(np.mean(rel)))
        ok = rel.max() <= C.READ_MAX_LIMIT and rel.mean() <= C.READ_MEAN_LIMIT
        if dev.profile == "memristor":
            m = C.waveform_metrics(tw["time"], tw["I"], ref["time"], ref["I"])
            out.update(quantity="I", unit="A", nrmse_pct=100 * m["nrmse"],
                       max_err_pct=100 * m["maxerr"])
            ok = ok and m["nrmse"] <= C.NRMSE_LIMIT
            out["rule"] = "I NRMSE <= 1 %, read G max <= 3 %, mean <= 1 %"
        else:
            out.update(quantity="G", unit="S")
            out["rule"] = "read G max <= 3 %, mean <= 1 %"
        out["verdict"] = "PASS" if ok else "FAIL"
        return out
    # fecap: polarization (monitor pin) and the charge that flowed through the
    # terminals, both under the waveform rule.  The current itself is reported
    # but not ruled on: a twin's current is usually a per-step average (dQ/dt)
    # and the simulator's an instantaneous value, which differ on fast edges
    # while their integrals agree.
    mi = C.waveform_metrics(tw["time"], tw["I"], ref["time"], ref["I"])
    q_tw = _cumtrapz(tw["time"], tw["I"])
    q_ref = _cumtrapz(ref["time"], ref["I"])
    mq = C.waveform_metrics(tw["time"], q_tw, ref["time"], q_ref)
    out.update(i_nrmse_pct=100 * mi["nrmse"], i_max_err_pct=100 * mi["maxerr"],
               q_nrmse_pct=100 * mq["nrmse"], q_max_err_pct=100 * mq["maxerr"])
    mon = dev.monitors[0] if dev.monitors else None
    pk = _pol_key(tw, dev.spec, mon)
    ok = C.verdict(mq) == "PASS"
    if pk and mon:
        mp = C.waveform_metrics(tw["time"], tw["obs:" + pk], ref["time"], ref["mon:" + mon])
        out.update(quantity="P", unit=pk, pol_twin=pk, pol_ref=mon,
                   nrmse_pct=100 * mp["nrmse"], max_err_pct=100 * mp["maxerr"])
        ok = ok and C.verdict(mp) == "PASS"
        out["rule"] = ("P and charge (integral of I) each NRMSE <= 1 %, max <= 3 %; "
                       f"current reported: NRMSE {100 * mi['nrmse']:.3g} %")
    else:
        out.update(quantity="Q", unit="C", nrmse_pct=100 * mq["nrmse"],
                   max_err_pct=100 * mq["maxerr"],
                   rule="charge (integral of I) NRMSE <= 1 %, max <= 3 %")
    out["verdict"] = "PASS" if ok else "FAIL"
    return out


def _cumtrapz(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])


def compared_series(dev, tw, ref, s):
    if dev.profile == "ecfet" or dev.profile == "fefet":
        return tw["G"], ref["G"], "G (S)"
    if dev.profile == "fecap" and s.get("pol_twin"):
        return tw["obs:" + s["pol_twin"]], ref["mon:" + s["pol_ref"]], s["pol_twin"]
    return tw["I"], ref["I"], "I (A)"


# ------------------------------------------------------------------ Spectre
class Cancelled(RuntimeError):
    """Raised from a progress callback to stop a validation."""


class Progress:
    """Counts the work and estimates the time remaining from what is left in
    each stage, at the rate measured for that stage:

      Spectre runs  - time per run from the runs finished (unknown before the
                      first one finishes: eta None);
      twin steps    - time per step from the twin's steps so far, before that
                      from a short probe (a FeFET twin takes ms per step, so
                      its runs can outlast Spectre);
      replay        - one Spectre run plus the training (if planned).

    done/total count units (Spectre runs, tests, replay); `fraction` is the
    elapsed share of the estimated total time (the progress bar)."""

    EMIT_EVERY_S = 0.5

    def __init__(self, callback=None):
        self.cb = callback
        self.t0 = time.monotonic()
        self.total = 0
        self.done_base = 0          # units finished in earlier stages
        self.done = 0
        self.tests_total = self.tests_done = 0
        self.runs_total = self.runs_done = 0
        self.replay = self.replay_done = False
        self.twin_total = self.twin_done = 0
        self.twin_s = 0.0           # time spent in twin steps so far
        self.probe_s = None         # probed time per twin step
        self.t_spectre = None       # Spectre stage: start, and time per run
        self.run_s = None
        self.stage = ""
        self._last_emit = 0.0

    def plan(self, runs, tests, replay, twin_steps=0, sec_per_step=None):
        self.runs_total, self.tests_total = runs, tests
        self.replay = bool(replay)
        self.total = runs + tests + (2 if replay else 0)
        self.twin_total, self.probe_s = int(twin_steps), sec_per_step
        self.t_spectre = time.monotonic()
        self.emit("planned")

    def twin_plan(self, twin_steps):
        """The exact twin step count, once the steps are chosen."""
        self.twin_total = self.twin_done + int(twin_steps)

    def twin(self, n_steps, seconds):
        self.twin_done += n_steps
        self.twin_s += seconds
        if time.monotonic() - self._last_emit >= self.EMIT_EVERY_S:
            self.emit()

    def eta(self):
        rem = 0.0
        if self.runs_done < self.runs_total:
            if not self.runs_done:
                return None
            self.run_s = (time.monotonic() - self.t_spectre) / self.runs_done
            rem += self.run_s * (self.runs_total - self.runs_done)
        left = max(0, self.twin_total - self.twin_done)
        if left:
            rate = (self.twin_s / self.twin_done if self.twin_done >= 2000
                    else self.probe_s)
            if rate is None:
                return None
            rem += left * rate
        if self.replay and not self.replay_done:
            rem += (self.run_s or 0.0) + 5.0
        return rem

    def emit(self, stage=None):
        if stage:
            self.stage = stage
        if self.cb is None:
            return
        self._last_emit = time.monotonic()
        el = self._last_emit - self.t0
        eta = self.eta()
        frac = (el / (el + eta) if eta is not None and el + eta > 0
                else (self.done / self.total if self.total else 0.0))
        self.cb({"stage": self.stage, "done": self.done, "total": self.total,
                 "runs_done": self.runs_done, "runs_total": self.runs_total,
                 "tests_done": self.tests_done, "tests_total": self.tests_total,
                 "twin_steps_done": self.twin_done, "twin_steps_total": self.twin_total,
                 "elapsed_s": el, "eta_s": eta, "fraction": min(1.0, frac)})

    def spectre(self, done, total):
        self.runs_done = done
        self.done = self.done_base + done
        self.emit(f"Spectre {done}/{total} runs")

    def spectre_finished(self, n):
        if self.runs_done and self.t_spectre is not None:
            self.run_s = (time.monotonic() - self.t_spectre) / max(1, self.runs_total)
        self.done_base += n
        self.done = self.done_base
        self.runs_done = self.runs_total

    def unit(self, stage, tests=0, n=1, replay_done=False):
        self.done_base += n
        self.done = self.done_base
        self.tests_done += tests
        self.replay_done = self.replay_done or replay_done
        self.emit(stage)


def ssh_runner(timeout=3600, keep=False):
    """A runner that does all its Spectre work over ONE SSH login (opened at
    the first call, kept for every later call, closed by runner.close() -
    validate() closes it at the end)."""
    from .remote import Session, run_remote_batch

    class _Runner:
        def __init__(self):
            self.session = None

        def __call__(self, decks, va_path, out_dir, progress=None):
            if self.session is None:
                self.session = Session().open()
            runs = run_remote_batch(decks, va_path, timeout=timeout, keep=keep,
                                    out_dir=out_dir, progress=progress,
                                    session=self.session)
            return {n: load_waveform(r.raw_dir) for n, r in runs.items()}

        def close(self):
            if self.session is not None:
                self.session.close()
                self.session = None

    return _Runner()



def _call_runner(runner, decks, va_path, out_dir, progress=None):
    """Runners may or may not take a progress callback (tests use simple ones)."""
    import inspect
    try:
        takes = "progress" in inspect.signature(runner).parameters
    except (TypeError, ValueError):
        takes = False
    if takes:
        return runner(decks, va_path, out_dir, progress=progress)
    out = runner(decks, va_path, out_dir)
    if progress is not None:
        progress(len(decks), len(decks))
    return out

def twin_steps_for(profile, steps, converged=None):
    """Twin output steps for one test: a FeFET twin at the coarsest Spectre
    step whose reads converged (`converged`; the middle step as the estimate
    before Spectre has run) and twice that; other twins at the coarsest
    Spectre step / 1, 2, 4."""
    if profile == "fefet":
        sc = converged if converged is not None else sorted(steps)[1 if len(steps) > 1 else 0]
        return [sc, 2 * sc]
    dt = max(steps)
    return [dt, dt / 2, dt / 4]


def _n_steps(stim, dts):
    return sum(len(stim.grid(d)) - 1 for d in dts)


def conv_steps(profile, base, t_stop):
    f = 4.0 if profile == "ecfet" else 10.0
    steps = [base, base / f, base / f / f]
    floor = t_stop / MAX_POINTS
    return sorted({max(s, floor) for s in steps}, reverse=True)


# ------------------------------------------------------------------ plots / report
def _plot_l1(path, dev, stim, tw, ref, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y_tw, y_ref, label = compared_series(dev, tw, ref, s)
    n = 4 if dev.profile == "fecap" and s.get("pol_twin") else 3
    fig, ax = plt.subplots(n, 1, figsize=(9, 2.1 * n + 0.6), sharex=False)
    t = tw["time"]
    ax[0].plot(stim.t, stim.v, color="0.25", lw=0.9)
    ax[0].set_ylabel("drive (A)" if stim.kind == "current" else "drive (V)")
    ax[0].set_title(f"{stim.name}: {s['description']}", fontsize=9)
    ax[1].plot(ref["time"], y_ref, color="black", lw=1.2, label="Spectre")
    ax[1].plot(t, y_tw, "--", color="#D55E00", lw=1.0, label="NeuroVAT twin")
    if "read_times" in s and len(s["read_times"]):
        ax[1].plot(s["read_times"], s["g_ref"] if label.startswith("G") else
                   np.interp(s["read_times"], ref["time"], y_ref), "k.", ms=3)
    ax[1].set_ylabel(label)
    ax[1].legend(fontsize=8, loc="best")
    rng = float(np.max(y_ref) - np.min(y_ref)) or 1.0
    err = 100 * (y_tw - np.interp(t, ref["time"], y_ref)) / rng
    ax[2].plot(t, err, color="#0072B2", lw=0.8)
    ax[2].set_ylabel("error (% of range)")
    ax[2].set_xlabel("time (s)")
    if "read_times" in s and len(s["read_times"]):
        ax2 = ax[2].twinx()
        ax2.plot(s["read_times"], read_error_pct(dev.profile, s["g_twin"], s["g_ref"]),
                 "o", color="#009E73", ms=3)
        ax2.set_ylabel("read G error (% of G range)" if dev.profile == "fefet"
                       else "read G error (%)", color="#009E73")
    if n == 4:
        ax[3].plot(ref["V"], y_ref, color="black", lw=1.0, label="Spectre")
        ax[3].plot(tw["V"], y_tw, "--", color="#D55E00", lw=0.9, label="twin")
        ax[3].set_xlabel("V (V)")
        ax[3].set_ylabel(label)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def read_error_pct(profile, g_twin, g_ref):
    """Signed read-G error in % on the scale of the profile's read rule:
    FeFET - the G range of the test (of G when it does not change);
    memristor - G itself."""
    gt, gr = np.asarray(g_twin, float), np.asarray(g_ref, float)
    if not len(gr):
        return gr
    scale = np.abs(gr)
    if profile == "fefet":
        span = float(np.max(gr) - np.min(gr))
        if span > 0 and span > 1e-9 * float(np.max(np.abs(gr))):
            scale = span
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * (gt - gr) / scale


def _save_l1_data(path, dev, stim, tw, ref, s):
    """The series of one Layer-1 plot (drive, Spectre, twin, reads) for the
    GUI results viewer; numeric arrays plus a JSON string (no pickle)."""
    y_tw, y_ref, label = compared_series(dev, tw, ref, s)
    d = {"drive_t": np.asarray(stim.t, float), "drive_v": np.asarray(stim.v, float),
         "t_ref": np.asarray(ref["time"], float), "y_ref": np.asarray(y_ref, float),
         "t_tw": np.asarray(tw["time"], float), "y_tw": np.asarray(y_tw, float)}
    if "read_times" in s and len(s["read_times"]):
        d.update(read_t=np.asarray(s["read_times"], float),
                 g_ref=np.asarray(s["g_ref"], float), g_twin=np.asarray(s["g_twin"], float),
                 read_err_pct=read_error_pct(dev.profile, s["g_twin"], s["g_ref"]))
    meta = {"label": label, "drive_kind": stim.kind, "stimulus": stim.name,
            "profile": dev.profile, "reference_step_s": s.get("reference_step_s")}
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **d)


def load_l1_data(path):
    """Read back _save_l1_data -> (arrays dict, meta dict); never unpickles."""
    with np.load(path, allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return arrs, meta


def _plot_conv(path, per_stim):
    """Max error of each Spectre step against that test's finest step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name, rows in per_stim.items():
        r = [x for x in rows if x["what"] == "spectre vs finest spectre"]
        if r:
            ax.loglog([x["step_s"] for x in r], [max(x["max_err_pct"], 1e-9) for x in r],
                      "o-", label=name, lw=1)
    ax.axhline(3, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("Spectre maximum time step (s)")
    ax.set_ylabel("max error vs finest step (%)")
    ax.set_title("Spectre against its own finest run (3 % line = pass limit)", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_replay(path, rows, rec, sig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    k = int(np.argmax([r["max_err_pct"] for r in rows]))
    ax[0].bar([r["synapse"] for r in rows], [r["max_err_pct"] for r in rows], color="#D55E00")
    ax[0].axhline(3, color="0.5", ls="--", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("synapse")
    ax[0].set_ylabel("max read error (% of range)")
    from .replay import spectre_g
    g_ref = spectre_g(sig, k)
    ax[1].plot(sig["time"], g_ref * 1e6, color="black", lw=1.0, label="Spectre")
    ax[1].plot(rec.reads_t, rec.reads_g[:, k] * 1e6, "--", color="#D55E00", lw=0.9,
               label="engine read")
    ax[1].set_title(f"synapse {k} (worst)", fontsize=9)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("G (uS)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def _cls(v):
    return {"PASS": "P", "FAIL": "F"}.get(v, "I")


def _html(report, out):
    v = report["verdict"]
    col = {"PASS": "#1b7f3b", "FAIL": "#b3261e"}.get(v, "#a06800")
    h = [f"<!doctype html><meta charset='utf-8'><title>NeuroVAT validation: "
         f"{report['module']}</title>",
         "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:24px auto;"
         "padding:0 16px;color:#222}table{border-collapse:collapse;margin:8px 0 18px}"
         "td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px;text-align:left}"
         "th{background:#f2f2f2}img{max-width:100%;border:1px solid #ddd;margin:6px 0}"
         ".P{color:#1b7f3b;font-weight:bold}.F{color:#b3261e;font-weight:bold}"
         ".I{color:#a06800;font-weight:bold}</style>",
         f"<h1>NeuroVAT validation: {report['module']}</h1>",
         f"<p style='font-size:20px'>Overall: <b style='color:{col}'>{v}</b></p>",
         "<table>"]
    for k in ("va", "twin", "profile", "simulator", "date"):
        h.append(f"<tr><th>{k}</th><td>{_fmt(report.get(k, ''))}</td></tr>")
    h.append("</table><h3>Calibration (probed on the twin)</h3><table>")
    for k, val in report["calibration"].items():
        h.append(f"<tr><th>{k}</th><td>{_fmt(val)}</td></tr>")
    h.append("</table>")
    if report["shared_params"]:
        h.append(f"<p>{len(report['shared_params'])} parameters passed to the Spectre "
                 "instance with the twin's values: "
                 + ", ".join(sorted(report["shared_params"])) + "</p>")
    h += ["<h2>Layer 2: convergence of the reference</h2>",
          "<p>Every test runs in Spectre at three maximum steps. The twin is scored "
          "against the finest run, which is accepted only if the next-finest run agrees "
          "with it under the pass rule; otherwise the test is INCONCLUSIVE. The last "
          "column says whether the coarse deck step alone would have been enough.</p>",
          "<table><tr><th>test</th><th>steps (s)</th><th>finest vs next (max %)</th>"
          "<th>reference converged</th><th>coarsest converged step (s)</th>"
          "<th>deck step enough</th></tr>"]
    for c in report["layer2"]:
        ok = c["reference_converged"]
        steps = ", ".join(f"{x:g}" for x in c["steps"])
        h.append(f"<tr><td>{c['stimulus']}</td><td>{steps}</td>"
                 f"<td>{_fmt(c['next_vs_finest_max_pct'])}</td>"
                 f"<td class={'P' if ok else 'F'}>{'yes' if ok else 'NO'}</td>"
                 f"<td>{c['coarsest_converged_step_s']:g}</td>"
                 f"<td>{'yes' if c['deck_step_converged'] else 'no'}</td></tr>")
    h += ["</table><img src='layer2_convergence.png'>",
          "<h2>Layer 1: waveform agreement</h2><table><tr><th>test</th>"
          "<th>compared</th><th>reference step (s)</th><th>NRMSE (%)</th><th>max (%)</th>"
          "<th>read G max / mean (%)</th><th>rule</th><th>verdict</th></tr>"]
    for t in report["layer1"]:
        rg = (f"{_fmt(t['read_g_max_pct'])} / {_fmt(t['read_g_mean_pct'])}"
              if "read_g_max_pct" in t else "")
        note = f"<br><small>{t['note']}</small>" if t.get("note") else ""
        h.append(f"<tr><td>{t['stimulus']}</td><td>{t['quantity']}</td>"
                 f"<td>{_fmt(t.get('reference_step_s', ''))}</td>"
                 f"<td>{_fmt(t.get('nrmse_pct', ''))}</td><td>{_fmt(t.get('max_err_pct', ''))}"
                 f"</td><td>{rg}</td><td>{t['rule']}</td>"
                 f"<td class={_cls(t['verdict'])}>{t['verdict']}{note}</td></tr>")
    h.append("</table>")
    for t in report["layer1"]:
        h.append(f"<h4>{t['stimulus']}</h4><p>{t['description']}</p>"
                 f"<img src='l1_{t['stimulus']}.png'>")
    r3 = report.get("layer3")
    if r3:
        h += ["<h2>Layer 3: training-schedule replay</h2>",
              f"<p>{r3['info']['n_pulses']} write pulses on {len(r3['rows'])} synapses "
              f"({r3['info']['seconds']} s of training, seed {r3['info']['seed']}); "
              f"<b>{r3['passed']}/{len(r3['rows'])}</b> synapses pass, worst max error "
              f"{_fmt(r3['worst_max_err_pct'])} %.</p><img src='layer3_replay.png'>"]
    h.append("<p style='color:#666;font-size:12px'>PASS: every test agrees under its rule. "
             "FAIL: at least one test disagrees against a converged reference. "
             "INCONCLUSIVE: no FAIL, but a reference did not converge or the comparison "
             "is undefined (see the notes). Agreement is a tolerance check on this fixed "
             "test set, not a proof of equivalence.</p>")
    (out / "report.html").write_text("\n".join(h), encoding="utf-8")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()
                if k not in ("read_times", "g_twin", "g_ref")}
    if isinstance(o, list):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# ------------------------------------------------------------------ main entry
def validate(va_path, twin=None, workspace=None, out_dir=None, params=None,
             profile=None, width=None, runner=None, replay=True, open_report=True,
             v_ds=0.1, log=print, read_level=None, answers=None, progress=None):
    """answers: the user's setup (setup.interactive_setup or a saved
    <name>.validate.json): twin, profile, width, amp_pos, amp_neg, read_level,
    v_ds, twin_read_param.  Explicit arguments win over answers.
    progress(dict) receives the stage, done/total work units, Spectre runs and
    tests done, the elapsed time and the estimated time remaining (s)."""
    prog = Progress(progress)
    runner = runner or ssh_runner()
    try:
        return _validate(**{k: v for k, v in locals().items() if k != 'prog'}, prog=prog)
    finally:
        if runner is not None and hasattr(runner, 'close'):
            runner.close()


def _validate(va_path, twin, workspace, out_dir, params, profile, width, runner, replay,
              open_report, v_ds, log, read_level, answers, progress, prog):
    a = answers or {}
    twin = twin or a.get("twin")
    profile = profile or a.get("profile")
    width = width or a.get("width")
    read_level = read_level if read_level is not None else a.get("read_level")
    if a.get("v_ds") is not None and v_ds == 0.1:
        v_ds = float(a["v_ds"])
    given = {k: a[k] for k in ("amp_pos", "amp_neg", "width") if a.get(k) is not None}
    params = dict(params or {})
    if a.get("twin_read_param") and read_level is not None:
        params.setdefault(a["twin_read_param"], float(read_level))
    workspace = str(workspace or REPO)
    if (profile or "") == "fefet" or a.get("profile") == "fefet":
        try:                     # a FeFET twin with a drain-bias parameter reads at V_ds
            from dataclasses import fields as _f
            if "Vds" in {f.name for f in _f(load_twin(twin, workspace)["params_class"])}:
                params.setdefault("Vds", float(v_ds))
        except Exception:        # noqa: BLE001 - twin resolved again below
            pass
    params = params or None
    va_path = Path(va_path)
    text = va_path.read_text(encoding="utf-8", errors="replace")
    module, ports, va_params, bound = va_interface(text)
    key = twin or match_twin(va_path.name, module, workspace)
    spec = load_twin(key, workspace)
    suite = build_suite(spec, params, profile, width, read_level, given)
    inst = {**shared_params(spec, va_params, params),
            **{k: v for k, v in (params or {}).items() if k in va_params}}
    dev = Device(suite.profile, va_path, module, ports, inst, bound, v_ds, spec,
                 pins=a.get("pins"))
    # under the WORKSPACE (in the Windows exe the code lives in _internal/,
    # where the results window would not find the run)
    out = Path(out_dir or Path(workspace) / "results" / "validation" /
               f"{module}_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    runner = runner or ssh_runner()
    log(f"{va_path.name}: module {module}({', '.join(ports)}), twin {spec['key']}, "
        f"profile {suite.profile}, {len(suite.stimuli)} stimuli")
    log(f"  calibration: " + ", ".join(f"{k} {v:.4g}" for k, v in suite.calibration.items()))

    # ---- layers 2 and 1: every test at three steps, scored against its finest
    plan = {st.name: conv_steps(suite.profile, suite.base_step, st.t_stop)
            for st in suite.stimuli}
    n = sum(len(v) for v in plan.values())
    do_replay = replay and dev.profile == "ecfet" and spec["key"] == "v3"
    est = sum(_n_steps(st, twin_steps_for(suite.profile, plan[st.name]))
              for st in suite.stimuli)
    try:
        probe = seconds_per_step(spec, suite.stimuli[0],
                                 min(plan[suite.stimuli[0].name]), params)
    except Exception:            # noqa: BLE001 - the estimate only
        probe = None
    prog.plan(n, len(suite.stimuli), do_replay, twin_steps=est, sec_per_step=probe)
    log(f"Spectre: {len(suite.stimuli)} tests x 3 steps = {n} runs ...")
    decks = {f"{st.name}__s{i}": dev.deck(st, step)
             for st in suite.stimuli for i, step in enumerate(plan[st.name])}
    (out / "spectre").mkdir(parents=True, exist_ok=True)
    for name, text in decks.items():      # the netlists Spectre ran (results viewer)
        (out / "spectre" / f"{name}.scs").write_text(str(text), encoding="utf-8")
    res = _call_runner(runner, decks, va_path, out / "spectre", prog.spectre)
    prog.spectre_finished(n)
    info = next(iter(res.values()))[1]
    # A FeFET's state is read at the read plateaus: Spectre's own convergence
    # and the twin are judged there (domain switching makes the drain current
    # between reads step-sensitive).  Its twin runs at the coarsest Spectre
    # step whose reads converged, and at twice that as its own check.
    all_refs, all_reads, tw_dts = {}, {}, {}
    for st in suite.stimuli:
        steps = plan[st.name]
        all_refs[st.name] = {step: dev.reference(res[f"{st.name}__s{i}"][0])
                             for i, step in enumerate(steps)}
        all_reads[st.name] = (st.read_times(suite.read_level)
                              if dev.profile == "fefet" else None)
        sc_step = (coarsest_converged(all_refs[st.name], "G", all_reads[st.name])
                   if dev.profile == "fefet" else None)
        tw_dts[st.name] = twin_steps_for(dev.profile, steps, sc_step)
    prog.twin_plan(sum(_n_steps(st, tw_dts[st.name]) for st in suite.stimuli))
    conv, l1, conv_rows = [], [], {}
    for st in suite.stimuli:
        steps = plan[st.name]
        refs = all_refs[st.name]
        dt = max(steps)
        reads = all_reads[st.name]
        prog.emit(f"twin: {st.name} (test {len(l1) + 1}/{len(suite.stimuli)})")
        twins = {d: twin_view(dev, run_twin(spec, st, d, params, progress=prog.twin))
                 for d in tw_dts[st.name]}
        qkey, skey, cross = "I", None, True
        if dev.profile == "ecfet":
            qkey = "G"
        elif dev.profile == "fefet":            # the twin's G is a read value
            qkey = "G"
        elif dev.profile == "fecap" and dev.monitors:
            pk = _pol_key(twins[dt], spec, dev.monitors[0])
            if pk:
                refs = {k: dict(r, P=r["mon:" + dev.monitors[0]]) for k, r in refs.items()}
                twins = {k: dict(t, P=t["obs:" + pk]) for k, t in twins.items()}
                qkey = "P"
        rows, summ = convergence(refs, st, dev.profile, twins, quantity_key=qkey,
                                 cross=cross, spectre_key=skey, reads=reads)
        conv_rows[st.name] = rows
        fine = sorted(steps)
        nxt = next(r for r in rows if r["what"] == "spectre vs finest spectre"
                   and r["step_s"] == fine[1])
        ref_ok = nxt["verdict"] == "PASS"
        conv.append({"stimulus": st.name, "steps": steps, "rows": rows,
                     "next_vs_finest_max_pct": nxt["max_err_pct"],
                     "reference_converged": ref_ok,
                     "coarsest_converged_step_s": summ["coarsest_converged_step_s"],
                     "deck_step_converged": max(steps) not in summ["unconverged_steps_s"]})
        ref = refs[fine[0]]
        tw = twins[min(twins)]
        sc = score(dev, suite, st, tw, ref)
        sc["reference_step_s"] = fine[0]
        if not ref_ok:
            conv_note = (f"Spectre's result depends on the time step: at {fine[1]:.3g} s it "
                         f"differs from {fine[0]:.3g} s by up to {nxt['max_err_pct']:.3g} %")
            if sc["verdict"] == "FAIL":
                sc["verdict"] = "INCONCLUSIVE"
                conv_note += "; no converged reference to judge the twin against"
            elif sc["verdict"] == "PASS":
                conv_note += "; the twin agrees with the finest run"
            sc["note"] = conv_note + (f". {sc['note']}" if sc.get("note") else "")
        _plot_l1(out / f"l1_{st.name}.png", dev, st, tw, ref, sc)
        _save_l1_data(out / f"l1_{st.name}.npz", dev, st, tw, ref, sc)
        l1.append(sc)
        prog.unit(f"test {len(l1)}/{len(suite.stimuli)} scored: {st.name} {sc['verdict']}",
                  tests=1)
        parts = []
        if "nrmse_pct" in sc:
            parts.append(f"NRMSE {sc['nrmse_pct']:.3g} %, max {sc['max_err_pct']:.3g} %")
        if "read_g_max_pct" in sc:
            parts.append(f"read G max {sc['read_g_max_pct']:.3g} %")
        log(f"  {st.name:20s} ref {fine[0]:.3g} s {'conv.' if ref_ok else 'NOT conv.'}  "
            f"{sc['quantity']}: {', '.join(parts) or '-'}  {sc['verdict']}")
    _plot_conv(out / "layer2_convergence.png", conv_rows)

    # ---- layer 3
    l3 = None
    if do_replay:
        from .deck import make_replay_deck
        from .replay import gate_waves, record_training, score_replay
        log("Layer 3: training-schedule replay (16 synapses, 4 s) ...")
        prog.emit("replay: training run")
        rec = record_training(16, 4.0, 1, v3_overrides=params)
        prog.unit("replay: engine run recorded")
        waves, t_stop = gate_waves(rec)
        deck = make_replay_deck(va_path.name, module, waves, t_stop, 1e-3,
                                {"Rinit": 1.0 / rec.g_start, "A_stdp": 0.0},
                                ports=ports, pins=a.get("pins"),
                                kind=spec.get("input_kind", "current"))
        sig = _call_runner(runner, {"replay": deck}, va_path, out / "spectre")["replay"][0]
        prog.unit("replay: Spectre done", replay_done=True)
        rows3 = score_replay(rec, sig)
        _plot_replay(out / "layer3_replay.png", rows3, rec, sig)
        l3 = {"info": rec.info, "rows": rows3,
              "passed": sum(r["verdict"] == "PASS" for r in rows3),
              "worst_max_err_pct": max(r["max_err_pct"] for r in rows3)}
        log(f"  {l3['passed']}/{len(rows3)} synapses pass, worst {l3['worst_max_err_pct']:.3g} %")

    verdicts = [t["verdict"] for t in l1]
    if l3 is not None:
        verdicts.append("PASS" if l3["passed"] == len(l3["rows"]) else "FAIL")
    overall = ("FAIL" if "FAIL" in verdicts else
               "INCONCLUSIVE" if "INCONCLUSIVE" in verdicts else "PASS")
    report = {"verdict": overall, "module": module, "va": str(va_path),
              "twin": f"{spec['key']} ({spec.get('label', '')})", "profile": dev.profile,
              "simulator": f"{info.get('simulator', '')} {info.get('version', '')}".strip(),
              "date": time.strftime("%Y-%m-%d %H:%M"), "setup": a,
              "calibration": suite.calibration, "shared_params": dev.inst,
              "layer2": conv,
              "layer1": l1, "layer3": l3}
    _html(report, out)
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2, default=str),
                                      encoding="utf-8")
    prog.emit("finished")
    log(f"Overall: {report['verdict']}  ->  {out / 'report.html'}")
    if open_report:
        try:
            webbrowser.open((out / "report.html").resolve().as_uri())
        except Exception:
            pass
    return report, out


def va_from_virtuoso(cellview, dest_dir):
    """Read LIB/CELL/VIEW through skillbridge and save it as <cell>.va."""
    from ..virtuoso import VirtuosoLink
    lib, cell, view = cellview.split("/")
    link = VirtuosoLink()
    try:
        link.connect()
        r = link.read_source(lib, cell, view)
    finally:
        link.disconnect()
    if not r["ok"]:
        raise RuntimeError(f"{cellview}: {r['note']}")
    if not re.match(r"^[A-Za-z0-9_.-]+$", cell):
        raise ValueError(f"unusual cell name: {cell!r}")
    p = Path(dest_dir) / f"{cell}.va"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(r["text"], encoding="utf-8")
    return p
