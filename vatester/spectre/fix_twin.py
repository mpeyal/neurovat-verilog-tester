"""Fix a twin that failed validation: the assistant edits the twin, NeuroVAT
re-scores it, until it passes or a few rounds are used up.

    result = fix_twin(run_dir, workspace, agent, rounds=4, log=print)

How one round works
  1. The assistant (agent.send, file edits allowed, NO shell) gets the failed
     tests (metrics, rule, where the error is largest) and edits ONLY the twin
     file.  The .va, the Spectre results and the report are fenced UNTRUSTED
     context (agent._wrap_ctx).
  2. Every other file under twins/, verilog/, simulation_core/, vatester/ and
     tools/ is compared with a snapshot taken before the round: a changed
     source file is restored; a new source file under twins/ or verilog/ is
     removed (twins/ files are executed on load).  Binaries and other new
     files are never deleted, only reported.
  3. validate() re-runs the twin and scores it against the Spectre results
     that the failed run already downloaded - no Spectre run, no SSH login.
     The netlists must be identical to the ones Spectre ran (a twin parameter
     shared with the .va goes into the netlist); if not, the round is rejected
     and the assistant is told why.
  4. PASS ends the loop (the fixed twin stays).  Otherwise the new errors are
     the next round's input.  The assistant may end early with "IMPOSSIBLE:".

If no round passes, the original twin is restored, the best attempt is kept
as <run>/fix/best_attempt_<twin>.py, and the verdict "fix not possible" comes
with the assistant's diagnosis and suggestions plus NeuroVAT's own checks
(unconverged Spectre references, failures only at the reads, ...).

Only twins in the workspace's twins/ folder are fixed here; the built-in
ECFET models (simulation_core) are not.
"""

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

SNAPSHOT_DIRS = ("twins", "verilog", "simulation_core", "vatester", "tools")
ROUNDS = 4

PROMPT = """A Python twin of a Verilog-A model FAILED NeuroVAT's validation against
Cadence Spectre. Fix the twin so it reproduces the .va as Spectre simulates it.

Edit ONLY this file: {twin_file}
Rules:
- Spectre and the .va are the reference: do NOT edit the .va or any other file
  (other edits are undone automatically).
- Keep the twin's interface (TWIN_SPEC, the params dataclass, the model class
  and its methods) so NeuroVAT can still load and drive it.
- Do NOT change the default value of a parameter the .va also declares
  ({shared}): those values go into the Spectre netlist, and the saved Spectre
  results would no longer apply (such a round is rejected). Fix the twin's
  equations, state update, clamping, units or numerics instead.
- Compare the twin's code with the .va line by line where the error is
  largest; small numerical differences (integration order, limiting
  functions, event timing) are the usual cause.
- You cannot run Spectre; NeuroVAT re-scores your edit after you finish.

After editing, end your answer with these lines:
CHANGE: <one line: what you changed and why>
DIAGNOSIS: <one line: the cause of the mismatch as you understand it>
SUGGESTION: <one line per suggestion for the user, if any>
If the twin cannot be made to match (e.g. the .va uses something the twin's
framework cannot express), make no edit and answer with
IMPOSSIBLE: <reason>
plus DIAGNOSIS / SUGGESTION lines.

Round {round} of {rounds}. The failed tests are in the UNTRUSTED CONTEXT below."""


class DeckChanged(RuntimeError):
    """The twin's edit changed a Spectre netlist: saved results do not apply."""


# ------------------------------------------------------------------ helpers
def twin_file(key, workspace):
    """Path of the twins/ file that defines `key` (None for built-ins)."""
    from ..twin_loader import load_twins
    for path, payload, _err in load_twins(os.path.join(workspace, "twins")):
        if payload is not None and payload[1].get("key") == key:
            return Path(path)
        if payload is None and Path(path).stem == key:
            return Path(path)
    return None


SOURCE_SUFFIXES = {".py", ".va", ".vams", ".scs", ".json", ".md", ".txt", ".toml",
                   ".cfg", ".ini", ".csv"}
REMOVE_NEW_IN = ("twins", "verilog")     # new files there are executed / referenced
KEEP_MAX = 4_000_000                     # bytes kept per file for a restore


def _digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _files(workspace):
    for d in SNAPSHOT_DIRS:
        root = Path(workspace) / d
        if root.is_dir():
            for p in root.rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    yield p


def snapshot(workspace, exclude):
    """Every file under SNAPSHOT_DIRS is recorded (so none looks 'new' later);
    the content of source-sized files is kept for a restore."""
    snap = {}
    for p in _files(workspace):
        if p == exclude:
            continue
        st = p.stat()
        keep = st.st_size <= KEEP_MAX and p.suffix.lower() in SOURCE_SUFFIXES
        snap[p] = ((p.read_bytes(), _digest(p)) if keep else None,
                   (st.st_size, st.st_mtime_ns))
    return snap


def enforce_snapshot(workspace, snap, exclude):
    """Undo the assistant's changes outside the twin file -> list of actions.
    Changed source files are restored; new source files under twins/ and
    verilog/ are removed (twins/ files are executed on load).  Nothing else
    is ever deleted: other new or changed files are only reported."""
    undone = []
    for p, (kept, stat) in snap.items():
        rel = p.relative_to(workspace)
        if not p.exists():
            if kept is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(kept[0])
                undone.append(f"restored deleted {rel}")
            else:
                undone.append(f"NOTE: {rel} was deleted (not restorable)")
            continue
        st = p.stat()
        if (st.st_size, st.st_mtime_ns) == stat:
            continue
        if kept is not None and _digest(p) != kept[1]:
            p.write_bytes(kept[0])
            undone.append(f"restored {rel}")
        elif kept is None:
            undone.append(f"NOTE: {rel} changed (left as is)")
    for p in _files(workspace):
        if p == exclude or p in snap:
            continue
        rel = p.relative_to(workspace)
        if rel.parts[0] in REMOVE_NEW_IN and p.suffix.lower() in SOURCE_SUFFIXES:
            p.unlink()
            undone.append(f"removed new file {rel}")
        else:
            undone.append(f"NOTE: new file {rel} (left in place)")
    return undone


def offline_runner(run_dir):
    """A validate() runner that returns the run's saved Spectre results,
    after checking each netlist is the one Spectre ran."""
    from .psf import load_waveform
    sdir = Path(run_dir) / "spectre"

    def runner(decks, va_path, out_dir):
        out = {}
        for name, text in decks.items():
            saved = sdir / f"{name}.scs"
            if not saved.exists():
                raise DeckChanged(f"the run has no saved netlist {name}.scs (re-run "
                                  f"the validation once to keep them)")
            if saved.read_text(encoding="utf-8") != str(text):
                raise DeckChanged(f"netlist {name} differs from the one Spectre ran")
            sig, _ = load_waveform(sdir / f"{name}.raw")
            out[name] = (sig, {"simulator": "spectre", "version": "(saved results)"})
        return out
    return runner


def _worst(run_dir, name, profile):
    """Where test `name` is worst: a few lines for the assistant."""
    from .validate import load_l1_data
    p = Path(run_dir) / f"l1_{name}.npz"
    if not p.exists():
        return []
    d, meta = load_l1_data(p)
    out = []
    if profile != "fefet":
        yr = np.interp(d["t_tw"], d["t_ref"], d["y_ref"])
        rng = float(np.nanmax(d["y_ref"]) - np.nanmin(d["y_ref"])) or 1.0
        e = (d["y_tw"] - yr) / rng
        k = int(np.nanargmax(np.abs(e)))
        out.append(f"largest waveform error {100 * e[k]:+.3g} % of range at t = "
                   f"{d['t_tw'][k]:.6g} s: twin {d['y_tw'][k]:.6g}, Spectre {yr[k]:.6g} "
                   f"({meta.get('label', '')})")
        drv = np.interp(d["t_tw"][k], d["drive_t"], d["drive_v"])
        out.append(f"  drive there: {drv:.6g} {'A' if meta.get('drive_kind') == 'current' else 'V'}")
    if "read_err_pct" in d:
        e = np.asarray(d["read_err_pct"], float)
        order = np.argsort(-np.abs(np.nan_to_num(e)))[:5]
        out.append("worst reads (time, Spectre G, twin G, error %):")
        for i in sorted(order):
            out.append(f"  {d['read_t'][i]:.6g} s  {d['g_ref'][i]:.6g}  "
                       f"{d['g_twin'][i]:.6g}  {e[i]:+.3g}")
    return out


def failure_report(run_dir, summary):
    """The failed tests as text (fenced as untrusted context for the agent)."""
    lines = [f"model {summary.get('module')}, twin {summary.get('twin')}, device type "
             f"{summary.get('profile')}, overall {summary.get('verdict')}"]
    conv = {c["stimulus"]: c for c in summary.get("layer2") or []}
    for t in summary.get("layer1") or []:
        name = t["stimulus"]
        lines.append(f"\n[{name}] {t.get('verdict')}: {t.get('description', '')}")
        for k in ("nrmse_pct", "max_err_pct", "read_g_max_pct", "read_g_mean_pct"):
            if k in t:
                lines.append(f"  {k} = {t[k]:.4g}")
        if t.get("rule"):
            lines.append(f"  rule: {t['rule']}")
        if t.get("note"):
            lines.append(f"  note: {t['note']}")
        c = conv.get(name, {})
        if c:
            lines.append(f"  Spectre converged at the reference step: "
                         f"{c.get('reference_converged')}")
        if t.get("verdict") != "PASS":
            lines += ["  " + s for s in _worst(run_dir, name, summary.get("profile"))]
    return "\n".join(lines)


def _tagged(text, tag):
    return [m.strip() for m in re.findall(rf"^\s*{tag}:\s*(.+)$", text or "", re.M)]


def _score(summary):
    """Lower is better: the worst error over the tests (for 'best attempt')."""
    worst = 0.0
    for t in summary.get("layer1") or []:
        for k in ("max_err_pct", "read_g_max_pct"):
            v = t.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                worst = max(worst, float(v))
    return worst


def own_checks(summary):
    """NeuroVAT's own suggestions from a failed run."""
    out = []
    conv = {c["stimulus"]: c for c in summary.get("layer2") or []}
    unconv = [n for n, c in conv.items() if not c.get("reference_converged")]
    if unconv:
        out.append(f"Spectre itself was not converged in {', '.join(unconv)}: re-run "
                   "with a finer step before blaming the twin.")
    tests = summary.get("layer1") or []
    reads_only = [t["stimulus"] for t in tests if t.get("verdict") != "PASS"
                  and t.get("nrmse_pct", 0) <= 1 and "read_g_max_pct" in t]
    if reads_only:
        out.append(f"{', '.join(reads_only)}: the waveform matches but the reads do not - "
                   "check the read path (read bias, the conductance definition, state "
                   "change during the read).")
    if any(t.get("verdict") == "INCONCLUSIVE" for t in tests):
        out.append("Some tests are INCONCLUSIVE (no usable reference): check the setup "
                   "(amplitudes, read bias) before fixing the twin.")
    return out


# ------------------------------------------------------------------ the loop
def fix_twin(run_dir, workspace, agent, rounds=ROUNDS, log=print, stop=None):
    """-> dict(verdict 'FIXED' | 'NOT FIXED' | 'NOT POSSIBLE', rounds, out_dir,
    changes, diagnosis, suggestions, twin_file, best_attempt)."""
    from .validate import validate
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    key = (summary.get("twin") or "").split(" (")[0]
    tf = twin_file(key, workspace)
    if tf is None:
        raise RuntimeError(f"twin {key!r} is not a twins/ file (built-in models are not "
                           "fixed automatically)")
    if not list((run_dir / "spectre").glob("*.scs")):
        raise RuntimeError("this run has no saved netlists: run the validation again "
                           "(runs from now on keep them), then fix")
    va = summary["va"]
    setup = summary.get("setup") or {}
    fix_dir = run_dir / "fix"
    fix_dir.mkdir(exist_ok=True)
    original = tf.read_bytes()
    (fix_dir / f"original_{tf.name}").write_bytes(original)
    from .deck import va_interface
    _m, _p, va_params, _b = va_interface(Path(va).read_text(encoding="utf-8",
                                                            errors="replace"))
    va_text = Path(va).read_text(encoding="utf-8", errors="replace")[:30000]
    report_text = failure_report(run_dir, summary)
    changes, diagnosis, suggestions = [], [], []
    best = (_score(summary), None)
    for r in range(1, rounds + 1):
        if stop and stop():
            break
        log(f"round {r}/{rounds}: the assistant edits {tf.name} ...")
        snap = snapshot(workspace, tf)
        prompt = PROMPT.format(twin_file=str(tf), shared=", ".join(sorted(va_params)) or "-",
                               round=r, rounds=rounds)
        ctx = (f"--- NeuroVAT validation result ---\n{report_text}\n\n"
               f"--- {Path(va).name} (the reference) ---\n{va_text}")
        res = agent.send(prompt, context=ctx, allow_edits=True, allow_bash=False,
                         timeout=900)
        undone = enforce_snapshot(workspace, snap, tf)
        if undone:
            log("  undone (outside the twin file): " + "; ".join(undone))
        if not res.get("ok"):
            log(f"  the assistant failed: {res.get('error')}")
            break
        text = res.get("text", "")
        changes += [f"round {r}: {c}" for c in _tagged(text, "CHANGE")]
        diagnosis = _tagged(text, "DIAGNOSIS") or diagnosis
        suggestions = _tagged(text, "SUGGESTION") or suggestions
        for c in _tagged(text, "CHANGE"):
            log(f"  change: {c}")
        if _tagged(text, "IMPOSSIBLE"):
            reason = _tagged(text, "IMPOSSIBLE")[0]
            log(f"  the assistant: not possible - {reason}")
            tf.write_bytes(original)
            return _result("NOT POSSIBLE", r, None, changes, [reason] + diagnosis,
                           suggestions + own_checks(summary), tf, best[1])
        attempt = fix_dir / f"round{r}_{tf.name}"
        shutil.copyfile(tf, attempt)
        if tf.read_bytes() == original and r == 1:
            log("  the assistant made no edit")
        out = fix_dir / f"round{r}"
        log(f"  re-scoring against the saved Spectre results (no Spectre run) ...")
        try:
            rep, _ = validate(va, workspace=workspace, out_dir=out,
                              runner=offline_runner(run_dir), replay=False,
                              open_report=False, answers=setup,
                              log=lambda m: log("    " + m))
        except DeckChanged as e:
            log(f"  rejected: {e} (a parameter shared with the .va changed?)")
            report_text = (f"YOUR LAST EDIT WAS REJECTED: {e}. A default that goes "
                           f"into the Spectre netlist changed; keep those defaults.\n\n"
                           + report_text)
            continue
        except Exception as e:               # noqa: BLE001 - the twin may be broken
            log(f"  the edited twin does not run: {type(e).__name__}: {e}")
            report_text = (f"YOUR LAST EDIT BROKE THE TWIN: {type(e).__name__}: {e}\n\n"
                           + report_text)
            continue
        new = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        log(f"  round {r}: {rep['verdict']}  (worst error {_score(new):.3g} %)")
        if rep["verdict"] == "PASS":
            return _result("FIXED", r, out, changes, diagnosis, suggestions, tf, attempt)
        if _score(new) < best[0]:
            best = (_score(new), attempt)
        summary, report_text = new, failure_report(out, new)
    tf.write_bytes(original)
    keep = None
    if best[1] is not None:
        keep = fix_dir / f"best_attempt_{tf.name}"
        shutil.copyfile(best[1], keep)
    log(f"not fixed after {rounds} rounds: the original twin is restored"
        + (f"; best attempt kept as {keep}" if keep else ""))
    return _result("NOT FIXED", rounds, None, changes, diagnosis,
                   suggestions + own_checks(summary), tf, keep)


def _result(verdict, rounds, out, changes, diagnosis, suggestions, tf, best):
    seen, sug = set(), []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            sug.append(s)
    return {"verdict": verdict, "rounds": rounds, "out_dir": str(out) if out else None,
            "changes": changes, "diagnosis": diagnosis, "suggestions": sug,
            "twin_file": str(tf), "best_attempt": str(best) if best else None,
            "time": time.strftime("%Y-%m-%d %H:%M")}
