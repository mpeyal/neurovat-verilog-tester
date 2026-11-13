"""Robustness sweep of a stream network (GUI-free, no DearPyGui).

After learning, test the network (STDP off) while ONE input factor at a time
departs from its nominal value - spike-time jitter, missing (deleted) spikes,
extra (noise) spikes - the robustness panels of Masquelier, Guyonneau &
Thorpe (2008) (jitter 1 / 3 ms, 10 / 30 % deletion, 5 / 26 Hz extra spikes;
their levels, our network). The same levels are run on an UNTRAINED copy of
the network (same parameters and frozen patterns, start weights): the
Test-before-Train baseline, so a curve shows what training bought.

* Every level runs ``Session.run(seconds, learn=False, stream_key=key)``:
  the threshold, the weights and the device stay as they are; a keyed run
  does not count as one of the session's runs, so the session is left as it
  was (its next ordinary run is the one it would have been). One key for
  every level and both networks: the trained and untrained networks see the
  same stream at each level, and pressing the sweep again gives identical
  numbers.
* ``noise_hz`` levels are the ABSOLUTE extra-spike rate per afferent (the
  stream's ``noise_hz``), not added to the nominal; ``bg_hz`` (images: the
  Poisson rate of the empty sections) stays nominal throughout.
  ``spike_deletion`` is a fraction (shown as %).
* Score: every pattern window and every other window of the WHOLE run (no
  learning happens, so nothing needs a burn-in), with the hit rule the
  Trainer's live chart uses (:func:`stream_view.section_hit`): images - at
  least one output spike in the window; Poisson - exactly one spike of some
  neuron. A false window is any other window (empty, background or a
  distractor) with at least one output spike. Balanced accuracy = mean of
  the hit share and the share of quiet other windows. The engine's own
  per-neuron window counts (the last third of the run, "fires at least
  once") are kept alongside as ``engine_windows``.
* Device sessions: ``learn=False`` sends no write pulse, so the device does
  not change (conductances are only read). The read energy of the sweep is
  NOT added to the session's energy meter unless ``energy=True``. An
  untrained device baseline needs a fresh device, which only the caller can
  build: pass ``untrained=`` (a Session or a callable that returns one), else
  that row is skipped.

RETRAIN mode (``retrain=True``) asks a different question: not "how does the
trained network cope with a worse input" but "what does a network trained
UNDER that input learn". For every level a fresh copy (:func:`fresh_like`:
start weights, same patterns, threshold and parameters) is trained with STDP
on for ``train_seconds`` at that level, on the keyed training stream
``(train_key, r)`` - one key for every level, so only the factor differs and
a repeat sweep is identical - then tested at the same level, STDP off, on
the sweep's test stream ``key``. The threshold is held (never re-derived);
the untrained copy is tested as in recall. The nominal row is an
independent training on a different stream ("retrained at nominal"), so it
does not equal the user's network. ``repeats`` = N independent trainings
per level (keys ``(train_key, 0..N-1)``): the row holds the mean hit % /
false % (counts summed over the trainings) and their min-max. Each
training's final weights are kept (``row["weights"]``) so a level can be
replayed. The caller's session is never run: its weights, runs and stream
stay as they were. Device sessions are refused (a device copy cannot be
made).
"""
import math
import time

import numpy as np

from . import stream_view as sv

FACTORS = ("jitter_ms", "spike_deletion", "noise_hz")
LABELS = {"jitter_ms": ("jitter", "ms", 1.0),
          "spike_deletion": ("missing spikes", "%", 100.0),
          "noise_hz": ("extra spikes", "Hz", 1.0)}
# the per-run stream settings a sweep holds at nominal (Session.run changes)
STREAM_SETTINGS = ("jitter_ms", "spike_deletion", "noise_hz", "bg_hz",
                   "pattern_frac", "spacing")
ALIASES = {"jitter": "jitter_ms", "jitter_ms": "jitter_ms",
           "delete": "spike_deletion", "deletion": "spike_deletion",
           "missing": "spike_deletion", "spike_deletion": "spike_deletion",
           "noise": "noise_hz", "extra": "noise_hz", "noise_hz": "noise_hz"}
FIGURE_LEVELS = {"jitter_ms": (1.0, 3.0, 5.0),
                 "spike_deletion": (0.0, 0.1, 0.3, 0.5),
                 "noise_hz": (5.0, 26.0)}
SWEEP_KEY = 2008                 # default stream key (any fixed int)
TRAIN_KEY = 2009                 # retrain: training streams (TRAIN_KEY, r)
RECALL, RETRAIN = "recall", "retrain"
RETRAIN_NOMINAL = "retrained at nominal"
DEVICE_REFUSAL = ("retraining needs ideal weights: a device copy cannot be "
                  "made")


def factor_name(name):
    key = ALIASES.get(str(name).strip().lower())
    if key is None:
        raise ValueError(f"unknown sweep factor {name!r}: use one of "
                         f"{', '.join(FACTORS)}")
    return key


def default_levels(factor, nominal_value):
    """The figure's levels plus the nominal value, ascending, no repeats."""
    factor = factor_name(factor)
    vals = {round(float(v), 9) for v in FIGURE_LEVELS[factor]}
    vals.add(round(float(nominal_value), 9))
    return sorted(vals)


def nominal_settings(session, nominal=None):
    """The stream settings every level holds: the session's parameters,
    overridden by ``nominal`` (the per-run ``changes`` the network was
    trained / tested with - the GUI passes them per run, so ``session.P``
    alone holds only the defaults)."""
    P = session.P
    out = {k: getattr(P, k) for k in STREAM_SETTINGS}
    for k, v in (nominal or {}).items():
        if k not in STREAM_SETTINGS:      # threshold, weights ... stay fixed
            raise ValueError(f"{k!r} is not a stream setting a sweep holds "
                             f"(one of {', '.join(STREAM_SETTINGS)})")
        out[k] = v
    return out


def fresh_like(session):
    """An untrained copy of an ideal-synapse session: same parameters, the
    same frozen patterns (and pattern afferents), start weights. None for a
    device session (it needs a fresh device; see the module note)."""
    from neurovat.masquelier import Session
    if session.device is not None:
        return None
    kw = {k: v for k, v in session.kw.items() if k != "energy"}
    if session.given:
        return Session(session.P, patterns=session.copies,
                       distractors=session.distractors, **kw)
    u = Session(session.P, n_patterns=len(session.members), **kw)
    u.members = np.array(session.members, bool)
    u.copies = session.copies
    return u


def measure(session, seconds, changes, key=SWEEP_KEY, images=None,
            stop=None):
    """One recall run (STDP off, keyed stream). Returns the counts, or None
    when ``stop()`` became true during it."""
    images = bool(session.given if images is None else images)
    acc = {"pat": 0, "hit": 0, "oth": 0, "oth_hit": 0}

    def on_section(_t, info):
        if info["label"] >= 0:
            acc["pat"] += 1
            acc["hit"] += bool(sv.section_hit(info["spikes"], images))
        else:
            acc["oth"] += 1
            acc["oth_hit"] += bool(info["spikes"])
        return not (stop and stop())

    t0 = time.perf_counter()
    res = session.run(seconds, learn=False, on_section=on_section,
                      stream_key=key, **changes)
    if res.get("stopped"):
        return None
    hit = acc["hit"] / acc["pat"] if acc["pat"] else float("nan")
    false = acc["oth_hit"] / acc["oth"] if acc["oth"] else float("nan")
    return {"hit_pct": 100.0 * hit, "false_pct": 100.0 * false,
            "balanced_accuracy": 0.5 * (hit + 1.0 - false),
            "pattern_windows": acc["pat"], "hits": acc["hit"],
            "other_windows": acc["oth"], "false_windows": acc["oth_hit"],
            "post_spikes": int(res["post_spikes"]),
            "engine_windows": [dict(q["windows"]) for q in res["per_neuron"]],
            "wall_s": time.perf_counter() - t0}


def retrain_once(session, train_seconds, seconds, changes, key=SWEEP_KEY,
                 train_key=TRAIN_KEY, images=None, stop=None, on_phase=None):
    """Train a :func:`fresh_like` copy of ``session`` (STDP on) for
    ``train_seconds`` on the keyed training stream ``train_key`` under
    ``changes``, then test it there (STDP off, stream ``key``). ``session``
    is not run. Returns :func:`measure`'s dict plus ``weights`` (the copy's
    final weights) and ``train_post_spikes``, or None when ``stop()``
    became true. ``on_phase("training" | "testing")`` is called before each
    part."""
    if session.device is not None:
        raise ValueError(DEVICE_REFUSAL)
    copy = fresh_like(session)
    if on_phase:
        on_phase("training")
    t0 = time.perf_counter()
    res = copy.run(float(train_seconds), learn=True, stream_key=train_key,
                   on_section=lambda _t, _i: not (stop and stop()),
                   **changes)
    if res.get("stopped") or (stop and stop()):
        return None
    t_train = time.perf_counter() - t0
    if on_phase:
        on_phase("testing")
    m = measure(copy, seconds, changes, key, images, stop)
    if m is None:
        return None
    m["weights"] = np.array(copy.weights, float)
    m["train_post_spikes"] = int(res["post_spikes"])
    m["train_wall_s"] = t_train
    return m


def _pct(vals):
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if not v:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(v)), min(v), max(v)


def aggregate(runs):
    """One row cell of N retrainings (each :func:`measure`-shaped): hit % /
    false % / balanced accuracy are the means, the window counts and spikes
    are summed over the trainings (every training is tested on the same
    stream, so hits / windows is the mean hit share), plus ``n_runs`` and
    the min-max (``hit_min``, ``hit_max``, ``false_min``, ``false_max``).
    One run: that run's dict with the min-max fields added."""
    if len(runs) == 1:
        out = {k: v for k, v in runs[0].items() if k != "weights"}
    else:
        out = {k: int(sum(r[k] for r in runs)) for k in (
            "pattern_windows", "hits", "other_windows", "false_windows",
            "post_spikes")}
        out["balanced_accuracy"] = _pct(
            r["balanced_accuracy"] for r in runs)[0]
        out["wall_s"] = float(sum(r["wall_s"] for r in runs))
    hit, hlo, hhi = _pct(r["hit_pct"] for r in runs)
    false, flo, fhi = _pct(r["false_pct"] for r in runs)
    out.update(hit_pct=hit, false_pct=false, hit_min=hlo, hit_max=hhi,
               false_min=flo, false_max=fhi, n_runs=len(runs))
    return out


def _plan(session, factors, nom):
    if factors is None:
        factors = FACTORS
    if isinstance(factors, dict):
        items = [(factor_name(f), [float(v) for v in lv] if lv is not None
                  else None) for f, lv in factors.items()]
    else:
        items = [(factor_name(f), None) for f in factors]
    plan = []
    for f, lv in items:
        levels = default_levels(f, nom[f]) if lv is None else list(
            dict.fromkeys(lv))
        if f == "spike_deletion" and any(not 0.0 <= v < 1.0 for v in levels):
            raise ValueError("spike_deletion levels are fractions in [0, 1)")
        if any(v < 0.0 for v in levels):
            raise ValueError(f"{f} levels must be >= 0")
        plan.append((f, levels))
    return plan


def sweep(session, factors=None, seconds=30.0, nominal=None, on_progress=None,
          stop=None, untrained=None, key=SWEEP_KEY, images=None,
          energy=False, retrain=False, train_seconds=None, repeats=1,
          train_key=TRAIN_KEY):
    """Test ``session`` (STDP off) at each level of each factor, one factor
    at a time, the others at ``nominal`` (see :func:`nominal_settings`).

    ``retrain=True``: for each level train ``repeats`` fresh copies for
    ``train_seconds`` (required) at that level on the training streams
    ``(train_key, r)``, then test them (see the module note); ``session``
    itself is not run. ValueError for a device session. A level stopped
    half-way (also during training) is dropped. Rows then also hold
    ``repeats`` (each training's test dict) and ``weights`` (each
    training's final weights); ``trained`` is their :func:`aggregate`.

    ``factors``: None (jitter_ms, spike_deletion, noise_hz at
    :func:`default_levels`), a list of factor names, or {name: levels}.
    ``untrained``: None = :func:`fresh_like` (skipped for a device session),
    False = no baseline, a Session, or a callable returning one.
    ``on_progress(i, n, text)`` is called before each run (i of n done);
    ``stop()`` is checked during and between runs - the result then holds
    the levels finished so far and ``stopped`` is True. A level stopped
    half-way is dropped.

    Returns {"factors": [{"factor", "label", "unit", "scale", "rows": [{
    "level", "nominal", "trained", "untrained"}]}], "nominal", "seconds",
    "key", "images", "rule", "untrained_note", "stopped", "wall_s", "mode"}
    (retrain adds "train_seconds", "repeats", "train_key"); each run's dict
    is :func:`measure`'s."""
    t_start = time.perf_counter()
    images = bool(session.given if images is None else images)
    retrain = bool(retrain)
    if retrain:
        if session.device is not None:
            raise ValueError(DEVICE_REFUSAL)
        if train_seconds is None:
            raise ValueError("retrain needs train_seconds (the GUI passes the "
                             "seconds the network was trained)")
        train_seconds = float(train_seconds)
        if not math.isfinite(train_seconds) or train_seconds <= 0.0:
            raise ValueError("train_seconds must be > 0")
        repeats = int(repeats)
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
    nom = nominal_settings(session, nominal)
    plan = _plan(session, factors, nom)
    note = "fresh copy at the start weights, same patterns"
    if untrained is False:
        base, note = None, "not run"
    elif untrained is None:
        base = None if session.device is not None else "auto"
        if base is None:
            note = ("skipped: a device network needs a fresh device "
                    "(pass untrained=)")
    elif callable(untrained) and not hasattr(untrained, "run"):
        base, note = untrained(), "given by the caller"
    else:
        base, note = untrained, "given by the caller"
    per_level = (2 * repeats if retrain else 1) + (base is not None)
    n_runs = sum(len(lv) for _f, lv in plan) * per_level
    out = {"factors": [], "nominal": dict(nom), "seconds": float(seconds),
           "key": key, "images": images, "rule": sv.hit_rule(images),
           "untrained_note": note, "stopped": False,
           "mode": RETRAIN if retrain else RECALL}
    if retrain:
        out.update(train_seconds=train_seconds, repeats=repeats,
                   train_key=train_key)
    saved_kw = session.kw
    if not energy and "energy" in session.kw:
        session.kw = {k: v for k, v in session.kw.items() if k != "energy"}
    done = 0
    try:
        for f, levels in plan:
            label, unit, scale = LABELS[f]
            block = {"factor": f, "label": label, "unit": unit,
                     "scale": scale, "rows": []}
            out["factors"].append(block)
            for v in levels:
                if stop and stop():
                    out["stopped"] = True
                    return out
                changes = dict(nom)
                changes[f] = v
                lvl = f"{label} {v * scale:g} {unit}"
                extra = {}
                if retrain:
                    runs = []
                    for r in range(repeats):
                        rep = (f" (training {r + 1} of {repeats})"
                               if repeats > 1 else "")

                        def phase(what, rep=rep):
                            nonlocal done
                            if what == "testing":
                                done += 1          # the training is done
                            if on_progress:
                                on_progress(done, n_runs,
                                            f"{what}...: {lvl}{rep}")
                        one = retrain_once(session, train_seconds, seconds,
                                           changes, key, (train_key, r),
                                           images, stop, phase)
                        if one is None:
                            out["stopped"] = True
                            return out
                        done += 1
                        runs.append(one)
                    tr = aggregate(runs)
                    extra = {"repeats": [{k: x for k, x in one.items()
                                          if k != "weights"} for one in runs],
                             "weights": [one["weights"] for one in runs]}
                else:
                    if on_progress:
                        on_progress(done, n_runs, f"{lvl}: trained")
                    tr = measure(session, seconds, changes, key, images, stop)
                    if tr is None:
                        out["stopped"] = True
                        return out
                    done += 1
                ut = None
                if base is not None:
                    if base == "auto":
                        base = fresh_like(session)
                    if on_progress:
                        on_progress(done, n_runs, (f"testing...: {lvl} "
                                    "untrained") if retrain else
                                    f"{lvl}: untrained")
                    ut = measure(base, seconds, changes, key, images, stop)
                    if ut is None:
                        out["stopped"] = True
                        return out
                    done += 1
                block["rows"].append({
                    "level": float(v),
                    "nominal": math.isclose(float(v), float(nom[f]),
                                            abs_tol=1e-12),
                    "trained": tr, "untrained": ut, **extra})
        if on_progress:
            on_progress(done, n_runs, "done")
        return out
    finally:
        session.kw = saved_kw
        out["wall_s"] = time.perf_counter() - t_start


def format_table(result):
    """Plain-text table of a sweep result (one line per level)."""
    retrain = result.get("mode") == RETRAIN
    n_rep = int(result.get("repeats") or 1)
    if retrain:
        lines = [f"robustness sweep, RETRAIN at each level: a fresh copy "
                 f"(start weights, same patterns, threshold held) trained "
                 f"{result['train_seconds']:g} s with STDP on at the level "
                 f"(training stream key ({result['train_key']}, r)), then "
                 f"tested {result['seconds']:g} s STDP off at the same level "
                 f"on stream key {result['key']}; "
                 + ("one training run per level" if n_rep == 1 else
                    f"{n_rep} training runs per level (counts summed, "
                    "hit % / false % = mean [min-max])")
                 + f"; hit = {result['rule']}; false = other window with "
                 ">= 1 output spike; whole run scored"]
    else:
        lines = [f"robustness sweep (recall): {result['seconds']:g} s per "
                 f"level, STDP off, stream key {result['key']}; hit = "
                 f"{result['rule']}; false = other window with >= 1 output "
                 "spike; whole run scored"]
    lines += ["nominal: " + ", ".join(f"{k} {v}" for k, v in
                                      result["nominal"].items()),
              f"untrained baseline: {result['untrained_note']}"]

    def cell(r):
        if r is None:
            return f"{'-':>9s} {'-':>9s} {'-':>5s} {'-':>6s}"
        return (f"{r['hits']:>4d}/{r['pattern_windows']:<4d} "
                f"{r['false_windows']:>4d}/{r['other_windows']:<4d} "
                f"{r['balanced_accuracy']:5.2f} {r['post_spikes']:>6d}")

    def spread(r, what):
        if r is None or f"{what}_min" not in r:
            return ""
        return (f"{r[f'{what}_pct']:5.1f} [{r[f'{what}_min']:.0f}-"
                f"{r[f'{what}_max']:.0f}]")

    who = "RETRAINED" if retrain else "TRAINED"
    head = (f"{'factor':15s} {'level':>8s}  | {who:>9s} {'false':>9s} "
            f"{'bal':>5s} {'spikes':>6s} | {'UNTRAINED':>9s} {'false':>9s} "
            f"{'bal':>5s} {'spikes':>6s}")
    if retrain and n_rep > 1:
        head += f" | {'hit %':>15s} {'false %':>15s}"
    lines += [head, "-" * len(head)]
    for b in result["factors"]:
        for row in b["rows"]:
            lev = f"{row['level'] * b['scale']:g} {b['unit']}"
            mark = "*" if row["nominal"] else " "
            line = (f"{b['label']:15s} {lev:>8s}{mark} | "
                    f"{cell(row['trained'])} | {cell(row['untrained'])}")
            if retrain and n_rep > 1:
                line += (f" | {spread(row['trained'], 'hit'):>15s} "
                         f"{spread(row['trained'], 'false'):>15s}")
            lines.append(line)
    if retrain:
        lines.append(f"* = {RETRAIN_NOMINAL}: an independent training on "
                     "another stream, so it is not the network you trained")
    else:
        lines.append("* = nominal level")
    lines[-1] += ("; hit (first column) and false as windows answered / "
                  "windows; bal = balanced accuracy; spikes = output spikes "
                  "of the test run" + (" (summed over the trainings)"
                                       if retrain and n_rep > 1 else ""))
    if result.get("stopped"):
        lines.append("STOPPED: partial result")
    if "wall_s" in result:
        lines.append(f"wall time {result['wall_s']:.1f} s")
    return "\n".join(lines)
