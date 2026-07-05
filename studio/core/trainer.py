"""Adapter — routes Studio's train_net / probe_lif to the REAL vatester.neuro
crossbar+LIF trainer and LIF probe (the same code the Dear PyGui app drives in
app.py::_nt_train_worker / _nt_probe_neuron). No physics is reimplemented here:
this only marshals the web's flat arg dict into neuro's dataclasses, runs it,
and packages JSON. When the parent repo isn't importable the web UI keeps its
in-page JS demo as the offline fallback (bridge returns {"engine": null}).

Bounded on purpose (epochs/grid/outputs capped) — the stdlib server runs this
synchronously on the request thread.
"""

from . import engine as _engine


def available():
    """True when both ecfet and vatester.neuro are importable (in-repo)."""
    if _engine._load() is None:
        return False
    try:
        import numpy  # noqa: F401
        from vatester import neuro  # noqa: F401
        return True
    except Exception:
        return False


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _device_factory(ecfet, device):
    cls = {"v2": ecfet.EcfetV2, "v3": ecfet.EcfetV3, "fefet": ecfet.FeFET}.get(device, ecfet.EcfetV2)
    pcls = {"v2": ecfet.V2Params, "v3": ecfet.V3Params, "fefet": ecfet.FeFETParams}.get(device, ecfet.V2Params)
    kind = "voltage" if device == "fefet" else "current"

    def make():
        p = pcls()
        # silence the device's intrinsic STDP lock-in — in the crossbar the
        # NETWORK rule decides plasticity direction (matches app._nt_device_factory).
        for silent in ("A_stdp", "A_stdp_V"):
            if hasattr(p, silent):
                setattr(p, silent, 0.0)
        return cls(p)

    return make, kind


# Loaded external dataset (MNIST / URL / uploaded), encoded to Trainer patterns.
# Bounded (small grid + few classes) so device-crossbar training stays fast.
_DATASET = {"loaded": False}


def load_dataset(kind="mnist", url="", res=8, invert=False, per_class=10, n_classes=6, **_):
    """Really download/decode a dataset (vatester.datasets) and encode it to
    input-pattern banks the trainer uses. Contract: {ok, kind, res, n_classes,
    n_train, n_test, class_names}."""
    import os as _os
    from vatester import datasets as nds
    if _engine._load() is None:
        raise RuntimeError("engine unavailable")
    cache = _os.path.join(_engine.repo_root(), "datasets", "_studio_cache")
    res = int(_clamp(res, 5, 10, 8))
    n_classes = int(_clamp(n_classes, 2, 6, 6))
    per_class = int(_clamp(per_class, 4, 24, 10))
    if kind == "file" and _.get("data"):
        import base64
        _os.makedirs(cache, exist_ok=True)
        name = _os.path.basename(str(_.get("name", "upload.dat"))) or "upload.dat"
        path = _os.path.join(cache, "_upload_" + name)
        with open(path, "wb") as f:
            f.write(base64.b64decode(_["data"]))
        images, labels = nds.load_any(path)
    elif kind == "url" and url:
        path = nds.download_url(str(url), cache)
        images, labels = nds.load_any(path)
    else:
        kind = "mnist"
        images, labels = nds.load_npz(nds.download_mnist(cache))
    tr_p, tr_t, te_p, te_t, names = nds.to_patterns_split(
        images, labels, res, res, train_per_class=per_class, test_per_class=8,
        n_classes=n_classes, invert=bool(invert))
    _DATASET.clear()
    _DATASET.update(loaded=True, kind=kind, res=res, names=names,
                    train=(tr_p, tr_t), test=(te_p, te_t))
    return {"ok": True, "kind": kind, "res": res, "n_classes": len(names),
            "n_train": len(tr_p), "n_test": len(te_p), "class_names": names}


def clear_dataset(**_):
    _DATASET.clear(); _DATASET["loaded"] = False
    return {"ok": True}


def _confusion_payload(tr, neuro, n_out):
    """Real confusion matrix + per-class P/R/F1 from the trained network."""
    import numpy as np
    yt, yp = tr.evaluate(use_test=bool(tr.test_patterns), max_n=None)
    if not yt:
        yt, yp = tr.evaluate(use_test=False, max_n=None)
    cm = neuro.confusion(yt, yp, n_out)
    m = neuro.prf1(cm)
    rows = []
    for k in range(n_out):
        rows.append({"name": (tr.class_names[k] if k < len(tr.class_names) else str(k)),
                     "prec": round(float(m["prec"][k]), 3),
                     "rec": round(float(m["rec"][k]), 3),
                     "f1": round(float(m["f1"][k]), 3),
                     "support": int(m["support"][k])})
    return {"matrix": cm.tolist(), "rows": rows,
            "acc": round(float(m["acc"]) * 100, 2),
            "macro_f1": round(float(m["macro_f1"]), 3),
            "class_names": [r["name"] for r in rows]}


def train_net(args=None):
    """Contract: -> {engine, epochs:[{epoch,train_acc,test_acc}], weights_uS,
    n_out, grid, final_train_acc, device}. Raises if neuro is unavailable so the
    bridge can fall back to the JS demo."""
    import numpy as np
    from vatester import neuro
    ecfet = _engine._load()
    if ecfet is None:
        raise RuntimeError("real engine unavailable")

    import math
    a = args or {}
    device = a.get("device", "v2")
    make, kind = _device_factory(ecfet, device)

    # map the UI 'learning rule' label onto what neuro actually implements
    # (learn_rule in {stdp, surrogate} x mode in {supervised, unsupervised}):
    #   STDP (device-local)/STDP(surrogate-trace) -> stdp, supervised
    #   surrogate/backprop(STE)/R-STDP(reward~grad) -> surrogate, supervised
    #   Hebbian                                      -> stdp, unsupervised
    rule_lbl = (a.get("rule") or "").lower()
    if "hebб".replace("б", "b") in rule_lbl:      # 'hebbian'
        learn_rule, forced_mode = "stdp", "unsupervised"
    elif "surrogate" in rule_lbl or "backprop" in rule_lbl or "ste" in rule_lbl or "r-stdp" in rule_lbl or "reward" in rule_lbl:
        learn_rule, forced_mode = "surrogate", None
    else:
        learn_rule, forced_mode = "stdp", None
    surrogate = learn_rule == "surrogate"

    n_out = int(_clamp(a.get("outputs", 4), 2, 6, 4))
    epochs = int(_clamp(a.get("epochs", 20), 2, 40, 20))
    seed = int(_clamp(a.get("seed", 1), 1, 1e6, 1))
    grid_h = int(_clamp(a.get("grid_h", 5), 3, 12, 5))
    grid_w = int(_clamp(a.get("grid_w", 5), 3, 12, 5))
    hidden = int(_clamp(a.get("hidden", 0), 0, 16, 0))
    hidden_layers = (hidden,) if hidden >= 1 else ()
    pattern_set = a.get("patterns", "bars") or "bars"
    encoding = a.get("encoding", "rate") or "rate"
    mode = forced_mode or ("unsupervised" if str(a.get("mode", "supervised")).lower().startswith("unsup") else "supervised")
    # a loaded external dataset overrides the built-in pattern set + grid/outputs
    use_ds = bool(a.get("use_dataset")) and _DATASET.get("loaded")
    if use_ds:
        grid_h = grid_w = _DATASET["res"]
        n_out = len(_DATASET["names"])
    amp_si = 1.0 if device == "fefet" else 1e-12   # pA -> A for ECFET, V for FeFET
    pot = _clamp(a.get("pot_amp", 170), 1, 1e6, 170) * amp_si
    dep = _clamp(a.get("dep_amp", 170), 1, 1e6, 170) * amp_si

    N = neuro.NeuronParams(
        tau_m=_clamp(a.get("tau_m", 20), 1, 200, 20),
        v_threshold=_clamp(a.get("vth", 1.0), 0.1, 5, 1.0),
        theta_plus=_clamp(a.get("theta", 0.06), 0.0, 0.5, 0.06),
        teacher=_clamp(a.get("teacher", 1.4), 0.2, 5, 1.4),
        epsp_gain=_clamp(a.get("epsp", 11.0), 1, 40, 11.0),
        inhibition=_clamp(a.get("wta", 0.9), 0.0, 3, 0.9))
    S = neuro.STDPParams(
        a_plus=_clamp(a.get("a_plus", 1.0), 0.05, 5, 1.0),
        a_minus=_clamp(a.get("a_minus", 1.0), 0.05, 5, 1.0),
        tau_pre=_clamp(a.get("tau_pre", 20), 1, 200, 20),
        offset=_clamp(a.get("split", 0.5), 0.05, 0.95, 0.25),
        pot_amp=pot, dep_amp=dep,
        pulse_width=_clamp(a.get("pulse_w", 10), 1, 100, 10) * 1e-3)
    cfg = neuro.NetConfig(
        grid_h=grid_h, grid_w=grid_w, n_out=n_out, mode=mode, learn_rule=learn_rule,
        hidden_layers=hidden_layers, present_ms=_clamp(a.get("present_ms", 120), 20, 240, 120),
        dt_ms=1.0, seed=seed, pattern_set=pattern_set, encoding=encoding,
        input_noise=_clamp(a.get("noise", 0.05), 0.0, 0.5, 0.05))

    if use_ds:
        tr_p, tr_t = _DATASET["train"]
        tr = neuro.Trainer(make, kind, N, S, cfg, patterns=tr_p, targets=tr_t)
        te_p, te_t = _DATASET["test"]
        tr.test_patterns, tr.test_targets = te_p, te_t
        tr.class_names = list(_DATASET["names"])
    else:
        tr = neuro.Trainer(make, kind, N, S, cfg)
    rng = np.random.default_rng(seed + 777)
    curve = []
    hist = []                     # per-epoch device conductances (weight transients)
    best_acc, best_w = -1.0, None
    for e in range(epochs):
        if surrogate:                          # anneal LR 1.0 -> 0.15 (matches desktop)
            frac = e / max(epochs - 1, 1)
            tr.lr_scale = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * frac))
        for k in rng.permutation(len(tr.patterns)):
            k = int(k)
            tr.train_step(tr.patterns[k][1], tr.target_of[k])
        tr_acc = tr.eval_accuracy(use_test=False, max_n=40)
        te_acc = tr.eval_accuracy(use_test=True, max_n=40) if tr.test_patterns else tr.eval_accuracy(use_test=False, max_n=40)
        curve.append({"epoch": e + 1, "train_acc": round(tr_acc * 100, 2),
                      "test_acc": round(te_acc * 100, 2)})
        Wc = tr.weights_uS(0)
        hist.append([round(float(x), 1) for row in Wc for x in row])   # flat per epoch
        if surrogate and tr_acc > best_acc + 1e-9:
            best_acc, best_w = tr_acc, tr.capture_weights()
    # surrogate oscillates around the solution — restore the best checkpoint
    if surrogate and best_w is not None:
        tr.restore_weights(best_w)

    W = tr.weights_uS(0)          # (n_out x n_in) real device conductances (uS)
    conf = _confusion_payload(tr, neuro, n_out)   # REAL confusion matrix + P/R/F1
    return {"engine": "neuro", "device": device, "n_out": n_out, "rule": learn_rule,
            "mode": mode, "grid": [cfg.grid_h, cfg.grid_w],
            "dataset": (_DATASET.get("kind") if use_ds else pattern_set),
            "epochs": curve,
            "weights_uS": [[round(float(x), 2) for x in row] for row in W],
            "weights_hist": hist,           # [epoch][synapse] real G(uS) trajectory
            "confusion": conf,              # real matrix, per-class P/R/F1, macro-F1
            "final_train_acc": round(max(best_acc * 100, curve[-1]["train_acc"]) if surrogate else curve[-1]["train_acc"], 2) if curve else 0.0}


def eval_net(args=None):
    """Real held-out evaluation of a freshly-built+trained net (Test button).
    Bounded re-run so the number reflects the actual neuro network, not a client
    toy classifier. Returns the confusion payload."""
    r = train_net(dict(args or {}, epochs=int(_clamp((args or {}).get("epochs", 12), 2, 30, 12))))
    return {"ok": True, "engine": "neuro", "confusion": r.get("confusion"),
            "test_acc": r.get("epochs", [{}])[-1].get("test_acc") if r.get("epochs") else None,
            "final_train_acc": r.get("final_train_acc")}


def probe_lif(args=None):
    """Real LIF probe. mode step/pulse/ramp drive the membrane with a current;
    mode 'epsp' drives it with an EXCITATORY SPIKE TRAIN (probe_lif_psp) so you
    see EPSP summation, and the f-I curve becomes rate-vs-input-rate.
    Contract: {engine, t, V, th, spikes, in_spikes?, rate_hz, n_spikes,
               fI:{drive,rate,xlabel}, mode}."""
    import numpy as np
    from vatester import neuro
    a = args or {}
    N = neuro.NeuronParams(
        tau_m=_clamp(a.get("tau_m", 20), 1, 200, 20),
        v_threshold=_clamp(a.get("vth", 1.0), 0.1, 5, 1.0),
        t_refractory=_clamp(a.get("refrac", 5), 0, 50, 5),
        tau_syn=_clamp(a.get("tau_syn", 8), 1, 100, 8),
        theta_plus=_clamp(a.get("theta", 0.06), 0.0, 0.5, 0.06),
        epsp_gain=_clamp(a.get("epsp", 11.0), 0.5, 40, 11.0))
    inj = _clamp(a.get("iInj", 1.6), 0.0, 10, 1.6)   # x rheobase (v_threshold units)
    mode = a.get("mode", "step")
    present_ms, dt = 240.0, 0.5

    if mode == "epsp":
        in_hz = max(1.0, inj * 40.0)             # slider -> input spike rate
        epsp_w = N.epsp_gain / 11.0 * 2.0        # PSP height per input spike (tuned to fire)
        t, V, TH, outs, ins, psp = neuro.probe_lif_psp(N, in_hz, epsp_w, present_ms, dt)
        hz, rates = neuro.fi_curve_psp(N, epsp_w, present_ms, dt, max(40.0, inj * 50.0))
        rate = len(outs) / (present_ms * 1e-3)
        return {"engine": "neuro", "mode": "epsp", "t": t.tolist(), "V": V.tolist(),
                "th": TH.tolist(), "spikes": [round(float(s), 2) for s in outs],
                "in_spikes": [round(float(s), 2) for s in ins],
                "rate_hz": round(rate, 1), "n_spikes": len(outs),
                "fI": {"drive": hz.tolist(), "rate": rates.tolist(), "xlabel": "input rate (Hz)"}}

    n = int(present_ms / dt)
    drive = inj * N.v_threshold
    if mode == "pulse":
        arr = np.zeros(n); arr[n // 4:3 * n // 4] = drive
    elif mode == "ramp":
        arr = np.linspace(0.0, drive, n)
    else:
        arr = np.full(n, drive)
    t, V, TH, spikes = neuro.probe_lif_wave(N, arr, dt)
    dr, rates = neuro.fi_curve(N, present_ms, dt, max(3.0, inj * 1.3) * N.v_threshold)
    rate = len(spikes) / (present_ms * 1e-3)
    return {"engine": "neuro", "mode": mode, "t": t.tolist(), "V": V.tolist(), "th": TH.tolist(),
            "spikes": [round(float(s), 2) for s in spikes],
            "rate_hz": round(rate, 1), "n_spikes": len(spikes),
            "fI": {"drive": (dr / max(N.v_threshold, 1e-9)).tolist(),
                   "rate": rates.tolist(), "xlabel": "I_inj (× rheobase)"}}
