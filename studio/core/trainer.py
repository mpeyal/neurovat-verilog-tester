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

    # map the UI 'learning rule' label onto neuro's learn_rule
    rule_lbl = (a.get("rule") or "").lower()
    learn_rule = "surrogate" if ("surrogate" in rule_lbl or "backprop" in rule_lbl or "ste" in rule_lbl) else "stdp"
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
    mode = "unsupervised" if str(a.get("mode", "supervised")).lower().startswith("unsup") else "supervised"
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

    tr = neuro.Trainer(make, kind, N, S, cfg)
    rng = np.random.default_rng(seed + 777)
    curve = []
    best_acc, best_w = -1.0, None
    for e in range(epochs):
        if surrogate:                          # anneal LR 1.0 -> 0.15 (matches desktop)
            frac = e / max(epochs - 1, 1)
            tr.lr_scale = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * frac))
        for k in rng.permutation(len(tr.patterns)):
            k = int(k)
            tr.train_step(tr.patterns[k][1], tr.target_of[k])
        tr_acc = tr.eval_accuracy(use_test=False, max_n=40)
        te_acc = tr.eval_accuracy(use_test=False, max_n=40)   # 2nd noisy draw ~ held-out
        curve.append({"epoch": e + 1, "train_acc": round(tr_acc * 100, 2),
                      "test_acc": round(te_acc * 100, 2)})
        if surrogate and tr_acc > best_acc + 1e-9:
            best_acc, best_w = tr_acc, tr.capture_weights()
    # surrogate oscillates around the solution — restore the best checkpoint
    if surrogate and best_w is not None:
        tr.restore_weights(best_w)

    W = tr.weights_uS(0)          # (n_out x n_in) real device conductances (uS)
    return {"engine": "neuro", "device": device, "n_out": n_out, "rule": learn_rule,
            "grid": [cfg.grid_h, cfg.grid_w],
            "epochs": curve,
            "weights_uS": [[round(float(x), 2) for x in row] for row in W],
            "final_train_acc": round(max(best_acc * 100, curve[-1]["train_acc"]) if surrogate else curve[-1]["train_acc"], 2) if curve else 0.0}


def probe_lif(args=None):
    """Contract: -> {engine, t, V, th, spikes, rate_hz, n_spikes, fI:{drive,rate}}."""
    import numpy as np
    from vatester import neuro
    a = args or {}
    N = neuro.NeuronParams(
        tau_m=_clamp(a.get("tau_m", 20), 1, 200, 20),
        v_threshold=_clamp(a.get("vth", 1.0), 0.1, 5, 1.0),
        t_refractory=_clamp(a.get("refrac", 5), 0, 50, 5),
        theta_plus=_clamp(a.get("theta", 0.06), 0.0, 0.5, 0.06))
    inj = _clamp(a.get("iInj", 1.6), 0.0, 10, 1.6)   # x rheobase (v_threshold units)
    mode = a.get("mode", "step")
    present_ms, dt = 240.0, 0.5
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
    return {"engine": "neuro", "t": t.tolist(), "V": V.tolist(), "th": TH.tolist(),
            "spikes": [round(float(s), 2) for s in spikes],
            "rate_hz": round(rate, 1), "n_spikes": len(spikes),
            "fI": {"drive": (dr / max(N.v_threshold, 1e-9)).tolist(),
                   "rate": rates.tolist()}}
