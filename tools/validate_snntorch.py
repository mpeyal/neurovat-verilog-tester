"""Compare the actual app engine with independent snnTorch forward/autograd.

The reference uses snn.Leaky without internal reset plus explicit differentiable
immediate subtraction, matching NeuroVAT's surrogate mode. Both sides use the
same device programming routine: this validates neuron math and learning, NOT
the device physics. No GUI or snnTorch dependency is required by the app itself.
"""
import copy
import dataclasses
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import snntorch as snn
from snntorch import surrogate
from simulation_core import EcfetV2, FeFET
from vatester.neuro import Trainer, NetConfig, NeuronParams, STDPParams

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


class FastSigmoid64(surrogate.FastSigmoid):
    """Keep float64 output; snnTorch 1.0's forward hardcasts spikes to float32.

    The installed library's spike decision and backward formula are unchanged.
    """
    @staticmethod
    def forward(ctx, input_, slope=25):
        return surrogate.FastSigmoid.forward(ctx, input_, slope).to(input_.dtype)


def reference(tr, spikes, target=None):
    """Independent forward and autograd; no app neuron/gradient calls."""
    snn.Leaky.instances.clear()  # release completed graphs retained by the library registry
    weights = [torch.tensor(tr.weights_norm(c), requires_grad=target is not None)
               for c in range(tr.n_layers)]
    pre = torch.tensor(spikes, dtype=torch.float64)
    layers = [pre]
    membranes = []
    for c, weight in enumerate(weights):
        gain = tr.N.epsp_gain * (tr.N.hidden_gain if c+1 < tr.n_layers else 1.)
        current = gain * (pre @ weight.T) / (pre.sum(1, keepdim=True)+1.)
        cell = snn.Leaky(beta=torch.tensor(math.exp(-tr.cfg.dt_ms/tr.N.tau_m)),
                         threshold=torch.tensor(tr.N.v_threshold),
                         spike_grad=lambda x: FastSigmoid64.apply(x, tr.S.sg_beta),
                         reset_mechanism="none")
        mem = torch.zeros(weight.shape[0])
        ss, uu, vv = [], [], []
        for step in current:
            spike, voltage = cell(step, mem)
            mem = voltage-tr.N.v_threshold*spike
            ss.append(spike); uu.append(voltage); vv.append(mem)
        pre = torch.stack(ss)
        layers.append(pre)
        membranes.append(torch.stack(vv))
    logits = torch.stack(uu).mean(0)
    loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([target])) if target is not None else None
    if loss is not None:
        loss.backward()
    return {"loss": float(loss.detach()) if loss is not None else None,
            "winner": int(logits.argmax()),
            "grads": [w.grad.detach().numpy() for w in weights] if loss is not None else [],
            "layers": [s.detach().numpy() for s in layers],
            "membrane": membranes[-1].detach().numpy()}


def data(seed):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(30):
        target = i % 2
        base = np.array([.9, .1, .8, .2]) if target == 0 else np.array([.1, .9, .2, .8])
        result.append((np.clip(base+rng.normal(0, .045, 4), 0, 1), target))
    return result[:16], result[16:22], result[22:]


def reference_stdp(tr, spikes, target=None, learn=False):
    """snnTorch Euler-LIF plus independent Torch traces/local write decisions.

    Refractory, teacher, WTA and adaptive threshold are explicit because a
    stock snn.Leaky layer alone does not implement NeuroVAT's network rule.
    """
    N, S, cfg = tr.N, tr.S, tr.cfg
    snn.Leaky.instances.clear()
    sizes = tr.layer_sizes
    layers = [torch.tensor(spikes, dtype=torch.float64)] + [torch.zeros((len(spikes), n)) for n in sizes[1:]]
    x = [torch.zeros(n) for n in sizes]
    pre = [torch.zeros(n) for n in sizes]
    v = [torch.full((n,), N.v_rest) for n in sizes[1:]]
    refr = [torch.zeros(n) for n in sizes[1:]]
    theta = [torch.tensor(t) for t in tr.theta]
    acc = [torch.zeros((sizes[c+1], sizes[c])) for c in range(tr.n_layers)]
    counts = [torch.zeros(n) for n in sizes[1:]]
    weights = [torch.tensor(tr.weights_norm(c)) for c in range(tr.n_layers)]
    cells = [snn.Leaky(beta=torch.tensor(1-cfg.dt_ms/N.tau_m),
                      reset_mechanism="none") for _ in weights]
    membrane = torch.zeros((len(spikes), tr.n_out))
    with torch.no_grad():
        for t in range(len(spikes)):
            fired_input = layers[0][t].bool()
            x[0][fired_input] = 1.; pre[0][fired_input] = 1.
            for c, w in enumerate(weights):
                eff = w*x[c]
                gain = N.hidden_gain if c+1 < tr.n_layers else 1.
                if c == 0:
                    drive = gain*(N.epsp_gain*(eff*torch.tensor(tr.exc0)).sum(1)
                                  - N.ipsp_gain*(eff*torch.tensor(tr.inh0)).sum(1))/(x[c].sum()+1.)
                else:
                    drive = gain*N.epsp_gain*eff.sum(1)/(x[c].sum()+1.)
                if c == tr.n_layers-1 and cfg.mode == "supervised" and target is not None:
                    drive[target] += N.teacher
                active = refr[c] <= 0
                cells[c].threshold = N.v_threshold+theta[c]
                spike, proposed = cells[c]((cfg.dt_ms/N.tau_m)*(N.v_rest+drive), v[c])
                v[c] = torch.where(active, proposed, torch.tensor(N.v_reset))
                refr[c][~active] -= cfg.dt_ms
                fired = torch.nonzero((spike > 0) & active).flatten()
                if len(fired):
                    if N.inhibition > 0 and sizes[c+1] > 1 and c == tr.n_layers-1:
                        win = fired[torch.argmax(v[c][fired])]
                        others = torch.ones(sizes[c+1], dtype=torch.bool); others[win] = False
                        v[c][others] -= N.inhibition
                        fired = win.reshape(1)
                    for j in fired.tolist():
                        layers[c+1][t, j] = 1.
                        v[c][j] = N.v_reset; refr[c][j] = N.t_refractory
                        if c == tr.n_layers-1:
                            theta[c][j] += N.theta_plus
                        x[c+1][j] = 1.; pre[c+1][j] = 1.
                        if learn:
                            acc[c][j] += pre[c]; counts[c][j] += 1.
                if c == tr.n_layers-1:
                    membrane[t] = v[c]
                    membrane[t, fired] = N.v_spike_peak
            for c in range(len(sizes)):
                x[c] *= math.exp(-cfg.dt_ms/N.tau_syn)
                pre[c] *= math.exp(-cfg.dt_ms/S.tau_pre)
            if N.tau_theta > 0:
                for th in theta:
                    th *= math.exp(-cfg.dt_ms/N.tau_theta)
        tr.theta = [th.numpy().copy() for th in theta]
        if learn:
            for c, count in enumerate(counts):
                for j in range(len(count)):
                    if count[j] < 1:
                        continue
                    mean = acc[c][j]/count[j]
                    offset = max(S.offset, 1e-6)
                    gain = min(tr.learn_cap, float(count[j]))
                    for i, value in enumerate(mean.tolist()):
                        pot = value >= offset
                        strength = S.a_plus*(value-offset)/(1-offset) if pot else S.a_minus*(offset-value)/offset
                        if pot or S.a_minus > 0:
                            tr._program(c, j, i, pot, strength, gain)
    totals = layers[-1].sum(0)
    return {"loss": 0., "winner": int(totals.argmax()) if totals.sum() else -1,
            "layers": [s.numpy() for s in layers], "membrane": membrane.numpy(), "grads": []}


def compare(cls, hidden, seed, rule="surrogate", *, input_kind=None, write_amp=None, evidence_path=None):
    train, validation, test = data(seed)
    kind = input_kind or ("voltage" if cls is FeFET else "current")
    amp = write_amp if write_amp is not None else (1. if kind == "voltage" else 50e-12)
    def make_device():
        device = cls()
        # Same policy as App._nt_device_factory: network plasticity owns timing.
        for field in ("A_stdp", "A_stdp_V"):
            if hasattr(device.p, field):
                setattr(device.p, field, 0.)
        return device
    app = Trainer(make_device, kind, NeuronParams(),
                  STDPParams(pot_amp=amp, dep_amp=amp, sg_lr=.04),
                  NetConfig(grid_h=2, grid_w=2, n_out=2, hidden_layers=hidden,
                            present_ms=40, seed=seed, learn_rule=rule),
                  patterns=[(str(i), x) for i, (x, y) in enumerate(train)],
                  targets=[y for x, y in train])
    ref = copy.deepcopy(app)
    encode = app.encode_spikes
    errors = dict(loss=0., gradient=0., membrane=0., weight_S=0.,
                  matched_state_gradient=0., spike_mismatches=0, prediction_mismatches=0)
    history = []
    best_score, best_loss, best_states = -1., float("inf"), None
    rng = np.random.default_rng(seed+777)

    def score(samples, seed_offset):
        saved_app, saved_ref = app.capture_state(), ref.capture_state()
        app.theta = [np.zeros_like(t) for t in app.theta]
        ref.theta = [np.zeros_like(t) for t in ref.theta]
        ap, rp = [], []
        for i, (x, y) in enumerate(samples):
            spikes = encode(x, np.random.default_rng(seed_offset+i), noise=False)
            app.encode_spikes = lambda *a, **k: spikes
            a = app.infer(x)
            r = reference(ref, spikes) if rule == "surrogate" else reference_stdp(ref, spikes)
            ap.append(a["winner"]); rp.append(r["winner"])
        errors["prediction_mismatches"] += sum(a != b for a, b in zip(ap, rp))
        targets = [y for x, y in samples]
        app.restore_state(saved_app); ref.restore_state(saved_ref)
        return float(np.mean(np.array(ap) == targets)), float(np.mean(np.array(rp) == targets))

    for epoch in range(8):
        if rule == "surrogate":
            app.lr_scale = ref.lr_scale = .15 + .85*.5*(1+math.cos(math.pi*epoch/7))
        epoch_loss = 0.
        for i in rng.permutation(len(train)):
            x, target = train[i]
            spikes = encode(x, np.random.default_rng(seed*10000+epoch*100+int(i)), noise=False)
            app.encode_spikes = lambda *a, **k: spikes
            # Tight local gradient check at identical states separates a math
            # error from amplified floating-point drift during device updates.
            local_reference = reference(copy.deepcopy(app), spikes, target) if rule == "surrogate" else None
            gradients = {}
            original = app._apply_grad
            def record(c, grad):
                gradients[c] = grad.copy()
                original(c, grad)
            app._apply_grad = record
            a = app.train_step(x, target)
            epoch_loss += a.get("loss", 0.)
            app._apply_grad = original
            r = reference(ref, spikes, target) if rule == "surrogate" else reference_stdp(ref, spikes, target, True)
            errors["loss"] = max(errors["loss"], abs(a.get("loss", 0.)-r["loss"]))
            errors["membrane"] = max(errors["membrane"], float(np.max(np.abs(a["v_trace"]-r["membrane"]))))
            for c in range(app.n_layers):
                if rule == "surrogate":
                    errors["matched_state_gradient"] = max(errors["matched_state_gradient"],
                        float(np.max(np.abs(gradients[c]-local_reference["grads"][c]))))
                    errors["gradient"] = max(errors["gradient"], float(np.max(np.abs(gradients[c]-r["grads"][c]))))
                    ref._apply_grad(c, r["grads"][c])
                errors["weight_S"] = max(errors["weight_S"], float(np.max(np.abs(app.weights_S(c)-ref.weights_S(c)))))
                actual = np.zeros_like(r["layers"][c+1])
                for j, times in enumerate(a["layer_spikes"][c+1]):
                    actual[np.rint(np.array(times)/app.cfg.dt_ms).astype(int), j] = 1
                errors["spike_mismatches"] += int(np.count_nonzero(actual != r["layers"][c+1]))
            # Match the app's elapsed presentation time for time-dependent twins.
            ref._t_dev += app.cfg.present_ms*1e-3
        val_acc = score(validation, seed*1000+500)
        train_acc = score(train, seed*1000+700)
        history.append({"epoch": epoch+1, "app_validation": val_acc[0], "snntorch_validation": val_acc[1],
                        "app_train": train_acc[0], "snntorch_train": train_acc[1]})
        if val_acc[0] > best_score or (val_acc[0] == best_score and epoch_loss < best_loss):
            best_score = val_acc[0]
            best_loss = epoch_loss
            best_states = app.capture_state(), ref.capture_state()
    app.restore_state(best_states[0]); ref.restore_state(best_states[1])
    test_acc = score(test, seed*1000+900)
    if evidence_path is not None:
        x, target = test[0]
        spikes = encode(x, np.random.default_rng(seed*1000+900), noise=False)
        app.encode_spikes = lambda *a, **k: spikes
        app.theta = [np.zeros_like(t) for t in app.theta]
        ref.theta = [np.zeros_like(t) for t in ref.theta]
        actual = app.infer(x)
        expected = reference(ref, spikes) if rule == "surrogate" else reference_stdp(ref, spikes)
        output = np.zeros_like(expected["layers"][-1])
        for j, times in enumerate(actual["layer_spikes"][-1]):
            output[np.rint(np.array(times)/app.cfg.dt_ms).astype(int), j] = 1
        np.savez_compressed(evidence_path, input_spikes=spikes, app_spikes=output,
                            snntorch_spikes=expected["layers"][-1], app_membrane=actual["v_trace"],
                            snntorch_membrane=expected["membrane"], dt_ms=app.cfg.dt_ms,
                            target=target)
    passed = (errors["matched_state_gradient"] < 1e-10 and errors["loss"] < 1e-9 and errors["gradient"] < 1e-8 and errors["membrane"] < 1e-8
              and errors["weight_S"] < 1e-11 and errors["spike_mismatches"] == 0 and errors["prediction_mismatches"] == 0)
    return {"device": cls.__name__, "rule": rule, "layers": app.layer_sizes, "seed": seed,
            "config": dataclasses.asdict(app.cfg), "neuron": dataclasses.asdict(app.N),
            "plasticity": dataclasses.asdict(app.S), "device_params": dataclasses.asdict(app.crossbars[0][0][0].p),
            "errors": errors, "history": history, "app_test_accuracy": test_acc[0],
            "snntorch_test_accuracy": test_acc[1], "passed": passed}


def main():
    rows = []
    for rule in ("surrogate", "stdp"):
        for cls in (c for c in (EcfetV2, FeFET) if c is not None):   # FeFET is private
            for hidden in ((), (3,)):
                for seed in (1, 7, 23):
                    row = compare(cls, hidden, seed, rule)
                    rows.append(row)
                    print(rule, row["device"], row["layers"], seed, row["passed"], row["errors"], flush=True)
    report = {"torch": torch.__version__, "snntorch": snn.__version__, "dtype": "float64 CPU",
              "epochs": 8, "train_validation_test_counts": [16, 6, 8],
              "scope": "Matched equations, spikes, initialization and device pulse updates for surrogate and supervised STDP modes. Independent snnTorch neurons/autograd and Torch eligibility rule; device physics/programming is shared, not independently validated. Synthetic two-class data, not MNIST or hardware validation. Float64-preserving wrapper around snnTorch FastSigmoid; explicit immediate reset and app-specific refractory/homeostasis/WTA.",
              "runs": rows, "passed": all(r["passed"] for r in rows)}
    path = Path(__file__).resolve().parents[1]/"results"/"snntorch_validation.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
