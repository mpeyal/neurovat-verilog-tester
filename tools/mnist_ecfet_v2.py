"""Reproducible real-MNIST app-engine versus matched snnTorch experiment."""
import copy
import csv
import dataclasses
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import snntorch
from simulation_core import EcfetV2, V2Params
from vatester import datasets
from vatester.neuro import Trainer, NetConfig, NeuronParams, STDPParams, confusion
from tools.validate_snntorch import reference

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results"/"mnist_ecfet_v2"
SEED, EPOCHS = 17, 8


def save(name, obj):
    payload = json.dumps(obj, indent=2, allow_nan=False)
    if "_epoch_" in name:
        with gzip.open(OUT/(name+".gz"), "wt", encoding="utf-8") as f:
            f.write(payload)
    else:
        (OUT/name).write_text(payload, encoding="utf-8")


def checkpoint(name, epoch):
    path = OUT/f"{name}_epoch_{epoch}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def prepare():
    path = Path(datasets.download_mnist(str(ROOT/"datasets")))
    with np.load(path, allow_pickle=False) as archive:
        d = {key: archive[key] for key in archive.files}
        assert d["x_train"].shape == (60000, 28, 28) and d["x_test"].shape == (10000, 28, 28)
        rng = np.random.default_rng(SEED)
        ids = {"train": [], "validation": [], "test": []}
        for label in range(10):
            order = rng.permutation(np.flatnonzero(d["y_train"] == label))
            ids["train"].extend(order[:100].tolist())
            ids["validation"].extend(order[100:120].tolist())
            ids["test"].extend(rng.permutation(np.flatnonzero(d["y_test"] == label))[:50].tolist())
        data = {}
        for split, chosen in ids.items():
            source = "test" if split == "test" else "train"
            frames = []
            for idx in chosen:
                g = datasets._resize(d[f"x_{source}"][idx], 14, 14)
                g -= g.min()
                frames.append((g/(g.max() or 1.)).reshape(-1).astype(np.float32))
            data[split] = np.array(frames), d[f"y_{source}"][chosen].astype(int)
    assert not set(ids["train"]) & set(ids["validation"])
    save("data_manifest.json", {"url": datasets.MNIST_NPZ_URL,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "seed": SEED,
         "indices": ids, "preprocessing": "App datasets._resize Pillow bilinear 28x28 to 14x14; per-image min/max normalization; float32 intensities"})
    np.savez_compressed(OUT/"prepared_data.npz", **{f"{s}_{part}": arr for s, pair in data.items() for part, arr in zip(("x", "y"), pair)})
    return data


def make_trainer(data):
    x, y = data["train"]
    return Trainer(lambda: EcfetV2(V2Params(A_stdp=0.)), "current", NeuronParams(),
                   STDPParams(sg_lr=.1),
                   NetConfig(grid_h=14, grid_w=14, n_out=10, present_ms=60.,
                             max_rate_hz=160., learn_rule="surrogate", seed=SEED),
                   patterns=[(str(int(t)), v) for v, t in zip(x, y)], targets=y.tolist())


def app_result(tr, x, spikes, learn=False, target=None):
    original = tr.encode_spikes
    tr.encode_spikes = lambda *a, **k: spikes
    try:
        return tr.train_step(x, target) if learn else tr.infer(x)
    finally:
        tr.encode_spikes = original


def raster(result, steps=60, neurons=10):
    matrix = np.zeros((steps, neurons), dtype=np.uint8)
    for j, times in enumerate(result["out_spikes"]):
        matrix[np.asarray(times, dtype=int), j] = 1
    return matrix


def evaluate(app, ref, data, split, bank, examples=False):
    x, y = data[split]
    states = app.capture_state(), ref.capture_state()
    ap, rp, mismatch, worst_mem = [], [], 0, 0.
    samples = {}
    for i, (vec, target) in enumerate(zip(x, y)):
        spikes = bank[i]
        a = app_result(app, vec, spikes)
        r = reference(ref, spikes)
        ap.append(a["winner"]); rp.append(r["winner"])
        ar = raster(a)
        rr = r["layers"][-1].astype(np.uint8)
        mismatch += int(np.count_nonzero(ar != rr))
        worst_mem = max(worst_mem, float(np.max(np.abs(a["v_trace"]-r["membrane"]))))
        if examples and str(target) not in samples:
            samples[str(target)] = {"split_index": i, "image": vec.reshape(14, 14).tolist(),
                "app_prediction": a["winner"], "snntorch_prediction": r["winner"],
                "input_spikes": spikes.astype(int).tolist(), "app_spikes": ar.tolist(),
                "snntorch_spikes": rr.tolist(), "app_membrane": a["v_trace"].tolist(),
                "snntorch_membrane": r["membrane"].tolist()}
    app.restore_state(states[0]); ref.restore_state(states[1])
    return {"app_accuracy": float(np.mean(np.array(ap) == y)),
            "snntorch_accuracy": float(np.mean(np.array(rp) == y)),
            "prediction_disagreements": int(np.count_nonzero(np.array(ap) != rp)),
            "spike_mismatches": mismatch, "max_membrane_error": worst_mem,
            "targets": y.tolist(), "app_predictions": ap, "snntorch_predictions": rp,
            "examples": samples}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = prepare()
    app = make_trainer(data)
    ref = copy.deepcopy(app)
    settings = {"cfg": dataclasses.asdict(app.cfg), "neuron": dataclasses.asdict(app.N),
                "stdp": dataclasses.asdict(app.S), "device": dataclasses.asdict(app.crossbars[0][0][0].p),
                "epochs": EPOCHS, "torch": torch.__version__, "snntorch": snntorch.__version__,
                "selection": "Each implementation independently selects highest validation accuracy then lowest training loss. Test evaluated only after all training and selection are frozen.",
                "scope": "App backend training engine vs snnTorch float64 immediate-reset matched reference; shared ECFET device programming; single seed and balanced subset, not a full MNIST benchmark."}
    save("settings.json", settings)
    bank = {s: np.array([app.encode_spikes(v, np.random.default_rng(SEED*100000+offset+i), noise=False)
                        for i, v in enumerate(pair[0])])
            for s, pair, offset in ((s, data[s], offset) for s, offset in (("train", 10000), ("validation", 20000), ("test", 30000)))}
    np.savez_compressed(OUT/"evaluation_spikes.npz", **bank)
    history, best = [], {"app": (-1., 1e100, 0), "snntorch": (-1., 1e100, 0)}
    rng = np.random.default_rng(SEED+777)
    start = time.perf_counter()
    for epoch in range(EPOCHS+1):
        errors = {"max_gradient_error": 0., "max_loss_error": 0., "max_weight_error_S": 0., "spike_mismatches": 0}
        losses = [0., 0.]
        if epoch:
            app.lr_scale = ref.lr_scale = .15+.85*.5*(1+math.cos(math.pi*(epoch-1)/max(EPOCHS-1, 1)))
            spike_digest = hashlib.sha256()
            for count, idx in enumerate(rng.permutation(len(data["train"][0]))):
                x, target = data["train"][0][idx], int(data["train"][1][idx])
                spikes = app.encode_spikes(x, np.random.default_rng(SEED*1000000+epoch*10000+int(idx)), noise=False)
                spike_digest.update(spikes.tobytes())
                grads = {}
                original = app._apply_grad
                def record(c, grad):
                    grads[c] = grad.copy()
                    original(c, grad)
                app._apply_grad = record
                a = app_result(app, x, spikes, True, target)
                app._apply_grad = original
                r = reference(ref, spikes, target)
                for c, gradient in enumerate(r["grads"]):
                    errors["max_gradient_error"] = max(errors["max_gradient_error"], float(np.max(np.abs(grads[c]-gradient))))
                    ref._apply_grad(c, gradient)
                ref._t_dev += ref.cfg.present_ms*1e-3
                losses[0] += a["loss"]; losses[1] += r["loss"]
                errors["max_loss_error"] = max(errors["max_loss_error"], abs(a["loss"]-r["loss"]))
                errors["max_weight_error_S"] = max(errors["max_weight_error_S"], float(np.max(np.abs(app.weights_S()-ref.weights_S()))))
                errors["spike_mismatches"] += int(np.count_nonzero(raster(a) != r["layers"][-1]))
                if (count+1) % 200 == 0:
                    print(f"epoch {epoch}/{EPOCHS}: {count+1}/1000 presentations", flush=True)
            errors["training_spikes_sha256"] = spike_digest.hexdigest()
        row = {"epoch": epoch, "train": evaluate(app, ref, data, "train", bank["train"]),
               "validation": evaluate(app, ref, data, "validation", bank["validation"]),
               "mean_train_loss": [v/len(data["train"][0]) for v in losses] if epoch else None,
               "errors": errors, "elapsed_s": time.perf_counter()-start}
        for name, tr, loss in (("app", app, losses[0]), ("snntorch", ref, losses[1])):
            save(f"{name}_epoch_{epoch}.json", tr.capture_state())
            score = row["validation"][f"{name}_accuracy"]
            if epoch and (score > best[name][0] or (score == best[name][0] and loss < best[name][1])):
                best[name] = score, loss, epoch
        history.append(row)
        save("progress.json", {"history": history, "best": best, "test_evaluated": False})
        print(f"EPOCH {epoch}: train {row['train']['app_accuracy']:.1%}/{row['train']['snntorch_accuracy']:.1%}; validation {row['validation']['app_accuracy']:.1%}/{row['validation']['snntorch_accuracy']:.1%}; {time.perf_counter()-start:.0f}s", flush=True)
    save("frozen_selection.json", {"best": best, "training_complete": True, "test_evaluated": False})
    # No test sample is scored until all parameters and checkpoint choices freeze.
    for row in history:
        epoch = row["epoch"]
        app.restore_state(checkpoint("app", epoch))
        ref.restore_state(checkpoint("snntorch", epoch))
        row["test"] = evaluate(app, ref, data, "test", bank["test"])
        print(f"posthoc test epoch {epoch}: {row['test']['app_accuracy']:.1%}/{row['test']['snntorch_accuracy']:.1%}", flush=True)
    app.restore_state(checkpoint("app", best['app'][2]))
    ref.restore_state(checkpoint("snntorch", best['snntorch'][2]))
    final = {s: evaluate(app, ref, data, s, bank[s], examples=s == "test") for s in data}
    save("comparison.json", {"settings": settings, "history": history, "best": best, "final": final,
         "elapsed_s": time.perf_counter()-start, "counts": {s: len(data[s][0]) for s in data},
         "final_max_weight_error_S": float(np.max(np.abs(app.weights_S()-ref.weights_S()))),
         "final_weight_rms_error_S": float(np.sqrt(np.mean((app.weights_S()-ref.weights_S())**2)))})
    save("selected_app_state.json", app.capture_state())
    save("selected_snntorch_state.json", ref.capture_state())
    np.savez_compressed(OUT/"final_weights.npz", app=app.weights_S(), snntorch=ref.weights_S())
    with (OUT/"accuracy.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["epoch"]+[f"{s}_{m}" for s in data for m in ("app", "snntorch")])
        for row in history:
            writer.writerow([row["epoch"]]+[row[s][f"{m}_accuracy"] for s in data for m in ("app", "snntorch")])
    save("progress.json", {"history": history, "best": best, "training_complete": True,
                           "test_evaluated": True})
    print("COMPLETE", {k: (v['app_accuracy'], v['snntorch_accuracy']) for k, v in final.items()}, flush=True)


if __name__ == "__main__":
    main()
