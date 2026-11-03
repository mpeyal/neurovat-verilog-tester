"""Native sequential SNN engine. PyTorch is the tensor engine; no snnTorch dependency.

Each call starts with zero membrane state. LIF uses delayed, detached reset.
Learning uses BPTT and the selected surrogate derivative, not a device write rule.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model_spec import Project


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage, slope, kind):
        ctx.save_for_backward(voltage)
        ctx.slope, ctx.kind = slope, kind
        return (voltage > 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        voltage, = ctx.saved_tensors
        if ctx.kind == "Fast sigmoid":
            derivative = 1/(1+ctx.slope*voltage.abs())**2
        else:
            derivative = (ctx.slope/2)/(1+(math.pi*ctx.slope*voltage/2)**2)
        return gradient*derivative, None, None


class SequentialSNN(nn.Module):
    def __init__(self, project):
        super().__init__()
        project.validate()
        self.project = deepcopy(project)
        self.shapes = project.validate()
        self.ops = nn.ModuleList()
        for spec, row in zip(project.layers, self.shapes):
            kind = spec["type"]
            if kind == "Conv2d":
                op = nn.Conv2d(row["input"][0], spec["channels"], spec["kernel"], padding=spec["padding"], bias=spec["bias"])
            elif kind == "Dense":
                op = nn.Linear(row["input"][0], spec["units"], bias=spec["bias"])
            elif kind == "MaxPool2d":
                op = nn.MaxPool2d(spec["kernel"])
            elif kind == "Flatten":
                op = nn.Flatten(1)
            else:
                op = nn.Identity()
            self.ops.append(op)
        self.float()  # Do not inherit another tool's global float64 default.

    def encode(self, images, generator=None, sample_ids=None):
        p = self.project
        value = (images*p.input_gain).clamp(0, 1) if p.encoding != "Static current" else images*p.input_gain
        if p.encoding == "Static current":
            return value.unsqueeze(0).expand(p.steps, *value.shape)
        if p.encoding == "Latency":
            when = torch.round((1-value)*(p.steps-1)).long()
            ticks = torch.arange(p.steps, device=value.device).reshape(-1, 1, 1, 1, 1)
            return ((ticks == when.unsqueeze(0)) & (value.unsqueeze(0) > 0)).to(value.dtype)
        if sample_ids is not None:
            # Stable per-image inference, independent of evaluation batch size.
            samples = []
            for sample, idx in zip(value, sample_ids):
                rng = torch.Generator(device=value.device).manual_seed(p.seed+1000003+int(idx))
                samples.append((torch.rand((p.steps, *sample.shape), generator=rng, device=value.device) < sample).float())
            return torch.stack(samples, dim=1)
        return (torch.rand((p.steps, *value.shape), generator=generator, device=value.device) < value.unsqueeze(0)).to(value.dtype)

    def forward(self, images, record=False, generator=None, sample_ids=None, stop=None):
        if tuple(images.shape[1:]) != tuple(self.project.input_shape):
            raise ValueError(f"Expected [batch, {self.project.input_shape}], received {list(images.shape)}")
        encoded = self.encode(images, generator, sample_ids)
        state, outputs, membranes, activity = {}, [], [], {}
        for value in encoded:
            if stop is not None and stop.is_set():
                raise InterruptedError("Stopped")
            for idx, (spec, op) in enumerate(zip(self.project.layers, self.ops)):
                if spec["type"] != "LIF":
                    value = op(value)
                    continue
                old = state.get(idx, torch.zeros_like(value))
                reset = (old > spec["threshold"]).to(value.dtype).detach()
                mem = spec["beta"]*old + value
                mem = mem-reset*spec["threshold"] if spec["reset"] == "subtract" else mem*(1-reset)
                state[idx] = mem
                value = SurrogateSpike.apply(mem-spec["threshold"], self.project.slope, self.project.surrogate)
                if record:
                    activity.setdefault(idx, []).append(value.detach().flatten(1).mean(1))
            outputs.append(value)
            membranes.append(state[len(self.ops)-1])
        result = {"spikes": torch.stack(outputs), "membrane": torch.stack(membranes)}
        if record:
            result.update(input=encoded, activity={i: torch.stack(v) for i, v in activity.items()})
        return result


def loss_and_scores(result, targets, kind):
    spikes = result["spikes"]
    if kind == "Membrane":
        scores = result["membrane"].mean(0)
        return F.cross_entropy(scores, targets), scores
    scores = spikes.sum(0)
    if kind == "Spike count":
        loss = F.cross_entropy(scores, targets)
    else:
        loss = F.cross_entropy(spikes.flatten(0, 1), targets.repeat(spikes.shape[0]))
    return loss, scores


def _checked_split(x, y, project, name):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim == 3 and project.input_shape[0] == 1:
        x = x[:, None]
    if x.ndim != 4 or tuple(x.shape[1:]) != tuple(project.input_shape) or len(x) != len(y):
        raise ValueError(f"{name}: expected images [N, C, H, W] matching the project and labels [N].")
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or len(y) < 2 or y.min() < 0 or y.max() >= project.classes:
        raise ValueError(f"{name}: labels must be integer class indices 0-{project.classes-1}.")
    if x.dtype == np.uint8:
        x = x.astype(np.float32)/255.
    if not (np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.integer)) or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError(f"{name}: images must be uint8 [0,255] or finite numeric values [0,1].")
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.int64))


def stratified_subset(labels, count, seed):
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(labels == c)) for c in np.unique(labels)]
    # Round-robin sampling preserves smaller classes and includes the remainder.
    order = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
    ids = np.asarray(order[:count], dtype=np.int64)
    return rng.permutation(ids)


def load_data(root, project):
    project.validate()
    if project.dataset == "Demo patterns":
        rng = np.random.default_rng(project.seed)
        templates = np.zeros((4, 1, 8, 8), dtype=np.float32)
        templates[0, 0, :, 2:4] = 1
        templates[1, 0, 2:4, :] = 1
        templates[2, 0] = np.eye(8)
        templates[3, 0] = np.fliplr(np.eye(8))
        raw = {}
        for split, n in (("train", project.train_limit), ("test", project.test_limit)):
            y = np.arange(n, dtype=np.int64) % 4
            x = np.clip(templates[y]+rng.normal(0, .12, (n, 1, 8, 8)), 0, 1).astype(np.float32)
            raw[split] = (x, y)
    else:
        if project.dataset == "MNIST":
            from .datasets import download_mnist
            path = download_mnist(str(Path(root)/"datasets"))
        else:
            path = project.dataset_path
            if not path:
                raise ValueError("Select an NPZ dataset with x_train, y_train, x_test and y_test.")
        with np.load(path, allow_pickle=False) as arrays:
            raw = {s: (arrays[f"x_{s}"], arrays[f"y_{s}"]) for s in ("train", "test")}
    result = {}
    for split, limit in (("train", project.train_limit), ("test", project.test_limit)):
        x, y = _checked_split(*raw[split], project, split)
        if limit > len(y):
            raise ValueError(f"Requested {limit:,} {split} images, but this dataset contains {len(y):,}.")
        ids = stratified_subset(y.numpy(), limit, project.seed+(split == "test"))
        result[split] = (x[ids], y[ids])
    x, y = result["train"]
    rng = np.random.default_rng(project.seed+2)
    train_ids, valid_ids = [], []
    for c in range(project.classes):
        ids = rng.permutation(np.flatnonzero(y.numpy() == c))
        if len(ids) < 2:
            raise ValueError(f"Class {c} needs at least two training samples to create a separate validation split.")
        n = max(1, min(len(ids)-1, round(len(ids)*project.validation_fraction)))
        valid_ids.extend(ids[:n]); train_ids.extend(ids[n:])
    result["train"] = (x[train_ids], y[train_ids])
    result["validation"] = (x[valid_ids], y[valid_ids])
    return {split: result[split] for split in ("train", "validation", "test")}


@torch.no_grad()
def evaluate(net, data, stop=None):
    net.eval()
    x, y = data
    preds, losses, active, bins, silent = [], 0., 0, 0, 0
    for start in range(0, len(y), net.project.batch_size):
        end = min(len(y), start+net.project.batch_size)
        out = net(x[start:end], sample_ids=range(start, end), stop=stop)
        loss, scores = loss_and_scores(out, y[start:end], net.project.loss)
        preds.append(scores.argmax(1))
        losses += float(loss)*(end-start)
        active += int(out["spikes"].sum()); bins += out["spikes"].numel()
        silent += int((out["spikes"].sum((0, 2)) == 0).sum())
    pred = torch.cat(preds)
    cm = torch.bincount(y*net.project.classes+pred, minlength=net.project.classes**2).reshape(net.project.classes, -1)
    return {"accuracy": float((pred == y).float().mean()), "loss": losses/len(y), "count": len(y),
            "correct": int((pred == y).sum()), "predictions": pred.tolist(), "targets": y.tolist(),
            "confusion": cm.tolist(), "occupancy": active/bins, "silent_samples": silent}


def save_weights(net, path):
    np.savez_compressed(path, **{name: value.detach().cpu().numpy() for name, value in net.named_parameters()})


def load_weights(net, path):
    # Validate every tensor before changing any parameters (transactional load).
    with np.load(path, allow_pickle=False) as arrays:
        expected = dict(net.named_parameters())
        if set(arrays.files) != set(expected):
            raise ValueError("Checkpoint layers do not match this project.")
        values = {}
        for name, parameter in expected.items():
            value = arrays[name]
            if value.shape != tuple(parameter.shape) or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
                raise ValueError(f"Invalid weights: {name}")
            converted = value.astype(np.float32)
            if not np.isfinite(converted).all():
                raise ValueError(f"Weights outside float32 range: {name}")
            values[name] = torch.from_numpy(converted.copy())
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(values[name])


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def train(root, folder, project, callback=lambda event: None, stop=None):
    project.validate()
    stop = stop or threading.Event()
    torch.set_num_threads(project.threads)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    project.save(folder/"project.json")
    started = time.monotonic()
    # Isolate initialization from unrelated callers' random state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(project.seed)
        net = SequentialSNN(project)
    history, best, best_epoch, data = [], (-1., float('-inf')), 0, None
    def status(state, **values):
        write_json(folder/"status.json", {"state": state, "elapsed_s": time.monotonic()-started, **values})
    status("loading")
    try:
        callback({"kind": "status", "message": "Loading data and creating a separate validation split..."})
        data = load_data(root, project)
        if stop.is_set():
            raise InterruptedError("Stopped")
        opt = torch.optim.Adam(net.parameters(), lr=project.lr) if project.optimizer == "Adam" else torch.optim.SGD(net.parameters(), lr=project.lr)
        rng = torch.Generator().manual_seed(project.seed+3)
        x, y = data["train"]
        callback({"kind": "data", "counts": {s: len(v[1]) for s, v in data.items()}})
        status("training")
        for epoch in range(1, project.epochs+1):
            net.train()
            order = torch.randperm(len(y), generator=rng)
            total, loss_sum, correct = 0, 0., 0
            for start in range(0, len(y), project.batch_size):
                ids = order[start:start+project.batch_size]
                opt.zero_grad(set_to_none=True)
                result = net(x[ids], generator=rng, stop=stop)
                loss, scores = loss_and_scores(result, y[ids], project.loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Training loss is not finite. Reduce the learning rate or input gain.")
                loss.backward()
                # Reject nonfinite gradients even when clipping is disabled.
                torch.nn.utils.clip_grad_norm_(net.parameters(), project.gradient_clip or float("inf"), error_if_nonfinite=True)
                opt.step()
                total += len(ids); loss_sum += float(loss.detach())*len(ids)
                correct += int((scores.detach().argmax(1) == y[ids]).sum())
                callback({"kind": "batch", "epoch": epoch, "seen": total, "count": len(y), "loss": loss_sum/total,
                          "online_accuracy": correct/total, "progress": ((epoch-1)+total/len(y))/project.epochs})
            validation = evaluate(net, data["validation"], stop)
            row = {"epoch": epoch, "loss": loss_sum/total, "online_accuracy": correct/total,
                   "validation_accuracy": validation["accuracy"], "validation_loss": validation["loss"],
                   "occupancy": validation["occupancy"], "silent_samples": validation["silent_samples"]}
            history.append(row)
            save_weights(net, folder/"last_weights.npz")
            score = (validation["accuracy"], -validation["loss"])
            if score > best:
                best, best_epoch = score, epoch
                save_weights(net, folder/"weights.npz")
            write_json(folder/"history.json", history)
            status("training", completed_epochs=epoch, best_epoch=best_epoch)
            callback({"kind": "epoch", "history": deepcopy(history)})
        load_weights(net, folder/"weights.npz")
        callback({"kind": "status", "message": f"Evaluating best validation checkpoint (epoch {best_epoch})..."})
        metrics = {s: evaluate(net, batch, stop) for s, batch in data.items()}
        result = {"history": history, "best_epoch": best_epoch, "metrics": metrics,
                  "elapsed_s": time.monotonic()-started, "readout": project.loss,
                  "engine": "NeuroVAT native LIF / PyTorch autograd", "synthetic": project.dataset == "Demo patterns"}
        write_json(folder/"result.json", result)
        status("complete", best_epoch=best_epoch)
        callback({"kind": "done", "result": result})
        return net, data, result
    except InterruptedError:
        status("stopped", completed_epochs=len(history), best_epoch=best_epoch)
        callback({"kind": "stopped", "message": "Stopped. Completed checkpoints remain in the run folder."})
        return net, data, None
    except Exception as exc:
        status("failed", error=str(exc), completed_epochs=len(history))
        raise
