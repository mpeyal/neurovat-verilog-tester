"""Native convolutional SNN benchmark, matching loaigandeel/snn-mnist.

Ideal signed weights and biases, not an ECFET device approximation. PyTorch
provides convolutions/autograd; the app implements its own LIF recurrence.
snnTorch is imported only for the independent reference implementation.
"""
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ConvConfig:
    steps: int = 50
    beta: float = .5
    slope: float = 25.
    batch_size: int = 128
    epochs: int = 5
    lr: float = .01
    seed: int = 17
    train_limit: int = 60000
    test_limit: int = 10000
    threads: int = 4

    def __post_init__(self):
        from .model_spec import finite_range, positive_int
        for key, maximum in (("steps", 200), ("batch_size", 1024), ("epochs", 1000), ("threads", 32), ("train_limit", 60000), ("test_limit", 10000)):
            positive_int(getattr(self, key), key, maximum)
        for key, bounds, inclusive in (("beta", (0., 1.), True), ("slope", (0., 1000.), False), ("lr", (0., 1.), False)):
            finite_range(getattr(self, key), key, *bounds, include_low=inclusive)
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32).")
        if self.train_limit < self.batch_size:
            raise ValueError("Training count must contain at least one full batch.")
        if self.train_limit % 10 or self.test_limit % 10:
            raise ValueError("Balanced sample counts must be multiples of ten.")


class FastSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage, slope):
        ctx.save_for_backward(voltage)
        ctx.slope = slope
        return (voltage > 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        voltage, = ctx.saved_tensors
        return gradient/(ctx.slope*voltage.abs()+1.)**2, None


class ConvSNN(nn.Module):
    def __init__(self, cfg=None, reference=False):
        super().__init__()
        self.cfg = cfg or ConvConfig()
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.fc1 = nn.Linear(1024, 10)
        self.reference = reference
        if reference:
            import snntorch as snn
            from snntorch import surrogate
            self.lifs = nn.ModuleList([snn.Leaky(beta=self.cfg.beta,
                spike_grad=surrogate.fast_sigmoid(slope=self.cfg.slope)) for _ in range(3)])

    def forward(self, images, record=False):
        mem = [None, None, None]
        spikes, voltages = [], []
        hidden = [[], []]
        # The notebook repeats a static image. Caching this deterministic
        # convolution is algebraically equivalent; gradients accumulate into it.
        drive1 = F.max_pool2d(self.conv1(images), 2)
        for _ in range(self.cfg.steps):
            value = drive1
            for layer in range(3):
                if mem[layer] is None:
                    mem[layer] = torch.zeros_like(value)
                if self.reference:
                    spike, mem[layer] = self.lifs[layer](value, mem[layer])
                else:
                    reset = (mem[layer] > 1.).to(value.dtype).detach()
                    mem[layer] = self.cfg.beta*mem[layer] + value - reset
                    spike = FastSpike.apply(mem[layer]-1., self.cfg.slope)
                if layer < 2 and record:
                    hidden[layer].append(spike)
                if layer == 0:
                    value = F.max_pool2d(self.conv2(spike), 2)
                elif layer == 1:
                    value = self.fc1(spike.flatten(1))
            spikes.append(spike)
            if record:
                voltages.append(mem[-1])
        result = {"spikes": torch.stack(spikes)}
        if record:
            result.update(membrane=torch.stack(voltages), hidden=[torch.stack(s) for s in hidden])
        return result


def rate_loss(spikes, targets):
    # Match ce_rate_loss: cross entropy of binary spikes at every step,
    # NOT cross entropy of the sum of spikes.
    log_probability = F.log_softmax(spikes, dim=-1)
    loss = torch.zeros(1, dtype=spikes.dtype, device=spikes.device)
    for step in log_probability:
        loss = loss + F.nll_loss(step, targets)
    return (loss/spikes.shape[0]).squeeze()


def parameter_arrays(net):
    return {name: value.detach().cpu().numpy() for name, value in net.named_parameters()}


def transfer(source, destination):
    with torch.no_grad():
        src = dict(source.named_parameters())
        for name, weight in destination.named_parameters():
            weight.copy_(src[name])


def load_data(root, cfg):
    from .datasets import download_mnist
    with np.load(download_mnist(str(Path(root)/"datasets")), allow_pickle=False) as data:
        result = {}
        for split, limit in (("train", cfg.train_limit), ("test", cfg.test_limit)):
            x, y = data[f"x_{split}"], data[f"y_{split}"]
            if not 1 <= limit <= len(y):
                raise ValueError(f"Invalid {split} image count")
            # Full MNIST uses original order; subsets are deterministic and balanced.
            if limit < len(y):
                rng = np.random.default_rng(cfg.seed)
                ids = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:limit//10] for c in range(10)])
                if len(ids) != limit:
                    raise ValueError("Subset counts must be multiples of ten")
                x, y = x[ids], y[ids]
            result[split] = (torch.from_numpy(x.copy()).float().unsqueeze(1)/255., torch.from_numpy(y.copy()).long())
    return result


@torch.no_grad()
def evaluate(net, data, batch_size=128, stop=None):
    net.eval()
    x, y = data
    predictions, occupied, bins = [], 0, 0
    for start in range(0, len(y), batch_size):
        if stop and stop.is_set():
            raise InterruptedError("Evaluation stopped")
        out = net(x[start:start+batch_size])["spikes"]
        predictions.append(out.sum(0).argmax(1))
        occupied += int(out.sum()); bins += out.numel()
    pred = torch.cat(predictions)
    return {"accuracy": float((pred == y).float().mean()), "correct": int((pred == y).sum()),
            "count": len(y), "predictions": pred.tolist(), "targets": y.tolist(),
            "output_occupancy": occupied/bins}


def train(root, output, cfg, callback=lambda data: None, stop=None, reference=False):
    """Used by both the native panel and CLI. No GUI calls from this worker."""
    stop = stop or threading.Event()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    net = ConvSNN(cfg, reference=reference)
    if reference:
        from snntorch.functional import ce_rate_loss
        loss_function = ce_rate_loss()
    else:
        loss_function = rate_loss
    data = load_data(root, cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, betas=(.9, .999))
    generator = torch.Generator().manual_seed(cfg.seed+1)
    history = []
    started = time.monotonic()
    (output/"settings.json").write_text(json.dumps(asdict(cfg), indent=2))
    np.savez_compressed(output/"initial_weights.npz", **parameter_arrays(net))
    try:
        for epoch in range(1, cfg.epochs+1):
            net.train()
            x, y = data["train"]
            order = torch.randperm(len(y), generator=generator)
            losses, correct, total = 0., 0, 0
            # Notebook drops incomplete TRAIN batches (full data => 59,904/epoch).
            for start in range(0, len(y)-cfg.batch_size+1, cfg.batch_size):
                if stop.is_set():
                    raise InterruptedError("Training stopped")
                ids = order[start:start+cfg.batch_size]
                optimizer.zero_grad(set_to_none=True)
                spikes = net(x[ids])["spikes"]
                loss = loss_function(spikes, y[ids])
                loss.backward(); optimizer.step()
                losses += float(loss.detach())*len(ids)
                correct += int((spikes.detach().sum(0).argmax(1) == y[ids]).sum())
                total += len(ids)
                if start//cfg.batch_size % 10 == 0:
                    callback({"kind": "batch", "epoch": epoch, "seen": total, "count": len(y),
                              "loss": losses/total, "online_accuracy": correct/total,
                              "elapsed_s": time.monotonic()-started})
            np.savez_compressed(output/f"epoch_{epoch}.npz", **parameter_arrays(net))
            row = {"epoch": epoch, "loss": losses/total, "online_train_accuracy": correct/total,
                   "presentations": total, "elapsed_s": time.monotonic()-started}
            history.append(row)
            (output/"history.json").write_text(json.dumps(history, indent=2))
            callback({"kind": "epoch", **row})
        # Fixed last epoch: no test-based checkpoint selection or tuning.
        result = {"config": asdict(cfg), "implementation": "snntorch" if reference else "app",
                  "history": history, "elapsed_s": time.monotonic()-started,
                  "train": evaluate(net, data["train"], cfg.batch_size, stop),
                  "test": evaluate(net, data["test"], cfg.batch_size, stop)}
        np.savez_compressed(output/"model.npz", **parameter_arrays(net))
        (output/"result.json").write_text(json.dumps(result, indent=2))
        callback({"kind": "done", "result": result})
        return net, result
    except InterruptedError:
        callback({"kind": "stopped"})
        return net, None


def load_weights(net, path):
    with np.load(path, allow_pickle=False) as arrays:
        params = dict(net.named_parameters())
        if set(arrays.files) != set(params):
            raise ValueError("Checkpoint tensors do not match the network.")
        values = {}
        for name, parameter in params.items():
            value = arrays[name]
            if value.shape != tuple(parameter.shape) or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
                raise ValueError(f"Invalid weight tensor {name}")
            converted = torch.from_numpy(value.copy()).to(parameter.dtype)
            if not torch.isfinite(converted).all():
                raise ValueError(f"Weight tensor outside numeric range: {name}")
            values[name] = converted
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(values[name])
