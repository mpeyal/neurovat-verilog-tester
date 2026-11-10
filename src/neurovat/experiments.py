"""Config-driven experiments and parameter sweeps.

A config is a TOML (or JSON) file::

    [data]      path = "datasets/mnist.npz"   # or  builtin = "bars"
    [network]   NetConfig fields    (grid_h, grid_w, n_out, learn_rule, jitter_ms, ...)
    [neuron]    NeuronParams fields
    [stdp]      STDPParams fields
    [device]    model = "softbound", write_noise = 0.0
    [device.params]      model parameters
    [device.variation]   device-to-device sigma per parameter
    [train]     epochs = 3
    [power]     PowerParams fields (v_read, t_read, e_spike, ...) and
                neuron_file = "lif_energy.json"; unset values run on flagged
                placeholders
    [sweep]     parameter = "network.jitter_ms", values = [0, 1, 2, 4]
                seeds = [1, 2, 3], apply = "train+test" | "test"

Every (value, seed) pair is one independent run seeded only by the config, so
a result file can be regenerated bit-for-bit from the ``config.json`` saved
next to it.
"""
import copy
import csv
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

from . import data as _data
from .config import NeuronParams, STDPParams, NetConfig
from .devices import device_factory
from .network import CrossbarSNN
from .patterns import make_patterns
from . import power as _power
from .power import PowerParams

DEFAULTS = {
    "name": "experiment",
    "data": {"path": "", "builtin": "", "fetch_mnist": False,
             "train_per_class": 100, "val_per_class": 20, "test_per_class": 50,
             "n_classes": 0, "seed": 0, "invert": False},
    "network": {}, "neuron": {}, "stdp": {},
    "device": {"model": "softbound", "params": {}, "variation": {},
               "write_noise": 0.0},
    "train": {"epochs": 3, "anneal": True},
    "power": dict({name: None for name in PowerParams.names()}, neuron_file=""),
    "sweep": {"parameter": "", "values": [], "seeds": [1], "apply": "train+test"},
    "output": {"dir": "results/sweeps", "energy_maps": True},
}
_OPEN = ("network", "neuron", "stdp")       # sections validated by a dataclass
_OPEN_DEVICE = ("params", "variation")      # validated by the device model


def load_config(path):
    """Read a .toml / .json config and fill in the defaults."""
    if str(path).lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        try:
            import tomllib
        except ModuleNotFoundError:            # Python < 3.11
            import tomli as tomllib
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    return resolve(raw)


def resolve(raw):
    """Merge a (partial) config dict over the defaults; reject unknown keys."""
    cfg = copy.deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key not in cfg:
            raise ValueError(f"Unknown config section {key!r}")
        if not isinstance(cfg[key], dict):
            cfg[key] = value
            continue
        for k, v in value.items():
            known = (key in _OPEN or k in cfg[key])
            if not known:
                raise ValueError(f"Unknown config key {key}.{k}")
            cfg[key][k] = v
    for section, cls in (("network", NetConfig), ("neuron", NeuronParams),
                         ("stdp", STDPParams)):
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(cfg[section]) - names
        if unknown:
            raise ValueError(f"Unknown {section} field(s): {sorted(unknown)}")
    if cfg["sweep"]["apply"] not in ("train+test", "test"):
        raise ValueError("sweep.apply must be 'train+test' or 'test'")
    return cfg


def set_path(cfg, dotted, value):
    """Set ``cfg["a"]["b"]["c"] = value`` for dotted = "a.b.c"."""
    *parents, leaf = dotted.split(".")
    node = cfg
    for k in parents:
        if not isinstance(node.get(k), dict):
            raise ValueError(f"Unknown sweep parameter {dotted!r}")
        node = node[k]
    top, open_ok = (parents[0] if parents else ""), False
    if top in _OPEN and len(parents) == 1:
        cls = {"network": NetConfig, "neuron": NeuronParams, "stdp": STDPParams}[top]
        open_ok = leaf in {f.name for f in dataclasses.fields(cls)}
    if top == "device" and len(parents) == 2 and parents[1] in _OPEN_DEVICE:
        open_ok = True
    if leaf not in node and not open_ok:
        raise ValueError(f"Unknown sweep parameter {dotted!r}")
    node[leaf] = value


def load_data(cfg):
    """-> (train patterns, train targets, Dataset or None) for a config."""
    d, net = cfg["data"], cfg["network"]
    gh = net.get("grid_h", NetConfig.grid_h)
    gw = net.get("grid_w", NetConfig.grid_w)
    if not d["path"] and not d["fetch_mnist"]:
        name = d["builtin"] or net.get("pattern_set", NetConfig.pattern_set)
        return make_patterns(name, gh, gw), None, None
    path = d["path"] or os.path.join("datasets", "mnist.npz")
    if d["fetch_mnist"]:
        path = _data.fetch_mnist(os.path.dirname(path) or ".")
    ds = _data.load_dataset(path, gh, gw, d["train_per_class"],
                            d["val_per_class"], d["test_per_class"],
                            d["n_classes"] or None, d["seed"], d["invert"])
    pats, tgts = _data.as_patterns(ds.train, ds.class_names)
    return pats, tgts, ds


def build(cfg, seed, loaded=None):
    """A ready-to-train network for one resolved config and seed."""
    pats, tgts, ds = loaded if loaded is not None else load_data(cfg)
    netcfg = dict(cfg["network"], seed=int(seed))
    if "hidden_layers" in netcfg:
        netcfg["hidden_layers"] = tuple(netcfg["hidden_layers"])
    dev = cfg["device"]
    net = CrossbarSNN(
        device_factory(dev["model"], dev["params"] or None,
                       dev["variation"] or None, dev["write_noise"]),
        NeuronParams(**cfg["neuron"]), STDPParams(**cfg["stdp"]),
        NetConfig(**netcfg), patterns=pats, targets=tgts,
        power=PowerParams.from_config(cfg["power"]))
    if ds is not None:
        _data.attach(net, ds)
    return net


def _accuracy(net, noisy):
    """Clean or noisy accuracy on the held-out test set (training set when
    the data has no test split)."""
    split = "test" if net.test_patterns else None
    return net.eval_accuracy(split=split, noisy=noisy)


def _score(net):
    """Clean and noisy test accuracy with the energy of each evaluation. The
    clean pass runs last, so the meter's "eval" phase (and the saved energy
    maps) describe clean inference."""
    net.energy.reset("eval")
    noisy_acc = _accuracy(net, True)
    noisy = net.energy.report("eval", parameters=False)
    net.energy.reset("eval")
    acc = _accuracy(net, False)
    clean = net.energy.report("eval", parameters=False)
    return {"test_acc": acc, "test_acc_noisy": noisy_acc,
            "infer_energy_J": clean["e_inference_J"],
            "infer_energy_noisy_J": noisy["e_inference_J"],
            "infer_power_W": clean["avg_power_W"],
            "pj_per_sop": clean["e_sop_pJ"],
            "pj_per_sop_array": clean["e_sop_array_pJ"],
            "pj_per_neuron_spike": clean["e_neuron_spike_pJ"],
            "energy_eval": clean, "energy_eval_noisy": noisy}


def run_one(cfg, seed, loaded=None):
    """Train one network; -> (result row, trained net)."""
    net = build(cfg, seed, loaded)
    t0 = time.perf_counter()
    history = net.fit(cfg["train"]["epochs"], anneal=cfg["train"]["anneal"])
    train_s = time.perf_counter() - t0
    trained = net.energy.report("train", parameters=False)
    row = {"seed": int(seed), "train_s": train_s,
           "train_energy_J": trained["energy_J"]["total"],
           "train_write_energy_J": trained["energy_J"]["array_write"],
           "train_power_W": trained["avg_power_W"],
           "energy_train": trained,
           "energy_init": net.energy.report("init", parameters=False),
           "train_acc": net.eval_accuracy(),
           "writes": net.write_count, "write_effort": net.write_effort,
           "history": history}
    if net.validation_patterns:
        row["val_acc"] = net.eval_accuracy(split="validation")
    row.update(_score(net))
    return row, net


def provenance(cfg, datasets=()):
    from . import __version__
    info = {"neurovat": __version__, "python": platform.python_version(),
            "numpy": np.__version__, "platform": platform.platform(),
            "argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config_sha256": hashlib.sha256(json.dumps(
                cfg, sort_keys=True).encode()).hexdigest(),
            "data": [d.meta for d in datasets if d is not None]}
    try:
        git = lambda *a: subprocess.run(("git",) + a, capture_output=True,
                                        text=True, timeout=10).stdout.strip()
        info["git_commit"] = git("rev-parse", "HEAD") or None
        info["git_dirty"] = bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = None
    return info


_METRICS = ("test_acc", "test_acc_noisy", "val_acc", "train_acc", "train_s",
            "writes", "write_effort",
            "infer_energy_J", "infer_energy_noisy_J", "infer_power_W",
            "pj_per_sop", "pj_per_sop_array", "pj_per_neuron_spike",
            "train_energy_J", "train_write_energy_J", "train_power_W")


def summarize(runs):
    """Mean and seed-to-seed std of every metric, per sweep value."""
    out = []
    for value in dict.fromkeys(json.dumps(r["value"]) for r in runs):
        group = [r for r in runs if json.dumps(r["value"]) == value]
        row = {"value": json.loads(value), "n_seeds": len(group)}
        for m in _METRICS:
            vals = [r[m] for r in group if r.get(m) is not None]
            if vals:
                row[m + "_mean"] = float(np.mean(vals))
                row[m + "_std"] = float(np.std(vals))
        out.append(row)
    return out


def run_experiment(cfg, out_dir=None, log=print):
    """Run a config (single run or sweep); write and return the results.

    Output directory ``<output.dir>/<name>/``: ``config.json`` (resolved),
    ``results.json`` (provenance, every run, summary), ``summary.csv``.
    apply="test" trains once per seed at the base config and re-scores the
    trained network under each swept ``network.*`` value (inference-time
    sensitivity); "train+test" retrains for every value. With
    ``output.energy_maps`` every run also writes its per-synapse / per-neuron
    energy arrays (``energy/<run>.npz``) and heatmaps (``.png``)."""
    cfg = resolve(cfg)
    sweep = cfg["sweep"]
    param, values = sweep["parameter"], list(sweep["values"])
    if not param:
        values = [None]
    test_only = sweep["apply"] == "test" and bool(param)
    if test_only and not param.startswith("network."):
        raise ValueError("sweep.apply = 'test' needs a network.* parameter")
    if param:
        set_path(copy.deepcopy(cfg), param, values[0])     # validate early
    info, runs, cache = provenance(cfg), [], {}
    out = out_dir or os.path.join(cfg["output"]["dir"], cfg["name"])
    first = copy.deepcopy(cfg)
    if param.startswith("power."):             # a swept value is not a placeholder
        set_path(first, param, values[0])
    power = PowerParams.from_config(first["power"])
    if power.placeholders():
        log("power model: PLACEHOLDER values for "
            + ", ".join(power.placeholders()))

    def loaded_for(c):
        key = json.dumps([c["data"], c["network"].get("grid_h"),
                          c["network"].get("grid_w"),
                          c["network"].get("pattern_set")], sort_keys=True)
        if key not in cache:
            cache[key] = load_data(c)
            if cache[key][2] is not None:
                info["data"].append(cache[key][2].meta)
        return cache[key]

    def record(value, row, net):
        row = dict({"parameter": param or None, "value": value}, **row)
        runs.append(row)
        log(f"{param or 'run'}={value} seed={row['seed']}: "
            f"test {row['test_acc']:.3f} (noisy {row['test_acc_noisy']:.3f}), "
            f"train {row['train_s']:.1f}s, "
            f"{_power._eng(row['infer_energy_J'], 'J')}/inference")
        if cfg["output"]["energy_maps"]:
            tag = f"seed{row['seed']}"
            if param:
                tag = "".join(ch if ch.isalnum() or ch in "-._" else "_"
                              for ch in f"{param.split('.')[-1]}={value}") + "_" + tag
            folder = os.path.join(out, "energy")
            os.makedirs(folder, exist_ok=True)
            _power.save_maps(net.energy, os.path.join(folder, tag + ".npz"))
            for phase in ("train", "eval"):
                _power.plot_maps(net.energy,
                                 os.path.join(folder, f"{tag}_{phase}.png"),
                                 phase, f"{cfg['name']} {tag} - {phase}")

    t0 = time.perf_counter()
    for seed in sweep["seeds"]:
        if test_only:
            row, net = run_one(cfg, seed, loaded_for(cfg))
            field = param.split(".", 1)[1]
            for value in values:
                setattr(net.cfg, field, value)
                record(value, dict(row, **_score(net)), net)
        else:
            for value in values:
                c = copy.deepcopy(cfg)
                if param:
                    set_path(c, param, value)
                record(value, *run_one(c, seed, loaded_for(c)))
    info["elapsed_s"] = time.perf_counter() - t0
    result = {"provenance": info, "config": cfg,
              "power": {"parameters": power.table(
                            cfg["network"].get("dt_ms", NetConfig.dt_ms) * 1e-3),
                        "placeholders": power.placeholders(),
                        "metadata": power.metadata},
              "summary": summarize(runs), "runs": runs}

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    cols = list(dict.fromkeys(k for row in result["summary"] for k in row))
    with open(os.path.join(out, "summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(result["summary"])
    log(f"wrote {out}")
    result["out_dir"] = out
    return result
