"""Speed and equivalence of the neurovat core against the pre-package engine.

    python benchmarks/bench_mnist.py [--train 200] [--test 200] [--out results/benchmark]

``legacy_neuro.py`` is the frozen per-device Python-loop engine. Each case
trains both engines on the same balanced MNIST subset in the same order from
the same seed, then scores the same held-out images, and reports wall time,
the largest conductance difference after training and how many predictions
differ.

  twins       both engines step the same scalar twin per synapse; the speedup
              is the vectorized network (the device loop remains)
  vectorized  the new engine uses a vectorized DeviceArray for the same model
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).resolve().parent)]

import numpy as np

import neurovat
from neurovat import CrossbarSNN, NetConfig, NeuronParams, STDPParams, data
from neurovat.devices import device_factory, twin_factory
import legacy_neuro
import simulation_core as sc


def devices():
    """name -> (scalar twin factory, write amplitude, vectorized factory|None)."""
    out = {"ecfet_v2": (lambda: sc.EcfetV2(sc.V2Params(A_stdp=0.)), 50e-12, None)}
    if sc.FeFET is not None and "fefet" in neurovat.available_devices():
        p = sc.FeFETParams()
        out["fefet"] = (lambda: sc.FeFET(sc.FeFETParams()), float(p.Vc),
                        device_factory("fefet"))
    return out


def run(net, order, test, detail=None):
    kw = {} if detail is None else {"detail": detail}
    t0 = time.perf_counter()
    for k in order:
        net.train_step(net.patterns[k][1], int(net.target_of[k]), **kw)
    t1 = time.perf_counter()
    net.test_patterns, net.test_targets = test
    _, pred = net.evaluate(use_test=True)
    return {"train_s": t1 - t0, "eval_s": time.perf_counter() - t1,
            "G": net.weights_S(), "pred": pred}


def case(name, grid, rule, dev, ds, seed=17):
    make, amp, vec = dev
    pats, tgts = data.as_patterns(ds.train, ds.class_names)
    test = data.as_patterns(ds.test, ds.class_names)
    N = NeuronParams()
    S = STDPParams(pot_amp=amp, dep_amp=amp, sg_lr=.1)
    cfg = NetConfig(grid_h=grid, grid_w=grid, n_out=10, present_ms=60.,
                    learn_rule=rule, seed=seed)
    order = np.random.default_rng(seed).permutation(len(pats)).tolist()
    old = run(legacy_neuro.Trainer(make, "", N, S, cfg, pats, tgts), order, test)
    new = run(CrossbarSNN(vec or twin_factory(make), N, S, cfg, pats, tgts),
              order, test, detail=False)
    span = float(old["G"].max() - old["G"].min()) or 1.0
    row = {"case": name, "crossbar": f"{grid * grid}x10", "rule": rule,
           "device": "vectorized" if vec else "twins",
           "n_train": len(pats), "n_test": len(test[0]),
           "legacy_train_s": old["train_s"], "new_train_s": new["train_s"],
           "train_speedup": old["train_s"] / new["train_s"],
           "legacy_eval_s": old["eval_s"], "new_eval_s": new["eval_s"],
           "eval_speedup": old["eval_s"] / new["eval_s"],
           "max_dG_S": float(np.abs(old["G"] - new["G"]).max()),
           "max_dG_rel": float(np.abs(old["G"] - new["G"]).max() / span),
           "pred_mismatch": int(np.sum(np.array(old["pred"]) != np.array(new["pred"]))),
           "test_acc": float(np.mean(np.array(new["pred"]) == np.array(test[1])))}
    print(f"{name:10s} {row['crossbar']:>7s} {rule:9s} {row['device']:10s} "
          f"train {old['train_s']:7.2f}s -> {new['train_s']:6.2f}s "
          f"({row['train_speedup']:5.1f}x)  eval {old['eval_s']:6.2f}s -> "
          f"{new['eval_s']:5.2f}s ({row['eval_speedup']:5.1f}x)  "
          f"max dG {row['max_dG_rel']:.1e} of range, "
          f"{row['pred_mismatch']} preds differ", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", type=int, default=200, help="training images")
    ap.add_argument("--test", type=int, default=200, help="test images")
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmark"))
    args = ap.parse_args()
    path = data.fetch_mnist(str(ROOT / "datasets"))
    rows = []
    for grid in (4, 14):
        ds = data.load_dataset(path, grid, grid, args.train // 10, 0,
                               args.test // 10, seed=17)
        for name, dev in devices().items():
            for rule in ("surrogate", "stdp"):
                rows.append(case(name, grid, rule, dev, ds))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_mnist.json").write_text(json.dumps({
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "data_sha256": data.MNIST_SHA256, "rows": rows}, indent=2))
    print(f"wrote {out / 'bench_mnist.json'}")


if __name__ == "__main__":
    main()
