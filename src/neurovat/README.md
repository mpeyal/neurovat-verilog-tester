# neurovat

Vectorized simulator for spiking networks whose synapses are a crossbar of
analog memory devices. It is the GUI-free core of the NeuroVAT test bench:
NumPy only, seeded, and driven by config files so results can be regenerated.

- **Network** — crossbar(s) of devices, LIF neurons, winner-take-all
  inhibition, adaptive thresholds, optional hidden layers.
- **Learning** — device-local STDP (pre-synaptic eligibility trace, one
  consolidated write pulse per synapse per presentation) or surrogate-gradient
  BPTT applied through the devices.
- **Encoding** — rate (Poisson), temporal and latency codes, with input noise,
  spike-time jitter and background activity.
- **Devices** — any model behind the `DeviceArray` interface. `softbound` is
  the built-in reference synapse; existing scalar device twins run unchanged
  through `ScalarTwinArray`.

## Install

```
pip install -e .            # from the repository root
pip install -e .[test]      # plus pytest
```

## Use

```python
from neurovat import CrossbarSNN, NetConfig, NeuronParams, STDPParams, device_factory

net = CrossbarSNN(
    device_factory("softbound", variation={"tau_w": 0.1}),
    NeuronParams(), STDPParams(pot_amp=0.05, dep_amp=0.05),
    NetConfig(grid_h=4, grid_w=4, n_out=4, pattern_set="bars", seed=1))
history = net.fit(epochs=5)
print(net.eval_accuracy(), net.eval_accuracy(noisy=True))
```

`neurovat.data.load_dataset("mnist.npz", 4, 4)` turns any keras-style `.npz`
(MNIST, Fashion-MNIST, CIFAR-10; colour is converted to grayscale) into
balanced, seeded train / validation / test splits at the network's pixel grid.

## Experiments and sweeps

```
neurovat run configs/crossbar_16x10.toml
neurovat run configs/sweep_device_variation.toml
neurovat run configs/sweep_jitter.toml
neurovat run configs/sweep_noise.toml --set train.epochs=3 --set "sweep.seeds=[1,2,3,4,5]"
```

A config names the dataset, network, neuron, learning rule, device and one
swept parameter (`device.variation.<param>`, `device.write_noise`,
`network.jitter_ms`, `network.input_noise`, ...). `sweep.apply = "test"` trains
once per seed and re-scores under each value. Each run writes, under
`results/sweeps/<name>/`:

- `config.json` — the resolved config; re-running it reproduces the results,
- `results.json` — every run (accuracy clean and noisy, validation accuracy,
  per-epoch history, training time, write count, write effort) with
  provenance (git commit, versions, dataset SHA-256),
- `summary.csv` — mean and seed-to-seed standard deviation per swept value.

`write_effort` is the sum of |amplitude| x pulse width over all write pulses,
in the device's drive unit. It is a programming-activity proxy, not a power
model.

## Adding a device model

```python
from neurovat import DeviceArray, register_device

class FTJArray(DeviceArray):
    def __init__(self, n_post, n_pre, params=None, variation=None,
                 write_noise=0.0, seed=None): ...
    def conductance(self): ...                 # (n_post, n_pre) Siemens
    def program(self, rows, cols, amps, width, t=0.0): ...   # -> t after
    def state_dict(self): ...
    def load_state_dict(self, state): ...
    def nominal(self): ...                     # fresh 1x1 nominal device

register_device("ftj", FTJArray)               # then: [device] model = "ftj"
```

The network reads `conductance()` and calls `program()`; switching kinetics,
bounds, variation and write noise belong to the device.
`neurovat/devices/softbound.py` is a complete example, and
`tests/test_devices.py` checks every registered model against the interface.

## Tests and benchmark

```
python -m pytest
python benchmarks/bench_mnist.py --train 1000 --test 1000
```

The benchmark trains and scores the pre-package engine
(`benchmarks/legacy_neuro.py`) and this package on the same MNIST subset from
the same seed, and reports wall time and the difference in the trained
conductances and predictions.

## Citing

See `CITATION.cff` in the repository root.
