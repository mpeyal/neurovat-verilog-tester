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
neurovat run configs/sweep_energy_accuracy.toml
```

A config names the dataset, network, neuron, learning rule, device and one
swept parameter (`device.variation.<param>`, `device.write_noise`,
`network.jitter_ms`, `network.input_noise`, ...). `sweep.apply = "test"` trains
once per seed and re-scores under each value. Each run writes, under
`results/sweeps/<name>/`:

- `config.json` — the resolved config; re-running it reproduces the results,
- `results.json` — every run (accuracy clean and noisy, validation accuracy,
  per-epoch history, training time, write count, write effort, energy
  breakdown of initialisation / training / evaluation) with provenance (git
  commit, versions, dataset SHA-256) and the power parameters used,
- `summary.csv` — mean and seed-to-seed standard deviation per swept value,
  accuracy and energy side by side,
- `energy/<run>.npz` and `energy/<run>_train.png` / `_eval.png` — per-synapse
  and per-neuron energy maps (`output.energy_maps = false` turns them off).

`write_effort` is the sum of |amplitude| x pulse width over all write pulses,
in the device's drive unit. It is a programming-activity proxy; the energy
figures come from the power model below.

## Power and energy

Every `CrossbarSNN` carries an `EnergyMeter` (`net.energy`) that accumulates
energy while the network trains and evaluates. It only observes; results are
identical with or without it.

| Component | Energy |
|---|---|
| array read | `v_read^2 * G * t_read` per device, per input spike on its row, with the conductance the MVM actually used |
| array write | per programming pulse and per device: `amplitude * I * width`. `I` is `power.write_current`, else the device's `write_current()` if it has one, else ohmic `amplitude * mean(G before, G after)`. Current-driven devices: `abs(I_pulse) * v_write * width` |
| neurons | `e_spike` per spike |
| peripherals | `e_row_driver` per input spike per crossbar, `e_col_sense` per column per timestep with an active row, `e_wta` per inhibition event (reported as three lines) |
| static | `p_leak` per neuron and `p_peripheral_static`, times simulated time |

```python
print(neurovat.power.format_report(net.energy.report("eval")))
```

`report(phase)` (`"init"`, `"train"`, `"eval"`) gives the breakdown table,
energy per synaptic operation (pJ/SOP, one SOP = one input spike arriving at
one synapse), energy per neuron spike, energy per presentation (inference),
and average power at `timestep_s` (default: the simulation step
`network.dt_ms`). `net.energy.maps(phase)` returns the per-synapse and
per-neuron arrays; `power.save_maps` / `power.plot_maps` write them out.

Parameters go in the config (SI units):

```toml
[power]
v_read = 0.1                 # V
t_read = 100e-9              # s
neuron_file = "lif_energy.json"   # values extracted from the LIF circuit
```

`neuron_file` is the hook for circuit-extracted neuron values: a `.json`,
`.toml` or `.csv` (`name,value` rows) with `e_spike` (J per spike) and/or
`p_leak` (W per neuron), plus optional `source`, `corner`, `vdd`,
`temperature`, `cell`, `date`, `notes`. The path is relative to the working
directory. It is read as data only.

**No parameter has a silent default.** Anything not supplied runs on the
value in `neurovat.power.PLACEHOLDERS`, and is listed as a PLACEHOLDER in the
run log, in every report and in `results.json` (`power.placeholders`). The
counts behind each energy are stored, not the energies, so
`net.energy.params.load_neuron_file(...)` after a run re-prices it without
re-simulating.

`power.*` values can be swept like any other parameter, and every sweep
reports `infer_energy_J`, `infer_power_W`, `pj_per_sop`,
`pj_per_neuron_spike`, `train_energy_J`, `train_write_energy_J` next to the
accuracies.

Not modelled: wire and sneak-path losses, and write time in the average-power
clock (pulses are assumed to fit inside the timestep). WTA events are counted
on the STDP path only.

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
    def write_current(self, rows, cols, amps, width): ...    # optional, A

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

### Unsupervised pattern finding (Masquelier et al. 2008)

```
python benchmarks/masquelier2008.py --seeds 20 --jobs 8
python benchmarks/masquelier2008.py --seeds 4 --device softbound --write-window-ms 50
```

`neurovat.masquelier` reproduces Masquelier, Guyonneau & Thorpe (2008,
PLoS ONE 3:e1377): one SRM neuron, 2000 afferents firing continuously at
64 Hz, a 50 ms pattern on half of them recurring in a quarter of the time
under 1 ms jitter and 10 Hz noise, and nearest-spike additive STDP, for
450 s. A run succeeds when, over the last 150 s, the hit rate is above 98 %,
there is no false alarm and the mean latency after pattern onset is below
10 ms (the paper: 96 % of 100 runs). The STDP pairing and the 0.1 ms neuron
step follow the Brian replication (Hathway & Goodman 2018, ReScience 4:6);
with a 1 ms step every weight potentiates and nothing is learnt. `--device`
puts a registered device model behind each synapse and is reported, not
asserted.

Competitive version (the mechanism of Masquelier, Guyonneau & Thorpe 2009,
Neural Computation 21:1259): several neurons, several patterns, and lateral
inhibition so that each neuron takes one pattern.

```
python benchmarks/masquelier2008.py --neurons 3 --patterns 3 --seeds 10 --jobs 10
```

Each pattern sits on its own random half of the afferents and the patterns
together fill `--pattern-frac` of the time. A neuron that fires resets the
others and holds them back by `inh_alpha` x threshold, decaying with
`tau_inh`. A run succeeds when every pattern is answered by its own neuron in
at least 90 % of its occurrences with fewer than 1 false alarm per second.
At paper scale (2000 afferents, 150 s) 6 of 10 seeds learn all three
patterns and the other 4 learn two (one neuron stays silent or two share a
pattern), with no false alarm in any run.

These are choices made here, not values from the 2009 paper, so the result
is reported and never asserted: `inh_alpha = 0.4`, `tau_inh = 20 ms`, the
40 % pattern share, independent random halves, and the success test.

`Session` trains and then tests one population on the same hidden patterns:

```python
from neurovat.masquelier import MasquelierParams, Session

s = Session(MasquelierParams(n_out=3, pattern_frac=0.4, seed=1), n_patterns=3)
s.run(150.0)                                   # STDP on
r = s.run(20.0, learn=False, jitter_ms=3.0)    # frozen weights, new stream
print(r["pattern_owner"], r["false_alarms_per_s"], r["spikes_per_window"])
```

Every `run` draws new background activity, jitter and noise around the same
patterns and carries the weights over. Pass a `DeviceArray` as `device` to
store the weights as conductances (`write_gain` / `dep_gain` map a weight
change to a pulse amplitude, `write_window_ms` merges the updates of one
window into one pulse per synapse); that mapping is a free parameter and
device runs are experiments. The GUI's Trainer exposes this as the **Stream**
tab.

The MNIST benchmark trains and scores the pre-package engine
(`benchmarks/legacy_neuro.py`) and this package on the same MNIST subset from
the same seed, and reports wall time and the difference in the trained
conductances and predictions.

## Citing

See `CITATION.cff` in the repository root.
