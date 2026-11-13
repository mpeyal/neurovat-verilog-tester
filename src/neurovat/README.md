# neurovat

Vectorized simulator for spiking networks whose synapses are a crossbar of
analog memory devices. It is the GUI-free core of the NeuroVAT test bench:
NumPy only, seeded, and driven by config files so results can be regenerated.

- **Network** — crossbar(s) of devices, LIF neurons, winner-take-all
  inhibition, adaptive thresholds, optional hidden layers.
- **Learning** — device-local STDP (pre-synaptic eligibility trace, one
  consolidated write pulse per synapse per presentation) or surrogate-gradient
  BPTT applied through the devices.
- **Encoding** — rate (Poisson), temporal, latency and column-scan
  (spatio-temporal) codes, with input noise, spike-time jitter and background
  activity. Column scan (`encoding = "scan"`) sweeps the image left to right:
  one input per pixel row, one time bin of `scan_bin_ms` per pixel column, a
  spike where the pixel is at or above `scan_threshold`. Patterns whose rows
  hold the same amount of ink look alike to it once integrated (all vertical
  bars and both diagonals of the `bars` set do). The crossbar trainer does
  not yet learn scanned patterns well: over 15 epochs and 3 seeds, letters
  and digits reach 25-42 % against 92-100 % with rate coding, although every
  pattern there has a distinct per-row ink profile. The stream benchmark
  has its own image codes in `encoding.stream_events`: column scan, row
  scan, latency and frozen Poisson (see "Stream encodings" below).
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
patterns and carries the weights over.

Pass a `DeviceArray` as `device` to store the weights as conductances. A
requested weight change becomes a pulse whose amplitude is `write_gain` (or
`dep_gain`) times the change, and `write_window_ms` merges the updates of one
window into one pulse per synapse. The gain decides everything: a pulse that
moves the device far less than one STDP step leaves the weights frozen.
`calibrate_write` finds the amplitudes at which one pulse on a device at
`w_init` equals one step (`a_plus` up, `a_minus` down):

```python
from neurovat.masquelier import calibrate_write

cal = calibrate_write(device, P, width=1e-3, amp0=1.0)   # amp0: usual drive level
s = Session(P, 1, device, write_window_ms=50.0,
            **{k: cal[k] for k in ("write_gain", "dep_gain", "pot_sign", "pulse_width")})
```

Larger requests are scaled linearly from that point, which a real device
does not follow exactly, so device runs remain experiments. With calibrated
pulses and 1000 afferents the built-in `softbound` synapse does not become
selective in 60 s: its weights stay in the middle of the range. The
benchmark takes `--device softbound --calibrate`. The GUI's Trainer exposes
all of this as the learning mode **unsupervised stream**.

The stream can also hide scanned images (`make_event_stream`, with
`encoding.scan_events`: one afferent per pixel row, pixel column c fires at
c x bin ms). `encoding.stream_events` also has a row scan (one afferent per
pixel column) and two codes with one afferent per pixel, latency and frozen
Poisson (see "Stream encodings" below). The GUI's image input sets the threshold to 0.85 x the lowest
peak potential one clean presentation drives at the STARTING weights
(`pattern_peak`; 0.85 is a free choice) and counts a window as answered when
the neuron fires at least once in it, since it answers an image with several
spikes. Because the threshold is tied to the starting weights, a neuron that
answers the image windows may have learnt nothing: compare with the
untrained baseline (a fresh `Session` run with `learn=False`; in the GUI,
Test before Train).

Measured with the first MNIST "0" (28 rows, 10 ms per column, a 280 ms
section, even spacing with `pattern_frac` 0.5, `bg_hz` 10 in the empty
sections, 1 ms jitter, one neuron), scored on 30 s test runs (the last
third: 17 image windows, 18 empty windows). The false windows are the empty
ones, where `noise_hz` adds to `bg_hz`.

- The untrained network did not depend on `w_init` (all weights equal, the
  threshold scaled to the peak at `w_init`). Measured on this "0": 30 s
  untrained at 0.475 and at 1.0, `noise_hz` 10, one neuron, seeds 1 and 2,
  gave the same spikes (8 and 7 of 18 false, 17/17 hits); 10 s with three
  neurons (seed 1) too. With `noise_hz` 0 the untrained neuron is clean
  (0/18 false; measured at 0.475).
- `w_init` 0.475 (the paper value; measured only at this start weight):
  after 60 s of training 17/17 hits, 0-4/18 false at noise 0 and 18/18 at
  noise 10. STDP potentiates the weights on the digit's rows while the
  threshold stays put, so training makes it worse.
- `w_init` 1.0 (threshold 0.85 x the peak at 1.0), `noise_hz` 10, three
  seeds: untrained 17/17 hits and 8, 7, 11 of 18 false; after 60 s of
  training 17/17 hits and 0-2 of 18 false. Weights can only fall from 1.0,
  so STDP prunes the false answers, as in Masquelier. The first-spike
  latency does not change (about 77-81 ms after onset). This is why the
  GUI's image input starts at weight 1.0.
- Longer training (`w_init` 1.0, seed 1, in the GUI): 300 s of training,
  then a 30 s test: 17/17 hits and 0/18 false (untrained 8/18). In the live
  2 s chunks of the training run every pattern window was answered for all
  300 s, and the share of empty windows answered fell to 0 after about 60 s
  of training (one later chunk, at 78 s, at 33 %). Weights at 300 s: 16 of
  28 above 0.9, 11 below 0.1.
- The first MNIST "1" (`w_init` 1.0, `noise_hz` 10, in the GUI, 60 s of
  training): untrained 17/17 hits and 18 / 16 of 18 false, trained 17/17 and
  1 / 2 of 18 (seeds 1 / 2).
- ECFET v2 synapses (`verilog/ecfet_v2.va` twin, calibrated pulses, seed 1,
  otherwise as above, in the GUI): untrained 17/17 hits and 8/18 false,
  after 60 s of training 17/17 and 1/18. The pulse is calibrated at weight
  0.5 (at 1.0 one direction cannot move; 0.1 / 0.9 / 0.5 are free choices):
  there one pulse makes one STDP step each way; at weight 1.0 the
  potentiation pulse makes no step and the depression pulse about one. The
  weights ended between 0.50 and 1.00 (15 of 28 above 0.9). One run only.
- Built-in 28 x 28 images of `make_patterns` (digits '0', '1', '7' and
  letters 'L', 'T') at `w_init` 1.0, `noise_hz` 10 and 20, seeds 1-2, 60 s
  of training: clean untrained (0/18 false; once 2/18, the '7' at noise 20)
  and after training (0/18), 17/17 hits throughout. Their clean peak at
  weight 1 is far higher than the MNIST "0"'s (about 36 for the digit '0'
  and 50 for the 'L', against about 17), so the threshold sits far above the
  noise; they show detection, not learning. So does the 14 x 14 built-in
  digit '0' at noise 0 and 10.

Known limit: several images are not separated by several neurons (every
image drives the same rows). MNIST "0", "1", "2" hidden, 3 neurons, `w_init`
1.0, `noise_hz` 10, seed 1, 120 s of training, 30 s tests (5-6 windows per
digit scored): neurons 1 and 3 both answer the "0" best (5/5 hits each, with
9 and 17 of 30 other windows), neuron 3 also answers the "1" and "2" (about
2-4 spikes per window of each), neuron 2 answers 3 of 6 "2" windows, and no
neuron takes the "1". One run. A neuron for one image also answers other
images shown as distractors. With the other
MNIST digits 1-9 as distractors (`w_init` 1.0, `noise_hz` 0): untrained 13
and 16 of 18 false windows, after 120 s of training 14 and 14 of 18 - no
learning.

#### Robustness sweep

`vatester.stream_robust.sweep` (the GUI: the Trainer's stream mode,
Robustness tab) tests a trained network while ONE input factor at a time
departs from its nominal value: spike-time jitter, missing (deleted) pattern
spikes, and extra (noise) spikes, at the levels of the 2008 paper's
robustness panels (jitter 1 / 3 ms, 10 / 30 % deletion, 5 / 26 Hz) plus the
value the network was trained with. It is recall: every level runs with
STDP off and the threshold held, on one fixed keyed stream
(`Session.run(stream_key=...)`), so the trained and untrained networks see
the same stream, a repeat sweep gives identical numbers, and the session
(weights, runs, stream, device) is left as it was. The untrained row is a
fresh copy at the start weights with the same patterns. Every window of the
whole run is counted (no learning, so no burn-in): a different denominator
from Test's score over the last third, so the two do not compare directly.
The extra-spike levels are the absolute noise rate, not added to the
nominal.

Measured: trained column scan of the first MNIST "0" (28 rows, `w_init`
1.0, one neuron, seed 1, even spacing, share 0.5, `bg_hz` and `noise_hz`
10, 1 ms jitter, 60 s of training with no Test before it), 30 s per level,
53 image windows and 54 other windows per run; * = nominal. (With the
untrained Test run first, the 60 s of training see a different stream; that
network gave 0/54 false at every level except 26 Hz, and 44 and 9 hits at
30 and 50 % missing spikes. The untrained column was the same.)

| factor | level | trained hits | trained false | untrained hits | untrained false |
|---|---|---|---|---|---|
| jitter | 1* / 3 / 5 ms | 53 / 53 / 53 | 3 / 3 / 3 | 53 / 53 / 53 | 25 / 25 / 25 |
| missing spikes | 0* / 10 / 30 / 50 % | 53 / 53 / 46 / 7 | 3 at each | 53 / 53 / 53 / 41 | 25 at each |
| extra spikes | 5 / 10* / 26 Hz | 53 / 53 / 53 | 0 / 3 / 54 | 53 / 53 / 53 | 2 / 25 / 54 |

- At 26 Hz both networks answer every window: the threshold is held and
  there are only 28 inputs (the paper's neuron had 2000).
- The trained network keeps its hits less well at 30 % deletion than the
  untrained one, because STDP pruned its weights.
- With 10 ms per column, jitter up to 5 ms does not change any window's
  answer.

**Retrain at each level.** Recall asks how the trained network copes with a
worse input; retrain (`sweep(..., retrain=True, train_seconds=T,
repeats=N)`, the tab's "retrain at each level") asks what a network trained
*under* that input learns. Per level, a fresh copy (start weights, the same
frozen patterns, the threshold held) is trained with STDP on for T s at that
level on one keyed training stream (the same key for every level, so only
the factor differs), then tested at that level with STDP off on the sweep's
test stream. The user's network is never run. The nominal row is "retrained
at nominal": an independent training on another stream, not the network you
trained. N trainings per level (keys (2009, r)) give a mean and a min-max.
Device synapses are refused (a device copy cannot be made).

Measured: MNIST "0", size 28, column scan, 3 neurons, 60 s of training,
30 s test per level, ONE training run per level (611 s wall; recall took
227 s). Hits of 45 image windows / false of 62 other windows:

| level | recall: hits / false | retrain: hits / false |
|---|---|---|
| jitter 1 ms (nominal) | 45 / 8 | 45 / 10 |
| jitter 3 ms | 45 / 8 | 45 / 6 |
| jitter 5 ms | 45 / 8 | 45 / 4 |
| missing spikes 10 % | 45 / 8 | 45 / 10 |
| missing spikes 30 % | 44 / 8 | 45 / 15 |
| missing spikes 50 % | 23 / 8 | 34 / 22 |
| extra spikes 5 Hz | 45 / 0 | 45 / 0 |
| extra spikes 26 Hz | 45 / 62 | 45 / 62 |

The clearest change is at 50 % missing spikes: a network trained there
keeps more hits (23 -> 34 of 45) but answers more empty windows (8 -> 22 of
62). With one training run per level, differences of a few windows are not
trends.

#### Stream encodings

How the image is turned into spikes decides what STDP can learn. Measured
on the first MNIST "0", untrained -> trained (60 s of training), 30 s tests
with STDP off, counted over the whole test run (not Test's last-third
score); hits held unless stated.

- Column scan (28 inputs = pixel rows), noise and background 10 Hz: false
  windows 25-27 -> 2 of 54 (two seeds, even spacing). Random spacing at
  share 0.4: 27/64 -> 5/61.
- Row scan (28 inputs = pixel columns), 10 Hz: 40-44 -> 2 of 54. Random
  spacing at 0.4: 53/64 -> 4/61.
- Latency (784 inputs, one per pixel; a brighter pixel fires earlier within
  a 50 ms window after a 15 ms lead margin): at 10 Hz the neuron fires in
  every window (188/188 -> 188/188 false), nothing learnt; at 1 Hz it is
  clean untrained (0 false), nothing to learn; at 3 Hz 188/188 -> 1/188 on
  both seeds (random spacing at 0.4: 226/226 -> 1/228); at 5 Hz 188/188 ->
  0-4/188. The GUI defaults latency to 3 Hz noise and background. It
  needs the longer training: 30 s of training in the GUI gave only
  63/63 -> 39/63 false windows.
- Frozen Poisson (784 inputs, one fixed draw at 100 Hz x intensity over 50
  ms, about 5 spikes per inked line) starts at weight 0.5, not 1, and the
  GUI sets its noise and background to 10 Hz. From weight 1 the hits
  collapsed after training: STDP depressed each line's spikes after the
  output spike, the digit's own weights fell to a mean of 0.51-0.77, and
  the clean image's peak at the trained weights fell below the threshold
  (0.64-0.96 x), so the neuron stopped firing on it and STDP stopped. From
  0.5 the digit's weights stayed near their start (mean 0.44-0.51) while
  the other lines fell (mean 0.02-0.30), and the peak stayed at 1.10-1.27 x
  the threshold. The image is clean untrained at 1-5 Hz (nothing to learn), and
  learns at 10 Hz.

Frozen Poisson fix search. MNIST "0" (`x_train/255`, size 28, ink 0.5),
one neuron, random spacing, share 0.4, jitter 1 ms, `noise_hz` = `bg_hz`,
auto threshold. Protocol: untrained 30 s Test on a fresh session, then
60 s of training, then a 30 s Test, all whole-run counts. Cells: 1, 3, 5,
10 Hz x stream seeds 1, 2 x frozen draws `[d, 0]` (`d` = 0, 1; the GUI
draws image i with `[app seed, i]`). A setting passes when, in every cell,
the trained hit rate is at least 95 % of the untrained one and the false
windows fall wherever the untrained network had any. "4 cells" = screened
on stream seed 1, draw 1 at each Hz and rejected at the first failure;
other counts are runs cut short. Every rejected setting lost hits in at
least one cell, and the failing cell moves with the setting (no noise
range works). Hits compare as rates: the Tests are different stretches of
stream, so there are 149 or 154 image windows untrained and 147 or 148
trained.

| setting | cells | hits held | worst trained hits (cell) | peak / thr after training | result |
|---|---|---|---|---|---|
| as before (thr 0.85, 100 Hz, w0 1) | 16 | 3 of 16 | 9/147 (3 Hz, draw 1) | 0.64-0.96 | fail |
| auto threshold 0.70 x peak | 13 | 7 of 13 | 35/147 (1 Hz, draw 1) | 0.74-1.16 | fail |
| auto threshold 0.60 x peak | 4 | 3 of 4 | 0/147 (10 Hz) | 0.41-0.97 | fail |
| rate 40 Hz | 4 | 1 of 4 | 0/147 (5 Hz) | 0.20-1.16 | fail |
| first spike of each line only | 7 | 5 of 7 | 114/148 (1 Hz); 10 Hz: false 226 -> 228 | 0.70-1.16 | fail |
| one spike per inked pixel, uniform time | 16 | 8 of 16 | 0/147 (3 Hz); 5-10 Hz: false never fall | 0.02-1.17 | fail |
| start weight 0.7 | 4 | 1 of 4 | 104/147 (10 Hz) | 0.86-1.01 | fail |
| start weight 0.6 | 4 | 4 of 4 | 147/147 | 0.95-1.05 | screen passed, peak at the threshold |
| **start weight 0.5** | **16** | **16 of 16** | 147/147 | **1.10-1.27** | **pass** |
| start weight 0.5, 180 s training | 4 | 4 of 4 | 147/147 | 1.20-1.26 | pass |
| start weight 0.4 | 4 | 4 of 4 | 147/147 | 1.15-1.31 | screen passed |

Start weight 0.5, per cell (untrained -> trained):

| noise Hz | stream seed, draw | hits | false windows | peak / thr |
|---|---|---|---|---|
| 1 | 1, 0 / 1, 1 / 2, 0 / 2, 1 | 149 -> 147 of 147, 149 -> 147, 154 -> 148 of 148, 154 -> 148 | 0 -> 0 in all four | 1.27 / 1.24 / 1.26 / 1.25 |
| 3 | same | same | 0 -> 0 | 1.23 / 1.10 / 1.26 / 1.21 |
| 5 | same | same | 0 -> 0 | 1.23 / 1.19 / 1.23 / 1.23 |
| 10 | same | same | 226 -> 0 of 228, 226 -> 0, 221 -> 0 of 227, 221 -> 0 | 1.18 / 1.20 / 1.20 / 1.20 |

Free parameters chosen here: the 50 ms window, the 15 ms margin,
the 100 Hz rate, the frozen draw's seed (0 in `stream_events` when no rng is
given; the GUI draws from the app seed, per image), the 3 Hz pixel-code
noise default (latency), and frozen Poisson's start weight 0.5 and 10 Hz
noise default (`stream_view.W0_CODE`, `CODE_NOISE_HZ`).

The MNIST benchmark trains and scores the pre-package engine
(`benchmarks/legacy_neuro.py`) and this package on the same MNIST subset from
the same seed, and reports wall time and the difference in the trained
conductances and predictions.

## Citing

See `CITATION.cff` in the repository root.
