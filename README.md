# NeuroVAT — Neuromorphic Verilog-A Test Bench (Python)

Fast iteration on neuromorphic synaptic device models (ECFET / ECRAM /
FeFET) without Cadence. Three built-in ECFET behavioral models are included:

| | |
|---|---|
| **v1** ([simulation_core/model_v1.py](simulation_core/model_v1.py)) | Faithful Python port of the original Verilog-A : use it to reproduce/debug Spectre behavior |
| **v2** ([simulation_core/model_v2.py](simulation_core/model_v2.py)) | Upgraded practical ECFET model — tune it here, then take [ecfet_v2.va](verilog/ecfet_v2.va) (same equations) back to Virtuoso |
| **v3** ([simulation_core/model_v3.py](simulation_core/model_v3.py)) | Paper-faithful ECFET model with conserved intercalation state and saturating LTP/LTD — twin of [ecfet_v3.va](verilog/ecfet_v3.va) |

**Your own devices (FeFET, FeCAP, RRAM, ...):** add a `.va` source to `verilog/`
and its Python twin to `twins/` (see [twins/README.md](twins/README.md)); the app
registers it at startup, or the built-in agent can build the twin for you.
Voltage-driven FeFET/FeCAP twins get the voltage drive, STDP and polarization
views. The authors' own FeFET and FeCAP models are private and not included.

## GUI

**One-click setup:** install Python 3.9 or newer from
[python.org](https://www.python.org/downloads/), then double-click
`setup_windows.bat` (Windows) or `setup_mac.command` (macOS). The first run
creates a `.venv` and installs the requirements; later runs just start the app.
If macOS blocks the file, right-click it and choose **Open**, or run
`chmod +x setup_mac.command` once in Terminal. The app is developed on Windows;
on macOS it uses DearPyGui's built-in font in place of Segoe UI.

**Manual setup:**

```
pip install -r requirements.txt
python run_gui.py
```

Native Python desktop app using DearPyGui; no browser or web server is needed.

Keep Verilog-A source models in **`verilog/`**. The application discovers this
folder automatically; Python behavioral twins stay in `simulation_core/` and `twins/`.
The source editor's **Save as** dialog and local copies imported from Virtuoso
default to `verilog/`. Launch from the project root with `python run_gui.py`.
Older workspaces with `.va` files at their root are still supported.

### Code organization and development checks

| Location | Responsibility |
|---|---|
| `run_gui.py` | Desktop entry point; workspace and launch options |
| `simulation_core/`, `twins/`, `verilog/` sources | Device equations, simulation engine and source models |
| `vatester/app.py` | Application state, lifecycle, worker dispatch and workspace assembly |
| `vatester/registry.py` | Built-in and plugin device/pattern registration; reload helpers |
| `vatester/design_system.py`, `ui_constants.py`, `desktop_layout.py` | Theme, labels, display units and viewport layout |
| `vatester/plot_interaction.py` | Shared zoom, probes, legends and plot menus |
| `vatester/network_studio.py` | Network architecture controls and Verilog-A source binding |
| `vatester/network_data.py` | Dataset controls, custom patterns and encoding previews |
| `vatester/network_visualization.py` | Network, cell, weight and activity rendering |
| `src/neurovat/` | GUI-free simulation core (crossbar, LIF, STDP, surrogate BPTT, device interface, sweeps) — see [src/neurovat/README.md](src/neurovat/README.md) |
| `configs/`, `benchmarks/`, `tests/` | Experiment configs, legacy-vs-core benchmark, pytest suite for the core |
| `vatester/neuro.py`, `trainer_state.py` | GUI `Trainer` on top of the `neurovat` core; state serialization |
| `vatester/analysis.py`, `stdp_protocol.py`, `compute.py` | Device measurements, timing experiments and compute calculations |
| `tools/`, root `test_*.py` | Research utilities and regression checks |
| `results/`, `datasets/`, `saved_model/` | Research outputs, downloaded data and saved networks |

The UI modules retain the existing `App` callbacks. They share the application's
state and send worker results through its queue; they do not import the GUI
coordinator. Device equations and training calculations remain in their engines.
UI changes require an app restart; device and analysis hot reload remains available.

Run the numerical regression suite:

```powershell
python -m unittest test_compute test_trainer_integrity test_stdp_protocol test_snn_builder -v
```

Run the simulation-core tests (`pip install pytest`):

```powershell
python -m pytest
```

Run the native desktop checks (each opens a test window briefly):

```powershell
python test_gui_logic.py
python test_gui_builder.py
python test_gui_compute.py
python test_design_system.py
```

They do not rerun the full MNIST/Kaggle benchmarks. Existing
research outputs are retained; saved networks and local scratch outputs are ignored
by Git. Continue to run `python selftest.py` after device-physics changes.

### Device compute experiments

Open **Device Lab > Compute**, choose a mode, open **Configure**, then use
**Run compute** or F5. Both plots fit the viewport. Long settings scroll inside
the anchored configuration popup. This uses the existing device library, rather
than adding another network builder.

- **Matrix multiply:** enter weights `A [outputs, inputs]` and vectors
  `X [inputs, vectors]`. Rows accept newlines or semicolons; entries accept spaces
  or commas. Up to 64 x 64 weights and 512 vectors are supported. Differential
  device pairs implement signed weights. Compare decoded outputs with NumPy's
  `A @ X`, with error plots, conductance quantization and seeded mismatch.
- **Logic / ALU:** exhaustively check NAND, AND, OR, XOR, NOT or a 1-4 bit
  ripple-carry adder, including carry-in and carry-out. The modeled circuit uses
  device-weighted NAND scores and ideal threshold comparators. Failure counts
  reflect the computed outputs; imposed mismatch can cause failures. This is
  functional verification, not a transistor-level ALU timing or hazard check.
- **Parallel banks:** choose replicated banks, read duration and overhead to
  estimate batch latency, throughput, resource count and scheduling speedup.
  Bank replicas share a representative mismatch map. These are hardware
  scheduling estimates, not measured parallel CPU/GPU performance.
- **Energy:** matrix/logic runs calculate array read energy from both signed
  branches using `sum(V^2 * G) * read_duration`. Enter a peripheral energy
  estimate per matrix vector or logic gate read; zero excludes it. **Analysis /
  assumptions** shows energy, accounted average power and throughput.
  **Terminal energy** separately integrates supplied CSV samples with headers
  `t_s,v_1,i_1` (optional matched `v_2,i_2`, etc.), in seconds, volts and amps.
  Currents are positive into the device. Net terminal energy includes energy
  storage/return and is not necessarily dissipated heat. All relevant terminals
  must be supplied for a complete balance; the initial trace is an example.

Matrix/logic runs require a mapped ECFET v1/v2/v3 (or your own FeFET) `.va` source and use
the current inspector parameters. **Inspect device parameters** selects the
matching inspector without overwriting edits. They initialize ideal conductance
states, quantize them and impose optional mismatch; they do not simulate
programming pulses, programming energy, parasitic wiring or nonlinear read bias.
Python twin execution is not a Verilog-A solver. Capacitor-type devices such as a
FeCAP do not provide this conductance mapping, but their externally supplied
terminal traces can be analyzed in energy mode.

**Export results** writes JSON plus CSV under `results/compute/`. JSON retains
settings, numerical outputs, assumptions, device parameters and source SHA-256
(or the supplied input trace). Logic CSV includes each input case and pass/fail;
matrix CSV includes every output/vector pair and current.

Checks: `python -m unittest test_compute -v` and `python test_gui_compute.py`.

### Network Studio

One network workspace handles architecture, data, training, simulation and analysis.
Every network requires a `.va` source mapped to a supported conductance-based
Python twin. The source picker shows the exact file and twin; saved models include
its SHA-256 fingerprint. A changed or missing source blocks training/evaluation
until rebuild, and blocks loading an incompatible saved network. Legacy saves
without a fingerprint require a currently available source for their twin.

1. Select a **Verilog-A synapse**. **Parameters / source** opens its inspector.
   Device parameters are captured when the network is built. Selecting a source
   does not overwrite parameter edits; use Device Lab's Apply parameters action
   to import parsable source defaults when desired.
2. Use **Architecture** to add, resize, reorder or remove hidden Dense/LIF layers.
   Input size, output count, neuron settings and learning controls are in the
   left inspector. **Build network** applies the settings to the single trainer.
3. Choose patterns or a dataset in **Data**. **Configure source** opens custom
   painting or import controls; **Noise settings** opens encoding perturbations.
   The clean/noisy spike previews fit the remaining window height. Training
   metrics appear only in **Results**, as a compact summary. Then **Train** with device-local
   STDP or surrogate gradients. **Test** evaluates the built network.
4. Inspect **Simulation**, **Neuron**, **Cell**, **Weights**, **Raster**,
   **Activity** and **Results**. Save/load the network through **Manage network**.

Training executes Python behavioral twins, not a Verilog-A circuit solver.
File-to-twin mapping alone does not verify equation equivalence, so validate
your own twins against their sources.
Capacitors and unmapped sources are not accepted as conductance synapses.

The separate ideal-weight Model Builder has been removed from the GUI. Dense
crossbars and LIF layers are currently supported (up to four hidden layers of
64 neurons). Device-backed convolution/pooling and a complete snnTorch feature
replacement are not yet implemented. The tensor reference utilities and previous
results remain available for research; they are not a second app training mode.
Previously stopped full MNIST/Kaggle benchmarks have not been restarted.

Offline desktop verification: `python test_gui_builder.py`.

Features:

* **Auto-detected Verilog-A files** — every `.va` in the workspace is scanned
  (module name, `parameter` declarations) and mapped to its Python twin;
  parameters can be applied to the model with one click and the source can
  be viewed/edited in-app.
* **Neuromorphic signal designer** — single spikes, pulse trains, LTP/LTD,
  paired pulses (PPF), Poisson spike trains, bursts, STDP pre/post pairs,
  amplitude staircases, and free-form custom patterns; current or voltage
  mode with pA…A / mV…V units.
* **Live plots** — stimulus, R(t), G(t) with linked axes, plus an Analysis
  tab with the G-vs-pulse-number synaptic curve and per-branch ΔG metrics.
* **Claude agent chat** (right panel) — generates spike patterns (loadable
  into the designer with one click), explains/reviews the `.va` sources, and
  can modify them when "allow file edits" is checked. Uses the `claude` CLI
  headless mode when installed (falls back to the Anthropic SDK +
  `ANTHROPIC_API_KEY`).
* CSV export (for Spectre comparison) and matplotlib PNG export; `F5` runs.

## Desktop workspaces

Run `python run_gui.py` to open the native Python desktop application.

The two workspaces share a slate-and-teal desktop theme, consistent controls
and typography. Switch between Device Lab and Network Studio without losing the
built network. F5 follows the active workspace.
The header uses one compact navigation row. Plot canvases resize to the available
window height; the main plot area does not scroll. STDP shows its spike preview
and timing curve together, with controls in **Experiment settings**. Long parameter
lists and experiment settings scroll independently. Transient, noise, raster and
training plots share bounded, resizable canvases.

The teal **Run** button stays beside the workspace tabs. Its label and F5 action
follow the active page: simulation, STDP, measurement, noise analysis, compute or training.
Plot **View options** and **Manage network** open beside their buttons and contain secondary actions. Dialogs
use right-aligned actions; longer forms scroll above their action footer.

Known ideal network primitives remain available as Verilog-A source assets and
do not trigger the missing-device-twin prompt. Other unmapped device models still
offer explicit twin generation through the file-selection dialog.

- **Device Lab:** choose a device, configure its stimulus, simulate transients,
  and inspect pulse response, STDP, polarization, noise and retention. Source
  and logs remain accessible in the same workspace.
- **Network Studio:** choose training data, configure the network and learning
  rule, then build, train and evaluate. Network, neuron, cell, weight, raster
  and activity views expose the existing simulation and analysis tools.
- **Parameter inspector:** search names, model comments or groups; filter to
  modified values; hover a name for its original symbol, definition and default.
  Reset restores that model's defaults. Values retain their model-native units.
- **Use in Network Studio** selects the inspected device as the network's synapse.
  Build the network to capture its current parameters. A capacitor is not offered
  as a conductance weight. Network learning rules remain distinct from Device
  Lab's waveform-driven STDP characterization.
- Stimulus edits refresh the waveform preview automatically after a short pause.
- **Assistant** opens an optional drawer. Custom pulse text and dataset editors
  appear only when relevant. Plot gestures and protocol assumptions live in
  contextual help. The Cell view offers illustration playback and speed controls;
  these do not alter simulation time. F5 runs the active experiment or training.

## Trainer integrity and independent comparison

The trainer computes spike activity and updates Python device states. It does
not run a Verilog-A circuit solver. Network STDP is an eligibility-trace rule
with writes consolidated after each presentation; it is separate from Device
Lab's waveform-overlap experiment. Neuron voltages are in normalized units.

- Rejected/no-spike outputs count as errors and have a separate confusion-matrix
  column. Recall and F1 include those samples.
- Checkpoints and new JSON saves preserve device state (including polarization,
  diffusion pools and device RNGs), neuron thresholds, time and training RNG.
  Legacy conductance-only saves cannot reconstruct lost dynamic history.
- Dataset builds reserve a stratified validation subset from training data,
  retaining at least two training samples per class. Epoch checkpoint selection
  uses validation accuracy (up to 40 samples). Tiny demo sets without a validation
  subset explicitly use training accuracy. Final test data never selects weights.
  The Validation and Test tiles have distinct meanings.
- Evaluation preserves the training RNG, thresholds and device clock. A requested
  test evaluation without a held-out test set raises an error rather than silently
  reporting training accuracy.

To reproduce the independent comparison:

```powershell
python -m pip install -r requirements-validation.txt
python -m unittest test_trainer_integrity
python tools/validate_snntorch.py
```

The comparison uses the actual app engine and independent snnTorch neurons /
PyTorch autograd on the same synthetic data, spike streams, initial states and
learning settings. Both sides deliberately use the same device-programming
model. It covers supervised STDP and surrogate learning, ECFET v2 (plus FeFET when installed), direct and
hidden-layer networks, and three seeds. It writes every run, parameter and
maximum error to `results/snntorch_validation.json`. The script exits nonzero
if any tolerance fails, even when all predictions agree. At identical states
the gradient must agree to 1e-10 and the weights after one step to 1e-9
(relative); spikes and predictions must be identical. The two separately
trained trajectories are allowed 1e-6 of drift, because rounding differences
of ~1e-15 between the two gradients are amplified by every device write;
each run also reports whether it stays within the former, tighter limits.

The snnTorch reference explicitly matches the app's immediate reset, membrane
readout, normalization, and (for STDP) refractory, inhibition and homeostasis.
It preserves float64 in the library's fast-sigmoid output. An unconfigured
stock snnTorch network or ordinary SGD is a different experiment. Agreement
here validates the implemented algorithms, not hardware fidelity, arbitrary
models/settings or performance on a real-world dataset. See the official
[snnTorch Leaky documentation](https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html)
and [surrogate gradient documentation](https://snntorch.readthedocs.io/en/stable/snntorch.surrogate.html).

## STDP: device plus programming waveform

The STDP tab now separates a device's switching response from an assumed
timing rule. A nonvolatile state alone does not establish an intrinsic timing
window. Overlapping pre/post waveforms can translate their timing difference
into a different voltage across a nonlinear device. Other devices may also
have genuine relaxation dynamics; these remain in the selected model.

Choose **overlapping spikes**, then choose an **exponential tail**, **linear
tail**, or **RC high-pass** shaper. Edit the signed pre/post amplitudes, width,
tail ratio and time, or R and C. The preview shows both signed contributions
and their sum. For two-terminal voltage programming enter `-Vpre` and `+Vpost`;
a FeFET model receives an equivalent differential gate programming voltage.
Voltage-driven families start with shaped spikes. Defaults are illustrative,
not device-specific experimental calibration.

RC mode models a rectangular source through an ideal buffered, unloaded
high-pass with `tau = R*C`. Its positive portion decays as `A*exp(-t/tau)`;
after the input ends at width `w`, its opposite tail is
`-A*(1-exp(-w/tau))*exp(-(t-w)/tau)`. This is **not a simulation of a passive
RC network loaded by the device**. Exponential tails are truncated at six time
constants and represented by 96 midpoint segments; linear tails end at the
specified tail time. Actual circuit loading and parasitics require circuit
simulation and measurement.

Every timing starts from the same initial model state and includes four runs:
the pair, pre only, post only, and idle, all with identical time grids.

* **Pair state change** is the actual modeled state change at the read delay.
* **Interaction** is `pair - pre - post + idle`. It isolates non-additivity;
  it must not be reported as the raw synaptic update.
* Single-spike disturbance and idle drift are displayed. Inspect these before
  describing individual spikes as non-disturbing. A symmetric or biologically
  shaped curve is not guaranteed; simultaneous opposite spikes can cancel.

The app's known added timing rules (`A_stdp`, `A_stdp_V`) are disabled for STDP
experiments unless **Include model's added timing rule** is checked. Enabling
that option uses the amplitudes in the model parameter panel; a zero amplitude
remains off. These are phenomenological rules, not proof of intrinsic device
timing memory. For a FeFET twin, `A_stdp_V` should default to zero outside the
STDP tab, and its STDP observable is the total threshold voltage, not a synthetic
timing-offset variable.

**Export last STDP experiment** saves `results/stdp_experiment.json` and `.csv`,
including model parameters, waveform/RC settings, timing-rule status, read delay,
timing grid, units and all four control responses. Clicking a curve point runs
the same pair waveform and read delay in Results. Shape settings persist when
switching between device families.

Examples of waveform-driven experimental STDP:
[ferroelectric tunnel junctions](https://www.nature.com/articles/s41467-020-15249-1)
and [pulse-shape dependence in ion-conducting memristors](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2016.00097/full).
The legacy `analysis.stdp_sweep` remains available for ECFET reference tests;
the desktop uses `vatester/stdp_protocol.py` for the controlled experiment.

## CLI quick start

```
python selftest.py                                  # 35 sanity checks (27 v2/v1 + 8 v3)
python run_ecfet.py spike                           # single 100 pA / 10 ms gate spike
python run_ecfet.py potentiate --amp-pA 50 --n 30   # conductance-up pulse train
python run_ecfet.py depress                         # conductance-down train
python run_ecfet.py ltp-ltd --n 30                  # synaptic LTP/LTD curve + metrics
python run_ecfet.py retention --hold-s 60           # write then watch relaxation
python run_ecfet.py compare                         # v1 vs v2, identical stimulus
```

Options: `--model v1|v2|both`, `--width-ms`, `--period-ms`, `--kappa-v`,
`--nu`, `--sigma-c2c 0.05` (write noise), `--extras` (internal-state panel),
`--csv` (dump waveforms for Spectre comparison). Plots land in `results/`.

Use as a library:

```python
from simulation_core import Waveform, EcfetV2, V2Params, simulate, plotting
wf = Waveform.pulse_train(amp=-100e-12, width=10e-3, period=50e-3, n=20)
r  = simulate(EcfetV2(V2Params(kappa_v=0.4)), wf, t_stop=2.0)
plotting.plot_transient(r, "out.png")
```

**Sign convention** (matches the original .va): positive gate current raises
R_mem / lowers conductance. "Potentiate" scenarios therefore use negative
current; flip with `V2Params(polarity=-1)` if your device is the other way.

## Scaling cheat-sheet (v1)

`dM/dt = Rdrift1c * I_gate = 20 Ω/s per pA`. So a 100 pA × 10 ms spike moves
M by 20 Ω; a 1 pA × 1 ms spike moves it by 20 µΩ — invisible next to the
fixed 20 Ω diffusion jump (issue 3 below). If your paper's pulses are
pA × ms and the conductance change is supposed to be a few % per pulse,
`Rdrift1c` must be raised, or use v2 where the per-pulse step is set
directly by `Q_ref` / `n_states`.

## Issues found in the original Verilog-A

1. **Integrator sign bug.** `I(cap) <+ Idrive` together with
   `I(cap) <+ ddt(V(cap))` gives, by KCL on the floating `capactive` node,
   `dV/dt = -Idrive`: positive gate current drives V(cap) *down*, while every
   guard in the module (`M_cap >= Rmax && Ieff > 0`, `source_dir`) assumes it
   goes up. Fix: `I(cap) <+ -Idrive;`. The Python v1 port implements the
   *intended* (+) sign.
2. **Switch-branch hazard.** The same `cap` branch gets `V(cap) <+` (pulse
   start, diffusion phase) and `I(cap) <+` (drift) contributions. In
   Verilog-AMS the voltage contribution silently wins for that iteration —
   discontinuous, and a classic source of Spectre convergence trouble.
3. **Fixed-amplitude "diffusion".** At every pulse end M instantly jumps by
   `c1+c2+c3 = 20 Ω` and then *recovers toward* `M_before_diffusion`. The
   relaxation magnitude is independent of how much the pulse actually wrote,
   and the retained state ends up ≈ the fully-drifted value — i.e. there is
   effectively no volatile loss, just a 20 Ω glitch.
4. **Hard 1 s diffusion window** truncates the τ₃ = 19 s tail with a slope
   discontinuity, and the state freezes mid-relaxation.
5. **τ₁/τ₂ vs window mismatch:** τ₁ ≈ 10 ms, τ₂ = 20 ms decay almost fully
   within the first 100 ms, so the 1 s window exists only for the tiny
   c3 = 0.1 component.

## What v2 does instead

* **Charge-controlled update** in the conductance domain:
  pulse of charge `Q_ref` (default 1 pC = 100 pA × 10 ms) moves G by
  `dG_unit·window`, where `dG_unit = (Gmax−Gmin)/n_states` and
  `window = exp(−ν·x)` is the standard soft-bound nonlinearity
  (x = normalized state). Gives realistic, saturating LTP/LTD curves;
  `nu_p`/`nu_d` set the asymmetry you fit from paper data.
* **Proportional volatile relaxation:** fraction `kappa_v` of every written
  ΔG goes into three pools relaxing with τ₁ = w²/2D, τ₂ = l²/2D, τ₃ —
  the post-pulse transient scales with the write, decays forever (no 1 s
  cutoff), and the retained fraction is `1 − kappa_v`.
* **Retention drift:** G_nv relaxes toward `1/R_eq` with `tau_retention`.
* **Cycle-to-cycle write noise** (`sigma_c2c`, seeded/reproducible).
* [ecfet_v2.va](verilog/ecfet_v2.va) implements the same equations with all state on
  internal nodes (1 V ≡ 1 S) — no switch branches, smooth `limexp` windows,
  clean convergence.

## Fitting to your paper data

1. `n_states` ← (number of distinguishable conductance levels) or
   full-range ΔG / per-pulse ΔG at mid-range.
2. `Q_ref` ← your experimental pulse charge (amp × width).
3. `nu_p`, `nu_d` ← fit to the curvature of measured LTP/LTD curves
   (`run_ecfet.py ltp-ltd` prints mean ΔG per branch).
4. `kappa_v` ← (peak − retained)/peak conductance right after a pulse.
5. `w, l, D, tao3, c1..c3` ← keep your physical diffusion constants; they
   set the relaxation time profile exactly as in v1.

## Validating against Cadence

Run any scenario with `--csv`, run the same piecewise-constant `isource`
stimulus on the .va in Spectre, export and overlay. For exact Verilog-A
execution outside Cadence, the `.va` also compiles with
[OpenVAF](https://openvaf.semimod.de/) → OSDI → ngspice.
