# NeuroVAT — Neuromorphic Verilog-A Test Bench (Python)

Fast iteration on neuromorphic synaptic device models (ECFET / ECRAM /
FeFET) without Cadence. Three behavioral models are implemented:

| | |
|---|---|
| **v1** ([ecfet/model_v1.py](ecfet/model_v1.py)) | Faithful Python port of the original Verilog-A — use it to reproduce/debug Spectre behavior |
| **v2** ([ecfet/model_v2.py](ecfet/model_v2.py)) | Upgraded practical ECFET model — tune it here, then take [ecfet_v2.va](ecfet_v2.va) (same equations) back to Virtuoso |
| **FeFET** ([ecfet/model_fefet.py](ecfet/model_fefet.py)) | Voltage-driven ferroelectric synapse (Merz-law switching) — twin of [FeFET.va](FeFET.va) |

## GUI

```
pip install -r requirements.txt
python run_gui.py
```

DearPyGui desktop app with:

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

## CLI quick start

```
python selftest.py                                  # 13 sanity checks
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
from ecfet import Waveform, EcfetV2, V2Params, simulate, plotting
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
* [ecfet_v2.va](ecfet_v2.va) implements the same equations with all state on
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
