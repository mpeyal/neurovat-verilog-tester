# Gate A validation: Python twins vs Spectre

This folder checks that each Python device twin (`simulation_core/model_v2.py`,
`model_v3.py`) reproduces a circuit-simulator run of the matching Verilog-A
model (`verilog/ecfet_v2.va`, `verilog/ecfet_v3.va`). Both sides get the same
fixed set of gate-current stimuli.

Two references are used:

| reference | what it is | where its data lives |
|---|---|---|
| **twin** | `simulate()` from `simulation_core`, run by `run_twin.py` | `data/twin/<model>/` |
| **Spectre** | the `.va` model run in Cadence Spectre. You run the generated decks on your Cadence machine, export the results, and import them with `scrub_spectre.py`. | `data/spectre/<model>/` |

The twin also has a third check that needs no simulator: `compare.py --self-check`
runs the twin at `dt`, `dt/2` and `dt/4`. This shows that the twin's own
time-step error is far below the pass rule.

## Pass rule

The rule is applied per model, per stimulus and per reference, on the observed
conductance `G`:

* **NRMSE** `= rms(G_twin - G_ref) / range(G_ref)` must be **<= 1 %**.
* **max error** `= max|G_twin - G_ref| / range(G_ref)` must be **<= 3 %**.

Here `range(G_ref)` is the reference's max minus min over that stimulus. Both
metrics are computed on one shared grid: the twin's own time grid, which is
refined at every edge, clipped to the time span both runs cover. The reference
is linearly interpolated onto that grid.

Before scoring, `compare.py` checks the gate current Spectre actually applied
(`X1:IGsc`). It must have the same sign as the stimulus and the same total
charge to within 1 %. A reversed or mis-scaled source is marked FAIL instead of
being scored as a different experiment.

## Layout

```
validation/
  README.md            this file
  stimuli.py           fixed, versioned stimulus set (STIMULUS_SET_VERSION)
  common.py            model registry, canonical CSV schema, twin stepping loop
  run_twin.py          twin runs   -> data/twin/<model>/<stim>.csv (+ .json wall time)
  spice/make_netlists.py  writes the Spectre decks -> spice/spectre/<model>/<stim>.scs
  spice/spectre/       GENERATED decks + run_all.sh (do not hand-edit)
  scrub_spectre.py     Spectre PSF-ascii/CSV -> data/spectre/<model>/<stim>.csv
  compare.py           scoring, figures, summary.csv, --self-check
  data/summary.csv     twin vs Spectre table (written once Spectre data exists)
  data/self_check.csv  twin dt / dt/2 / dt/4 convergence table
  figures/             <model>_<stim>.png overlays, selfcheck/<model>_<stim>.png
```

## How to run

```
python validation/run_twin.py                 # all twin data (under 1 s)
python validation/compare.py --self-check     # time-step convergence
python validation/spice/make_netlists.py      # (re)generate the Spectre decks
# ... run the decks with Spectre and import them (next section) ...
python validation/compare.py                  # twin vs Spectre
python validation/compare.py --all            # both
python -m pytest -q tests/test_validation.py
```

`compare.py` exits with status 1 if any row FAILs.

## Stimulus set (`gateA-1`)

A positive current potentiates in both models (G up, R down). All pulses are
rectangular in the twin. The Spectre PWL ramps every edge over 1 us, starting
at the ideal edge, so each pulse carries exactly the same charge.

| name | content | t_stop |
|---|---|---|
| `single_pulse` | +50 pA / 10 ms (Fig. 3a), then relaxation | 2.0 s |
| `ltp_ltd_train` | 20 x +500 pA then 20 x -500 pA, 10 ms every 50 ms | 3.01 s |
| `pulse_width_sweep` | +200 pA, widths 2/5/10/20/40 ms, every 0.5 s | 3.01 s |
| `amplitude_sweep` | 10 ms, +25/50/100/200/400 pA, every 0.5 s | 3.01 s |
| `gap_sweep` | +200 pA / 10 ms pairs, gaps 5/20/50/200/1000 ms, 1.5 s apart | 8.885 s |
| `retention_hold` | 10 x +500 pA / 10 ms, then no drive until 20 s | 20 s |
| `stdp_pair_sweep` | opposite-polarity 50 pA / 10 ms pairs, dt = +-20/50/200 ms, 2 s apart | 12.51 s |
| `random_spike_train` | 40 random-sign 200 pA / 5 ms spikes, exponential gaps (mean 60 ms), seed 20261113 | 3.132 s |

**STDP.** Both modules have a single gate pin, `IGsc`. "Pre" and "post" are
therefore opposite-polarity pulses on that one pin. This matches
`tools/ecfet_v3_tb.scs` and the twins' lock-in rule: dt > 0 means a depressing
pulse followed by a potentiating one. There are no separate pre and post
terminals to drive.

The twin and the deck both use `dt_max` as their maximum step: the twin's
`time_grid` `dt_max`, and Spectre's `maxstep`.

Changing any stimulus changes `stimuli.fingerprint()`. Every CSV header
carries the version and the fingerprint, so data produced from an older set
can be recognised. Bump `STIMULUS_SET_VERSION` whenever you change the set.

## Running the Spectre decks

`make_netlists.py` writes one deck per model and stimulus (16 in total) to
`spice/spectre/v2/` and `spice/spectre/v3/`, plus a `run_all.sh` that runs them
all. Each deck:

* pulls in the model with `ahdl_include`. The path defaults to
  `../../../../verilog/ecfet_vN.va`, relative to the deck. If you copy the
  decks elsewhere, regenerate them with `--va-dir <dir holding the .va files>`.
* instantiates `X1 (n_in n_out n_igsc n_rmem) ecfet_vN` with the defaults.
  These equal the twin's dataclass defaults.
* applies a 10 mV read bias, `vread (n_in 0)`, across In-Out. Out is tied to
  node 0 so the read loop has a ground reference.
* drives the gate with `igate (0 n_igsc) isource type=pwl wave=[...]`. This is
  meant to make a positive value flow **into** the IGsc pin. It has not yet
  been verified on a real Spectre run (see "Gate source orientation" below).
* sets `options reltol=1e-6 vabstol=1e-12 iabstol=1e-18 gmin=1e-15`. The tight
  `vabstol` is required: the state nodes hold conductance as voltage
  (1 V == 1 S, about 1e-4 V), and Spectre's default of 1 uV is about one
  write step.
* runs `tran tran stop=<t_stop> maxstep=<dt_max> errpreset=conservative ic=node`.
* saves exactly these signals:

| saved signal | meaning | canonical column |
|---|---|---|
| `n_rmem` | V(Rmem) = R = 1/G_total | `R_ohm`, `G_S = 1/R` |
| `X1.n_nv` (v2) | nonvolatile G (1 V == 1 S), without the STDP lock-in | `G_ret_S = n_nv + n_stdp` |
| `X1.n_x` (v3) | conserved x in [0, 1] | `x_Li`, `G_ret_S = G_of_x(x) + n_stdp` |
| `X1.n_v1..3` | volatile EDL pools (S) | `G_vol_S = n_v1 + n_v2 + n_v3` |
| `X1.n_stdp` | STDP-locked part (S) | `stdp_S` |
| `vread:p` | read-branch current | `i_read_A = abs(i)` |
| `X1:IGsc` | gate current into the pin | `drive_A`, used by the sign and charge check |

### Exact commands (on the Cadence machine)

**Run `v3/single_pulse.scs` first.** At t = 20 ms, R should *fall*, from about
4400 ohm to about 4370 ohm.

* If R rises instead, the gate source is reversed. Regenerate the decks with
  `python validation/spice/make_netlists.py --flip-gate` before running the
  other 15.
* Importing the run and running `compare.py` shows the same problem as
  `drive_check = sign flipped`.

```
cd validation/spice/spectre/v3
spectre +escchars -format psfascii -raw single_pulse.raw +log single_pulse.log single_pulse.scs
# or every deck of both models:
sh validation/spice/spectre/run_all.sh
```

Each run leaves `<stim>.raw/tran.tran` (PSF-ascii) and `<stim>.log` next to
its deck.

### Importing the results

Copy the whole `spice/spectre/` tree, or just the `*.raw/tran.tran` and
`*.log` files with the same layout, back into this repo, then run:

```
python validation/scrub_spectre.py --dir validation/spice/spectre      # batch
# or one file:
python validation/scrub_spectre.py --model v3 --stimulus single_pulse \
    path/to/single_pulse.raw/tran.tran --log path/to/single_pulse.log
```

The importer also accepts a CSV export instead of PSF-ascii. The CSV must have
a time column, then one column per saved signal named as in the table above.
ViVA "`<signal> X`, `<signal> Y`" column pairs work if every X column holds the
same time points.

`scrub_spectre.py` **rebuilds** the output rather than copying it. The output
has:

* one neutral provenance line, `# source: Spectre <major.minor>; model: ...; stimulus_set: ...`
* the canonical header
* numeric rows only

Dates, design paths, host and user fields from the PSF header and the log are
dropped. Only the transient's elapsed time is read from the log, and it is
stored in a `.json` sidecar. As a final check, every output line is matched
against e-mail, user@host, IP, path, host-name, date and keyword patterns, and
against the local user and host names. If anything matches, the importer
refuses to write the file and exits with status 2.

`*.raw/`, `*.log` and other Spectre work files under `spice/spectre/` are
git-ignored, because they contain host and user names. Commit only the
scrubbed CSVs.

## Known twin vs `.va` differences to expect

These are differences in the sources, not bugs in this harness. They are the
first places to look if Spectre FAILs:

* **v2 STDP trace.** `model_v2.py` caps each eligibility trace at its
  single-pulse weight and zeroes (consumes) the opposite trace on lock-in.
  `ecfet_v2.va` does neither: it adds `scn_i` to its own trace on every onset
  and never consumes. Same-polarity runs therefore build a larger trace in the
  `.va`. Expect v2 differences on `stdp_pair_sweep`, `random_spike_train` and
  the LTP->LTD turn of `ltp_ltd_train`. `ecfet_v3.va` does saturate and
  consume, like its twin.
* **v2 lock-in storage.** The twin adds the STDP lock-in into `G_nv`, which is
  subject to retention. The `.va` keeps it in `stdp_lock`, which is not. With
  `tau_retention` = 1.31e6 s this is negligible over these stimuli.
* **v3 pulse-onset detection.** `ecfet_v3.va` detects the onset through
  module variables (`prev_drift`) that change on *every* evaluation, including
  Newton iterations and rejected time steps. Spectre keeps such variables, so
  a rejected step at an edge could fire or skip a lock-in. If STDP stimuli
  differ only at pulse onsets, check this first. `ecfet_v2.va` uses
  `@(cross)`, which Spectre handles natively.
* **Write integration.** The twin integrates the write with sub-steps inside
  each time step and splits the pool decay from the write (first order in dt;
  see `self_check.csv`). Spectre integrates the same ODEs with its own
  adaptive steps.
* **Gate source orientation.** The existing hand-written
  `tools/ecfet_v3_tb.scs` wires `igate (n_igsc 0)` with `val1=50p`. By the
  SPICE/Spectre isource convention (positive current flows from the first node
  through the source to the second), that *appears* to pull current out of the
  IGsc pin, which would be a depressing pulse. This is unverified. The
  generated decks use `(0 n_igsc)`, and `--flip-gate` switches to the old
  orientation. The `single_pulse` check above, the `X1:IGsc` save and
  `compare.py`'s drive check settle the question on the first real run.

## Paper overlay: `sharbati-1` (Sharbati et al., Adv. Mater. 30, 1802353, 2018)

This is a separate stimulus set built from the paper's own protocols, in
`paper/`. It does not change the `gateA-1` set, its data or its tests. It
gives three scores, reported separately:

1. twin vs Spectre, using the Gate A rule above;
2. Spectre vs paper;
3. twin vs paper.

| file | role |
|---|---|
| `paper/sharbati_stimuli.py` | the set (`STIMULUS_SET_VERSION = "sharbati-1"`), built with the Gate A `Stimulus` class |
| `paper/sharbati2018_targets.json` | text-scalar targets. Each row has its value, unit, page/figure, role (`calibration` / `held-out`), source type and extraction rule |
| `paper/curves/*.csv` | **empty** curve slots (provenance header only) for future digitized or original data. Never add invented points |
| `paper/run_twin_paper.py` | twin runs -> `data/sharbati/twin/<model>/`; `--self-check` -> `data/sharbati/self_check.csv` |
| `paper/make_decks_paper.py` | Spectre decks -> `spice/spectre_sharbati/<model>/` + `run_first.sh`, `run_all.sh` |
| `paper/scrub_paper.py` | Spectre import -> `data/sharbati/spectre/<model>/` (same scrubber and leak check as Gate A) |
| `paper/score.py` | tables -> `data/sharbati/score_table.csv`, `twin_vs_spectre.csv`, `curve_scores.csv`; figures -> `figures/sharbati/` |

Every pulse is 50 pA / 10 ms, as in the paper. The stimuli are:

* `sp_pot` and `sp_dep`: one pulse at 5 s (Fig. 3a/b), run to 205 s.
* `ltp_ltd`: 400 pre-depressing pulses, then 1700 LTP and 1700 LTD pulses, one every 50 ms (Fig. 3c).
* 14 `pp_dt*` stimuli: same-polarity pulse pairs (Fig. 4b), with onset-to-onset dt from 15 ms to 1800 ms. Each dt is its own transient.
* `retention_13h`: one pulse, a 300 s settle at 10 ms steps, then a 13 h hold at 60 s steps.

Values that the main text does not give are marked `assumed` in each
stimulus's params:

* the Fig. 4b amplitude and width;
* the LTP pulse interval and read timing;
* the retention state, because Fig. S5 is in the SI and not available.

**Every target is a calibration target: nothing is held out.** The model
defaults were tuned to these same numbers. A match is therefore a consistency
check of the twin and `.va` against the numbers they were fitted to, not a
prediction.

The saved CSVs are thinned. They keep every waveform edge, plus points no
closer together than a per-protocol spacing (`core.SAVE`).

The `retention_13h` self-check is exact by construction. The twin's no-drive
update is closed-form, so the coarse 60 s hold step cannot add error, and the
check passes trivially. It says nothing about Spectre at `maxstep=60`. That is
settled only by the twin-vs-Spectre score of the retention deck.

### How to run it

```
python validation/paper/run_twin_paper.py                 # twin data (~10 s)
python validation/paper/run_twin_paper.py --self-check    # dt, dt/2, dt/4
python validation/paper/make_decks_paper.py               # (re)generate the decks
python validation/paper/score.py                          # tables (Spectre column "pending" until imported)
python -m pytest -q tests/test_paper_overlay.py
```

On the Cadence machine:

1. **Run the v3 single-pulse deck first** with
   `sh validation/spice/spectre_sharbati/run_first.sh`. At t = 5 s, R must
   **fall**, from about 4400 ohm to about 4370 ohm.
2. If R rises instead, the gate source is reversed. Regenerate the decks with
   `python validation/paper/make_decks_paper.py --flip-gate`, then rerun
   step 1.
3. Then run the other decks:
   `sh validation/spice/spectre_sharbati/run_all.sh`. There are 36 decks. The
   `ltp_ltd` deck has 3800 pulses (190 s), and `retention_13h` has
   `maxstep=60`.
4. Copy the `*.raw/tran.tran` and `*.log` files back, keeping the same
   layout.
5. Import and score them:

```
python validation/paper/scrub_paper.py --dir validation/spice/spectre_sharbati
python validation/paper/score.py
```

## Background: OpenVAF

This harness uses Spectre only. For reference, two OpenVAF limits would block a
port of these models:

* OpenVAF does not support `@(cross)`, which is why `ecfet_v3.va` replaced it.
  `ecfet_v2.va` still uses it.
* OpenVAF has no hidden state: module variables are not kept between
  evaluations. The `stdp_lock` accumulators in both models and the
  `prev_drift` onset detector in v3 rely on that state, so they would not
  behave as they do in Spectre.

Check both points against the OpenVAF version you have before trying it.

## Stanford-PKU RRAM: `stanford-1` (twin vs Spectre, and vs MemTorch)

This set adds a second, public device: the Stanford-PKU metal-oxide RRAM
compact model v1.0.0, by Z. Jiang, S. Yu, X. Guan and H.-S. P. Wong (Z. Jiang
et al., IEEE TED 63(5), 2016; nanoHUB doi:10.4231/D37H1DN48). The
model was developed by the **Stanford Nanoelectronics Research Group of Prof.
H.-S. Philip Wong**.

Its licence is the NEEDS modified CMC licence, quoted in the `.va` and in
`twins/stanford_pku_rram.py`. It allows free modification and redistribution
on four conditions:

* keep the copyright notices;
* acknowledge the group;
* cite the papers;
* do not charge for the Stanford code.

This set is separate from `gateA-1` and `sharbati-1` and does not change them.

### Getting the `.va` (not tracked in git)

1. Download "Verilog-A" from https://nanohub.org/publications/19. No login was
   needed on 2026-11-14. The file is `rram_v_1_0_0.va`: 6762 bytes, sha256
   `ab0eebd3d8a04176f06a4f0a791444aa9f78786f79f1f3e1c54bb1f56c63d1ab`.
2. Save it, unmodified, as `verilog/stanford_pku_rram.va`.

`verilog/.gitignore` keeps the file out of git; `git check-ignore -v` reports
`verilog/.gitignore:3:*`. The scanner maps both this name and the original
`rram_v_1_0_0` name to the `stanford_pku` twin.

### Files

| file | role |
|---|---|
| `twins/stanford_pku_rram.py` | the twin: both Stanford copyright blocks, `TWIN_SPEC` key `stanford_pku`, voltage-driven, TE/BE |
| `stanford/stimuli.py` | `VoltageStimulus`: PWL voltage points in integer ps. Twin and deck use the same points |
| `stanford/core.py` | twin runs on the PWL grid, canonical columns, scoring, literal-Euler reference |
| `stanford/run_twin.py` | twin data -> `data/stanford/twin/`, figures -> `figures/stanford/`; `--self-check` |
| `stanford/make_decks.py` | Spectre decks -> `spice/spectre_stanford/` + `run_first.sh`, `run_all.sh`, `.gitignore` |
| `stanford/scrub.py` | Spectre import -> `data/stanford/spectre/` (same readers and leak check as Gate A) |
| `stanford/score.py` | twin vs Spectre -> `data/stanford/summary.csv`, `figures/stanford/spectre_*.png` |
| `memtorch_compare.py` | MemTorch Stanford_PKU vs the twin. Runs only where MemTorch is installed; skips cleanly otherwise |

### Model and twin semantics

**State and wind-up.**
* The model's state is the `idt` output y, and `gap = clip(y, gap_min, gap_max)`.
* The clamp acts on the output only. While the field stays above threshold,
  y keeps moving past the bound ("winds up"), and a later opposite pulse must
  first unwind it before the gap moves.
* The twin reproduces this. `clamp_state=1` gives MemTorch's clamped state
  instead.
* Examples from the twin runs:
  * `pulse_set_ref` (1.8 V / 100 ns, the HSPICE template) ends with y = -406 nm.
  * The 0.5 V/ms template sweeps wind y to +50 um and to -8 mm.
  * In `dc_loop` the second RESET sweep cannot undo the SET-side wind-up, so
    loop 2 stays in LRS (`figures/stanford/twin_iv_hysteresis.png`).
* This is why the official decks run one half-sweep per transient.

**Evaluation order.**
* gamma reads the previous evaluation's clamped `gap`.
* T uses the probed branch current `I(TE,BE)`.
* Their continuous-time limit is `gap = clip(y)` at the same instant. The twin
  integrates that limit with adaptive Dormand-Prince 5(4) inside every
  `step()`, with V linear between grid points.

**Field threshold.**
* RESET (V < 0) stops at g* = 1 nm * ((16 - F_min tox/|V|)/beta)^(1/3).
* That point is below gap_max only for |V| < 1.392 V. Above that, RESET winds
  up at gap_max.
* On a ramp the state slides along g*. The twin handles this discontinuity
  explicitly; Spectre has to resolve it with its own step control.

**Read conductance.**
* `.G` is the read conductance I(0.1 V)/0.1.
* A 0.1 V read leaves the gap exactly unchanged, because the rate is exactly 0
  below the field threshold. Reads therefore do not disturb the state.
* The I-V is sinh-shaped, so I = G*V holds only at 0.1 V. The instantaneous
  current is the observable `I_TE (A)`.

**Variability** (`model_switch=1`) is off by default.
* The `.va` draws a fresh `$rdist_normal` on every evaluation, so its noise
  depends on the step size.
* The twin draws one sample per `step()`, from a generator seeded with
  `rand_seed_ini`.

### Stimulus set (`stanford-1`, sha256 `158bdb6aa16bc5f0`)

Pulses are trapezoids with 50 ps edges and an area of amp x width. Reads are
0.1 V / 4 ns plateaus. Every stimulus starts in LRS (gap_ini = 0.2 nm) unless
its row says otherwise.

| name | content | t_stop |
|---|---|---|
| `pulse_reset_ref` | HSPICE `pulseSweep.sp` RESET, -1.3 V / 100 ns (10 ps edges), then a read | 205 ns |
| `pulse_set_ref` | HSPICE SET, +1.8 V / 100 ns from gap_ini 1.9 nm, then a read | 205 ns |
| `dc_reset_ref`, `dc_set_ref` | HSPICE `DCSweep.sp` half sweeps, 0 -> -2 / +2 V -> 0 over 4 ms + 4 ms (SET starts at 1.9 nm) | 8.1 ms |
| `dc_loop` | two bipolar loops in one transient at 1 V/us: 0 -> -1.35 -> 0 -> +1.5 -> 0 V | 11.5 us |
| `reset_train` | 20 x -1.3 V / 10 ns, 10 ns gaps, a read in every gap | 407 ns |
| `set_train` | 20 x +1.45 V / 10 ns from gap_ini = 1.7 nm (the HRS SET threshold is ~1.39 V) | 407 ns |
| `pulse_width_sweep` | -1.3 V pulses of 1/2/5/10/20 ns, then +1.35 V pulses of the same widths | 183 ns |
| `amplitude_sweep` | 10 ns pulses at -1.0..-1.4 V, then at +1.0..+1.4 V | 207 ns |
| `read_disturb_hold` | 5 x -1.3 V, a 1 us hold, then 10 x 5 ns stress at 0.1/0.5/0.9/+1.2/-1.2 V, with a read after each group | 1.65 us |
| `s1_pm1p3`, `s1_pm1p4` | MemTorch S1: 10 x -A then 10 x +A, 10 ns / 10 ns, with A = 1.3 / 1.4 V | 407 ns |

**Pass rule** (per stimulus; provisional, sized for ns switching and currents
from uA to 0.7 A):
* per-read G: max relative error <= 3 % and mean relative error <= 1 %;
* I(t): NRMSE <= 1 % of the reference range;
* gap: NRMSE <= 1 %, where the gap can be inverted from I and V (|V| >= 50 mV).

### How to run (twin side, no simulator)

```
python validation/stanford/run_twin.py                # 12 stimuli, ~2 s of twin time
python validation/stanford/run_twin.py --self-check   # dt, dt/2, dt/4 + literal-Euler check
python validation/stanford/make_decks.py              # (re)generate the Spectre decks
python validation/stanford/score.py                   # twin vs Spectre, once imported
python -m pytest -q tests/test_stanford_scanner.py tests/test_stanford_twin.py tests/test_stanford_validation.py
```

**Self-check.** Every stimulus PASSES. The largest dt-vs-dt/4 difference is
0.039 % in per-read G (`set_train`), and every I NRMSE is <= 0.009 %. These
numbers are small almost by construction, because the twin integrates
adaptively inside each step.

**Independent check.** A literal transcription of the `.va` (forward Euler,
one evaluation per time point) was run at 0.2 ps and Richardson-extrapolated
against 0.4 ps. Results (`data/stanford/brute_check.csv`):

* On a RESET pulse plus a runaway SET pulse (the constant-V path), it agrees
  with the twin to 2.5e-16 m, including the wind-up below gap_min.
* On a 1e7 V/s triangle loop (the ramp path, sliding on the RESET stop gap;
  extrapolated from 0.5 / 1 ps), the twin is within 6e-15 m. The worst point
  is the end point, at 1e-13 m: it lands exactly on gap_min (a knife edge),
  where the extrapolated reference is itself uncertain at that level.

### Running the Spectre decks (on the Cadence machine)

The decks use `ahdl_include "../../../verilog/stanford_pku_rram.va"`, so copy
the `.va` into `verilog/` first. If you move the decks, regenerate them with
`--va-dir`.

1. **Run `sh validation/spice/spectre_stanford/run_first.sh` first.** It runs
   `pulse_reset_ref`, the shortest deck. Check that:
   * the `.va` compiles;
   * `X1:TE` is **negative**: about -35 mA at t = 1 ns, with the magnitude
     falling as the gap opens;
   * the read at ~170 ns gives about +6.6 uA (G ~ 66 uS).
2. If Spectre's AHDL parser rejects the `.va`, the likely causes are its comma
   ranges (`from[0, 1]`, `from(390, 460)`) and the two `` `include ``s on one
   line. Run `python validation/stanford/make_decks.py --patched`, then repeat
   step 1. The flag writes `verilog/stanford_pku_rram_spectre.va`
   (git-ignored), points the decks at it, and changes only:
   * those two syntax points, nothing else;
   * a note added at the top;
   * both copyright notices kept.
3. Run `sh validation/spice/spectre_stanford/run_all.sh` (12 decks).
   * A failing deck is reported and the remaining decks still run.
   * The two 8.1 ms template DC decks run last. With a 1 us step bound, one
     step at RESET onset can move the gap about 1 nm, so Spectre's step
     control, not the model, decides how the RESET slide is resolved there.
   * **`dc_loop` (1 ns bound) is the hysteresis comparison of record.** If
     `dc_reset_ref` or `dc_set_ref` fails or FAILs scoring, first re-run it
     with a smaller `time_step=` / `maxstep=` (edit the generated deck
     locally).
4. Copy the `*.raw/tran.tran` and `*.log` files back into
   `validation/spice/spectre_stanford/`, then run the commands below. The
   batch import reports a refused or truncated run and still imports the
   others (exit status 2 if any run was refused).

```
python validation/stanford/scrub.py --dir validation/spice/spectre_stanford
python validation/stanford/score.py
```

**Deck settings.**
* Options: `reltol=1e-5 vabstol=1e-9 iabstol=1e-15 gmin=1e-15`.
* Transient: `tran maxstep=dt_max errpreset=conservative`.
* The instance gets `q=1.602176462e-19 kb=1.3806503e-23` explicitly. The `.va`
  would otherwise take the simulator's own `` `P_Q `` / `` `P_K ``.
* `time_step` (the `.va`'s `$bound_step`, default 1 ns) is set to dt_max:
  0.1 ns for the pulse decks and 1 us for the two 8.1 ms DC decks, which
  would otherwise take 8 million steps.
* Saved signals: `n_te`, `X1:TE`, `vte:p`.
* The importer rejects a run whose V*I integral is negative (a flipped
  current probe).

### Spectre concerns (unverified: the authors list only HSPICE H-2013.03-SP2)

* **Syntax:** the comma parameter ranges and the two `` `include ``s on one
  line are non-standard. See `--patched` above.
* **Probe before contribution:** `Itb = I(TE,BE)` probes the branch current
  before that branch's own contribution. Spectre may warn, and the value
  returned is the current iteration's contribution.
* **Hidden state:** the module variable `gap` is read before it is assigned,
  so it carries state across evaluations, including Newton iterations and
  rejected steps. Spectre keeps that state; OpenVAF would not.
* **Discontinuous rate:** the hard `if` on the field threshold makes the rate
  jump at g*. A slow RESET ramp slides along g*, which can cost many time
  steps or end in a time-step-too-small error.
* **Random draws:** `$rdist_normal` runs on every evaluation, even with
  `model_switch=0`, where its output is multiplied by 0.
* **Wind-up:** a Spectre run should show the `idt` wind-up too. If Spectre's
  I(t) differs from the twin's only after a saturating pulse, compare
  `pulse_set_ref` and `dc_loop` first.

### MemTorch comparison (`memtorch_compare.py`)

This script runs only in a venv where MemTorch is installed. MemTorch is
GPL-3: the script imports it but never copies its code.

Install recipe (Windows, Python 3.13). The last install step compiles a C++
extension, so MSVC Build Tools are required.

```
py -3.13 -m venv venv
venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\python -m pip install numpy scipy pandas matplotlib scikit-learn lmfit seaborn ipython ninja
venv\Scripts\python -m pip download memtorch-cpu==1.1.6 --no-deps --no-binary :all: --no-build-isolation -d dl
venv\Scripts\python -m pip install --no-build-isolation --no-deps dl\memtorch-cpu-1.1.6.tar.gz
venv\Scripts\python validation/memtorch_compare.py
```

**Setup.**
* Both tools get the same parameters. `r_on`/`r_off` are chosen so that
  MemTorch's derived gap bounds come out at exactly 0.2 and 1.7 nm, and the
  bounds are then also set directly.
* Both get the same V samples: the midpoint of each dt, held over the step.
* The twin runs in MemTorch's reading (`clamp_state=1`, q = 1.6e-19), so only
  the numerics differ.
* The reference is the adaptive twin on the exact PWL.

**Results.** Per-read G error (max %) against the reference, measured
2026-11 (`data/stanford/memtorch/memtorch_compare.csv`). In the "twin
adaptive" column the value in brackets is the error against the exact ODE on
the same held samples, i.e. the integration error alone.

| stimulus | dt | MemTorch | twin Euler | twin adaptive | MemTorch wall | twin adaptive wall |
|---|---|---|---|---|---|---|
| S1 +-1.3 V | 1 ns | 21.7 | 29.1 | 0.59 (0) | 0.004 s | 0.006 s |
| S1 +-1.3 V | 0.1 ns | 3.1 | 4.3 | 0.59 (0) | 0.03 s | 0.06 s |
| S1 +-1.3 V | 10 ps | 0.35 | 0.46 | 0.025 (0) | 0.27 s | 0.51 s |
| S1 +-1.4 V | 1 ns | 68.7 | 74.5 | 0.54 (0) | 0.003 s | 0.006 s |
| S1 +-1.4 V | 0.1 ns | 20.7 | 24.3 | 0.54 (0) | 0.03 s | 0.05 s |
| S1 +-1.4 V | 10 ps | 2.8 | 3.3 | 0.028 (0) | 0.32 s | 0.55 s |

How to read the table:
* The twin adaptive's 0.5-0.6 % at coarse dt is not integration error. It
  comes from the held samples missing the 50 ps edges.
* The maximum error is always on the last, runaway SET pulse.
* Current NRMSE for MemTorch at +-1.3 V is 0.99 / 0.12 / 0.011 % (1 ns /
  0.1 ns / 10 ps).
* MemTorch at 1 ps converges to the reference (0.036 % at +-1.3 V, 0.28 % at
  +-1.4 V).
* MemTorch computes T from the previous step's current. This lag makes it
  slightly *more* accurate than a plain same-step Euler.
* Running the twin with the `.va` semantics (wind-up, q = `` `P_Q ``) instead
  of the matched settings changes G by up to 1.2 %, almost entirely through
  the charge constant.

**Parameter mismatches** between MemTorch and the `.va` (`param_map.csv`):
* MemTorch derives gap_min/gap_max from r_on/r_off. Its defaults give
  0.20008 / 1.70000 nm.
* MemTorch hard-codes q = 1.6e-19; the `.va` uses `` `P_Q ``.
* MemTorch's default dt is 1e-4 s, which cannot represent ns pulses. The
  `.va`'s `$bound_step` is 1 ns.
* MemTorch uses `gamma_init` for both polarities. The `.va` hard-codes 16 for
  V < 0, which is the same value at the default.
* MemTorch clamps the state every step (no wind-up) and computes T from the
  previous step's current.
