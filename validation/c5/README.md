# C5 tier 1: does training with a v3 device synapse in NeuroVAT match Spectre?

Open-loop replay. NeuroVAT runs a stream-mode STDP training with ECFET v3
device synapses in physical time. Every gate write pulse the engine applied
is recorded. The same pulses then drive one `ecfet_v3` instance per synapse
in Spectre. Per-synapse G(t) is compared under the Gate A rule: NRMSE <= 1 %
and max <= 3 % of that synapse's Spectre G range.

## Pipeline

```
python validation/c5/run_engine.py --afferents 16 --seconds 4 --seed 1   # engine run, schedule, reads
python validation/c5/make_deck.py                                        # -> spice/spectre_c5/v3/train.scs
# on the Spectre host, from a copy of the repo layout (deck includes ../../../../verilog/ecfet_v3.va):
#   spectre +escchars -format psfascii -raw train.raw +log train.log train.scs
python validation/c5/compare_c5.py scrub <train.raw/tran.tran.tran> <train.log>
python validation/c5/compare_c5.py score       # summary.csv, cost.json, figure
python validation/c5/diagnose_engine.py        # why the engine's full read differs (no Spectre needed)
```

## Setup (`data/c5/engine.json`)

- **Engine:** `neurovat.masquelier.run`, unchanged. One neuron, 16 afferents
  (8 in the pattern), a pattern every 4th 50 ms section, seed 1, 4 s of
  stream.
- **Device:** from `vector_twin_factory` (the vectorised `EcfetV3Array`,
  self-checked against the scalar twin), with `V3Params.paper(A_stdp=0)`.
  Physical time, full read, re-read every 1 ms.
- **Writes:**
  - 10 ms pulses, with a 10 ms write window.
  - Calibrated on the retained step: 1.50 nA potentiates by a_plus at the
    start weight, and 1.30 nA depresses by a_minus.
  - Start weight 0.475 (G = 609 uS, R = 1642 ohm).
- **Recording:** `program` and `peek` are wrapped on the instance, so neither
  the class nor the engine changes. This gives the schedule (`schedule.csv`:
  3608 pulses, 0.2 to 1563 pA) and every read the engine made
  (`engine_reads.csv`: full G and G_nv per synapse).
- **Activity:** 281 output spikes. Each synapse got 195 to 264 pulses, so it
  is inside a write pulse about 55 % of the time.
- **Spectre deck:**
  - Gate A options, maxstep 1 ms, 1 us PWL edges, strobe 0.2 ms.
  - Each instance has `Rinit = 1/G_start` and `A_stdp = 0`.
  - Back-to-back pulses share one edge, because times are held on a 0.1 us
    integer grid.

## Results

See `data/c5/summary.csv` and `figures/c5/tier1_synapses.png`. Three
comparisons are made per synapse:

| comparison | what it tests |
|---|---|
| **engine** (primary) | the conductance the network actually read during training, against Spectre at the same times |
| **twin** | the scalar twin replaying the same pulses on the Gate A grid, against Spectre (model equivalence under this schedule) |
| **G_nv** | the stored part only, engine against Spectre |

**Spectre run.** 2.12 s transient, 34153 accepted steps, 20301 strobe
points. Every synapse's gate charge matches the schedule to 5e-14, with the
right sign.

Ranges over the 16 synapses, in % of each synapse's Spectre G range
(215 to 599 uS):

| comparison | NRMSE % | max % | final G error % | verdict |
|---|---|---|---|---|
| twin replay, full G | 0.021 - 0.063 | 0.065 - 0.178 | <= 0.13 | **16/16 PASS** |
| engine read, full G, all reads | 5.1 - 14.5 | 18 - 52 | <= 3.93 | **16/16 FAIL** |
| engine read, full G, outside pulses | 0.64 - 1.50 | 1.96 - 4.89 | - | NRMSE > 1 % on 8/16, max > 3 % on 10/16 |
| engine read, G_nv, outside pulses | - | <= 7.3e-4 | <= 6.6e-4 | exact |
| engine read, G_nv, all reads | 1.4 - 4.1 | 5.3 - 14.6 | - | FAIL (read timing) |

Reading the table:

- **The v3 model matches Spectre under the training's own pulse schedule.**
  This covers 3608 pulses of 0.2 to 1563 pA, including synapse 2 reaching the
  G_total output clamp at Gmax. Synapse 2 does not reach the x rail: its final
  G_nv is 994 uS, so the rail taper is not exercised here.
- **The stored conductance is exact.** What the engine stores, G_nv and so
  the stored weight, matches Spectre to below 1e-5 of range outside pulses
  and at the end.
- **The full read does not match.** What the network *reads* during training,
  the full G, does not meet the Gate A rule. The cause is the engine, not the
  model; see below. A claim of the form "training with a device synapse in
  NeuroVAT matches Spectre" holds for the stored state (G_nv) outside write
  pulses and at the end of the run. It does not hold inside pulses: the engine
  applies each stored step at the pulse start, up to 10 ms early, and the
  devices are in a pulse about 56 % of the time. It does not yet hold for the
  full read the neuron uses.

## Why the engine's full read is off (`diagnose_engine.py`, `data/c5/diagnose.json`)

`EcfetV3Array._step` is a line-by-line copy of `model_v3.step`. In physical
time a pulse is ONE step of dt = 10 ms.

1. **Splitting.** Part (C) decays the volatile pools over the whole dt. Part
   (E) then injects the pulse's EDL, which therefore never decays inside the
   pulse (tau1 = 22 ms against the 10 ms pulse). The pool, and so the full
   read, is too high right after every write.
2. **Read timing.** `peek()` reads a device inside a pulse "as it is", which
   is already the post-pulse state. The network therefore sees each write's
   whole step up to 10 ms early. At the calibrated 1.5 nA one pulse moves the
   full read by tens of % of a synapse's range, so the in-pulse reads carry
   the largest errors.

Neither effect touches G_nv: x is conserved and linear in charge, so the
stored state is exact. The engine's G_nv equals the fine-grid twin to about
3e-9 of range.

Measured against the fine-grid twin, worst synapse of 16:

| engine pulse handling | out-of-pulse NRMSE % (max / mean) | out-of-pulse max % |
|---|---|---|
| 1 step per pulse (as in-tree) | 1.45 / 0.99 | 4.71 |
| pulse split into 10 sub-pulses | 0.091 / 0.062 | 0.30 |
| pulse split into 100 sub-pulses | 0.035 / 0.024 | 0.11 |

The in-pulse reads are off by up to 51.7 % of range against the twin at the
same time. Against the twin at that pulse's end they are off by at most 5.2 %,
which is the splitting error at its worst point. So the in-pulse gap is the
`peek` look-ahead, and the rest is splitting.

This was reported here and is now fixed (next section). Gate A did not
catch it, because `simulate()` steps the twin on a fine grid inside each
pulse.

## After the engine fix (`validation/enginefix/`)

Both causes are fixed: the v3 driven step integrates the pools exactly
(exponential integrator per x sub-step, so the EDL decays within the pulse),
and in physical time `program` schedules a pulse that `advance` / `peek`
integrate up to the read time (a read inside a pulse is the state at that
time). A closed-loop re-run changes the reads, hence the spikes and the
schedule, so the committed Spectre run is compared with an OPEN-LOOP replay
of this schedule through the fixed engine's device code (`program` at each
write time, `peek` every 1 ms): `python validation/enginefix/replay_c5.py`
writes `validation/data/c5_fixed/` (`acceptance.json`, `summary.csv`). The
same replay run on the master code reproduces the table above
(`validation/enginefix/c5_replay_master.json`: in-pulse max 51.6 %,
out-of-pulse max 4.89 %, final G 3.93 %).

| engine read vs Spectre, full G (% of range) | NRMSE | max | verdict |
|---|---|---|---|
| all reads | 0.0004 - 0.0010 | 0.0010 - 0.0029 | **16/16 PASS** |
| inside pulses | 0.0004 - 0.0013 | 0.0010 - 0.0029 | |
| outside pulses | 0.0001 - 0.0003 | 0.0007 - 0.0019 | |
| G_nv, all reads | 0.0001 - 0.0003 | 0.0003 - 0.0007 | PASS |
| final G error | - | <= 0.0017 | |
| twin replay (fine grid), full G | 0.0023 - 0.0066 | 0.0076 - 0.026 | 16/16 PASS |

A closed-loop check (new schedule, new Spectre run) is in
`validation/enginefix/` (tier-1 long run).

## Spectre cost

The figures are in `data/c5/cost.json`. Wall time is the transient only,
parsed from the log; the AHDL compile, about 2 s, comes on top. The host is
shared, so timings are indicative.

- 16 instances over 4.06 s: 2.12 s, 34153 accepted steps.
- That is **0.59 ms per v3 pulse** and **0.033 s per synapse-second**.
- Steps per pulse are only about 9.5, because the engine's write window
  aligns most pulses on a common 10 ms grid, so the instances share
  breakpoints.
- For comparison, a single instance with free-running pulses (Sharbati
  ltp_ltd) took 1.6 ms per pulse and 113 steps per pulse.
- The engine itself took 10.4 s for the same run, including the Python
  logging wrappers.
