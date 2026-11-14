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

RESULTS_PLACEHOLDER

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

This is reported, not fixed. Changing `EcfetV3Array` / `model_v3.step`
(sub-stepping a pulse, or a causal in-pulse read) would move the Gate B and
C6 numbers and needs its own decision. Gate A did not catch it, because
`simulate()` steps the twin on a fine grid inside each pulse.

## Spectre cost

COST_PLACEHOLDER
