# C6 ablation: pre-registered confirmation run

Written and committed before any run of this confirmation. Nothing below is
changed after the runs start; any deviation is reported as a deviation.

## Why

The first run (`out/first_run/`, `out/attribution_gap450/`,
`out/gain_control_gap450/`, seeds 1-10) was made on branch `gateb-agent`,
before two later fixes to `src/neurovat/masquelier.py::run` (same-step
double pre spikes depress once, by the first; refractory / eta-window tests
with a 1e-6*dt tolerance). Its "learnt" threshold and its attribution arms
were chosen after seeing the data. This run tests the first run's main
finding on new seeds with everything fixed in advance.

## Code

- Merged master at commit `9842b33` (includes Gate B and both masquelier
  fixes). `run_ablation.py`, `neurosim_curve.py` and `masquelier.run` are used
  unchanged.
- Analysis: `validation/ablation/prereg_analysis.py`, committed with this
  file. Checked on the first-run data only (it reproduces learnt 10/10/9/9,
  10/9/8/9, 3/3/1/7 at 50/150/450 ms, attribution 1/3/1/7, gain control 2).

## Design

- Network: 1000 afferents (500 in the pattern), 1 neuron, Masquelier 2008
  rule and neuron, threshold 250, 300 evenly spaced presentations, read every
  1 ms; device arms 10 ms pulses, 10 ms write window, calibrated on the
  retained step, v3 STDP lock-in off. These are the `run_ablation.py`
  settings of the first run.
- Gaps: 50, 150, 450 ms (`--k 2 4 10`). Attribution and gain control at
  450 ms only (`--k 10`).
- Seeds: 101-130 (30 seeds, `--seed0 101 --seeds 30`), disjoint from the
  first run's 1-10. All arms see the same stream for a seed.
- Arms, as named in `run_ablation.py`:
  - main: `ideal`, `neurosim` (the existing fitted A-parameter curve),
    `v3_static`, `v3_time`;
  - attribution (same device state as `v3_time`, only the read differs):
    `v3_time_nv` (G_nv only, physical time), `v3_time_p1` / `_p2` / `_p3`
    (G_nv + the 22.2 ms / 312.5 ms / 19 s pool). The "full read" row is the
    main run's `v3_time` at 450 ms (not a separate run);
  - gain-matched static control: `v3_static_gain` (v3_static with every
    pulse x `pool3_gain()` = 1.36).

## Commands (run from the repo root, one at a time, at most 10 processes)

    python validation/ablation/run_ablation.py --seed0 101 --seeds 30 --k 10 2 4 \
        --arms ideal neurosim v3_static v3_time --presentations 300 \
        --afferents 1000 --read-ms 1.0 --jobs 10 --out validation/ablation/out/prereg
    python validation/ablation/run_ablation.py --seed0 101 --seeds 30 --k 10 \
        --arms v3_time_nv v3_time_p1 v3_time_p2 v3_time_p3 --presentations 300 \
        --afferents 1000 --read-ms 1.0 --jobs 10 \
        --out validation/ablation/out/prereg/attribution_gap450
    python validation/ablation/run_ablation.py --seed0 101 --seeds 30 --k 10 \
        --arms v3_static_gain --presentations 300 --afferents 1000 --read-ms 1.0 \
        --jobs 10 --out validation/ablation/out/prereg/gain_control_gap450
    python validation/ablation/prereg_analysis.py \
        --main validation/ablation/out/prereg/runs.jsonl \
        --extra validation/ablation/out/prereg/attribution_gap450/runs.jsonl \
                validation/ablation/out/prereg/gain_control_gap450/runs.jsonl \
        --out validation/ablation/out/prereg/analysis.json

(`--k 10 2 4` only puts the 450 ms jobs first; the table is sorted.)

Exploratory, outside the confirmatory test: the first run's exact config
(seeds 1-10, `--k 2 4 10`, the four main arms) re-run on this code into
`out/first_run_rerun/`, to show what the masquelier fixes changed.

## Outcomes

- Primary: **learnt** = last-third hit rate > 0.9 (`hit_rate` in
  runs.jsonl, as in the first run). A non-finite or missing hit rate is not
  learnt.
- Secondary: the paper's strict success (hit > 98 %, no false alarm,
  latency < 10 ms, the `success` field), hit %, false alarms /s, latency.

## Tests

Exact two-sided sign test (McNemar) paired by seed. With b = seeds where
only arm A learnt and c = seeds where only arm B learnt,
p = min(1, 2 P(X <= min(b, c))), X ~ Bin(b + c, 1/2); p = 1 if b + c = 0.

- **Primary:** at 450 ms, `v3_time` vs `v3_static` on learnt; alpha 0.05.
- Secondary (learnt, 450 ms): `v3_time` vs `v3_static_gain`; `v3_time` vs
  `ideal`; `v3_time_p3` vs `v3_time_nv`. Also `v3_time` vs `v3_static` on
  strict success. No multiplicity correction; raw p reported.
- Descriptive: learnt / 30 and success / 30 per arm per gap; hit %, false
  alarms, latency mean +- sd (the `run_ablation.py` table).

## Decision rule

- C6 is **confirmed** only if the primary test gives p < 0.05 **and** the
  direction is v3_time > v3_static (b > c).
- If the primary test is not significant (or goes the other way), C6 is
  reported as **not confirmed**. Secondary tests cannot rescue a failed
  primary; they are reported as supporting or not.
- A crashed run is re-run with the same seed and arm; no seed is replaced or
  dropped. Results are reported whatever they are.
