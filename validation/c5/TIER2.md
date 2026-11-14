# C5 tier 2: closed-loop Spectre testbench (design note)

Tier 1 (`README.md`) replays the engine's recorded write pulses open loop.
Tier 2 closes the loop inside Spectre: a neuron, an STDP controller and the
v3 synapses all run in one transient. The question is whether a training run
gives the same learning outcome as the NeuroVAT stream engine. This note
fixes the design. Nothing here has been built yet.

## What tier 1 already tells us

These results determine choices (c) and (d) below.

**Stored state matches exactly outside write pulses and at the end.** For
the same pulse schedule, the stored state G_nv (from x) matches between the
engine's device, the fine-grid twin and Spectre at those times. Inside a
pulse the engine already holds the post-pulse value (see 2 below). The engine and the twin agree to about 3e-9 of each synapse's G
range. The `README.md` table has the Spectre numbers.

**The full read (G_nv + volatile pools) does not match.** The engine reads
are off in two ways, both caused by how `EcfetV3Array` / `model_v3.step`
handle one pulse as one 10 ms step:

1. **Splitting.** The pools are decayed over the whole step before the new
   EDL is injected. The injected EDL therefore misses its decay inside the
   pulse (tau1 = 22 ms against a 10 ms pulse). Outside pulses this gives
   errors up to a few % of range. Splitting each pulse into sub-pulses shrinks
   it (`diagnose_engine.py`, `data/c5/diagnose.json`).
2. **Read timing.** `peek()` inside a pulse returns the post-pulse state, so
   the network sees each write up to 10 ms early. Inside pulses this gives
   errors of tens of % of range.

A closed-loop comparison on the full read would therefore show a systematic
bias on top of the spike chaos. That bias would stay until the engine
sub-steps pulses and reads causally.

## (a) Neuron: a Verilog-A copy of the engine's SRM, not an LIF

The stream engine (`masquelier.run`) uses the Masquelier 2008 spike-response
model with exact kernels. It is not an LIF:

```
u(t) = K * (Am - As) - inh + thr * (K1 e^{-s/tau_m} - K2 (e^{-s/tau_m} - e^{-s/tau_s}))   (s = t - t_post)
Am, As = sum_j w_j e^{-(t - t_j)/tau_m}, e^{-(t - t_j)/tau_s}   (reset at each post spike)
```

The parameters are tau_m = 10 ms, tau_s = 2.5 ms, K1 = 2, K2 = 4,
t_ref = 1 ms, and the kernels are cut at 7 tau.

Swapping in an LIF on both sides would test a different network from the one
that NeuroVAT trains. Port the SRM instead, as `verilog/srm_neuron.va`:

- **Kernel sums as two RC states.** Am and As are exactly two first-order
  states driven by weighted input impulses, so in Verilog-A they become two
  `idt` nodes with time constants tau_m and tau_s.
- **Input impulses.** Each input spike adds `w_j(t)` to both nodes. Feed it as
  a short current pulse of fixed charge times `w_j`, taken from that synapse's
  read current.
- **Threshold and reset.** Use `@(cross(u - thr, +1))`, which Spectre places
  exactly. At the crossing, reset the states (`Am = As = 0`, via a reset
  switch or a re-initialised `idt`) and start the after-potential kernel from
  `t_post`.
- **Cut-off.** The engine ends each kernel at 7 tau. A `.va` decays forever.
  The tail beyond 7 tau is e^-7 (about 0.1 %) of the kernel. Either accept it
  and state it, or zero it with a timed event at 7 tau.
- **Time step.** The engine steps the neuron on a 0.1 ms grid, and its spike
  times are quantised to that grid. Spectre's spikes fall at the exact
  crossings. This is a source of divergence by itself (see (d)).

The input spike trains are a frozen stream exported from
`make_pattern_stream` (seed, pattern and jitter as in tier 1). They become PWL
or `vsource type=pwl` spike pulses, the same on both sides.

## (b) STDP: a `.va` controller that emits the engine's calibrated pulses

STDP lives in `verilog/stdp_ctrl.va`, one instance per synapse (or one
vectorised module). It reproduces `masquelier.run` exactly:

- **Nearest-spike rule.**
  - LTP: at a post spike, each afferent with a pre spike since the last post
    gets `a_plus * e^{-dt/tau_plus}`, with dt up to 7 tau_plus.
  - LTD: at the first pre spike after a post, it gets
    `-a_minus * e^{-dt/tau_minus}`, with dt up to 7 tau_minus.
- **Write window.** Updates are summed per synapse and flushed every 10 ms
  as one pulse. Its amplitude is `pot_sign * gain * dw`, with `write_gain` for
  dw > 0 and `dep_gain` for dw < 0, from `calibrate_write`. Both gains go in
  as parameters taken from tier 1's `engine.json`.
  - The pulse is 10 ms wide, with 1 us edges.
  - It drives the synapse's IGsc through a current source controlled by the
    `.va`, for example `I(n_ig) <+ amp * pulse_window`.
  - The flush tick is a `@(timer(k * 10 ms))` event, so it lands on the same
    grid as the engine (flush at `t_ms >= next_flush`, aligned to the 0.1 ms
    neuron grid).
- **Device STDP lock-in: OFF** (`A_stdp = 0`) on both sides, as in Gate B,
  C6 and tier 1. The network rule decides, and the device's own
  edge-triggered lock-in (with its `prev_drift` evaluation-order caveat in
  the `.va`) stays out of the closed loop.

## (c) Read: G_nv on both sides as the primary test, full read as secondary

- **Primary: read G_nv (`read="nv"`, and `X.n_x` mapped through S(x) in
  Spectre).** Tier 1 shows that the stored state is identical across the
  engine, the twin and Spectre outside write pulses and at the end. This read
  removes the splitting error. It does NOT remove the read-timing error: the
  engine still applies each stored step at the pulse start, up to 10 ms early.
  In tier 1 the engine's G_nv over all reads was 1.4-4.1 % NRMSE and
  5.3-14.6 % max against Spectre, at about 56 % pulse duty. So even with this
  read, a closed-loop difference = spike chaos + the SRM port + that 10 ms
  look-ahead. That holds until the engine reads causally during a pulse, or
  until the `.va` side copies the look-ahead, which it should not.
- **Secondary: the full read, `1/V(Rmem)` in Spectre against
  `read="full"`.** Report it separately, and expect it to be biased until the
  engine's pulse sub-stepping and causal in-pulse read are fixed. Run it once
  more after that fix.

The weight is `clip((G - Gmin)/(Gmax - Gmin), 0, 1)` on both sides. In the
`.va` neuron it is computed from the sensed G of each synapse, refreshed
continuously. The engine refreshes it every 1 ms; that difference is small,
and the 1 ms refresh can be mimicked with a sample-and-hold `@(timer)`.

## (d) Comparison: statistics over seeds, not traces

Spike chaos is expected. A 0.1 ms shift of one post spike changes which
afferents get LTP, and the runs diverge after that. Traces are therefore not
compared after the first divergence. Instead, the two sides are compared over
**N seeds** (one frozen stream per seed, identical on both sides), on:

- **Time to first divergence.** The first post spike whose time differs by
  more than 0.1 ms (expected to be early). This is a diagnostic, not a pass
  criterion.
- **Final weight distribution.**
  - Pattern-afferent share with w > 0.9.
  - Background share with w < 0.1.
  - Bimodal share.
  - Mean pattern and background w.
  - Kolmogorov-Smirnov distance between the pooled final-w histograms.
- **Learning outcome.**
  - Hit rate, false alarms per s and latency over the last third
    (`masquelier._score`).
  - Success count (hit > 98 %, no false alarm, latency < 10 ms).
- **Rule.** For each metric, the two-sided 95 % CI of the per-seed paired
  difference (Spectre - engine) must contain 0 or lie within a stated
  equivalence margin. Suggested margins: 5 % of w range for the weight
  shares, 5 points of hit rate, 0.5 false alarms /s, 1 ms latency. Use a TOST
  equivalence test with those margins.
- **Number of seeds.** 10 seeds give a paired-difference SE of about
  sd/sqrt(10). C6 showed per-seed sds of a few % on the weight shares, so
  10-20 seeds resolve the margins above.

## (e) Size: what Spectre can afford

The cost is measured in tier 1. Its numbers are in `README.md` and
`validation/data/c5/cost.json`.

Spectre advances all instances on one shared time grid, so the cost grows
roughly with (number of write pulses, summed over synapses) x (instances per
step).

Learning needs about 200 presentations (C6: `--presentations 200`, with a
pattern every k-th 50 ms section). At k = 4 that is 40 s of stream per seed.

Measured in tier 1 (16 synapses, 4 s, 3608 pulses, pulse duty about 56 %):
**0.59 ms per pulse, 0.033 s per synapse-second**. The transient took 2.12 s
for 34153 steps.

That figure is for synapses only, with the write window aligning pulses on
a 10 ms grid. A closed loop adds the neuron's spike events and the input
spike PWLs, which are not aligned. Assume a step count 3-10x higher, i.e.
**about 0.1-0.3 s per synapse-second**.

| size | synapse-seconds | estimated Spectre time |
|---|---|---|
| smallest useful: 1 neuron, 32 afferents, 40 s (200 presentations at k = 4), 10 seeds | 12 800 | 0.4 - 1 h |
| 1 neuron, 100 afferents, 40 s, 10 seeds | 40 000 | 1 - 3.5 h |
| the paper's 2000 afferents, 450 s, 1 seed | 900 000 | 1 - 3 days |

Recommendation:

- **Build at 32 afferents with 10-20 seeds,** at 2 Spectre jobs at a time
  (host rule).
- **100 afferents is affordable** for a confirmation run.
- **2000 afferents is out of scope** for Spectre.

Output stays small with a strobe and by saving only the read node and n_x
per synapse plus the neuron's spikes. Tier 1 wrote 31 MB of PSF-ascii for
16 synapses over 4 s at a 0.2 ms strobe.

Build order:

1. The SRM `.va` alone against the engine with STDP off and fixed weights. The
   spike times must match to 0.1 ms on a frozen stream.
2. Add the STDP controller on ideal weights (no device) against the ideal arm.
3. Add the v3 synapses with the G_nv read.
4. Add the full read last.
