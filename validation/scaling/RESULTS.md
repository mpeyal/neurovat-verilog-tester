# T6. Scaling: twin vs Spectre (same machine)

**Machine:** one shared Linux host, AMD EPYC 9554, 128 logical CPUs.
- Spectre 24.1 classic with `+mt=1` (transient CPU / elapsed ≈ 0.997).
- Python 3.9.25, numpy 2.0.2, with OMP/OPENBLAS/MKL threads set to 1.
- Each point is the median of 3 runs, one job at a time; spreads are about 2 % or less.
- 1-minute load average 2.9–5.6 during the runs.
- Spectre time is transient analysis only, excluding about 0.3–2.5 s of AHDL compile and parse per run.

**Accuracy at every point:** each twin was checked against that run's Spectre output on 16 probe devices (or all of them, if fewer) with the Gate A rule, and passed. Speed-ups are therefore at equal accuracy.

**Sources:** `validation/scaling/` and `validation/data/scaling/{v3_array,network_time,network_replay,rram_array}.csv`; `machine.json` and the figures are in `validation/figures/scaling/`.

## T6a. ECFET v3 array: N independent devices, C5 write schedule (4.06 s, about 225 pulses per device)

| N | Spectre (s) | vectorised twin (s) | speed-up | scalar engine adapter (s) | scalar Gate A twin (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.214 | 0.386 | 0.55× | 0.038 | 0.125 |
| 4 | 0.499 | 0.570 | 0.88× | 0.133 | 0.508 |
| 16 | 1.47 | 0.640 | 2.3× | 0.498 | 1.95 |
| 64 | 5.17 | 0.665 | 7.8× | 1.96 | 7.86 |
| 256 | 20.8 | 0.779 | 27× | 8.00 | 30.8 |
| 1024 | 86.7 | 1.13 | 77× | 32.4 | 125 |
| 4096 | 584 | 2.35 | **248×** | – | – |

Max twin error against Spectre is 0.0029 % of range (NRMSE ≤ 0.001 %) for every twin at every N.

## T6b. Network level: Masquelier stream network, 1 neuron, k = 10, 4 s of stream

Wall time per stream-second:

| afferents | NeuroVAT, ideal weights | NeuroVAT, v3 physical time (full network: neurons, STDP, devices) | Spectre open-loop replay of the same write pulses (devices only, no neuron or STDP) | ratio, Spectre replay / NeuroVAT v3 |
|---:|---:|---:|---:|---:|
| 64 | 0.248 | 0.435 | 1.37 | 3.1× |
| 256 | 0.392 | 0.618 | 5.73 | 9.3× |
| 1000 | 0.514 | 0.830 | 23.1 | **28×** |
| 2000 | 0.544 | 0.987 | not run | – |

- **Spectre replay cost:** a near-constant 0.021–0.023 s per synapse-second.
- **NeuroVAT cost:** grows only weakly with the afferent count (2000 afferents: 0.99 s per stream-second).
- **Engine accuracy:** the engine's reads match that Spectre replay on all 16 probes (max 0.0032 % of range).
- **What the Spectre column leaves out:** it contains no neuron or learning. A closed-loop Spectre testbench is not built; `validation/c5/TIER2.md` *extrapolates* 0.1–0.3 s per synapse-second, i.e. 100–300 s per stream-second at 1000 afferents. The figure labels this band as extrapolated.

## T6c. Two-terminal memristors: scalar twin vs Spectre, N independent devices

| model / stimulus | Spectre step that passes | N = 1 | N = 16 | N = 64 | N = 256 |
|---|---|---|---|---|---|
| Stanford-PKU `pulse_set_ref`, Spectre at 1 ps (the T2b reference step) | 1 ps | Spectre 1.69 s / twin 0.019 s (**90×**) | 4.76 / 0.299 (16×) | 14.0 / 1.19 (12×) | – |
| Stanford-PKU, Spectre at 10 ps (passes vs 1 ps: I NRMSE 0.86 %) | 10 ps | 0.170 / 0.019 (9×) | 0.479 / 0.299 (1.6×) | 1.44 / 1.19 (1.2×) | – |
| VTEAM `on_train`, Spectre at 100 ps (passes: G max 0.49 %) | 100 ps | 0.052 / 0.018 (2.8×) | 0.124 / 0.286 (**0.43×**) | 0.348 / 1.15 (0.30×) | 1.39 / 4.66 (0.30×) |

The Stanford and VTEAM twins are scalar, so their cost is linear in N. Spectre amortises its per-step overhead over many instances, so for N above a few it matches or beats the scalar twins. This happens wherever Spectre does not need a fine step: Stanford at 10 ps, and VTEAM.

## Honest reading (for the paper)

1. **The speed-up comes from vectorisation, not from a cheaper device model.**
   - The scalar v3 twin on the Gate A grid is as fast as Spectre or slower (0.6–1.5×).
   - The vectorised v3 twin is 77× faster than Spectre at 1024 devices and 248× at 4096.
   - Below 16 devices it is not faster.
2. **Spectre's step count stays flat, about 33k, for every N on this schedule.** But its time grows faster than N (6.7× from 1024 to 4096). At 4096 devices one 4 s run takes 10 min, and a 10 s network run would take about 25 min.
3. **At network level:**
   - NeuroVAT simulates the whole learning network with physical-time v3 synapses, at 1000 afferents, 28× faster than Spectre simulates the devices alone, with no neuron and no learning.
   - The closed-loop speed-up is larger but extrapolated, not measured.
4. **Two-terminal RRAM:** a speed advantage exists only where Spectre needs a fine step for accuracy (Stanford switching at 1 ps: 12–90×). VTEAM has smooth hard-threshold dynamics, a coarse step is enough, and Spectre is faster for N ≥ 16. A vectorised RRAM twin would be needed to claim speed there; it is not built.

**Caveats:**
- The host is shared and was loaded, and both sides ran single-threaded; multi-threaded or APS Spectre would be faster.
- The C5 schedule puts all pulses on a shared 10 ms grid, which keeps Spectre's step count low, so it favours Spectre.
- The vectorised twin reads every 1 ms, while Spectre and the scalar twin produce full waveforms.
- Python 3.9 on the host is older than the laptop's 3.13.
