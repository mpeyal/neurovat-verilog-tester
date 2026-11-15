# T4 (pre-registered re-run)

**Question (C6):** does the synapse's *time* dynamics change what STDP learns, beyond its per-pulse curve?

**Pre-registration:** `validation/ablation/PREREG.md` (commit `91dee5f`), with a dated addendum (`a888ddf`) committed before any outcome was looked at. The addendum covers the engine finding: the old physical-time read was wrong inside write pulses. Because of it, the physical-time arms were re-run on the fixed engine (master `8b6b7fc`), and those runs are the primary result. The old-engine physical-time numbers are a sensitivity row only.

**Setup** (same as the first run):
- 1000 afferents (500 in the pattern), 1 neuron, Masquelier 2008 rule and neuron.
- 300 evenly spaced presentations per run. Gap = (k−1)×50 ms.
- Device arms: 10 ms pulses, 10 ms write window, calibrated on the retained step.
- 30 new seeds (101–130), disjoint from the first run's 1–10. All arms share the stream of a seed.
- Scored on the last third of each run. **Learnt** = hit rate > 0.9. **Success** = hit > 98 %, no false alarm, latency < 10 ms.
- Static arms are not re-run: ideal, neurosim, v3_static and the gain control do not use the changed path. They were checked bit-identical on the fixed engine, 14/14 runs compared.

**Seeds that learnt the pattern (hit rate > 0.9), out of 30** (v3_time: fixed engine)

| gap | ideal | neurosim | v3_static | v3_time | v3_time, old engine (pre-fix) |
|---|---:|---:|---:|---:|---:|
| 50 ms | 30 | 30 | 28 | 28 | 25 |
| 150 ms | 28 | 27 | 26 | 24 | 20 |
| 450 ms | 22 | 13 | 16 | 21 | 19 |

Strict success, out of 30:

| gap | ideal | neurosim | v3_static | v3_time | v3_time, old engine (pre-fix) |
|---|---:|---:|---:|---:|---:|
| 50 ms | 20 | 17 | 12 | 15 | 13 |
| 150 ms | 15 | 13 | 10 | 6 | 6 |
| 450 ms | 9 | 8 | 4 | 8 | 7 |

Hit % at 450 ms (mean ± sd): ideal 73 ± 45, neurosim 44 ± 50, v3_static 57 ± 48, v3_time 82 ± 34 (old engine 75 ± 40).
Latency at 450 ms: 13.4 / 17.0 / 18.6 / 10.0 ms (old engine 11.8 ms).
False alarms are ≤ 0.02 /s in every arm.

**Attribution at 450 ms, fixed engine.** The device state is the same as in v3_time; only the read differs.

| read | learnt /30 | success /30 | hit % | old engine learnt /30 |
|---|---:|---:|---:|---:|
| G_nv only, physical time | 14 | 6 | 50 ± 50 | 13 |
| G_nv + pool τ1 = 22.2 ms | 16 | 5 | 58 ± 48 | 13 |
| G_nv + pool τ2 = 312.5 ms | 13 | 3 | 50 ± 47 | 16 |
| G_nv + pool τ3 = 19 s | 23 | 11 | 87 ± 27 | 20 |
| full read (v3_time) | 21 | 8 | 82 ± 34 | 19 |
| gain control: v3_static, every pulse ×1.36 | 14 | 6 | 48 ± 50 | (static: same) |

**Paired tests.** Exact two-sided sign test (McNemar), by seed, at 450 ms. "a:b" is the number of seeds where only the first arm learnt : only the second.

| test | role | learnt | discordant | p |
|---|---|---|---|---:|
| v3_time vs v3_static | **primary** | 21 vs 16 | 9:4 | **0.27** |
| v3_time vs v3_static, strict success | secondary | 8 vs 4 | 6:2 | 0.29 |
| v3_time vs gain control | secondary | 21 vs 14 | 11:4 | 0.12 |
| v3_time vs ideal | secondary | 21 vs 22 | 4:5 | 1.00 |
| τ3-only vs G_nv-only | secondary | 23 vs 14 | 10:1 | 0.012 |

The same tests on the old engine (sensitivity): primary 19 vs 16, 7:4, p = 0.55; vs gain 8:3, p = 0.23; vs ideal 2:5, p = 0.45; τ3 vs G_nv 12:5, p = 0.14.

**First run's exact config re-run on merged master** (seeds 1–10, old engine, masquelier fixes only):

| gap | ideal | neurosim | v3_static | v3_time |
|---|---|---|---|---|
| 50 ms | 9 (was 10) | 10 (10) | 10 (9) | 10 (9) |
| 150 ms | 10 (10) | 9 (9) | 9 (8) | 10 (9) |
| 450 ms | 4 (3) | 5 (3) | 4 (1) | 5 (7) |

- No run of the 120 is identical. 15 runs flip learnt, 24 flip success, and 40 change their hit rate.
- At 450 ms the first run's 7 vs 1 (v3_time vs v3_static, p = 0.03) becomes 5 vs 4 (3:2, p = 1).

**Honest reading:**
- **C6 is not confirmed.** The pre-registered primary test is not significant on the fixed engine (p = 0.27) or on the old one (p = 0.55).
- The direction still favours v3_time over v3_static at 450 ms, but the effect is small. Ideal STDP learns about as often as v3_time (22 vs 21). The first run's 7/1 contrast came from a 450 ms setup where ideal itself learnt only 3/10.
- That contrast was fragile. The two masquelier fixes alone, on the same seeds, make it disappear.
- One secondary test is nominally significant: reading the 19 s pool on top of G_nv learns more than G_nv alone (23 vs 14, 10:1, p = 0.012). This is a pre-registered secondary test with no multiplicity correction, so it cannot rescue the primary. It is a lead for a follow-up, not a result.
- The gain-matched static control (14/30) does not reproduce v3_time (11:4, p = 0.12). Not significant.
- v3_time is the only device arm with a mean latency at 10 ms at 450 ms; v3_static is at 18.6 ms.
- Wall time, 10 processes:
  - prereg main 7482 s, attribution 5005 s, gain 795 s (old engine);
  - fixed-engine v3_time 2285 s, attribution 5598 s;
  - first-run rerun 2255 s.
