# v3 at the x rails (bound step-collapse check)

## The problem

`ecfet_v3.va` used to stop the write at the rails with a hard switch:

```
if ((xc >= 1.0 && u_x > 0.0) || (xc <= 0.0 && u_x < 0.0)) u_x = 0.0;
```

Retention (`-(x - x_eq)/tau_ret`) keeps pulling x down while the switch
holds it up. So during a potentiating pulse at the upper rail, dx/dt jumps
from +|I|/Q_full (0.062 /s at 50 pA) to about -1e-6 /s across x = 1. The
state sat on the switch (sliding mode): x moved by about 5e-11 around 1, and
Spectre's step collapsed to about 1e-10 s.

- The Sharbati `ltp_ltd` deck (3800 pulses, 190.6 s) got stuck at
  t = 100.71 s. It was stopped after 9.5 h, having written 207 GB.
- The short reproduction here, `hi_short` (Rinit 872 ohm, 30 x +50 pA), got
  stuck at t = 0.801 s, the first pulse that reaches x = 1. It wrote 391 MB
  in 3 min and was killed by the file-size cap.

The lower rail did not stall, because retention pushes toward x_eq = 0, the
same way as the drive.

## The fix

Only the outward write is tapered, and only over the last `x_soft` (default
1e-5) of the window:

```
z   = clip(distance to the rail / x_soft, 0, 1)     (from the raw V(n_x))
u_x = u_x * z * (2 - z)
```

- At the band entry, w = z(2 - z) is 1 with w' = 0, so the write is C1 there.
- At the rail, w is about 2z, so the right-hand side is continuous and the rail
  is a smooth, stable fixed point. At 50 pA it sits about 1e-5 * x_soft
  (1e-10) inside the window.
- Inward drive is never tapered, so a write away from the rail starts at full
  rate.
- Off the band the equations are unchanged. The Gate A CSVs re-ran
  byte-identical.

The Python twin (`model_v3.py`) and `EcfetV3Array` keep their hard clamp,
which is the x_soft -> 0 limit. The G difference this taper can make is at
most span * dS/dx(rail) * x_soft, about 0.3 * x_soft = 3e-6 of the window.
`x_soft` is a `.va`-only parameter with no twin counterpart.

## Decks and result

`bound_check.py` writes the decks (`validation/spice/spectre_bound/v3/`),
imports the runs (`validation/data/bound/`) and scores them with the Gate A
rule (NRMSE <= 1 %, max <= 3 % of the Spectre G range). Each stimulus sets
its own Rinit.

| stimulus | content | NRMSE % | max % | G range uS | Spectre tran s |
|---|---|---|---|---|---|
| hi_short | Rinit 872, 30 x +50 pA into the upper rail | 0.032 | 0.388 | 2.64 | 0.047 |
| hi_50pA | Rinit 880, 120 x +50 pA, then 40 x -50 pA | 0.009 | 0.052 | 13.06 | 0.238 |
| hi_500pA | Rinit 900, 30 x +500 pA, then 20 x -500 pA | 0.022 | 0.070 | 72.90 | 0.106 |
| lo_50pA | Rinit 8000, 120 x -50 pA into the lower rail, then 40 x +50 pA | 0.007 | 0.077 | 13.85 | 0.211 |
| lo_500pA | Rinit 8000, 30 x -500 pA, then 20 x +500 pA | 0.016 | 0.061 | 83.56 | 0.089 |

Two more decks were run once, outside the committed set, at the C5 training's
write amplitude: 2 nA, 20 pulses into each rail, then 15 off it. Both pass:
the upper rail scores NRMSE 0.023 % and max 0.10 % (0.08 s), and the lower
rail 0.022 % and 0.10 % (0.07 s).

The Sharbati `ltp_ltd` deck, unchanged, now finishes. Its transient takes
5.96 s (8.6 s including the AHDL compile), with 430016 accepted steps. Against
the twin it scores NRMSE 0.19 % and max 0.35 % over the whole 1029 uS window
(PASS), and both end at G = 120.005 uS. Before the fix it did not finish
(stopped after 9.5 h).

```
python validation/bound/bound_check.py decks
python validation/bound/bound_check.py scrub <dir with <name>.raw/tran.tran.tran and <name>.log>
python validation/bound/bound_check.py score
```
