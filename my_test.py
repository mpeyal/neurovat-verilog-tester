"""Minimal example: define a gate-current input, simulate, plot input+output."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecfet import Waveform, EcfetV1, EcfetV2, simulate, plotting

# INPUT: any gate-current pattern you want, as (start_s, width_s, amplitude_A)
wf = Waveform([
    (0.05, 0.010, +100e-12),   # +100 pA for 10 ms  -> R goes UP
    (0.30, 0.010, +100e-12),   # another one
    (0.60, 0.020, -150e-12),   # -150 pA for 20 ms  -> R goes DOWN
    (1.00, 0.005,   +2e-12),   # tiny +2 pA spike, 5 ms
])

# SIMULATE both models with that input
r1 = simulate(EcfetV1(), wf, t_stop=2.0)   # your original Verilog-A behavior
r2 = simulate(EcfetV2(), wf, t_stop=2.0)   # upgraded practical ECFET

# PLOT: panel 1 = input current, panels 2-3 = output R and G
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "my_test.png")
plotting.plot_transient([r1, r2], out, title="My custom input/output test")
print(r1.summary())
print(r2.summary())
print("plot saved ->", out)
