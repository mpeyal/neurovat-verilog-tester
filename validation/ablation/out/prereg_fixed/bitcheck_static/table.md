Network: 1000 afferents (500 in the pattern), 1 neuron, Masquelier 2008 rule and neuron; 300 evenly spaced presentations per run; 2 seeds (streams shared by the arms); scored on the last third. Device arms: 10 ms pulses, 10 ms write window, calibrated on the retained step. mean ± sd over seeds; success = runs meeting hit > 98 %, no false alarm, latency < 10 ms.

| gap (ms) | arm | success | hit % | false alarms /s | latency ms | pattern w>0.9 % (read / stored) | background w<0.1 % (read / stored) | mean background w (read / stored) |
|---|---|---|---|---|---|---|---|---|
| 50 | ideal | 2/2 | 100.0 ± 0.0 | 0.00 ± 0.00 | 7.6 ± 1.3 | 16.2 ± 0.6 / 16.2 ± 0.6 | 73.2 ± 2.5 / 73.2 ± 2.5 | 0.078 ± 0.007 / 0.078 ± 0.007 |
| 50 | neurosim | 0/2 | 98.0 ± 2.8 | 0.00 ± 0.00 | 14.2 ± 11.0 | 15.4 ± 2.8 / 15.4 ± 2.8 | 79.3 ± 5.5 / 79.3 ± 5.5 | 0.067 ± 0.013 / 0.067 ± 0.013 |
| 50 | v3_static | 0/2 | 97.5 ± 3.5 | 0.00 ± 0.00 | 16.4 ± 13.7 | 17.4 ± 0.3 / 17.4 ± 0.3 | 83.5 ± 1.6 / 83.5 ± 1.6 | 0.051 ± 0.006 / 0.051 ± 0.006 |
| 50 | v3_static_gain | 2/2 | 100.0 ± 0.0 | 0.00 ± 0.00 | 7.0 ± 0.2 | 22.4 ± 0.6 / 22.4 ± 0.6 | 90.4 ± 2.8 / 90.4 ± 2.8 | 0.035 ± 0.005 / 0.035 ± 0.005 |
| 450 | ideal | 1/2 | 99.0 ± 1.4 | 0.00 ± 0.00 | 7.2 ± 1.7 | 23.5 ± 1.0 / 23.5 ± 1.0 | 92.4 ± 0.6 / 92.4 ± 0.6 | 0.032 ± 0.004 / 0.032 ± 0.004 |
| 450 | neurosim | 1/2 | 50.0 ± 70.7 | 0.00 ± 0.00 | 9.1 ± 0.0 | 13.1 ± 18.5 / 13.1 ± 18.5 | 61.6 ± 49.8 / 61.6 ± 49.8 | 0.115 ± 0.130 / 0.115 ± 0.130 |
| 450 | v3_static | 1/2 | 50.5 ± 70.0 | 0.00 ± 0.00 | 18.9 ± 13.4 | 12.2 ± 17.3 / 12.2 ± 17.3 | 56.2 ± 56.3 / 56.2 ± 56.3 | 0.116 ± 0.138 / 0.116 ± 0.138 |
| 450 | v3_static_gain | 0/2 | 0.0 ± 0.0 | 0.00 ± 0.00 | - | 0.0 ± 0.0 / 0.0 ± 0.0 | 23.4 ± 0.3 / 23.4 ± 0.3 | 0.215 ± 0.001 / 0.215 ± 0.001 |
