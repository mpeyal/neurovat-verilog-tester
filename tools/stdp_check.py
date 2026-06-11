#!/usr/bin/env python
"""Replicate the GUI _stdp_worker dt-sweep to verify v2 STDP behaviour."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecfet import Waveform, EcfetV2, V2Params, EcfetV1, V1Params, simulate

AMP_PRE = -100e-12   # potentiating (G up)
AMP_POST = 100e-12   # depressing (G down)
WIDTH = 5e-3
TAIL = 1.0           # settle 1000 ms
T0 = 10e-3           # t0c in _stdp_worker

DTS_MS = [-100, -50, -30, -20, -15, -10, -7.5, -5.05,
          5.05, 7.5, 10, 15, 20, 30, 50, 100]


def stdp_point(model, dt):
    # exact replica of app.py _stdp_worker single-pair measurement
    pre_t = T0 + max(0.0, -dt)
    post_t = pre_t + dt
    wf = Waveform([(pre_t, WIDTH, AMP_PRE), (post_t, WIDTH, AMP_POST)])
    t_stop = max(pre_t, post_t) + WIDTH + TAIL
    r = simulate(model, wf, t_stop, label=model.name)
    return (r.G[-1] - r.G[0]) * 1e6


def main():
    for cls, pcls in ((EcfetV2, V2Params),):
        print(f"== {cls.name} ==")
        print(" dt_ms    dG_uS")
        for dt_ms in DTS_MS:
            m = cls(pcls())
            dg = stdp_point(m, dt_ms * 1e-3)
            print(f"{dt_ms:+7.1f}  {dg:+9.4f}")


if __name__ == "__main__":
    main()
