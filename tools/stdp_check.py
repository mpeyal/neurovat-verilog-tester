#!/usr/bin/env python
"""Replicate the GUI STDP sweep to verify the anti-symmetric window.

Classic STDP: one pre/post pair (OPPOSITE polarity) per dt, dG = retained G
change after settling.  The A_stdp lock-in makes it order-dependent, so
pre-before-post and post-before-pre give OPPOSITE-sign dG (the +/- window).
With pre=-100/post=+100, dt>0 (post after pre) potentiates (+dG).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecfet import EcfetV2, V2Params
from vatester import analysis

AMP_PRE = -100e-12     # OPPOSITE polarity for the two spikes
AMP_POST = 100e-12
WIDTH = 5e-3
TAIL = 1.0
DTS = [d * 1e-3 for d in (-100, -50, -30, -20, -15, -10, -7.5, -5.05,
                          5.05, 7.5, 10, 15, 20, 30, 50, 100)]


def main():
    curves = analysis.stdp_sweep([EcfetV2(V2Params())],
                                 AMP_PRE, AMP_POST, WIDTH, DTS, TAIL)
    print(f"== EcfetV2 (A_stdp={V2Params().A_stdp}) ==")
    print(" dt_ms     dG_uS")
    for label, ys in curves.items():
        for dt, y in zip(DTS, ys):
            print(f"{dt * 1e3:+7.1f}  {y:+9.4f}")


if __name__ == "__main__":
    main()
