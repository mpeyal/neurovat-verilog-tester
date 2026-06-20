#!/usr/bin/env python
"""NeuroVAT - neuromorphic Verilog-A model tester GUI.

  python run_gui.py                 # workspace = this folder
  python run_gui.py --workspace D:\\my_models

Features: auto-detected .va files, ECFET v1/v2 + FeFET behavioral twins,
neuromorphic stimulus designer (spikes, trains, LTP/LTD, PPF, Poisson,
bursts, STDP, staircase, custom), live plots + LTP/LTD analysis, parameter
editing (importable from the .va), CSV/PNG export, an embedded Claude
agent that can generate patterns and read/modify the Verilog-A sources, and
a Neuromorphic Trainer studio that wires the selected device into a crossbar
of synapses + spiking (LIF) neurons and trains it with STDP, visualising the
weight (conductance) updates live.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vatester.app import main  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="folder to scan for .va files (default: app folder)")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="render N framessstarts then exit (self-test)")
    args = ap.parse_args()
    main(args.workspace, smoke_frames=args.smoke)
