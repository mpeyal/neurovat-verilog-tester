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


def _app_dir():
    """Folder holding verilog/ sources, twins/ and patterns/: the directory
    that CONTAINS the exe when frozen (PyInstaller), else this script's folder.
    A frozen build's __file__ points inside the bundle, not next to the exe."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",
                    default=_app_dir(),
                    help="folder to scan for .va files (default: app folder)")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="render N frames then exit (self-test)")
    # control bridge is ON by default (toggle live from the Tools menu); use
    # --no-bridge to start with it off. See tools/nvat_ctl.py.
    ap.add_argument("--no-bridge", dest="bridge", action="store_false",
                    help="start with the control bridge disabled "
                         "(it is enabled by default; toggle it in Tools menu)")
    ap.set_defaults(bridge=True)
    args = ap.parse_args()
    from vatester.app import main
    main(args.workspace, smoke_frames=args.smoke, bridge=args.bridge)
