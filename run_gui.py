#!/usr/bin/env python
"""NeuroVAT - neuromorphic Verilog-A model tester GUI.

  python run_gui.py                 # workspace = this folder
  python run_gui.py --workspace D:\\my_models
  python run_gui.py --web           # browser UI on http://127.0.0.1:8000
                                    # (NeuroVAT Studio, same physics engine)

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help="folder to scan for .va files (default: app folder)")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="render N fi i nramessstarts then exit (self-test)")
    # control bridge is ON by default (toggle live from the Tools menu); use
    # --no-bridge to start with it off. See tools/nvat_ctl.py.
    ap.add_argument("--no-bridge", dest="bridge", action="store_false",
                    help="start with the control bridge disabled "
                         "(it is enabled by default; toggle it in Tools menu)")
    ap.add_argument("--web", action="store_true",
                    help="serve the NeuroVAT Studio web GUI on localhost "
                         "instead of the Dear PyGui window (same engine)")
    ap.add_argument("--port", type=int, default=8000,
                    help="port for --web (default 8000)")
    ap.add_argument("--no-open", dest="open_browser", action="store_false",
                    help="with --web: don't auto-open the browser")
    ap.set_defaults(bridge=True, open_browser=True)
    args = ap.parse_args()
    if args.web:
        studio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio")
        sys.path.insert(0, studio_dir)
        os.chdir(studio_dir)   # server/bridge resolve their files relative to studio/
        import server          # noqa: E402  (studio/server.py)
        server.serve(port=args.port, open_browser=args.open_browser)
    else:
        from vatester.app import main
        main(args.workspace, smoke_frames=args.smoke, bridge=args.bridge)
