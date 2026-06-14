#!/usr/bin/env python
"""Prove the analysis layer hot-reloads with NO app restart.

Builds the real GUI, runs an STDP sweep, then edits vatester/analysis.py on
disk and fires the SAME reload path the live watcher uses (App._reload_models).
A second sweep must reflect the edited code. The original file is always
restored.

  python tools/verify_hotreload.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dearpygui.dearpygui as dpg
from vatester.app import App
from vatester import analysis
from ecfet import EcfetV2, V2Params

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(HERE, "vatester", "analysis.py")

ORIG = "ys = [y - base for y in ys]"
EDIT = "ys = [(y - base) * 5.0 for y in ys]   # HOTRELOAD-TEST"

# small fixed sweep used to compare before/after
MODELS = [EcfetV2(V2Params())]
ARGS = dict(amp_pre=100e-12, amp_post=100e-12, width=5e-3,
            dts=[-0.02, -0.01, 0.01, 0.02], tail=0.5)


def peak(curves):
    ys = next(iter(curves.values()))
    return max(abs(y) for y in ys)


def main():
    app = App(HERE)
    app.build()                       # real GUI (so _reload_models can run)

    before = peak(analysis.stdp_sweep(MODELS, **ARGS))
    print(f"before edit:  peak |dG| = {before:.4f} uS")

    src = open(ANALYSIS, encoding="utf-8").read()
    assert ORIG in src, "anchor line not found - did analysis.py change?"
    ok = False
    try:
        with open(ANALYSIS, "w", encoding="utf-8") as f:
            f.write(src.replace(ORIG, EDIT))
        app._reload_models()          # <-- exactly what the live watcher calls
        after = peak(analysis.stdp_sweep(MODELS, **ARGS))
        print(f"after  edit:  peak |dG| = {after:.4f} uS  (expect ~5x)")
        ratio = after / before if before else 0.0
        ok = abs(ratio - 5.0) < 0.05
        print(f"ratio = {ratio:.3f}  ->",
              "PASS - edit took effect with NO restart" if ok
              else "FAIL - reload did not pick up the edit")
    finally:
        with open(ANALYSIS, "w", encoding="utf-8") as f:
            f.write(src)              # always restore the original file
        app._reload_models()
        restored = peak(analysis.stdp_sweep(MODELS, **ARGS))
        print(f"restored:     peak |dG| = {restored:.4f} uS")
        assert abs(restored - before) < 1e-9, "failed to restore analysis.py!"

    dpg.destroy_context()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
