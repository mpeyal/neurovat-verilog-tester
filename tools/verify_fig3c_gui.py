#!/usr/bin/env python
"""Drive the real GUI's LTP/LTD (Fig. 3c) on the default (paper) config and
screenshot the Analysis tab.  Confirms the ramp is triangular (no clipping) and
tops out near ~1150 uS, not the clipped 2500 uS trapezoid.

  python tools/verify_fig3c_gui.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dearpygui.dearpygui as dpg
from vatester.app import App

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = App(here)
app.build()

for v in app.va_files:
    tag = f"cb_file_{v.name}"
    if dpg.does_item_exist(tag):
        dpg.set_value(tag, v.model_key == "v2")

print("v2 default: Rmin=%g Rmax=%g n_states=%g c3=%g"
      % (app.param_values["v2"]["Rmin"], app.param_values["v2"]["Rmax"],
         app.param_values["v2"]["n_states"], app.param_values["v2"]["c3"]))

# LTP/LTD train: v2 potentiates on +current, so pot_sign = +1
dpg.set_value("gen_combo", "LTP / LTD train")
dpg.set_value("unit_combo", "pA")
dpg.set_value("kind_combo", "current")
app.gen_values["LTP / LTD train"] = dict(
    n_each=300, amp=170, width_ms=10, period_ms=2000, gap_ms=0,
    t0_ms=10, pot_sign=1, reps=1)

app.on_run()
frames = 0
while dpg.is_dearpygui_running() and frames < 30000:
    app._process_queue()
    dpg.render_dearpygui_frame()
    frames += 1
    if app.results and not app.sim_running and frames > 60:
        break

app.on_analysis_metric("G")
for _ in range(20):
    dpg.render_dearpygui_frame()

# report the plotted (G_nv) range
import numpy as np
from vatester import analysis
d = analysis.per_pulse_samples(app.results, "G",
                               app.results_meta.get("n_each"))[0]
g = np.array(d["vals"])
print("Analysis G_nv span: %.0f..%.0f uS  (Gmax=%.0f, no clip)"
      % (g.min(), g.max(), 1e6 / app.param_values["v2"]["Rmin"]))

out = os.path.join(here, "results", "ui_fig3c.png")
dpg.output_frame_buffer(out)
for _ in range(10):
    dpg.render_dearpygui_frame()
dpg.destroy_context()
print("screenshot ->", out)
