#!/usr/bin/env python
"""Drive the real GUI's STDP tab and screenshot it.

Verifies the STDP curve is the dR-driven pairing gain: same-polarity pulses,
A_stdp = 0, a symmetric curve peaking at dt->0 and decaying to ~0 with |dt|
along the volatile relaxation (the delR write-then-relax).

  python tools/verify_stdp_gui.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dearpygui.dearpygui as dpg
from vatester.app import App

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = App(here)
app.build()

# enable only v2
for v in app.va_files:
    tag = f"cb_file_{v.name}"
    if dpg.does_item_exist(tag):
        dpg.set_value(tag, v.model_key == "v2")

print("A_stdp default:", app.param_values["v2"]["A_stdp"])
print("PRE/POST amp:", dpg.get_value("stdp_amp_pre"),
      dpg.get_value("stdp_amp_post"), dpg.get_value("unit_combo"))

app.on_plot_stdp()
frames = 0
while dpg.is_dearpygui_running() and frames < 12000:
    app._process_queue()
    app._tick_zoom_anim()
    dpg.render_dearpygui_frame()
    frames += 1
    if app._stdp_series and not app.sim_running and frames > 60:
        break

for ln in getattr(app, "_stdp_summary", []) or ["(no summary)"]:
    print("  ", ln)

dpg.set_value("center_tabs", "tab_stdp")
for _ in range(15):
    dpg.render_dearpygui_frame()
out = os.path.join(here, "results", "ui_stdp_delR.png")
dpg.output_frame_buffer(out)
for _ in range(10):
    dpg.render_dearpygui_frame()
dpg.destroy_context()
print("screenshot ->", out)
