#!/usr/bin/env python
"""Drive the real GUI through the paper's Fig. 3a experiment and screenshot it.

Reproduces in the actual DearPyGui app: v2 model, Rinit = 4410 ohm, a single
+50 pA / 10 ms intercalation pulse at t = 5 s, observed for 40 s.
Expected (Sharbati et al. 2018, Fig. 3a): dip to ~4380 ohm (dR ~ -30),
relaxing back to ~4400 ohm (dR' ~ -10).

  python tools/verify_fig3a_gui.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import dearpygui.dearpygui as dpg
from vatester.app import App

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = App(here)
app.build()

# enable only the v2 model
for v in app.va_files:
    tag = f"cb_file_{v.name}"
    if dpg.does_item_exist(tag):
        dpg.set_value(tag, v.model_key == "v2")

# paper Fig. 3a bias point
app.param_values["v2"]["Rinit"] = 4410.0

# Signal Designer: single +50 pA / 10 ms pulse at t = 5 s, 40 s total window
dpg.set_value("gen_combo", "Single spike")
dpg.set_value("unit_combo", "pA")
app.gen_values["Single spike"] = {"t0_ms": 5000.0, "width_ms": 10.0,
                                  "amp": 50.0}
dpg.set_value("tail_input", 34.99)        # t_stop = 5.01 + 34.99 = 40 s
print("designer:", dpg.get_value("gen_combo"),
      "| kind:", dpg.get_value("kind_combo"),
      "| unit:", dpg.get_value("unit_combo"))

app.on_run()
frames = 0
while dpg.is_dearpygui_running() and frames < 8000:
    app._process_queue()
    dpg.render_dearpygui_frame()
    frames += 1
    if app.results and not app.sim_running and frames > 90:
        break

if not app.results:
    print("FAIL: no simulation results"); dpg.destroy_context(); sys.exit(1)

r = app.results[0]
R0 = float(np.interp(4.99, r.t, r.R))
Rdip = float(r.R.min())
Rend = float(r.R[-1])
i_pk = float(np.max(np.abs(r.i_gate)))
print(f"stimulus peak: {i_pk*1e12:+.1f} pA")
print(f"GUI v2 run [{r.label}]:")
print(f"  R before pulse : {R0:8.1f} ohm")
print(f"  dip            : {Rdip:8.1f} ohm  (dR  = {Rdip-R0:+.1f}, paper -30)")
print(f"  R at 40 s      : {Rend:8.1f} ohm  (dR' = {Rend-R0:+.1f}, paper -10)")

ok = abs((Rdip - R0) + 30.0) < 4.0 and abs((Rend - R0) + 10.0) < 3.0
print("RESULT:", "PASS - GUI shows the paper's Fig. 3a behavior"
      if ok else "FAIL - outside tolerance")

for _ in range(10):
    dpg.render_dearpygui_frame()
out = os.path.join(here, "results", "ui_fig3a.png")
dpg.output_frame_buffer(out)
for _ in range(10):
    dpg.render_dearpygui_frame()
dpg.destroy_context()
print("screenshot ->", out)
sys.exit(0 if ok else 1)
