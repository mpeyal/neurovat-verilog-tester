"""Render the GUI with representative content and capture screenshots."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dearpygui.dearpygui as dpg
from vatester.app import App

here = os.path.dirname(os.path.abspath(__file__))
app = App(here)
app.build()

app.append_chat("you", "Generate a potentiating Poisson spike train: "
                "50 Hz, 1 s, -100 pA, 2 ms spikes.")
app.append_chat("agent", "Done - a 50 Hz Poisson train over 1 s with -100 pA "
                "/ 2 ms spikes. Negative current potentiates (G up) per the "
                "ECFET sign convention. Load it below and hit Run.")
app._add_pattern_card({"label": "Poisson 50 Hz potentiating",
                       "pulses": [(0.01 + i * 0.02, 0.002, -100e-12)
                                  for i in range(52)],
                       "kind": "current", "unit": "pA"})
app.on_preview()
app.on_run()

frames = 0
while dpg.is_dearpygui_running() and frames < 1500:
    app._process_queue()
    dpg.render_dearpygui_frame()
    frames += 1
    if app.results and frames > 90:
        break

# drop A/B probe markers on the R plot for the capture
if app.results:
    r = app.results[0]
    i, j = len(r.t) // 3, 2 * len(r.t) // 3
    app._set_probe("plot_r", "A",
                   *app._nearest_point("plot_r", float(r.t[i]),
                                       float(r.R[i])))
    app._set_probe("plot_r", "B",
                   *app._nearest_point("plot_r", float(r.t[j]),
                                       float(r.R[j])))
    for _ in range(8):
        dpg.render_dearpygui_frame()

out = os.path.join(here, "results")
os.makedirs(out, exist_ok=True)
dpg.output_frame_buffer(os.path.join(out, "ui_results.png"))
for _ in range(10):
    dpg.render_dearpygui_frame()

dpg.set_value("center_tabs", "tab_designer")
for _ in range(10):
    dpg.render_dearpygui_frame()
dpg.output_frame_buffer(os.path.join(out, "ui_designer.png"))
for _ in range(10):
    dpg.render_dearpygui_frame()

# STDP sweep capture
app.on_plot_stdp()
for _ in range(4000):
    app._process_queue()
    app._tick_zoom_anim()
    dpg.render_dearpygui_frame()
    if app._stdp_series and not app.sim_running:
        break
dpg.set_value("center_tabs", "tab_stdp")
for _ in range(15):
    dpg.render_dearpygui_frame()
dpg.output_frame_buffer(os.path.join(out, "ui_stdp.png"))
for _ in range(10):
    dpg.render_dearpygui_frame()

dpg.destroy_context()
print("screenshots saved")
