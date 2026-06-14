"""Headless-ish integration test: build the GUI, trigger preview + run,
pump the render loop until results are plotted."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dearpygui.dearpygui as dpg
from vatester.app import App

app = App(os.path.dirname(os.path.abspath(__file__)))
app.build()

# slash-command / model badge checks
app._handle_command("/model sonnet")
assert app.agent_model_id == "claude-sonnet-4-6", app.agent_model_id
assert "Sonnet 4.6" in dpg.get_item_configuration("model_badge")["label"]
app._handle_command("/model opus 4.8")
assert app.agent_model_id == "claude-opus-4-8", app.agent_model_id
app._handle_command("/model default")
assert app.agent_model_id is None
app._handle_command("/model")          # renders the in-chat picker
app._handle_command("/help")
app._handle_command("/bogus")          # unknown -> error bubble, no crash

app.on_preview()
app.on_run()

frames = 0
while dpg.is_dearpygui_running() and frames < 1200:
    app._process_queue()
    dpg.render_dearpygui_frame()
    frames += 1
    if app.results and frames > 20:
        break

# zoom / fit buttons, through the same dispatch shape DPG uses
xaxes = ("ax_i_x", "ax_r_x", "ax_g_x")
yaxes = ("ax_i_y", "ax_r_y", "ax_g_y")
both = xaxes + yaxes
for _ in range(5):
    dpg.render_dearpygui_frame()
before = dpg.get_axis_limits("ax_r_x")
app._on_zoom_btn(0, None, (both, 0.7))          # Zoom + button
assert app._zoom_anim, "zoom targets not queued"
for _ in range(80):
    app._tick_zoom_anim()
    dpg.render_dearpygui_frame()
    if not app._zoom_anim and not app._zoom_release:
        break
assert not app._zoom_anim, f"zoom animation did not settle: {app._zoom_anim}"
after = dpg.get_axis_limits("ax_r_x")
ratio = (after[1] - after[0]) / (before[1] - before[0])
assert 0.65 < ratio < 0.75, f"zoom button had no effect: ratio={ratio:.3f}"

app._on_fit_btn(0, None, xaxes)                 # Fit X button
app._on_fit_btn(0, None, yaxes)                 # Fit Y button
for _ in range(5):
    app._tick_zoom_anim()
    dpg.render_dearpygui_frame()
refit = dpg.get_axis_limits("ax_r_x")
assert (refit[1] - refit[0]) > (after[1] - after[0]), "Fit X did not refit"

# A/B probe markers: snap to nearest data point, readout shows delta
r = app.results[0]
mid, late = len(r.t) // 3, 2 * len(r.t) // 3
hit_a = app._nearest_point("plot_r", float(r.t[mid]), float(r.R[mid]))
assert hit_a, "nearest-point lookup failed"
app._set_probe("plot_r", "A", *hit_a)
hit_b = app._nearest_point("plot_r", float(r.t[late]), float(r.R[late]))
app._set_probe("plot_r", "B", *hit_b)
for _ in range(5):
    dpg.render_dearpygui_frame()
readout = dpg.get_value("probe_results")
assert "A:" in readout and "B:" in readout and "dX=" in readout, readout
assert "dy/dx=" in readout, readout
assert abs(hit_a[0] - r.t[mid]) < 1e-9, "probe A did not snap to data"
assert "ruler" in app._probes["plot_r"], "A-B ruler not drawn"

# arm-then-click flow: key arms, Esc disarms
app._on_probe_key(None, None, "A")
assert app._probe_armed == "A"
app._on_probe_key(None, None, "ESC")
assert app._probe_armed is None

app._clear_probes(("plot_r",))
assert dpg.get_value("probe_results") == ""
for _ in range(3):
    dpg.render_dearpygui_frame()

# STDP sweep: button -> worker -> curve plotted in the STDP tab
import time
app.on_plot_stdp()
assert app.sim_running, "STDP sweep did not start"
t0 = time.time()
while time.time() - t0 < 120:
    app._process_queue()
    app._tick_zoom_anim()
    dpg.render_dearpygui_frame()
    if not app.sim_running and app._stdp_series:
        break
assert app._stdp_series, "STDP sweep produced no curves"
assert app._stdp_summary, "no STDP metrics"
# scatter series carries every real point; the line series has a NaN gap-break
_scatter = [s for s in app._stdp_series
            if dpg.get_item_configuration(s).get("label", "").startswith("##")][0]
stdp_data = dpg.get_value(_scatter)
n_pts = len(app._stdp_ctx["dts"])
assert len(stdp_data[0]) == n_pts, \
    f"expected {n_pts} sweep points, got {len(stdp_data[0])}"
assert any(abs(v) > 0 for v in stdp_data[1]), "STDP curve is all zeros"
# the line series must contain a NaN break so the two branches don't join
_line = [s for s in app._stdp_series
         if not dpg.get_item_configuration(s).get("label", "").startswith("##")][0]
_ly = dpg.get_value(_line)[1]
assert any(v != v for v in _ly), "STDP line has no NaN gap-break"
for _ in range(3):
    dpg.render_dearpygui_frame()

# STDP sweep: run the threaded worker, pump the queue until the curve lands
app.on_plot_stdp()
for _ in range(4000):
    app._process_queue()
    app._tick_zoom_anim()
    dpg.render_dearpygui_frame()
    if app._stdp_series and not app.sim_running:
        break
assert app._stdp_series, "STDP curve not produced"
assert app._stdp_summary, "STDP summary not produced"
print("STDP:", " | ".join(app._stdp_summary))

# DT RANGE bounds the sweep: re-run at +-200 ms and check the span
dpg.set_value("stdp_range_ms", 200.0)
app.on_plot_stdp()
for _ in range(4000):
    app._process_queue(); app._tick_zoom_anim(); dpg.render_dearpygui_frame()
    if app._stdp_series and not app.sim_running:
        break
mx = max(abs(d) for d in app._stdp_ctx["dts"]) * 1e3
assert 150 <= mx <= 200 + 1e-6, f"DT RANGE not honored: max|dt|={mx} ms"
print(f"DT RANGE 200ms -> max|dt|={mx:.4g} ms")
dpg.set_value("stdp_range_ms", 1000.0)   # restore

# every swept dt must be strictly greater than the pulse width (a 1% gap so
# the pulses never touch); smallest |dt| should be ~1.01x width
w = app._stdp_ctx["width"]
mn = min(abs(d) for d in app._stdp_ctx["dts"])
assert mn > w, f"sweep includes |dt| <= width: min|dt|={mn}, width={w}"
assert abs(mn - w * 1.01) < 1e-9, \
    f"smallest |dt| should be 1.01x width, got {mn} vs {w*1.01}"
print(f"STDP sweep: {len(app._stdp_ctx['dts'])} pts, "
      f"min|dt|={mn*1e3:.4g} ms, width={w*1e3:.4g} ms (1% gap)")

# drill into a single STDP timing point -> transient + dT/dG/dR readout
assert app._stdp_ctx, "STDP context not stored for drilldown"
n_dt = len(app._stdp_ctx["dts"])
app._stdp_show_dt(n_dt - 3)            # a dt > 0 point
for _ in range(8):
    dpg.render_dearpygui_frame()
status = dpg.get_item_configuration("run_status")["label"]
assert "dt" in status and "dR" in status, status   # tolerant of format tweaks
n_models = len(app._stdp_ctx["models"])
assert len(app.results) == n_models, \
    f"drilldown made {len(app.results)} transients, expected {n_models}"
print("STDP drilldown:", status.lstrip("● ").encode("ascii", "replace").decode())

# drilldown must only fire ON a point, not in empty canvas
import vatester.app as _m
_dts = app._stdp_ctx["dts"]
_series = [s for s in app._stdp_series
           if not dpg.get_item_configuration(s).get("label", "").startswith("##")][0]
_d = dpg.get_value(_series)
_xj, _yj = float(_d[0][5]), float(_d[1][5])     # a real point
y0a, y1a = dpg.get_axis_limits("stdp_y")
app.run_status_before = dpg.get_item_configuration("run_status")["label"]
called = {"n": 0}
_orig = app._stdp_show_dt
app._stdp_show_dt = lambda i: called.__setitem__("n", called["n"] + 1)
# click far off the curve (same x, way off in y) -> must NOT fire
_m.dpg.get_plot_mouse_pos = lambda: (_xj, y1a + (y1a - y0a))
app._stdp_drilldown()
assert called["n"] == 0, "drilldown fired on empty-canvas click"
# click right on the point -> must fire
_m.dpg.get_plot_mouse_pos = lambda: (_xj, _yj)
app._stdp_drilldown()
assert called["n"] == 1, "drilldown did not fire on an actual point"
app._stdp_show_dt = _orig
print("STDP drilldown proximity gate OK")

# analysis metric switch: G <-> R_mem re-plots and relabels
app.on_analysis_metric("R")
for _ in range(5):
    dpg.render_dearpygui_frame()
assert app.analysis_metric == "R"
assert dpg.get_item_configuration("ana_y")["label"] == "R_mem (ohm)", \
    dpg.get_item_configuration("ana_y")["label"]
assert "R_mem" in dpg.get_value("ana_text"), dpg.get_value("ana_text")
app.on_analysis_metric("G")
for _ in range(5):
    dpg.render_dearpygui_frame()
assert dpg.get_item_configuration("ana_y")["label"] == "G (uS)"
assert "G " in dpg.get_value("ana_text")
print("analysis metric switch OK (G <-> R_mem)")

n_results = len(app.results)
n_series = len(app._series)
n_ana = len(app._ana_series)
n_models = len(app._enabled_keys())     # adapts to whichever .va files exist
va_names = [v.name for v in app.va_files]
dpg.destroy_context()

assert n_models >= 1, "no models enabled"
assert n_results == n_models, \
    f"expected {n_models} results (one per enabled model), got {n_results}"
assert n_series >= 1 + 2 * n_results, \
    f"expected stimulus + R/G per model, got {n_series}"
assert n_ana >= 1, f"expected analysis series, got {n_ana}"
assert any("fefet" in n for n in va_names), va_names
print(f"integration OK: {n_results} result(s), {n_series} plot series, "
      f"{n_ana} analysis series, va files: {va_names}")
