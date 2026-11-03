"""End-to-end probe-drag test with REAL OS mouse input (Windows)."""
import os
import sys
import ctypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
u32 = ctypes.windll.user32
try:
    u32.SetProcessDPIAware()
except Exception:
    pass

import dearpygui.dearpygui as dpg
from vatester.app import App

LEFTDOWN, LEFTUP = 0x0002, 0x0004


def render(app, n=3):
    for _ in range(n):
        app._process_queue()
        app._tick_zoom_anim()
        app._tick_probe_drag()
        app._hide_tip_if_stale()
        dpg.render_dearpygui_frame()


def put_cursor(sx, sy, app, frames=3):
    u32.SetCursorPos(int(sx), int(sy))
    render(app, frames)


app = App(os.getcwd())
app.build()
render(app, 30)
app.on_run()
for _ in range(800):
    render(app, 1)
    if app.results and not app.sim_running:
        break
dpg.set_value("center_tabs", "tab_results")
render(app, 10)

print("pan_button configurable:",
      "pan_button" in dpg.get_item_configuration("plot_r"))

# ---- screen <-> client calibration --------------------------------------
vx, vy = dpg.get_viewport_pos()
put_cursor(vx + 300, vy + 300, app, 5)
dmx, dmy = dpg.get_mouse_pos(local=False)
offx, offy = (vx + 300) - dmx, (vy + 300) - dmy
print(f"client->screen offset: ({offx:.0f}, {offy:.0f})")

# ---- client px <-> plot units calibration over plot_r --------------------
rmin = dpg.get_item_rect_min("plot_r")
rsz = dpg.get_item_rect_size("plot_r")
c0 = (rmin[0] + rsz[0] * 0.55, rmin[1] + rsz[1] * 0.5)
put_cursor(c0[0] + offx, c0[1] + offy, app, 5)
pp0 = dpg.get_plot_mouse_pos()
put_cursor(c0[0] + 120 + offx, c0[1] + offy, app, 5)
pp1 = dpg.get_plot_mouse_pos()
put_cursor(c0[0] + offx, c0[1] + 80 + offy, app, 5)
pp2 = dpg.get_plot_mouse_pos()
ux = (pp1[0] - pp0[0]) / 120.0          # plot-x units per px
uy = (pp2[1] - pp0[1]) / 80.0           # plot-y units per px
print(f"plot mouse at center: {pp0} | ux={ux:.3g}/px uy={uy:.3g}/px")
assert abs(ux) > 0, "plot mouse pos not tracking the real cursor"

# ---- place probe A with a REAL click at plot center ----------------------
put_cursor(c0[0] + offx, c0[1] + offy, app, 4)
app._arm_probe("A")
render(app, 2)
u32.mouse_event(LEFTDOWN, 0, 0, 0, 0)
render(app, 2)
u32.mouse_event(LEFTUP, 0, 0, 0, 0)
render(app, 6)
probes = app._probes.get("plot_r", {})
assert "A" in probes, "real click did not place probe A"
m = probes["A"]
x_start = m["x"]
print(f"probe A placed at x={x_start:.4f}")

# ---- drag it: cursor onto the probe, press, move right, release ---------
ppx = c0[0] + (m["x"] - pp0[0]) / ux
ppy = c0[1] + (m["y"] - pp0[1]) / uy
put_cursor(ppx + offx, ppy + offy, app, 6)      # hover -> pan parks
pan_now = dpg.get_item_configuration("plot_r").get("pan_button")
print("pan_button while on probe:", pan_now)
lim_before = tuple(dpg.get_axis_limits("ax_r_x"))

u32.mouse_event(LEFTDOWN, 0, 0, 0, 0)
render(app, 3)
print("dragging flag after press:", app._dragging)
MOVE = 0x0001
for k in range(1, 13):
    u32.mouse_event(MOVE, 14, 0, 0, 0)       # true injected relative move
    render(app, 2)
    if k in (1, 6, 12):
        print(f"  step {k}: io mouse={dpg.get_mouse_pos(local=False)} "
              f"probe x={app._probes['plot_r']['A']['x']:.4f}")
u32.mouse_event(LEFTUP, 0, 0, 0, 0)
render(app, 8)

lim_after = tuple(dpg.get_axis_limits("ax_r_x"))
x_end = app._probes["plot_r"]["A"]["x"]
moved = x_end - x_start
expect = 14 * 12 * ux
print(f"probe moved dx={moved:.4f} (expected ~{expect:.4f})")
print(f"axis x-limits before {lim_before} after {lim_after}")
panned = abs(lim_after[0] - lim_before[0]) > 1e-9

dpg.destroy_context()
assert moved > 0, "probe moved the WRONG DIRECTION"
assert abs(expect) * 0.5 < moved < abs(expect) * 2.0, \
    f"probe motion miscalibrated: {moved:.4f} vs {expect:.4f}"
assert not panned, "CANVAS PANNED instead of dragging the probe"
print("REAL-INPUT PROBE DRAG: PASS (probe moved correctly, no pan)")
