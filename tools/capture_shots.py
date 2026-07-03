"""Live screenshot harness for the NeuroVAT GUI.

Drives the REAL app (vatester.app.App) through its main screens off the normal
event loop and saves a PNG of each via dpg.output_frame_buffer. Deterministic and
focus-independent (reads the GL back buffer, not the OS window).

Run from the repo root:
    python tools/capture_shots.py

Outputs PNGs into results/deck/.  Read-only against the app: it only imports and
calls public action methods, never edits app code.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dearpygui.dearpygui as dpg          # noqa: E402
from vatester.app import App               # noqa: E402

OUTDIR = os.path.join(ROOT, "results", "deck")
MODEL = os.path.join(ROOT, "saved_model", "ecfetv3_xor_neuro_model.json")

os.makedirs(OUTDIR, exist_ok=True)
app = App(ROOT)
app.build()                                # creates + shows the viewport

# Bigger viewport => crisper captures.
try:
    dpg.set_viewport_width(1680)
    dpg.set_viewport_height(1014)
except Exception as e:                      # noqa: BLE001
    print("[cap] viewport resize skipped:", e)


def tick():
    """One frame: replicate App.run()'s per-frame work, then render."""
    app._process_queue()
    app._tick_zoom_anim()
    app._tick_probe_drag()
    app._hide_tip_if_stale()
    app._tick_menu_dismiss()
    app._nt_tick_layout()
    app._nt_tick_paint()
    app._nt_tick_cell_anim()
    app._watch_code()
    app._nt_watch_patterns()
    dpg.render_dearpygui_frame()


def pump(frames):
    for _ in range(frames):
        if not dpg.is_dearpygui_running():
            break
        tick()


def pump_until(done, timeout=150.0, settle=25):
    """Render frames until done() is True (or timeout), then settle frames."""
    t0 = time.perf_counter()
    while not done():
        if not dpg.is_dearpygui_running():
            return
        if time.perf_counter() - t0 > timeout:
            print(f"[cap]   timeout after {timeout:.0f}s waiting on completion")
            break
        tick()
    pump(settle)


def shot(name):
    """Save the current frame to results/deck/<name>.png."""
    path = os.path.join(OUTDIR, f"{name}.png")
    pump(6)                                 # let layout settle
    try:
        dpg.output_frame_buffer(file=path)
        tick()                              # flush the capture
        pump(2)
    except Exception as e:                  # noqa: BLE001
        print(f"[cap] FAILED {name}: {e!r}")
        return False
    ok = os.path.isfile(path) and os.path.getsize(path) > 5000
    print(f"[cap] {'saved ' if ok else 'EMPTY '} {name}.png"
          + ("" if ok else "  (will fall back)"))
    return ok


def center(tag):
    dpg.set_value("center_tabs", tag)
    try:
        app._on_center_tab()                # swap the context-sensitive left panel
    except Exception:                       # noqa: BLE001
        pass
    pump(10)


def subtab(label):
    """Select a Neuro-Trainer viz sub-tab by its (stripped) label."""
    kids = dpg.get_item_children("nt_viz_tabs", 1) or []
    for k in kids:
        lbl = (dpg.get_item_label(k) or "").strip()
        if lbl == label.strip():
            dpg.set_value("nt_viz_tabs", k)
            try:
                app._on_nt_viz_tab()
            except Exception:               # noqa: BLE001
                pass
            pump(12)
            return True
    print(f"[cap]   sub-tab '{label}' not found")
    return False


def main():
    pump(40)                                # initial settle (rescan, panels)

    # ---- 1. Signal Designer (default LTP/LTD generator is pre-populated) ----
    print("[cap] Signal Designer")
    center("tab_designer")
    shot("01_signal_designer")

    # ---- 2./3. Run a transient -> Results + Analysis -----------------------
    print("[cap] Running transient simulation...")
    app.on_run()
    pump_until(lambda: not app.sim_running, timeout=180)
    center("tab_results")
    shot("02_results")
    center("tab_analysis")
    shot("03_analysis")

    # ---- 4. STDP sweep -----------------------------------------------------
    print("[cap] Running STDP sweep...")
    center("tab_stdp")
    try:
        app.on_plot_stdp()
        pump_until(lambda: not app.sim_running, timeout=180)
    except Exception as e:                  # noqa: BLE001
        print("[cap]   STDP run error:", e)
    shot("04_stdp")

    # ---- 5./6./7. Neuro Trainer: load trained XOR model, train, evaluate ---
    print("[cap] Neuro Trainer")
    center("tab_trainer")
    if os.path.isfile(MODEL):
        app.on_nt_load(MODEL)
        pump(20)
    else:
        print("[cap]   model JSON missing, building fresh:", MODEL)
        app.on_nt_build()
        pump(20)

    # Cap epochs so the run is quick but still draws live curves/rasters.
    try:
        cur = int(app._nt_get("nt_epochs", 20))
        dpg.set_value("nt_epochs", min(cur, 25) if cur else 20)
    except Exception:                       # noqa: BLE001
        pass

    if app.trainer is not None:
        print("[cap]   training...")
        app.on_nt_train()
        pump_until(lambda: not app.trainer_running, timeout=240)
        print("[cap]   evaluating...")
        app.on_nt_test()
        pump_until(lambda: not app.trainer_running, timeout=120)

    subtab("Weights")
    shot("05_trainer_weights")
    if subtab("All weights"):
        shot("06_trainer_allweights")
    if subtab("Out spikes"):
        shot("07_trainer_raster")
    if subtab("Metrics"):
        shot("08_trainer_metrics")
    if subtab("Activity"):
        shot("09_trainer_activity")

    # ---- 8. Verilog-A source editor ---------------------------------------
    print("[cap] Verilog-A source")
    center("tab_source")
    shot("10_verilog_source")

    print("[cap] DONE -> ", OUTDIR)
    app.virtuoso.disconnect()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
