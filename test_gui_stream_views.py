"""Stream mode of the real desktop Trainer: Poisson and scanned-image input,
live raster snapshots, replay, Network / Patterns tabs, input and mode
switches, error text in place."""
from pathlib import Path
import shutil
import tempfile
import time

import dearpygui.dearpygui as dpg

from vatester.app import App
from vatester import stream_view as sv


def frames(app, n=1):
    for _ in range(n):
        app._process_queue()
        app._nt_tick_stream_view()
        app.desktop.tick()
        dpg.render_dearpygui_frame()


def pump(app, timeout=240):
    deadline = time.monotonic() + timeout
    seen = []
    handler = app._on_nt_stream_view
    app._on_nt_stream_view = lambda snap: (seen.append(1), handler(snap))
    try:
        while app.trainer_running and time.monotonic() < deadline:
            frames(app)
        frames(app, 3)
    finally:
        del app._on_nt_stream_view
    assert not app.trainer_running
    return len(seen)


def errors(app):
    return [s for s in app._log_lines
            if "[ui] error" in s or "[stream] error" in s
            or "view error" in s or "preview error" in s]


def main():
    repo = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="stream_gui_") as temp:
        root = Path(temp)
        shutil.copytree(repo / "verilog", root / "verilog")
        shutil.copytree(repo / "twins", root / "twins",
                        ignore=shutil.ignore_patterns("__pycache__"))
        app = App(temp, bridge=False)
        try:
            app.build()
            frames(app, 3)
            set_ = app._nt_set_control
            assert set_("nt_mode", "unsupervised stream")
            frames(app, 2)
            # Poisson input, defaults, short run
            set_("nt_st_aff", 200)
            set_("nt_st_seconds", 4)
            app._nt_stream_start(True)
            views = pump(app)
            r = app._nt_stream_last
            assert r and r["input"] == sv.POISSON and r["pattern_ms"] == 50.0
            assert views > 0, "no live raster snapshot drawn"
            assert app._nt_st_rec is not None
            assert dpg.get_value("nt_st_r_bg")[0], "empty Poisson raster"
            assert "exactly one" in dpg.get_value("nt_st_verdict")
            assert "TP" in dpg.get_value("nt_st_verdict")
            for tab in ("nt_st_tab_net", "nt_st_tab_pat", "nt_st_tab_raster"):
                assert set_("nt_st_tabs", tab)
                frames(app, 2)
            # image input: noise default swaps, rows switch
            assert set_("nt_st_input", "images")
            assert dpg.get_value("nt_st_input") == sv.IMAGES
            assert dpg.get_value("nt_st_noise") == 0.0
            row = dpg.get_item_parent("nt_st_aff")
            assert not dpg.get_item_configuration(row)["show"]
            set_("nt_st_neurons", 1)
            set_("nt_st_spacing", "even")
            set_("nt_st_seconds", 6)
            app._nt_stream_start(True)
            pump(app)
            r = app._nt_stream_last
            assert r["input"] == sv.IMAGES and r["pattern_ms"] == 140.0
            assert "at least one" in r["hit_rule"]
            app._nt_stream_start(False)
            pump(app)
            assert app._nt_stream_last["phase"] == "test"
            assert app._nt_st_arch["images"]
            assert dpg.get_value("nt_st_r_p0")[0], "no pattern spikes drawn"
            # replay: move, play a little, pause
            set_("nt_st_view_t", 2.0)
            assert set_("nt_st_play", 1) and app._nt_st_playing
            t0 = dpg.get_value("nt_st_view_t")
            end = time.monotonic() + 0.4
            while time.monotonic() < end:
                frames(app)
            assert dpg.get_value("nt_st_view_t") > t0
            set_("nt_st_play", 0)
            assert not app._nt_st_playing
            # distractors
            set_("nt_st_img_distract", "all")
            set_("nt_st_seconds", 3)
            app._nt_stream_start(True)
            pump(app)
            assert dpg.get_value("nt_st_r_dis")[0], "no distractor spikes"
            # errors shown in place, not raised
            set_("nt_st_img_pick", "99")
            assert "out of range" in app._nt_st_arch["error"]
            app._nt_stream_start(True)
            assert not app.trainer_running
            assert "out of range" in app._nt_stream_error
            set_("nt_st_img_pick", "0")
            set_("nt_st_img_distract", "")
            set_("nt_st_imgset", "dataset")
            if not app._nt_ds:
                assert "no dataset" in app._nt_st_arch["error"]
            mnist = repo / "datasets" / "mnist.npz"
            if mnist.exists():      # the Load MNIST path, local file only
                (root / "datasets").mkdir(exist_ok=True)
                shutil.copy(mnist, root / "datasets" / "mnist.npz")
                set_("nt_st_img_size", 28)
                app.on_nt_download_mnist()
                deadline = time.monotonic() + 120
                while app._nt_busy_load and time.monotonic() < deadline:
                    frames(app)
                frames(app, 3)
                assert not app._nt_st_arch.get("error"), app._nt_st_arch
                assert app._nt_st_arch["pattern_ms"] == 280.0
                set_("nt_st_seconds", 3)
                app._nt_stream_start(True)
                pump(app)
                assert app._nt_stream_last["afferents"] == 28
                set_("nt_st_img_size", 14)
            set_("nt_st_imgset", "digits")
            # switch input back and mode out and in
            assert set_("nt_st_input", "Poisson")
            assert dpg.get_value("nt_st_noise") == 10.0
            assert set_("nt_mode", "supervised")
            frames(app, 2)
            assert set_("nt_mode", "unsupervised stream")
            frames(app, 2)
            errs = errors(app)
            assert not errs, errs
            print("PASS: stream Poisson + image input, live raster, replay, "
                  "distractors, errors in place, input / mode switches")
        finally:
            app.virtuoso.disconnect()
            dpg.destroy_context()


if __name__ == "__main__":
    main()
