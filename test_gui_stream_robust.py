"""Robustness tab of the stream mode in the real desktop Trainer: refusals
(no trained network, settings changed, busy), a short sweep with its plots
and table, Stop, Show in Raster, a worker exception, and a device network
(no untrained baseline; device and energy meter unchanged)."""
from pathlib import Path
import hashlib
import shutil
import tempfile
import time

import dearpygui.dearpygui as dpg
import numpy as np

from vatester.app import App
from vatester import network_stream_robust as rb
from vatester import stream_robust as sr
from vatester import stream_view as sv


def frames(app, n=1):
    for _ in range(n):
        app._process_queue()
        app._nt_tick_stream_view()
        app.desktop.tick()
        dpg.render_dearpygui_frame()


def pump(app, timeout=300, on_frame=None):
    deadline = time.monotonic() + timeout
    while app.trainer_running and time.monotonic() < deadline:
        frames(app)
        if on_frame:
            on_frame()
    frames(app, 3)
    assert not app.trainer_running, "busy flag stuck"


def errors(app):
    return [s for s in app._log_lines
            if "[ui] error" in s or "[stream] error" in s
            or "view error" in s or "preview error" in s
            or "[robust] error" in s or "raster error" in s]


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, float).tobytes()).hexdigest()


def series_points(stem, who, what):
    return dpg.get_value(f"nt_st_rb_{stem}_{who}_{what}")[0]


def retrain_checks(app, s, recall, w0, runs0, stream0):
    """Retrain mode: a tiny sweep, the mode toggle switching the plots,
    Show in Raster with the retrained weights, Stop during training and a
    worker exception. Leaves the tab in recall mode."""
    set_ = app._nt_set_control
    assert app._nt_rb_result is recall
    assert set_("nt_st_rb_mode", "retrain")
    assert dpg.get_value("nt_st_rb_mode") == rb.MODES[1]
    # no retrain result yet: empty plots, labelled
    assert app._nt_rb_result is None and not app._nt_rb_entries
    assert not series_points("del", "trained", "hit")
    assert "no sweep in this mode" in dpg.get_value("nt_st_rb_showing")
    assert "RETRAIN" in dpg.get_value("nt_st_rb_showing")
    assert set_("nt_st_rb_jitter", "3")
    assert set_("nt_st_rb_delete", "30")
    assert set_("nt_st_rb_noise", "40")
    assert set_("nt_st_rb_seconds", 2)
    assert set_("nt_st_rb_train_s", 0)          # = the trained seconds
    assert set_("nt_st_rb_repeats", 2)
    assert "2 training runs per level" in dpg.get_value("nt_st_rb_mode_note")
    texts = []
    handler = app._on_nt_robust_progress
    app._on_nt_robust_progress = lambda d: (texts.append(d.get("text", "")),
                                            handler(d))
    assert app._nt_rb_start(), app._nt_rb_error
    # the mode is held while it runs
    assert not set_("nt_st_rb_mode", "recall")
    assert dpg.get_value("nt_st_rb_mode") == rb.MODES[1]
    pump(app)
    del app._on_nt_robust_progress
    rr = app._nt_rb_result
    assert rr and rr["mode"] == sr.RETRAIN and not rr["stopped"], rr
    assert rr["repeats"] == 2
    assert rr["train_seconds"] == app._nt_stream_trained_s
    assert any(t.startswith("level 1/3, training") for t in texts), texts
    assert any(t.startswith("level 3/3, testing") for t in texts), texts
    assert any("retrained at nominal" in e[0] for e in app._nt_rb_entries) \
        is False                      # no nominal level in this sweep
    showing = dpg.get_value("nt_st_rb_showing")
    assert "RETRAIN" in showing and "2 training runs per level" in showing
    assert dpg.get_value("nt_st_rb_leg_who") == "retrained: hits"
    assert "RETRAIN at each level" in dpg.get_value("nt_st_rb_table")
    assert series_points("del", "trained", "hit") == [30.0]
    assert dpg.get_value("nt_st_rb_noi_band_hit")[0] == [40.0]
    assert digest(s.weights) == w0 and s.runs == runs0
    assert s.stream is stream0                    # the session is not run
    # the GUI result is the core's
    one = sr.sweep(s, {"spike_deletion": [0.3]}, seconds=2.0,
                   nominal=app._nt_stream_trained_with, retrain=True,
                   train_seconds=rr["train_seconds"], repeats=2)
    a = one["factors"][0]["rows"][0]
    b = rb.find_row(rr, "spike_deletion", 0.3)
    for who in ("trained", "untrained"):
        for k in ("hits", "false_windows", "post_spikes"):
            assert a[who][k] == b[who][k], (who, k)
    np.testing.assert_array_equal(a["weights"][0], b["weights"][0])

    # toggle: the plots, table and Raster list switch between the results
    assert set_("nt_st_rb_mode", "recall")
    assert app._nt_rb_result is recall
    assert series_points("del", "trained", "hit") == [0.0, 30.0]
    assert not dpg.get_value("nt_st_rb_noi_band_hit")[0]
    assert dpg.get_value("nt_st_rb_table").startswith("robustness sweep (recall)")
    assert dpg.get_value("nt_st_rb_leg_who") == "trained: hits"
    assert len(dpg.get_item_configuration("nt_st_rb_pick")["items"]) == \
        len(rb.entries(recall))
    assert set_("nt_st_rb_mode", "retrain at each level")
    assert app._nt_rb_result is rr
    assert series_points("del", "trained", "hit") == [30.0]
    assert dpg.get_item_configuration("nt_st_rb_pick")["items"] == \
        [e[0] for e in rb.entries(rr)]

    # Show in Raster replays THAT level's retrained weights (first training):
    # pick a level where they give another spike count than the session's
    pick = None
    for i, (_text, factor, level) in enumerate(rb.entries(rr)):
        own = s.run(2.0, learn=False, stream_key=sr.SWEEP_KEY,
                    **dict(app._nt_stream_trained_with, **{factor: level}))
        row = rb.find_row(rr, factor, level)
        if row["repeats"][0]["post_spikes"] != own["post_spikes"]:
            pick = (i, row, own["post_spikes"])
            break
    assert pick is not None, "retrained == trained at every level"
    assert set_("nt_st_rb_pick", pick[0])
    assert app._nt_rb_show_raster(), app._nt_rb_error
    pump(app)
    want = pick[1]["repeats"][0]["post_spikes"]
    assert app._nt_rb_raster_last == want, (app._nt_rb_raster_last, want)
    assert app._nt_rb_raster_last != pick[2]       # not the session's network
    assert "retrained network" in dpg.get_value("nt_st_rb_status")
    assert app._nt_st_tab_is("nt_st_tab_raster")
    assert digest(s.weights) == w0 and s.runs == runs0
    assert set_("nt_st_tabs", "nt_st_tab_robust")

    # Stop during the training of level 2: level 1 kept, level 2 dropped
    seen = []
    handler = app._on_nt_robust_progress
    app._on_nt_robust_progress = lambda d: (seen.append(d.get("text", "")),
                                            handler(d))

    def stop_in_training():
        if any(t.startswith("level 2/3, training") for t in seen):
            app._nt_rb_stop()
    set_("nt_st_rb_repeats", 1)
    assert app._nt_rb_start(), app._nt_rb_error
    pump(app, on_frame=stop_in_training)
    del app._on_nt_robust_progress
    rs = app._nt_rb_result
    assert rs["stopped"] and rs["mode"] == sr.RETRAIN
    assert sum(len(x["rows"]) for x in rs["factors"]) == 1
    assert "stopped" in dpg.get_value("nt_st_rb_status")
    assert not any(t.startswith("level 2/3, testing") for t in seen), seen
    assert dpg.get_item_configuration("nt_st_rb_mode")["enabled"]

    # a worker exception in retrain mode still ends the run
    real = sr.retrain_once

    def boom(*_a, **_k):
        raise RuntimeError("injected retrain failure")
    sr.retrain_once = boom
    try:
        assert app._nt_rb_start(), app._nt_rb_error
        pump(app)
    finally:
        sr.retrain_once = real
    assert not app.trainer_running
    assert "injected retrain failure" in app._nt_rb_error
    assert "retrain sweep failed" in dpg.get_value("nt_st_rb_status")
    assert app._nt_rb_result is rs              # the kept result is shown
    assert dpg.get_item_configuration("nt_st_rb_mode")["enabled"]
    app._log_lines[:] = [x for x in app._log_lines if "injected" not in x]
    assert set_("nt_st_rb_mode", "recall") and app._nt_rb_result is recall
    assert set_("nt_st_rb_train_s", 0) and set_("nt_st_rb_repeats", 1)
    assert digest(s.weights) == w0 and s.runs == runs0
    return ("retrain sweep (repeats 2, toggle, Show in Raster with the "
            "retrained weights, Stop in training, exception); ")


def main():
    repo = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="stream_robust_") as temp:
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
            assert set_("nt_st_input", "images")
            set_("nt_st_imgset", "digits")
            set_("nt_st_img_size", 10)
            set_("nt_st_bin", 5.0)
            set_("nt_st_neurons", 1)
            set_("nt_st_spacing", "even")
            set_("nt_st_share", 0.5)
            set_("nt_st_seconds", 4)
            assert set_("nt_st_tabs", "nt_st_tab_robust")
            frames(app, 2)

            # refused: no network at all
            assert not app._nt_rb_start()
            assert app._nt_rb_error == rb.NO_NETWORK
            assert rb.NO_NETWORK in dpg.get_value("nt_st_rb_status")
            assert not app.trainer_running
            # refused: only the untrained baseline exists (nothing trained)
            app._nt_stream_start(False)
            pump(app)
            assert app._nt_stream_session is not None
            assert not app._nt_rb_start() and app._nt_rb_error == rb.NO_NETWORK

            # Train with jitter 2 ms; then move the live jitter control: the
            # nominal must be the TRAINED value
            set_("nt_st_jitter", 2.0)
            app._nt_stream_start(True)
            pump(app)
            assert app._nt_stream_trained_s > 0
            assert app._nt_stream_trained_with["jitter_ms"] == 2.0
            set_("nt_st_jitter", 4.0)
            s = app._nt_stream_session
            w0, runs0, stream0 = digest(s.weights), s.runs, s.stream

            # refused: a network setting changed; restored: allowed
            set_("nt_st_thr", 25.0)
            assert not app._nt_rb_start()
            assert app._nt_rb_error == rb.SETTINGS_CHANGED
            assert rb.SETTINGS_CHANGED in dpg.get_value("nt_st_rb_status")
            set_("nt_st_thr", 0.0)

            # a short sweep; the progress messages draw partial results
            assert set_("nt_st_rb_jitter", "3")          # + nothing else
            assert set_("nt_st_rb_delete", "0, 30")
            assert set_("nt_st_rb_noise", "")            # default levels
            assert set_("nt_st_rb_seconds", 2)
            assert set_("nt_st_rb_untrained", 1)
            progress = []
            handler = app._on_nt_robust_progress
            app._on_nt_robust_progress = lambda d: (progress.append(d),
                                                    handler(d))
            assert app._nt_rb_start(), app._nt_rb_error
            assert app.trainer_running
            # busy: Train / Test cannot start mid-sweep, nor a second sweep
            app._nt_stream_start(True)
            assert app._nt_stream_session is s
            assert not app._nt_rb_start() and "busy" in app._nt_rb_error
            app._nt_rb_error = None
            pump(app)
            del app._on_nt_robust_progress
            r = app._nt_rb_result
            assert r and not r["stopped"], r
            assert any(d.get("result") for d in progress), "no partial result"
            assert r["nominal"]["jitter_ms"] == 2.0        # trained, not live
            levels = {b["factor"]: [x["level"] for x in b["rows"]]
                      for b in r["factors"]}
            assert levels["jitter_ms"] == [3.0]
            assert levels["spike_deletion"] == [0.0, 0.3]
            assert levels["noise_hz"] == sr.default_levels("noise_hz", 10.0)
            for b in r["factors"]:
                for row in b["rows"]:
                    assert row["trained"] and row["untrained"]
            for _f, _tag, stem, _xl in rb.FACTOR_TAGS:
                for who in ("trained", "untrained"):
                    for what in ("hit", "false"):
                        assert series_points(stem, who, what), (stem, who, what)
            assert dpg.get_value("nt_st_rb_jit_nom")[0] == [2.0]
            table = dpg.get_value("nt_st_rb_table")
            assert table.startswith("robustness sweep") and "UNTRAINED" in table
            # the session is left as it was
            assert digest(s.weights) == w0 and s.runs == runs0
            assert s.stream is stream0
            # the merged per-level sweep is the single sweep of the core
            one = sr.sweep(s, {"spike_deletion": [0.0, 0.3]}, seconds=2.0,
                           nominal=app._nt_stream_trained_with)
            for a, b in zip(one["factors"][0]["rows"],
                            next(x for x in r["factors"]
                                 if x["factor"] == "spike_deletion")["rows"]):
                for who in ("trained", "untrained"):
                    for k in ("hits", "false_windows", "post_spikes"):
                        assert a[who][k] == b[who][k], (who, k)

            # Show in Raster: the same run as that level of the sweep
            items = dpg.get_item_configuration("nt_st_rb_pick")["items"]
            assert len(items) == len(rb.entries(r)) >= 5
            assert set_("nt_st_rb_pick", 2)
            assert app._nt_rb_show_raster(), app._nt_rb_error
            pump(app)
            factor, level = rb.entries(r)[2][1:]
            row = next(x for b in r["factors"] if b["factor"] == factor
                       for x in b["rows"] if x["level"] == level)
            assert app._nt_rb_raster_last == row["trained"]["post_spikes"]
            assert app._nt_st_tab_is("nt_st_tab_raster")
            assert app._nt_st_rec is not None
            assert dpg.get_value("nt_st_r_bg")[0], "empty raster"
            assert digest(s.weights) == w0 and s.runs == runs0
            retrain_txt = retrain_checks(app, s, r, w0, runs0, stream0)

            # Stop: a long sweep stopped after the first level
            assert set_("nt_st_tabs", "nt_st_tab_robust")
            set_("nt_st_rb_seconds", 3)
            set_("nt_st_rb_jitter", "1, 3, 5")

            def stop_after_first():
                if app._nt_rb_kind == "sweep" and any(
                        b["rows"] for b in (getattr(app, "_rb_seen", None)
                                            or {"factors": []})["factors"]):
                    app._nt_rb_stop()
            seen_handler = app._on_nt_robust_progress

            def keep(d):
                if d.get("result"):
                    app._rb_seen = d["result"]
                seen_handler(d)
            app._on_nt_robust_progress = keep
            assert app._nt_rb_start(), app._nt_rb_error
            pump(app, on_frame=stop_after_first)
            del app._on_nt_robust_progress
            r2 = app._nt_rb_result
            assert r2["stopped"], "Stop did not stop the sweep"
            n_rows = sum(len(b["rows"]) for b in r2["factors"])
            assert 1 <= n_rows < 3 + 4 + 3, n_rows
            assert "stopped" in dpg.get_value("nt_st_rb_status")
            assert "STOPPED" in dpg.get_value("nt_st_rb_table")
            # the stop flag does not stick: the next Train runs in full
            set_("nt_st_jitter", 2.0)

            # an exception in the worker still ends the run
            real = sr.sweep

            def boom(*_a, **_k):
                raise RuntimeError("injected sweep failure")
            sr.sweep = boom
            try:
                assert app._nt_rb_start(), app._nt_rb_error
                pump(app)
            finally:
                sr.sweep = real
            assert not app.trainer_running
            assert "injected sweep failure" in app._nt_rb_error
            assert "injected" in dpg.get_value("nt_st_rb_status")
            assert any("[robust] error" in x for x in app._log_lines)
            app._log_lines[:] = [x for x in app._log_lines
                                 if "injected" not in x]
            # and Train runs afterwards
            app._nt_stream_start(True)
            assert app.trainer_running
            pump(app)
            assert not app._nt_stream_last["stopped"]

            # a pixel code (latency): sweep and Show in Raster
            if dpg.does_item_exist("nt_st_code"):     # the encoding combo
                assert set_("nt_st_code", "latency")
                app.on_nt_stream_reset()
                set_("nt_st_seconds", 2)
                app._nt_stream_start(True)
                assert app.trainer_running, app._nt_stream_error
                pump(app)
                set_("nt_st_rb_jitter", "3")
                set_("nt_st_rb_delete", "30")
                set_("nt_st_rb_noise", "")
                set_("nt_st_rb_seconds", 1)
                assert app._nt_rb_start(), app._nt_rb_error
                pump(app)
                rl = app._nt_rb_result
                noise = app._nt_stream_trained_with["noise_hz"]
                assert [x["level"] for b in rl["factors"]
                        if b["factor"] == "noise_hz" for x in b["rows"]] == \
                    sr.default_levels("noise_hz", noise)
                assert series_points("noi", "untrained", "false")
                assert set_("nt_st_rb_pick", "missing spikes 30")
                assert app._nt_rb_show_raster(), app._nt_rb_error
                pump(app)
                row = next(x for b in rl["factors"]
                           if b["factor"] == "spike_deletion"
                           for x in b["rows"])
                assert app._nt_rb_raster_last == row["trained"]["post_spikes"]
                assert set_("nt_st_code", "column scan")
                app.on_nt_stream_reset()
                code_txt = "latency code sweep + raster; "
            else:
                code_txt = ""

            # a device network: no untrained baseline, the device and the
            # energy meter are not changed by the sweep
            set_("nt_st_synapse", "selected device")
            app.on_nt_stream_reset()
            set_("nt_st_seconds", 2)
            app._nt_stream_start(True)
            if app.trainer_running:
                pump(app, timeout=600)
                ds = app._nt_stream_session
                assert ds is not None and ds.device is not None, \
                    app._nt_stream_error
                # retrain is refused for device synapses (recall works)
                assert set_("nt_st_rb_mode", "retrain")
                assert not app._nt_rb_start()
                assert app._nt_rb_error == rb.DEVICE_RETRAIN
                assert sr.DEVICE_REFUSAL in dpg.get_value("nt_st_rb_status")
                assert sr.DEVICE_REFUSAL in dpg.get_value("nt_st_rb_mode_note")
                assert not app.trainer_running
                assert set_("nt_st_rb_mode", "recall")
                dev = ds.device
                g0 = digest(dev.conductance())
                meter = ds.kw["energy"]
                before = repr(meter.report("eval"))
                set_("nt_st_rb_jitter", "3")
                set_("nt_st_rb_delete", "30")
                set_("nt_st_rb_noise", "26")
                set_("nt_st_rb_seconds", 1)
                assert app._nt_rb_start(), app._nt_rb_error
                pump(app, timeout=600)
                rd = app._nt_rb_result
                assert rd["untrained_note"] == rb.DEVICE_NOTE
                assert all(x["untrained"] is None for b in rd["factors"]
                           for x in b["rows"])
                assert rb.DEVICE_NOTE in dpg.get_value("nt_st_rb_nominal")
                assert not series_points("jit", "untrained", "hit")
                assert series_points("jit", "trained", "hit")
                assert digest(dev.conductance()) == g0
                assert repr(meter.report("eval")) == before
                assert ds.kw["energy"] is meter
                device_txt = "device sweep left the device and meter alone"
            else:
                device_txt = f"device check skipped ({app._nt_stream_error})"
            frames(app, 2)
            errs = errors(app)
            assert not errs, errs
            print("PASS: robustness refusals (no network, untrained only, "
                  "settings changed, busy), sweep with trained nominal, "
                  "progressive plots, table, session unchanged, merge == "
                  "single sweep, Show in Raster, Stop, worker exception; "
                  + retrain_txt + code_txt + device_txt)
        finally:
            app.virtuoso.disconnect()
            dpg.destroy_context()


if __name__ == "__main__":
    main()
