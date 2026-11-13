"""Stream mode of the real desktop Trainer: Poisson and scanned-image input,
live raster snapshots, replay, Network / Patterns tabs, input and mode
switches, error text in place."""
from pathlib import Path
import shutil
import tempfile
import time

import dearpygui.dearpygui as dpg
import numpy as np

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


def row_shown(tag):
    return dpg.get_item_configuration(dpg.get_item_parent(tag))["show"]


def row_label(tag):
    return dpg.get_value(dpg.get_item_children(dpg.get_item_parent(tag), 1)[0])


def check_codes(app, set_):
    """The encoding combo: input count, rows, noise default, key, views."""
    assert dpg.get_value("nt_st_code") == "column scan"
    assert set_("nt_st_img_size", 28)
    assert dpg.get_value("nt_st_noise") == dpg.get_value("nt_st_bg") == 10.0
    want = {"column scan": (28, "row", "ms / column"),
            "row scan": (28, "col", "ms / row"),
            "latency": (784, "pixel", None),
            "frozen Poisson": (784, "pixel", None)}
    for code, (n, layout, label) in want.items():
        assert set_("nt_st_code", code)
        assert dpg.get_value("nt_st_code") == code
        inp = app._nt_st_input_cfg()
        assert inp["aff"] == inp["n_pattern"] == n, (code, inp["aff"])
        assert inp["image"]["layout"] == layout
        arch = app._nt_st_arch
        assert not arch.get("error"), arch
        assert arch["n_aff"] == n and arch["layout"] == layout
        assert len(arch["net_idx"]) == n
        assert row_shown("nt_st_bin") == (layout != "pixel")
        assert row_shown("nt_st_code_win") == (layout == "pixel")
        if label:
            assert row_label("nt_st_bin") == label
        noise = sv.noise_default(layout)
        # both pixel codes (latency, frozen Poisson) default to 3 Hz
        assert noise == (sv.PIXEL_NOISE_HZ if layout == "pixel" else 10.0)
        assert sv.PIXEL_NOISE_HZ == 3.0
        assert dpg.get_value("nt_st_noise") == noise, code
        assert dpg.get_value("nt_st_bg") == noise, code
        # the start weight follows the code: 0.5 for frozen Poisson, else 1
        w0 = 0.5 if code == "frozen Poisson" else 1.0
        assert dpg.get_value("nt_st_w0") == w0, code
        assert inp["rule"]["w_init"] == w0, code
        if layout == "pixel":
            assert arch["n_shown"] < n and arch["n_inked"] > 0
            vec = arch["vecs"][0]
            inked = {i for i, v in enumerate(vec) if v >= 0.5}
            assert inked <= set(arch["shown"].tolist())
        else:
            assert arch["n_shown"] == 28
        for tab in ("nt_st_tab_net", "nt_st_tab_pat"):
            assert set_("nt_st_tabs", tab)
            frames(app, 2)
    # a bridge / agent name is normalised, a bad one refused
    assert set_("nt_st_code", "row")
    assert dpg.get_value("nt_st_code") == "row scan"
    assert not set_("nt_st_code", "morse")
    assert dpg.get_value("nt_st_code") == "row scan"
    # a noise value the user changed is not swapped; the untouched bg is
    set_("nt_st_noise", 7.0)
    assert set_("nt_st_code", "ttfs")
    assert dpg.get_value("nt_st_code") == "latency"
    assert dpg.get_value("nt_st_noise") == 7.0
    assert dpg.get_value("nt_st_bg") == sv.PIXEL_NOISE_HZ
    assert set_("nt_st_code", "column scan")
    assert dpg.get_value("nt_st_noise") == 7.0
    assert dpg.get_value("nt_st_bg") == 10.0
    set_("nt_st_noise", 10.0)
    # Poisson input counts as a scan: back to 10
    assert set_("nt_st_code", "latency")
    assert dpg.get_value("nt_st_noise") == sv.PIXEL_NOISE_HZ
    assert set_("nt_st_input", "Poisson")
    assert dpg.get_value("nt_st_noise") == 10.0
    assert set_("nt_st_input", "images")
    assert dpg.get_value("nt_st_noise") == sv.PIXEL_NOISE_HZ
    # latency <-> frozen Poisson (same layout): the start weight moves, the
    # noise stays at 3 Hz; a start weight the user changed stays
    assert set_("nt_st_code", "frozen Poisson")
    assert dpg.get_value("nt_st_noise") == dpg.get_value("nt_st_bg") == \
        sv.PIXEL_NOISE_HZ
    assert dpg.get_value("nt_st_w0") == 0.5
    assert set_("nt_st_code", "latency")
    assert dpg.get_value("nt_st_noise") == sv.PIXEL_NOISE_HZ
    assert dpg.get_value("nt_st_w0") == 1.0
    set_("nt_st_w0", 0.8)
    assert set_("nt_st_code", "frozen Poisson")
    assert dpg.get_value("nt_st_w0") == 0.8
    assert set_("nt_st_code", "latency")
    assert dpg.get_value("nt_st_w0") == 0.8
    set_("nt_st_w0", 1.0)
    # Poisson input: back to the paper's 0.475
    assert set_("nt_st_input", "Poisson")
    assert dpg.get_value("nt_st_w0") == 0.475
    assert set_("nt_st_input", "images")
    assert dpg.get_value("nt_st_w0") == 1.0
    # a latency network: baseline, train, test; the pixel views
    set_("nt_st_seconds", 3)
    app.on_nt_stream_reset()
    for learn in (False, True, False):
        app._nt_stream_start(learn)
        assert app.trainer_running, app._nt_stream_error
        pump(app)
    r = app._nt_stream_last
    assert r["phase"] == "test" and r["afferents"] == 784
    assert r["images"]["code"] == "latency" and r["pattern_ms"] == 80.0
    assert app._nt_stream_session.P.n_afferents == 784
    verdict = dpg.get_value("nt_st_verdict")
    assert "encoding latency" in verdict and "trained TP" in verdict
    assert dpg.get_value("nt_st_r_p0")[0], "no latency spikes drawn"
    ys = dpg.get_value("nt_st_r_p0")[1]
    assert max(ys) < app._nt_st_arch["n_shown"]
    assert "pixel (r, c)" in dpg.get_item_configuration("nt_st_ry")["label"]
    assert np.shape(app._nt_st_net_w) == (1, 784)
    assert set_("nt_st_tabs", "nt_st_tab_net")
    frames(app, 2)
    # the heat map: the image and one texture per neuron, drawn once
    kids = dpg.get_item_children("nt_st_net_dl", 2)
    images = [i for i in kids if "DrawImage" in dpg.get_item_type(i)]
    assert len(images) == 2, len(images)
    texts = [dpg.get_item_configuration(i).get("text") for i in kids]
    assert texts.count("n1 w") == 1 and texts.count("afferents") == 1
    assert ("w", 0) in app._nt_st_texs
    assert set_("nt_st_tabs", "nt_st_tab_pat")
    frames(app, 2)
    # the code changes on a trained network: Test refuses
    session = app._nt_stream_session
    for code in ("frozen Poisson", "column scan"):
        assert set_("nt_st_code", code)
        app._nt_stream_start(False)
        assert not app.trainer_running
        assert "settings changed" in app._nt_stream_error
        assert app._nt_stream_session is session
    # so does a window change of a pixel code
    assert set_("nt_st_code", "latency")
    set_("nt_st_code_win", 40.0)
    app._nt_stream_start(False)
    assert not app.trainer_running and "settings changed" in \
        app._nt_stream_error
    set_("nt_st_code_win", 50.0)
    # frozen Poisson and row scan run end to end (fresh networks)
    for code in ("frozen Poisson", "row scan"):
        assert set_("nt_st_code", code)
        app.on_nt_stream_reset()
        app._nt_stream_start(False)
        assert app.trainer_running, app._nt_stream_error
        pump(app)
        assert app._nt_stream_last["images"]["code"] == code
        for tab in ("nt_st_tab_net", "nt_st_tab_pat", "nt_st_tab_raster"):
            assert set_("nt_st_tabs", tab)
            frames(app, 2)
    assert "pixel column" in dpg.get_item_configuration("nt_st_ry")["label"]
    assert set_("nt_st_code", "column scan")
    assert dpg.get_value("nt_st_noise") == 10.0
    set_("nt_st_img_size", 14)
    app.on_nt_stream_reset()


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
            # Test first: the untrained baseline freezes the Poisson patterns
            app._nt_stream_start(False)
            assert app.trainer_running, app._nt_stream_error
            pump(app)
            r = app._nt_stream_last
            assert r["phase"] == "untrained baseline" and r["writes"] == 0
            assert "UNTRAINED baseline" in dpg.get_value("nt_st_verdict")
            frozen = app._nt_stream_session.copies
            assert frozen and all(len(a) for a, _t in frozen)
            app._nt_stream_start(True)
            views = pump(app)
            r = app._nt_stream_last
            assert r and r["input"] == sv.POISSON and r["pattern_ms"] == 50.0
            assert r["phase"] == "train"
            for (a1, t1), (a2, t2) in zip(frozen,
                                          app._nt_stream_session.copies):
                assert (a1 == a2).all() and (t1 == t2).all()
            assert "Untrained baseline of this network" in dpg.get_value(
                "nt_st_verdict")
            assert views > 0, "no live raster snapshot drawn"
            # the live series, the other-windows one included
            for k in ("hit", "oth", "fa"):
                xs, ys = dpg.get_value(f"nt_st_s_{k}")[:2]
                assert xs, f"empty live series {k}"
            assert all(0.0 <= v <= 100.0 for v in
                       dpg.get_value("nt_st_s_oth")[1])
            assert app._nt_st_rec is not None
            assert dpg.get_value("nt_st_r_bg")[0], "empty Poisson raster"
            assert "exactly one" in dpg.get_value("nt_st_verdict")
            assert "TP" in dpg.get_value("nt_st_verdict")
            for tab in ("nt_st_tab_net", "nt_st_tab_pat", "nt_st_tab_raster"):
                assert set_("nt_st_tabs", tab)
                frames(app, 2)
            # image input: the start weight swaps to 1 (noise stays), rows
            # switch
            assert set_("nt_st_input", "images")
            assert dpg.get_value("nt_st_input") == sv.IMAGES
            assert dpg.get_value("nt_st_noise") == 10.0
            assert dpg.get_value("nt_st_w0") == sv.W0_DEFAULT[sv.IMAGES]
            assert any("start weight set to 1" in s for s in app._log_lines)
            row = dpg.get_item_parent("nt_st_aff")
            assert not dpg.get_item_configuration(row)["show"]
            set_("nt_st_neurons", 1)
            set_("nt_st_spacing", "even")
            set_("nt_st_seconds", 6)
            # the trained Poisson network still exists: a Test with other
            # settings refuses rather than replace it by an untrained one
            poisson = app._nt_stream_session
            app._nt_stream_start(False)
            assert not app.trainer_running
            assert "settings changed" in app._nt_stream_error
            assert app._nt_stream_session is poisson
            # Test first: the untrained baseline on a fresh network
            app.on_nt_stream_reset()
            assert app._nt_stream_session is None
            app._nt_stream_start(False)
            assert app.trainer_running, app._nt_stream_error
            pump(app)
            r = app._nt_stream_last
            assert r["phase"] == "untrained baseline" and r["untrained"]
            assert r["input"] == sv.IMAGES and r["pattern_ms"] == 140.0
            assert r["rule"]["w_init"] == 1.0
            assert app._nt_stream_baseline is r
            verdict = dpg.get_value("nt_st_verdict")
            assert "UNTRAINED baseline" in verdict and "learnt" not in verdict
            assert any("UNTRAINED baseline" in s for s in app._log_lines)
            session = app._nt_stream_session
            assert session is not None and session.runs == 1
            copies = session.copies
            app._nt_stream_start(True)
            pump(app)
            r = app._nt_stream_last
            assert app._nt_stream_session is session         # the same network
            assert len(session.copies) == len(copies)        # and patterns
            for (a1, t1), (a2, t2) in zip(copies, session.copies):
                assert (a1 == a2).all() and (t1 == t2).all()
            assert "at least one" in r["hit_rule"]
            assert app._nt_stream_trained_s == r["simulated_s"]
            app._nt_stream_start(False)
            pump(app)
            r = app._nt_stream_last
            assert r["phase"] == "test" and not r["untrained"]
            verdict = dpg.get_value("nt_st_verdict")
            assert "untrained baseline:" in verdict and "untrained TP" in verdict
            assert "trained TP" in verdict and "of training" in verdict
            assert all(v is None or 0.0 <= v < 140.0
                       for v in r["first_spike_latency_ms"])
            ctx = "\n".join(app._nt_stream_context())
            assert "untrained baseline of this network" in ctx
            assert "detection" in ctx and "patterns learnt" not in ctx
            # Train, then a key setting changes: Test refuses and leaves the
            # trained network, its baseline and the last result alone
            last, base = app._nt_stream_last, app._nt_stream_baseline
            trained = app._nt_stream_trained_s
            set_("nt_st_thr", 25.0)
            app._nt_stream_start(False)
            assert not app.trainer_running
            assert app._nt_stream_error == (
                "settings changed since this network was built: press New "
                "network, or restore them")
            assert app._nt_stream_error in dpg.get_value("nt_st_status")
            assert app._nt_stream_session is session
            assert app._nt_stream_trained_s == trained > 0.0
            assert app._nt_stream_baseline is base
            assert app._nt_stream_last is last
            # restored: Test runs on the trained network again
            set_("nt_st_thr", 0.0)
            app._nt_stream_start(False)
            assert app.trainer_running, app._nt_stream_error
            pump(app)
            r = app._nt_stream_last
            assert r["phase"] == "test" and not r["untrained"]
            assert app._nt_stream_session is session
            assert app._nt_stream_baseline is base
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
            check_codes(app, set_)
            # switch input back and mode out and in
            assert set_("nt_st_input", "Poisson")
            assert dpg.get_value("nt_st_noise") == 10.0
            assert dpg.get_value("nt_st_w0") == sv.W0_DEFAULT[sv.POISSON]
            # a start weight the user changed is not swapped
            set_("nt_st_w0", 0.6)
            assert set_("nt_st_input", "images")
            assert dpg.get_value("nt_st_w0") == 0.6
            assert set_("nt_st_input", "Poisson")
            assert dpg.get_value("nt_st_w0") == 0.6
            set_("nt_st_w0", sv.W0_DEFAULT[sv.POISSON])
            assert set_("nt_mode", "supervised")
            frames(app, 2)
            assert set_("nt_mode", "unsupervised stream")
            frames(app, 2)
            errs = errors(app)
            assert not errs, errs
            print("PASS: stream Poisson + image input, untrained baseline -> "
                  "train -> test, Test refused after a setting change, "
                  "start-weight swap, live raster, replay, "
                  "distractors, errors in place, input / mode switches, "
                  "the four encodings (input count, noise default, Test "
                  "refused after a code change, pixel views)")
        finally:
            app.virtuoso.disconnect()
            dpg.destroy_context()


if __name__ == "__main__":
    main()
