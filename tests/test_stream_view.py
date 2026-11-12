"""GUI-free helpers of the stream-mode views (vatester.stream_view)."""
from pathlib import Path

import numpy as np
import pytest

from neurovat.masquelier import MasquelierParams, Session, pattern_peak
from vatester import stream_view as sv

ROOT = Path(__file__).resolve().parents[1]


def test_parse_indices():
    assert sv.parse_indices("", 5) == []
    assert sv.parse_indices(" 0, 3 ,3", 5) == [0, 3]
    assert sv.parse_indices("1-3", 5) == [1, 2, 3]
    assert sv.parse_indices("all", 4, exclude=[1]) == [0, 2, 3]
    with pytest.raises(ValueError, match="out of range"):
        sv.parse_indices("7", 5)
    with pytest.raises(ValueError, match="not a number"):
        sv.parse_indices("x", 5)


def test_input_kind_normalises():
    assert sv.input_kind("scanned images") == sv.IMAGES
    assert sv.input_kind("images") == sv.IMAGES
    assert sv.input_kind("Poisson") == sv.POISSON
    assert sv.input_kind("") == sv.POISSON


def test_build_image_input_and_errors():
    bank = sv.image_bank("digits", 14)
    img = sv.build_image_input(bank, "0", "all", 14, 10.0, 0.5)
    assert img["picks"] == [0]
    assert img["distract"] == list(range(1, len(bank)))
    assert img["pattern_ms"] == 140.0
    rows, t = img["patterns"][0]
    assert rows.max() < 14 and t.max() < img["pattern_ms"]
    with pytest.raises(ValueError, match="at least one"):
        sv.build_image_input(bank, "", "", 14, 10.0, 0.5)
    with pytest.raises(ValueError, match="both hidden"):
        sv.build_image_input(bank, "0,1", "1", 14, 10.0, 0.5)
    blank = [("empty", np.zeros(14 * 14))]
    with pytest.raises(ValueError, match="no pixel"):
        sv.build_image_input(blank, "0", "", 14, 10.0, 0.5)
    with pytest.raises(ValueError, match="no dataset"):
        sv.image_bank("dataset", 14, ds=None)
    # a non-integer bin still gives a whole-ms section that holds the scan
    img = sv.build_image_input(bank, "0", "", 14, 7.3, 0.5)
    assert img["pattern_ms"] == np.ceil(14 * 7.3)


def test_dataset_images_one_per_class_in_order():
    rng = np.random.default_rng(0)
    imgs = rng.random((12, 20, 20)).astype(np.float32)
    labels = np.array([3, 1, 1, 0, 3, 2, 0, 2, 1, 0, 3, 2])
    out = sv.dataset_images(imgs, labels, 10)
    assert [c for c, _ in out] == ["0", "1", "2", "3"]
    v = out[0][1]
    assert v.shape == (100,) and v.min() == 0.0 and v.max() == 1.0
    inv = sv.dataset_images(imgs, labels, 10, invert=True)
    assert np.allclose(inv[0][1], 1.0 - v)


@pytest.mark.skipif(not (ROOT / "datasets" / "mnist.npz").exists(),
                    reason="datasets/mnist.npz not present")
def test_dataset_images_real_mnist():
    from vatester.datasets import load_npz
    imgs, labels = load_npz(str(ROOT / "datasets" / "mnist.npz"))[:2]
    bank = sv.dataset_images(imgs, labels, 28)
    assert [c for c, _ in bank] == [str(c) for c in range(10)]
    img = sv.build_image_input(bank, "0", "", 28, 10.0, 0.5)
    assert img["pattern_ms"] == 280.0
    assert 50 < img["patterns"][0][0].size < 400


def test_auto_threshold_is_fraction_of_weakest_peak():
    bank = sv.image_bank("digits", 14)
    img = sv.build_image_input(bank, "0,1", "", 14, 10.0, 0.5)
    P = MasquelierParams(n_afferents=14, n_pattern=14, threshold=1.0,
                         pattern_ms=img["pattern_ms"])
    thr = sv.auto_threshold(img["patterns"], P)
    peaks = [pattern_peak(p, P) for p in img["patterns"]]
    assert thr == pytest.approx(sv.AUTO_THR * min(peaks))


def test_shown_afferents_and_events_digest():
    assert sv.shown_afferents(30, 60).tolist() == list(range(30))
    s = sv.shown_afferents(1000, 60)
    assert s.size <= 60 and s[0] == 0 and s[-1] == 999
    a = [(np.array([0, 1]), np.array([0.0, 10.0]))]
    b = [(np.array([0, 1]), np.array([0.0, 20.0]))]
    assert sv.events_digest(a) == sv.events_digest(a)
    assert sv.events_digest(a) != sv.events_digest(b)


def test_section_hit_rules():
    assert not sv.section_hit([], True)
    assert sv.section_hit([(1.0, 0), (5.0, 0), (9.0, 0)], True)
    assert not sv.section_hit([(1.0, 0), (5.0, 0)], False)
    assert sv.section_hit([(1.0, 0), (5.0, 1), (7.0, 1)], False)


def test_view_window_auto():
    assert sv.view_window_ms(0, False, 50.0, 0.4) == 1500.0
    assert sv.view_window_ms(0, True, 280.0, 0.4) == 6 * 2 * 280.0
    assert sv.view_window_ms(2.5, True, 280.0, 0.4) == 2500.0


def _image_session(distract=""):
    bank = sv.image_bank("digits", 10)
    img = sv.build_image_input(bank, "0", distract, 10, 10.0, 0.5)
    P = MasquelierParams(n_afferents=10, n_pattern=10, threshold=1.0, n_out=2,
                         pattern_ms=img["pattern_ms"], spacing="even",
                         bg_hz=10.0, noise_hz=0.0, pattern_frac=0.4, seed=1)
    from dataclasses import replace
    P = replace(P, threshold=sv.auto_threshold(img["patterns"], P))
    return P, Session(P, patterns=img["patterns"],
                      distractors=img["distractors"])


def test_run_view_snapshot_contents():
    P, s = _image_session()
    res = s.run(3.0, record_u=True)
    S = s.stream
    rv = sv.RunView(S, P.pattern_ms, P.n_out, P.threshold, 10.0,
                    np.arange(10), 10, 1000.0, True)
    snap = rv.snapshot(2000.0, res["post_t_ms"], res["post_neuron"],
                       res["u_ms"])
    assert snap["t0"] == pytest.approx(11.0) and snap["t1"] == pytest.approx(12.0)
    xs = snap["pat"][0][0] + snap["bg"][0]
    assert xs and min(xs) >= 11.0 and max(xs) <= 12.0
    # every pattern spike lies in a shaded window of its pattern
    starts, ends = snap["win"][0]
    for x in snap["pat"][0][0]:
        assert any(a <= x <= b for a, b in zip(starts, ends))
    ux, uy = snap["u"]
    assert len(uy) == P.n_out and len(ux) == len(uy[0]) == 1000
    n_in = int(((res["post_t_ms"] > 1000.0) & (res["post_t_ms"] <= 2000.0)).sum())
    assert snap["n_post"] == n_in == sum(snap["post_per_neuron"])
    # the window is clipped to the stream: nothing past the end
    end = rv.snapshot(1e9, res["post_t_ms"], res["post_neuron"])
    assert end["t1"] == pytest.approx(10.0 + S.duration_ms / 1000.0)


def test_run_view_distractors_and_poisson_sample():
    P, s = _image_session("1,2")
    s.run(2.0)
    S = s.stream
    rv = sv.RunView(S, P.pattern_ms, P.n_out, P.threshold, 0.0,
                    np.arange(10), 10, 2000.0, True)
    snap = rv.snapshot(2000.0, [], [])
    assert snap["dis"][0] and not snap["bg"][0]        # distractors, no bg
    assert len(snap["dwin"]) == 2
    # Poisson input: only the sampled afferents, mapped to 0..n_shown-1
    Pp = MasquelierParams(n_afferents=200, n_pattern=100, threshold=50.0,
                          seed=2)
    sp = Session(Pp, 1)
    sp.run(1.0)
    shown = sv.shown_afferents(200, 60)
    rv = sv.RunView(sp.stream, Pp.pattern_ms, 1, Pp.threshold, 0.0, shown,
                    200, 1000.0, False)
    snap = rv.snapshot(1000.0, [], [])
    ys = snap["bg"][1] + sum((p[1] for p in snap["pat"]), [])
    assert ys and max(ys) < shown.size
    x, y = sv.frozen_raster(sp.stream.copies[0], shown)
    assert len(x) == len(y) and (not y or max(y) < shown.size)


def test_first_spike_latency_takes_the_first_spike_of_each_answered_window():
    # windows of 100 ms starting at 0, 200, 400; neuron 0 answers 0 and 400
    post_t = [12.0, 30.0, 55.0, 160.0, 408.0, 420.0, 230.0]
    post_j = [0, 0, 0, 0, 0, 0, 1]
    post_lab = [0, 0, 0, -1, 0, 0, 0]
    post_lat = [12.0, 30.0, 55.0, np.nan, 8.0, 20.0, 30.0]
    assert sv.first_spike_latency(post_t, post_j, post_lab, post_lat, 0.0,
                                  0, 0) == pytest.approx(10.0)    # (12 + 8) / 2
    # only windows starting in the scored part count
    assert sv.first_spike_latency(post_t, post_j, post_lab, post_lat, 300.0,
                                  0, 0) == pytest.approx(8.0)
    assert sv.first_spike_latency(post_t, post_j, post_lab, post_lat, 0.0,
                                  0, 1) is None


def test_detection_and_training_effect_wording():
    clean = {"tp": 17, "fn": 0, "fp": 0, "tn": 18, "balanced_accuracy": 1.0}
    noisy = {"tp": 17, "fn": 0, "fp": 8, "tn": 10, "balanced_accuracy": 0.78}
    assert sv.detects(clean) and not sv.detects(noisy) and not sv.detects(None)
    assert sv.window_rates(noisy) == pytest.approx((1.0, 8 / 18))
    learnt, text = sv.training_effect(noisy, clean)
    assert learnt and text.startswith("learnt") and "now detects" in text
    # 11/18 -> 2/18 false with every hit held: learnt, though 2/18 is just
    # above the 10 % detection line
    eleven = {"tp": 17, "fn": 0, "fp": 11, "tn": 7, "balanced_accuracy": 0.69}
    two = {"tp": 17, "fn": 0, "fp": 2, "tn": 16, "balanced_accuracy": 0.94}
    assert not sv.detects(two)
    learnt, text = sv.training_effect(eleven, two)
    assert learnt and "still above" in text
    lost = {"tp": 10, "fn": 7, "fp": 0, "tn": 18, "balanced_accuracy": 0.79}
    assert not sv.training_effect(noisy, lost)[0]       # hits not held
    learnt, text = sv.training_effect(clean, clean)
    assert not learnt and "already" in text
    learnt, text = sv.training_effect(clean, noisy)
    assert not learnt and "worse" in text
    assert sv.stream_differs({"noise": 10.0, "bg": 10.0},
                             {"noise": 20.0, "bg": 10.0}) == ["noise"]
    per = [{"neuron": 0, "best_pattern": 0, "windows": noisy},
           {"neuron": 1, "best_pattern": 0, "windows": clean}]
    rows = sv.pattern_rows(per, 2)
    assert rows[0]["neuron"] == 1 and rows[0]["detects"]
    assert rows[1]["neuron"] is None and not rows[1]["detects"]


def test_image_input_starts_at_weight_one():
    assert sv.W0_DEFAULT[sv.POISSON] == 0.475       # the paper value, unchanged
    assert sv.W0_DEFAULT[sv.IMAGES] == 1.0


def test_code_names_layouts_and_noise_defaults():
    assert sv.code_name("row") == "row scan"
    assert sv.code_name("ttfs") == "latency"
    assert sv.code_name(None) == "column scan"
    with pytest.raises(ValueError):
        sv.code_name("morse")
    assert [sv.code_layout(c) for c in sv.CODES] == ["row", "col", "pixel",
                                                     "pixel"]
    assert sv.PIXEL_NOISE_HZ == 3.0 and sv.NOISE_HZ == 10.0
    assert sv.noise_default("pixel") == sv.PIXEL_NOISE_HZ
    assert sv.noise_default("row") == sv.noise_default(None) == sv.NOISE_HZ
    # the default follows the layout; a value the user edited stays
    assert sv.swapped_default(10.0, "row", "pixel") == 3.0
    assert sv.swapped_default(3.0, "pixel", "col") == 10.0
    assert sv.swapped_default(7.0, "row", "pixel") == 7.0
    assert sv.swapped_default(10.0, "pixel", "row") == 10.0   # edited to 10
    assert set(sv.CODE_TIPS) == set(sv.CODES)
    assert "unreliable" in sv.CODE_TIPS["frozen Poisson"]


def test_line_labels_timing_and_explainer():
    assert sv.line_label("row", 3, 28) == "row 3"
    assert sv.line_label("col", 3, 28) == "col 3"
    assert sv.line_label("pixel", 2 * 28 + 5, 28) == "(2,5)"
    assert sv.LINE_NAME == {"row": "pixel row", "col": "pixel column",
                            "pixel": "pixel (r, c)"}
    assert sv.code_timing({"layout": "row", "bin_ms": 10.0,
                           "code": "column scan"}) == \
        "column scan, 10 ms / column"
    assert sv.code_timing({"layout": "col", "bin_ms": 5.0,
                           "code": "row scan"}) == "row scan, 5 ms / row"
    assert sv.code_timing({"layout": "pixel", "window_ms": 50.0,
                           "code": "latency"}) == "latency, 50 ms window"
    for code in sv.CODES:
        text = sv.code_explainer(code, 28, 10.0, 50.0, 80.0, 1)
        assert "Section 80 ms" in text
    assert "pixel row" in sv.code_explainer("column scan", 28, 10, 0, 280)
    assert "pixel column" in sv.code_explainer("row scan", 28, 10, 0, 280)
    assert "784 lines" in sv.code_explainer("latency", 28, 10, 50, 80)
    assert "seed 4" in sv.code_explainer("frozen Poisson", 28, 10, 50, 80, 4)


def test_pixel_lines_hold_every_inked_pixel():
    bank = sv.image_bank("digits", 28)
    vec = bank[0][1]
    inked = np.nonzero(vec >= 0.5)[0]
    shown, n_inked = sv.pixel_lines([vec], 0.5, 784)
    assert n_inked == inked.size and set(inked) <= set(shown.tolist())
    assert shown.size == inked.size + sv.PIXEL_OTHERS
    assert (np.diff(shown) > 0).all() and shown.max() < 784
    # the inked pixels are the only lines a latency pattern drives
    lat = sv.build_image_input(bank, "0", "", 28, 10.0, 0.5, code="latency")
    assert set(lat["patterns"][0][0].tolist()) <= set(shown.tolist())
    # more inked pixels than room: an even sample, still capped
    full = np.ones(784)
    shown, n_inked = sv.pixel_lines([full], 0.5, 784, cap=100, others=10)
    assert shown.size == n_inked == 90
    blank, n0 = sv.pixel_lines([np.zeros(784)], 0.5, 784)
    assert n0 == 0 and blank.size == sv.PIXEL_OTHERS


def test_build_image_input_codes_keep_the_column_scan_default():
    bank = sv.image_bank("digits", 14)
    old = sv.build_image_input(bank, "0", "1", 14, 10.0, 0.5)
    same = sv.build_image_input(bank, "0", "1", 14, 10.0, 0.5,
                                code="column scan")
    assert old["code"] == "column scan" and old["layout"] == "row"
    assert old["n_afferents"] == 14 and old["pattern_ms"] == 140.0
    assert sv.events_digest(old["patterns"]) == sv.events_digest(same["patterns"])
    for key in ("patterns", "distractors", "picks", "distract", "labels",
                "dlabels", "vecs", "dvecs", "size", "bin_ms", "pattern_ms"):
        assert key in old
    row = sv.build_image_input(bank, "0", "", 14, 10.0, 0.5, code="row scan")
    assert row["layout"] == "col" and row["n_afferents"] == 14
    lat = sv.build_image_input(bank, "0", "", 14, 10.0, 0.5, code="latency")
    assert lat["layout"] == "pixel" and lat["n_afferents"] == 196
    assert lat["patterns"][0][0].max() < 196 and lat["window_ms"] == 50.0
    poi = sv.build_image_input(bank, "0", "1", 14, 10.0, 0.5,
                               code="frozen Poisson")
    again = sv.build_image_input(bank, "0", "1", 14, 10.0, 0.5,
                                 code="frozen Poisson")
    assert sv.events_digest(poi["patterns"] + poi["distractors"]) == \
        sv.events_digest(again["patterns"] + again["distractors"])
    assert sv.events_digest(old["patterns"], "column scan") != \
        sv.events_digest(old["patterns"])
    assert sv.CODES == list(__import__("neurovat.encoding", fromlist=["x"])
                            .STREAM_CODES)
    # every code's events fit the engine, and auto_threshold takes the dict
    for img in (old, row, lat, poi):
        P = MasquelierParams(n_afferents=img["n_afferents"],
                             n_pattern=img["n_afferents"], w_init=1.0,
                             pattern_ms=img["pattern_ms"], threshold=1.0)
        assert sv.auto_threshold(img, P) == sv.auto_threshold(img["patterns"], P)
        Session(P, patterns=img["patterns"]).run(1.0, learn=False)
