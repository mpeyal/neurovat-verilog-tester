"""Spike encoding: rate/Poisson statistics, noise, jitter, determinism."""
import numpy as np
import pytest

from neurovat import (CrossbarSNN, NetConfig, NeuronParams, STDPParams,
                      device_factory)
from neurovat.encoding import encode_spikes, scan_events


def enc(vec, seed=0, noise=True, mask=None, **cfg):
    c = NetConfig(grid_h=4, grid_w=4, present_ms=1000.0, **cfg)
    return encode_spikes(vec, c, np.random.default_rng(seed), mask, noise)


def test_rate_code_is_poisson_at_the_configured_rate():
    vec = np.linspace(0, 1, 16)
    S = np.mean([enc(vec, seed=s, max_rate_hz=100.0).sum(0) for s in range(40)], 0)
    expected = 1000 * (1 - np.exp(-vec * 100.0 * 1e-3))   # spikes per second
    assert S[0] == 0
    np.testing.assert_allclose(S[1:], expected[1:], rtol=0.12)


def test_encoding_is_deterministic_per_seed():
    vec = np.random.default_rng(1).random(16)
    kw = dict(input_noise=0.2, jitter_ms=3.0, bg_rate_hz=10.0)
    np.testing.assert_array_equal(enc(vec, 5, **kw), enc(vec, 5, **kw))
    assert not np.array_equal(enc(vec, 5, **kw), enc(vec, 6, **kw))


def test_clean_spikes_nest_inside_the_noisy_raster():
    vec = (np.arange(16) % 2).astype(float)
    clean = enc(vec, 3, noise=False, bg_rate_hz=20.0)
    noisy = enc(vec, 3, noise=True, bg_rate_hz=20.0)
    assert np.all(noisy[clean])
    assert clean[:, vec == 0].sum() == 0 and noisy[:, vec == 0].sum() > 0


def test_background_rate_and_signal_mask():
    mask = np.zeros(16)
    mask[:8] = 1.0
    S = enc(np.ones(16), 2, mask=mask, bg_rate_hz=20.0, max_rate_hz=200.0)
    assert S[:, :8].mean() > 5 * S[:, 8:].mean()         # masked: noise only
    assert abs(S[:, 8:].sum(0).mean() - 20.0) < 6.0
    assert enc(np.ones(16), 2, noise=False, mask=mask)[:, 8:].any()


def test_jitter_moves_spikes_but_not_between_inputs():
    vec = np.ones(16)
    base = enc(vec, 4, encoding="temporal", max_rate_hz=50.0)
    jit = enc(vec, 4, encoding="temporal", max_rate_hz=50.0, jitter_ms=4.0)
    assert not np.array_equal(base, jit)
    assert np.all(jit.sum(0) <= base.sum(0)) and jit.sum() > 0.8 * base.sum()
    t0, t1 = np.nonzero(base[:, 0])[0], np.nonzero(jit[:, 0])[0]
    assert len(t0) == len(t1) and 1.0 < np.abs(t1 - t0).std() < 8.0


def test_input_noise_perturbs_rates_and_pattern_window_gates_them():
    vec = np.zeros(16)
    assert enc(vec, 0).sum() == 0
    assert enc(vec, 0, input_noise=0.5).sum() > 0
    S = enc(np.ones(16), 0, pattern_ms=200.0, max_rate_hz=200.0)
    assert S[:400].sum() == 0 and S[600:].sum() == 0 and S[400:600].sum() > 0


def test_latency_code_brighter_pixels_fire_first():
    vec = np.linspace(0.2, 1, 16)
    S = enc(vec, 1, encoding="latency", max_rate_hz=400.0)
    first = np.array([np.nonzero(S[:, i])[0][0] for i in range(16)])
    assert first[-1] < first[0]
    assert np.all(first >= ((1 - vec) * 0.6 * 1000).astype(int))


def _scan_cfg(**kw):
    kw = {"present_ms": 80.0, "scan_bin_ms": 10.0, **kw}
    return NetConfig(grid_h=4, grid_w=6, encoding="scan", **kw)


def test_column_scan_rows_are_inputs_and_columns_are_time_bins():
    cfg = _scan_cfg()
    img = np.zeros((4, 6))
    img[:, 2] = 1.0                                 # a vertical bar, column 2
    S = encode_spikes(img.reshape(-1), cfg, np.random.default_rng(0), noise=False)
    assert S.shape == (80, 4)                       # 4 input lines, not 24
    assert np.nonzero(S.any(1))[0].tolist() == [20]  # all rows, in bin 2 only
    assert S[20].all()
    img = np.zeros((4, 6))
    img[1, :] = 1.0                                 # a horizontal bar, row 1
    S = encode_spikes(img.reshape(-1), cfg, np.random.default_rng(0), noise=False)
    assert np.nonzero(S[:, 1])[0].tolist() == [0, 10, 20, 30, 40, 50]
    assert S[:, [0, 2, 3]].sum() == 0               # only that row's line


def test_column_scan_fires_once_per_inked_pixel():
    cfg = _scan_cfg(scan_threshold=0.5)
    img = np.random.default_rng(3).random((4, 6))
    S = encode_spikes(img.reshape(-1), cfg, np.random.default_rng(0), noise=False)
    assert S.sum() == (img >= 0.5).sum()


def test_column_scan_must_fit_the_presentation():
    cfg = _scan_cfg(present_ms=50.0)                # 6 columns x 10 ms = 60 ms
    with pytest.raises(ValueError, match="does not fit"):
        encode_spikes(np.zeros(24), cfg, np.random.default_rng(0), noise=False)


def test_scan_events_are_the_spikes_of_the_clean_scan_raster():
    cfg = NetConfig(grid_h=5, grid_w=7, encoding="scan", present_ms=30.0,
                    dt_ms=0.5, scan_bin_ms=3.0, scan_threshold=0.4)
    img = np.random.default_rng(8).random((5, 7))
    S = encode_spikes(img.reshape(-1), cfg, np.random.default_rng(0), noise=False)
    rows, times = scan_events(img, 3.0, 0.4)
    step, row = np.nonzero(S)
    assert rows.size == row.size > 0
    assert sorted(zip(rows.tolist(), np.round(times, 9).tolist())) == \
        sorted(zip(row.tolist(), np.round(step * cfg.dt_ms, 9).tolist()))


def test_column_scan_network_has_one_input_per_row():
    net = CrossbarSNN(device_factory("softbound"), NeuronParams(), STDPParams(),
                      _scan_cfg(n_out=2, pattern_set="bars"))
    assert net.n_in == 4 and net.layer_sizes == [4, 2]
    assert net.train_step(net.patterns[0][1], 0)["winner"] in (-1, 0, 1)


# --- image codes of the event stream (stream_events) -------------------------

def _img(seed=3, shape=(6, 9)):
    return np.random.default_rng(seed).random(shape)


def test_stream_column_scan_is_scan_events():
    from neurovat.encoding import stream_events
    img = _img()
    ev = stream_events(img, "column scan", 7.3, 0.4)
    rows, t = scan_events(img, 7.3, 0.4)
    np.testing.assert_array_equal(ev["afferents"], rows)
    np.testing.assert_array_equal(ev["times"], t)
    assert ev["layout"] == "row" and ev["n_afferents"] == 6
    assert ev["pattern_ms"] == np.ceil(9 * 7.3)


def test_row_scan_is_the_column_scan_of_the_transpose():
    from neurovat.encoding import stream_events
    img = _img()
    row = stream_events(img, "row scan", 10.0, 0.5)
    col = stream_events(img.T, "column scan", 10.0, 0.5)
    for k in ("afferents", "times", "n_afferents", "pattern_ms"):
        np.testing.assert_array_equal(row[k], col[k])
    assert row["layout"] == "col" and row["n_afferents"] == 9
    assert row["pattern_ms"] == 60.0              # 6 rows x 10 ms
    # line = pixel column c, time = pixel row r x bin
    r, c = 4, 7
    img2 = np.zeros((6, 9))
    img2[r, c] = 1.0
    ev = stream_events(img2, "row scan", 10.0, 0.5)
    assert ev["afferents"].tolist() == [c] and ev["times"].tolist() == [40.0]


def test_latency_brighter_fires_earlier_with_margins_and_pixel_lines():
    from neurovat.encoding import STREAM_MARGIN_MS, stream_events
    img = np.zeros((3, 4))
    img[0, 1], img[2, 3], img[1, 0], img[1, 1] = 1.0, 0.75, 0.5, 0.2
    ev = stream_events(img, "latency", 10.0, 0.5, window_ms=40.0)
    assert ev["layout"] == "pixel" and ev["n_afferents"] == 12
    t = dict(zip(ev["afferents"].tolist(), ev["times"].tolist()))
    assert sorted(t) == [1, 4, 11]                 # r * W + c; 0.2 is below
    m = STREAM_MARGIN_MS
    assert t[1] == m                                # brightest at the margin
    assert t[11] == m + 20.0                        # halfway to the threshold
    assert t[4] == m + 40.0                         # the threshold fires last
    assert ev["pattern_ms"] == 2 * m + 40.0 and max(t.values()) < ev["pattern_ms"]
    # near-binary image: a volley one margin after the section start
    ev = stream_events((_img() > 0.5).astype(float), "ttfs", 10.0, 0.5)
    assert np.all(ev["times"] == m) and ev["pattern_ms"] == 2 * m + 50.0


def test_frozen_poisson_is_frozen_per_seed_and_follows_intensity():
    from neurovat.encoding import STREAM_MARGIN_MS, stream_events
    img = np.zeros((10, 10))
    img[:5] = 1.0
    img[5:, :5] = 0.5
    a = stream_events(img, "frozen Poisson", 10.0, 0.4, window_ms=200.0, rng=7)
    b = stream_events(img, "frozen Poisson", 10.0, 0.4, window_ms=200.0,
                      rng=np.random.default_rng(7))
    c = stream_events(img, "frozen Poisson", 10.0, 0.4, window_ms=200.0, rng=8)
    np.testing.assert_array_equal(a["afferents"], b["afferents"])
    np.testing.assert_array_equal(a["times"], b["times"])
    assert not np.array_equal(a["times"], c["times"])
    d = stream_events(img, "frozen Poisson", 10.0, 0.4, window_ms=200.0)
    e = stream_events(img, "frozen Poisson", 10.0, 0.4, window_ms=200.0)
    np.testing.assert_array_equal(d["times"], e["times"])   # default seed 0
    n = np.bincount(a["afferents"], minlength=100)
    v = img.reshape(-1)
    assert n[v == 0].sum() == 0                      # dark pixels are silent
    full, half = n[v == 1].mean(), n[v == 0.5].mean()
    assert 15 < full < 25                            # 100 Hz x 200 ms = 20
    assert 0.3 < half / full < 0.7                   # rate ~ intensity
    assert a["times"].min() >= STREAM_MARGIN_MS
    assert a["times"].max() < STREAM_MARGIN_MS + 200.0 < a["pattern_ms"]
    assert a["layout"] == "pixel" and a["n_afferents"] == 100


def test_unknown_stream_code_is_rejected():
    from neurovat.encoding import stream_code, stream_events
    assert stream_code(None) == "column scan"
    assert stream_code("Row") == "row scan"
    with pytest.raises(ValueError, match="unknown stream encoding"):
        stream_events(np.ones((3, 3)), "wavelet")
