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
