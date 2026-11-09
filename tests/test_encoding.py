"""Spike encoding: rate/Poisson statistics, noise, jitter, determinism."""
import numpy as np

from neurovat import NetConfig
from neurovat.encoding import encode_spikes


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
