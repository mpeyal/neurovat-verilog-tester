"""Engine cross-check (validation/engine): NeuroVAT's stream engine vs Brian2,
its LIF vs snnTorch, and the Brian2 port of ECFET v3. The Brian2 / snnTorch
tests skip cleanly when those packages are not installed; the helper tests
always run."""
import math

import numpy as np
import pytest

from neurovat import masquelier as M
from validation.engine import common as C


# ---- helpers (no optional dependency) -------------------------------------
def test_on_grid_times_bin_back_to_their_step():
    k = np.arange(0, 200000, 7)
    for dt in (0.1, 0.05, 0.025, 0.4):
        t = C.on_grid(k, dt)
        assert (np.floor(t / dt).astype(np.int64) == k).all()
        assert np.abs(t - k * dt).max() < 1e-9


def test_dedupe_keeps_the_first_spike_of_each_afferent_and_bin():
    aff = np.array([3, 3, 1, 3, 1])
    k = np.array([5, 5, 5, 6, 5])
    keep = C.dedupe(aff, k)
    assert keep.tolist() == [True, False, True, True, False]


def test_match_spikes_and_first_divergence():
    a = np.array([1.0, 2.0, 3.0, 10.0])
    b = np.array([1.05, 2.0, 3.5, 10.0])
    m = C.match_spikes(a, b, 0.1)
    assert m["matched"] == 3 and m["match_frac"] == 0.75
    assert C.first_divergence(a, b, 0.1) == 3.0
    assert C.first_divergence(a, a, 1e-9) is None


# ---- Brian2 ------------------------------------------------------------------
def _small(seconds=1.0, seed=3):
    P = M.MasquelierParams(n_afferents=200, n_pattern=100, threshold=50.0,
                           base_s=seconds, n_repeats=1, seed=seed)
    S = M.make_stream(P, np.random.default_rng(seed))
    G, _moved, _dropped = C.grid_stream(S, P.sim_dt_ms)
    return P, G


@pytest.mark.parametrize("learn", [False, True])
def test_brian2_reproduces_the_stream_engine_on_grid_input(learn):
    pytest.importorskip("brian2", exc_type=ImportError)
    from validation.engine import brian2_model as B
    P, S = _small()
    w0 = None if learn else np.random.default_rng(5).uniform(0, 1, P.n_afferents)
    r = M.run(P, stream=S, learn=learn, weights=w0, record_u=True)
    b = B.run(P, S, weights=w0, learn=learn, record_u=True)
    assert len(r["post_t_ms"]) > 20
    np.testing.assert_allclose(b["post_t_ms"], r["post_t_ms"], atol=1e-9)
    np.testing.assert_allclose(b["weights"], np.ravel(r["weights"]), atol=1e-12)
    um = B.per_ms_max(b["u"], b["ready"], P.sim_dt_ms)
    nu = r["u_ms"][:, 0].astype(float)
    n = min(um.size, nu.size)
    assert (np.isnan(um[:n]) == np.isnan(nu[:n])).all()
    ok = np.isfinite(nu[:n])
    np.testing.assert_allclose(um[:n][ok], nu[:n][ok], rtol=1e-5, atol=1e-4)


def test_brian2_pair_window_matches_neurovat_and_the_analytic_rule():
    pytest.importorskip("brian2", exc_type=ImportError)
    from validation.engine import brian2_compare as BC
    summ, rows = BC.test_b()
    assert summ["post_match"] == 1.0
    assert summ["max_abs_dw_nv_b2"] < 1e-12
    assert summ["max_abs_dw_nv_analytic"] < 1e-12
    assert any(r["dw_analytic"] > 0 for r in rows)
    assert any(r["dw_analytic"] < 0 for r in rows)


def test_brian2_v3_port_tracks_the_twin_on_a_single_pulse():
    pytest.importorskip("brian2", exc_type=ImportError)
    from validation import common, stimuli
    from validation.compare import metrics
    from validation.engine import brian2_v3 as V
    st = stimuli.BY_NAME["single_pulse"]
    t, G, _wall = V.run(st, dt=5e-4)
    tw, _ = common.run_twin("v3", st)
    m = metrics(np.interp(tw["t_s"], t, G), tw["G_S"])
    assert m["nrmse"] < 0.01 and m["maxerr"] < 0.03


# ---- snnTorch ----------------------------------------------------------------
def test_snntorch_euler_map_is_the_neurovat_lif_and_exact_map_differs_by_euler_error():
    pytest.importorskip("snntorch", exc_type=ImportError)
    from dataclasses import replace
    from neurovat.config import NeuronParams
    from neurovat.lif import probe_lif_wave
    from validation.engine import snntorch_compare as T
    N = NeuronParams()
    dt = 0.5
    _t, d = T.drive_wave(dt)
    _, Vn, _, sn = probe_lif_wave(N, d, dt)
    Ve, se = T.snn_neuron(N, d, dt, exact=False)
    assert np.abs(Vn - Ve).max() < 1e-12
    np.testing.assert_allclose(se, sn)
    Nsub = replace(N, v_threshold=1e9)
    errs = []
    for h in (0.5, 0.25):
        _t, dh = T.drive_wave(h)
        _, Vn, _, _ = probe_lif_wave(Nsub, dh, h)
        Vx, _ = T.snn_neuron(Nsub, dh, h, exact=True)
        errs.append(np.abs(Vn - Vx).max())
    # Euler truncation: first order in dt
    assert errs[0] > 1e-4
    assert errs[0] / errs[1] == pytest.approx(2.0, rel=0.15)
