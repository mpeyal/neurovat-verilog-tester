"""Crossbar read path against explicit dense / per-step references."""
import numpy as np

from neurovat.crossbar import input_drive, mvm, normalize, trace_matrix


def test_mvm_matches_dense_matrix_product():
    rng = np.random.default_rng(0)
    W = rng.random((10, 16))                      # 16 inputs x 10 outputs
    x = rng.random(16)
    dense = np.array([sum(W[j, i] * x[i] for i in range(16)) for j in range(10)])
    np.testing.assert_allclose(mvm(W, x), dense, rtol=1e-12)
    np.testing.assert_allclose(mvm(W, x), W @ x, rtol=1e-12)


def test_mvm_batches_over_leading_axes():
    rng = np.random.default_rng(1)
    W = rng.random((10, 16))
    X = rng.random((5, 7, 16))
    out = mvm(W, X)
    assert out.shape == (5, 7, 10)
    for b in range(5):
        for t in range(7):
            np.testing.assert_allclose(out[b, t], W @ X[b, t], rtol=1e-12)


def test_normalize_maps_device_range_to_unit_interval():
    G = np.array([0.5e-6, 1e-6, 5.05e-5, 1e-4, 2e-4])
    np.testing.assert_allclose(normalize(G, 1e-6, 1e-4), [0, 0, 0.5, 1, 1])
    assert np.all(normalize(G, 1e-6, 1e-6) == np.clip(G - 1e-6, 0, 1))


def test_trace_matrix_matches_stepwise_recursion():
    rng = np.random.default_rng(2)
    S = rng.random((50, 16)) < 0.1
    decay = 0.88
    x, ref = np.zeros(16), []
    for t in range(50):
        x[S[t]] = 1.0
        ref.append(x.copy())
        x *= decay
    np.testing.assert_allclose(trace_matrix(S, decay), ref, rtol=1e-12, atol=0)


def test_input_drive_matches_stepwise_dense_products():
    rng = np.random.default_rng(3)
    W = rng.random((10, 16))
    X = trace_matrix(rng.random((40, 16)) < 0.15, 0.9)
    inh = np.zeros(16)
    inh[[2, 9]] = 1.0
    exc = 1.0 - inh
    out = input_drive(W, X, exc, inh, 11.0, 1.5, gain=1.7)
    for t in range(40):
        x = X[t]
        ref = 1.7 * (11.0 * (W @ (x * exc)) - 1.5 * (W @ (x * inh))) / (x.sum() + 1.0)
        np.testing.assert_allclose(out[t], ref, rtol=1e-12, atol=1e-15)


def test_network_membrane_integrates_the_dense_crossbar_product(make_net):
    """With firing disabled, the output membrane is a leaky integral of
    Wn @ trace - recomputed here from the device conductances."""
    net = make_net(n_out=10, neuron={"v_threshold": 1e9, "ipsp_gain": 0.0})
    rng = np.random.default_rng(4)
    net.arrays[0].w = rng.random((10, 16))
    vec = rng.random(16)
    res = net.present(vec, learn=False, clean=True)

    G = net.arrays[0].conductance()
    Wn = (G - net.g_min) / (net.g_max - net.g_min)
    S = net.encode_spikes(vec, np.random.default_rng(20240517), noise=False)
    N, dt = net.N, net.cfg.dt_ms
    x, v = np.zeros(16), np.zeros(10)
    for t in range(S.shape[0]):
        x[S[t]] = 1.0
        drive = N.epsp_gain * (Wn @ x) / (x.sum() + 1.0)
        v = v + dt / N.tau_m * (N.v_rest - v + drive)
        np.testing.assert_allclose(res["v_trace"][t], v, rtol=1e-10, atol=1e-13)
        x *= np.exp(-dt / N.tau_syn)
