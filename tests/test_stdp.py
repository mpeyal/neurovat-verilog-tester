"""Device-local STDP: eligibility, write decisions, and the resulting
conductance changes in a crossbar."""
import numpy as np

from neurovat import STDPParams
from neurovat import stdp


def test_eligibility_matches_the_event_driven_loop():
    rng = np.random.default_rng(0)
    pre = rng.random((60, 16)) < 0.12
    post = rng.random((60, 10)) < 0.05
    decay = np.exp(-1.0 / 20.0)
    x, acc, npost = np.zeros(16), np.zeros((10, 16)), np.zeros(10)
    for t in range(60):
        x[pre[t]] = 1.0
        for j in np.nonzero(post[t])[0]:
            acc[j] += x
            npost[j] += 1
        x *= decay
    got_acc, got_n = stdp.eligibility(pre, post, decay)
    np.testing.assert_allclose(got_acc, acc, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(got_n, npost)


def test_consolidation_potentiates_recent_inputs_and_depresses_quiet_ones():
    S = STDPParams(a_plus=1.0, a_minus=0.5, offset=0.25)
    acc = np.array([[2.0, 0.5, 0.0],      # neuron 0 fired twice
                    [9.0, 9.0, 9.0],      # neuron 1 never fired
                    [1.0, 0.25, 0.1]])    # neuron 2 fired once
    npost = np.array([2.0, 0.0, 1.0])
    mask, pot, strength, gain = stdp.consolidation(acc, npost, S, learn_cap=12.0)
    np.testing.assert_array_equal(mask, [[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    np.testing.assert_array_equal(pot[[0, 2]], [[1, 1, 0], [1, 1, 0]])
    # mean pre trace 1.0 -> full potentiation; 0.0 -> full (a_minus) depression
    np.testing.assert_allclose(strength[0], [1.0, 0.0, 0.5])
    np.testing.assert_allclose(strength[2], [1.0, 0.0, 0.5 * 0.15 / 0.25])
    np.testing.assert_array_equal(gain.ravel(), [2.0, 0.0, 1.0])


def test_no_depression_when_a_minus_is_zero():
    S = STDPParams(a_minus=0.0)
    mask, pot, _s, _g = stdp.consolidation(
        np.array([[1.0, 0.0]]), np.array([1.0]), S, 12.0)
    np.testing.assert_array_equal(mask, [[True, False]])


def test_pulses_are_signed_scaled_row_major_and_skip_negligible_writes():
    S = STDPParams(pot_amp=2.0, dep_amp=3.0)
    mask = np.array([[True, True], [True, False]])
    pot = np.array([[True, False], [False, True]])
    strength = np.array([[0.5, 1.5], [1e-5, 0.9]])     # 1.5 clips to 1
    rows, cols, amps, p, s = stdp.pulses(mask, pot, strength, 2.0, S, pot_sign=-1.0)
    np.testing.assert_array_equal(rows, [0, 0])        # (1,0) below MIN_PULSE
    np.testing.assert_array_equal(cols, [0, 1])
    np.testing.assert_allclose(s, [1.0, 2.0])
    np.testing.assert_allclose(amps, [-2.0 * 1.0, +3.0 * 2.0])


def test_supervised_presentation_writes_the_target_row_only(make_net):
    # weak feed-forward drive: only the teacher-driven target can fire
    net = make_net(n_out=4, pattern_set="bars",
                   neuron={"ipsp_gain": 0.0, "epsp_gain": 2.0})
    vec = np.zeros(16)
    vec[:4] = 1.0                                   # top bar
    before = net.weights_S()
    res = net.train_step(vec, target=2)
    after = net.weights_S()
    assert res["n_out_spikes"][2] > 0
    silent = [0, 1, 3]
    assert all(res["n_out_spikes"][j] == 0 for j in silent)
    np.testing.assert_array_equal(after[silent], before[silent])
    assert np.all(after[2, :4] > before[2, :4])     # active inputs: LTP
    assert np.all(after[2, 4:] < before[2, 4:])     # quiet inputs: LTD
    writes = {k: v for k, v in net.last_write.items() if k[1] == 2}
    assert all(writes[(0, 2, i)][0] for i in range(4))
    assert not any(writes[(0, 2, i)][0] for i in range(4, 16))


def test_learning_disabled_leaves_devices_untouched(make_net):
    net = make_net()
    before = net.capture_weights()
    net.present(net.patterns[0][1], target=0, learn=False)
    net.infer(net.patterns[0][1])
    np.testing.assert_array_equal(net.weights_S(), before[0])
    assert net.write_count == 0


def test_stdp_learns_receptive_fields_and_classifies_bars(make_net):
    net = make_net(n_out=4, pattern_set="bars")
    for _ in range(5):
        for k, (_lbl, vec) in enumerate(net.patterns):
            net.train_step(vec, net.target_of[k])
    rows = net.classify()
    assert sum(r["winner"] == r["target"] for r in rows) == len(rows)
    Wn = net.weights_norm()
    for k, (_lbl, vec) in enumerate(net.patterns[:4]):
        on, off = Wn[net.target_of[k]][vec > 0.5], Wn[net.target_of[k]][vec < 0.5]
        assert on.mean() > off.mean() + 0.3


def test_surrogate_gradient_matches_finite_differences():
    from neurovat import surrogate
    rng = np.random.default_rng(5)
    S_in = (rng.random((30, 6)) < 0.3).astype(float)
    Wn = [rng.random((5, 6)), rng.random((3, 5))]
    gains, alpha, thr, beta = [11.0 * 1.7, 11.0], 0.95, 1.0, 5.0
    spk, cache = surrogate.forward(S_in, Wn, gains, alpha, thr)
    loss, grads = surrogate.loss_and_grads(cache, Wn, gains, alpha, thr, beta, 1)
    assert [g.shape for g in grads] == [w.shape for w in Wn]
    # While the spike pattern is unchanged the loss is exactly differentiable
    # in the output weights, and equals the surrogate gradient in the limit
    # of an infinitely sharp surrogate. Check that against central differences.
    _l, exact = surrogate.loss_and_grads(cache, Wn, gains, alpha, thr, 1e12, 1)

    def loss_at(W1):
        s, c = surrogate.forward(S_in, [Wn[0], W1], gains, alpha, thr)
        assert np.array_equal(s[-1], spk[-1])
        return surrogate.loss_and_grads(c, [Wn[0], W1], gains, alpha, thr, beta, 1)[0]
    h = 1e-7
    for (j, i) in [(0, 0), (1, 3), (2, 4)]:
        Wp, Wm = Wn[1].copy(), Wn[1].copy()
        Wp[j, i] += h
        Wm[j, i] -= h
        np.testing.assert_allclose((loss_at(Wp) - loss_at(Wm)) / (2 * h),
                                   exact[1][j, i], rtol=1e-4, atol=1e-7)
    assert loss == loss_at(Wn[1])


def test_surrogate_training_reduces_loss(make_net):
    net = make_net(rule="surrogate", n_out=4, pattern_set="bars",
                   stdp={"pot_amp": 0.02, "dep_amp": 0.02, "sg_lr": 0.05})
    history = net.fit(6)
    assert history[-1]["loss"] < history[0]["loss"]
    assert net.eval_accuracy() == 1.0
