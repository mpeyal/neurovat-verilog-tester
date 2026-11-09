"""Seeded reproducibility: runs, checkpoints, evaluation and experiments."""
import json

import numpy as np
import pytest

from neurovat.devices import device_factory
from conftest import train

NOISY = dict(input_noise=0.1, jitter_ms=2.0, bg_rate_hz=5.0)


@pytest.mark.parametrize("rule", ["stdp", "surrogate"])
def test_same_seed_gives_identical_training(make_net, rule):
    a = train(make_net(rule=rule, seed=11, **NOISY), 2)
    b = train(make_net(rule=rule, seed=11, **NOISY), 2)
    np.testing.assert_array_equal(a.weights_S(), b.weights_S())
    assert json.dumps(a.capture_state()) == json.dumps(b.capture_state())
    c = train(make_net(rule=rule, seed=12, **NOISY), 2)
    assert not np.array_equal(a.weights_S(), c.weights_S())


@pytest.mark.parametrize("rule", ["stdp", "surrogate"])
def test_fit_history_is_reproducible(make_net, rule):
    def run():
        net = make_net(rule=rule, seed=5, **NOISY)
        hist = net.fit(3)
        return [{k: v for k, v in h.items() if k != "train_s"} for h in hist], net
    (ha, a), (hb, b) = run(), run()
    assert ha == hb
    np.testing.assert_array_equal(a.weights_S(), b.weights_S())


@pytest.mark.parametrize("rule", ["stdp", "surrogate"])
def test_checkpoint_resume_equals_uninterrupted_run(make_net, rule):
    kw = dict(rule=rule, seed=7, neuron={"v_noise": 0.02}, **NOISY)
    straight = train(make_net(**kw), 2)
    first = train(make_net(**kw), 1)
    state = json.loads(json.dumps(first.capture_state()))      # via JSON
    resumed = make_net(**kw)
    resumed.restore_state(state)
    train(resumed, 1)
    np.testing.assert_array_equal(resumed.weights_S(), straight.weights_S())
    assert json.dumps(resumed.capture_state()) == json.dumps(straight.capture_state())


@pytest.mark.parametrize("rule", ["stdp", "surrogate"])
def test_evaluation_is_deterministic_and_leaves_state_alone(make_net, rule):
    net = train(make_net(rule=rule, neuron={"v_noise": 0.05}, **NOISY))
    before = json.dumps(net.capture_state())
    clean = net.evaluate()
    noisy = net.evaluate(noisy=True)
    assert net.evaluate() == clean and net.evaluate(noisy=True) == noisy
    assert json.dumps(net.capture_state()) == before


def test_noisy_evaluation_uses_the_configured_front_end(make_net):
    net = train(make_net(rule="surrogate", n_out=4), 3)
    net.cfg.input_noise, net.cfg.jitter_ms = 0.0, 0.0
    quiet = net.eval_accuracy(noisy=True)
    net.cfg.input_noise, net.cfg.jitter_ms = 1.0, 20.0
    assert net.eval_accuracy(noisy=True) < quiet
    assert net.eval_accuracy() == net.eval_accuracy()         # clean: unaffected


def test_batched_scoring_equals_per_sample_scoring(make_net):
    net = train(make_net(rule="surrogate", hidden_layers=(6,), **NOISY))
    vecs = [v for _l, v in net.patterns]
    for noisy in (False, True):
        batched = net.predict(vecs, noisy=noisy, batch=3)
        net.infer = net.infer                     # instance hook -> per-sample path
        assert net.predict(vecs, noisy=noisy) == batched
        del net.infer


def test_device_variation_has_its_own_random_stream(make_net):
    plain = make_net(seed=4, signal_frac=0.5)
    varied = make_net(device_factory("softbound", variation={"tau_w": 0.3, "g_max": 0.1}),
                      seed=4, signal_frac=0.5)
    assert plain.rng.bit_generator.state == varied.rng.bit_generator.state
    np.testing.assert_array_equal(plain.signal_mask, varied.signal_mask)
    again = make_net(device_factory("softbound", variation={"tau_w": 0.3, "g_max": 0.1}),
                     seed=4, signal_frac=0.5)
    np.testing.assert_array_equal(varied.weights_S(), again.weights_S())
    assert np.ptp(varied.weights_S()) > 0 and np.ptp(plain.weights_S()) == 0


def test_validation_split_is_seeded_and_disjoint(make_net):
    rng = np.random.default_rng(0)
    pats = [(f"s{k}", rng.random(16)) for k in range(20)]
    targets = [k % 2 for k in range(20)]

    def split():
        net = make_net(n_out=2)
        net.patterns, net.target_of = list(pats), list(targets)
        net.split_validation(0.3)
        return net
    a, b = split(), split()
    held = [l for l, _ in a.validation_patterns]
    assert held == [l for l, _ in b.validation_patterns]
    assert len(held) == 6 and sorted(a.validation_targets) == [0] * 3 + [1] * 3
    assert len(a.patterns) + len(held) == 20
    assert not {l for l, _ in a.patterns} & set(held)
