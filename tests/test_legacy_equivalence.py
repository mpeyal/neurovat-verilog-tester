"""The vectorized engine reproduces the pre-package per-device engine
(benchmarks/legacy_neuro.py) on the same devices, seed and inputs."""
import json

import numpy as np
import pytest

import legacy_neuro
from neurovat import NetConfig, NeuronParams, STDPParams
from neurovat.devices import twin_factory
from neurovat import CrossbarSNN
from conftest import ToyTwin, STDP_AMPS

CASES = {
    "plain": dict(),
    "hidden": dict(hidden_layers=(7, 5)),
    "noisy": dict(input_noise=0.15, jitter_ms=2.0, bg_rate_hz=8.0,
                  signal_frac=0.7, pattern_ms=40.0),
    "latency": dict(encoding="latency"),
    "temporal": dict(encoding="temporal", jitter_ms=1.0),
    "unsupervised": dict(mode="unsupervised"),
}


def pair(rule, neuron=None, **cfg):
    N = NeuronParams(**(neuron or {}))
    S = STDPParams(**STDP_AMPS)
    c = NetConfig(grid_h=4, grid_w=4, n_out=5, present_ms=60.0, seed=9,
                  learn_rule=rule, pattern_set="digits", **cfg)
    return (legacy_neuro.Trainer(ToyTwin, "voltage", N, S, c),
            CrossbarSNN(twin_factory(ToyTwin), N, S, c))


def same_result(a, b):
    assert a.keys() == b.keys()
    np.testing.assert_allclose(a["v_trace"], b["v_trace"], rtol=1e-12, atol=1e-13)
    for k in ("out_spikes", "in_spikes", "layer_spikes", "winner",
              "n_out_spikes", "layer_sizes", "n_steps"):
        assert a[k] == b[k], k
    if "loss" in a:
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("rule", ["stdp", "surrogate"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_training_matches_the_legacy_engine(rule, case):
    neuron = {"v_noise": 0.03} if case == "noisy" else None
    old, new = pair(rule, neuron, **CASES[case])
    for c in range(old.n_layers):                      # same initial weights
        np.testing.assert_array_equal(old.weights_S(c), new.weights_S(c))
    np.testing.assert_array_equal(old.signal_mask, new.signal_mask)
    np.testing.assert_array_equal(old.inh0, new.inh0)
    assert old.pot_sign == new.pot_sign
    for _epoch in range(2):
        for k, (_lbl, vec) in enumerate(old.patterns):
            same_result(old.train_step(vec, old.target_of[k]),
                        new.train_step(vec, new.target_of[k]))
    for c in range(old.n_layers):
        np.testing.assert_allclose(new.weights_S(c), old.weights_S(c),
                                   rtol=1e-11, atol=0)
    np.testing.assert_array_equal(old.spike_count, new.spike_count)
    assert old._t_dev == pytest.approx(new._t_dev, rel=1e-12)
    assert old.rng.bit_generator.state == new.rng.bit_generator.state
    assert old.last_write.keys() == new.last_write.keys()
    for key, (pot, strength) in old.last_write.items():
        assert new.last_write[key][0] == pot
        assert new.last_write[key][1] == pytest.approx(strength, rel=1e-10)

    same_result(old.infer(old.patterns[0][1]), new.infer(new.patterns[0][1]))
    assert old.evaluate() == new.evaluate()
    assert old.evaluate(max_n=4) == new.evaluate(max_n=4)
    assert old.classify() == new.classify()
    for th_old, th_new in zip(old.theta, new.theta):
        np.testing.assert_allclose(th_new, th_old, rtol=1e-12, atol=1e-300)


def test_checkpoints_are_interchangeable_with_the_legacy_engine():
    old, new = pair("surrogate")
    for k, (_lbl, vec) in enumerate(old.patterns):
        old.train_step(vec, old.target_of[k])
    new.restore_state(json.loads(json.dumps(old.capture_state())))
    np.testing.assert_array_equal(new.weights_S(), old.weights_S())
    fresh, _ = pair("surrogate")
    fresh.restore_state(json.loads(json.dumps(new.capture_state())))
    np.testing.assert_array_equal(fresh.weights_S(), old.weights_S())


def test_gui_trainer_shim_keeps_the_legacy_surface():
    from vatester import neuro
    c = NetConfig(grid_h=4, grid_w=4, n_out=3, learn_rule="surrogate", seed=2)
    tr = neuro.Trainer(ToyTwin, "voltage", NeuronParams(), STDPParams(**STDP_AMPS), c)
    assert isinstance(tr.crossbars[0][0][0], ToyTwin) and tr.syn is tr.crossbars[0]
    assert tr.input_kind == "voltage" and tr.make_device is ToyTwin
    tr.train_step(tr.patterns[0][1], 0)
    G = tr.weights_S()
    assert G[1, 2] == tr.crossbars[0][1][2].G           # weights read the twins
    key = next(iter(tr.last_write))
    assert tr.last_write.get(key) is not None and len(key) == 3
    assert tr.last_write.get((0, 99, 99), (True, 0.0)) == (True, 0.0)
    tr.set_weights(0, np.full((3, 16), 2e-5))
    assert tr.crossbars[0][0][0].G == 2e-5
    for name in ("make_patterns", "confusion", "prf1", "device_weight",
                 "calibrate_pot_sign", "probe_lif", "probe_lif_wave", "fi_curve",
                 "probe_lif_psp", "fi_curve_psp", "NeuronParams", "STDPParams",
                 "NetConfig"):
        assert hasattr(neuro, name)
