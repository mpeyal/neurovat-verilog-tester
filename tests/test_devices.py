"""The pluggable device interface: every registered model, the scalar-twin
adapter, and a model registered from outside the package."""
import json

import numpy as np
import pytest

import neurovat
from neurovat import devices
from neurovat.devices import DeviceArray, ScalarTwinArray, device_factory
from neurovat.state import pack, unpack
from conftest import ToyTwin, train

MODELS = neurovat.available_devices()


def pulse_amp(arr):
    """A drive amplitude that visibly moves this model."""
    p = arr.p
    return float(getattr(p, "a_ref", 0) or 2 * getattr(p, "Vc", 1.0))


@pytest.mark.parametrize("name", MODELS)
def test_registered_model_honours_the_interface(name):
    arr = device_factory(name)(3, 5, seed=1)
    assert isinstance(arr, DeviceArray) and arr.shape == (3, 5)
    G0 = np.array(arr.conductance())
    assert G0.shape == (3, 5) and arr.g_min < arr.g_max
    assert np.all((G0 >= arr.g_min) & (G0 <= arr.g_max))

    a = pulse_amp(arr)
    sign = arr.potentiation_sign(a, 10e-3)
    rows, cols = np.array([0, 2]), np.array([1, 4])
    t = arr.program(rows, cols, sign * np.array([a, -a]), 10e-3, t=1.0)
    assert t == pytest.approx(1.0 + 2 * 10e-3)
    G1 = np.array(arr.conductance())
    assert G1[0, 1] > G0[0, 1] and G1[2, 4] < G0[2, 4]
    untouched = np.ones((3, 5), bool)
    untouched[rows, cols] = False
    np.testing.assert_array_equal(G1[untouched], G0[untouched])

    for _ in range(200):                           # saturates inside the bounds
        arr.program(rows, cols, sign * np.array([a, -a]), 10e-3)
    G = arr.conductance()
    assert arr.g_min <= G[2, 4] < G[0, 1] <= arr.g_max


@pytest.mark.parametrize("name", MODELS)
def test_state_roundtrips_through_json(name):
    arr = device_factory(name, write_noise=0.1)(3, 5, seed=2)
    rows, cols = np.nonzero(np.ones((3, 5), bool))
    amps = np.linspace(-1, 1, 15) * pulse_amp(arr)
    arr.program(rows, cols, amps, 10e-3)
    state = json.loads(json.dumps(pack(arr.state_dict())))
    twin = device_factory(name, write_noise=0.1)(3, 5, seed=99)
    twin.load_state_dict(unpack(state))
    np.testing.assert_array_equal(twin.conductance(), arr.conductance())
    arr.program(rows, cols, amps, 10e-3)           # same write-noise stream
    twin.program(rows, cols, amps, 10e-3)
    np.testing.assert_array_equal(twin.conductance(), arr.conductance())
    with pytest.raises(ValueError):
        device_factory(name)(2, 5, seed=0).load_state_dict(unpack(state))


@pytest.mark.parametrize("name", MODELS)
def test_set_conductance_roundtrip(name):
    src = device_factory(name)(2, 3, seed=0)       # conductances the model reaches
    rows, cols = np.nonzero(np.ones((2, 3), bool))
    src.program(rows, cols, np.linspace(-1, 1, 6) * pulse_amp(src), 10e-3)
    G = np.array(src.conductance())
    assert np.ptp(G) > 0
    arr = device_factory(name)(2, 3, seed=0)
    arr.set_conductance(G)
    np.testing.assert_allclose(arr.conductance(), G, rtol=1e-8)


@pytest.mark.parametrize("name", MODELS)
def test_variation_and_write_noise_are_seeded(name):
    arr = device_factory(name)(1, 1)
    param = next(k for k in ("tau_w", "Vc") if hasattr(arr.p, k))

    def written(seed, **kw):
        a = device_factory(name, **kw)(4, 6, seed=seed)
        rows, cols = np.nonzero(np.ones((4, 6), bool))
        a.program(rows, cols, np.full(24, 0.7 * pulse_amp(a)), 5e-3)
        return np.array(a.conductance())
    nominal = written(0)
    assert np.ptp(nominal) == 0
    for kw in ({"variation": {param: 0.2}}, {"write_noise": 0.2}):
        assert np.ptp(written(0, **kw)) > 0
        np.testing.assert_array_equal(written(0, **kw), written(0, **kw))
        assert not np.array_equal(written(0, **kw), written(1, **kw))
    with pytest.raises(ValueError):
        device_factory(name, variation={"no_such_parameter": 0.1})(2, 2)


def test_unknown_model_lists_the_available_ones():
    with pytest.raises(KeyError, match="softbound"):
        device_factory("no-such-device")


def test_scalar_twin_adapter_tracks_its_twins():
    arr = ScalarTwinArray(ToyTwin, 2, 3)
    rows, cols, amps = np.array([0, 1]), np.array([2, 0]), np.array([0.5, -0.5])
    assert arr.program(rows, cols, amps, 10e-3, t=0.0) == pytest.approx(20e-3)
    ref = [[ToyTwin() for _ in range(3)] for _ in range(2)]
    ref[0][2].step(0.0, 10e-3, 0.5)
    ref[1][0].step(10e-3, 10e-3, -0.5)
    np.testing.assert_array_equal(arr.conductance(),
                                  [[d.G for d in row] for row in ref])
    assert arr.potentiation_sign(0.5, 10e-3) == 1.0
    assert (arr.g_min, arr.g_max) == (1e-6, 1e-4)


def test_a_new_device_model_plugs_in_without_touching_the_network(make_net):
    class Binary(DeviceArray):
        """Two-state synapse: any positive pulse sets, any negative resets."""
        name = "binary"
        g_min, g_max = 1e-6, 1e-5

        def __init__(self, n_post, n_pre, params=None, variation=None,
                     write_noise=0.0, seed=None):
            super().__init__(n_post, n_pre)
            self.on = np.zeros(self.shape, bool)

        def conductance(self):
            return np.where(self.on, self.g_max, self.g_min)

        def program(self, rows, cols, amps, width, t=0.0):
            self.on[rows, cols] = np.asarray(amps) > 0
            return t + width * len(rows)

        def state_dict(self):
            return {"on": self.on.copy()}

        def load_state_dict(self, state):
            self.on = np.asarray(state["on"], bool).copy()

        def nominal(self):
            return Binary(1, 1)

    devices.register_device("binary", Binary)
    try:
        assert "binary" in neurovat.available_devices()
        net = train(make_net("binary", pattern_set="bars"), 3)
        assert set(np.unique(net.weights_norm())) == {0.0, 1.0}
        assert net.write_count > 0 and len(net.classify()) == len(net.patterns)
        other = make_net("binary", pattern_set="bars")
        other.restore_state(json.loads(json.dumps(net.capture_state())))
        np.testing.assert_array_equal(other.weights_S(), net.weights_S())
    finally:
        devices._REGISTRY.pop("binary")


def test_vectorized_model_matches_its_scalar_twin(make_net):
    """Private device model (not in public builds): the array implementation
    must reproduce the scalar twin it was derived from."""
    sc = pytest.importorskip("simulation_core")
    if sc.FeFET is None or "fefet" not in MODELS:
        pytest.skip("private device model not present")
    amp = sc.FeFETParams().Vc
    for rule in ("stdp", "surrogate"):
        kw = dict(rule=rule, n_out=5, hidden_layers=(6,), jitter_ms=1.0,
                  stdp={"pot_amp": amp, "dep_amp": amp})
        fast = train(make_net("fefet", **kw), 2)
        slow = train(make_net(devices.twin_factory(sc.FeFET), **kw), 2)
        for c in range(2):
            np.testing.assert_allclose(fast.weights_S(c), slow.weights_S(c),
                                       rtol=1e-9, atol=0)
        assert fast.evaluate() == slow.evaluate()
