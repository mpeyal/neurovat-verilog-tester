"""Gate B: physical-time device synapses.

* the vectorised ECFET v3 array equals the scalar twin adapter to machine
  tolerance on random event sequences (both polarities, idle gaps from
  0.1 ms to minutes, pulses near the rails, both read modes, both timings);
* idle stepping is exact: many short idle steps == one long ``step``;
* a long idle gap is one step (O(1)).
"""
import math

import numpy as np
import pytest

from neurovat.devices import (EcfetV3Array, ScalarTwinArray,
                              vector_twin_factory)
from simulation_core.model_v2 import EcfetV2, V2Params
from simulation_core.model_v3 import EcfetV3, V3Params

RTOL = 1e-11


def make_v3(**kw):
    return lambda: EcfetV3(V3Params.paper(**kw))


def random_sequence(rng, n_events, n_post, n_pre, amp_max=2e-9):
    """(t, rows, cols, amps) events with unique cells per event and gaps
    spanning the three v3 pool time constants and beyond."""
    t, out = 0.0, []
    for _ in range(n_events):
        t += float(rng.choice([1e-4, 2e-3, 0.022, 0.3, 19.0, 300.0]))
        k = int(rng.integers(1, n_post * n_pre + 1))
        sel = rng.choice(n_post * n_pre, k, replace=False)
        amps = rng.choice([-1.0, 1.0], k) * rng.uniform(1e-12, amp_max, k)
        out.append((t, sel // n_pre, sel % n_pre, amps))
    return out


@pytest.mark.parametrize("timing", ["physical", "serial"])
@pytest.mark.parametrize("read", ["full", "nv"])
@pytest.mark.parametrize("A_stdp", [0.0, 6e-6])
def test_vector_v3_equals_scalar_twin(timing, read, A_stdp):
    make = make_v3(A_stdp=A_stdp)
    vec = EcfetV3Array(make().p, 3, 4, timing, read)
    ref = ScalarTwinArray(make, 3, 4, timing, read)
    rng = np.random.default_rng(7)
    worst = 0.0
    for t, rows, cols, amps in random_sequence(rng, 120, 3, 4):
        assert vec.program(rows, cols, amps, 10e-3, t) == pytest.approx(
            ref.program(rows, cols, amps, 10e-3, t))
        pairs = [(vec.conductance().copy(), ref.conductance().copy()),
                 (vec.retained(), ref.retained())]
        pairs.append((vec.read_at(t + 0.004, [1, 3]).copy(),
                      ref.read_at(t + 0.004, [1, 3]).copy()))
        for got, want in pairs:
            worst = max(worst, float(np.max(np.abs(got - want) / want)))
    vec.advance(1e4)
    ref.advance(1e4)
    worst = max(worst, float(np.max(np.abs(vec.conductance()
                                           - ref.conductance())
                                    / ref.conductance())))
    assert worst < RTOL, worst
    if timing == "physical":
        np.testing.assert_allclose(vec.t_dev.reshape(3, 4), ref.t_dev,
                                   rtol=1e-15)


def test_vector_v3_drives_to_both_rails_like_the_twin():
    make = make_v3(A_stdp=0.0)
    vec = EcfetV3Array(make().p, 1, 2)
    ref = ScalarTwinArray(make, 1, 2, "physical", "full")
    rows, cols = np.array([0, 0]), np.array([0, 1])
    t = 0.0
    for _ in range(40):                     # big pulses: many sub-steps
        amps = np.array([3e-9, -3e-9])
        vec.program(rows, cols, amps, 10e-3, t)
        ref.program(rows, cols, amps, 10e-3, t)
        t += 0.05
    np.testing.assert_allclose(vec.retained(), ref.retained(), rtol=RTOL)
    G = vec.retained()[0]
    assert G[0] == pytest.approx(vec.g_max) and G[1] == pytest.approx(vec.g_min)


def test_state_dict_roundtrip_and_set_conductance():
    make = make_v3(A_stdp=0.0)
    vec = EcfetV3Array(make().p, 2, 2)
    ref = ScalarTwinArray(make, 2, 2, "physical", "full")
    G = np.array([[2e-4, 5e-4], [9e-4, 1.3e-4]])
    vec.set_conductance(G)
    ref.set_conductance(G)
    np.testing.assert_allclose(vec.conductance(), ref.conductance(), rtol=RTOL)
    rows, cols = np.array([0, 1]), np.array([1, 0])
    vec.program(rows, cols, np.array([1e-10, -1e-10]), 10e-3, 0.5)
    other = EcfetV3Array(make().p, 2, 2)
    other.load_state_dict(vec.state_dict())
    for a in (vec, other):
        a.program(rows, cols, np.array([-5e-11, 2e-10]), 10e-3, 0.7)
    np.testing.assert_array_equal(other.read_at(3.0), vec.read_at(3.0))


@pytest.mark.parametrize("make", [
    make_v3(A_stdp=6e-6), lambda: EcfetV2(V2Params.paper_fig3())],
    ids=["v3", "v2"])
def test_idle_steps_equal_one_long_step(make):
    """After a write, N short idle steps == one step over their sum."""
    a, b = make(), make()
    for dev in (a, b):
        dev.step(0.0, 10e-3, 80e-12)
        dev.step(10e-3, 5e-3, 0.0)
        dev.step(15e-3, 10e-3, -40e-12)
    gaps = np.random.default_rng(3).uniform(1e-4, 2.0, 400)
    t = 25e-3
    for g in gaps:
        a.step(t, float(g), 0.0)
        t += float(g)
    b.step(25e-3, float(gaps.sum()), 0.0)
    for k, v in a.observables().items():
        assert v == pytest.approx(b.observables()[k], rel=1e-10, abs=1e-18), k
    assert a.G == pytest.approx(b.G, rel=1e-12)


def test_idle_gap_is_one_step_whatever_its_length():
    calls = []

    class Counting(EcfetV3):
        def step(self, t, dt, I):
            calls.append(dt)
            super().step(t, dt, I)

    arr = ScalarTwinArray(lambda: Counting(V3Params.paper(A_stdp=0.0)), 1, 1,
                          "physical", "full")
    arr.program(np.array([0]), np.array([0]), np.array([5e-11]), 10e-3, 0.0)
    calls.clear()
    arr.advance(3600.0 * 24)                 # one day of idle time
    assert calls == [pytest.approx(3600.0 * 24 - 10e-3)]
    # the volatile pools are gone; retention moved x toward x_eq
    d = arr.devices[0][0]
    assert abs(sum(d.pools)) < 1e-12 * d.G_nv
    assert arr.conductance()[0, 0] == pytest.approx(d.G_nv)


def test_full_read_sees_the_volatile_pools_and_nv_read_does_not():
    make = make_v3(A_stdp=0.0)
    full = EcfetV3Array(make().p, 1, 1, "physical", "full")
    nv = EcfetV3Array(make().p, 1, 1, "physical", "nv")
    g0 = float(full.conductance()[0, 0])
    for a in (full, nv):
        a.program(np.array([0]), np.array([0]), np.array([5e-11]), 10e-3, 0.0)
    ret = float(full.retained()[0, 0]) - g0
    early = float(full.read_at(0.010)[0, 0]) - g0
    late = float(full.read_at(200.0)[0, 0])
    p = make().p
    # instantaneous excess = kappa_v / (1 - kappa_v) of the retained step
    assert early / ret == pytest.approx(1.0 + p.kappa_v / (1 - p.kappa_v),
                                        rel=0.02)
    assert late == pytest.approx(float(full.retained()[0, 0]), rel=1e-7)
    assert float(nv.read_at(0.010)[0, 0]) - g0 == pytest.approx(ret, rel=1e-9)


def test_pulses_of_one_call_are_parallel_in_physical_time():
    make = make_v3(A_stdp=0.0)
    arr = ScalarTwinArray(make, 1, 3, "physical", "full")
    t = arr.program(np.array([0, 0, 0]), np.array([0, 1, 2]),
                    np.full(3, 5e-11), 10e-3, 1.0)
    assert t == pytest.approx(1.01)
    np.testing.assert_allclose(arr.t_dev, [[1.01, 1.01, 1.01]])
    # a write that arrives during the pulse starts when the pulse ends
    arr.program(np.array([0]), np.array([0]), np.array([5e-11]), 10e-3, 1.005)
    assert arr.t_dev[0, 0] == pytest.approx(1.02)


def test_factory_vectorises_only_the_canonical_noiseless_v3():
    assert vector_twin_factory(make_v3(A_stdp=0.0)).kind == "vector"
    assert vector_twin_factory(make_v3(sigma_c2c=0.1)).kind == "scalar"
    assert vector_twin_factory(lambda: EcfetV2(V2Params())).kind == "scalar"

    class Edited(EcfetV3):
        pass
    assert vector_twin_factory(lambda: Edited(V3Params.paper())).kind == "scalar"
    arr = vector_twin_factory(make_v3(A_stdp=0.0))(2, 5)
    assert isinstance(arr, EcfetV3Array) and arr.time_dynamic


@pytest.mark.parametrize("read", ["full", "nv"])
def test_peek_reads_without_stepping_and_matches_an_idle_step(read):
    make = make_v3(A_stdp=0.0)
    a = EcfetV3Array(make().p, 2, 3, "physical", read)
    rows, cols = np.array([0, 1, 1]), np.array([0, 2, 1])
    a.program(rows, cols, np.array([2e-10, -1e-10, 5e-11]), 10e-3, 0.3)
    before = a.state_dict()
    for t in (0.305, 0.31, 0.4, 5.0, 900.0):
        got = a.peek(t)
        assert np.array_equal(a.state_dict()["pools"], before["pools"])
        b = EcfetV3Array(make().p, 2, 3, "physical", read)
        b.load_state_dict(before)
        np.testing.assert_allclose(got, b.read_at(t), rtol=1e-13)
