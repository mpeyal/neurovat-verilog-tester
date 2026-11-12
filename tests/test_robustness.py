"""Robustness sweep of a stream network (vatester.stream_robust)."""
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from neurovat.devices import device_factory
from neurovat.masquelier import MasquelierParams, Session
from vatester import stream_robust as sr
from vatester import stream_view as sv

NOMINAL = dict(jitter_ms=1.0, spike_deletion=0.0, noise_hz=10.0, bg_hz=10.0,
               pattern_frac=0.5, spacing="even")


def _digest(w):
    return hashlib.sha256(np.ascontiguousarray(w, float).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def trained():
    bank = sv.image_bank("digits", 14)
    img = sv.build_image_input(bank, "0", "", 14, 10.0, 0.5)
    P = MasquelierParams(n_afferents=14, n_pattern=14, w_init=1.0,
                         pattern_ms=img["pattern_ms"], threshold=1.0, seed=2)
    P = replace(P, threshold=sv.auto_threshold(img, P))
    s = Session(P, patterns=img["patterns"])
    s.run(3.0, **NOMINAL)
    return s


def test_default_levels_hold_the_figure_and_the_nominal():
    assert sr.default_levels("jitter", 1.0) == [1.0, 3.0, 5.0]
    assert sr.default_levels("jitter_ms", 2.0) == [1.0, 2.0, 3.0, 5.0]
    assert sr.default_levels("missing", 0.0) == [0.0, 0.1, 0.3, 0.5]
    assert sr.default_levels("noise_hz", 10.0) == [5.0, 10.0, 26.0]
    with pytest.raises(ValueError, match="unknown sweep factor"):
        sr.default_levels("speed", 1.0)


def test_nominal_level_is_a_plain_keyed_test_and_nothing_changes(trained):
    s = trained
    w0, runs, stream = _digest(s.weights), s.runs, s.stream
    seen = []
    res = sr.sweep(s, {"noise_hz": [10.0, 200.0], "spike_deletion": [0.9],
                       "jitter_ms": [1.0, 20.0]},
                   seconds=2.0, nominal=NOMINAL,
                   on_progress=lambda i, n, t: seen.append((i, n)))
    assert _digest(s.weights) == w0 and s.runs == runs and s.stream is stream
    assert seen[0] == (0, 10) and seen[-1] == (10, 10)
    nom = res["factors"][0]["rows"][0]
    assert nom["nominal"] and nom["level"] == 10.0
    plain = s.run(2.0, learn=False, stream_key=sr.SWEEP_KEY, **NOMINAL)
    assert nom["trained"]["post_spikes"] == plain["post_spikes"]
    assert nom["trained"]["engine_windows"][0] == plain["per_neuron"][0]["windows"]
    assert nom["trained"]["pattern_windows"] > 0
    # the untrained copy runs at the start weights on the same stream
    u = sr.fresh_like(s)
    base = u.run(2.0, learn=False, stream_key=sr.SWEEP_KEY, **NOMINAL)
    assert nom["untrained"]["post_spikes"] == base["post_spikes"]
    assert np.all(u.weights == s.P.w_init)
    # an extreme level really changes the result (an ignored level would not)
    loud = res["factors"][0]["rows"][1]["trained"]
    gone = res["factors"][1]["rows"][0]["trained"]
    assert loud["false_windows"] > nom["trained"]["false_windows"]
    assert gone["hits"] < nom["trained"]["hits"]
    jit = res["factors"][2]["rows"]
    assert jit[0]["trained"] == {**nom["trained"], "wall_s": jit[0]["trained"]["wall_s"]}
    assert jit[1]["trained"]["post_spikes"] != nom["trained"]["post_spikes"]
    # pressing it again gives identical numbers
    again = sr.sweep(s, {"noise_hz": [200.0]}, seconds=2.0, nominal=NOMINAL)
    r = again["factors"][0]["rows"][0]
    for k in ("trained", "untrained"):
        for m in ("hits", "false_windows", "post_spikes", "engine_windows"):
            assert r[k][m] == res["factors"][0]["rows"][1][k][m]
    assert "extra spikes" in sr.format_table(res)


def test_noise_levels_are_absolute_and_the_hit_rule_follows_the_input(trained):
    res = sr.sweep(trained, {"noise_hz": [0.0]}, seconds=2.0,
                   nominal=dict(NOMINAL, noise_hz=50.0), untrained=False)
    plain = trained.run(2.0, learn=False, stream_key=sr.SWEEP_KEY,
                        **dict(NOMINAL, noise_hz=0.0))
    row = res["factors"][0]["rows"][0]
    assert row["trained"]["post_spikes"] == plain["post_spikes"]
    assert row["untrained"] is None and not row["nominal"]
    assert res["rule"] == sv.hit_rule(True)
    with pytest.raises(ValueError, match="not a stream setting"):
        sr.sweep(trained, nominal={"threshold": 1.0})


def test_stop_returns_the_levels_finished_so_far(trained):
    calls = {"n": 0}

    def progress(i, n, text):
        calls["n"] = i

    res = sr.sweep(trained, {"jitter_ms": [1.0, 3.0, 5.0]}, seconds=1.0,
                   nominal=NOMINAL, untrained=False, on_progress=progress,
                   stop=lambda: calls["n"] >= 1)
    assert res["stopped"]
    assert [r["level"] for r in res["factors"][0]["rows"]] == [1.0]
    assert "STOPPED" in sr.format_table(res)


def test_poisson_session_untrained_copy_shares_its_patterns():
    P = MasquelierParams(n_afferents=60, n_pattern=30, threshold=30.0,
                         n_out=2, seed=5)
    s = Session(P, n_patterns=2)
    s.run(2.0, pattern_frac=0.4)
    u = sr.fresh_like(s)
    np.testing.assert_array_equal(u.members, s.members)
    for (a1, t1), (a2, t2) in zip(u.copies, s.copies):
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(t1, t2)
    res = sr.sweep(s, {"jitter_ms": [3.0]}, seconds=1.0,
                   nominal={"pattern_frac": 0.4})
    assert res["rule"] == sv.hit_rule(False) and not res["images"]
    assert res["factors"][0]["rows"][0]["untrained"] is not None


def test_device_session_is_not_changed_by_a_sweep():
    from neurovat.power import EnergyMeter
    bank = sv.image_bank("bars", 8)
    img = sv.build_image_input(bank, "0", "", 8, 5.0, 0.5)
    P = MasquelierParams(n_afferents=8, n_pattern=8, threshold=1.0,
                         pattern_ms=img["pattern_ms"], seed=1)
    P = replace(P, threshold=sv.auto_threshold(img, P))
    dev = device_factory("softbound")(1, 8)
    meter = EnergyMeter([(1, 8)], ["voltage"], None, 1e-3)
    s = Session(P, device=dev, patterns=img["patterns"], energy=meter)
    s.run(2.0)
    state = dev.state_dict()
    G = dev.conductance().copy()
    before = repr(meter.report("eval"))
    res = sr.sweep(s, {"noise_hz": [10.0, 40.0]}, seconds=1.0)
    assert res["untrained_note"].startswith("skipped")
    assert res["factors"][0]["rows"][0]["untrained"] is None
    np.testing.assert_array_equal(dev.state_dict()["w"], state["w"])
    np.testing.assert_array_equal(dev.conductance(), G)
    assert repr(meter.report("eval")) == before   # sweep reads not metered
    assert s.kw["energy"] is meter                 # and the meter is restored
