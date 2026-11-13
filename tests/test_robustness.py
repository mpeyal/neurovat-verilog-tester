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


# ---- retrain mode -------------------------------------------------------

@pytest.fixture(scope="module")
def low_start():
    """Start weight 0.475: STDP potentiates there, so what a copy learns
    depends on the input it is trained under (at 1.0 it only prunes)."""
    bank = sv.image_bank("digits", 14)
    img = sv.build_image_input(bank, "0", "", 14, 10.0, 0.5)
    P = MasquelierParams(n_afferents=14, n_pattern=14, w_init=0.475,
                         pattern_ms=img["pattern_ms"], threshold=1.0, seed=2)
    P = replace(P, threshold=sv.auto_threshold(img, P))
    s = Session(P, patterns=img["patterns"])
    s.run(3.0, **NOMINAL)
    return s


COUNTS = ("hits", "pattern_windows", "false_windows", "other_windows",
          "post_spikes")


def _same(a, b):
    return all(a[k] == b[k] for k in COUNTS)


def test_retrain_is_a_keyed_training_of_a_fresh_copy(low_start):
    s = low_start
    w0, runs, stream = _digest(s.weights), s.runs, s.stream
    levels = {"spike_deletion": [0.0, 0.5], "noise_hz": [200.0]}
    seen = []
    res = sr.sweep(s, levels, seconds=2.0, nominal=NOMINAL, retrain=True,
                   train_seconds=3.0,
                   on_progress=lambda i, n, t: seen.append((i, n, t)))
    # the user's session is not run
    assert _digest(s.weights) == w0 and s.runs == runs and s.stream is stream
    assert res["mode"] == sr.RETRAIN and res["repeats"] == 1
    assert res["train_seconds"] == 3.0 and not res["stopped"]
    assert seen[0][:2] == (0, 9) and seen[-1][:2] == (9, 9)
    assert any(t.startswith("training...") for _i, _n, t in seen)
    assert any(t.startswith("testing...") for _i, _n, t in seen)
    nom, gone = res["factors"][0]["rows"]
    loud = res["factors"][1]["rows"][0]
    # each row is exactly: fresh copy, keyed training at that level, test
    for row, f in ((gone, "spike_deletion"), (loud, "noise_hz")):
        ch = dict(NOMINAL, **{f: row["level"]})
        c = sr.fresh_like(s)
        c.run(3.0, learn=True, stream_key=(sr.TRAIN_KEY, 0), **ch)
        ref = sr.measure(c, 2.0, ch)
        assert _same(row["trained"], ref) and _same(row["repeats"][0], ref)
        np.testing.assert_array_equal(row["weights"][0], c.weights)
        assert row["trained"]["hit_min"] == row["trained"]["hit_max"]
    # the training level matters: the trained weights differ by level and the
    # result differs from recall (an ignored retrain flag / level fails here)
    assert _digest(gone["weights"][0]) != _digest(nom["weights"][0])
    assert _digest(loud["weights"][0]) != _digest(nom["weights"][0])
    rec = sr.sweep(s, levels, seconds=2.0, nominal=NOMINAL)
    assert rec["mode"] == sr.RECALL and "weights" not in rec["factors"][0]["rows"][0]
    r_gone = rec["factors"][0]["rows"][1]
    r_loud = rec["factors"][1]["rows"][0]
    assert gone["trained"]["hits"] != r_gone["trained"]["hits"]
    assert loud["trained"]["post_spikes"] != r_loud["trained"]["post_spikes"]
    # the untrained comparison is the recall one
    for a, b in ((gone, r_gone), (loud, r_loud)):
        assert _same(a["untrained"], b["untrained"])
    # nominal row: an independent training, labelled as such
    assert nom["nominal"] and nom["trained"]["pattern_windows"] > 0
    table = sr.format_table(res)
    assert sr.RETRAIN_NOMINAL in table and "RETRAINED" in table
    assert "one training run per level" in table
    # a repeat sweep is identical
    again = sr.sweep(s, {"spike_deletion": [0.5]}, seconds=2.0,
                     nominal=NOMINAL, retrain=True, train_seconds=3.0,
                     untrained=False)
    row = again["factors"][0]["rows"][0]
    assert _same(row["trained"], gone["trained"])
    np.testing.assert_array_equal(row["weights"][0], gone["weights"][0])
    assert _digest(s.weights) == w0 and s.runs == runs and s.stream is stream


def test_retrain_repeats_give_the_mean_and_the_min_max(low_start):
    s = low_start
    one = sr.sweep(s, {"noise_hz": [40.0]}, seconds=2.0, nominal=NOMINAL,
                   retrain=True, train_seconds=2.0, untrained=False)
    two = sr.sweep(s, {"noise_hz": [40.0]}, seconds=2.0, nominal=NOMINAL,
                   retrain=True, train_seconds=2.0, untrained=False,
                   repeats=2)
    r1, r2 = one["factors"][0]["rows"][0], two["factors"][0]["rows"][0]
    t = r2["trained"]
    assert two["repeats"] == 2 and t["n_runs"] == 2
    assert len(r2["repeats"]) == len(r2["weights"]) == 2
    assert _same(r2["repeats"][0], r1["trained"])       # repeat 0 = (key, 0)
    assert _digest(r2["weights"][0]) != _digest(r2["weights"][1])
    hits = [x["hit_pct"] for x in r2["repeats"]]
    falses = [x["false_pct"] for x in r2["repeats"]]
    assert t["hit_pct"] == pytest.approx(np.mean(hits))
    assert (t["hit_min"], t["hit_max"]) == (min(hits), max(hits))
    assert (t["false_min"], t["false_max"]) == (min(falses), max(falses))
    assert t["hits"] == sum(x["hits"] for x in r2["repeats"])
    assert t["pattern_windows"] == 2 * r1["trained"]["pattern_windows"]
    table = sr.format_table(two)
    assert "2 training runs per level" in table and "hit %" in table


def test_retrain_stop_during_training_drops_that_level(low_start):
    trainings = {"n": 0}

    def progress(_i, _n, text):
        if text.startswith("training"):
            trainings["n"] += 1
    res = sr.sweep(low_start, {"jitter_ms": [1.0, 3.0, 5.0]}, seconds=1.0,
                   nominal=NOMINAL, retrain=True, train_seconds=2.0,
                   untrained=False, on_progress=progress,
                   stop=lambda: trainings["n"] >= 2)
    assert res["stopped"] and trainings["n"] == 2
    assert [r["level"] for r in res["factors"][0]["rows"]] == [1.0]
    assert "STOPPED" in sr.format_table(res)


def test_retrain_refuses_a_device_and_needs_train_seconds(low_start):
    bank = sv.image_bank("bars", 8)
    img = sv.build_image_input(bank, "0", "", 8, 5.0, 0.5)
    P = MasquelierParams(n_afferents=8, n_pattern=8, threshold=1.0,
                         pattern_ms=img["pattern_ms"], seed=1)
    dev = device_factory("softbound")(1, 8)
    s = Session(P, device=dev, patterns=img["patterns"])
    with pytest.raises(ValueError, match="retraining needs ideal weights: a "
                       "device copy cannot be made"):
        sr.sweep(s, {"noise_hz": [5.0]}, seconds=1.0, retrain=True,
                 train_seconds=1.0)
    with pytest.raises(ValueError, match="train_seconds"):
        sr.sweep(low_start, {"noise_hz": [5.0]}, seconds=1.0, retrain=True)
    with pytest.raises(ValueError, match="repeats"):
        sr.sweep(low_start, {"noise_hz": [5.0]}, seconds=1.0, retrain=True,
                 train_seconds=1.0, repeats=0)
