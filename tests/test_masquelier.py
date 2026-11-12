"""Masquelier et al. (2008) paradigm: input statistics, the alternating
nearest-spike rule, and pattern finding on a shortened run."""
from dataclasses import replace
from pathlib import Path

import numpy as np

import pytest

from neurovat.devices import device_factory
from neurovat.encoding import scan_events
from neurovat.masquelier import (MasquelierParams, Session, calibrate_write,
                                 event_members, make_event_stream,
                                 make_pattern_stream, make_stream,
                                 pattern_peak, pulse_step, random_members, run,
                                 _pattern_sections, _sections)
from neurovat.patterns import make_patterns


def test_input_matches_the_paper_statistics():
    P = MasquelierParams(base_s=10.0)
    S = make_stream(P, np.random.default_rng(3))
    assert 60.0 < S.rate_hz < 68.0                  # paper: 64 Hz on average
    on = S.pattern_on
    assert 0.18 < on.mean() < 0.32                  # a quarter of the sections
    assert not (on[1:] & on[:-1]).any()             # never two in a row
    assert on[S.source_section]                     # no hidden extra copy


def test_pattern_sections_hit_the_requested_share():
    on = _pattern_sections(200000, 0.25, np.random.default_rng(0))
    assert abs(on.mean() - 0.25) < 0.01
    assert not (on[1:] & on[:-1]).any()


def test_the_pattern_repeats_and_the_other_afferents_do_not():
    P = MasquelierParams(base_s=5.0, jitter_ms=0.0, noise_hz=0.0)
    S = make_stream(P, np.random.default_rng(5))
    k1, k2 = np.nonzero(S.pattern_on)[0][:2]

    def section(k, lo, hi):
        m = ((S.t // P.pattern_ms) == k) & (S.aff >= lo) & (S.aff < hi)
        return sorted(zip(S.aff[m], np.round(S.t[m] - k * P.pattern_ms, 6)))

    assert section(k1, 0, P.n_pattern) == section(k2, 0, P.n_pattern)
    assert section(k1, P.n_pattern, P.n_afferents) != section(k2, P.n_pattern, P.n_afferents)


def test_stdp_finds_the_pattern_within_thirty_seconds():
    r = run(MasquelierParams(base_s=30.0, n_repeats=1, seed=1))
    t, lat = r["post_t_ms"], r["post_latency_ms"]
    late = t > 20e3
    assert late.sum() > 10                          # still firing
    assert not np.isnan(lat[late]).any()            # only inside the pattern
    assert np.nanmean(lat[late]) < 10.0             # near its start
    assert r["weights"][:1000].mean() > r["weights"][1000:].mean()


def test_several_patterns_share_the_stream():
    P = MasquelierParams(base_s=10.0, pattern_frac=0.4)
    members = random_members(3, P, np.random.default_rng(0))
    S = make_pattern_stream(members, P, np.random.default_rng(4))
    assert members.sum(1).tolist() == [P.n_pattern] * 3
    assert 60.0 < S.rate_hz < 68.0                  # still no rate cue
    assert 0.32 < S.pattern_on.mean() < 0.48
    assert set(S.label[S.pattern_on]) == {0, 1, 2}
    assert all(S.label[k] == p for p, k in enumerate(S.sources))


def test_spike_deletion_thins_the_pattern_only():
    members = np.zeros((1, 400), bool)
    members[0, :200] = True
    kw = dict(n_afferents=400, n_pattern=200, base_s=5.0, noise_hz=0.0)
    full = make_pattern_stream(members, MasquelierParams(**kw),
                               np.random.default_rng(2))
    thin = make_pattern_stream(members, MasquelierParams(spike_deletion=0.5, **kw),
                               np.random.default_rng(2))

    def count(S, in_pattern, lo, hi):
        sec = (S.t // 50.0).astype(int)
        m = (S.pattern_on[sec] == in_pattern) & (S.aff >= lo) & (S.aff < hi)
        return int(m.sum())

    assert count(thin, True, 0, 200) < 0.65 * count(full, True, 0, 200)
    assert count(thin, True, 200, 400) == count(full, True, 200, 400)


def test_inhibition_gives_each_pattern_its_own_neuron():
    P = MasquelierParams(n_out=2, n_afferents=1000, n_pattern=500,
                         threshold=250.0, pattern_frac=0.4, base_s=30.0,
                         n_repeats=1, seed=1)
    r = run(P, members=random_members(2, P, np.random.default_rng(1)))
    assert sorted(r["pattern_owner"]) == [0, 1]                 # both learnt
    assert len(set(r["pattern_owner"].values())) == 2           # by different neurons
    assert r["false_alarms_per_s"] < 1.0
    for p, j in r["pattern_owner"].items():                     # and only its own
        assert r["spikes_per_window"][j, 1 - p] < 0.05


def test_a_session_tests_on_a_new_stream_with_frozen_weights():
    P = MasquelierParams(n_afferents=1000, n_pattern=500, threshold=250.0,
                         seed=1)
    s = Session(P)
    s.run(25.0)
    w, pattern = s.weights.copy(), s.copies
    r = s.run(8.0, learn=False, jitter_ms=2.0)
    assert np.array_equal(s.weights, w)             # nothing was learnt
    assert s.copies[0][0] is pattern[0][0]          # the same hidden pattern
    hit = r["spikes_per_window"][0]
    assert hit[0] > 0.9 and hit[1] < 0.05           # found in unseen input


def test_an_untrained_first_run_freezes_the_patterns_that_training_then_uses():
    """The GUI's untrained baseline: Test before Train. The Poisson patterns
    are frozen by the first run whether it learns or not, the weights stay
    at w_init, and a later Train works on the same patterns."""
    P = MasquelierParams(n_afferents=200, n_pattern=100, threshold=50.0,
                         n_out=2, seed=3)
    s = Session(P, n_patterns=2)
    s.run(1.0, learn=False)
    frozen = s.copies
    assert len(frozen) == 2 and all(len(a) for a, _t in frozen)
    assert np.all(s.weights == P.w_init)            # nothing learnt yet
    s.run(1.0)
    assert not np.all(s.weights == P.w_init)        # training changed them
    for (a1, t1), (a2, t2) in zip(frozen, s.copies):
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(t1, t2)


def test_on_section_reports_each_section_and_can_stop_the_run():
    seen = []

    def on_section(t_ms, info):
        seen.append((t_ms, info["label"]))
        return len(seen) < 20

    P = MasquelierParams(n_afferents=400, n_pattern=200, threshold=100.0)
    r = Session(P).run(10.0, on_section=on_section)
    assert len(seen) == 20
    assert abs(r["simulated_s"] - 1.0) < 1e-6       # 20 sections of 50 ms
    assert {lab for _, lab in seen} <= {-1, 0}


def test_calibrated_pulses_make_one_stdp_step():
    P = MasquelierParams()
    device = device_factory("softbound")(1, 4)
    cal = calibrate_write(device, P, 1e-3)
    a_minus = P.a_minus_ratio * P.a_plus
    assert cal["pot_step"] == pytest.approx(P.a_plus, rel=0.03)
    assert cal["dep_step"] == pytest.approx(a_minus, rel=0.03)
    up = pulse_step(device, cal["pot_sign"] * cal["write_gain"] * P.a_plus,
                    1e-3, P.w_init)
    down = pulse_step(device, -cal["pot_sign"] * cal["dep_gain"] * a_minus,
                      1e-3, P.w_init)
    assert up == pytest.approx(P.a_plus, rel=0.03)
    assert down == pytest.approx(-a_minus, rel=0.03)


def test_calibration_skips_a_direction_that_is_switched_off():
    P = MasquelierParams(a_minus_ratio=0.0)
    cal = calibrate_write(device_factory("softbound")(1, 1), P, 1e-3)
    assert cal["dep_gain"] == 0.0 and cal["dep_amp"] == 0.0
    assert cal["pot_step"] == pytest.approx(P.a_plus, rel=0.03)


def test_calibration_reports_a_step_that_is_out_of_reach():
    P = MasquelierParams(a_plus=0.9)        # softbound cannot jump that far
    with pytest.raises(ValueError, match="cannot make a step"):
        calibrate_write(device_factory("softbound")(1, 1), P, 1e-3)


# --- given patterns (make_event_stream) -------------------------------------

EVENTS = (np.array([0, 3, 5, 5, 9]), np.array([0.0, 4.2, 7.5, 12.0, 19.9]))


def _ev(**kw):
    kw = {"n_afferents": 10, "n_pattern": 5, "pattern_ms": 20.0, "base_s": 1.0,
          "jitter_ms": 0.0, "noise_hz": 0.0, "bg_hz": 0.0,
          "spike_deletion": 0.0, **kw}
    return MasquelierParams(**kw)


def _in_section(S, P, k):
    """Index mask of the spikes in section k."""
    return (S.t // P.pattern_ms).astype(int) == k


def test_event_stream_pastes_the_given_events_and_nothing_else():
    P = _ev()
    S = make_event_stream([EVENTS], P, np.random.default_rng(0))
    assert S.label.size == 50 and S.pattern_on.any()
    want = sorted(zip(EVENTS[0].tolist(), EVENTS[1].tolist()))
    for k in range(S.label.size):
        m = _in_section(S, P, k)
        if S.label[k] == 0:
            got = sorted(zip(S.aff[m].tolist(),
                             np.round(S.t[m] - k * P.pattern_ms, 9).tolist()))
            assert got == want
            assert (S.kind[m] == 1).all()
        else:
            assert not m.any()
    assert S.kind.size == S.aff.size
    assert (S.extra == -1).all()


@pytest.mark.parametrize("frac, k", [(0.5, 2), (0.25, 4)])
def test_even_spacing_puts_a_pattern_in_every_kth_section(frac, k):
    P = _ev(spacing="even", pattern_frac=frac)
    want = np.arange(k - 1, 50, k)
    S = make_event_stream([EVENTS], P, np.random.default_rng(0))
    assert np.nonzero(S.pattern_on)[0].tolist() == want.tolist()
    members = np.zeros((1, 10), bool)
    members[0, :5] = True
    S = make_pattern_stream(members, P, np.random.default_rng(0))
    assert np.nonzero(S.pattern_on)[0].tolist() == want.tolist()


def test_unknown_spacing_is_rejected():
    P = _ev(spacing="regular")
    with pytest.raises(ValueError, match="spacing"):
        _sections(10, P, np.random.default_rng(0))
    with pytest.raises(ValueError, match="spacing"):
        make_event_stream([EVENTS], P, np.random.default_rng(0))


def test_background_fills_only_the_empty_sections_at_the_requested_rate():
    P = _ev(base_s=4.0, bg_hz=50.0)
    S = make_event_stream([EVENTS], P, np.random.default_rng(1))
    on = S.pattern_on[(S.t // P.pattern_ms).astype(int)]
    assert (S.kind[~on] == 0).all() and (S.kind[on] == 1).all()
    n_off = int((~S.pattern_on).sum())
    rate = (~on).sum() / (P.n_afferents * n_off * P.pattern_ms * 1e-3)
    assert 40.0 < rate < 60.0
    S = make_event_stream([EVENTS], replace(P, bg_hz=0.0, noise_hz=50.0),
                          np.random.default_rng(1))
    on = S.pattern_on[(S.t // P.pattern_ms).astype(int)]
    assert (S.kind[on] == 0).any()                  # noise inside the patterns


def test_distractors_fill_the_other_sections_unlabelled():
    P = _ev()
    dis = [(np.array([1, 2]), np.array([3.0, 6.0])),
           (np.array([4, 8, 8]), np.array([1.0, 2.0, 15.0]))]
    S = make_event_stream([EVENTS], P, np.random.default_rng(2), dis)
    assert (S.extra[S.pattern_on] == -1).all()
    off = np.nonzero(~S.pattern_on)[0]
    assert (S.extra[off] >= 0).all() and (S.label[off] == -1).all()
    assert set(S.extra[off].tolist()) == {0, 1}
    for k in off:
        m = _in_section(S, P, k)
        d_aff, d_t = dis[S.extra[k]]
        assert (S.kind[m] == 2).all()
        assert sorted(zip(S.aff[m].tolist(),
                          np.round(S.t[m] - k * P.pattern_ms, 9).tolist())) == \
            sorted(zip(d_aff.tolist(), d_t.tolist()))


def test_deletion_thins_each_presentation_and_jitter_stays_in_its_section():
    n = 200
    events = (np.arange(n), np.linspace(0.0, 19.99, n))
    P = _ev(n_afferents=n, n_pattern=n, base_s=2.0, spike_deletion=0.5)
    S = make_event_stream([events], P, np.random.default_rng(3))
    counts = [int((_in_section(S, P, k) & (S.kind == 1)).sum())
              for k in np.nonzero(S.pattern_on)[0]]
    assert all(60 < c < 140 for c in counts)
    assert abs(np.mean(counts) - n / 2) < 10
    S = make_event_stream([events], replace(P, spike_deletion=0.0, jitter_ms=3.0),
                          np.random.default_rng(3))
    for k in np.nonzero(S.pattern_on)[0]:
        m = _in_section(S, P, k)
        assert m.sum() == n and (S.kind[m] == 1).all()
    assert (S.kind == 1).sum() == n * S.pattern_on.sum()


def test_event_stream_rejects_events_that_do_not_fit():
    P = _ev()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="afferent"):
        make_event_stream([(np.array([10]), np.array([1.0]))], P, rng)
    with pytest.raises(ValueError, match="lasts"):
        make_event_stream([(np.array([1]), np.array([20.0]))], P, rng)
    with pytest.raises(ValueError, match="multiple"):
        make_event_stream([EVENTS], replace(P, pattern_ms=20.5), rng)


def test_pattern_stream_tags_pattern_spikes():
    P = MasquelierParams(n_afferents=40, n_pattern=20, base_s=2.0)
    members = np.zeros((1, 40), bool)
    members[0, :20] = True
    S = make_pattern_stream(members, P, np.random.default_rng(4))
    assert S.kind.size == S.aff.size
    size = S.copies[0][0].size
    assert size > 0
    for k in range(S.label.size):
        m = _in_section(S, P, k)
        if S.pattern_on[k]:
            assert (m & (S.kind == 1)).sum() == size
        else:
            assert (S.kind[m] == 0).all()
    assert (S.extra == -1).all()


def test_event_members_marks_each_patterns_afferents():
    m = event_members([EVENTS, (np.array([1]), np.array([0.0]))], 10)
    assert m.shape == (2, 10)
    assert np.nonzero(m[0])[0].tolist() == [0, 3, 5, 9]
    assert np.nonzero(m[1])[0].tolist() == [1]


def test_pattern_peak_is_w_init_per_simultaneous_spike():
    P = MasquelierParams()
    one = (np.array([0]), np.array([3.0]))
    assert pattern_peak(one, P) == pytest.approx(P.w_init, abs=1e-3)
    five = (np.arange(5), np.full(5, 3.0))
    assert pattern_peak(five, P) == pytest.approx(5 * P.w_init, abs=5e-3)
    assert pattern_peak((np.zeros(0, int), np.zeros(0)), P) == 0.0


# --- recording and window-level scoring -------------------------------------

def _small(**kw):
    kw = {"n_afferents": 50, "n_pattern": 25, "base_s": 2.0, "n_repeats": 1,
          "threshold": 25.0, "n_out": 2, **kw}
    P = MasquelierParams(**kw)
    members = np.zeros((1, P.n_afferents), bool)
    members[0, :P.n_pattern] = True
    return P, make_pattern_stream(members, P, np.random.default_rng(6))


def test_recording_the_potential_changes_nothing_and_crosses_at_each_spike():
    P, S = _small()
    seen = []

    def on_section(t_ms, info):
        seen.append((info["stream"], info["u_ms"]))

    plain = run(P, stream=S)
    rec = run(P, stream=S, record_u=True, on_section=on_section)
    assert "u_ms" not in plain
    assert plain["post_t_ms"].size > 0
    np.testing.assert_array_equal(plain["post_t_ms"], rec["post_t_ms"])
    np.testing.assert_array_equal(plain["post_neuron"], rec["post_neuron"])
    np.testing.assert_array_equal(plain["weights"], rec["weights"])
    u = rec["u_ms"]
    assert u.shape == (int(round(S.duration_ms)), P.n_out)
    for t, j in zip(rec["post_t_ms"], rec["post_neuron"]):
        ms = (int(round(t / P.sim_dt_ms)) - 1) // int(round(1.0 / P.sim_dt_ms))
        assert u[ms, j] >= P.threshold
    assert len(seen) == S.label.size
    assert all(s is S and isinstance(a, np.ndarray) and a.shape[1] == P.n_out
               for s, a in seen)


def test_on_section_has_no_recording_unless_asked():
    P, S = _small(base_s=0.2)
    seen = []
    run(P, stream=S, on_section=lambda t, info: seen.append(info["u_ms"]))
    assert seen and all(a is None for a in seen)


def test_window_counts_cover_every_scored_section():
    P, S = _small(base_s=3.0)
    r = run(P, stream=S)
    t_done = r["simulated_s"] * 1000.0
    t0 = t_done - r["score_window_s"] * 1000.0
    n_win = int(np.floor(t_done / P.pattern_ms + 1e-9)
                - np.ceil(t0 / P.pattern_ms - 1e-9))
    assert n_win == 20
    for q in r["per_neuron"]:
        w = q["windows"]
        assert w["tp"] + w["fn"] + w["fp"] + w["tn"] == n_win
        assert w["accuracy"] == pytest.approx((w["tp"] + w["tn"]) / n_win)


def test_a_silent_neuron_scores_half_balanced_accuracy():
    P, S = _small(base_s=3.0, threshold=1e9)
    r = run(P, stream=S)
    assert r["post_spikes"] == 0
    for q in r["per_neuron"]:
        w = q["windows"]
        assert w["tp"] == 0 and w["fp"] == 0 and w["fn"] > 0
        assert w["balanced_accuracy"] == pytest.approx(0.5)


# --- sessions on given patterns ---------------------------------------------

def test_a_trained_neuron_detects_a_column_scanned_digit():
    # detection only: without noise and with the auto threshold (0.85 x the
    # peak at the start weight) the built-in digit is detected before any
    # training (see the learning test below for what STDP adds)
    img = np.asarray(make_patterns("digits", 14, 14)[0][1]).reshape(14, 14)
    events = scan_events(img, 10.0)
    P = MasquelierParams(n_afferents=14, n_pattern=14, pattern_ms=140.0,
                         n_out=1, pattern_frac=0.5, spacing="even",
                         bg_hz=10.0, noise_hz=0.0)
    P = replace(P, threshold=0.85 * pattern_peak(events, P))
    s = Session(P, patterns=[events])
    s.run(20.0)
    w = s.run(10.0, learn=False)["per_neuron"][0]["windows"]
    assert w["tp"] >= 0.9 * (w["tp"] + w["fn"])
    assert w["fp"] <= 0.2 * (w["fp"] + w["tn"])


MNIST = Path(__file__).resolve().parents[1] / "datasets" / "mnist.npz"


@pytest.mark.skipif(not MNIST.exists(), reason="datasets/mnist.npz not present")
def test_stdp_prunes_the_false_answers_to_an_mnist_digit():
    """Learning, not only detection: the GUI flow on one network - untrained
    baseline Test, Train, Test. The fresh neuron also answers empty windows
    (10 Hz background + 10 Hz noise); at start weight 1 STDP can only lower
    weights, and training keeps the hits and removes most false answers.
    Without training the false answers stay (checked: this test then fails
    on the 'after' assertion). No built-in image showed this
    (their peaks sit far above the noise, so they are clean untrained)."""
    with np.load(MNIST, allow_pickle=False) as d:
        x, y = d["x_train"], d["y_train"]
    img = np.asarray(x[int(np.nonzero(y == 0)[0][0])], float)
    img = (img - img.min()) / (img.max() - img.min())          # the first '0'
    events = scan_events(img, 10.0, 0.5)
    P = MasquelierParams(n_afferents=28, n_pattern=28, pattern_ms=280.0,
                         n_out=1, pattern_frac=0.5, spacing="even",
                         bg_hz=10.0, noise_hz=10.0, w_init=1.0, seed=2)
    P = replace(P, threshold=0.85 * pattern_peak(events, P))
    s = Session(P, patterns=[events])
    before = s.run(15.0, learn=False)["per_neuron"][0]["windows"]
    assert np.all(s.weights == 1.0)                 # the baseline learns nothing
    s.run(40.0)
    after = s.run(15.0, learn=False)["per_neuron"][0]["windows"]
    for w in (before, after):                       # hits held
        assert w["tp"] >= 0.9 * (w["tp"] + w["fn"])
    assert before["fp"] >= 0.3 * (before["fp"] + before["tn"])
    assert after["fp"] <= 0.15 * (after["fp"] + after["tn"])
    assert after["fp"] < before["fp"]


def test_a_session_hides_the_given_patterns_in_each_new_stream():
    P = _ev(noise_hz=20.0, threshold=5.0)
    with pytest.raises(ValueError):
        Session(P, patterns=[])
    s = Session(P, patterns=[EVENTS])
    assert s.stream is None
    seen = []
    s.run(1.0)
    first = s.stream
    s.run(0.6, on_section=lambda t, info: seen.append(info["stream"]))
    second = s.stream
    assert second is not first and all(x is second for x in seen)
    assert second.label.size == 30                  # the last run's stream
    assert len(first.copies) == len(second.copies) == 1
    for (a1, t1), (a2, t2) in zip(first.copies, second.copies):
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(second.copies[0][0], EVENTS[0])
    n1, n2 = first.t[first.kind == 0], second.t[second.kind == 0]
    assert n1.size and n2.size
    assert not np.array_equal(n1[:min(n1.size, n2.size)], n2[:min(n1.size, n2.size)])


# --- keyed streams (Session.run(stream_key=...)) -----------------------------

def _keyed_session():
    P = MasquelierParams(n_afferents=40, n_pattern=20, threshold=20.0,
                         pattern_frac=0.4, seed=4)
    return Session(P)


def test_a_keyed_run_is_repeatable_and_leaves_the_session_as_it_was():
    a, b = _keyed_session(), _keyed_session()
    for s in (a, b):
        s.run(2.0)
    runs, w, stream = a.runs, a.weights.copy(), a.stream
    k1 = a.run(1.0, learn=False, stream_key=5, jitter_ms=2.0)
    k2 = a.run(1.0, learn=False, stream_key=5, jitter_ms=2.0)
    np.testing.assert_array_equal(k1["post_t_ms"], k2["post_t_ms"])
    assert k1["input_rate_hz"] == k2["input_rate_hz"]
    assert a.runs == runs and a.stream is stream
    np.testing.assert_array_equal(a.weights, w)
    other = a.run(1.0, learn=False, stream_key=(5, 1), jitter_ms=2.0)
    assert other["input_rate_hz"] != k1["input_rate_hz"]
    # the next ordinary run is the one it would have been without the probes
    ra, rb = a.run(1.0, learn=False), b.run(1.0, learn=False)
    np.testing.assert_array_equal(ra["post_t_ms"], rb["post_t_ms"])
    assert ra["input_rate_hz"] == rb["input_rate_hz"] and a.runs == b.runs


def test_a_keyed_stream_is_not_the_stream_of_a_numbered_run():
    def stream_of(**kw):
        seen = {}

        def grab(_t, info):
            seen["S"] = info["stream"]
        _keyed_session().run(1.0, learn=False, on_section=grab, **kw)
        return seen["S"]
    numbered, keyed = stream_of(), stream_of(stream_key=0)
    assert numbered.t.size != keyed.t.size or not np.array_equal(
        numbered.t, keyed.t)
    again = stream_of()                  # the default stream is unchanged
    np.testing.assert_array_equal(again.t, numbered.t)


@pytest.mark.parametrize("given", [False, True])
def test_keyed_runs_keep_the_background_whatever_the_deletion_and_jitter(given):
    """Only the pattern presentations change between sweep levels: the
    sections, background and noise of a keyed stream stay the same."""
    P = MasquelierParams(n_afferents=40, n_pattern=20, threshold=20.0,
                         pattern_frac=0.4, seed=4, bg_hz=5.0, pattern_ms=50.0)
    ev = (np.arange(20, dtype=np.int32), np.linspace(2.0, 40.0, 20))

    def stream(**kw):
        s = Session(P, patterns=[ev]) if given else Session(P)
        seen = {}

        def grab(_t, info):
            seen["S"] = info["stream"]
        s.run(1.0, learn=False, stream_key=3, on_section=grab, **kw)
        return seen["S"]
    ref = stream()
    for kw in ({"spike_deletion": 0.5}, {"jitter_ms": 4.0}):
        S = stream(**kw)
        np.testing.assert_array_equal(S.label, ref.label)
        np.testing.assert_array_equal(S.t[S.kind == 0], ref.t[ref.kind == 0])
        np.testing.assert_array_equal(S.aff[S.kind == 0], ref.aff[ref.kind == 0])
        assert not np.array_equal(S.t[S.kind == 1], ref.t[ref.kind == 1])
