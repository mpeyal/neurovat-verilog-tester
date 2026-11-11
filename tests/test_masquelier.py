"""Masquelier et al. (2008) paradigm: input statistics, the alternating
nearest-spike rule, and pattern finding on a shortened run."""
import numpy as np

from neurovat.masquelier import (MasquelierParams, Session,
                                 make_pattern_stream, make_stream,
                                 random_members, run, _pattern_sections)


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
