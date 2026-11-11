"""Masquelier et al. (2008) paradigm: input statistics, the alternating
nearest-spike rule, and pattern finding on a shortened run."""
import numpy as np

from neurovat.masquelier import MasquelierParams, make_stream, run, _pattern_sections


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
