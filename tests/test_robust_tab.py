"""GUI-free helpers of the Robustness tab (vatester.network_stream_robust)."""
import json
from dataclasses import replace

import pytest

from neurovat.masquelier import MasquelierParams, Session
from vatester import network_stream_robust as rb
from vatester import stream_robust as sr
from vatester import stream_view as sv

NOMINAL = dict(jitter_ms=1.0, spike_deletion=0.0, noise_hz=10.0, bg_hz=10.0,
               pattern_frac=0.5, spacing="even")


@pytest.fixture(scope="module")
def trained():
    bank = sv.image_bank("digits", 10)
    img = sv.build_image_input(bank, "0", "", 10, 5.0, 0.5)
    P = MasquelierParams(n_afferents=10, n_pattern=10, w_init=1.0,
                         pattern_ms=img["pattern_ms"], threshold=1.0, seed=3)
    P = replace(P, threshold=sv.auto_threshold(img, P))
    s = Session(P, patterns=img["patterns"])
    s.run(2.0, **NOMINAL)
    return s


def test_parse_levels_display_units_and_errors():
    assert rb.parse_levels("", "jitter_ms") is None
    assert rb.parse_levels("  ", "spike_deletion") is None
    assert rb.parse_levels("1, 3,5", "jitter") == [1.0, 3.0, 5.0]
    assert rb.parse_levels("0 10% 30;30", "spike_deletion") == [0.0, 0.1, 0.3]
    assert rb.parse_levels("26,5", "noise_hz") == [26.0, 5.0]
    with pytest.raises(ValueError, match="not a number"):
        rb.parse_levels("1, x", "jitter_ms")
    with pytest.raises(ValueError, match=">= 0"):
        rb.parse_levels("-1", "noise_hz")
    with pytest.raises(ValueError, match="below 100"):
        rb.parse_levels("100", "spike_deletion")
    assert rb.format_levels([0.0, 0.1, 0.5], "spike_deletion") == "0, 10, 50"


def test_stream_changes_match_the_worker():
    cfg = {"share": 0.4, "jitter": 1.0, "noise": 3.0, "delete": 0.1,
           "spacing": "even", "bg": 10.0, "image": None}
    ch = rb.stream_changes(cfg)
    assert "bg_hz" not in ch                   # Poisson: bg stays the default
    assert ch == {"pattern_frac": 0.4, "jitter_ms": 1.0, "noise_hz": 3.0,
                  "spike_deletion": 0.1, "spacing": "even"}
    cfg["image"] = {"bg": 10.0}
    assert rb.stream_changes(cfg)["bg_hz"] == 10.0
    assert set(rb.stream_changes(cfg)) <= set(sr.STREAM_SETTINGS)


def test_merge_of_per_level_sweeps_is_the_single_sweep(trained):
    factors = {"jitter_ms": [1.0, 3.0], "noise_hz": [5.0]}
    one = sr.sweep(trained, factors, seconds=1.5, nominal=NOMINAL)
    acc = None
    for f, levels in factors.items():
        for v in levels:
            acc = rb.merge(acc, sr.sweep(trained, {f: [v]}, seconds=1.5,
                                         nominal=NOMINAL))
    assert [b["factor"] for b in acc["factors"]] == ["jitter_ms", "noise_hz"]
    assert not acc["stopped"] and acc["wall_s"] > 0
    for b1, b2 in zip(one["factors"], acc["factors"]):
        assert len(b1["rows"]) == len(b2["rows"])
        for r1, r2 in zip(b1["rows"], b2["rows"]):
            assert r1["level"] == r2["level"]
            assert r1["nominal"] == r2["nominal"]
            for who in ("trained", "untrained"):
                for k in ("hits", "pattern_windows", "false_windows",
                          "other_windows", "post_spikes"):
                    assert r1[who][k] == r2[who][k]
    assert sr.format_table(acc).startswith("robustness sweep")


def test_compact_series_entries_are_plain_and_drop_missing(trained):
    res = sr.sweep(trained, {"spike_deletion": [0.0, 0.3]}, seconds=1.0,
                   nominal=NOMINAL, untrained=False)
    res["factors"][0]["rows"][1]["trained"]["hit_pct"] = float("nan")
    c = rb.compact(res)
    json.dumps(c, allow_nan=False)             # JSON-safe, NaN -> None
    row = c["factors"][0]["rows"][1]
    assert row["trained"]["hit_pct"] is None and row["untrained"] is None
    assert row["shown"] == pytest.approx(30.0)
    assert "engine_windows" not in row["trained"]
    s = rb.series(c["factors"][0])
    assert s["trained_hit"][0] == [0.0]        # the NaN point is dropped
    assert s["trained_false"][0] == pytest.approx([0.0, 30.0])
    assert s["untrained_hit"] == ([], [])
    e = rb.entries(res)
    assert e[0] == ("missing spikes 0 % (nominal)", "spike_deletion", 0.0)
    assert e[1][0] == "missing spikes 30 %"
    lines = rb.summary_lines(res)
    assert len(lines) == 2 and "untrained -" in lines[0]
    text = rb.table_text(res, width=80)
    assert all(len(x) <= 80 or "|" in x or " " not in x.strip()
               for x in text.splitlines())
    assert text.split() == sr.format_table(res).split()   # nothing lost
