"""stanford-1 validation set: stimulus fingerprint, PWL round trip, decks,
Spectre importer (leak check) and scoring."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation import stimuli as gate_a  # noqa: E402
from validation.stanford import core, make_decks, scrub  # noqa: E402
from validation.stanford import stimuli as S  # noqa: E402

PINNED = "158bdb6aa16bc5f0"      # bump STIMULUS_SET_VERSION + this when the set changes


def test_fingerprint_pinned_and_rebuild_stable():
    assert S.STIMULUS_SET_VERSION == "stanford-1"
    assert S.fingerprint(S.build_stimuli()) == S.fingerprint()
    assert S.fingerprint()[:16] == PINNED


def test_gate_a_set_untouched():
    assert gate_a.STIMULUS_SET_VERSION == "gateA-1"
    assert not hasattr(gate_a.Stimulus, "v")


def test_fingerprint_changes_with_any_point():
    st = S.STIMULI[0]
    pts = list(st.points)
    pts[1] = (pts[1][0], pts[1][1] + 1e-6)
    alt = [S.VoltageStimulus(st.name, st.description, tuple(pts), st.t_stop_ps, st.dt_max,
                             st.reads, st.instance, st.params)] + S.STIMULI[1:]
    assert S.fingerprint(alt) != S.fingerprint()


@pytest.mark.parametrize("st", S.STIMULI, ids=lambda s: s.name)
def test_pwl_round_trip_and_reads(st):
    back = S.parse_spectre_pwl(st.spectre_pwl())
    assert np.allclose(back, st.pwl_points(), rtol=1e-12, atol=0)
    assert all(b > a for (a, _), (b, _) in zip(st.points, st.points[1:]))
    if len(st.reads):
        assert np.allclose(st.v(st.read_times), S.V_READ)
    ts = core.grid(st)
    assert set(np.round(st.t_points / 1e-15)) <= set(np.round(ts / 1e-15))
    assert np.max(np.diff(ts)) <= st.dt_max * (1 + 1e-9)


def test_pulse_area_is_amp_times_width():
    b = S._Builder()
    b.wait(1000)
    b.pulse(-1.3, 10 * S.NS)
    st = b.build("x", "x", 1000, 1e-10)
    t = np.linspace(0, st.t_stop, 200001)
    assert np.trapezoid(st.v(t), t) == pytest.approx(-1.3 * 10e-9, rel=1e-6)


def test_decks(tmp_path):
    make_decks.main(["--out", str(tmp_path)])
    for st in S.STIMULI:
        txt = (tmp_path / f"{st.name}.scs").read_text(encoding="utf-8")
        assert 'ahdl_include "' in txt and core.VA_NAME in txt
        assert f"X1 (n_te 0) {core.MODULE}" in txt
        assert "q=1.602176462e-19" in txt and "kb=1.3806503e-23" in txt
        assert f"maxstep={st.dt_max:g}" in txt and f"time_step={st.dt_max:.12g}" in txt
        assert "save n_te X1:TE vte:p" in txt
        for k, v in st.instance:
            assert f"{k}={v:.12g}" in txt
        assert np.allclose(S.parse_spectre_pwl(txt[txt.index("wave=["):]), st.pwl_points())
    assert (tmp_path / ".gitignore").read_text().startswith("# Spectre work files")
    assert "pulse_reset_ref.scs" in (tmp_path / "run_first.sh").read_text()


def test_patch_source_only_fixes_syntax():
    src = ('/* Copyright @ 2014 Stanford University (Stanford) */\n'
           '`include "constants.vams"`include "disciplines.vams"\n'
           'parameter integer model_switch = 0 from[0, 1];\n'
           'parameter real T_crit = 450 from(390, 460);\n'
           'parameter real deltaGap0 = 0.02 from[0, 0.1);\n'
           'parameter real g0 = 0.25e-9 from(0:2e-9);\n')
    out = make_decks.patch_source(src)
    assert "Copyright @ 2014 Stanford University (Stanford)" in out
    assert 'constants.vams"\n`include "disciplines.vams"' in out
    assert "from[0:1]" in out and "from(390:460)" in out and "from[0:0.1)" in out
    assert "from(0:2e-9)" in out and ", " not in out.split("separate lines.")[1].replace(
        "(Stanford)", "")


def _fake_spectre_csv(path, cols, flip=False):
    sgn = -1.0 if flip else 1.0
    lines = ["time,n_te,X1:TE"]
    for t, v, i in zip(cols["t_s"], cols["V_TE_V"], cols["I_TE_A"]):
        lines.append(f"{t:.15g},{v:.15g},{sgn * i:.15g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_scrub_and_score_round_trip(tmp_path):
    st = S.BY_NAME["s1_pm1p3"]
    tw, _, _ = core.run_twin(st)
    src = tmp_path / "s1.csv"
    _fake_spectre_csv(src, tw)
    dest = scrub.scrub_one(st.name, src, out_root=tmp_path)
    text = dest.read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("# source: Spectre; model: stanford_pku; stimulus_set: stanford-1 ")
    from validation import common
    sp, _ = common.read_csv(dest)
    assert np.all(np.isnan(sp["gap_state_m"]))
    ok = np.isfinite(sp["gap_m"])
    assert np.allclose(sp["gap_m"][ok], tw["gap_m"][ok], rtol=0, atol=1e-15)
    row = core.score(tw, sp, st)
    assert row["n_reads"] == 20 and row["g_read_max_pct"] < 1e-6
    assert core.verdict(row) == "PASS"


def test_scrub_refuses_flipped_current(tmp_path):
    st = S.BY_NAME["reset_train"]
    tw, _, _ = core.run_twin(st)
    src = tmp_path / "r.csv"
    _fake_spectre_csv(src, tw, flip=True)
    with pytest.raises(scrub.ScrubError, match="wrong sign"):
        scrub.scrub_one(st.name, src, out_root=tmp_path)


def test_leak_check_refuses_identifiers():
    good = scrub.render({c: np.zeros(2) for c in core.COLUMNS}, {"version": "21.1.0"})
    scrub.check_clean(good)
    for bad in ("# source: Spectre 21.1; model: stanford_pku; stimulus_set: x at someone@host",
                "# source: Spectre /home/x; model: stanford_pku; stimulus_set: stanford-1 0123456789abcdef"):
        with pytest.raises(scrub.ScrubError):
            scrub.check_clean([bad] + good[1:])
    with pytest.raises(scrub.ScrubError):
        scrub.check_clean(good[:2] + ["1,2,host"])
    with pytest.raises(scrub.ScrubError):
        scrub.check_clean(good, extra_ids=["stanford_pku"])


def test_score_detects_a_real_difference():
    st = S.BY_NAME["reset_train"]
    tw, _, _ = core.run_twin(st)
    other, _, _ = core.run_twin(st, Vel0=11.0)
    row = core.score(other, tw, st)
    assert core.verdict(row) == "FAIL" and row["g_read_max_pct"] > 3
    assert not math.isnan(row["gap_nrmse_pct"])
