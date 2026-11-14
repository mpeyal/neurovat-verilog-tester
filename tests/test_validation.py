"""Fast checks of the Gate A validation harness (validation/)."""

import re

import numpy as np
import pytest

from simulation_core import simulate
from validation import common, compare, scrub_spectre, stimuli
from validation.spice import make_netlists


# ---- stimulus set --------------------------------------------------------
def test_stimulus_set_is_deterministic():
    a, b = stimuli.build_stimuli(), stimuli.build_stimuli()
    assert [s.as_dict() for s in a] == [s.as_dict() for s in b]
    assert stimuli.fingerprint(a) == stimuli.fingerprint(b) == stimuli.fingerprint()
    names = [s.name for s in a]
    assert len(names) == len(set(names)) == 8
    assert "random_spike_train" in names and "stdp_pair_sweep" in names


def test_random_train_is_seeded_and_mixed_sign():
    st = stimuli.BY_NAME["random_spike_train"]
    signs = {np.sign(p[2]) for p in st.pulses}
    assert signs == {-1.0, 1.0}
    assert st.pulses == stimuli.build_stimuli()[-1].pulses


@pytest.mark.parametrize("st", stimuli.STIMULI, ids=lambda s: s.name)
def test_pwl_round_trips_to_the_same_waveform(st):
    pts = stimuli.parse_spectre_pwl(st.spectre_pwl())
    ref = st.pwl_points()
    assert len(pts) == len(ref)
    np.testing.assert_allclose(np.array(pts), np.array(ref), rtol=1e-11, atol=1e-18)
    # plateau values match the twin waveform away from the 1 us edges
    edges = st.waveform().edges
    probes = [0.5 * (a + b) for a, b in zip([0.0] + edges, edges + [st.t_stop])]
    np.testing.assert_allclose(stimuli.pwl_eval(pts, probes), st.drive(probes),
                               rtol=0, atol=1e-20)
    # and the PWL carries exactly the same charge as the rectangular pulses
    t, i = map(np.array, zip(*pts))
    q_pwl = np.sum(0.5 * (i[1:] + i[:-1]) * np.diff(t))
    q_rect = sum(w * a for _, w, a in st.pulses)
    assert q_pwl == pytest.approx(q_rect, rel=1e-9, abs=1e-24)


def test_twin_loop_matches_simulate_exactly():
    st = stimuli.BY_NAME["stdp_pair_sweep"]
    for key in common.MODELS:
        cols, _ = common.run_twin(key, st)
        res = simulate(common.new_model(key), st.waveform(), st.t_stop,
                       dt_min=stimuli.DT_MIN_S, dt_max=st.dt_max)
        np.testing.assert_array_equal(cols["t_s"], res.t)
        np.testing.assert_array_equal(cols["G_S"], res.G)
        np.testing.assert_array_equal(cols["drive_A"], res.i_gate)


def test_refined_grid_contains_base_points():
    ts = np.array([0.0, 0.1, 0.3, 1.0])
    fine = common.refine_grid(ts, 4)
    assert len(fine) == 13
    np.testing.assert_array_equal(fine[::4], ts)


# ---- metrics -------------------------------------------------------------
def test_metrics_identical_is_zero():
    g = np.linspace(1e-4, 2e-4, 101)
    m = compare.metrics(g, g)
    assert m["nrmse"] == 0.0 and m["maxerr"] == 0.0
    assert compare.verdict(m) == "PASS"


def test_metrics_known_offset():
    g = np.linspace(1e-4, 2e-4, 101)          # range 1e-4 S
    m = compare.metrics(g + 0.5e-6, g)        # constant 0.5 % of range
    assert m["nrmse"] == pytest.approx(0.005)
    assert m["maxerr"] == pytest.approx(0.005)
    assert compare.verdict(m) == "PASS"
    m = compare.metrics(g + 2e-6, g)          # 2 % -> fails the NRMSE rule
    assert m["nrmse"] == pytest.approx(0.02)
    assert compare.verdict(m) == "FAIL"


def test_metrics_sine_error_nrmse():
    t = np.linspace(0, 1, 100001)
    g = 1e-4 + 1e-4 * t
    m = compare.metrics(g + 1e-6 * np.sin(2 * np.pi * 7 * t), g)
    assert m["nrmse"] == pytest.approx(0.01 / np.sqrt(2), rel=1e-3)
    assert m["maxerr"] == pytest.approx(0.01, rel=1e-3)


def test_flat_reference_is_not_a_pass():
    m = compare.metrics(np.ones(10), np.ones(10))
    assert compare.verdict(m) == "FAIL"


def test_drive_check_catches_reversed_source():
    st = stimuli.BY_NAME["single_pulse"]
    twin, _ = common.run_twin("v3", st)
    t, i = map(np.array, zip(*st.pwl_points()))
    ref = {"t_s": t, "drive_A": i}
    assert compare.drive_check(twin, ref) == "ok"
    assert compare.drive_check(twin, {"t_s": t, "drive_A": -i}) == "sign flipped"
    assert compare.drive_check(twin, {"t_s": t, "drive_A": 1.1 * i}).startswith(
        "charge mismatch")


# ---- Spectre import ------------------------------------------------------
def _fake_psf(key, st, host_header=True):
    cols, _ = common.run_twin(key, st)
    t = cols["t_s"]
    head = ['"PSFversion" "1.00"', '"simulator" "spectre"',
            '"version" "21.1.0.389.isr12"']
    if host_header:
        head += ['"date" "2:15:03 PM, Tue Nov 11, 2025"',
                 '"design" "/home/jdoe/sim/single_pulse.scs"',
                 '"host" "cad07.engr.example.edu"', '"user" "jdoe"']
    state = "X1.n_nv" if key == "v2" else "X1.n_x"
    state_v = cols["G_ret_S"] if key == "v2" else cols["x_Li"]
    traces = ["n_rmem", state, "X1.n_v1", "X1.n_v2", "X1.n_v3", "X1.n_stdp",
              "vread:p", "X1:IGsc"]
    vol = cols["G_vol_S"]
    vals = [cols["R_ohm"], state_v, vol, 0 * vol, 0 * vol, 0 * vol,
            -cols["i_read_A"], cols["drive_A"]]
    out = ["HEADER", *head, "TYPE", '"V" FLOAT DOUBLE PROP( "units" "V" )',
           "SWEEP", '"time" "sweep" PROP( "units" "s" )', "TRACE",
           *[f'"{n}" "V"' for n in traces], "VALUE"]
    for k in range(len(t)):
        out.append(f'"time" {t[k]:.15e}')
        out += [f'"{n}" {v[k]:.15e}' for n, v in zip(traces, vals)]
    out.append("END")
    return "\n".join(out) + "\n", cols


@pytest.mark.parametrize("key", list(common.MODELS))
def test_scrub_strips_host_user_header(tmp_path, key):
    st = stimuli.BY_NAME["single_pulse"]
    text, twin = _fake_psf(key, st)
    src = tmp_path / "tran.tran"
    src.write_text(text)
    log = tmp_path / "x.log"
    log.write_text("Total time required for tran analysis `tran': "
                   "CPU = 1.5 s, elapsed = 2.25 s.\nuser jdoe on cad07\n")
    dest = scrub_spectre.scrub_one(key, st.name, src, log=log, out_root=tmp_path)
    body = dest.read_text()
    head = "\n".join(body.splitlines()[:2])
    for bad in ("jdoe", "cad07", "example.edu", "/home", "Nov", "2025", "2:15"):
        assert bad not in head
    assert all(scrub_spectre._ROW_OK.match(r) for r in body.splitlines()[2:])
    first = body.splitlines()[0]
    assert first.startswith("# source: Spectre 21.1; model: " + key)
    assert body.splitlines()[1] == ",".join(common.COLUMNS)
    meta = common.read_meta(dest.with_suffix(".json"))
    assert meta["wall_s"] == 2.25 and meta["drive_measured"] is True
    assert "jdoe" not in dest.with_suffix(".json").read_text()
    ref, _ = common.read_csv(dest)
    m = compare.metrics(twin["G_S"], ref["G_S"])
    assert m["nrmse"] < 1e-6          # only the 10-digit CSV rounding remains
    assert compare.drive_check(twin, ref) == "ok"


def test_scrub_refuses_leaky_line():
    lines = scrub_spectre.render("v3", {c: np.zeros(2) for c in common.COLUMNS},
                                 {"version": "21.1"})
    scrub_spectre.check_clean(lines)                     # neutral -> accepted
    for leak in ("# source: Spectre on cad07.example.edu; model: v3; x",
                 "# source: Spectre 21.1 jdoe@cad07; model: v3"):
        with pytest.raises(scrub_spectre.ScrubError):
            scrub_spectre.check_clean([leak] + lines[1:])
    with pytest.raises(scrub_spectre.ScrubError):
        scrub_spectre.check_clean(lines + ["/home/jdoe/run,1,2"])
    with pytest.raises(scrub_spectre.ScrubError):     # a known local identifier
        scrub_spectre.check_clean(
            [lines[0].replace("model: v3", "model: v3 cad07")] + lines[1:],
            extra_ids=["cad07"])


def test_missing_drive_is_reported_not_substituted(tmp_path):
    st = stimuli.BY_NAME["single_pulse"]
    text, twin = _fake_psf("v3", st, host_header=False)
    lines = [ln for ln in text.splitlines() if "X1:IGsc" not in ln]
    src = tmp_path / "tran.tran"
    src.write_text("\n".join(lines) + "\n")
    dest = scrub_spectre.scrub_one("v3", st.name, src, out_root=tmp_path)
    ref, _ = common.read_csv(dest)
    assert np.all(np.isnan(ref["drive_A"]))
    assert compare.drive_check(twin, ref) == "drive not measured"
    assert common.read_meta(dest.with_suffix(".json"))["drive_measured"] is False


def test_psf_value_lines_with_type_strings(tmp_path):
    st = stimuli.BY_NAME["single_pulse"]
    text, twin = _fake_psf("v3", st, host_header=False)
    head, _, values = text.partition("VALUE\n")
    typed = [re.sub(r'^("[^"]+") ', r'\1 "V" ', ln) if ln.startswith('"') and
             not ln.startswith('"time"') else ln for ln in values.splitlines()]
    src = tmp_path / "tran.tran"
    src.write_text(head + "VALUE\n" + "\n".join(typed) + "\n")
    ref, _ = common.read_csv(scrub_spectre.scrub_one("v3", st.name, src,
                                                     out_root=tmp_path))
    assert compare.metrics(twin["G_S"], ref["G_S"])["nrmse"] < 1e-6


def test_csv_export_import(tmp_path):
    st = stimuli.BY_NAME["single_pulse"]
    twin, _ = common.run_twin("v3", st)
    names = ["time (s)", "V(n_rmem)", "V(X1.n_x)", "V(X1.n_v1)", "V(X1.n_v2)",
             "V(X1.n_v3)", "V(X1.n_stdp)", "I(vread:p)", "I(X1:IGsc)"]
    z = 0 * twin["t_s"]
    cols = [twin["t_s"], twin["R_ohm"], twin["x_Li"], twin["G_vol_S"], z, z,
            twin["stdp_S"], -twin["i_read_A"], twin["drive_A"]]
    rows = [",".join(names)] + [",".join(f"{c[k]:.15e}" for c in cols)
                                for k in range(len(z))]
    src = tmp_path / "export.csv"
    src.write_text("\n".join(rows) + "\n")
    ref, _ = common.read_csv(scrub_spectre.scrub_one("v3", st.name, src,
                                                     out_root=tmp_path))
    assert compare.metrics(twin["G_S"], ref["G_S"])["nrmse"] < 1e-6
    np.testing.assert_allclose(ref["x_Li"], twin["x_Li"], rtol=1e-9)
    assert compare.drive_check(twin, ref) == "ok"


def test_numeric_rows_are_not_flagged():
    lines = scrub_spectre.render("v2", {c: np.array([1.0e-3, -2.5e-12, np.nan])
                                        for c in common.COLUMNS}, {})
    scrub_spectre.check_clean(lines)


# ---- Spectre decks -------------------------------------------------------
def test_decks_are_generated_with_matching_pwl(tmp_path):
    make_netlists.main(["--out", str(tmp_path)])
    for key in common.MODELS:
        for st in stimuli.STIMULI:
            text = (tmp_path / key / f"{st.name}.scs").read_text()
            assert f'ahdl_include "' in text and common.MODELS[key][2] in text
            assert "igate (0 n_igsc) isource type=pwl" in text
            assert "X1:IGsc" in text and "n_rmem" in text
            wave = text[text.index("wave=["):text.index("]", text.index("wave=[")) + 1]
            np.testing.assert_allclose(np.array(stimuli.parse_spectre_pwl(wave)),
                                       np.array(st.pwl_points()), rtol=1e-11,
                                       atol=1e-18)
    assert (tmp_path / "run_all.sh").is_file()
    make_netlists.main(["--out", str(tmp_path / "flip"), "--flip-gate"])
    text = (tmp_path / "flip" / "v3" / "single_pulse.scs").read_text()
    assert "igate (n_igsc 0) isource type=pwl" in text
