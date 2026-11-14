"""Checks of the sharbati-1 paper overlay (validation/paper/)."""

import json
import math

import numpy as np
import pytest

from validation import common, compare, stimuli as gate_a
from validation.paper import core, make_decks_paper, scrub_paper, score
from validation.paper import metrics as M
from validation.paper import sharbati_stimuli as S


# ---- stimulus set --------------------------------------------------------
def test_stimulus_set_is_deterministic_and_separate():
    a, b = S.build_stimuli(), S.build_stimuli()
    assert [s.as_dict() for s in a] == [s.as_dict() for s in b]
    assert S.fingerprint(a) == S.fingerprint(b) == S.fingerprint()
    assert S.STIMULUS_SET_VERSION == "sharbati-1"
    assert gate_a.STIMULUS_SET_VERSION == "gateA-1"
    assert S.fingerprint() != gate_a.fingerprint()
    names = [s.name for s in a]
    assert len(names) == len(set(names)) == 18
    assert {"sp_pot", "sp_dep", "ltp_ltd", "retention_13h"} <= set(names)
    assert len(gate_a.STIMULI) == 8                     # Gate A untouched


def test_protocols_follow_the_paper():
    for s in S.STIMULI:                                  # 50 pA / 10 ms everywhere
        assert {abs(p[2]) for p in s.pulses} == {50e-12}
        assert {p[1] for p in s.pulses} == {10e-3}
    assert S.BY_NAME["sp_pot"].pulses[0][2] > 0 > S.BY_NAME["sp_dep"].pulses[0][2]
    dts = []
    for n in S.PAIR_NAMES:                               # Fig. 4b: two INTERCALATION pulses
        st = S.BY_NAME[n]
        assert len(st.pulses) == 2 and all(p[2] > 0 for p in st.pulses)
        dts.append(st.params["dt_onset_s"])
        assert st.pulses[1][0] - st.pulses[0][0] == pytest.approx(st.params["dt_onset_s"])
    assert dts == sorted(dts) and dts[-1] == pytest.approx(1.8) and len(dts) == 14
    lt = S.BY_NAME["ltp_ltd"].params
    assert lt["n_each"] > 250
    ret = S.BY_NAME["retention_13h"]
    assert ret.t_stop - ret.params["t_settle_s"] == pytest.approx(13 * 3600)


@pytest.mark.parametrize("st", S.STIMULI, ids=lambda s: s.name)
def test_pwl_charge_equals_rectangular_charge(st):
    pts = gate_a.parse_spectre_pwl(st.spectre_pwl())
    np.testing.assert_allclose(np.array(pts), np.array(st.pwl_points()), rtol=1e-11, atol=1e-18)
    t, i = map(np.array, zip(*pts))
    q_pwl = np.sum(0.5 * (i[1:] + i[:-1]) * np.diff(t))
    q_rect = sum(w * a for _, w, a in st.pulses)
    assert q_pwl == pytest.approx(q_rect, rel=1e-9, abs=1e-24)


def test_retention_grid_is_fine_then_coarse():
    st = S.BY_NAME["retention_13h"]
    ts = S.grid(st)
    assert ts[0] == 0.0 and ts[-1] == st.t_stop and np.all(np.diff(ts) > 0)
    t_set = st.params["t_settle_s"]
    assert t_set in ts
    for e in st.waveform().edges:
        assert np.min(np.abs(ts - e)) < 1e-12
    d = np.diff(ts)
    assert d[ts[1:] <= t_set].max() <= S.RET_DT_FINE * (1 + 1e-9)
    assert d[ts[:-1] >= t_set].max() == pytest.approx(S.RET_DT_COARSE)
    assert st.dt_max == S.RET_DT_COARSE                  # Spectre maxstep


def test_thinning_keeps_every_edge():
    for name in ("ltp_ltd", "pp_dt0015ms", "retention_13h"):
        st = S.BY_NAME[name]
        ts = S.grid(st)
        kept = ts[core.keep_mask(ts, st)]
        for e in st.waveform().edges:
            assert np.min(np.abs(kept - e)) < 1e-12
        assert kept[0] == 0.0 and kept[-1] == st.t_stop


# ---- targets file --------------------------------------------------------
REQUIRED = {"id", "quantity", "value", "unit", "comparison", "tolerance", "tolerance_basis",
            "page_figure", "role", "source_type", "stimulus", "metric", "extraction"}


def test_targets_schema():
    t = json.loads(core.TARGETS.read_text(encoding="utf-8"))
    assert t["stimulus_set"] == S.STIMULUS_SET_VERSION
    ids = [r["id"] for r in t["targets"]]
    assert len(ids) == len(set(ids)) >= 15
    for r in t["targets"]:
        assert REQUIRED <= set(r), r["id"]
        assert r["role"] in ("calibration", "held-out")
        assert r["source_type"] in ("text", "digitized", "SI-missing")
        assert r["comparison"] in ("equal", "gt", "lt")
        assert isinstance(r["value"], (int, float))
        assert r["tolerance"] is None                    # paper gives no error bars
        assert r["page_figure"].startswith("p.")
        assert r["metric"].split(".")[0] in ("sp_pot", "sp_dep", "ltp", "pp", "ret")
    # the summary must agree with the per-row roles
    if not any(r["role"] == "held-out" for r in t["targets"]):
        assert t["held_out_summary"].startswith("NONE")
    for slot in t["curve_slots"]:
        head, x, y = score.read_curve(core.PAPER_DIR / slot["file"])
        assert len(x) == 0 and len(y) == 0               # empty until real data exists
        for k in ("figure", "source_type", "digitizer", "date", "x_unit", "y_unit"):
            assert k in head


@pytest.fixture(scope="module")
def v3_twin_metrics():
    cache = {}

    def loader(name):
        if name not in cache:
            cache[name] = core.run_twin("v3", S.BY_NAME[name])[0]
        return cache[name]
    vals, series = score.extract_all("twin", "v3", loader=loader)
    return vals, series


def test_every_target_metric_is_extracted(v3_twin_metrics):
    vals, _ = v3_twin_metrics
    t = score.load_targets()
    for r in t["targets"]:
        assert r["metric"] in vals, r["metric"]
        assert math.isfinite(vals[r["metric"]]), r["metric"]
    # Rinit = 4400 ohm, plus 5 s of self-discharge before the pulse
    assert vals["sp_pot.R0_ohm"] == pytest.approx(4400.0, rel=1e-5)
    assert vals["sp_pot.dR_peak_ohm"] < 0 < vals["sp_dep.dR_peak_ohm"]


# ---- scorer on synthetic data --------------------------------------------
def test_three_exp_fit_recovers_known_taus():
    dt = np.array(S.PP_DTS_MS) * 1e-3
    c = (2.0e-6, 1.5e-6, 1.0e-6)
    y = M.three_exp(dt, c[0], 0.022, c[1], 0.315, c[2], 19.0)
    free, fixed = M.fit_three_exp(dt, y)
    assert free["ok"]
    assert free["tau_s"][0] == pytest.approx(0.022, rel=0.02)
    assert free["tau_s"][1] == pytest.approx(0.315, rel=0.02)
    assert free["tau_s"][2] == pytest.approx(19.0, rel=0.25)   # weakly identified
    assert fixed["r2"] == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(fixed["c_S"], c, rtol=1e-6)
    flat, _ = M.fit_three_exp(dt, np.full(len(dt), 1e-6))
    assert not flat["ok"] and "flat" in flat["note"]


def _cols(t, g):
    t = np.asarray(t, dtype=float)
    g = np.asarray(g, dtype=float)
    return {"t_s": t, "G_S": g, "R_ohm": 1.0 / g}


def test_single_pulse_extraction_synthetic():
    st = S.BY_NAME["sp_pot"]
    t = np.linspace(0, S.SP_T_STOP, 200001)
    R = np.full(len(t), 4410.0)
    after = t >= S.SP_T_ON + 0.01
    R[after] = 4400.0 - 20.0 * np.exp(-(t[after] - S.SP_T_ON - 0.01) / 2.0)
    m = M.single_pulse(_cols(t, 1 / R), st)
    assert m["R0_ohm"] == pytest.approx(4410.0)
    assert m["dR_peak_ohm"] == pytest.approx(-30.0, abs=0.05)
    assert m["dR_asym_ohm"] == pytest.approx(-10.0, abs=1e-6)
    assert m["dR_fig_ohm"] == pytest.approx(-10.0, abs=1e-3)
    assert m["step_pct_settled"] == pytest.approx(100 * 10 / 4410)


def test_ltp_extraction_synthetic_linear():
    st = S.BY_NAME["ltp_ltd"]
    p = st.params
    t = np.linspace(0, st.t_stop, 400001)
    t = np.union1d(t, [p["t_ltp_s"], p["t_ltd_s"]])
    g = np.interp(t, [0, p["t_ltp_s"], p["t_ltd_s"], st.t_stop],
                  [1e-4, 1e-4, 1.1e-3, 1e-4])
    m = M.ltp_ltd(_cols(t, g), st)
    assert m["r2_10_90"] == pytest.approx(1.0, abs=1e-9)
    assert m["n_states"] == p["n_each"]
    assert m["modulation_pct"] == pytest.approx(1000.0, rel=1e-9)


def test_paired_pulse_and_retention_synthetic():
    st = S.BY_NAME["pp_dt0095ms"]
    p = st.params
    # canonical-column convention: the sample AT an onset is the pre-pulse value
    t = np.union1d(np.linspace(0, st.t_stop, 100001), [p["t1_s"], p["t2_s"]])
    g = np.full(len(t), 2e-4)
    g[(t > p["t1_s"]) & (t <= p["t2_s"])] = 2.03e-4
    g[t > p["t2_s"]] = 2.05e-4
    g[t >= p["t2_s"] + 0.05] = 2.04e-4
    d = M.paired_pulse(_cols(t, g), st)
    assert d["dG_S"] == pytest.approx(0.02e-4)
    assert d["dG_alt_S"] == pytest.approx(0.02e-4)
    st = S.BY_NAME["retention_13h"]
    t = np.linspace(0, st.t_stop, 5001)
    g = 2e-4 * np.exp(-np.maximum(t - st.params["t_settle_s"], 0) / 1e6)
    assert M.retention_pct(_cols(t, g), st) == pytest.approx(
        100 * (math.exp(-46800 / 1e6) - 1), rel=1e-4)


def test_score_rows_judge_and_pending():
    t = score.load_targets()
    vals = {("v3", "twin"): {"sp_pot.dR_peak_ohm": -33.0, "ltp.n_states": 300,
                             "sp_pot.step_pct_settled": 0.5}}
    rows = {r["target"]: r for r in score.score_rows(t, vals)}
    assert rows["fig3a_dR_peak"]["twin_vs_paper"] == "-10.0%"
    assert rows["fig3a_dR_peak"]["spectre_vs_paper"] == "pending"
    assert rows["fig3c_states"]["twin_vs_paper"] == "met"
    assert rows["p2_step_pct"]["twin_vs_paper"] == "NOT met"
    assert rows["fig4b_tau1"]["twin_vs_paper"] == "pending"


def test_curve_score_synthetic():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    s = score.curve_score(x, 2 * x, x, 2 * x)
    assert s["r2"] == pytest.approx(1.0) and s["rmse"] == pytest.approx(0.0)


# ---- decks and Spectre import -------------------------------------------
def test_decks_generated(tmp_path):
    make_decks_paper.main(["--out", str(tmp_path)])
    for key in common.MODELS:
        for st in S.STIMULI:
            text = (tmp_path / key / f"{st.name}.scs").read_text()
            assert "reltol=1e-6 vabstol=1e-12 iabstol=1e-18" in text
            assert "igate (0 n_igsc) isource type=pwl" in text
            assert "sharbati-1" in text and S.fingerprint()[:16] in text
            assert f"maxstep={st.dt_max:g}" in text
            wave = text[text.index("wave=["):text.index("]", text.index("wave=[")) + 1]
            np.testing.assert_allclose(np.array(gate_a.parse_spectre_pwl(wave)),
                                       np.array(st.pwl_points()), rtol=1e-11, atol=1e-18)
    assert "maxstep=60 " in (tmp_path / "v3" / "retention_13h.scs").read_text()
    assert "sp_pot.scs" in (tmp_path / "run_first.sh").read_text()
    assert "*.raw/" in (tmp_path / ".gitignore").read_text()
    make_decks_paper.main(["--out", str(tmp_path / "flip"), "--flip-gate"])
    assert "igate (n_igsc 0) isource" in (tmp_path / "flip" / "v3" / "sp_pot.scs").read_text()


def test_scrub_round_trip_with_thinning(tmp_path):
    st = S.BY_NAME["pp_dt0045ms"]
    cols, _ = core.run_twin("v3", st)
    # fake a Spectre run: PWL drive (1 us ramps) on a time base that contains them
    pts = np.array(st.pwl_points())
    t = np.union1d(cols["t_s"], pts[:, 0])
    drive = np.interp(t, pts[:, 0], pts[:, 1])
    R = np.interp(t, cols["t_s"], cols["R_ohm"])
    x = np.interp(t, cols["t_s"], cols["x_Li"])
    vol = np.interp(t, cols["t_s"], cols["G_vol_S"])
    head = ['"PSFversion" "1.00"', '"simulator" "spectre"', '"version" "21.1.0"',
            '"host" "cad07.engr.example.edu"', '"user" "jdoe"']
    names = ["n_rmem", "X1.n_x", "X1.n_v1", "X1.n_v2", "X1.n_v3", "X1.n_stdp", "vread:p", "X1:IGsc"]
    vals = [R, x, vol, 0 * vol, 0 * vol, 0 * vol, -0.01 / R, drive]
    out = ["HEADER", *head, "TYPE", '"V" FLOAT DOUBLE PROP( "units" "V" )', "SWEEP",
           '"time" "sweep" PROP( "units" "s" )', "TRACE", *[f'"{n}" "V"' for n in names], "VALUE"]
    for k in range(len(t)):
        out.append(f'"time" {t[k]:.15e}')
        out += [f'"{n}" {v[k]:.15e}' for n, v in zip(names, vals)]
    out.append("END")
    src = tmp_path / "tran.tran"
    src.write_text("\n".join(out) + "\n")
    dest = scrub_paper.scrub_one("v3", st.name, src, out_root=tmp_path)
    body = dest.read_text()
    assert body.startswith("# source: Spectre 21.1; model: v3; stimulus_set: sharbati-1 ")
    for bad in ("jdoe", "cad07", "example.edu"):
        assert bad not in body
    ref, _ = common.read_csv(dest)
    assert len(ref["t_s"]) < len(t)                      # thinned
    twin = core.thin(cols, st)
    assert compare.drive_check(twin, ref) == "ok"        # charge kept exactly
    tg = compare.common_grid(twin["t_s"], ref["t_s"])
    m = compare.metrics(np.interp(tg, twin["t_s"], twin["G_S"]),
                        np.interp(tg, ref["t_s"], ref["G_S"]))
    assert m["nrmse"] < 1e-3
    pp_twin = M.paired_pulse(cols, st)
    pp_ref = M.paired_pulse(ref, st)
    assert pp_ref["dG_S"] == pytest.approx(pp_twin["dG_S"], rel=1e-3)
