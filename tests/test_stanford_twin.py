"""Stanford-PKU RRAM twin: determinism, reset hygiene, I-V hysteresis sanity,
.va integration semantics (idt wind-up) and agreement with a literal Euler
transcription of the .va."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.stanford import core  # noqa: E402
from validation.stanford import stimuli as S  # noqa: E402

M = core.twin_module()


def test_defaults_match_va_text():
    p = M.StanfordPKUParams()
    assert (p.gap_ini, p.gap_min, p.gap_max) == (2e-10, 2e-10, 17e-10)
    assert (p.g0, p.V0, p.Vel0, p.I0, p.beta, p.gamma0) == (0.25e-9, 0.25, 10, 1e-3, 0.8, 16)
    assert (p.Ea, p.a0, p.T_ini, p.F_min, p.Rth, p.tox) == (0.6, 0.25e-9, 298, 1.4e9, 2.1e3, 12e-9)
    assert p.model_switch == 0 and p.time_step == 1e-9


@pytest.mark.skipif(not core.VA_PATH.is_file(), reason="Stanford .va not downloaded")
def test_defaults_match_local_va_file():
    from vatester import tools_ext, va_scan
    va = va_scan.parse_va(str(core.VA_PATH))
    assert va.module == core.MODULE
    defaults = {f: getattr(M.StanfordPKUParams(), f) for f in M.StanfordPKUParams.__dataclass_fields__}
    va.model_key = "stanford_pku"
    res = tools_ext.check_consistency([va], {"stanford_pku": defaults}, {})
    assert res["n_mismatch"] == 0, res["reports"]
    text = core.VA_PATH.read_text(encoding="utf-8")
    assert "Copyright @ 2014 Stanford University" in text          # notice intact


def test_twin_header_keeps_notices():
    src = core.TWIN_PATH.read_text(encoding="utf-8")
    for s in ("Copyright @ 2014 Stanford University (Stanford)",
              "Copyright The Board Trustees of the Leland Stanford Junior University 2013",
              "Stanford Nanoelectronics Research Group of Prof. H.-S. Philip Wong",
              "4. The users agree to reproduce any copyright notice"):
        assert s in src


def test_deterministic_and_reset_has_no_leak():
    st = S.BY_NAME["reset_train"]
    a, _, _ = core.run_twin(st)
    b, _, _ = core.run_twin(st)
    for c in core.COLUMNS:
        assert np.array_equal(a[c], b[c])
    d = core.new_twin(st)
    ts = core.grid(st)
    x = core.simulate_twin(d, st, ts)
    y = core.simulate_twin(d, st, ts)          # simulate_twin resets first
    assert np.array_equal(x["gap_state_m"], y["gap_state_m"])


def test_variability_seeded_and_off_by_default():
    def run(**kw):
        d = core.new_twin(**kw)
        for _ in range(50):
            d.step(0, 1e-9, -1.3)
        return d.y
    assert run() == run()
    assert run(model_switch=1, rand_seed_ini=7) == run(model_switch=1, rand_seed_ini=7)
    assert run(model_switch=1, rand_seed_ini=7) != run(model_switch=1, rand_seed_ini=8)


def test_read_is_exactly_non_disturbing_and_G_is_read_conductance():
    d = core.new_twin(gap_ini=0.8e-9)
    y0 = d.y
    for v in (0.1, 0.5, 0.9, -0.9):
        d.step(0, 1e-3, v)
    assert d.y == y0
    assert d.G == pytest.approx(d.current(0.1) / 0.1, rel=1e-15)
    assert d.R == pytest.approx(1 / d.G)


def test_iv_hysteresis_sanity():
    r, _, _ = core.run_twin(S.BY_NAME["dc_reset_ref"])
    s, _, _ = core.run_twin(S.BY_NAME["dc_set_ref"])
    # RESET half sweep: LRS -> HRS; on the way back |I| at -0.5 V is lower
    assert r["gap_m"][0] == pytest.approx(2e-10) and r["gap_m"][-1] == pytest.approx(17e-10)
    k = np.flatnonzero(np.isclose(r["V_TE_V"], -0.5, atol=2.5e-4))
    assert abs(r["I_TE_A"][k[0]]) > 100 * abs(r["I_TE_A"][k[-1]])
    # switching voltages near the field threshold F_min * tox / gamma
    i_sw = np.argmax(np.diff(r["gap_m"]) > 0)
    assert -1.15 < r["V_TE_V"][i_sw] < -1.0
    # SET half sweep: HRS -> LRS, abrupt SET at ~1.39 V (gamma(1.7 nm) = 12.07)
    assert s["gap_m"][0] == pytest.approx(17e-10) and s["gap_m"][-1] == pytest.approx(2e-10)
    i_sw = np.argmax(np.diff(s["gap_m"]) < 0)
    assert 1.35 < s["V_TE_V"][i_sw] < 1.42
    k = np.flatnonzero(np.isclose(s["V_TE_V"], 0.5, atol=2.5e-4))
    assert abs(s["I_TE_A"][k[-1]]) > 100 * abs(s["I_TE_A"][k[0]])


def test_gradual_trains_monotone():
    for name, sign in (("reset_train", -1), ("set_train", +1)):
        st = S.BY_NAME[name]
        c, _, _ = core.run_twin(st)
        g = core.read_conductance(c, st)
        assert len(g) == 20
        assert np.all(sign * np.diff(g) > 0)


def test_idt_windup_matches_va_and_clamp_state_option():
    st = S.BY_NAME["pulse_set_ref"]
    c, _, _ = core.run_twin(st)
    assert c["gap_state_m"][0] == pytest.approx(1.9e-9)       # idt starts above gap_max
    assert c["gap_m"][0] == pytest.approx(1.7e-9)
    assert c["gap_m"][-1] == pytest.approx(2e-10)
    assert c["gap_state_m"][-1] < -1e-7                      # wound far below gap_min
    m, _, _ = core.run_twin(st, clamp_state=1)
    assert np.all(m["gap_state_m"] >= 2e-10 - 1e-24)
    # dc_loop: the second RESET sweep cannot undo the SET-side wind-up
    loop, _, _ = core.run_twin(S.BY_NAME["dc_loop"])
    n = len(loop["t_s"])
    assert loop["gap_m"][: n // 2].max() > 1.0e-9
    assert loop["gap_m"][n // 2:].max() < 0.25e-9


def test_reset_self_limits_at_stop_gap():
    d = core.new_twin()
    for _ in range(200):
        d.step(0, 10e-9, -1.3)                               # 2 us at -1.3 V
    g_star = 1e-9 * ((16 - 1.4e9 * 12e-9 / 1.3) / 0.8) ** (1 / 3)
    assert d.y == pytest.approx(g_star, rel=1e-9)
    assert d.y < 1.7e-9


def test_matches_literal_euler_transcription():
    pts = ((0, 0.0), (50, -1.3), (5000, -1.3), (5050, 0.0), (6000, 0.0),
           (6050, 1.5), (9000, 1.5), (9050, 0.0), (10000, 0.0))
    st = S.VoltageStimulus("t", "t", pts, 10000, 1e-10)
    c, _, _ = core.run_twin(st)
    tb, gb, yb = core.brute_reference(pts, st.t_stop, 1e-13)
    tb2, gb2, yb2 = core.brute_reference(pts, st.t_stop, 2e-13)
    rich = 2 * yb[-1] - yb2[-1]
    assert abs(c["gap_state_m"][-1] - rich) < 1e-14
    assert abs(yb[-1] - yb2[-1]) > abs(c["gap_state_m"][-1] - rich)   # twin beats Euler


def test_gui_step_with_long_dt_is_finite_and_fast():
    d = core.new_twin()
    for v in (-1.3, 0.0, 1.5, 0.0, -1.0, 0.1):
        d.step(0, 1e-3, v)                                   # ms-scale GUI steps
        assert math.isfinite(d.y) and math.isfinite(d.G)
    assert d.n_rhs < 20000


def test_ramp_path_matches_literal_euler_transcription():
    # 0 -> -1.5 -> 0 V at 5e7 V/s: the RESET stop gap moves with V (sliding /
    # freezing logic of the adaptive integrator)
    pts = ((0, 0.0), (30000, -1.5), (60000, 0.0))
    st = S.VoltageStimulus("r", "r", pts, 60000, 1e-10)
    c, _, _ = core.run_twin(st)
    _, _, yb = core.brute_reference(pts, st.t_stop, 2e-13)
    _, _, yb2 = core.brute_reference(pts, st.t_stop, 4e-13)
    rich = 2 * yb[-1] - yb2[-1]
    assert c["gap_state_m"][-1] > 0.5e-9                      # it did RESET
    assert abs(c["gap_state_m"][-1] - rich) < 5e-14


def test_G_setter_round_trip_and_json_safe_state():
    import json
    d = core.new_twin(model_switch=1, rand_seed_ini=3)
    d.G = 1e-4
    assert d.G == pytest.approx(1e-4, rel=1e-12)
    d.G = 1.0                                                # above LRS: clamped
    assert d.gap == pytest.approx(2e-10)
    d.step(0, 1e-9, -1.3)
    json.dumps({k: v for k, v in vars(d).items() if k != "p"})
