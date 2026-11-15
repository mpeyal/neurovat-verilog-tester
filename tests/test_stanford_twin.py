"""Stanford-PKU RRAM twin (twins/stanford_pku_rram.py): defaults vs the .va,
licence notices, seeded variability, non-disturbing read, RESET self-limit,
long GUI steps and JSON-safe state."""

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TWIN_PATH = ROOT / "twins" / "stanford_pku_rram.py"
VA_PATH = ROOT / "verilog" / "stanford_pku_rram.va"      # local copy, git-ignored
MODULE = "rram_v_1_0_0"


def _load_twin():
    """Import twins/stanford_pku_rram.py by path (trusted repo file)."""
    spec = importlib.util.spec_from_file_location("nvat_test_stanford_twin", TWIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_twin()


def new_twin(**overrides):
    return M.StanfordPKURRAM(M.StanfordPKUParams(**overrides))


def test_defaults_match_va_text():
    p = M.StanfordPKUParams()
    assert (p.gap_ini, p.gap_min, p.gap_max) == (2e-10, 2e-10, 17e-10)
    assert (p.g0, p.V0, p.Vel0, p.I0, p.beta, p.gamma0) == (0.25e-9, 0.25, 10, 1e-3, 0.8, 16)
    assert (p.Ea, p.a0, p.T_ini, p.F_min, p.Rth, p.tox) == (0.6, 0.25e-9, 298, 1.4e9, 2.1e3, 12e-9)
    assert p.model_switch == 0 and p.time_step == 1e-9


@pytest.mark.skipif(not VA_PATH.is_file(), reason="Stanford .va not downloaded")
def test_defaults_match_local_va_file():
    from vatester import tools_ext, va_scan
    va = va_scan.parse_va(str(VA_PATH))
    assert va.module == MODULE
    defaults = {f: getattr(M.StanfordPKUParams(), f) for f in M.StanfordPKUParams.__dataclass_fields__}
    va.model_key = "stanford_pku"
    res = tools_ext.check_consistency([va], {"stanford_pku": defaults}, {})
    assert res["n_mismatch"] == 0, res["reports"]
    text = VA_PATH.read_text(encoding="utf-8")
    assert "Copyright @ 2014 Stanford University" in text          # notice intact


def test_twin_header_keeps_notices():
    src = TWIN_PATH.read_text(encoding="utf-8")
    for s in ("Copyright @ 2014 Stanford University (Stanford)",
              "Copyright The Board Trustees of the Leland Stanford Junior University 2013",
              "Stanford Nanoelectronics Research Group of Prof. H.-S. Philip Wong",
              "4. The users agree to reproduce any copyright notice"):
        assert s in src


def test_variability_seeded_and_off_by_default():
    def run(**kw):
        d = new_twin(**kw)
        for _ in range(50):
            d.step(0, 1e-9, -1.3)
        return d.y
    assert run() == run()
    assert run(model_switch=1, rand_seed_ini=7) == run(model_switch=1, rand_seed_ini=7)
    assert run(model_switch=1, rand_seed_ini=7) != run(model_switch=1, rand_seed_ini=8)


def test_read_is_exactly_non_disturbing_and_G_is_read_conductance():
    d = new_twin(gap_ini=0.8e-9)
    y0 = d.y
    for v in (0.1, 0.5, 0.9, -0.9):
        d.step(0, 1e-3, v)
    assert d.y == y0
    assert d.G == pytest.approx(d.current(0.1) / 0.1, rel=1e-15)
    assert d.R == pytest.approx(1 / d.G)


def test_reset_self_limits_at_stop_gap():
    d = new_twin()
    for _ in range(200):
        d.step(0, 10e-9, -1.3)                               # 2 us at -1.3 V
    g_star = 1e-9 * ((16 - 1.4e9 * 12e-9 / 1.3) / 0.8) ** (1 / 3)
    assert d.y == pytest.approx(g_star, rel=1e-9)
    assert d.y < 1.7e-9


def test_gui_step_with_long_dt_is_finite_and_fast():
    d = new_twin()
    for v in (-1.3, 0.0, 1.5, 0.0, -1.0, 0.1):
        d.step(0, 1e-3, v)                                   # ms-scale GUI steps
        assert math.isfinite(d.y) and math.isfinite(d.G)
    assert d.n_rhs < 20000


def test_G_setter_round_trip_and_json_safe_state():
    import json
    d = new_twin(model_switch=1, rand_seed_ini=3)
    d.G = 1e-4
    assert d.G == pytest.approx(1e-4, rel=1e-12)
    d.G = 1.0                                                # above LRS: clamped
    assert d.gap == pytest.approx(2e-10)
    d.step(0, 1e-9, -1.3)
    json.dumps({k: v for k, v in vars(d).items() if k != "p"})
