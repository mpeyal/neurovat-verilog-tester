"""GUI power profiles: one file per device family, no cross-family fallback."""
import json

import pytest

neuro = pytest.importorskip("vatester.neuro")


def _write(folder, name, values):
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(json.dumps(values), encoding="utf-8")


def test_profile_is_picked_by_family(tmp_path):
    d = tmp_path / "power_profiles"
    _write(d, "alpha.json", {"e_spike": 1e-12, "corner": "tt", "temperature": "27 C"})
    _write(d, "beta.json", {"e_spike": 2e-12, "p_leak": 3e-9})
    a, note_a = neuro.load_power_params("Alpha", str(d))
    b, _ = neuro.load_power_params("BETA", str(d))
    assert a.e_spike == 1e-12 and "p_leak" in a.placeholders()
    assert "alpha.json" in note_a and "tt 27 C" in note_a
    assert b.e_spike == 2e-12 and b.p_leak == 3e-9


def test_no_fallback_to_another_family(tmp_path):
    d = tmp_path / "power_profiles"
    _write(d, "alpha.json", {"e_spike": 1e-12})
    p, note = neuro.load_power_params("gamma", str(d))
    assert p.sources == {} and "placeholder" in note
    p, _ = neuro.load_power_params(None, str(d))
    assert p.sources == {}


def test_bad_profile_is_reported_not_used(tmp_path):
    d = tmp_path / "power_profiles"
    _write(d, "alpha.json", {"e_spike": -1})
    p, note = neuro.load_power_params("alpha", str(d))
    assert p.sources == {} and "ignored" in note
