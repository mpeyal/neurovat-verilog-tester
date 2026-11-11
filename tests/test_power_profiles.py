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


def test_explicit_choice_and_default_file(tmp_path):
    d = tmp_path / "power_profiles"
    _write(d, "alpha.json", {"e_spike": 1e-12})
    _write(d, "beta.json", {"e_spike": 2e-12})
    assert neuro.list_power_profiles(str(d)) == ["alpha", "beta"]
    # an explicit choice wins over the device family
    p, note = neuro.load_power_params("alpha", str(d), choice="beta")
    assert p.e_spike == 2e-12 and "beta.json" in note
    # placeholders only, on request
    p, _ = neuro.load_power_params("alpha", str(d), choice=neuro.PROFILE_NONE)
    assert p.sources == {}
    # default: own family first, then default.json, then placeholders
    p, _ = neuro.load_power_params("alpha", str(d), choice=neuro.PROFILE_DEFAULT)
    assert p.e_spike == 1e-12
    p, _ = neuro.load_power_params("gamma", str(d))
    assert p.sources == {}
    _write(d, "default.json", {"e_spike": 5e-12})
    p, note = neuro.load_power_params("gamma", str(d))
    assert p.e_spike == 5e-12 and "default.json" in note


def test_new_params_reprice_a_recorded_run():
    import numpy as np
    from neurovat.power import EnergyMeter, PowerParams
    m = EnergyMeter([(1, 1)], ["current"], PowerParams(e_spike=1e-12), 1e-3)
    m.record_forward("eval", [np.zeros((1, 1))], [np.zeros((4, 1)), np.ones((4, 1))])
    assert m.report("eval")["energy_J"]["neurons"] == pytest.approx(4e-12)
    m.params = PowerParams(e_spike=3e-12)
    assert m.report("eval")["energy_J"]["neurons"] == pytest.approx(12e-12)


def test_power_components_add_up_to_the_total():
    import numpy as np
    from neurovat.power import EnergyMeter, PowerParams
    m = EnergyMeter([(2, 3)], ["current"], PowerParams(e_spike=1e-12), 1e-3)
    m.record_forward("eval", [np.full((2, 3), 1e-6)],
                     [np.ones((4, 3)), np.ones((4, 2))])
    rep = m.report("eval", parameters=False)
    top = sum(neuro.power_component(rep, g, k)
              for _, g, k in neuro.POWER_COMPONENTS if g is None)
    assert top == pytest.approx(rep["energy_J"]["total"])
    for group in ("peripherals", "static"):
        parts = sum(neuro.power_component(rep, g, k)
                    for _, g, k in neuro.POWER_COMPONENTS if g == group)
        assert parts == pytest.approx(rep["energy_J"][group])
