"""C6 ablation harness (validation/ablation): smoke run of every arm on a
tiny network, and the NeuroSim-style fitted curve."""
import sys
from pathlib import Path

import numpy as np
import pytest

ABL = Path(__file__).resolve().parents[1] / "validation" / "ablation"
sys.path.insert(0, str(ABL))

import neurosim_curve as nc   # noqa: E402
import run_ablation as ra     # noqa: E402


@pytest.fixture(scope="module")
def fit():
    return nc.fit_v3(ra.v3_params())


def test_neurosim_fit_reports_its_residual(fit):
    assert fit["p_max"] == round(ra.v3_params().Q_full / ra.v3_params().Q_ref)
    for br in ("ltp", "ltd"):
        assert fit[br]["A"] > 0 and np.isfinite(fit[br]["rmse_frac_range"])
        assert 0.0 <= fit[br]["max_err_frac_range"] < 0.5


def test_neurosim_array_moves_along_its_curves(fit):
    arr = nc.array_from_fit(fit, 1, 2)
    lo, hi = arr.g_min, arr.g_max
    arr.set_conductance(np.full((1, 2), lo))
    n_pulses = fit["p_max"]
    for _ in range(n_pulses):                 # LTP from Gmin: reaches Gmax
        arr.program(np.array([0]), np.array([0]), np.array([50e-12]), 10e-3)
    assert arr.conductance()[0, 0] == pytest.approx(hi, rel=1e-6)
    assert arr.conductance()[0, 1] == lo      # untouched
    g = arr.conductance()[0, 0]
    arr.program(np.array([0]), np.array([0]), np.array([-50e-12]), 10e-3)
    assert arr.conductance()[0, 0] < g
    # two half pulses == one pulse (charge-counted, fractional pulses)
    a, b = nc.array_from_fit(fit, 1, 1), nc.array_from_fit(fit, 1, 1)
    a.program(np.array([0]), np.array([0]), np.array([50e-12]), 10e-3)
    for _ in range(2):
        b.program(np.array([0]), np.array([0]), np.array([25e-12]), 10e-3)
    assert a.conductance()[0, 0] == pytest.approx(b.conductance()[0, 0],
                                                  rel=1e-12)


@pytest.mark.parametrize("arm", ra.ARMS)
def test_every_arm_runs_on_a_tiny_network(arm, fit):
    r = ra.one_job((arm, 1, 2, 12, 100, 1.0, fit))
    assert r["arm"] == arm and r["seconds"] == pytest.approx(1.2)
    assert r["gap_ms"] == 50.0 and r["writes"] >= 0
    assert len(r["w_read"]["hist"]) == 10
    if arm == "ideal":
        assert r["calibration"] is None
        assert r["w_read"] == r["w_stored"]
    else:
        cal = r["calibration"]
        assert cal["pot_step"] == pytest.approx(0.03125, rel=0.03)
    table = ra.summarise([r])
    assert table[0]["arm"] == arm and "|" in ra.markdown(table, {
        "afferents": 100, "presentations": 12, "seeds": 1})
