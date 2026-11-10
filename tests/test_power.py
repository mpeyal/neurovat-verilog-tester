"""Energy model: the arithmetic is checked by hand on a 2x2 crossbar with
known conductances and spikes."""
import json

import numpy as np
import pytest

from neurovat import (CrossbarSNN, NetConfig, NeuronParams, PowerParams,
                      STDPParams, experiments)
from neurovat import power as pw
from neurovat.power import EnergyMeter, PLACEHOLDERS

from conftest import STDP_AMPS

# conductances (S): rows = post neuron j, columns = input i
G = np.array([[1e-6, 2e-6],
              [3e-6, 4e-6]])
# 4 time steps; input 0 spikes at t=0,2; input 1 at t=1
PRE = np.array([[1, 0], [0, 1], [1, 0], [0, 0]], bool)
# neuron 0 spikes once (t=2); neuron 1 is silent
POST = np.array([[0, 0], [0, 0], [1, 0], [0, 0]], bool)

FULL = dict(v_read=0.2, t_read=1e-6, v_write=1.5, e_spike=2e-12, p_leak=1e-9,
            e_row_driver=1e-13, e_col_sense=5e-13, e_wta=7e-13,
            p_peripheral_static=2e-9)


def meter(kind="voltage", **params):
    return EnergyMeter([(2, 2)], [kind], PowerParams(**dict(FULL, **params)),
                       dt_s=1e-3)


# ---- hand-checked arithmetic ----------------------------------------------

def test_read_energy_per_device_is_v2_g_t_per_input_spike():
    m = meter()
    m.record_forward("eval", [G], [PRE, POST])
    # V^2 t = 0.04 * 1e-6 = 4e-8; input 0 is read twice, input 1 once
    expected = np.array([[4e-8 * 1e-6 * 2, 4e-8 * 2e-6 * 1],
                         [4e-8 * 3e-6 * 2, 4e-8 * 4e-6 * 1]])
    np.testing.assert_allclose(m.read_energy("eval", 0), expected, rtol=1e-12)
    np.testing.assert_allclose(expected.sum(), 5.6e-13, rtol=1e-12)


def test_breakdown_on_the_2x2_crossbar():
    m = meter()
    m.record_forward("eval", [G], [PRE, POST], wta_ops=1)
    r = m.report("eval")
    e = r["energy_J"]
    assert e["array_read"] == pytest.approx(5.6e-13)
    assert e["array_write"] == 0.0
    assert e["neurons"] == pytest.approx(2e-12)              # 1 spike x 2 pJ
    # 3 row-drives x 0.1 pJ; 3 active steps x 2 columns x 0.5 pJ; 1 WTA event
    assert r["peripherals_J"] == pytest.approx(
        {"row_driver": 3e-13, "column_sense": 3e-12, "wta": 7e-13})
    assert e["peripherals"] == pytest.approx(4e-12)
    # 4 steps x 1 ms = 4 ms: 2 neurons x 1 nW, plus 2 nW of peripherals
    assert r["static_J"] == pytest.approx(
        {"neuron_leak": 8e-12, "peripheral_static": 8e-12})
    assert e["static"] == pytest.approx(16e-12)
    assert e["total"] == pytest.approx(22.56e-12)
    assert r["counts"]["sops"] == 6                          # 3 spikes x 2 posts
    assert r["e_sop_array_pJ"] == pytest.approx(0.56 / 6)
    assert r["e_sop_pJ"] == pytest.approx(22.56 / 6)
    assert r["e_neuron_spike_pJ"] == pytest.approx(10.0)     # (2 + 8) pJ / 1 spike
    assert r["e_inference_J"] == pytest.approx(22.56e-12)
    assert r["avg_power_W"] == pytest.approx(22.56e-12 / 4e-3)
    assert r["placeholders"] == []


def test_clock_period_sets_static_energy_and_average_power():
    m = meter(timestep_s=1e-6)                 # 1 us hardware clock per step
    m.record_forward("eval", [G], [PRE, POST])
    r = m.report("eval")
    assert r["sim_time_s"] == pytest.approx(4e-6)
    assert r["static_J"]["neuron_leak"] == pytest.approx(2 * 1e-9 * 4e-6)
    assert r["avg_power_W"] == pytest.approx(r["energy_J"]["total"] / 4e-6)


def test_write_energy_voltage_pulse_is_ohmic():
    m = meter()
    # 0.5 V for 1 ms on device (1, 0): 1 uS before, 3 uS after -> mean 2 uS
    m.record_write("train", 0, np.array([1]), np.array([0]), np.array([-0.5]),
                   1e-3, np.array([1e-6]), np.array([3e-6]))
    expected = np.zeros((2, 2))
    expected[1, 0] = 0.5 ** 2 * 2e-6 * 1e-3                  # 0.5 nJ
    np.testing.assert_allclose(m.write_energy("train", 0), expected)
    assert m.report("train")["e_write_pulse_pJ"] == pytest.approx(500.0)


def test_write_energy_uses_given_or_device_current():
    args = (0, np.array([0]), np.array([1]), np.array([0.5]), 1e-3,
            np.array([1e-6]), np.array([1e-6]))
    m = meter(write_current=2e-6)              # |V| I t = 0.5 * 2e-6 * 1e-3
    m.record_write("train", *args)
    assert m.write_energy("train", 0)[0, 1] == pytest.approx(1e-9)
    m = meter()                                # the device model reports 4 uA
    m.record_write("train", *args, current=np.array([4e-6]))
    assert m.write_energy("train", 0)[0, 1] == pytest.approx(2e-9)
    assert m.write_source(0) == "device model write_current()"


def test_write_energy_current_pulse_uses_compliance_voltage():
    m = meter(kind="current")
    # 1 nA for 1 ms at 1.5 V
    m.record_write("train", 0, np.array([0]), np.array([0]), np.array([1e-9]),
                   1e-3, np.array([1e-6]), np.array([1e-6]))
    assert m.write_energy("train", 0)[0, 0] == pytest.approx(1.5e-12)


def test_read_and_write_are_accounted_separately_and_per_phase():
    m = meter()
    m.record_forward("train", [G], [PRE, POST])
    m.record_write("train", 0, np.array([0]), np.array([0]), np.array([0.5]),
                   1e-3, np.array([1e-6]), np.array([1e-6]))
    m.record_forward("eval", [G], [PRE, POST])
    train, ev = m.report("train"), m.report("eval")
    assert train["energy_J"]["array_read"] == pytest.approx(5.6e-13)
    assert train["energy_J"]["array_write"] == pytest.approx(2.5e-10)
    assert ev["energy_J"]["array_write"] == 0.0
    assert m.maps("train")["write_pulses_c0"].tolist() == [[1, 0], [0, 0]]
    m.reset("eval")
    assert m.report("eval")["energy_J"]["total"] == 0.0
    assert m.report("eval")["e_sop_pJ"] is None


def test_batched_rasters_add_up():
    m, b = meter(), meter()
    for _ in range(3):
        m.record_forward("eval", [G], [PRE, POST])
    b.record_forward("eval", [G], [np.stack([PRE] * 3), np.stack([POST] * 3)])
    assert b.report("eval") == m.report("eval")
    assert b.report("eval")["counts"]["presentations"] == 3


# ---- placeholders and the neuron-value hook -------------------------------

def test_unset_parameters_are_flagged_never_silent():
    p = PowerParams(v_read=0.2)
    assert p.placeholders() == [n for n in PLACEHOLDERS if n != "v_read"]
    assert p.get("e_spike") == PLACEHOLDERS["e_spike"][0]
    table = p.table(dt_s=1e-3)
    assert table["v_read"]["source"] == "config"
    assert table["e_spike"]["source"] == "PLACEHOLDER"
    assert table["timestep_s"] == {"value": 1e-3, "unit": "s",
                                   "meaning": table["timestep_s"]["meaning"],
                                   "source": "network.dt_ms"}
    m = EnergyMeter([(2, 2)], ["voltage"], p)
    m.record_forward("eval", [G], [PRE, POST])
    rep = m.report("eval")
    assert rep["placeholders"] == p.placeholders()
    assert "PLACEHOLDER values (not measured): t_read" in pw.format_report(rep)
    assert PowerParams(**FULL).placeholders() == []
    with pytest.raises(ValueError):
        PowerParams(e_spike=-1.0)


def test_neuron_file_hook(tmp_path):
    path = tmp_path / "lif_tt_27c.json"
    path.write_text(json.dumps({"e_spike": 3.1e-12, "p_leak_W": 4.2e-9,
                                "corner": "tt", "vdd": 1.8}))
    p = PowerParams.from_config({"v_read": 0.2, "neuron_file": str(path)})
    assert (p.e_spike, p.p_leak) == (3.1e-12, 4.2e-9)
    assert p.sources["e_spike"] == "file:lif_tt_27c.json"
    assert p.metadata == {"corner": "tt", "vdd": 1.8}
    assert "e_spike" not in p.placeholders() and "p_leak" not in p.placeholders()

    csv_path = tmp_path / "lif.csv"
    csv_path.write_text("name,value,unit\n# extracted\ne_spike,2e-12,J\n")
    assert PowerParams().load_neuron_file(str(csv_path)).e_spike == 2e-12

    # the same value from two places is a conflict, not a silent override
    with pytest.raises(ValueError, match="set both"):
        PowerParams.from_config({"e_spike": 1e-12, "neuron_file": str(path)})
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"v_read": 0.3}))          # not a neuron value
    with pytest.raises(ValueError, match="unknown parameter"):
        PowerParams().load_neuron_file(str(bad))


def test_energies_follow_parameters_loaded_after_the_run():
    m = EnergyMeter([(2, 2)], ["voltage"], PowerParams(**FULL), dt_s=1e-3)
    m.record_forward("eval", [G], [PRE, POST])
    m.p.e_spike = 5e-12
    assert m.report("eval")["energy_J"]["neurons"] == pytest.approx(5e-12)


# ---- the meter inside the network -----------------------------------------

def tiny_net(rule="stdp", **power):
    net = CrossbarSNN(
        "softbound", NeuronParams(), STDPParams(**STDP_AMPS),
        NetConfig(grid_h=1, grid_w=2, n_out=2, learn_rule=rule, seed=1,
                  present_ms=40.0, pattern_set="bars"),
        patterns=[("a", np.array([1.0, 0.0])), ("b", np.array([0.0, 1.0]))],
        targets=[0, 1], power=PowerParams(**dict(FULL, **power)))
    net.set_weights(0, G * 10)
    return net


def test_network_read_energy_matches_its_spikes():
    net = tiny_net()
    res = net.present(np.array([1.0, 0.6]), learn=False)
    n_in = np.array([len(s) for s in res["layer_spikes"][0]], float)
    n_out = sum(res["n_out_spikes"])
    assert n_in.sum() > 0
    r = net.energy.report("eval")
    expected = 0.2 ** 2 * 1e-6 * (G * 10) * n_in[None, :]
    np.testing.assert_allclose(net.energy.read_energy("eval", 0), expected)
    assert r["counts"]["sops"] == 2 * n_in.sum()
    assert r["counts"]["neuron_spikes"] == n_out
    assert r["counts"]["steps"] == res["n_steps"] == 40
    assert r["energy_J"]["neurons"] == pytest.approx(2e-12 * n_out)
    assert r["static_J"]["neuron_leak"] == pytest.approx(2 * 1e-9 * 40e-3)
    assert "train" not in net.energy.tallies              # nothing was learned


def test_network_write_energy_of_one_pulse():
    net = tiny_net()
    g0 = net.weights_S()[1, 0]
    net._program(0, 1, 0, True, 1.0)           # one full-strength 0.05 V pulse
    g1 = net.weights_S()[1, 0]
    assert g1 > g0
    e = net.energy.write_energy("train", 0)
    width = net.S.pulse_width
    assert e[1, 0] == pytest.approx(0.05 ** 2 * 0.5 * (g0 + g1) * width)
    assert e.sum() == e[1, 0] and net.write_count == 1


def test_training_accumulates_and_surrogate_init_is_separate():
    net = tiny_net("surrogate")
    assert net.energy.report("init")["counts"]["write_pulses"] > 0
    net.fit(2)
    train, init = net.energy.report("train"), net.energy.report("init")
    assert train["counts"]["presentations"] == 4
    assert train["counts"]["write_pulses"] + init["counts"]["write_pulses"] \
        == net.write_count
    assert train["energy_J"]["array_read"] > 0


def test_batched_and_single_evaluation_agree(make_net):
    net = make_net(rule="surrogate")
    vecs = [vec for _lbl, vec in net.patterns]
    with net._frozen():
        net.energy.reset("eval")
        net.predict(vecs)
        batched = net.energy.report("eval")
        net.energy.reset("eval")
        for v in vecs:
            net.infer(v, detail=False)
        single = net.energy.report("eval")
    assert batched["counts"] == single["counts"]
    assert batched["energy_J"] == pytest.approx(single["energy_J"])


def test_meter_does_not_change_the_simulation(make_net):
    a, b = make_net(), make_net()
    b.energy.record_forward = lambda *args, **kw: None
    b.energy.record_write = lambda *args, **kw: None
    for net in (a, b):
        net.fit(1)
    np.testing.assert_array_equal(a.weights_S(), b.weights_S())


# ---- maps and sweeps -------------------------------------------------------

def test_maps_are_saved_as_arrays_and_heatmaps(tmp_path):
    net = tiny_net()
    net.fit(1)
    net.evaluate()
    path = pw.save_maps(net.energy, str(tmp_path / "maps.npz"))
    with np.load(path, allow_pickle=False) as z:
        assert z["train_read_J_c0"].shape == (2, 2)
        assert z["train_write_J_c0"].shape == (2, 2)
        assert z["eval_neuron_J_l1"].shape == (2,)
        assert z["eval_row_driver_J_c0"].shape == (2,)
        np.testing.assert_allclose(z["eval_read_J_c0"],
                                   net.energy.read_energy("eval", 0))
    pytest.importorskip("matplotlib")
    png = pw.plot_maps(net.energy, str(tmp_path / "maps.png"), "train")
    assert (tmp_path / "maps.png").stat().st_size > 1000 and png


def test_sweep_reports_accuracy_and_energy(tmp_path):
    cfg = {"name": "p", "data": {"builtin": "bars"},
           "network": {"grid_h": 4, "grid_w": 4, "n_out": 4, "present_ms": 40.0},
           "stdp": STDP_AMPS, "train": {"epochs": 1},
           "power": {"t_read": 1e-6, "e_spike": 2e-12},
           "sweep": {"parameter": "power.v_read", "values": [0.1, 0.2],
                     "seeds": [1, 2]}}
    result = experiments.run_experiment(cfg, out_dir=str(tmp_path), log=lambda s: None)
    low, high = result["summary"]
    for key in ("test_acc_mean", "infer_energy_J_mean", "infer_power_W_mean",
                "pj_per_sop_mean", "pj_per_neuron_spike_mean",
                "train_energy_J_mean", "train_write_energy_J_std"):
        assert key in low
    # same network and spikes; the array read energy scales with V_read^2
    assert low["test_acc_mean"] == high["test_acc_mean"]
    assert high["pj_per_sop_array_mean"] == pytest.approx(
        4 * low["pj_per_sop_array_mean"])
    assert high["infer_energy_J_mean"] > low["infer_energy_J_mean"]
    flagged = result["power"]["placeholders"]
    assert "p_leak" in flagged and "e_wta" in flagged
    assert not {"v_read", "t_read", "e_spike"} & set(flagged)
    run = result["runs"][0]
    assert run["energy_eval"]["placeholders"] == flagged
    assert set(run["energy_train"]["energy_J"]) == {
        "array_read", "array_write", "neurons", "peripherals", "static", "total"}
    header = (tmp_path / "summary.csv").read_text().splitlines()[0]
    assert "pj_per_sop_mean" in header and "test_acc_mean" in header
    assert (tmp_path / "energy" / "v_read_0.1_seed1.npz").exists()
    json.loads((tmp_path / "results.json").read_text())


def test_power_config_is_validated():
    with pytest.raises(ValueError):
        experiments.resolve({"power": {"v_reed": 0.1}})
    cfg = experiments.resolve({})
    experiments.set_path(cfg, "power.e_spike", 1e-12)
    assert cfg["power"]["e_spike"] == 1e-12
