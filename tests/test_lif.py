"""Leaky integrate-and-fire dynamics: single neuron and in-network."""
import numpy as np

from neurovat import NeuronParams
from neurovat.lif import fi_curve, probe_lif, probe_lif_wave


def test_subthreshold_membrane_follows_the_analytic_charging_curve():
    N = NeuronParams(tau_m=20.0, tau_theta=0.0)
    drive, dt = 0.8, 1.0                          # below threshold 1.0
    _t, V, _th, spikes = probe_lif(N, drive, 200.0, dt)
    k = np.arange(1, len(V) + 1)
    np.testing.assert_allclose(V, drive * (1 - (1 - dt / N.tau_m) ** k), rtol=1e-12)
    assert spikes == []
    assert abs(V[-1] - drive) < 1e-3              # settles at v_rest + drive


def test_constant_drive_fires_at_the_analytic_rate():
    N = NeuronParams(tau_m=20.0, t_refractory=5.0, tau_theta=0.0, theta_plus=0.0)
    drive, dt = 2.0, 1.0
    _t, _V, _th, spikes = probe_lif(N, drive, 1000.0, dt)
    # steps to threshold from reset: smallest n with drive(1-(1-dt/tau)^n) >= thr
    n = int(np.ceil(np.log(1 - N.v_threshold / drive) / np.log(1 - dt / N.tau_m)))
    isi = np.diff(spikes)
    assert np.all(isi == (n + N.t_refractory / dt) * dt)
    assert spikes[0] == (n - 1) * dt


def test_refractory_period_holds_the_membrane_at_reset():
    N = NeuronParams(t_refractory=7.0, tau_theta=0.0, theta_plus=0.0)
    t, V, _th, spikes = probe_lif(N, 3.0, 100.0, 1.0)
    for s in spikes[:-1]:
        k = int(s)
        assert np.all(V[k + 1:k + 8] == N.v_reset)


def test_adaptive_threshold_rises_per_spike_and_slows_firing():
    base = dict(tau_m=20.0, t_refractory=2.0)
    fixed = probe_lif(NeuronParams(theta_plus=0.0, **base), 2.0, 600.0, 1.0)[3]
    _t, _V, TH, adapt = probe_lif(
        NeuronParams(theta_plus=0.2, tau_theta=400.0, **base), 2.0, 600.0, 1.0)
    assert len(adapt) < len(fixed)
    assert TH[int(adapt[0]) + 1] > TH[int(adapt[0])]
    assert np.diff(adapt)[-1] > np.diff(adapt)[0]


def test_wave_probe_agrees_with_constant_probe_and_fi_curve_is_monotonic():
    N = NeuronParams()
    a = probe_lif(N, 1.8, 150.0, 1.0)
    b = probe_lif_wave(N, np.full(150, 1.8), 1.0)
    np.testing.assert_allclose(a[1], b[1])
    assert a[3] == b[3]
    drives, rates = fi_curve(N, 300.0, 1.0, 4.0)
    assert np.all(np.diff(rates) >= 0) and rates[0] == 0 and rates[-1] > 0


def test_network_neuron_matches_the_single_neuron_probe(make_net):
    """One output neuron under teacher drive only (silent input) follows
    probe_lif exactly: same leak, threshold adaptation, reset, refractory."""
    net = make_net(n_out=1, mode="supervised", max_rate_hz=0.0)
    res = net.present(np.zeros(16), target=0, learn=False, clean=True)
    _t, V, _th, spikes = probe_lif(net.N, net.N.teacher, 60.0, 1.0)
    assert res["out_spikes"][0] == spikes and spikes
    fired = np.zeros(60, bool)
    fired[np.array(spikes, int)] = True
    np.testing.assert_allclose(res["v_trace"][~fired, 0], V[~fired], rtol=1e-12)
    assert np.all(res["v_trace"][fired, 0] == net.N.v_spike_peak)


def test_winner_take_all_allows_one_output_spike_per_step(make_net):
    net = make_net(n_out=6, neuron={"epsp_gain": 60.0, "inhibition": 0.9,
                                    "t_refractory": 0.0})
    res = net.present(np.ones(16), learn=False, clean=True)
    times = [t for ts in res["out_spikes"] for t in ts]
    assert times and len(times) == len(set(times))

    free = make_net(n_out=6, neuron={"epsp_gain": 60.0, "inhibition": 0.0,
                                     "t_refractory": 0.0})
    res = free.present(np.ones(16), learn=False, clean=True)
    times = [t for ts in res["out_spikes"] for t in ts]
    assert len(times) > len(set(times))            # identical neurons co-fire
