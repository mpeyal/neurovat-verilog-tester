"""Leaky integrate-and-fire dynamics: single-neuron probes used for f-I
curves and the same update rule the network applies per layer."""
import math

import numpy as np


def probe_lif(N, drive, present_ms, dt):
    """Simulate ONE leaky integrate-and-fire output neuron driven by a CONSTANT
    `drive` for present_ms (no crossbar, no weights) - exactly the network's
    output-neuron dynamics (leak toward v_rest+drive, fire at threshold + the
    adaptive homeostatic theta, reset, refractory).  Returns
    (t_ms, V, threshold_trace, spike_times_ms) so the GUI can plot the sawtooth.
    """
    n = max(1, int(round(present_ms / dt)))
    leak = dt / max(N.tau_m, 1e-6)
    d_theta = math.exp(-dt / N.tau_theta) if N.tau_theta > 0 else 0.0
    v, theta, refr = N.v_rest, 0.0, 0.0
    t = np.arange(n) * dt
    V = np.empty(n)
    TH = np.empty(n)
    spikes = []
    rng = np.random.default_rng(12345)
    for k in range(n):
        active = refr <= 0
        if active:
            v += leak * (N.v_rest - v + drive)
            if N.v_noise > 0:
                v += rng.normal(0.0, N.v_noise)
        else:
            v = N.v_reset
            refr -= dt
        if N.tau_theta > 0:
            theta *= d_theta
        V[k] = v
        TH[k] = N.v_threshold + theta
        if active and v >= N.v_threshold + theta:
            spikes.append(k * dt)
            v = N.v_reset
            refr = N.t_refractory
            theta += N.theta_plus
    return t, V, TH, spikes


def probe_lif_wave(N, drive, dt):
    """LIF output neuron driven by a PER-STEP `drive` array - lets the bench show
    a constant step, a rectangular pulse or a ramp with the same dynamics as
    probe_lif().  Returns (t_ms, V, threshold_trace, spike_times_ms)."""
    drive = np.asarray(drive, float)
    n = len(drive)
    leak = dt / max(N.tau_m, 1e-6)
    d_theta = math.exp(-dt / N.tau_theta) if N.tau_theta > 0 else 0.0
    v, theta, refr = N.v_rest, 0.0, 0.0
    t = np.arange(n) * dt
    V = np.empty(n)
    TH = np.empty(n)
    spikes = []
    rng = np.random.default_rng(12345)
    for k in range(n):
        active = refr <= 0
        if active:
            v += leak * (N.v_rest - v + drive[k])
            if N.v_noise > 0:
                v += rng.normal(0.0, N.v_noise)
        else:
            v = N.v_reset
            refr -= dt
        if N.tau_theta > 0:
            theta *= d_theta
        V[k] = v
        TH[k] = N.v_threshold + theta
        if active and v >= N.v_threshold + theta:
            spikes.append(k * dt)
            v = N.v_reset
            refr = N.t_refractory
            theta += N.theta_plus
    return t, V, TH, spikes


def fi_curve(N, present_ms, dt, drive_max, n_pts=26):
    """Firing rate (Hz) vs constant drive - the neuron's f-I transfer curve."""
    drives = np.linspace(0.0, drive_max, n_pts)
    rates = np.array([len(probe_lif(N, float(d), present_ms, dt)[3])
                      / (present_ms * 1e-3) for d in drives])
    return drives, rates


def probe_lif_psp(N, in_hz, epsp_w, present_ms, dt):
    """Drive ONE LIF neuron with a regular EXCITATORY spike train (one input
    spike every 1000/in_hz ms).  Each input spike adds `epsp_w` to a synaptic
    PSP trace that decays with tau_syn; the membrane leaks toward that PSP - so
    you SEE each EPSP bump, the leak between them, sub-threshold SUMMATION, and
    the threshold crossing that finally triggers a spike + reset.  Returns
    (t_ms, V, threshold_trace, out_spike_times, in_spike_times, psp_trace)."""
    n = max(1, int(round(present_ms / dt)))
    leak = dt / max(N.tau_m, 1e-6)
    d_syn = math.exp(-dt / max(N.tau_syn, 1e-6))
    d_theta = math.exp(-dt / N.tau_theta) if N.tau_theta > 0 else 0.0
    period = max(1, int(round((1000.0 / max(in_hz, 1e-3)) / dt)))
    v, theta, refr, x = N.v_rest, 0.0, 0.0, 0.0
    t = np.arange(n) * dt
    V = np.empty(n)
    TH = np.empty(n)
    P = np.empty(n)
    outs, ins = [], []
    rng = np.random.default_rng(7)
    for k in range(n):
        x *= d_syn
        if k % period == 0:                # an EPSP arrives
            x += epsp_w
            ins.append(k * dt)
        active = refr <= 0
        if active:
            v += leak * (N.v_rest - v + x)
            if N.v_noise > 0:
                v += rng.normal(0.0, N.v_noise)
        else:
            v = N.v_reset
            refr -= dt
        if N.tau_theta > 0:
            theta *= d_theta
        V[k] = v
        TH[k] = N.v_threshold + theta
        P[k] = x
        if active and v >= N.v_threshold + theta:
            outs.append(k * dt)
            v = N.v_reset
            refr = N.t_refractory
            theta += N.theta_plus
    return t, V, TH, outs, ins, P


def fi_curve_psp(N, epsp_w, present_ms, dt, hz_max, n_pts=24):
    """Output firing rate (Hz) vs INPUT spike rate - the spike transfer curve."""
    hz = np.linspace(1.0, max(hz_max, 2.0), n_pts)
    out = np.array([len(probe_lif_psp(N, float(h), epsp_w, present_ms, dt)[3])
                    / (present_ms * 1e-3) for h in hz])
    return hz, out
