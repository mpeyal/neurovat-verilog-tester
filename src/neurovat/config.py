"""Network, neuron and plasticity parameters."""

from dataclasses import dataclass


@dataclass
class NeuronParams:
    """Leaky integrate-and-fire output neuron + post-synaptic potential tuning.
    Membrane potential is in arbitrary units with v_rest = 0 and threshold 1,
    so the gains below are what you actually tune."""
    tau_m: float = 20.0        # membrane time constant (ms)
    v_threshold: float = 1.0   # fire when v >= threshold (+ adaptive theta)
    v_reset: float = 0.0       # reset potential after a spike
    v_rest: float = 0.0        # leak target
    t_refractory: float = 5.0  # refractory period (ms)
    v_spike_peak: float = 1.6  # value drawn into the membrane trace ON the step a
    #                            neuron fires, so the spike shows AT the threshold
    #                            crossing (the reset + refractory floor then follow)
    tau_syn: float = 8.0       # synaptic (PSP) trace decay (ms)

    epsp_gain: float = 11.0    # excitatory PSP weight (EPSP height); drive is
    #                            activity-normalised, so this is grid-independent
    ipsp_gain: float = 1.0     # inhibitory PSP weight (IPSP depth)
    inhibition: float = 0.9    # lateral winner-take-all strength (0 = none)

    theta_plus: float = 0.06   # homeostatic threshold bump per output spike
    tau_theta: float = 180.0   # adaptive-threshold decay (ms); 0 disables

    teacher: float = 1.4       # supervised teacher drive onto the target neuron
    v_noise: float = 0.0       # membrane (thermal/neuronal) noise sigma per step
    hidden_gain: float = 1.7   # extra drive on HIDDEN layers (they have no
    #                            teacher, so they need a boost to fire on their
    #                            own and pass a code to the next crossbar)


@dataclass
class STDPParams:
    """Recent-pre eligibility rule that programs the device synapses.

    At each postsynaptic spike, accumulate the presynaptic trace. At the end
    of a presentation, inputs above offset potentiate and those below it
    depress. Writes are consolidated into amplitude-scaled pulses. This is
    an algorithmic local rule, not a physical waveform-overlap STDP circuit.
    Amplitudes are SI (A or V), pulse_width is seconds, tau_pre is milliseconds.
    """
    a_plus: float = 1.0        # potentiation learning rate (scales pulse amp)
    a_minus: float = 1.0       # (heterosynaptic) depression learning rate
    tau_pre: float = 20.0      # pre eligibility-trace decay (ms)
    offset: float = 0.25       # pre-trace split: above -> LTP, below -> LTD
    pot_amp: float = 50e-12    # SI amplitude of a full potentiating pulse
    dep_amp: float = 50e-12    # SI amplitude of a full depressing pulse
    pulse_width: float = 10e-3 # programming pulse width (s)
    pot_sign: float = 0.0      # +1/-1 drive polarity that potentiates;
    #                            0 = auto-calibrate from the device
    sg_lr: float = 0.04        # surrogate-gradient learning rate (BPTT step)
    sg_beta: float = 5.0       # surrogate steepness (fast-sigmoid derivative)


@dataclass
class NetConfig:
    grid_h: int = 5
    grid_w: int = 5
    n_out: int = 4
    hidden_layers: tuple = ()  # sizes of HIDDEN LIF layers between input and
    #                            output, e.g. (8,) or (8, 4); each consecutive
    #                            pair of layers is joined by its own crossbar
    mode: str = "supervised"   # "supervised" | "unsupervised"
    learn_rule: str = "stdp"   # "stdp" (device-local) | "surrogate" (BPTT,
    #                            supervised, gradient applied through the device)
    present_ms: float = 120.0  # duration of one pattern presentation
    dt_ms: float = 1.0         # integration step
    max_rate_hz: float = 160.0 # peak input spike rate for a fully-on pixel
    seed: int = 1
    pattern_set: str = "bars"  # which built-in pattern bank to train on

    # ---- spike encoding + input noise (front-end "sensor" realism) ----
    encoding: str = "rate"     # "rate" (Poisson) | "latency" (time-to-first
    #                            spike: brighter pixel fires earlier) | "temporal"
    #                            (phase code: one precisely-timed spike per cycle,
    #                            phase set by intensity - repeating spike times)
    #                            | "scan" (column scan: one input per pixel ROW,
    #                            one time bin per pixel COLUMN, a spike where
    #                            the pixel is inked - a spatio-temporal pattern)
    scan_bin_ms: float = 10.0  # "scan": duration of one column
    scan_threshold: float = 0.5  # "scan": a pixel at or above this is inked
    bg_rate_hz: float = 0.0    # spontaneous background rate on EVERY input (Hz)
    input_noise: float = 0.0   # per-presentation pixel noise (Gaussian + salt
    #                            & pepper), as a fraction in [0, 1]
    signal_frac: float = 1.0   # fraction of afferents carrying the REAL pattern;
    #                            the rest fire only background noise (a FIXED
    #                            random subset, the embedded-pattern paradigm)
    jitter_ms: float = 0.0     # Gaussian temporal jitter (sigma) on every spike
    pattern_ms: float = 0.0    # how long the PATTERN is shown within present_ms
    #                            (centred); the rest of the window is background-
    #                            noise only. 0 or >= present_ms = full window.
