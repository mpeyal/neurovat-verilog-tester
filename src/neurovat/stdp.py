"""Device-local plasticity: the recent-pre eligibility rule and its
consolidation into amplitude-scaled programming pulses."""
import numpy as np

from .crossbar import trace_matrix

MIN_PULSE = 1e-3      # pulses weaker than this fraction of full scale are skipped


def eligibility(pre_raster, post_raster, decay):
    """Accumulate the pre trace at every post spike.

    Returns (acc, npost): ``acc[j, i]`` is the summed trace of input i over
    the spikes of neuron j, ``npost[j]`` that neuron's spike count."""
    post = np.asarray(post_raster, float)
    return post.T @ trace_matrix(pre_raster, decay), post.sum(0)


def consolidation(acc, npost, S, learn_cap):
    """One write decision per synapse of every neuron that fired.

    Inputs whose mean trace at the neuron's spikes is >= offset potentiate,
    quieter ones are heterosynaptically depressed. Returns
    (mask, potentiate, strength, gain): which synapses to write, in which
    direction, how hard in [0, 1], and the per-row charge gain."""
    off = max(S.offset, 1e-6)
    fired = npost >= 1
    mean_pre = acc / np.maximum(npost, 1.0)[:, None]
    potentiate = mean_pre >= off
    strength = np.where(potentiate,
                        S.a_plus * (mean_pre - off) / (1.0 - off),
                        S.a_minus * (off - mean_pre) / off)
    mask = fired[:, None] & (potentiate | (S.a_minus > 0))
    gain = np.minimum(learn_cap, npost)[:, None]
    return mask, potentiate, strength, gain


def pulses(mask, potentiate, strength, gain, S, pot_sign):
    """Write decisions -> (rows, cols, signed amplitudes, effective strengths).
    Row-major order, so a per-device model sees the same sequence the scalar
    engine produced."""
    s = np.clip(strength, 0.0, 1.0) * gain
    rows, cols = np.nonzero(mask & (s > MIN_PULSE))
    pot = np.broadcast_to(potentiate, s.shape)[rows, cols]
    s = s[rows, cols]
    amp = np.where(pot, S.pot_amp, S.dep_amp) * s
    return rows, cols, np.where(pot, pot_sign, -pot_sign) * amp, pot, s
