"""Crossbar read path: weight normalisation, spike traces and the
matrix-vector product that turns pre-synaptic activity into post-synaptic
drive. Whole presentations are one matrix product, not a per-step loop."""
import numpy as np


def normalize(G, g_min, g_max):
    """Conductance (S) -> synaptic efficacy in [0, 1] over the device range."""
    span = (g_max - g_min) or 1.0
    return np.clip((G - g_min) / span, 0.0, 1.0)


def mvm(W, X):
    """Crossbar matrix-vector product. ``W`` is (n_post, n_pre); ``X`` is
    (n_pre,) or (..., n_pre) activity. Returns (..., n_post)."""
    return X @ W.T


def trace_matrix(S, decay):
    """Exponential spike trace of a (T, n) raster, for all steps at once.

    A spike sets the trace to 1; it is multiplied by ``decay`` after every
    step. Row t is the trace as seen AT step t (spikes of step t included)."""
    S = np.asarray(S, bool)
    t = np.arange(S.shape[0])[:, None]
    last = np.maximum.accumulate(np.where(S, t, -1), axis=0)
    return np.where(last >= 0, float(decay) ** (t - np.maximum(last, 0)), 0.0)


def input_drive(Wn, X, exc, inh, epsp_gain, ipsp_gain, gain=1.0):
    """Activity-normalised drive of the input crossbar for every step.

    ``X`` is the (T, n_pre) synaptic trace; excitatory afferents (``exc``)
    add EPSPs and the fixed inhibitory ones (``inh``) subtract IPSPs."""
    signed = X * (epsp_gain * exc - ipsp_gain * inh)
    return gain * mvm(Wn, signed) / (X.sum(-1) + 1.0)[..., None]
