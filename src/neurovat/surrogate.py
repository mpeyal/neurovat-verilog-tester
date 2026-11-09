"""Surrogate-gradient BPTT through a stack of crossbars.

Soft-reset LIF layers, fast-sigmoid surrogate 1/(1 + beta|u - thr|)^2 and a
softmax cross-entropy readout on the mean output membrane."""
import math

import numpy as np

from .crossbar import mvm


def forward(S_in, Wn, gains, alpha, thr):
    """Forward pass. ``S_in`` is (T, n_in) or a batch (B, T, n_in) of 0/1
    spikes. Returns (spk, cache): ``spk[c]`` the spikes of layer c, and per
    crossbar ``(u, norm, pre)`` with ``u`` the pre-reset membrane."""
    spk, cache = [S_in], []
    for c, W in enumerate(Wn):
        pre = spk[c]
        norm = pre.sum(-1) + 1.0
        I = gains[c] * mvm(W, pre) / norm[..., None]
        T = I.shape[-2]
        v = np.zeros(I.shape[:-2] + I.shape[-1:])
        u = np.empty(I.shape)
        s = np.empty(I.shape)
        for t in range(T):
            uu = alpha * v + I[..., t, :]
            sp = (uu >= thr).astype(float)
            v = uu - thr * sp                 # soft reset
            u[..., t, :] = uu
            s[..., t, :] = sp
        spk.append(s)
        cache.append((u, norm, pre))
    return spk, cache


def loss_and_grads(cache, Wn, gains, alpha, thr, beta, target):
    """Cross-entropy loss and dL/dWn for one (unbatched) sample.

    Yields nothing; returns (loss, grads) with ``grads[c]`` shaped like
    ``Wn[c]``."""
    K = len(Wn)
    u_out = cache[K - 1][0]
    T = u_out.shape[0]
    logit = u_out.mean(0)
    z = logit - logit.max()
    p = np.exp(z)
    normalizer = float(p.sum())               # >= 1 after subtracting max(logit)
    loss = float(math.log(normalizer) - z[target])
    p = p / normalizer
    g_logit = p.copy(); g_logit[target] -= 1.0
    g_read = np.tile(g_logit / T, (T, 1))     # dL/du_out[t] (readout)
    grads = [None] * K
    gS = None                                 # spike grad from the layer above
    for c in range(K - 1, -1, -1):
        u, norm, pre = cache[c]
        sg = 1.0 / (1.0 + beta * np.abs(u - thr)) ** 2
        dI = np.empty_like(u)
        dv_next = np.zeros(u.shape[1])
        for t in range(T - 1, -1, -1):
            if c == K - 1:                    # output: membrane readout
                du = g_read[t] + dv_next * (1.0 - thr * sg[t])
            else:                             # hidden: spike path from above
                du = gS[t] * sg[t] + dv_next * (1.0 - thr * sg[t])
            dI[t] = du
            dv_next = du * alpha
        scaled = gains[c] * dI / norm[:, None]
        grads[c] = scaled.T @ pre
        if c > 0:
            gS = scaled @ Wn[c]
            current = gains[c] * (pre @ Wn[c].T) / norm[:, None]
            gS -= (dI * current).sum(1)[:, None] / norm[:, None]
    return loss, grads
