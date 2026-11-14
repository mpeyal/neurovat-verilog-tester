"""NeuroVAT's unsupervised-stream neuron and STDP rule written in Brian2.

The NeuroVAT side is :func:`neurovat.masquelier.run` (ideal synapses,
``n_out = 1``). Its model, read from the source:

* SRM neuron. ``u(t) = K * sum_j w_j (exp(-(t-t_j)/tau_m) - exp(-(t-t_j)/tau_s))
  + T (K1 e^{-s/tau_m} - K2 (e^{-s/tau_m} - e^{-s/tau_s}))`` with ``s`` the time
  since the last output spike (eta term only for ``s < 7 tau_m``); K
  normalises the PSP peak to 1. The sums are kept as two exponentially
  decaying accumulators (exact, no ODE error), sampled at ``sim_dt`` (0.1 ms).
  A pre spike at exact time t_j inside step k adds
  ``w_j exp(-(t_end - t_j)/tau)``, t_end = (k+1) dt.
* Threshold T tested at the END of each step; the output spike is stamped
  t_end. Refractory: the neuron may fire only when ``t_end - t_post > t_ref``
  (strict). A spike clears both PSP accumulators.
* STDP: additive, nearest-spike with strict LTP/LTD alternation per synapse
  (Hathway & Goodman 2018). Post spike: each afferent with a pre spike since
  the previous post gets ``+a_plus exp(-(t_post - t_lastpre)/tau_plus)``. Pre
  spike: if it is the afferent's first since the last post, it gets
  ``-a_minus exp(-(t_pre - t_post)/tau_minus)``. Both windows end at
  7 tau; weights clipped to [0, 1]. Within a step NeuroVAT does
  decay -> LTD of this step's pre spikes -> PSP add with the updated weight
  -> threshold -> LTP -> reset.

Brian2 port. Same equations as ODEs, integrated with ``method='exact'``
(the accumulators and eta factors are linear decays). STDP uses
event-driven traces that are SET to 1 at each spike (nearest-spike) and
gated by open/closed flags (alternation); the 7 tau cut is ``trace >=
exp(-7)``. To reproduce NeuroVAT's within-step order the schedule is
``groups, thresholds, resets, synapses`` with the post pathway before the
pre pathway, so a pre spike in the step right after an output spike is
depressed and survives the PSP reset, as in NeuroVAT. Brian's time-step
convention then maps one-to-one: a Brian output spike at t = n dt is the
NeuroVAT spike stamped t_end = n dt, a Brian input at t = k dt is the
NeuroVAT input of step k.

Input: Brian2's SpikeGeneratorGroup puts each spike on the dt grid and
refuses two spikes of one neuron in one step, so the input is binned at
floor(t / dt) (NeuroVAT's own binning) and same-bin duplicates are dropped.
"""
import math
import time

import numpy as np

import brian2 as b2

from validation.engine.common import bins, dedupe

b2.prefs.codegen.target = "numpy"
b2.BrianLogger.log_level_error()


def _kernel_norm(tm, ts):
    s_pk = tm * ts / (tm - ts) * math.log(tm / ts)
    return 1.0 / (math.exp(-s_pk / tm) - math.exp(-s_pk / ts))


def run(P, S, weights=None, learn=True, dt_ms=None, record_u=False,
        method="exact"):
    """Simulate stream S (one pass, ``P.n_repeats`` ignored) on Brian2.

    Returns dict: post_t_ms (array), weights (final, (n_afferents,)), u
    (per-step potential at the threshold test, steps 1..n) and ready mask
    when ``record_u``, n_input (spikes fed), n_dropped (same-bin
    duplicates), wall_s."""
    if P.n_out != 1:
        raise ValueError("the Brian2 port covers n_out = 1")
    dt = float(dt_ms if dt_ms is not None else P.sim_dt_ms)
    n_base = int(round(S.duration_ms / dt))
    b2.start_scope()
    b2.defaultclock.dt = dt * b2.ms
    k = np.minimum(bins(S.t, dt), n_base - 1)
    keep = dedupe(S.aff, k)
    aff, k = S.aff[keep], k[keep]
    o = np.lexsort((aff, k))
    aff, k = aff[o], k[o]

    ms = b2.ms
    ns = dict(tau_m=P.tau_m * ms, tau_s=P.tau_s * ms, thr=P.threshold,
              K=_kernel_norm(P.tau_m, P.tau_s), K1=P.K1, K2=P.K2,
              t_ref=P.t_ref_ms * ms, eta_win=P.cutoff_tau * P.tau_m * ms,
              a_plus=P.a_plus, a_minus=P.a_minus_ratio * P.a_plus,
              tau_plus=P.tau_plus * ms, tau_minus=P.tau_minus * ms,
              cut=math.exp(-P.cutoff_tau), learn=1.0 if learn else 0.0)
    eqs = """
    dxm/dt = -xm / tau_m : 1
    dxs/dt = -xs / tau_s : 1
    dem/dt = -em / tau_m : 1
    des/dt = -es / tau_s : 1
    tpost : second
    u = K * (xm - xs) + thr * (K1 * em - K2 * (em - es)) * int((t - tpost) < eta_win) : 1
    ready = int((t - tpost) > t_ref + 1e-3 * dt) : 1
    """
    G = b2.NeuronGroup(1, eqs, threshold="ready > 0.5 and u >= thr",
                       reset="xm = 0; xs = 0; em = 1; es = 1; tpost = t",
                       method=method, namespace=ns)
    G.tpost = -1e9 * b2.second
    inp = b2.SpikeGeneratorGroup(P.n_afferents, aff, k * dt * ms)
    syn_eqs = """
    w : 1
    ltp_open : 1
    ltd_open : 1
    dapre/dt = -apre / tau_plus : 1 (event-driven)
    dapost/dt = -apost / tau_minus : 1 (event-driven)
    """
    on_pre = """
    w = clip(w - learn * ltd_open * int(apost >= cut) * a_minus * apost, 0, 1)
    ltd_open = 0
    ltp_open = 1
    apre = 1
    xm_post += w
    xs_post += w
    """
    on_post = """
    w = clip(w + learn * ltp_open * int(apre >= cut) * a_plus * apre, 0, 1)
    ltp_open = 0
    ltd_open = 1
    apost = 1
    """
    syn = b2.Synapses(inp, G, syn_eqs, on_pre=on_pre, on_post=on_post,
                      namespace=ns)
    syn.connect()
    syn.pre.order = 1
    syn.post.order = -1
    w0 = (np.full(P.n_afferents, P.w_init) if weights is None
          else np.asarray(weights, float).reshape(-1))
    syn.w[:] = w0[np.asarray(syn.i[:])]
    spk = b2.SpikeMonitor(G)
    objs = [G, inp, syn, spk]
    mon = None
    if record_u:
        mon = b2.StateMonitor(G, ["u", "ready"], record=0, when="thresholds",
                              order=-1)
        objs.append(mon)
    net = b2.Network(*objs)
    net.schedule = ["start", "groups", "thresholds", "resets", "synapses", "end"]
    t0 = time.perf_counter()
    net.run((n_base + 1) * dt * ms)
    wall = time.perf_counter() - t0
    w = np.zeros(P.n_afferents)
    w[np.asarray(syn.i[:])] = np.asarray(syn.w[:])
    out = {"post_t_ms": np.asarray(spk.t / ms, float), "weights": w,
           "n_input": int(aff.size), "n_dropped": int((~keep).sum()),
           "wall_s": wall, "dt_ms": dt}
    if mon is not None:
        out["u"] = np.asarray(mon.u[0])[1:]
        out["ready"] = np.asarray(mon.ready[0])[1:] > 0.5
    return out


def per_ms_max(u, ready, dt):
    """NeuroVAT's ``record_u`` reduction: the max of u over the ready steps of
    each millisecond, NaN when none is ready. u[k] is the value tested at
    the end of step k."""
    u_per = max(int(round(1.0 / dt)), 1)
    n = -(-u.size // u_per)
    v = np.where(ready, u, -np.inf)
    pad = np.full(n * u_per, -np.inf)
    pad[:v.size] = v
    m = pad.reshape(n, u_per).max(1)
    m[~np.isfinite(m)] = np.nan
    return m
