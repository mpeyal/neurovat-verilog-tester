"""Brian2 hand port of the ECFET v3 synapse (simulation_core/model_v3.py,
verilog/ecfet_v3.va).

Continuous part as Brian2 ODEs (from the .va ``idt`` lines):
    dx/dt  = u_x - (x - x_eq)/tau_ret,   u_x = sgn |I_eff| eta / Q_full
             (u_x = 0 at a rail it pushes against)
    dv_i/dt = c_i kappa/(1-kappa) span S'(x) u_x - v_i / tau_i     (3 EDL pools)
    dtr/dt = -tr / tau_i                (6 STDP eligibility traces, p and d)
    G = clip(Gmin + span S(x) + v1 + v2 + v3 + lock, Gmin, Gmax)
Discrete part (pulse ONSET, |I| crossing I_drift_th) as run_regularly code
at the start of every step, before the integration over that step (the
twin's order): harvest the opposite trace into ``lock`` (+A_stdp sum td on a
potentiating onset, -A_stdp sum tp on a depressing one), consume it, set the
own trace to its single-pulse weight, clamp lock so G_nv + lock stays in
[Gmin, Gmax].

Drive: the stimulus is fed as a TimedArray holding the step-AVERAGED gate
current (exact charge per step); every gateA-1 edge but the random train's
is on the 0.1 ms grid.
"""
import math
import time

import numpy as np

import brian2 as b2

from simulation_core import V3Params

b2.prefs.codegen.target = "numpy"
b2.BrianLogger.log_level_error()

EQS = """
I : 1
sgn : 1
ux : 1
sg = 1 / (1 + exp(-k * (x - 0.5))) : 1
Sx = (sg - sa) / sden : 1
dSdx = k * sg * (1 - sg) / sden : 1
dx/dt = (ux - (x - x_eq) / tau_ret) / second : 1
dv1/dt = (cn1 * ef * span * dSdx * ux - v1 / tau1) / second : 1
dv2/dt = (cn2 * ef * span * dSdx * ux - v2 / tau2) / second : 1
dv3/dt = (cn3 * ef * span * dSdx * ux - v3 / tau3) / second : 1
dtp1/dt = -tp1 / (tau1 * second) : 1
dtp2/dt = -tp2 / (tau2 * second) : 1
dtp3/dt = -tp3 / (tau3 * second) : 1
dtd1/dt = -td1 / (tau1 * second) : 1
dtd2/dt = -td2 / (tau2 * second) : 1
dtd3/dt = -td3 / (tau3 * second) : 1
lock : 1
prev : 1
Gnv = Gmin + span * Sx : 1
G = clip(Gnv + v1 + v2 + v3 + lock, Gmin, Gmax) : 1
"""

ONSET = """
I = drive(t)
x = clip(x, 0, 1)
Ieff = I * int(abs(I) >= Ith) + leak * I * int(abs(I) < Ith)
sgn = -polarity * (int(Ieff > 0) - int(Ieff < 0))
ux = sgn * abs(Ieff) / Q_full * eta
ux = ux * (1 - int(x >= 1 and ux > 0)) * (1 - int(x <= 0 and ux < 0))
on = int(abs(I) >= Ith)
onset = on * (1 - prev)
pot = int(sgn > 0)
lock += onset * A * (pot * (td1 + td2 + td3) - (1 - pot) * (tp1 + tp2 + tp3))
tp1 = (1 - onset) * tp1 + onset * pot * w1
tp2 = (1 - onset) * tp2 + onset * pot * w2
tp3 = (1 - onset) * tp3 + onset * pot * w3
td1 = (1 - onset) * td1 + onset * (1 - pot) * w1
td2 = (1 - onset) * td2 + onset * (1 - pot) * w2
td3 = (1 - onset) * td3 + onset * (1 - pot) * w3
lock = lock + onset * (clip(lock, Gmin - Gnv, Gmax - Gnv) - lock)
prev = on
"""


def step_average(pulses, t_stop, dt):
    """Gate current averaged over each [n dt, (n+1) dt] (A), exact charge."""
    n = int(math.ceil(t_stop / dt - 1e-9))
    edges = np.arange(n + 1) * dt
    q = np.zeros(n + 1)
    for t0, w, a in pulses:
        q += a * np.clip(edges - t0, 0.0, w)
    return np.diff(q) / dt


def run(stim, p=None, dt=1e-4, method="rk4"):
    """Simulate one gateA-1 stimulus. Returns (t_s, G_S, wall_s)."""
    p = p or V3Params()
    b2.start_scope()
    b2.defaultclock.dt = dt * b2.second
    I = step_average(stim.pulses, stim.t_stop, dt)
    drive = b2.TimedArray(I, dt=dt * b2.second)
    t1, t2, t3 = p.taus
    cn = p.pool_weights
    ws = p.stdp_weights
    sa, sb = 1 / (1 + math.exp(0.5 * p.k)), 1 / (1 + math.exp(-0.5 * p.k))
    eta = 1.0
    if p.kinetic_on:
        eta = p.eta_inf + (1 - p.eta_inf) * (1 - math.exp(-p.t_on / p.tau_kin))
    ns = dict(drive=drive, Ith=p.I_drift_th, leak=p.leak_drift_scale, k=p.k,
              sa=sa, sden=sb - sa, polarity=p.polarity, Q_full=p.Q_full, eta=eta,
              x_eq=p.x_eq, tau_ret=p.tau_ret, tau1=t1, tau2=t2, tau3=t3,
              cn1=cn[0], cn2=cn[1], cn3=cn[2],
              ef=p.kappa_v / max(1 - p.kappa_v, 1e-9), span=p.Gmax - p.Gmin,
              Gmin=p.Gmin, Gmax=p.Gmax, A=p.A_stdp, w1=ws[0], w2=ws[1], w3=ws[2])
    grp = b2.NeuronGroup(1, EQS, method=method, namespace=ns)
    grp.x = p.x_of_G(min(max(1 / p.Rinit, p.Gmin), p.Gmax))
    grp.run_regularly(ONSET, when="start", order=0)
    mon = b2.StateMonitor(grp, "G", record=0, when="start", order=1)
    net = b2.Network(grp, mon)
    t0 = time.perf_counter()
    net.run(len(I) * dt * b2.second)
    wall = time.perf_counter() - t0
    return np.asarray(mon.t / b2.second), np.asarray(mon.G[0]), wall


def source_lines():
    """Lines of code of the port's model definition (equations + event code
    + the run() wiring), excluding comments/docstrings/blank lines."""
    import inspect
    src = (EQS + ONSET).splitlines() + inspect.getsource(run).splitlines() \
        + inspect.getsource(step_average).splitlines()
    n = 0
    in_doc = False
    for line in src:
        s = line.strip()
        if s.startswith('"""'):
            in_doc = not (s.count('"""') == 2) and not in_doc
            continue
        if in_doc or not s or s.startswith("#"):
            continue
        n += 1
    return n
