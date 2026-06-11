import sys
sys.path.insert(0, ".")
from ecfet.model_v2 import EcfetV2, V2Params
from ecfet import simulate, Waveform


def single(params, amp):
    width = 5e-3; tail = 1.0; t0c = 10e-3
    m = EcfetV2(params)
    wf = Waveform([(t0c, width, amp)])
    r = simulate(m, wf, t0c + width + tail, label=m.name)
    return (r.G[-1] - r.G[0]) * 1e6


def sweep(params):
    width = 5e-3; tail = 1.0; t0c = 10e-3
    amp_pre, amp_post = -100e-12, 100e-12
    dts = [-1.0, -0.5, -0.2, -0.05, -0.02, -0.0051,
           0.0051, 0.02, 0.05, 0.2, 0.5, 1.0]
    out = []
    for dt in dts:
        m = EcfetV2(params)
        pre_t = t0c + max(0.0, -dt)
        post_t = pre_t + dt
        wf = Waveform([(pre_t, width, amp_pre), (post_t, width, amp_post)])
        t_stop = max(pre_t, post_t) + width + tail
        r = simulate(m, wf, t_stop, label=m.name)
        out.append((dt * 1e3, (r.G[-1] - r.G[0]) * 1e6))
    return out


p = V2Params()
print("single pre(-100pA) net dG uS = %.4f" % single(p, -100e-12))
print("single post(+100pA) net dG uS = %.4f" % single(p, 100e-12))
print("pair sweep dt_ms -> dG_uS:")
rows = sweep(p)
base = 0.5 * (rows[0][1] + rows[-1][1])
for dtms, dg in rows:
    print("  %+8.3f  raw %+9.4f   baseline-sub %+9.4f" % (dtms, dg, dg - base))
print("baseline (mean of tails) = %.4f uS" % base)
