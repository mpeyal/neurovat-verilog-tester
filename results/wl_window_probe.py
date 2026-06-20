"""Measure the STDP learning-window width vs device geometry (w, l) for v3.

Window width metrics, per geometry:
  * tau1 = w^2/2D, tau2 = l^2/2D  (fast core), tau3 = 19 s (fixed tail)
  * HWHM : dt where the window falls to 50% of its dt->0 peak
  * w10  : dt where it falls to 10% of peak
  * tail level at dt = 200 ms / 1 s (fraction of peak) -> the fixed c3 floor
The window itself is computed by the REAL analysis layer (stdp_sweep), so it
includes the single-pulse-subtraction the GUI uses.
"""
import numpy as np
from ecfet.model_v3 import EcfetV3, V3Params
from vatester.analysis import stdp_sweep

D = 3.6e-10
# fine dt grid (positive side of the anti-symmetric window), seconds
dts = np.concatenate([np.linspace(1e-3, 0.5, 60), np.linspace(0.5, 20.0, 80)])
amp = 50e-12
width = 10e-3

def window_metrics(w, l):
    p = V3Params.paper(w=w, l=l)
    m = EcfetV3(p)
    curves = stdp_sweep([m], amp_pre=-amp, amp_post=+amp, width=width,
                        dts=list(dts), tail=2.0)
    y = np.array(curves[m.name])
    peak = y[0]
    yn = y / peak
    def cross(frac):
        below = np.where(yn <= frac)[0]
        return dts[below[0]] if len(below) else np.inf
    tau1 = w*w/(2*D); tau2 = l*l/(2*D)
    return dict(w=w, l=l, tau1=tau1, tau2=tau2, peak_uS=peak,
                hwhm=cross(0.5), w10=cross(0.10),
                tail_200ms=float(np.interp(0.2, dts, yn)),
                tail_1s=float(np.interp(1.0, dts, yn)))

print(f"{'w(um)':>6} {'l(um)':>6} {'t1(ms)':>8} {'t2(ms)':>8} "
      f"{'peak(uS)':>9} {'HWHM(ms)':>9} {'w10(ms)':>9} {'@200ms':>8} {'@1s':>7}")
geoms = [(4e-6, 15e-6),   # paper default
         (4e-6, 8e-6),
         (2e-6, 4e-6),
         (1e-6, 2e-6),
         (0.5e-6, 1e-6),
         (0.2e-6, 0.4e-6)]
for w, l in geoms:
    r = window_metrics(w, l)
    hwhm = r['hwhm']*1e3; w10 = r['w10']*1e3
    print(f"{w*1e6:6.2f} {l*1e6:6.2f} {r['tau1']*1e3:8.1f} {r['tau2']*1e3:8.1f} "
          f"{r['peak_uS']:9.3f} {hwhm:9.1f} {w10:9.1f} "
          f"{r['tail_200ms']*100:7.1f}% {r['tail_1s']*100:6.1f}%")
