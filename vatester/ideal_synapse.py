"""Python equation twin of verilog/ideal_signed_synapse.va.

Weight storage/programming is external. This is an active signed
transconductance primitive, not a passive memristor or ECFET device model.
"""
import numpy as np


def branch_current(pre_voltage, weight=0., g_scale=1e-6, control_voltage=None):
    if not np.isfinite(g_scale) or g_scale <= 0:
        raise ValueError("g_scale must be positive and finite")
    w = weight if control_voltage is None else control_voltage
    return -g_scale*np.asarray(w)*np.asarray(pre_voltage)
