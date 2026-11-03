"""Shared desktop labels, palette aliases and display units.

Keep these dependency-light: UI components import this module, never app.py.
Canonical units match the existing simulator and trainer interfaces.
"""
from . import design_system as ds

APP_TITLE = "NeuroVAT - Neuromorphic Verilog-A Tester"
LEFT_W, RIGHT_W = 350, 312
BUBBLE_W = RIGHT_W - 44
BUBBLE_INDENT = 26

# ---------------------------------------------------------------- palette --
C_BG      = ds.BG
C_SURF    = ds.SURFACE
C_SURF2   = ds.SURFACE_2
C_SURF3   = ds.SURFACE_3
C_BORDER  = ds.BORDER
C_ACC     = ds.ACCENT
C_ACC_H   = ds.ACCENT_HOVER
C_ACC_A   = ds.ACCENT_ACTIVE
C_TEXT    = ds.TEXT
C_TEXT2   = ds.TEXT_SECONDARY
C_MUTED   = ds.TEXT_MUTED
C_GREEN   = ds.POSITIVE
C_AMBER   = ds.WARNING
C_RED     = ds.NEGATIVE
C_AGENT   = (255, 178, 102)

C_BUB_USER   = (37, 50, 86)
C_BUB_USER_B = (66, 88, 148)
C_BUB_AGENT  = (30, 34, 43)
C_BUB_AGENT_B = (50, 56, 70)
C_BUB_ERR    = (62, 30, 36)
C_BUB_ERR_B  = (130, 56, 64)

# ---- selectable display units --------------------------------------------
# Each dimension has an ordered unit list; the CANONICAL unit (factor 1.0) is the
# one the engine expects (pA for current, V for voltage, ms for time, Hz for
# frequency).  A field stores its number in the currently-selected unit; reads go
# through _nt_getu() which multiplies by the factor below to get the canonical
# value, and switching a unit rescales the number so the physical value is fixed.
UNIT_SETS = {
    "current": ["pA", "nA", "uA", "mA", "A", "kA", "MA"],
    "voltage": ["uV", "mV", "V", "kV"],
    "time":    ["ns", "us", "ms", "s"],
    "freq":    ["mHz", "Hz", "kHz", "MHz"],
}
UNIT_FACTOR = {
    "pA": 1.0, "nA": 1e3, "uA": 1e6, "mA": 1e9, "A": 1e12, "kA": 1e15, "MA": 1e18,
    "uV": 1e-6, "mV": 1e-3, "V": 1.0, "kV": 1e3,
    "ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3,
    "mHz": 1e-3, "Hz": 1.0, "kHz": 1e3, "MHz": 1e6,
}
UNIT_DIM = {u: dim for dim, us in UNIT_SETS.items() for u in us}
CANON_UNIT = {"current": "pA", "voltage": "V", "time": "ms", "freq": "Hz"}

# STDP tab stimulus modes (index 1 = analysis.stdp_sweep_overlap)
STDP_MODES = ("rectangular pulses", "overlapping spikes")
# measurements offered by the Polarization tab.  Device classes without a
# GUI_PROFILE only get the P-V loop; a ferroelectric capacitor twin lists all.
POLAR_MEASUREMENTS = ("P-V loop", "I-V switching current", "C-V (dQ/dV)",
                      "PUND", "Switching kinetics")
POLAR_DESC = {
    "P-V loop": "Triangular voltage sweep; polarization vs voltage over the last "
                "(settled) cycle. Remanence at V = 0, coercive voltage at the "
                "zero crossings. A faster sweep widens the loop (switching is "
                "rate dependent).",
    "I-V switching current": "Same triangular sweep, plotted as the displacement "
                "current dQ/dt vs voltage. The switching-current peaks sit at the "
                "coercive voltage; their area is the switched charge.",
    "C-V (dQ/dV)": "Same sweep, plotted as the large-signal capacitance dQ/dV vs "
                "voltage (butterfly). The peaks mark switching; the flat level "
                "away from them is the dielectric capacitance, which is mostly "
                "what a small-signal C-V meter reports.",
    "PUND": "Negative preset, then Positive, Up, Negative, Down pulses. P - U is "
            "the switched polarization (2*Pr); dielectric and leakage charge "
            "cancel. Plot: the current response. Numbers: log.",
    "Switching kinetics": "Preset fully negative, apply ONE pulse per width and "
            "amplitude, read the switched fraction at zero bias. Plotted vs "
            "log10(width): position, spread and slope calibrate tau0/Ea/alpha, "
            "sigma_eta and beta. t50 per amplitude goes to the log.",
}
# widgets whose values belong to a device class: saved when you leave a class
# and restored when you come back, so FeCAP volts never leak into ECFET pA runs
CLASS_SETTING_TAGS = (
    "gen_combo", "unit_combo", "delay_ms", "tail_input", "custom_text",
    "stdp_mode", "stdp_amp_pre", "stdp_amp_post", "stdp_width_ms",
    "stdp_range_ms", "stdp_settle_ms", "stdp_npts", "stdp_tail_ratio",
    "stdp_tail_tau_ms", "polar_meas", "polar_vamp", "polar_period",
    "polar_cycles", "polar_pts", "pund_amp", "pund_width_ms", "pund_delay_ms",
    "kin_amps", "kin_wmin_s", "kin_wmax_s", "kin_npts",
    "stdp_shape", "stdp_r_kohm", "stdp_c_nf", "stdp_preview_dt",
    "stdp_measurement", "stdp_include_rule")
