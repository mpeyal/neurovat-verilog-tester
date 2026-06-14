"""Dynamic loader for user/agent device twins kept OUTSIDE the app source.

Device behavioural models ("twins") for NEW devices live in a separate
top-level `twins/` folder, not in the GUI package (`vatester/`) or the core
engine (`ecfet/`).  Each twin file is a self-contained Python module that
declares a `TWIN_SPEC` dict describing how the GUI should register and drive it:

    # twins/my_device.py
    TWIN_SPEC = {
        "key":          "rram",                 # unique short id
        "label":        "RRAM (my model)",      # shown in the model list
        "device_class": "RRAM",                 # groups it in the DEVICE CLASS
                                                #   selector (new or existing)
        "input_kind":   "voltage",              # "voltage" | "current" drive
        "va_keywords":  ("rram", "reram"),      # map a .va to this twin by name
        "model_class":  MyRRAM,                 # step(t,dt,drive)/.R/.G/reset()/
                                                #   observables()
        "params_class": MyRRAMParams,           # dataclass of parameters
        # optional profile (else sensible defaults): the characterisation state
        "stdp": {"obs": "Vth (V)", "label": "dVt", "unit": "mV", "scale": 1e3},
        "result_plots": [("Vth (V)", "Vth", "mV", 1e3)],
        "analysis_metrics": [("Vth (V)", "Vth", "mV", 1e3)],
        "polar_obs": "P (uC/cm2)",              # enables the Polarization tab
    }

SECURITY NOTE: importing a Python twin EXECUTES it - there is no sandbox.  The
folder separation keeps the GUI/core source from being edited by mistake and
constrains the agent to this directory; it is organisational safety, not a
defence against malicious code.  Only put twins you trust here.
"""

import glob
import importlib.util
import os


def load_twins(twins_dir):
    """Import every twins/*.py and collect their TWIN_SPEC.

    Returns a list of (path, payload, error): payload is (module, spec) on
    success, else None with an error string.  Never raises - a bad twin file is
    reported, not fatal.
    """
    out = []
    if not os.path.isdir(twins_dir):
        return out
    for path in sorted(glob.glob(os.path.join(twins_dir, "*.py"))):
        base = os.path.basename(path)
        if base.startswith("_"):
            continue
        modname = "twins_" + os.path.splitext(base)[0]
        try:
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:                      # noqa: BLE001 - report any
            out.append((path, None, f"{type(e).__name__}: {e}"))
            continue
        ts = getattr(mod, "TWIN_SPEC", None)
        if not isinstance(ts, dict):
            out.append((path, None, "no TWIN_SPEC dict"))
            continue
        missing = [k for k in ("key", "label", "model_class", "params_class")
                   if k not in ts]
        if missing:
            out.append((path, None, f"TWIN_SPEC missing {missing}"))
            continue
        out.append((path, (mod, ts), None))
    return out
