"""Dynamic loader for user/agent device twins kept OUTSIDE the app source.

Device behavioural models ("twins") for NEW devices live in a separate
top-level `twins/` folder, not in the GUI package (`vatester/`) or the core
engine (`simulation_core/`).  Each twin file is a self-contained Python module that
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
        "stdp": {"obs": "Vth (V)", "label": "dVt", "unit": "mV", "scale": 1e3,
                 # optional: STDP tab stimulus loaded when this class is picked
                 "stimulus": {"mode": "overlap", "unit": "V", "amp": 0.55,
                              "width_ms": 1.0, "tail_ratio": 1.0,
                              "tail_tau_ms": 20.0, "range_ms": 160.0,
                              "settle_ms": 20.0, "npts": 24}},
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
import sys
import types


def load_twins(twins_dir):
    """Import every twins/*.py and collect their TWIN_SPEC.

    Returns a list of (path, payload, error): payload is (module, spec) on
    success, else None with an error string.  Never raises - a bad twin file is
    reported, not fatal.
    """
    out = []
    if not os.path.isdir(twins_dir):
        return out
    twins_dir = os.path.abspath(twins_dir)
    # Sibling imports (e.g. `from twins.fecap_nls import FeCAPNLS`) must
    # resolve to the very same modules that the GUI watches and reloads.
    # Point the namespace at the selected workspace, not the application's
    # bundled twins, including when --workspace selects an external folder.
    package = sys.modules.get("twins")
    if package is None or list(getattr(package, "__path__", ())) != [twins_dir]:
        for name in list(sys.modules):
            if name.startswith("twins.") or name.startswith("twins_"):
                del sys.modules[name]
        package = types.ModuleType("twins")
        package.__path__ = [twins_dir]
        package.__package__ = "twins"
        package.__spec__ = importlib.util.spec_from_loader("twins", loader=None,
                                                         is_package=True)
        package.__spec__.submodule_search_locations = [twins_dir]
        sys.modules["twins"] = package
    for path in sorted(glob.glob(os.path.join(twins_dir, "*.py"))):
        base = os.path.basename(path)
        if base.startswith("_"):
            continue
        stem = os.path.splitext(base)[0]
        modname = "twins." + stem
        try:
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = sys.modules.get(modname)
            if mod is None:
                mod = importlib.util.module_from_spec(spec)
            # Publish before execution so sibling imports share this identity.
            sys.modules[modname] = mod
            sys.modules["twins_" + stem] = mod  # compatibility for existing tools
            setattr(package, stem, mod)
            with open(path, encoding="utf-8") as source:
                exec(compile(source.read(), path, "exec"), mod.__dict__)
        except Exception as e:                      # noqa: BLE001 - report any
            sys.modules.pop(modname, None)
            sys.modules.pop("twins_" + stem, None)
            if hasattr(package, stem):
                delattr(package, stem)
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
