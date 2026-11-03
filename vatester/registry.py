"""Shared device and pattern registries used by the desktop.

The dictionaries are updated in place so existing App imports and hot reload
continue to see the same objects. Importing this module does not load workspace
plugins; registration is explicit and uses the existing trusted-code loaders.
"""
import dataclasses
import importlib
import os

from simulation_core import (EcfetV1, V1Params, EcfetV2, V2Params,
                   EcfetV3, V3Params, FeFET, FeFETParams)
from simulation_core import (model_v1 as _m_v1, model_v2 as _m_v2,
                   model_v3 as _m_v3)
# the FeFET model is private and absent from public builds
_m_fefet = importlib.import_module("simulation_core.model_fefet") if FeFET else None
from . import va_scan as _va_scan_mod
from . import twin_loader, pattern_loader


@dataclasses.dataclass
class ModelSpec:
    key: str
    label: str
    cls: object
    params_cls: object
    input_kind: str


MODEL_SPECS = [
    ModelSpec("v1", "ECFET v1 (Verilog-A port)", EcfetV1, V1Params, "current"),
    ModelSpec("v2", "ECFET v2 (practical ECRAM)", EcfetV2, V2Params, "current"),
    ModelSpec("v3", "ECFET v3 (paper-faithful)", EcfetV3, V3Params, "current"),
]
SPEC_BY_KEY = {s.key: s for s in MODEL_SPECS}

# device classes group the model keys; selecting a class reconfigures the GUI
# (which models are enabled, current vs voltage drive, ΔG vs ΔVt, polarization)
DEVICE_FAMILIES = {"ECFET": ("v1", "v2", "v3")}
DEVICE_OF_KEY = {k: dev for dev, keys in DEVICE_FAMILIES.items() for k in keys}
DEVICE_KIND = {"ECFET": "current", "FeFET": "voltage"}   # default drive

# model_key -> the Python "twin" source the GUI actually simulates
TWIN_FILE = {"v1": "simulation_core/model_v1.py", "v2": "simulation_core/model_v2.py",
             "v3": "simulation_core/model_v3.py"}
LABEL_TO_KEY = {s.label: s.key for s in MODEL_SPECS}

# modules to hot-reload when the agent edits a twin, so the GUI re-simulates
# with the new code (key -> (module, class name, params class name))
RELOAD_MODULES = {
    "v1": (_m_v1, "EcfetV1", "V1Params"),
    "v2": (_m_v2, "EcfetV2", "V2Params"),
    "v3": (_m_v3, "EcfetV3", "V3Params"),
}

if FeFET is not None:                 # register the private FeFET model when present
    _fefet = ModelSpec("fefet", "FeFET (Merz/Preisach-lite)", FeFET, FeFETParams, "voltage")
    MODEL_SPECS.append(_fefet)
    SPEC_BY_KEY["fefet"] = _fefet
    LABEL_TO_KEY[_fefet.label] = "fefet"
    DEVICE_FAMILIES["FeFET"] = ("fefet",)
    DEVICE_OF_KEY["fefet"] = "FeFET"
    TWIN_FILE["fefet"] = "simulation_core/model_fefet.py"
    RELOAD_MODULES["fefet"] = (_m_fefet, "FeFET", "FeFETParams")
else:                                 # a user's FeFET .va maps to their own twin, or
    _va_scan_mod.MODEL_HINTS[:] = [h for h in _va_scan_mod.MODEL_HINTS
                                   if h[0] != "fefet"]   # prompts for one

# every file the live watcher polls + reloads (twins + the analysis layer)
WATCH_FILES = list(TWIN_FILE.values()) + ["vatester/analysis.py", "vatester/stdp_protocol.py"]

# user/agent input-PATTERN plugins (the `patterns/` folder, hot-watched). Built
# in the GUI; plugins ADD to it without editing the app or restarting.
BUILTIN_PATSETS = ["bars", "letters", "digits", "random", "nand", "xor"]
PATTERN_PLUGINS = {}            # key -> PATTERN_SPEC dict (make(cfg)->pats,tgts)


def register_pattern_plugins(workdir):
    """(Re)load every patterns/*.py into PATTERN_PLUGINS. Returns (keys,errs)."""
    results = pattern_loader.load_patterns(os.path.join(workdir, "patterns"))
    reg, errs = {}, []
    for path, payload, err in results:
        if err:
            errs.append((os.path.basename(path), err))
            continue
        _mod, ps = payload
        reg[ps["key"]] = ps
    PATTERN_PLUGINS.clear()
    PATTERN_PLUGINS.update(reg)
    return list(reg), errs


def _register_twin(mod, ts):
    """Register one dynamic twin (its TWIN_SPEC) into the model registry."""
    key = ts["key"]
    if key in SPEC_BY_KEY:
        return key                                  # built-in or already loaded
    mcls, pcls = ts["model_class"], ts["params_class"]
    kind = ts.get("input_kind", "current")
    spec = ModelSpec(key, ts["label"], mcls, pcls, kind)
    MODEL_SPECS.append(spec)
    SPEC_BY_KEY[key] = spec
    LABEL_TO_KEY[spec.label] = key
    RELOAD_MODULES[key] = (mod, mcls.__name__, pcls.__name__)
    if getattr(mod, "__file__", None):
        TWIN_FILE[key] = mod.__file__
        WATCH_FILES.append(mod.__file__)
    dev = ts.get("device_class", "ECFET")
    DEVICE_FAMILIES[dev] = tuple(DEVICE_FAMILIES.get(dev, ())) + (key,)
    DEVICE_OF_KEY[key] = dev
    DEVICE_KIND.setdefault(dev, kind)
    _va_scan_mod.MODEL_HINTS.insert(0, (key, tuple(ts.get("va_keywords", (key,)))))
    # optional profile -> set as class attributes the GUI already reads
    p = ts.get("stdp")
    if p:
        mcls.STDP_OBS = p.get("obs", "G")
        mcls.STDP_LABEL = p.get("label", "dG")
        mcls.STDP_UNIT = p.get("unit", "uS")
        mcls.STDP_SCALE = p.get("scale", 1e6)
        if p.get("stimulus"):
            mcls.STDP_STIMULUS = dict(p["stimulus"])
    if ts.get("result_plots"):
        mcls.RESULT_PLOTS = tuple(tuple(x) for x in ts["result_plots"])
    if ts.get("analysis_metrics"):
        mcls.ANALYSIS_METRICS = tuple(tuple(x) for x in ts["analysis_metrics"])
    if ts.get("polar_obs"):
        mcls.POLAR_OBS = ts["polar_obs"]
    if ts.get("analyses"):
        mcls.ANALYSES = tuple(ts["analyses"])
    return key


def register_dynamic_twins(workdir):
    """Load + register every twin under <workdir>/twins. (loaded, errors)."""
    loaded, errors = [], []
    for path, payload, err in twin_loader.load_twins(
            os.path.join(workdir, "twins")):
        if err:
            errors.append((os.path.basename(path), err))
            continue
        mod, ts = payload
        try:
            loaded.append(_register_twin(mod, ts))
        except Exception as e:                       # noqa: BLE001
            errors.append((os.path.basename(path), f"register failed: {e}"))
    return loaded, errors


def _defaults_of(params_cls):
    inst = params_cls()
    return {f.name: getattr(inst, f.name)
            for f in dataclasses.fields(params_cls)}


def _reload_module(mod):
    """Reload source in place without accepting stale timestamp-based bytecode.

    Preserve the module identity used by sibling twin imports. The caller
    reloads base twins before their dependents so subclasses bind to new code.
    """
    path = getattr(mod, "__file__", None)
    if not path or not path.endswith(".py"):
        importlib.reload(mod)
        return
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    exec(compile(src, path, "exec"), mod.__dict__)
