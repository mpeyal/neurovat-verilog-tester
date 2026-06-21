"""Dynamic loader for user/agent input-PATTERN plugins kept OUTSIDE the app
source - the trainer's equivalent of `twins/` for devices.

A new training pattern source (a logic gate, a custom stimulus bank, ...) lives
in a top-level `patterns/` folder, not in the GUI (`vatester/app.py`) or the
engine (`vatester/neuro.py`).  Each file is a self-contained module declaring a
`PATTERN_SPEC` dict, so new patterns appear in the trainer's PATTERNS dropdown
with NO edit to the GUI and NO restart (the folder is hot-watched):

    # patterns/my_gate.py
    import numpy as np

    def _make(cfg):
        # may set the grid + class count for this pattern
        cfg.grid_h, cfg.grid_w, cfg.n_out = 2, 2, 2
        pats = [("A=0 B=0 -> 1", np.array([0., 1., 0., 1.], np.float32)), ...]
        targets = [1, ...]
        return pats, targets          # patterns = [(label, vec01_flat_[0,1])]

    PATTERN_SPEC = {
        "key":   "mygate",            # unique short id (the combo value)
        "label": "mygate",            # shown in the PATTERNS dropdown
        "make":  _make,               # make(cfg) -> (patterns, targets)
    }

SECURITY NOTE: importing a Python plugin EXECUTES it - there is no sandbox.  The
folder separation is organisational safety, not a defence against malicious
code.  Only put files you trust here.
"""

import glob
import importlib.util
import os
import sys


def load_patterns(patterns_dir):
    """Import every patterns/*.py and collect their PATTERN_SPEC.

    Returns a list of (path, payload, error): payload is (module, spec) on
    success, else None with an error string.  Never raises - a bad file is
    reported, not fatal.
    """
    out = []
    if not os.path.isdir(patterns_dir):
        return out
    for path in sorted(glob.glob(os.path.join(patterns_dir, "*.py"))):
        base = os.path.basename(path)
        if base.startswith("_"):
            continue
        modname = "patterns_" + os.path.splitext(base)[0]
        try:
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod           # so importlib.reload works later
            spec.loader.exec_module(mod)
        except Exception as e:                   # noqa: BLE001 - report any
            out.append((path, None, f"{type(e).__name__}: {e}"))
            continue
        ps = getattr(mod, "PATTERN_SPEC", None)
        if not isinstance(ps, dict):
            out.append((path, None, "no PATTERN_SPEC dict"))
            continue
        missing = [k for k in ("key", "label", "make") if k not in ps]
        if missing:
            out.append((path, None, f"PATTERN_SPEC missing {missing}"))
            continue
        if not callable(ps["make"]):
            out.append((path, None, "PATTERN_SPEC['make'] is not callable"))
            continue
        out.append((path, (mod, ps), None))
    return out
