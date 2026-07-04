"""OpenVAF seam — compile Verilog-A to an OSDI module, with graceful fallback.

If the `openvaf` binary is on PATH it is used for a real compile; otherwise we
parse the source for a parameter count and report the behavioural twin as the
active engine. Wire your simulator into `run_osdi()` to make run_sim use the
compiled model instead of the twin (see the TODO below).
"""

import os
import re
import shutil
import subprocess
import tempfile


def openvaf_available():
    return shutil.which("openvaf") is not None


def _param_count(va):
    return len(re.findall(r"\bparameter\s+(?:real|integer)\b", va or ""))


def compile_va(va="", cell="ecfet_v2", **_):
    """Contract: -> { ok, params, osdi, engine }. Raises RuntimeError on a real
    OpenVAF compile error so the front-end can surface it in the log."""
    n_params = _param_count(va)

    if not openvaf_available():
        # No OpenVAF installed — parse-only, behavioural twin stays active.
        return {"ok": True, "params": n_params, "osdi": None, "engine": "twin"}

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, cell + ".va")
        out = os.path.join(d, cell + ".osdi")
        with open(src, "w", encoding="utf-8") as f:
            f.write(va)
        try:
            subprocess.run(["openvaf", src, "-o", out],
                           check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("OpenVAF compile error:\n" + (e.stderr or e.stdout or str(e)))
        # Keep the built module next to the repo so run_osdi() can find it.
        dest_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, cell + ".osdi")
        shutil.copyfile(out, dest)
        return {"ok": True, "params": n_params, "osdi": dest, "engine": "openvaf"}


def run_osdi(osdi_path, pulses, essentials):
    """TODO (your simulator): drive the compiled .osdi with a Verilog-A capable
    engine (ngspice+OSDI, Xyce, or Spectre) and return the same dict shape as
    twin.simulate(). Until this is implemented, run_sim falls back to the twin.
    """
    raise NotImplementedError
