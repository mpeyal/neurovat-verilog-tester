"""Bridge — the single API surface shared by both front-end transports.

Every method takes ONE dict argument and returns a JSON-serialisable dict,
matching the `// CONTRACT:` comments in the HTML. The desktop launcher exposes
this object directly as pywebview's js_api; the server launcher wraps each
method as POST /api/<name>. Add a method here and it is instantly callable from
the front-end as backend.call("<name>", args) — no other wiring needed.

Engine preference: when studio/ lives inside the NeuroVAT repo, run_sim uses
the REAL `ecfet` models (the same physics the Dear PyGui app runs), compile_va
uses the app's OpenVAF locator (vatester.tools_ext), and the Virtuoso panel
reads/writes the repo's actual .va sources. Each falls back to the bundled
standalone implementation in core/ so studio also runs on its own.
"""

import os
import re
import tempfile

from core import twin, openvaf, virtuoso, engine

_CELL_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$")   # plain file stems only


def _check_cell(cell):
    """Reject any cell name that isn't a plain file stem (no separators/dots-only)
    BEFORE it reaches any handler — the core/ fallback joins it into a path."""
    if not cell or not _CELL_RE.match(cell):
        raise ValueError("invalid cell name: %r" % (cell,))
    return cell


def _workspace_va(cell):
    """Path of the repo's real .va for this cell (case-insensitive), or None."""
    root = engine.repo_root()
    try:
        for fn in os.listdir(root):
            if fn.lower() == cell.lower() + ".va":
                return os.path.join(root, fn)
    except OSError:
        pass
    return None


class Bridge:
    # --- simulation -------------------------------------------------------
    def run_sim(self, args=None):
        a = args or {}
        kw = dict(pulses=a.get("pulses", []), essentials=a.get("essentials"),
                  gen=a.get("gen", "train"), device=a.get("device", "v2"))
        if engine.available():
            try:
                return engine.run_sim(**kw)
            except Exception:
                pass  # fall back to the analytic twin below
        return twin.simulate(**kw)

    def compile_va(self, args=None):
        a = args or {}
        va = a.get("va", "")
        cell = a.get("cell", "") or "ecfet_v2"
        if not _CELL_RE.match(cell):
            cell = "ecfet_v2"
        try:
            from vatester import tools_ext
        except Exception:
            return openvaf.compile_va(va, cell)

        n_params = len(re.findall(r"\bparameter\s+(?:real|integer)\b", va or ""))
        exe = tools_ext.find_openvaf()
        if not exe:
            return {"ok": True, "params": n_params, "osdi": None, "engine": "twin"}

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, cell + ".va")
            with open(src, "w", encoding="utf-8") as f:
                f.write(va)
            r = tools_ext.openvaf_compile(src, exe=exe)
            if not r.get("ok"):
                raise RuntimeError("OpenVAF compile error:\n" + (r.get("output") or "unknown"))
            dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, cell + ".osdi")
            if r.get("osdi"):
                import shutil
                shutil.copyfile(r["osdi"], dest)
            return {"ok": True, "params": n_params, "osdi": dest, "engine": "openvaf"}

    # --- Cadence Virtuoso / workspace sources ------------------------------
    # When inside the repo, "Load source" serves the SAME .va files the desktop
    # app scans, and "Write back" edits them in place (local, git-versioned).
    # Outside the repo it falls back to core/virtuoso.py (skillbridge or ./va).
    def load_from_virtuoso(self, args=None):
        a = args or {}
        cell = _check_cell(a.get("cell", "") or "ecfet_v2")
        path = _workspace_va(cell)
        if path:
            with open(path, "r", encoding="utf-8", newline="") as f:
                return {"source": f.read(), "engine": "workspace"}
        return virtuoso.load_from_virtuoso(a.get("lib", ""), cell, a.get("view", "veriloga"))

    def write_to_virtuoso(self, args=None):
        a = args or {}
        cell = _check_cell(a.get("cell", ""))
        path = _workspace_va(cell)
        if path:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(a.get("source", ""))
            return {"ok": True, "engine": "workspace"}
        return virtuoso.write_to_virtuoso(
            a.get("lib", ""), cell, a.get("view", "veriloga"), a.get("source", ""))

    # --- health -----------------------------------------------------------
    def health(self, args=None):
        try:
            from vatester import tools_ext
            has_openvaf = tools_ext.find_openvaf() is not None
        except Exception:
            has_openvaf = openvaf.openvaf_available()
        return {"ok": True, "openvaf": has_openvaf,
                "engine": "ecfet" if engine.available() else "twin",
                "workspace": engine.repo_root() if engine.available() else None}
