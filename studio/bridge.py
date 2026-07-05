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

from core import twin, openvaf, virtuoso, engine, trainer
from core import analysis as _analysis

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
                  gen=a.get("gen", "train"), device=a.get("device", "v2"),
                  va=a.get("va"), unit=a.get("unit"))
        if engine.available():
            try:
                return engine.run_sim(**kw)
            except Exception:
                pass  # fall back to the analytic twin below
        # the analytic twin ignores the .va text + unit; drop them to match its sig
        return twin.simulate(**{k: v for k, v in kw.items() if k not in ("va", "unit")})

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

    # --- neuromorphic trainer + LIF probe (reuse the desktop's vatester.neuro) --
    def train_net(self, args=None):
        """Real crossbar+LIF training via vatester.neuro. Raises if neuro isn't
        importable so the front-end keeps its in-page JS demo (offline)."""
        return trainer.train_net(args or {})

    def probe_lif(self, args=None):
        """Real LIF neuron probe (membrane trace + f-I curve) via vatester.neuro."""
        return trainer.probe_lif(args or {})

    # --- device-measured analyses (real STDP window + FeFET P-V loop) ---------
    def stdp_window(self, args=None):
        """Real anti-symmetric STDP window measured from the device model."""
        a = args or {}
        return _analysis.stdp_window(device=a.get("device", "v2"), va=a.get("va"),
                                     dt_max_ms=a.get("dt_max_ms", 1800), n=a.get("n", 28))

    def polarization(self, args=None):
        """Real FeFET P-V hysteresis loop (empty/available:False for ECFET)."""
        a = args or {}
        return _analysis.polarization(device=a.get("device", "fefet"), va=a.get("va"),
                                      v_amp=a.get("v_amp", 3.0))

    # --- embedded agent (real Claude, READ-ONLY on the web path) -------------
    _agent = None

    def _get_agent(self):
        """Lazily build the desktop ClaudeAgent (same CLI backend). None if the
        agent module / CLI isn't available (UI then keeps its canned demo)."""
        if Bridge._agent is None:
            try:
                from vatester.agent import ClaudeAgent
                a = ClaudeAgent(engine.repo_root())
                # only usable if a real backend was found
                Bridge._agent = a if (getattr(a, "cli", None) or getattr(a, "sdk_ok", False)) else False
            except Exception:
                Bridge._agent = False
        return Bridge._agent or None

    def agent_chat(self, args=None):
        """One real agent turn. READ-ONLY (Read/Grep/Glob only — no Edit/Write/
        Bash, no bypassPermissions) so the localhost web agent can explain the
        .va/sources without the file-edit/shell RCE surface. Contract:
        {ok, text, backend} or {ok:False} to let the UI fall back to its demo."""
        a = args or {}
        agent = self._get_agent()
        if agent is None:
            return {"ok": False, "text": "", "backend": None}
        try:
            r = agent.send(str(a.get("message", "")), context=str(a.get("context", "")),
                           allow_edits=False, allow_bash=False,
                           model=a.get("model") or "default", timeout=180)
            return {"ok": bool(r.get("ok")), "text": r.get("text", "") or r.get("error", ""),
                    "backend": agent.backend_label(), "model": a.get("model") or "default"}
        except Exception as e:
            return {"ok": False, "text": str(e), "backend": None}

    # --- health -----------------------------------------------------------
    def health(self, args=None):
        try:
            from vatester import tools_ext
            has_openvaf = tools_ext.find_openvaf() is not None
        except Exception:
            has_openvaf = openvaf.openvaf_available()
        return {"ok": True, "openvaf": has_openvaf,
                "engine": "ecfet" if engine.available() else "twin",
                "neuro": trainer.available(),
                "agent": self._get_agent() is not None,
                "workspace": engine.repo_root() if engine.available() else None}
