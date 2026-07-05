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
    # A REAL Cadence Virtuoso connection (the same SSH-tunnel + skillbridge the
    # desktop app uses). Lazily built; None if vatester.virtuoso isn't importable.
    _vlink = None

    def _get_vlink(self):
        if Bridge._vlink is None:
            try:
                from vatester.virtuoso import VirtuosoLink
                Bridge._vlink = VirtuosoLink()
            except Exception:
                Bridge._vlink = False
        return Bridge._vlink or None

    def virtuoso_connect(self, args=None):
        """Open the real SSH tunnel + skillbridge workspace. On success returns
        the live libraries; on failure the real error (e.g. host unreachable /
        skill server not running) so the UI can show WHY."""
        link = self._get_vlink()
        if link is None:
            return {"ok": False, "connected": False,
                    "error": "Virtuoso link unavailable (vatester.virtuoso not importable)"}
        try:
            info = link.connect()
            return {"ok": True, "connected": True, "version": info.get("version"),
                    "tunnel": info.get("tunnel"), "libs": info.get("libraries", [])}
        except Exception as e:
            return {"ok": False, "connected": False, "error": str(e)}

    def virtuoso_disconnect(self, args=None):
        link = self._get_vlink()
        if link is not None:
            try:
                link.disconnect()
            except Exception:
                pass
        return {"ok": True, "connected": False}

    def virtuoso_cells(self, args=None):
        """Cells in a library from the live workspace (veriloga flagged), capped."""
        a = args or {}
        link = self._get_vlink()
        if link is None or not link.connected:
            return {"ok": False, "cells": [], "error": "not connected"}
        try:
            allc = link.list_cells(a.get("lib", ""), False)
            vac = set(link.list_cells(a.get("lib", ""), True))
            cells = [{"cell": c, "view": "veriloga" if c in vac else "schematic",
                      "va": c in vac} for c in allc[:80]]
            return {"ok": True, "cells": cells, "truncated": len(allc) > 80}
        except Exception as e:
            return {"ok": False, "cells": [], "error": str(e)}

    def load_from_virtuoso(self, args=None):
        a = args or {}
        cell = _check_cell(a.get("cell", "") or "ecfet_v2")
        # when a live Virtuoso session is up, read the REAL cellview source
        link = self._get_vlink()
        if link is not None and link.connected and a.get("lib"):
            try:
                r = link.read_source(a.get("lib", ""), cell, a.get("view", "veriloga"))
                if r.get("ok") and r.get("text"):
                    return {"source": r["text"], "engine": "skillbridge"}
            except Exception:
                pass
        path = _workspace_va(cell)
        if path:
            with open(path, "r", encoding="utf-8", newline="") as f:
                return {"source": f.read(), "engine": "workspace"}
        return virtuoso.load_from_virtuoso(a.get("lib", ""), cell, a.get("view", "veriloga"))

    def write_to_virtuoso(self, args=None):
        a = args or {}
        cell = _check_cell(a.get("cell", ""))
        # a live Virtuoso session + a lib -> push into the real cell (hiWriteback)
        link = self._get_vlink()
        if link is not None and link.connected and a.get("lib"):
            try:
                link.write_source(a.get("lib", ""), cell, a.get("view", "veriloga"), a.get("source", ""))
                return {"ok": True, "engine": "skillbridge"}
            except Exception as e:
                return {"ok": False, "engine": "skillbridge", "error": str(e)}
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

    def eval_net(self, args=None):
        """Real held-out evaluation (Test button) via vatester.neuro."""
        return trainer.eval_net(args or {})

    def load_dataset(self, args=None):
        """Really download/decode a dataset (MNIST / URL) and encode it to the
        trainer's input patterns (vatester.datasets). Raises on failure."""
        return trainer.load_dataset(**(args or {}))

    def clear_dataset(self, args=None):
        return trainer.clear_dataset()

    # --- real workspace scan + twin/.va consistency + CSV fit -----------------
    def rescan_va(self, args=None):
        """Actually list the .va files in the workspace (not a canned string)."""
        root = engine.repo_root()
        try:
            files = sorted(f for f in os.listdir(root) if f.lower().endswith(".va"))
        except OSError:
            files = []
        return {"ok": True, "files": files, "count": len(files), "workspace": root}

    def fit_csv(self, args=None):
        """Fit the LTP/LTD nonlinearity of a measured G-per-pulse CSV (text)."""
        a = args or {}
        text = a.get("csv", "") or ""
        g = []
        for line in text.splitlines():
            parts = re.split(r"[,\s;]+", line.strip())
            for tok in reversed(parts):          # last numeric field = G
                try:
                    g.append(float(tok)); break
                except ValueError:
                    continue
        if len(g) < 4:
            return {"ok": False, "error": "need >= 4 numeric G values (got %d)" % len(g)}
        try:
            from vatester import tools_ext
            res = tools_ext.fit_ltp_ltd(g)
            return {"ok": True, "n": len(g), "report": tools_ext.format_fit(res),
                    "result": {k: v for k, v in res.items()
                               if isinstance(v, (int, float, str, bool))}}
        except Exception as e:
            return {"ok": False, "error": str(e)}

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
