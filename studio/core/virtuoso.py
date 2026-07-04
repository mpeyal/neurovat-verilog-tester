"""Cadence Virtuoso seam — pull/push Verilog-A via skillbridge, with fallback.

If `skillbridge` is installed and a Virtuoso workspace is reachable it is used;
otherwise cells are read from / written to the local ./va folder so the rest of
the app keeps working without Cadence.
"""

import os
import re

VA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "va")


def _safe_cell(cell):
    """Cell names become local filenames — allow plain stems only, no path parts."""
    cell = cell or "ecfet_v2"
    if not re.match(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*$", cell):
        raise ValueError("invalid cell name: %r" % (cell,))
    return cell


def _workspace():
    try:
        from skillbridge import Workspace
        return Workspace.open()
    except Exception:
        return None


def load_from_virtuoso(lib="", cell="", view="veriloga", **_):
    """Contract: -> { source: "<verilog-a text>" }."""
    cell = _safe_cell(cell)
    ws = _workspace()
    if ws is not None:
        try:
            cv = ws.db.open_cell_view_by_type(lib, cell, view, "", "r")
            path = ws.db.get_prop(cv, "vfsFileName") or ""
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return {"source": f.read(), "engine": "skillbridge"}
        except Exception:
            pass  # fall through to local

    from .twin import default_va
    local = os.path.join(VA_DIR, (cell or "ecfet_v2") + ".va")
    if os.path.exists(local):
        with open(local, "r", encoding="utf-8") as f:
            return {"source": f.read(), "engine": "local"}
    return {"source": default_va(cell or "ecfet_v2"), "engine": "sample"}


def write_to_virtuoso(lib="", cell="", view="veriloga", source="", **_):
    """Contract: -> { ok: True }. Writes to Virtuoso if reachable, else ./va."""
    cell = _safe_cell(cell)
    ws = _workspace()
    if ws is not None:
        try:
            cv = ws.db.open_cell_view_by_type(lib, cell, view, "", "a")
            path = ws.db.get_prop(cv, "vfsFileName") or ""
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(source)
                ws.hi.write_back(cv)
                return {"ok": True, "engine": "skillbridge"}
        except Exception:
            pass

    os.makedirs(VA_DIR, exist_ok=True)
    with open(os.path.join(VA_DIR, (cell or "ecfet_v2") + ".va"), "w", encoding="utf-8") as f:
        f.write(source)
    return {"ok": True, "engine": "local"}
