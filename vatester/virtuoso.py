"""Skillbridge link to Cadence Virtuoso over an SSH tunnel.

Windows side of the bridge: starts (if needed) an SSH tunnel forwarding
localhost:7777 to the skillbridge Unix socket on the Linux host, then opens
a skillbridge Workspace over it.

Prereqs:
  - SSH key auth to the Linux host (no password prompts; BatchMode is used).
  - Virtuoso open on the Linux side with the skillbridge server running
    (pyStartServer / skill() in the CIW).
  - pip install skillbridge
"""

import socket
import subprocess
import sys
import threading
import time

HOST = "mahmudulpeyal@coen-cassia.boisestate.edu"
# The skill server is started PER USER (skill()/pyStartServer use ?id = the
# login name), so its socket is /tmp/skill-server-<user>.sock - NOT the shared
# "default" socket. On this multi-user host the "default" socket is already
# owned by another account and /tmp is sticky, so connecting to it just gets
# reset (the WinError 10054 we kept hitting). Derive our own per-user socket.
_USER = HOST.split("@", 1)[0]
REMOTE_SOCKET = f"/tmp/skill-server-{_USER}.sock"
PORT = 7777

# View names treated as "Verilog/behavioral" source for the browser filter.
VERILOG_VIEWS = ("veriloga", "verilogams", "verilogA", "ahdl", "vams",
                 "verilog", "functional")

# SKILL helpers defined once per connection so each browse query is a single
# tunnel round trip (iteration happens server-side, not over the socket).
_SKILL_HELPERS = [
    r'''(defun pyReadFile (p)
  (let ((s "") ln port)
    (when (and (isFile p) (setq port (infile p)))
      (while (gets ln port) (setq s (strcat s ln))) (close port)) s))''',
    r'''(defun pyWriteFile (p s)
  (let (port)
    (if (setq port (outfile p))
      (progn (fprintf port "%s" s) (close port) "OK")
      (strcat "ERR|cannot open for write: " p))))''',
    r'''(defun pyCells (libName vaOnly)
  (let ((res "")
        (vlist (list "veriloga" "verilogams" "verilogA" "ahdl" "vams"
                     "verilog" "functional")))
    (foreach cell (ddGetObj libName)->cells
      (if vaOnly
        (let ((hit nil))
          (foreach view cell->views
            (when (member view->name vlist) (setq hit t)))
          (when hit (setq res (strcat res cell->name "\n"))))
        (setq res (strcat res cell->name "\n"))))
    res))''',
    r'''(defun pyViews (libName cellName)
  (let ((res "") (cv (ddGetObj libName cellName)))
    (when cv (foreach view cv->views (setq res (strcat res view->name "\n"))))
    res))''',
    r'''(defun pyCvFiles (libName cellName viewName)
  (let ((cv (ddGetObj libName cellName viewName)))
    (if (null cv) "ERR|no such cellview"
      (strcat cv->readPath "|" (buildString (getDirFiles cv->readPath) ",")))))''',
]

# Source-file preference when a cellview holds several files. Lower index wins;
# exact names beat extensions. Binary OA files are never picked.
_SRC_EXACT = ["veriloga.va", "verilogams.vams", "verilog.v", "verilog.vams",
              "ahdl.def", "functional.v"]
_SRC_EXTS = [".va", ".vams", ".v", ".scs", ".sp", ".cir", ".def", ".m"]


def _pick_source(files):
    """Choose the most source-like text file from a cellview directory."""
    real = [f for f in files if f not in (".", "..") and not f.endswith(".oa")
            and f not in ("master.tag", "pc.db", "data.dm")]
    for name in _SRC_EXACT:
        if name in real:
            return name
    for ext in _SRC_EXTS:
        for f in real:
            if f.endswith(ext):
                return f
    return None


def _port_open(port=PORT, timeout=1.0):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


class VirtuosoLink:
    """Owns the tunnel subprocess and the skillbridge workspace."""

    def __init__(self, host=HOST, remote_socket=REMOTE_SOCKET, port=PORT):
        self.host = host
        self.remote_socket = remote_socket
        self.port = port
        self.ws = None
        self.tunnel = None      # Popen only if we started the tunnel ourselves
        self.info = {}
        self._helpers_ready = False
        # skillbridge's socket is not thread-safe; serialize every ws call so
        # overlapping UI actions (connect still finishing while a browse fires)
        # can't interleave on the wire.
        self._lock = threading.RLock()

    @property
    def connected(self):
        return self.ws is not None

    # ---------------- tunnel ----------------

    def _ensure_tunnel(self, wait_s=15.0):
        if _port_open(self.port):
            return "reused running tunnel"
        if self.tunnel and self.tunnel.poll() is None:
            self.tunnel.terminate()
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.tunnel = subprocess.Popen(
            ["ssh", "-N", "-o", "BatchMode=yes",
             "-o", "ExitOnForwardFailure=yes",
             "-o", "ConnectTimeout=10",
             "-L", f"{self.port}:{self.remote_socket}", self.host],
            creationflags=flags,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if _port_open(self.port):
                return "tunnel started"
            if self.tunnel.poll() is not None:
                err = (self.tunnel.stderr.read() or b"").decode(
                    errors="replace").strip()
                self.tunnel = None
                raise RuntimeError(
                    f"ssh tunnel exited: {err or 'unknown error'} "
                    "(is SSH key auth set up?)")
            time.sleep(0.4)
        raise RuntimeError(
            f"tunnel did not open port {self.port} within {wait_s:.0f}s")

    # ---------------- skillbridge ----------------

    def connect(self):
        """Bring up tunnel + workspace. Returns an info dict; raises on failure."""
        try:
            from skillbridge import Workspace
        except ImportError:
            raise RuntimeError(
                "skillbridge is not installed (pip install skillbridge)")
        how = self._ensure_tunnel()
        with self._lock:
            try:
                ws = Workspace.open()
                version = str(ws["getVersion"]())
            except Exception as e:
                raise RuntimeError(
                    f"tunnel is up but skillbridge failed: {e}\n\n"
                    "The SSH tunnel is fine - the skill server didn't answer "
                    "on your per-user socket:\n"
                    f"    {self.remote_socket}\n"
                    "Make sure THAT server is running on the host:\n"
                    "  1) Virtuoso is open on the host;\n"
                    "  2) in the CIW, start the skill server - run skill() (or "
                    "pyStartServer with ?id set to your login so it creates the "
                    "socket above; do NOT use the shared 'default' socket, it "
                    "may belong to another user);\n"
                    "  3) confirm with pyShowLog(), then click Connect again.")
            info = {"tunnel": how, "version": version}
            try:
                info["libraries"] = [l.name for l in ws.dd.get_lib_list()]
            except Exception:
                info["libraries"] = []
            self.ws = ws
            self.info = info
            self._helpers_ready = False
            try:
                self._ensure_helpers()
            except Exception:
                pass  # browsing helpers optional; connection itself is fine
        return info

    # ---------------- library / cell / view browsing ----------------

    def _ensure_helpers(self):
        if self._helpers_ready or self.ws is None:
            return
        for src in _SKILL_HELPERS:
            self.ws["evalstring"](src)
        self._helpers_ready = True

    def _require(self):
        if self.ws is None:
            raise RuntimeError("not connected to Virtuoso")
        self._ensure_helpers()

    @staticmethod
    def _lines(blob):
        return [s for s in (blob or "").strip().split("\n") if s]

    def list_libraries(self):
        """Library names (refreshed from the live workspace)."""
        with self._lock:
            if self.ws is None:
                return list(self.info.get("libraries", []))
            libs = [l.name for l in self.ws.dd.get_lib_list()]
            self.info["libraries"] = libs
            return libs

    def list_cells(self, lib, verilog_only=False):
        """Cell names in a library, optionally only those with a Verilog view."""
        with self._lock:
            self._require()
            return self._lines(self.ws["pyCells"](lib, bool(verilog_only)))

    def list_views(self, lib, cell):
        """View names of a cell."""
        with self._lock:
            self._require()
            return self._lines(self.ws["pyViews"](lib, cell))

    def read_source(self, lib, cell, view):
        """Read the text source backing a cellview.

        Returns a dict: {ok, path, text, files, note}. ok is False when the
        view holds only binary OA data (no readable source file).
        """
        with self._lock:
            self._require()
            raw = self.ws["pyCvFiles"](lib, cell, view) or ""
            if raw.startswith("ERR|"):
                return {"ok": False, "path": "", "text": "", "files": [],
                        "note": raw[4:] or "could not read cellview"}
            path, _, files_csv = raw.partition("|")
            files = [f for f in files_csv.split(",") if f]
            pick = _pick_source(files)
            if not pick:
                return {"ok": False, "path": path, "text": "", "files": files,
                        "note": "no readable text source (binary OA view)"}
            full = path + "/" + pick
            text = self.ws["pyReadFile"](full) or ""
            return {"ok": True, "path": full, "text": text, "files": files,
                    "note": ""}

    def write_source(self, lib, cell, view, text):
        """Write `text` back to the source file backing a cellview (overwrites
        it in the Cadence library). Returns {ok, path, note}. The library must
        be writable; recompile/re-netlist the cell in Cadence to pick it up."""
        with self._lock:
            self._require()        # loads pyWriteFile + the browse helpers
            raw = self.ws["pyCvFiles"](lib, cell, view) or ""
            if raw.startswith("ERR|"):
                return {"ok": False, "path": "", "note": raw[4:] or "no cellview"}
            path, _, files_csv = raw.partition("|")
            files = [f for f in files_csv.split(",") if f]
            pick = _pick_source(files)
            if not pick:
                return {"ok": False, "path": path,
                        "note": "no writable text source in this view"}
            full = path + "/" + pick
            res = (self.ws["pyWriteFile"](full, text) or "").strip()
            if res != "OK":
                note = res[4:] if res.startswith("ERR|") else (res or "write failed")
                return {"ok": False, "path": full, "note": note}
            return {"ok": True, "path": full, "note": ""}

    def disconnect(self):
        """Close the workspace and stop the tunnel if we started it."""
        with self._lock:
            self._helpers_ready = False
            if self.ws is not None:
                try:
                    self.ws.close()
                except Exception:
                    pass
                self.ws = None
        if self.tunnel is not None and self.tunnel.poll() is None:
            self.tunnel.terminate()
            try:
                self.tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.tunnel.kill()
        self.tunnel = None
        self.info = {}

    def short_version(self):
        """'IC23.1-64b' out of the long CDS version banner."""
        import re
        m = re.search(r"virtuoso version\s+(\S+)", self.info.get("version", ""))
        return m.group(1) if m else (self.info.get("version", "")[:24] or "?")
