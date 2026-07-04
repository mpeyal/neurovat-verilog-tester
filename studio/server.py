"""Zero-dependency web server — Python 3 standard library only.

Serves the bundled index.html and exposes the Bridge as POST /api/<name>.
No pip install required for this path. Run:  python run.py
"""

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bridge import Bridge

ROOT = os.path.dirname(os.path.abspath(__file__))
BRIDGE = Bridge()
SERVER_FLAG = b"<script>window.__NVAT_SERVER__=true;</script>"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    # ---- static ----------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._serve_index()
        # any other asset (none needed for the bundle, but kept for safety)
        safe = os.path.normpath(path).lstrip("/\\")
        full = os.path.join(ROOT, safe)
        if os.path.isfile(full) and full.startswith(ROOT):
            return self._serve_file(full)
        self.send_error(404, "not found")

    def _serve_index(self):
        with open(os.path.join(ROOT, "index.html"), "rb") as f:
            html = f.read()
        # tell the front-end a backend is present -> routes to /api/*
        if b"</head>" in html:
            html = html.replace(b"</head>", SERVER_FLAG + b"</head>", 1)
        else:
            html = SERVER_FLAG + html
        self._send(200, "text/html; charset=utf-8", html)

    def _serve_file(self, full):
        import mimetypes
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            self._send(200, ctype, f.read())

    # ---- api -------------------------------------------------------------
    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if not path.startswith("/api/"):
            return self.send_error(404, "not found")
        name = path[len("/api/"):].strip("/")
        method = getattr(BRIDGE, name, None)
        if method is None or name.startswith("_") or not callable(method):
            return self.send_error(404, "no such method: " + name)

        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            args = json.loads(raw or b"{}")
        except Exception:
            args = {}

        try:
            result = method(args)
            self._send(200, "application/json", json.dumps(result).encode("utf-8"))
        except Exception as e:  # surfaces in the front-end log, twin takes over
            self._send(500, "text/plain; charset=utf-8", str(e).encode("utf-8"))

    # ---- helper ----------------------------------------------------------
    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(port=8000, open_browser=True):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = "http://127.0.0.1:%d/" % port
    print("\n  NeuroVAT Studio  ->  " + url + "   (Ctrl+C to stop)\n")
    if open_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
        httpd.shutdown()


if __name__ == "__main__":
    serve()
