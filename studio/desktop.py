"""Desktop launcher — native window via pywebview (optional).

    pip install pywebview
    python run.py --desktop

Exposes the Bridge in-process (no server, no URL). The front-end detects
window.pywebview.api and routes every backend.call() straight to Python.
"""

import os

from bridge import Bridge

ROOT = os.path.dirname(os.path.abspath(__file__))


def main(width=1480, height=920):
    import webview  # imported lazily so the server path needs no install
    webview.create_window(
        "NeuroVAT Studio",
        url=os.path.join(ROOT, "index.html"),
        js_api=Bridge(),
        width=width,
        height=height,
        min_size=(1100, 720),
    )
    webview.start()


if __name__ == "__main__":
    main()
