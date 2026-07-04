"""NeuroVAT Studio launcher.

    python run.py              # local web server + opens your browser  (no installs)
    python run.py --port 9000  # pick a port
    python run.py --no-open    # don't auto-open the browser
    python run.py --desktop    # native window instead (needs: pip install pywebview)
"""

import sys


def main(argv):
    if "--desktop" in argv:
        import desktop
        return desktop.main()

    port = 8000
    if "--port" in argv:
        try:
            port = int(argv[argv.index("--port") + 1])
        except (IndexError, ValueError):
            pass
    open_browser = "--no-open" not in argv

    import server
    server.serve(port=port, open_browser=open_browser)


if __name__ == "__main__":
    main(sys.argv[1:])
