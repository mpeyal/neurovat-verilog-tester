#!/usr/bin/env python3
"""
connect_test.py  --  verify a skillbridge connection to Cadence Virtuoso.

Intended to run on WINDOWS, talking to Virtuoso on the Linux box
(coen-cassia.boisestate.edu / 10.24.1.45) through an SSH tunnel.

SETUP (run once, leave open, in a separate Windows terminal):
    ssh -N -L 7777:/tmp/skill-server-mahmudulpeyal.sock mahmudulpeyal@coen-cassia.boisestate.edu

The skillbridge client on Windows auto-uses TCP localhost:7777, which the
tunnel forwards to the Linux Unix socket /tmp/skill-server-mahmudulpeyal.sock.
(Use the PER-USER socket, not the shared -default.sock: on this multi-user host
the default socket is owned by another account and just resets the connection.)

On the Linux side, make sure the bridge is up first (type  skill()  in the CIW).

REQUIREMENT on Windows:  pip install skillbridge
RUN:                     python connect_test.py
"""

import sys


def main() -> int:
    try:
        from skillbridge import Workspace
    except ImportError:
        print("[FAIL] skillbridge is not installed on this machine.")
        print("       Fix:  pip install skillbridge")
        return 1

    # On Windows, Workspace.open() defaults to TCP localhost:7777 -- exactly
    # what the SSH tunnel exposes. (Pass a numeric id to use a different port.)
    try:
        ws = Workspace.open()
    except (ConnectionRefusedError, OSError) as exc:
        print(f"[FAIL] Could not reach the skillbridge server: {exc}")
        print("       Checklist:")
        print("         1. Is the SSH tunnel running? (the ssh -L ... command)")
        print("         2. Is Virtuoso open on Linux and did you run  skill()  in the CIW?")
        print("         3. Is the tunnel on port 7777 (the skillbridge default)?")
        return 1

    try:
        version = ws['getVersion']()
        window = ws['hiGetCurrentWindow']()
        try:
            libs = ws.dd.get_lib_list()
            lib_names = [l.name for l in libs][:10]
        except Exception as exc:
            lib_names = [f"(could not enumerate libraries: {exc})"]
    except Exception as exc:
        print(f"[FAIL] Connected, but a SKILL call failed: {exc}")
        return 1

    print("[OK] Bridge is live.")
    print(f"     Virtuoso version : {version}")
    print(f"     Current window   : {window}")
    print(f"     Libraries (<=10) : {', '.join(lib_names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
