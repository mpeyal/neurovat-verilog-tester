#!/usr/bin/env python
"""nvat_ctl - command-line client for the running NeuroVAT GUI's control bridge.

Launch the app with the bridge on:
    python run_gui.py --bridge

Then drive it from any shell (this is what the agent / Claude Code calls):
    python tools/nvat_ctl.py ping
    python tools/nvat_ctl.py state
    python tools/nvat_ctl.py set nt_epochs 25
    python tools/nvat_ctl.py get nt_epochs
    python tools/nvat_ctl.py tab tab_results
    python tools/nvat_ctl.py run                 # runs a transient, waits, dumps data
    python tools/nvat_ctl.py stdp                # STDP sweep
    python tools/nvat_ctl.py snapshot            # dump plotted data + weights to JSON
    python tools/nvat_ctl.py shot results/x.png  # screenshot the live window
    python tools/nvat_ctl.py reload              # hot-reload the model twins after an edit
    python tools/nvat_ctl.py nt_build            # build the trainer crossbar
    python tools/nvat_ctl.py nt_train            # train (waits for completion)
    python tools/nvat_ctl.py nt_test

Typical closed loop the driver runs:
    set params -> run -> read results/agent_snapshot.json -> (if wrong) edit the
    twin in ecfet/ -> reload -> run again -> shot to eyeball it.

The command result (JSON) is printed to stdout; exit code is 0 on ok, 1 on error.
Transport is a file mailbox under <workspace>/control/ (command.json/result.json).
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# commands that only need (tag) / (tag value) / (path) positional args
_ARG_SPECS = {
    "get": ["tag"], "set": ["tag", "value"], "tab": ["name"],
    "shot": ["path"], "nt_load": ["path"],
    "fit": ["path"], "openvaf": ["path"],       # openvaf path optional
}


def _cmd_dir(workspace):
    return os.path.join(os.path.abspath(workspace), "control")


def send(workspace, name, args, timeout=None, poll=0.05):
    d = _cmd_dir(workspace)
    if not os.path.isdir(d):
        raise SystemExit(f"[nvat_ctl] no control mailbox at {d}\n"
                         f"  is the app running with:  python run_gui.py --bridge ?")
    cmd_path = os.path.join(d, "command.json")
    res_path = os.path.join(d, "result.json")
    cid = time.time_ns()
    # remove any prior result so we only accept OUR id
    try:
        os.remove(res_path)
    except OSError:
        pass
    tmp = cmd_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"id": cid, "cmd": name, "args": args}, f)
    os.replace(tmp, cmd_path)           # atomic hand-off

    # default client timeout is generous for long ops (train); the bridge has its
    # own per-command deadline and will always answer or time out.
    deadline = time.perf_counter() + (timeout if timeout else 1200)
    while time.perf_counter() < deadline:
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                res = json.load(f)
        except (OSError, ValueError):
            time.sleep(poll)
            continue
        if res.get("id") == cid:
            return res
        time.sleep(poll)
    raise SystemExit(f"[nvat_ctl] timed out waiting for '{name}' result "
                     f"(is the app still running?)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", help="bridge command (ping/state/run/set/get/...)")
    ap.add_argument("rest", nargs="*", help="positional args for the command")
    ap.add_argument("--workspace", default=ROOT,
                    help="app workspace (default: repo root)")
    ap.add_argument("--json", action="store_true",
                    help="raw args as a JSON object instead of positionals")
    ap.add_argument("--timeout", type=float, default=None,
                    help="client wait seconds (default 1200)")
    a = ap.parse_args()

    if a.json:
        args = json.loads(a.rest[0]) if a.rest else {}
    else:
        keys = _ARG_SPECS.get(a.cmd, [])
        args = {k: v for k, v in zip(keys, a.rest)}

    res = send(a.workspace, a.cmd, args, timeout=a.timeout)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res.get("ok") else 1)


if __name__ == "__main__":
    main()
