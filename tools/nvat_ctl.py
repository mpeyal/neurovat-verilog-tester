#!/usr/bin/env python
"""nvat_ctl - command-line client for the running NeuroVAT GUI's control bridge.

Launch the app (the bridge is on by default; --no-bridge starts it off, and the
Tools menu toggles it live):
    python run_gui.py

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
    python tools/nvat_ctl.py noise_plot          # refresh the Noise & DR canvases
    python tools/nvat_ctl.py noise_data          # plotted arrays + parameters + budget
    python tools/nvat_ctl.py noise_export        # CSV spectra/DR and JSON snapshot
    python tools/nvat_ctl.py nt_build            # build the trainer crossbar
    python tools/nvat_ctl.py nt_train            # train (waits for completion)
    python tools/nvat_ctl.py nt_test
    python tools/nvat_ctl.py nt_stream           # switch to mode 'unsupervised stream' and train
    python tools/nvat_ctl.py nt_stream_test      # same patterns, new stream, STDP off
    python tools/nvat_ctl.py set nt_st_input "scanned images"   # or "Poisson"; image rows:
                                                 # nt_st_imgset / _img_size / _img_pick / _img_distract
    python tools/nvat_ctl.py set nt_st_code latency  # image encoding: column scan | row scan |
                                                 # latency | frozen Poisson (window: nt_st_code_win; frozen Poisson sets nt_st_w0 0.5, noise 10 Hz)
    python tools/nvat_ctl.py tab nt_st_tab_raster  # stream views: _raster _net _pat _learn _weights _energy _robust
    python tools/nvat_ctl.py set nt_st_view_t 3  # replay position (s into the last run)
    python tools/nvat_ctl.py set nt_st_play 1    # play the replay (0 = pause)
    python tools/nvat_ctl.py nt_load_mnist       # load <workspace>/datasets/mnist.npz (image set 'dataset')
    python tools/nvat_ctl.py nt_robust           # robustness sweep of the TRAINED stream network
                                                 # (STDP off, one fixed stream); waits, returns the table
    python tools/nvat_ctl.py nt_robust 10        # ... 10 s per level (else nt_st_rb_seconds, default 30)
    python tools/nvat_ctl.py set nt_st_rb_jitter "1,3,5"   # levels: _jitter ms, _delete %, _noise Hz
                                                 # ('' = paper levels + trained value); nt_st_rb_untrained 0/1
    python tools/nvat_ctl.py tab nt_st_tab_robust  # the Robustness tab (plots + table)
    python tools/nvat_ctl.py nt_robust 30 retrain  # RETRAIN at each level: a fresh copy trained at the
                                                 # level (STDP on), then tested there; positional:
                                                 # seconds retrain train_s repeats (retrain 1/0 or
                                                 # retrain/recall; train_s 0 = the trained seconds)
    python tools/nvat_ctl.py nt_robust 30 1 60 3   # ... 60 s training per level, 3 trainings (mean, min-max)
    python tools/nvat_ctl.py set nt_st_rb_mode recall  # show the kept recall result (or 'retrain ...')
    python tools/nvat_ctl.py nt_robust_raster 2  # show swept level #2 (or its text) in the Raster tab
                                                 # (retrain mode: that level's retrained weights)
    python tools/nvat_ctl.py nt_power_save       # Power tab report -> results/energy_reports/

Typical closed loop the driver runs:
    set params -> run -> read results/agent_snapshot.json -> (if wrong) edit the
    twin in simulation_core/ -> reload -> run again -> shot to eyeball it.

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
    "nt_robust": ["seconds", "retrain", "train_s", "repeats"], "nt_robust_raster": ["entry"],
}


def _cmd_dir(workspace):
    return os.path.join(os.path.abspath(workspace), "control")


def send(workspace, name, args, timeout=None, poll=0.05):
    d = _cmd_dir(workspace)
    if not os.path.isdir(d):
        raise SystemExit(f"[nvat_ctl] no control mailbox at {d}\n"
                         f"  is the app running (python run_gui.py) with the "
                         f"bridge on (not --no-bridge; Tools menu)?")
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
    # default client timeout is generous for long ops (train); the bridge has its
    # own per-command deadline and will always answer or time out.
    deadline = time.perf_counter() + (timeout if timeout else 1200)
    # Windows may deny replacement while the render thread is reading the old
    # command. Retry the SAME atomic hand-off, never publish a partial file or
    # dispatch a second command. Bound retries so real permission errors surface.
    publish_deadline = min(deadline, time.perf_counter() + 2.0)
    while True:
        try:
            os.replace(tmp, cmd_path)
            break
        except PermissionError:
            if time.perf_counter() >= publish_deadline:
                raise
            time.sleep(min(poll, 0.05))

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
