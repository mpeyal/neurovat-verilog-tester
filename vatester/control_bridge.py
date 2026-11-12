"""Control bridge - drive the running NeuroVAT GUI from outside, skillbridge-style.

This is the local analog of the Cadence skillbridge in vatester/virtuoso.py: there
the app is a *client* of Virtuoso's SKILL server; here the app IS the server. A
driver (the in-app agent via its Bash tool, or an external `claude -p` session)
issues high-level commands - run a sim, set a field, capture a screenshot, dump
the plotted data - and reads back a JSON result plus fresh snapshots/PNGs. That
closes the observe -> diagnose -> edit-code -> reload -> re-run loop against the
LIVE app, with DearPyGui still rendering so a human can watch.

Transport: a one-command-at-a-time file mailbox under <workdir>/control/.
  - driver writes   control/command.json  = {"id", "cmd", "args"}   (atomic rename)
  - bridge writes   control/result.json   = {"id", "cmd", "ok", "data", "error"}

Why file-mailbox and not a socket: DearPyGui is single-threaded and every fix in
this app marshals work onto the render thread. poll() is called once per frame
from App.run(), so every command EXECUTES ON THE RENDER THREAD - no locks, no
cross-thread DPG access. Long commands (run/train) don't block the frame: the
bridge starts them and then watches sim_running/trainer_running across frames,
writing the result only when the work actually finishes.

Security: the command set is a FIXED whitelist. There is no eval/exec of caller
strings; `set`/`get` only touch existing DearPyGui widget tags. The mailbox is
local to the workspace. (See CLAUDE.md.)
"""

import json
import os
import time

import dearpygui.dearpygui as dpg


class ControlBridge:
    def __init__(self, app, ctrl_dir=None, active=True):
        self.app = app
        self.dir = ctrl_dir or os.path.join(app.workdir, "control")
        self.cmd_path = os.path.join(self.dir, "command.json")
        self.res_path = os.path.join(self.dir, "result.json")
        self._last_id = None            # highest command id already handled
        self._pending = None            # async command in flight (dict) or None
        self._frame = 0
        self.active = False             # gate poll(); flipped by start()/stop()
        if active:
            self.start()

    # -- runtime enable / disable (Tools menu toggle) ---------------------
    def start(self):
        """Open the mailbox and begin accepting commands."""
        if self.active:
            return
        os.makedirs(self.dir, exist_ok=True)
        self._clear_mailbox()           # drop any stale command/result
        self._pending = None
        self.active = True
        self.app.log(f"[bridge] ON - control mailbox ready at {self.dir}")

    def stop(self):
        """Stop accepting commands and remove the mailbox files."""
        if not self.active:
            return
        self.active = False
        self._pending = None
        self._clear_mailbox()
        self.app.log("[bridge] OFF - control mailbox closed")

    def _clear_mailbox(self):
        for p in (self.cmd_path, self.res_path):
            try:
                os.remove(p)
            except OSError:
                pass

    # -- called once per frame from App.run() -----------------------------
    def poll(self):
        if not self.active:             # disabled via the Tools menu
            return
        self._frame += 1
        if self._pending is not None:
            self._advance_pending()
            return
        cmd = self._read_command()
        if cmd is None:
            return
        self._dispatch(cmd)

    # -- mailbox io -------------------------------------------------------
    def _read_command(self):
        try:
            with open(self.cmd_path, "r", encoding="utf-8") as f:
                cmd = json.load(f)
        except (OSError, ValueError):
            return None
        cid = cmd.get("id")
        if cid is None or cid == self._last_id:
            return None                 # nothing new / already handled
        self._last_id = cid
        return cmd

    def _write_result(self, cid, name, ok, data=None, error=None):
        payload = {"id": cid, "cmd": name, "ok": bool(ok),
                   "data": data if data is not None else {}, "error": error,
                   "sim_running": self.app.sim_running,
                   "trainer_running": self.app.trainer_running}
        tmp = self.res_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.res_path)
        except OSError as e:
            self.app.log(f"[bridge] result write failed: {e}")
        tag = "ok" if ok else f"ERR {error}"
        self.app.log(f"[bridge] {name} -> {tag}")

    # -- dispatch ---------------------------------------------------------
    def _dispatch(self, cmd):
        cid = cmd.get("id")
        name = str(cmd.get("cmd", "")).lower()
        args = cmd.get("args") or {}
        handler = self._HANDLERS.get(name)
        if handler is None:
            self._write_result(cid, name, False,
                               error=f"unknown command '{name}'. known: "
                                     + ", ".join(sorted(self._HANDLERS)))
            return
        try:
            result = handler(self, args)
        except Exception as e:          # never let a bad command crash the frame
            self._write_result(cid, name, False, error=f"{type(e).__name__}: {e}")
            return
        if isinstance(result, _Async):
            result.id = cid
            result.name = name
            result.deadline = time.perf_counter() + result.timeout_s
            self._pending = result
        else:
            self._write_result(cid, name, True, data=result)

    def _advance_pending(self):
        p = self._pending
        try:
            if p.done():
                data = p.finalize() or {}
                self._pending = None
                self._write_result(p.id, p.name, True, data=data)
            elif time.perf_counter() > p.deadline:
                self._pending = None
                self._write_result(p.id, p.name, False,
                                   error=f"timed out after {p.timeout_s:.0f}s")
        except Exception as e:
            self._pending = None
            self._write_result(p.id, p.name, False,
                               error=f"{type(e).__name__}: {e}")

    # =====================================================================
    # command handlers  (return a dict = done now, or _Async = watch frames)
    # =====================================================================
    def _cmd_ping(self, args):
        return {"pong": True, "frame": self._frame}

    def _cmd_state(self, args):
        a = self.app
        vp = {}
        try:
            vp = {"w": dpg.get_viewport_client_width(),
                  "h": dpg.get_viewport_client_height()}
        except Exception:
            pass
        tab = None
        if dpg.does_item_exist("center_tabs"):
            tab = dpg.get_value("center_tabs")
        return {"sim_running": a.sim_running, "trainer_running": a.trainer_running,
                "models_enabled": list(a._enabled_keys()),
                "n_results": len(a.results),
                "trainer_built": a.trainer is not None,
                "center_tab": tab, "viewport": vp, "frame": self._frame}

    def _cmd_get(self, args):
        tag = args["tag"]
        if not dpg.does_item_exist(tag):
            raise KeyError(f"no widget '{tag}'")
        return {"tag": tag, "value": dpg.get_value(tag)}

    def _cmd_set(self, args):
        tag, val = args["tag"], args["value"]
        if not dpg.does_item_exist(tag):
            raise KeyError(f"no widget '{tag}'")
        if tag.startswith("nt_"):
            ok = self.app._nt_set_control(tag, val)
            return {"tag": tag, "value": dpg.get_value(tag), "applied": ok}
        cur = dpg.get_value(tag)
        if isinstance(cur, bool):
            val = str(val).lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            val = int(float(val))
        elif isinstance(cur, float):
            val = float(val)
        dpg.set_value(tag, val)
        if tag.startswith('p_'):
            user_data = dpg.get_item_user_data(tag)
            if isinstance(user_data, (tuple, list)) and len(user_data) == 3:
                self.app._on_param_edit(tag, val, user_data)
        elif tag.startswith('cb_file_'):
            self.app._on_file_toggle(tag, val, tag[len('cb_file_'):])
        elif tag == 'param_model_sel':
            self.app.rebuild_param_panel()
        elif tag == 'gen_combo':
            self.app.rebuild_gen_params()
        elif tag.startswith('g_'):
            user_data = dpg.get_item_user_data(tag)
            if isinstance(user_data, (tuple, list)) and len(user_data) == 2:
                self.app._on_gen_edit(tag, val, user_data)
        return {"tag": tag, "value": dpg.get_value(tag), "applied": True}

    def _cmd_tab(self, args):
        name = args["name"]
        if not dpg.does_item_exist("center_tabs"):
            raise RuntimeError("center_tabs not built")
        dpg.set_value("center_tabs", name)
        try:
            self.app._on_center_tab()
        except Exception:
            pass
        return {"center_tab": dpg.get_value("center_tabs")}

    def _cmd_snapshot(self, args):
        a = self.app
        snap = a._write_agent_snapshot()
        nt = None
        if a.trainer is not None:
            try:
                a._write_nt_snapshot()
                nt = os.path.join(a.workdir, "results", "neuro_snapshot.json")
            except Exception:
                nt = None
        summary = [{"model": r.label,
                    "R_ohm": [float(r.R.min()), float(r.R.max())],
                    "G_uS": [float(r.G.min() * 1e6), float(r.G.max() * 1e6)]}
                   for r in a.results]
        return {"agent_snapshot": snap, "neuro_snapshot": nt,
                "results": summary}

    def _cmd_shot(self, args):
        path = args.get("path") or os.path.join(
            self.app.workdir, "results", "control_shot.png")
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # output_frame_buffer saves the frame; give it 3 frames to flush to disk.
        dpg.output_frame_buffer(file=path)
        start = self._frame

        def done():
            return self._frame - start >= 3

        def finalize():
            ok = os.path.isfile(path) and os.path.getsize(path) > 2000
            return {"path": path, "saved": ok,
                    "bytes": os.path.getsize(path) if os.path.isfile(path) else 0}
        return _Async(done, finalize, timeout_s=10)

    def _cmd_run(self, args):
        self.app.on_run()
        return self._await_sim("run", write_snapshot=True,
                               timeout_s=args.get("timeout_s", 180))

    def _cmd_stdp(self, args):
        self.app.on_plot_stdp()
        return self._await_sim("stdp", write_snapshot=True,
                               timeout_s=args.get("timeout_s", 240))

    def _cmd_reload(self, args):
        self.app._reload_models()
        return {"reloaded": True}

    def _cmd_log(self, args):
        """Return the tail of the console log and persist it to a file."""
        lines = list(self.app._log_lines)
        n = int(args.get("n", 200))
        path = os.path.join(self.app.workdir, "results", "session_log.txt")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError:
            path = None
        return {"total": len(lines), "file": path, "lines": lines[-n:]}

    def _cmd_noise_plot(self, args):
        panel = self.app.noise_panel
        try:
            panel.calculate()
        except Exception as exc:
            panel._error(exc)
            raise
        dpg.set_value('center_tabs', 'tab_noise')
        self.app._on_center_tab()
        return {'model': panel.data['model_key'], 'budget': panel.data['budget'],
                'points': len(panel.data['frequency_hz'])}

    def _cmd_noise_data(self, args):
        self._cmd_noise_plot({})
        if self.app.noise_panel.data is None:
            raise ValueError('No noise plot data. Run noise_plot first.')
        return self.app.noise_panel.data

    def _cmd_noise_export(self, args):
        return self.app.noise_panel.export_data()

    def _cmd_check_va(self, args):
        return self.app.check_va_consistency()

    def _cmd_openvaf(self, args):
        return self.app.openvaf_compile_va(args.get("path"))

    def _cmd_fit(self, args):
        return self.app.fit_curve_csv(args["path"])

    def _cmd_nt_build(self, args):
        if self.app.trainer_running:
            raise RuntimeError("Trainer is busy.")
        self.app.on_nt_build()
        if self.app._nt_is_stream():
            return {"stream_network": "new"}
        if self.app.trainer is None:
            raise RuntimeError(dpg.get_item_configuration('nt_status')['label'])
        return {"trainer_built": self.app.trainer is not None}

    def _cmd_nt_load(self, args):
        self.app.on_nt_load(args["path"])
        return {"trainer_built": self.app.trainer is not None}

    def _cmd_nt_train(self, args):
        if self.app.trainer_running:
            raise RuntimeError("Trainer is busy.")
        if self.app._nt_is_stream():
            return self._cmd_nt_stream(args)
        self.app.on_nt_train()
        if not self.app.trainer_running:
            raise RuntimeError(dpg.get_item_configuration('nt_status')['label'])
        return self._await_trainer("nt_train", write_nt=True,
                                   timeout_s=args.get("timeout_s", 900))

    def _cmd_nt_test(self, args):
        if self.app.trainer_running:
            raise RuntimeError("Trainer is busy.")
        if self.app._nt_is_stream():
            return self._cmd_nt_stream(args, learn=False)
        self.app.on_nt_test()
        if not self.app.trainer_running:
            raise RuntimeError(dpg.get_item_configuration('nt_status')['label'])
        return self._await_trainer("nt_test", write_nt=True,
                                   timeout_s=args.get("timeout_s", 300))

    def _cmd_nt_stream(self, args, learn=True):
        app = self.app
        if app.trainer_running:
            raise RuntimeError("Trainer is busy.")
        app._nt_set_control("nt_mode", "unsupervised stream")
        app._nt_stream_start(learn)
        if not app.trainer_running:
            raise RuntimeError(app._nt_stream_error or "stream did not start")

        def finalize():
            if app._nt_stream_error:
                raise RuntimeError(app._nt_stream_error)
            return {"stream": app._nt_stream_last}
        return _Async(lambda: not app.trainer_running, finalize,
                      timeout_s=args.get("timeout_s", 3600))

    def _cmd_nt_stream_test(self, args):
        return self._cmd_nt_stream(args, learn=False)

    def _cmd_nt_power_save(self, args):
        # no caller-supplied path: always results/energy_reports/<timestamp>.txt
        path = self.app.on_nt_power_save()
        if path is None:
            raise RuntimeError("no energy report yet: train or evaluate first")
        return {"path": path}

    # -- shared async waiters --------------------------------------------
    def _await_sim(self, name, write_snapshot, timeout_s):
        app = self.app

        def done():
            return not app.sim_running

        def finalize():
            data = {"n_results": len(app.results)}
            if write_snapshot:
                data["agent_snapshot"] = app._write_agent_snapshot()
                data["results"] = [
                    {"model": r.label,
                     "R_ohm": [float(r.R.min()), float(r.R.max())],
                     "G_uS": [float(r.G.min() * 1e6), float(r.G.max() * 1e6)]}
                    for r in app.results]
            return data
        return _Async(done, finalize, timeout_s=timeout_s)

    def _await_trainer(self, name, write_nt, timeout_s):
        app = self.app

        def done():
            return not app.trainer_running

        def finalize():
            data = {"trainer_built": app.trainer is not None}
            error = getattr(app, '_nt_training_error' if name == 'nt_train' else '_nt_eval_error', None)
            if error:
                raise RuntimeError(error)
            if name == 'nt_train':
                data['training'] = getattr(app, '_nt_last_training', {})
            if write_nt and app.trainer is not None:
                try:
                    app._write_nt_snapshot()
                    data["neuro_snapshot"] = os.path.join(
                        app.workdir, "results", "neuro_snapshot.json")
                except Exception:
                    pass
            return data
        return _Async(done, finalize, timeout_s=timeout_s)

    _HANDLERS = {
        "ping": _cmd_ping, "state": _cmd_state,
        "get": _cmd_get, "set": _cmd_set, "tab": _cmd_tab,
        "snapshot": _cmd_snapshot, "shot": _cmd_shot,
        "run": _cmd_run, "stdp": _cmd_stdp, "reload": _cmd_reload,
        "nt_build": _cmd_nt_build, "nt_load": _cmd_nt_load,
        "nt_train": _cmd_nt_train, "nt_test": _cmd_nt_test,
        "nt_stream": _cmd_nt_stream, "nt_stream_test": _cmd_nt_stream_test,
        "nt_power_save": _cmd_nt_power_save,
        "check_va": _cmd_check_va, "openvaf": _cmd_openvaf, "fit": _cmd_fit,
        "log": _cmd_log,
        "noise_plot": _cmd_noise_plot, "noise_data": _cmd_noise_data,
        "noise_export": _cmd_noise_export,
    }


class _Async:
    """A command whose completion is watched across frames."""
    def __init__(self, done, finalize, timeout_s=180):
        self.done = done
        self.finalize = finalize
        self.timeout_s = float(timeout_s)
        self.id = None
        self.name = None
        self.deadline = None
