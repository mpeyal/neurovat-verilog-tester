"""GUI: validate a twin against Cadence Spectre (Cadence Virtuoso panel).

The "Validate twin" button is enabled only while the app is connected to
Cadence (VirtuosoLink).  It opens a window where the user picks the .va and
sets (or lets the tool suggest) the device type, pulse amplitudes and width,
read bias and drain bias; "Start validation" runs vatester.spectre.validate in
a worker thread.  Progress (Spectre runs and tests done of the total, elapsed
time, estimated time remaining) and the log are shown live; at the end the
verdict is shown and the results window (validate_results.py: test table,
Spectre-vs-twin plots, step check, netlists) opens in the app.  "Validation
results" under the button shows earlier runs, also without Cadence.

Threading: the worker talks to the GUI only through app.q ("val_*" messages,
handled on the render thread by handle()); it always posts "val_done".
Twin building is NOT done here: that stays with the assistant.
"""

import os
import threading
import time

import dearpygui.dearpygui as dpg

from . import design_system as ds
from .ui_constants import C_AMBER, C_GREEN, C_MUTED, C_RED, C_TEXT2, LEFT_W
from .va_scan import va_paths
from .validate_results import ValidateResults

FIELDS = [  # tag, label, kind, help
    ("val_twin", "twin", "text", "key of the twin (twins/ file or v1/v2/v3); empty = matched "
                                 "from the .va"),
    ("val_profile", "device type", "combo", "ecfet, memristor, fecap or fefet; empty = "
                                            "from the twin"),
    ("val_amp_pos", "positive / program amplitude", "num", "V (ECFET: gate current in A); "
                                                          "SI suffixes allowed: 50p, 1.2"),
    ("val_amp_neg", "negative / erase amplitude", "num", "V, negative (not used for ECFET, "
                                                         "FeCAP)"),
    ("val_width", "pulse width", "num", "s, e.g. 10n, 1u, 10m"),
    ("val_read_level", "read bias", "num", "V: read voltage (memristor) or read gate voltage "
                                           "(FeFET)"),
    ("val_v_ds", "drain bias V_ds", "num", "V, FeFET only"),
    ("val_pins", "ECFET pins", "text", "in=...,out=...,gate=... (read channel and gate "
                                       "ports); empty = found by name"),
]
KEYS = {"val_twin": "twin", "val_profile": "profile", "val_amp_pos": "amp_pos",
        "val_amp_neg": "amp_neg", "val_width": "width", "val_read_level": "read_level",
        "val_v_ds": "v_ds", "val_pins": "pins"}


def _fmt_time(s):
    if s is None:
        return "estimating..."
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min {sec:02d} s" if m else f"{sec} s")


class ValidatePanel:
    def __init__(self, app):
        self.app = app
        self.running = False
        self.stop_requested = False
        self.connected = None
        self.out_dir = None
        self.log_lines = []
        self.va_list = []
        self.results = ValidateResults(app)

    # ------------------------------------------------------------ building
    def build_button(self):
        """Inside the Cadence Virtuoso panel (call within its container)."""
        dpg.add_spacer(height=4)
        b = dpg.add_button(label="Validate twin vs Spectre", tag="btn_val_open",
                           width=-1, enabled=False, callback=self.open)
        with dpg.tooltip(b):
            dpg.add_text("Run the fixed test set of the twin's device type in Spectre\n"
                         "and compare (three layers, plots in a report).\n"
                         "Needs the Cadence connection above.")
        self.app._small("connect to Cadence to validate a twin", tag="val_hint",
                        wrap=LEFT_W - 36)
        r = dpg.add_button(label="Validation results", tag="btn_val_results", width=-1,
                           callback=lambda: self.results.show())
        with dpg.tooltip(r):
            dpg.add_text("Plots and tables of finished validations\n"
                         "(Spectre vs twin per test, step check, netlists).\n"
                         "No Cadence connection needed.")

    def build_dialog(self):
        with dpg.window(label="Validate twin against Spectre", tag="val_win", show=False,
                        width=700, height=860, no_collapse=True):
            ds.heading(self.app, "Validate twin against Spectre",
                       "Pick the Verilog-A model. Leave a value empty to use the "
                       "suggestion (twin calibration; for a FeFET, a Spectre Id-Vg "
                       "sweep). Runs on the Cadence host.")
            self.app._small("VERILOG-A MODEL", color=C_MUTED)
            dpg.add_combo([], tag="val_va", width=-1, callback=self._on_va)
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(init_width_or_weight=0.45)
                dpg.add_table_column(init_width_or_weight=0.55)
                for tag, label, kind, help_ in FIELDS:
                    with dpg.table_row():
                        dpg.add_text(label, color=C_TEXT2)
                        if kind == "combo":
                            w = dpg.add_combo(["", "ecfet", "memristor", "fecap", "fefet"],
                                              tag=tag, width=-1)
                        else:
                            w = dpg.add_input_text(tag=tag, width=-1)
                        with dpg.tooltip(w):
                            dpg.add_text(help_)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Suggest values", tag="val_btn_suggest",
                               callback=self.suggest)
                ai = dpg.add_button(label="Ask AI to fill", tag="val_btn_ai",
                                    callback=self.ask_ai)
                with dpg.tooltip(ai):
                    dpg.add_text("The assistant reads the .va, the available twins and\n"
                                 "the automatic suggestion, fills the EMPTY fields and\n"
                                 "gives a reason for each (see the log). Your values\n"
                                 "are kept; edit anything before Start. Uses the\n"
                                 "assistant's model; no Cadence login.")
                b = dpg.add_button(label="Start validation", tag="val_btn_start",
                                   callback=self.start)
                dpg.bind_item_theme(b, self.app.themes["primary"])
                dpg.add_button(label="Stop", tag="val_btn_stop", enabled=False,
                               callback=self.request_stop)
                dpg.add_button(label="Show results", tag="val_btn_report", enabled=False,
                               callback=self.open_report)
                dpg.add_button(label="Close", callback=lambda: dpg.hide_item("val_win"))
            dpg.add_spacer(height=4)
            dpg.add_progress_bar(tag="val_bar", default_value=0.0, width=-1, overlay="")
            dpg.add_text("", tag="val_status", color=C_TEXT2, wrap=660)
            with dpg.group(horizontal=True):
                dpg.add_text("", tag="val_verdict")
                dpg.add_spacer(width=16)
                fx = dpg.add_button(label="Fix twin with AI", tag="val_btn_fix", show=False,
                                    callback=self.fix_with_ai)
                with dpg.tooltip(fx):
                    dpg.add_text("The validation did not pass. The assistant edits the\n"
                                 "twin and NeuroVAT re-scores it against this run's\n"
                                 "Spectre results (no new Spectre run), until it\n"
                                 "passes or a few rounds are used up. Progress and\n"
                                 "the verdict show in the results window.")
            dpg.add_input_text(tag="val_log", multiline=True, readonly=True, width=-1,
                               height=200)
        if "dialog" in self.app.themes:
            dpg.bind_item_theme("val_win", self.app.themes["dialog"])
        self.results.build()

    # ------------------------------------------------------------ state
    def tick(self):
        """Render thread, every frame: follow the Cadence connection."""
        con = bool(self.app.virtuoso.connected)
        if con == self.connected:
            return
        self.connected = con
        if dpg.does_item_exist("btn_val_open"):
            dpg.configure_item("btn_val_open", enabled=con)
            dpg.set_value("val_hint", "" if con else "connect to Cadence to validate a twin")
        if dpg.does_item_exist("val_btn_start"):
            dpg.configure_item("val_btn_start", enabled=con and not self.running)

    def _log(self, line):
        import textwrap
        if len(line) > 96 and not line.startswith(" "):   # the log box does not wrap
            line = "\n".join(textwrap.wrap(line, 96, subsequent_indent="      "))
        self.log_lines.append(line)
        self.log_lines = self.log_lines[-400:]
        dpg.set_value("val_log", "\n".join(self.log_lines))

    def _va(self):
        rel = dpg.get_value("val_va")
        return os.path.join(self.app.workdir, rel) if rel else ""

    def _answers_from_fields(self):
        from .spectre.setup import parse_number
        a = {}
        for tag, key in KEYS.items():
            v = str(dpg.get_value(tag) or "").strip()
            if not v:
                continue
            if key in ("twin", "profile"):
                a[key] = v
            elif key == "pins":
                from .spectre.setup import parse_pins
                a[key] = parse_pins(v)
            else:
                try:
                    a[key] = parse_number(v)
                except ValueError:
                    raise ValueError(f"{key}: not a number: {v!r}")
        return a

    def _fill(self, answers):
        for tag, key in KEYS.items():
            v = answers.get(key)
            if v is None or v == "" or v == {}:
                dpg.set_value(tag, "")
            elif isinstance(v, dict):
                from .spectre.setup import pins_text
                dpg.set_value(tag, pins_text(v))
            elif isinstance(v, float):
                dpg.set_value(tag, f"{v:.6g}")
            else:
                dpg.set_value(tag, str(v))

    # ------------------------------------------------------------ actions
    def open(self, *_):
        if not self.app.virtuoso.connected:
            self.app.log("[validate] connect to Cadence first (Cadence Virtuoso panel)")
            return
        wd = self.app.workdir
        self.va_list = [os.path.relpath(p, wd) for p in va_paths(wd)]
        dpg.configure_item("val_va", items=self.va_list)
        if self.va_list and dpg.get_value("val_va") not in self.va_list:
            dpg.set_value("val_va", self.va_list[0])
            self._on_va()
        dpg.show_item("val_win")
        dpg.focus_item("val_win")

    def _on_va(self, *_):
        from .spectre.setup import load_answers
        va = self._va()
        self._fill(load_answers(va) or {} if va else {})
        if va:
            self._log(f"{os.path.basename(va)}: "
                      + ("saved setup loaded" if load_answers(va) else
                         "no saved setup; empty fields are suggested at the start"))

    def suggest(self, *_):
        if self.running:
            return
        va = self._va()
        if not va:
            self._log("pick a Verilog-A model first")
            return
        try:
            given = self._answers_from_fields()
        except ValueError as e:
            self._log(str(e))
            return
        self.running = True
        self._busy(True, "suggesting values ...")
        q, wd = self.app.q, self.app.workdir

        def worker():
            try:
                from .spectre.setup import suggest_answers
                from .spectre.validate import ssh_runner
                a, notes = suggest_answers(va, wd, ssh_runner(), given)
                q.put(("val_suggest", a, notes, None))
            except Exception as e:             # noqa: BLE001 - shown to the user
                q.put(("val_suggest", None, [], f"{type(e).__name__}: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _agent_clone(self):
        """A fresh assistant with the app's provider and credentials, so this
        one-off request does not join (or disturb) the chat conversation."""
        from .agent import ClaudeAgent
        src = self.app.agent
        a = ClaudeAgent(self.app.workdir)
        a.set_provider(src.provider)
        for k in ("override_kind", "override_value", "openai_key", "gemini_key"):
            setattr(a, k, getattr(src, k, None))
        return a

    def ask_ai(self, *_):
        if self.running:
            return
        va = self._va()
        if not va:
            self._log("pick a Verilog-A model first")
            return
        try:
            given = self._answers_from_fields()
        except ValueError as e:
            self._log(str(e))
            return
        self.running = True
        self._busy(True, "asking the AI to fill the empty fields ...")
        q, wd = self.app.q, self.app.workdir
        agent = self._agent_clone()

        def worker():
            try:
                from .spectre.ai_fill import build_request, parse_reply
                prompt, context, ports, keys = build_request(va, wd, given)
                q.put(("val_log", f"AI ({agent.backend_label()}): reading "
                                  f"{os.path.basename(va)} ..."))
                res = agent.send(prompt, context=context, allow_edits=False,
                                 allow_bash=False, timeout=300)
                if not res.get("ok"):
                    raise RuntimeError(res.get("error") or "no answer")
                a, reasons, rejected = parse_reply(res.get("text", ""), ports, keys, given)
                q.put(("val_ai", a, reasons, rejected, given, None))
            except Exception as e:             # noqa: BLE001 - shown, never stuck
                q.put(("val_ai", None, {}, {}, given, f"{type(e).__name__}: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def start(self, *_):
        if self.running:
            return
        if not self.app.virtuoso.connected:
            self._log("not connected to Cadence: connect in the Cadence Virtuoso panel first")
            return
        va = self._va()
        if not va:
            self._log("pick a Verilog-A model first")
            return
        try:
            given = self._answers_from_fields()
        except ValueError as e:
            self._log(str(e))
            return
        self.running, self.stop_requested = True, False
        self._busy(True, "starting ...")
        dpg.configure_item("val_btn_report", enabled=False)
        dpg.configure_item("val_btn_fix", show=False)
        dpg.set_value("val_verdict", "")
        dpg.set_value("val_bar", 0.0)
        q, wd = self.app.q, self.app.workdir

        def progress(d):
            if self.stop_requested:
                from .spectre.validate import Cancelled
                raise Cancelled("stopped by the user")
            q.put(("val_progress", d))

        def worker():
            report, out, err = None, None, None
            try:
                from .spectre.setup import save_answers, suggest_answers
                from .spectre.validate import ssh_runner, validate
                runner = ssh_runner()
                q.put(("val_log", "filling empty values with suggestions ..."))
                answers, notes = suggest_answers(va, wd, runner, given)
                for n in notes:
                    q.put(("val_log", n))
                save_answers(va, answers)
                q.put(("val_suggest", answers, [], None))
                report, out = validate(va, workspace=wd, runner=runner, open_report=False,
                                       answers=answers, progress=progress,
                                       log=lambda m: q.put(("val_log", m)))
            except Exception as e:             # noqa: BLE001 - reported, never stuck
                err = f"{type(e).__name__}: {e}"
            finally:
                q.put(("val_done", report, str(out) if out else None, err))

        threading.Thread(target=worker, daemon=True).start()

    def request_stop(self, *_):
        if self.running:
            self.stop_requested = True
            self._log("stopping at the next progress point ...")

    def fix_with_ai(self, *_):
        """Open this run in the results window and start the AI twin fix there."""
        if self.out_dir:
            self.results.show(self.out_dir)
            self.results.start_fix()

    def open_report(self, *_):
        """The results window on this run (report.html: its Open report button)."""
        if self.out_dir:
            self.results.show(self.out_dir)

    def _busy(self, on, msg):
        dpg.configure_item("val_btn_start", enabled=not on and bool(self.connected))
        dpg.configure_item("val_btn_suggest", enabled=not on)
        dpg.configure_item("val_btn_ai", enabled=not on)
        dpg.configure_item("val_btn_stop", enabled=on)
        dpg.set_value("val_status", msg)

    # ------------------------------------------------------------ queue
    def handle(self, item):
        kind = item[0]
        if kind.startswith("val_fix"):          # AI twin fix (results window)
            self.results.handle(item)
            return
        if kind == "val_log":
            self._log(item[1])
        elif kind == "val_suggest":
            a, notes, err = item[1], item[2], item[3]
            for n in notes:
                self._log(n)
            if err:
                self._log(f"suggestion failed: {err}")
            if a:
                self._fill(a)
            if self.running and dpg.get_value("val_status") == "suggesting values ...":
                self.running = False
                self._busy(False, "values suggested; edit if needed, then Start")
        elif kind == "val_ai":
            a, reasons, rejected, given, err = item[1:6]
            self.running = False
            if err:
                self._log(f"AI fill failed: {err}")
                self._busy(False, "AI fill failed (see the log); fill the fields or use "
                                  "Suggest values")
                return
            filled = {k: v for k, v in a.items() if k not in given}
            self._fill({**given, **filled})
            for k in filled:
                self._log(f"AI  {k} = {a[k]}" + (f"  - {reasons[k]}" if k in reasons else ""))
            for k, why in rejected.items():
                self._log(f"AI  {k} not used ({why})")
            if not a.get("twin"):
                self._log("AI: no twin fits this .va - a twin has to be written first "
                          "(ask the assistant to build one), then validate")
            self._busy(False, f"AI filled {len(filled)} field(s); check them, edit if "
                              "needed, then Start")
        elif kind == "val_progress":
            d = item[1]
            frac = d.get("fraction")
            if frac is None:
                frac = d["done"] / d["total"] if d["total"] else 0.0
            dpg.set_value("val_bar", frac)
            dpg.configure_item("val_bar", overlay=f"{100 * frac:.0f} %")
            stage = "" if d["stage"].startswith("Spectre ") else d["stage"] + "\n"
            twin = ""
            if d.get("twin_steps_total"):
                twin = (f"   twin steps {100 * d['twin_steps_done'] / d['twin_steps_total']:.0f} %")
            dpg.set_value("val_status",
                          f"{stage}Spectre runs {d['runs_done']}/{d['runs_total']}   "
                          f"tests scored {d['tests_done']}/{d['tests_total']}{twin}   "
                          f"elapsed {_fmt_time(d['elapsed_s'])}   "
                          f"remaining {_fmt_time(d['eta_s'])}")
        elif kind == "val_done":
            report, out, err = item[1], item[2], item[3]
            self.running = False
            self.out_dir = out
            if err:
                self._busy(False, "stopped" if "stopped by the user" in err else "failed")
                self._log(err)
                dpg.set_value("val_verdict", "STOPPED" if "stopped" in err else "ERROR")
                dpg.configure_item("val_verdict", color=C_AMBER)
                return
            v = report["verdict"]
            col = {"PASS": C_GREEN, "FAIL": C_RED}.get(v, C_AMBER)
            dpg.set_value("val_verdict", f"Overall: {v}")
            dpg.configure_item("val_verdict", color=col)
            dpg.set_value("val_bar", 1.0)
            dpg.configure_item("val_bar", overlay="100 %")
            self._busy(False, f"finished: {v}; report in {out}")
            dpg.configure_item("val_btn_report", enabled=True)
            dpg.configure_item("val_btn_fix", show=(v != "PASS"))
            self.app.log(f"[validate] {report['module']}: {v} -> {out}")
            self.open_report()
