"""GUI: results of a twin-vs-Spectre validation (vatester.spectre.validate).

A window that shows a finished run without leaving the app:

  * the run (any folder under results/validation with a summary.json) and its
    overall verdict, twin, device type and Spectre version;
  * a table of the tests: verdict, waveform and read errors, the Spectre step
    the twin was judged against and whether Spectre had converged there;
  * "Twin vs Spectre": for the selected test, three linked plots - the drive
    applied to both, Spectre and the twin overlaid, and the error (waveform
    and at the reads) - zoomable, x axes linked;
  * "Spectre step check": every test's Spectre run at each step against its
    finest (Layer 2);
  * "Spectre netlist": the exact netlist Spectre ran for the test (Spectre is
    run directly from this netlist on the host; Maestro/ADE is not used).

Data come from the files validate() writes: summary.json, l1_<test>.npz
(loaded with allow_pickle=False) and spectre/<test>__s<i>.scs.  Runs from
before those files existed show the table and the report PNGs only.
Render thread only (no workers): loading one run takes well under a second.
"""

import json
import os
import webbrowser

import numpy as np
import dearpygui.dearpygui as dpg

from .ui_constants import C_AMBER, C_GREEN, C_MUTED, C_RED, C_TEXT2

C_SPECTRE = (225, 230, 240)
C_TWIN = (255, 140, 60)
C_READ = (60, 200, 140)
C_ERR = (110, 160, 255)
VERDICT_COL = {"PASS": C_GREEN, "FAIL": C_RED}
MAX_POINTS = 20000


def envelope(t, y, max_points=MAX_POINTS):
    """At most ~max_points of (t, y) for drawing: the min and max of each bin
    in time order, so pulses and spikes survive the reduction."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    n = len(t)
    if n <= max_points:
        return t, y
    bins = max(1, max_points // 2)
    size = int(np.ceil(n / bins))
    pad = bins * size - n
    yp = np.concatenate([y, np.full(pad, np.nan)]) if pad else y
    blk = yp.reshape(bins, size)
    with np.errstate(invalid="ignore"):
        i_min = np.nanargmin(np.where(np.isnan(blk), np.inf, blk), axis=1)
        i_max = np.nanargmax(np.where(np.isnan(blk), -np.inf, blk), axis=1)
    base = np.arange(bins) * size
    idx = np.unique(np.clip(np.concatenate([base + i_min, base + i_max, [0, n - 1]]),
                            0, n - 1))
    return t[idx], y[idx]


def list_runs(workdir):
    """Result folders (newest first) that hold a summary.json."""
    root = os.path.join(workdir, "results", "validation")
    if not os.path.isdir(root):
        return []
    runs = [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isfile(os.path.join(root, d, "summary.json"))]
    return sorted(runs, key=os.path.getmtime, reverse=True)


def _pct(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(v) else f"{v:.3g}"


class ValidateResults:
    def __init__(self, app):
        self.app = app
        self.run_dir = None
        self.summary = {}
        self.test = None
        self.runs = []
        self.fixing = False
        self.fix_stop = False
        self.fix_lines = []
        self._fix_agent = None

    @staticmethod
    def _rounds():
        from .spectre.fix_twin import ROUNDS
        return ROUNDS

    # ------------------------------------------------------------ AI twin fix
    def _fix_log(self, line):
        self.fix_lines = (self.fix_lines + [line])[-500:]
        dpg.set_value("vres_fix_log", "\n".join(self.fix_lines))

    def start_fix(self, *_):
        if self.fixing or not self.run_dir:
            return
        import threading
        run_dir, wd, q = self.run_dir, self.app.workdir, self.app.q
        agent = self.app.validate_panel._agent_clone()
        self._fix_agent = agent
        self.fixing, self.fix_stop, self.fix_lines = True, False, []
        dpg.configure_item("vres_btn_fix", enabled=False)
        dpg.configure_item("vres_btn_fix_stop", enabled=True)
        dpg.configure_item("vres_fix_hdr", default_open=True)
        dpg.set_value("vres_fix_hdr", True)
        dpg.set_value("vres_fix_status", f"fixing the twin with {agent.backend_label()} ...")
        dpg.configure_item("vres_fix_status", color=C_TEXT2)

        def worker():
            res, err = None, None
            try:
                from .spectre.fix_twin import fix_twin
                res = fix_twin(run_dir, wd, agent, log=lambda m: q.put(("val_fix_log", m)),
                               stop=lambda: self.fix_stop)
            except Exception as e:             # noqa: BLE001 - shown, never stuck
                err = f"{type(e).__name__}: {e}"
            finally:
                q.put(("val_fix_done", res, err))

        threading.Thread(target=worker, daemon=True).start()

    def stop_fix(self, *_):
        if self.fixing:
            self.fix_stop = True
            if self._fix_agent is not None:
                self._fix_agent.stop()
            self._fix_log("stopping after the current step ...")

    def handle(self, item):
        kind = item[0]
        if kind == "val_fix_log":
            self._fix_log(item[1])
            return
        res, err = item[1], item[2]
        self.fixing = False
        dpg.configure_item("vres_btn_fix_stop", enabled=False)
        if err:
            self._fix_log(f"fix failed: {err}")
            dpg.set_value("vres_fix_status", f"fix failed: {err}")
            dpg.configure_item("vres_fix_status", color=C_RED)
            dpg.configure_item("vres_btn_fix", enabled=True)
            return
        v = res["verdict"]
        lines = [{"FIXED": f"FIXED in round {res['rounds']}: the twin now passes against "
                           f"the saved Spectre results ({os.path.basename(res['twin_file'])} "
                           f"was edited; the original is in fix/).",
                  "NOT POSSIBLE": "FIX NOT POSSIBLE (the assistant's assessment); the twin "
                                  "is unchanged.",
                  "NOT FIXED": f"NOT FIXED after {res['rounds']} rounds; the original twin "
                               "is restored."}[v]]
        if res.get("diagnosis"):
            lines.append("Diagnosis: " + " ".join(res["diagnosis"]))
        if res.get("suggestions"):
            lines.append("Suggestions:\n" + "\n".join(f"  - {s}" for s in res["suggestions"]))
        if res.get("best_attempt"):
            lines.append(f"Best attempt kept: {res['best_attempt']}")
        dpg.set_value("vres_fix_status", "\n".join(lines))
        dpg.configure_item("vres_fix_status", color=C_GREEN if v == "FIXED" else C_AMBER)
        for ln in lines:
            self._fix_log(ln)
        self.app.log(f"[validate] twin fix: {v}")
        if v == "FIXED" and res.get("out_dir"):
            self.show(res["out_dir"])          # the passing re-score
        else:
            dpg.configure_item("vres_btn_fix", enabled=True)

    # ------------------------------------------------------------ building
    def _line_theme(self, col, weight, marker=False):
        with dpg.theme() as th:
            comp = dpg.mvScatterSeries if marker else dpg.mvLineSeries
            with dpg.theme_component(comp):
                dpg.add_theme_color(dpg.mvPlotCol_Line, col, category=dpg.mvThemeCat_Plots)
                if marker:
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, col,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, col,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, weight,
                                        category=dpg.mvThemeCat_Plots)
                else:
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, weight,
                                        category=dpg.mvThemeCat_Plots)
        return th

    def build(self):
        self.th = {"spectre": self._line_theme(C_SPECTRE, 2.4),
                   "twin": self._line_theme(C_TWIN, 1.3),
                   "drive": self._line_theme((170, 178, 196), 1.2),
                   "err": self._line_theme(C_ERR, 1.1),
                   "read": self._line_theme(C_READ, 3.5, marker=True),
                   "limit": self._line_theme((150, 150, 150), 1.0)}
        with dpg.window(label="Validation results", tag="vres_win", show=False,
                        width=1240, height=900, pos=(20, 20), no_collapse=True):
            with dpg.group(horizontal=True):
                dpg.add_text("RUN", color=C_MUTED)
                dpg.add_combo([], tag="vres_run", width=560, callback=self._on_run)
                dpg.add_button(label="Refresh", callback=lambda: self.show())
                dpg.add_button(label="Open report (browser)", callback=self.open_report)
                dpg.add_button(label="Open folder", callback=self.open_folder)
                dpg.add_button(label="Close", callback=lambda: dpg.hide_item("vres_win"))
            dpg.add_text("", tag="vres_head", color=C_TEXT2, wrap=1200)
            with dpg.group(horizontal=True):
                dpg.add_text("", tag="vres_verdict")
                dpg.add_spacer(width=20)
                fb = dpg.add_button(label="Fix twin with AI", tag="vres_btn_fix",
                                    enabled=False, callback=self.start_fix)
                with dpg.tooltip(fb):
                    dpg.add_text("For a run that did not pass: the assistant edits the\n"
                                 "twin file (only that file), NeuroVAT re-scores it\n"
                                 "against this run's saved Spectre results (no new\n"
                                 f"Spectre run, no login), up to {self._rounds()} rounds. "
                                 "Stops at PASS;\nelse the original twin is restored "
                                 "and the verdict\ncomes with a diagnosis and "
                                 "suggestions.")
                dpg.add_button(label="Stop fix", tag="vres_btn_fix_stop", enabled=False,
                               callback=self.stop_fix)
            with dpg.collapsing_header(label="AI twin fix", tag="vres_fix_hdr",
                                       default_open=False):
                dpg.add_text("", tag="vres_fix_status", wrap=1200)
                dpg.add_input_text(tag="vres_fix_log", multiline=True, readonly=True,
                                   width=-1, height=170)
            dpg.add_text("Click a test to see its plots.", color=C_MUTED)
            with dpg.table(tag="vres_table", header_row=True, resizable=True,
                           policy=dpg.mvTable_SizingStretchProp, borders_innerH=True,
                           borders_outerH=True, row_background=True, height=140,
                           scrollY=True):
                for name, w in (("test", 0.14), ("verdict", 0.08),
                                ("waveform NRMSE / max (%)", 0.15),
                                ("read G max / mean (%)", 0.14),
                                ("Spectre step judged at (s)", 0.13),
                                ("Spectre converged", 0.10), ("what the test does", 0.36)):
                    dpg.add_table_column(label=name, init_width_or_weight=w)
            with dpg.tab_bar():
                with dpg.tab(label="Twin vs Spectre"):
                    dpg.add_text("", tag="vres_test_info", wrap=1200, color=C_TEXT2)
                    with dpg.subplots(3, 1, tag="vres_sub", link_all_x=True, width=-1,
                                      height=500, row_ratios=[0.9, 2.0, 1.3]):
                        with dpg.plot(tag="vres_p_drive", height=-1, width=-1):
                            dpg.add_plot_axis(dpg.mvXAxis, tag="vres_x_drive", no_tick_labels=True)
                            dpg.add_plot_axis(dpg.mvYAxis, tag="vres_y_drive", label="drive")
                        with dpg.plot(tag="vres_p_cmp", height=-1, width=-1):
                            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
                            dpg.add_plot_axis(dpg.mvXAxis, tag="vres_x_cmp", no_tick_labels=True)
                            dpg.add_plot_axis(dpg.mvYAxis, tag="vres_y_cmp", label="")
                        with dpg.plot(tag="vres_p_err", height=-1, width=-1):
                            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
                            dpg.add_plot_axis(dpg.mvXAxis, tag="vres_x_err", label="time (s)")
                            dpg.add_plot_axis(dpg.mvYAxis, tag="vres_y_err", label="error (%)")
                    dpg.add_text("Scroll to zoom, drag to pan, double-click to fit; the "
                                 "three plots share the time axis.", color=C_MUTED)
                with dpg.tab(label="Spectre step check"):
                    dpg.add_text("Each test ran in Spectre at three maximum time steps. "
                                 "A step whose result differs from the finest by more "
                                 "than the pass limit means Spectre was not converged "
                                 "there; the twin is judged against a converged (or the "
                                 "finest) run.", wrap=1200, color=C_TEXT2)
                    with dpg.plot(tag="vres_p_conv", height=420, width=-1):
                        dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
                        dpg.add_plot_axis(dpg.mvXAxis, tag="vres_x_conv",
                                          label="Spectre maximum time step (s)",
                                          scale=dpg.mvPlotScale_Log10)
                        dpg.add_plot_axis(dpg.mvYAxis, tag="vres_y_conv",
                                          label="max error vs finest step (%)",
                                          scale=dpg.mvPlotScale_Log10)
                    dpg.add_input_text(tag="vres_conv_rows", multiline=True, readonly=True,
                                       width=-1, height=150)
                with dpg.tab(label="Spectre netlist"):
                    dpg.add_text("The netlist Spectre ran for the selected test (run "
                                 "directly with Spectre on the host, not through "
                                 "Maestro/ADE). The same drive (PWL) goes to the twin.",
                                 wrap=1200, color=C_TEXT2)
                    dpg.add_combo([], tag="vres_deck", width=420, callback=self._on_deck)
                    dpg.add_input_text(tag="vres_deck_text", multiline=True, readonly=True,
                                       width=-1, height=520)
        if "dialog" in self.app.themes:
            dpg.bind_item_theme("vres_win", self.app.themes["dialog"])

    # ------------------------------------------------------------ actions
    def show(self, run_dir=None):
        """Open the window on `run_dir` (default: the newest run)."""
        self.runs = list_runs(self.app.workdir)
        labels = [os.path.basename(r) for r in self.runs]
        dpg.configure_item("vres_run", items=labels)
        if run_dir is None:
            run_dir = self.run_dir if self.run_dir in self.runs else (
                self.runs[0] if self.runs else None)
        dpg.show_item("vres_win")
        dpg.focus_item("vres_win")
        if run_dir is None:
            dpg.set_value("vres_head", "no validation results yet (results/validation)")
            return
        dpg.set_value("vres_run", os.path.basename(run_dir))
        self.load(run_dir)

    def _on_run(self, *_):
        name = dpg.get_value("vres_run")
        hit = [r for r in self.runs if os.path.basename(r) == name]
        if hit:
            self.load(hit[0])

    def open_report(self, *_):
        if self.run_dir:
            p = os.path.join(self.run_dir, "report.html")
            if os.path.exists(p):
                webbrowser.open("file:///" + os.path.abspath(p).replace("\\", "/"))

    def open_folder(self, *_):
        if self.run_dir and os.path.isdir(self.run_dir) and hasattr(os, "startfile"):
            os.startfile(self.run_dir)            # noqa: S606 - a local folder

    # ------------------------------------------------------------ loading
    def load(self, run_dir):
        self.run_dir = run_dir
        with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
            self.summary = s = json.load(f)
        v = s.get("verdict", "?")
        dpg.set_value("vres_head",
                      f"{s.get('module', '?')}  -  twin {s.get('twin', '?')}  -  device "
                      f"type {s.get('profile', '?')}  -  {s.get('simulator', 'Spectre')}  -  "
                      f"{s.get('date', '')}")
        dpg.set_value("vres_verdict", f"Overall: {v}")
        dpg.configure_item("vres_verdict", color=VERDICT_COL.get(v, C_AMBER))
        dpg.configure_item("vres_btn_fix", enabled=(v != "PASS" and not self.fixing))
        conv = {c["stimulus"]: c for c in s.get("layer2") or []}
        for row in dpg.get_item_children("vres_table", 1) or []:
            dpg.delete_item(row)
        tests = s.get("layer1") or []
        for t in tests:
            c = conv.get(t["stimulus"], {})
            with dpg.table_row(parent="vres_table"):
                dpg.add_selectable(label=t["stimulus"], span_columns=True,
                                   user_data=t["stimulus"],
                                   callback=lambda sndr, a, u: self.select(u))
                dpg.add_text(t.get("verdict", "?"),
                             color=VERDICT_COL.get(t.get("verdict"), C_AMBER))
                dpg.add_text(f"{_pct(t.get('nrmse_pct'))} / {_pct(t.get('max_err_pct'))}")
                dpg.add_text(f"{_pct(t.get('read_g_max_pct'))} / "
                             f"{_pct(t.get('read_g_mean_pct'))}" if "read_g_max_pct" in t
                             else "-")
                dpg.add_text(f"{t.get('reference_step_s', float('nan')):.3g}"
                             if t.get("reference_step_s") else "-")
                ok = c.get("reference_converged")
                dpg.add_text("yes" if ok else ("no" if ok is not None else "-"),
                             color=C_GREEN if ok else C_AMBER)
                dpg.add_text(t.get("description", ""), wrap=420)
        self._plot_conv(conv)
        if tests:
            self.select(tests[0]["stimulus"])

    def select(self, name):
        self.test = name
        s = self.summary
        t = next((x for x in s.get("layer1") or [] if x["stimulus"] == name), {})
        note = f"   NOTE: {t['note']}" if t.get("note") else ""
        dpg.set_value("vres_test_info",
                      f"{name}: {t.get('description', '')}\nverdict {t.get('verdict', '?')}"
                      f" - rule: {t.get('rule', 'NRMSE <= 1 %, max <= 3 % of the range')}"
                      f"{note}")
        self._plot_test(name)
        self._conv_rows(name)
        self._decks(name)

    # ------------------------------------------------------------ plots
    @staticmethod
    def _clear(*axes):
        for ax in axes:
            dpg.delete_item(ax, children_only=True)

    def _series(self, parent, x, y, label, theme, scatter=False):
        x, y = (np.asarray(x, float), np.asarray(y, float)) if scatter else envelope(x, y)
        ok = np.isfinite(x) & np.isfinite(y)
        add = dpg.add_scatter_series if scatter else dpg.add_line_series
        sid = add(x[ok].tolist(), y[ok].tolist(), label=label, parent=parent)
        dpg.bind_item_theme(sid, self.th[theme])
        return sid

    def _plot_test(self, name):
        self._clear("vres_y_drive", "vres_y_cmp", "vres_y_err")
        path = os.path.join(self.run_dir, f"l1_{name}.npz")
        if not os.path.exists(path):
            dpg.set_value("vres_test_info", dpg.get_value("vres_test_info") +
                          "\n(this run has no plot data in the app; it predates the "
                          "results viewer - use Open report)")
            return
        from .spectre.validate import load_l1_data
        d, meta = load_l1_data(path)
        unit = "A" if meta.get("drive_kind") == "current" else "V"
        dpg.configure_item("vres_y_drive", label=f"drive ({unit})")
        dpg.configure_item("vres_y_cmp", label=meta.get("label", ""))
        self._series("vres_y_drive", d["drive_t"], d["drive_v"], "drive", "drive")
        self._series("vres_y_cmp", d["t_ref"], d["y_ref"], "Spectre", "spectre")
        self._series("vres_y_cmp", d["t_tw"], d["y_tw"], "NeuroVAT twin", "twin")
        if "read_t" in d:
            self._series("vres_y_cmp", d["read_t"], d["g_ref"] if meta.get("label", "")
                         .startswith("G") else np.interp(d["read_t"], d["t_ref"], d["y_ref"]),
                         "reads (Spectre)", "read", scatter=True)
        rng = float(np.nanmax(d["y_ref"]) - np.nanmin(d["y_ref"])) or 1.0
        err = 100 * (d["y_tw"] - np.interp(d["t_tw"], d["t_ref"], d["y_ref"])) / rng
        if meta.get("profile") != "fefet":       # FeFET: only the reads are compared
            self._series("vres_y_err", d["t_tw"], err, "waveform error (% of range)", "err")
        if "read_err_pct" in d:
            lab = ("read G error (% of G range)" if meta.get("profile") == "fefet"
                   else "read G error (%)")
            self._series("vres_y_err", d["read_t"], d["read_err_pct"], lab, "read",
                         scatter=True)
        for ax in ("vres_x_drive", "vres_x_cmp", "vres_x_err", "vres_y_drive",
                   "vres_y_cmp", "vres_y_err"):
            dpg.fit_axis_data(ax)

    def _plot_conv(self, conv):
        self._clear("vres_y_conv")
        for name, c in conv.items():
            r = [x for x in c.get("rows", []) if x.get("what") == "spectre vs finest spectre"
                 and np.isfinite(x.get("max_err_pct", np.nan))]
            if r:
                dpg.add_line_series([x["step_s"] for x in r],
                                    [max(x["max_err_pct"], 1e-9) for x in r],
                                    label=name, parent="vres_y_conv")
        lim = dpg.add_inf_line_series([3.0], horizontal=True, label="pass limit 3 %",
                                      parent="vres_y_conv")
        dpg.bind_item_theme(lim, self.th["limit"])
        dpg.fit_axis_data("vres_x_conv")
        dpg.fit_axis_data("vres_y_conv")

    def _conv_rows(self, name):
        c = next((x for x in self.summary.get("layer2") or [] if x["stimulus"] == name), {})
        lines = [f"{name}: Spectre steps {', '.join(f'{v:.3g}' for v in c.get('steps', []))} s"]
        for r in c.get("rows", []):
            lines.append(f"  {r['what']:30s} step {r['step_s']:.3g} s   max "
                         f"{_pct(r.get('max_err_pct'))} %   {r.get('verdict', '')}")
        dpg.set_value("vres_conv_rows", "\n".join(lines))

    def _decks(self, name):
        sdir = os.path.join(self.run_dir, "spectre")
        files = sorted(f for f in (os.listdir(sdir) if os.path.isdir(sdir) else [])
                       if f.startswith(name + "__") and f.endswith(".scs"))
        c = next((x for x in self.summary.get("layer2") or [] if x["stimulus"] == name), {})
        t = next((x for x in self.summary.get("layer1") or [] if x["stimulus"] == name), {})
        steps, ref = c.get("steps") or [], t.get("reference_step_s")
        self._deck_files, labels, pick = {}, [], None
        for f in files:
            lab = f
            try:
                i = int(f[len(name) + 3:-4])
                lab = f"{f}   (max step {steps[i]:.3g} s)"
                if ref is not None and abs(steps[i] - ref) <= 1e-9 * abs(ref):
                    lab += "  <- reference the twin was judged against"
                    pick = lab
            except (ValueError, IndexError):
                pass
            self._deck_files[lab] = f
            labels.append(lab)
        dpg.configure_item("vres_deck", items=labels)
        if labels:
            dpg.set_value("vres_deck", pick or labels[0])
            self._on_deck()
        else:
            dpg.set_value("vres_deck", "")
            dpg.set_value("vres_deck_text", "(the netlists of this run were not kept; runs "
                                            "made from now on keep them)")

    def _on_deck(self, *_):
        f = getattr(self, "_deck_files", {}).get(dpg.get_value("vres_deck"))
        if not f:
            return
        p = os.path.join(self.run_dir, "spectre", f)
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as e:
            text = f"cannot read {f}: {e}"
        dpg.set_value("vres_deck_text", text[:200000])
