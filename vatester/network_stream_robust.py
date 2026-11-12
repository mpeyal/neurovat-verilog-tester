"""Robustness tab of the unsupervised stream mode: after training, test the
network on a fixed stream with more jitter, missing spikes or extra spikes,
one factor at a time (core: :mod:`vatester.stream_robust`).

Render thread: the tab, the guards (no trained network, settings changed,
busy) and the drawing. The worker runs :func:`stream_robust.sweep` once per
level (same stream key and nominal, so the merged result is the one a single
sweep gives) and only posts queue messages: ``nt_robust_progress`` (text, and
after each level the partial result) and always a terminal
``nt_robust_done``. It shares the Trainer's busy flag ``trainer_running``
and its stop flag ``_trainer_stop``.

"Show in Raster" replays ONE keyed STDP-off run at a swept level through the
Raster tab (the same stream and numbers as that level of the sweep).

The module-level helpers are GUI-free (tested in tests/test_robust_tab.py).
"""
import math
import threading

import dearpygui.dearpygui as dpg
import numpy as np

from . import stream_robust as sr
from . import stream_view as sv
from .ui_constants import C_AMBER, C_GREEN, C_MUTED, C_RED, C_TEXT2

# (factor, level-list tag, plot tag stem, x-axis label)
FACTOR_TAGS = (("jitter_ms", "nt_st_rb_jitter", "jit", "jitter (ms)"),
               ("spike_deletion", "nt_st_rb_delete", "del",
                "missing spikes (%)"),
               ("noise_hz", "nt_st_rb_noise", "noi", "extra spikes (Hz)"))
SETTINGS_CHANGED = ("settings changed since this network was built: press "
                    "New network, or restore them")
NO_NETWORK = ("no trained network: press Train first (the sweep tests the "
              "trained network; Test before Train is the untrained baseline)")
DEVICE_NOTE = "untrained baseline not available for device synapses"
HIT_COL = (80, 160, 255)
FALSE_COL = (255, 140, 60)
NOM_COL = (205, 210, 225, 150)
UNTRAINED_ALPHA = 95
INTRO = ("After training, the network is tested (STDP off, threshold held) "
         "on one fixed test stream, one factor changed at a time; counted "
         "over the whole run, so the numbers differ from Test's last-third "
         "verdict.")

# (tag, one-line meaning) - added to the controls the agent / bridge may set
ROBUST_CONTROLS = [
    ("nt_st_rb_jitter", "robustness sweep: jitter levels ms, comma list "
                        "('' = 1, 3, 5 and the trained value)"),
    ("nt_st_rb_delete", "robustness sweep: missing-spike levels in %, comma "
                        "list ('' = 0, 10, 30, 50 and the trained value)"),
    ("nt_st_rb_noise", "robustness sweep: extra-spike levels Hz, ABSOLUTE "
                       "noise rate, comma list ('' = 5, 26 and the trained "
                       "value)"),
    ("nt_st_rb_seconds", "robustness sweep: seconds of stream per level "
                         "(1-600, default 30)"),
    ("nt_st_rb_untrained", "robustness sweep: also run the untrained copy "
                           "(ideal synapses only)"),
    ("nt_st_rb_pick", "robustness sweep: the swept level 'Show in Raster' "
                      "replays (an item of the list, or its index)"),
]


# ---- GUI-free helpers -------------------------------------------------

def stream_changes(cfg):
    """The per-run stream settings of a Train / Test config, as the stream
    worker passes them to ``Session.run`` (``bg_hz`` for images only)."""
    out = {"pattern_frac": cfg["share"], "jitter_ms": cfg["jitter"],
           "noise_hz": cfg["noise"], "spike_deletion": cfg["delete"],
           "spacing": cfg["spacing"]}
    if cfg.get("image") is not None:
        out["bg_hz"] = cfg["bg"]
    return out


def _scale(factor):
    return sr.LABELS[sr.factor_name(factor)][2]


def parse_levels(text, factor):
    """Levels typed in display units (deletion in %) -> the core's units,
    in the order given, no repeats. '' -> None (default levels)."""
    factor = sr.factor_name(factor)
    label, unit, scale = sr.LABELS[factor]
    parts = [p for p in str(text or "").replace(";", ",").replace(
        " ", ",").split(",") if p.strip()]
    if not parts:
        return None
    out = []
    for p in parts:
        try:
            v = float(p.strip().rstrip("%"))
        except ValueError:
            raise ValueError(f"{label} levels: {p.strip()!r} is not a "
                             "number") from None
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"{label} levels must be >= 0 {unit}")
        if factor == "spike_deletion" and v >= 100.0:
            raise ValueError("missing-spike levels are % below 100")
        v = round(v / scale, 9)
        if v not in out:
            out.append(v)
    return out


def format_levels(levels, factor):
    scale = _scale(factor)
    return ", ".join(f"{v * scale:g}" for v in levels)


def merge(acc, part):
    """Fold one :func:`stream_robust.sweep` result into ``acc`` (None to
    start): factor blocks of the same factor are joined, ``stopped`` is
    or-ed and the wall times add up. Returns ``acc``."""
    if acc is None:
        acc = {k: v for k, v in part.items() if k != "factors"}
        acc["factors"], acc["wall_s"], acc["stopped"] = [], 0.0, False
    for b in part["factors"]:
        dst = next((x for x in acc["factors"] if x["factor"] == b["factor"]),
                   None)
        if dst is None:
            dst = {k: b[k] for k in ("factor", "label", "unit", "scale")}
            dst["rows"] = []
            acc["factors"].append(dst)
        dst["rows"].extend(b["rows"])
    acc["stopped"] = bool(acc["stopped"] or part.get("stopped"))
    acc["wall_s"] = float(acc["wall_s"]) + float(part.get("wall_s") or 0.0)
    return acc


def _num(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    return v


def _run(r):
    if r is None:
        return None
    return {k: _num(r[k]) for k in (
        "hits", "pattern_windows", "false_windows", "other_windows",
        "hit_pct", "false_pct", "balanced_accuracy", "post_spikes")}


def compact(result):
    """JSON-safe copy without the per-neuron engine counts (NaN -> None);
    each row also carries ``shown``, the level in display units."""
    if result is None:
        return None
    out = {k: _num(v) for k, v in result.items()
           if k not in ("factors", "nominal")}
    out["nominal"] = {k: _num(v) for k, v in result["nominal"].items()}
    out["factors"] = [{
        "factor": b["factor"], "label": b["label"], "unit": b["unit"],
        "scale": b["scale"],
        "rows": [{"level": float(r["level"]),
                  "shown": float(r["level"]) * b["scale"],
                  "nominal": bool(r["nominal"]),
                  "trained": _run(r["trained"]),
                  "untrained": _run(r["untrained"])} for r in b["rows"]]}
        for b in result["factors"]]
    return out


def series(block):
    """Plot data of one factor block: x in display units per series
    (points whose value is missing are dropped)."""
    out = {}
    for who in ("trained", "untrained"):
        for what, key in (("hit", "hit_pct"), ("false", "false_pct")):
            xs, ys = [], []
            for r in block["rows"]:
                run = r.get(who)
                v = None if run is None else run.get(key)
                if v is None or not math.isfinite(float(v)):
                    continue
                xs.append(float(r["level"]) * block["scale"])
                ys.append(float(v))
            out[f"{who}_{what}"] = (xs, ys)
    return out


def entries(result):
    """[(text, factor, level)] of every swept level (the Raster combo)."""
    out = []
    for b in (result or {}).get("factors", []):
        for r in b["rows"]:
            mark = " (nominal)" if r["nominal"] else ""
            out.append((f"{b['label']} {r['level'] * b['scale']:g} "
                        f"{b['unit']}{mark}", b["factor"], float(r["level"])))
    return out


def table_text(result, width=96):
    """:func:`stream_robust.format_table`, its prose lines wrapped to
    ``width`` (the table rows, which hold '|', are kept as they are)."""
    import textwrap
    out = []
    for line in sr.format_table(result).splitlines():
        if "|" in line or len(line) <= width:
            out.append(line)
        else:
            out += textwrap.wrap(line, width, subsequent_indent="  ",
                                 break_long_words=False,
                                 break_on_hyphens=False)
    return "\n".join(out)


def summary_lines(result):
    """Short per-level lines (agent context, chat): hits and false windows
    as counts, trained | untrained."""
    def cell(r):
        if r is None:
            return "-"
        return (f"hits {r['hits']}/{r['pattern_windows']}, false "
                f"{r['false_windows']}/{r['other_windows']}")
    lines = []
    for b in result.get("factors", []):
        for r in b["rows"]:
            mark = "*" if r["nominal"] else ""
            lines.append(f"  {b['label']} {r['level'] * b['scale']:g}"
                         f"{mark} {b['unit']}: trained {cell(r['trained'])}"
                         f" | untrained {cell(r['untrained'])}")
    return lines


# ---- the tab (render thread) -------------------------------------------

class NetworkStreamRobustMixin:
    _nt_stream_trained_with = None  # stream settings of the last Train
    _nt_rb_result = None            # the last sweep (full, render thread)
    _nt_rb_meta = None              # ... which network it was run on
    _nt_rb_entries = ()             # [(text, factor, level)] for the combo
    _nt_rb_error = None
    _nt_rb_kind = None              # "sweep" | "raster" while one runs
    _nt_rb_agent = False            # the agent started the sweep in flight
    _nt_rb_th = None

    @staticmethod
    def _nt_rb_changes(cfg):
        return stream_changes(cfg)

    def _nt_rb_theme(self, col, weight, marker, msize):
        """Cached theme of a line series with markers."""
        if self._nt_rb_th is None:
            self._nt_rb_th = {}
        key = (tuple(col), weight, marker, msize)
        th = self._nt_rb_th.get(key)
        if th is not None:
            return th
        cat = dpg.mvThemeCat_Plots
        with dpg.theme() as th:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, col, category=cat)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, col, category=cat)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, col,
                                    category=cat)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, weight,
                                    category=cat)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, marker,
                                    category=cat)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, msize,
                                    category=cat)
        self._nt_rb_th[key] = th
        return th

    def _nt_st_robust_tab(self):
        """Contents of the Robustness tab (inside ``nt_st_tab_robust``)."""
        with dpg.child_window(width=-1, height=-1, border=False,
                              tag="nt_st_rb_scroll"):
            self._small(INTRO, color=C_TEXT2, wrap=0)
            with dpg.group(horizontal=True):
                for factor, tag, _stem, _xl in FACTOR_TAGS:
                    label, unit, _s = sr.LABELS[factor]
                    self._small(f"{label} {unit}", color=C_TEXT2)
                    dpg.add_input_text(tag=tag, width=150,
                                       hint=self._nt_rb_hint(factor, None))
                    with dpg.tooltip(tag):
                        dpg.add_text(
                            f"{label} levels in {unit}, comma-separated. "
                            "Empty = the levels of Masquelier et al. 2008 "
                            "plus the value the network was trained with."
                            + (" Absolute extra-spike rate per afferent, not "
                               "added to the trained value." if
                               factor == "noise_hz" else ""), wrap=300)
            with dpg.group(horizontal=True):
                self._small("seconds / level", color=C_TEXT2)
                dpg.add_input_double(tag="nt_st_rb_seconds", default_value=30.0,
                                     width=70, format="%.3g", step=0)
                with dpg.tooltip("nt_st_rb_seconds"):
                    dpg.add_text("Stream seconds of every level (1-600). Each "
                                 "level runs once trained and once untrained.",
                                 wrap=300)
                dpg.add_checkbox(label="compare with untrained", default_value=True,
                                 tag="nt_st_rb_untrained")
                with dpg.tooltip("nt_st_rb_untrained"):
                    dpg.add_text("Also run every level on an untrained copy "
                                 "(same patterns, start weights): what "
                                 "training bought. Ideal synapses only.",
                                 wrap=300)
                dpg.add_button(label=" Run sweep ", tag="nt_st_rb_run",
                               callback=lambda *_: self._nt_rb_start())
                dpg.add_button(label=" Stop ", tag="nt_st_rb_stop",
                               callback=lambda *_: self._nt_rb_stop())
            dpg.add_text("Robustness: train first, then press Run sweep.",
                         tag="nt_st_rb_status", color=C_TEXT2, wrap=0)
            self._small("", color=C_MUTED, tag="nt_st_rb_nominal", wrap=0)
            with dpg.group(horizontal=True):
                for (col, alpha), what in (((HIT_COL, 255), "trained: hits"),
                                           ((FALSE_COL, 255), "false windows"),
                                           ((HIT_COL, 150), "  untrained "
                                            "(faded, wide): hits"),
                                           ((FALSE_COL, 150), "false windows"),
                                           ((NOM_COL[:3], 255),
                                            "  | grey line = nominal")):
                    self._small(what, color=(*col, alpha))
            with dpg.table(header_row=False, borders_innerV=False,
                           borders_outerV=False, borders_innerH=False,
                           borders_outerH=False, pad_outerX=False,
                           policy=dpg.mvTable_SizingStretchSame):
                for _ in FACTOR_TAGS:
                    dpg.add_table_column()
                with dpg.table_row():
                    for factor, _tag, stem, xlabel in FACTOR_TAGS:
                        self._nt_rb_plot(stem, sr.LABELS[factor][0], xlabel)
            with dpg.group(horizontal=True):
                self._small("show in Raster", color=C_TEXT2)
                dpg.add_combo([], tag="nt_st_rb_pick", width=240)
                dpg.add_button(label=" Show in Raster ", tag="nt_st_rb_show",
                               callback=lambda *_: self._nt_rb_show_raster())
                with dpg.tooltip("nt_st_rb_show"):
                    dpg.add_text("Runs ONE keyed STDP-off run of the trained "
                                 "network at that level and shows it in the "
                                 "Raster tab: the same stream and spikes as "
                                 "that level of the sweep. It replaces the "
                                 "Raster's replay of the last Train / Test.",
                                 wrap=300)
            dpg.add_input_text(tag="nt_st_rb_table", multiline=True,
                               readonly=True, width=-1, height=230,
                               default_value="no sweep yet")
            fonts = getattr(self, "fonts", None) or {}
            if "mono" in fonts:
                dpg.bind_item_font("nt_st_rb_table", fonts["mono"])

    def _nt_rb_plot(self, stem, title, xlabel):
        with dpg.plot(label=title, width=-1, height=250, no_menus=True,
                      no_mouse_pos=True, tag=f"nt_st_rb_plot_{stem}"):
            dpg.add_plot_axis(dpg.mvXAxis, label=xlabel,
                              tag=f"nt_st_rb_x_{stem}")
            with dpg.plot_axis(dpg.mvYAxis, label="% of windows",
                               tag=f"nt_st_rb_y_{stem}"):
                s = dpg.add_inf_line_series([], label="nominal",
                                            tag=f"nt_st_rb_{stem}_nom")
                dpg.bind_item_theme(s, self._nt_st_theme("line", NOM_COL, 1.0))
                # untrained first (wide, faded), trained on top: they can
                # coincide exactly and both must stay visible
                for who, weight, marker, msize, alpha in (
                        ("untrained", 6.0, dpg.mvPlotMarker_Square, 5.0,
                         UNTRAINED_ALPHA),
                        ("trained", 1.8, dpg.mvPlotMarker_Circle, 3.0, 255)):
                    for what, col in (("hit", HIT_COL), ("false", FALSE_COL)):
                        s = dpg.add_line_series(
                            [], [], label=f"{who} {'hits' if what == 'hit' else 'false windows'}",
                            tag=f"nt_st_rb_{stem}_{who}_{what}")
                        dpg.bind_item_theme(s, self._nt_rb_theme(
                            (*col, alpha), weight, marker, msize))
            dpg.set_axis_limits(f"nt_st_rb_y_{stem}", -4.0, 104.0)

    # ---- status ---------------------------------------------------------

    @staticmethod
    def _nt_rb_hint(factor, nominal_value):
        levels = list(sr.FIGURE_LEVELS[factor])
        if nominal_value is not None:
            levels = sr.default_levels(factor, nominal_value)
            return "empty = " + format_levels(levels, factor)
        return "empty = " + format_levels(levels, factor) + " + trained"

    def _nt_rb_say(self, text, color=C_TEXT2):
        if dpg.does_item_exist("nt_st_rb_status"):
            dpg.set_value("nt_st_rb_status", text)
            dpg.configure_item("nt_st_rb_status", color=color)

    def _nt_rb_refuse(self, text):
        self._nt_rb_error = text
        self._nt_rb_say(text, C_RED)
        return False

    def _nt_rb_check(self):
        """The guards shared by the sweep and Show in Raster. Returns the
        session and the nominal ({settings}, how) or None (refused)."""
        if self.trainer_running:
            self._nt_rb_refuse("busy: a Train / Test / sweep is running - "
                               "stop it first")
            return None
        s = self._nt_stream_session
        if s is None or self._nt_stream_trained_s <= 0.0:
            self._nt_rb_refuse(NO_NETWORK)
            return None
        try:
            key, cfg = self._nt_stream_settings()
        except (ValueError, TypeError) as exc:
            self._nt_rb_refuse(str(exc))
            return None
        if key != self._nt_stream_key:
            self._nt_rb_refuse(SETTINGS_CHANGED)
            return None
        nominal, how = self._nt_stream_trained_with, "the settings of the last Train"
        if nominal is None:
            nominal = stream_changes(cfg)
            how = ("the LIVE controls: the settings this network was trained "
                   "with were not recorded")
        return s, dict(nominal), how, key

    # ---- sweep ----------------------------------------------------------

    def _nt_rb_start(self, *_):
        """Start the sweep (button, bridge, agent). False = refused, the
        reason in ``_nt_rb_error`` and the status line."""
        self._nt_rb_error = None
        got = self._nt_rb_check()
        if got is None:
            return False
        s, nominal, how, key = got
        try:
            nom = sr.nominal_settings(s, nominal)
            plan = []
            for factor, tag, _stem, _xl in FACTOR_TAGS:
                lv = parse_levels(self._nt_get(tag, ""), factor)
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, hint=self._nt_rb_hint(
                        factor, nom[factor]))
                plan.append((factor, lv if lv is not None else
                             sr.default_levels(factor, nom[factor])))
            seconds = float(self._nt_get("nt_st_rb_seconds", 30.0))
        except (ValueError, TypeError) as exc:
            return self._nt_rb_refuse(str(exc))
        seconds = min(max(seconds, 1.0), 600.0)
        device = s.device is not None
        want_base = bool(self._nt_get("nt_st_rb_untrained", True))
        untrained = None if want_base else False
        n_levels = sum(len(lv) for _f, lv in plan)
        n_runs = n_levels * (2 if want_base and not device else 1)
        self._nt_rb_meta = {"session": s, "trained_s": self._nt_stream_trained_s,
                            "key": key, "how": how, "device": device,
                            "nominal": dict(nom), "seconds": seconds}
        self.trainer_running = True
        self._trainer_stop = False
        self._nt_rb_kind = "sweep"
        self._nt_busy(True)
        self._nt_rb_clear()
        nom_txt = ", ".join(f"{k} {v:g}" if isinstance(v, (int, float))
                            else f"{k} {v}" for k, v in nom.items())
        if dpg.does_item_exist("nt_st_rb_nominal"):
            dpg.set_value("nt_st_rb_nominal", f"nominal = {how}: {nom_txt}"
                          + (f". {DEVICE_NOTE}." if device and want_base else ""))
        self._nt_status("robustness sweep", C_AMBER)
        self._nt_rb_say(f"sweep: {n_levels} levels, {n_runs} runs of "
                        f"{seconds:g} s - starting...", C_AMBER)
        self.log(f"[robust] sweep: {n_levels} levels x {seconds:g} s, "
                 f"untrained {'on' if want_base else 'off'}"
                 f"{' (device: not available)' if device and want_base else ''}"
                 f"; nominal = {how}: {nom_txt}")
        threading.Thread(target=self._nt_rb_worker,
                         args=(s, plan, seconds, nominal, untrained, device,
                               want_base), daemon=True).start()
        return True

    def _nt_rb_stop(self, *_):
        if self.trainer_running:
            self._trainer_stop = True
            self._nt_rb_say("stopping...", C_AMBER)
            self._nt_status("stopping...", C_AMBER)

    def _nt_rb_clear(self):
        for _f, _tag, stem, _xl in FACTOR_TAGS:
            for who in ("trained", "untrained"):
                for what in ("hit", "false"):
                    t = f"nt_st_rb_{stem}_{who}_{what}"
                    if dpg.does_item_exist(t):
                        dpg.set_value(t, [[], []])
            if dpg.does_item_exist(f"nt_st_rb_{stem}_nom"):
                dpg.set_value(f"nt_st_rb_{stem}_nom", [[]])

    # ---- worker (no DPG access) ------------------------------------------

    def _nt_rb_worker(self, session, plan, seconds, nominal, untrained,
                      device, want_base):
        out = {"kind": "sweep", "error": None, "result": None}
        try:
            stop = lambda: bool(self._trainer_stop)            # noqa: E731
            total = sum(len(lv) for _f, lv in plan)
            merged, k = None, 0
            for factor, levels in plan:
                for v in levels:
                    if stop():
                        if merged is not None:
                            merged["stopped"] = True
                        break

                    def progress(_i, _n, text, k=k):
                        self.q.put(("nt_robust_progress", {
                            "text": f"level {k + 1} of {total}: {text}"}))
                    part = sr.sweep(session, {factor: [v]}, seconds=seconds,
                                    nominal=nominal, on_progress=progress,
                                    stop=stop, untrained=untrained)
                    merged = merge(merged, part)
                    if device and want_base:
                        merged["untrained_note"] = DEVICE_NOTE
                    k += 1
                    self.q.put(("nt_robust_progress", {
                        "text": f"{k} of {total} levels done",
                        "result": compact(merged),
                        "table": table_text(merged)}))
                    if part["stopped"]:
                        break
                if merged is not None and merged["stopped"]:
                    break
            if merged is None:          # stopped before the first level
                merged = {"factors": [], "nominal": sr.nominal_settings(
                    session, nominal), "seconds": float(seconds),
                    "key": sr.SWEEP_KEY, "images": bool(session.given),
                    "rule": sv.hit_rule(bool(session.given)),
                    "untrained_note": "-", "stopped": True, "wall_s": 0.0}
            out["result"] = merged
        except Exception as ex:                          # noqa: BLE001
            out["error"] = str(ex) or repr(ex)
            self.q.put(("log", f"[robust] error: {ex!r}"))
        finally:
            self.q.put(("nt_robust_done", out))

    def _nt_rb_raster_worker(self, session, factor, level, nominal, seconds,
                             key, arch, view_ms):
        out = {"kind": "raster", "error": None, "factor": factor,
               "level": level}
        saved = session.kw
        try:
            changes = dict(sr.nominal_settings(session, nominal))
            changes[factor] = level
            P = session.P
            images = bool(session.given)
            cap = {}

            def on_section(_t, info):
                if "stream" not in cap:
                    cap["stream"] = info["stream"]
                return not self._trainer_stop
            if "energy" in session.kw:      # the sweep's reads are not metered
                session.kw = {k: v for k, v in session.kw.items()
                              if k != "energy"}
            record_u = seconds * P.n_out <= sv.U_BUDGET
            res = session.run(seconds, learn=False, on_section=on_section,
                              record_u=record_u, stream_key=key, **changes)
            if res.get("stopped") or "stream" not in cap:
                out["stopped"] = True
                return
            view = sv.RunView(cap["stream"], P.pattern_ms, P.n_out,
                              P.threshold, 0.0, arch["shown"], P.n_afferents,
                              view_ms, images)
            view.duration_ms = min(view.duration_ms,
                                   res["simulated_s"] * 1000.0)
            out.update(view=view, arch=arch,
                       post=(np.array(res["post_t_ms"], float),
                             np.array(res["post_neuron"], int)),
                       u=res.get("u_ms"), post_spikes=int(res["post_spikes"]),
                       share=float(changes["pattern_frac"]))
        except Exception as ex:                          # noqa: BLE001
            out["error"] = str(ex) or repr(ex)
            self.q.put(("log", f"[robust] raster error: {ex!r}"))
        finally:
            session.kw = saved
            self.q.put(("nt_robust_done", out))

    # ---- queue handlers (render thread) -----------------------------------

    def _on_nt_robust_progress(self, d):
        if d.get("result") is not None:
            self._nt_rb_draw(d["result"])
        if d.get("table"):
            dpg.set_value("nt_st_rb_table", d["table"])
        self._nt_rb_say("sweep running - " + d.get("text", ""), C_AMBER)

    def _on_nt_robust_done(self, out):
        self.trainer_running = False
        self._nt_busy(False)
        self._nt_rb_kind = None
        agent, self._nt_rb_agent = self._nt_rb_agent, False
        try:
            if out.get("kind") == "raster":
                self._nt_rb_raster_done(out)
                return
            if out["error"]:
                self._nt_rb_error = out["error"]
                self._nt_status("robustness sweep failed", C_RED)
                self._nt_rb_say("sweep failed: " + out["error"], C_RED)
                if agent:
                    self.append_chat("sys", "robustness sweep failed: "
                                     + out["error"])
                return
            r = self._nt_rb_result = out["result"]
            self._nt_rb_draw(compact(r))
            table = table_text(r)
            dpg.set_value("nt_st_rb_table", table)
            self._nt_rb_entries = entries(r)
            items = [e[0] for e in self._nt_rb_entries]
            dpg.configure_item("nt_st_rb_pick", items=items)
            if items and dpg.get_value("nt_st_rb_pick") not in items:
                dpg.set_value("nt_st_rb_pick", items[0])
            n = len(self._nt_rb_entries)
            if r.get("stopped"):
                self._nt_status("robustness sweep stopped", C_AMBER)
                self._nt_rb_say(f"stopped: {n} level(s) finished "
                                f"({r.get('wall_s', 0.0):.0f} s wall)", C_AMBER)
            else:
                self._nt_status("robustness sweep done", C_GREEN)
                self._nt_rb_say(f"done: {n} levels in {r.get('wall_s', 0.0):.0f} "
                                "s wall. The same sweep again gives the same "
                                "numbers (one fixed stream key).", C_GREEN)
            self.log("[robust] " + table.replace("\n", "\n  "))
            if agent:
                self.append_chat("sys", "robustness sweep done:\n"
                                 + "\n".join(summary_lines(r)))
        except Exception as exc:                        # noqa: BLE001
            self.log(f"[robust] view error: {exc!r}")
            self._nt_rb_say(f"view error: {exc}", C_RED)

    def _nt_rb_draw(self, res):
        """Draw a (compact) sweep result into the three plots."""
        by = {b["factor"]: b for b in res.get("factors", [])}
        nominal = res.get("nominal") or {}
        for factor, _tag, stem, _xl in FACTOR_TAGS:
            b = by.get(factor)
            scale = sr.LABELS[factor][2]
            data = series(b) if b else {}
            xs_all = []
            for who in ("trained", "untrained"):
                for what in ("hit", "false"):
                    xs, ys = data.get(f"{who}_{what}", ([], []))
                    dpg.set_value(f"nt_st_rb_{stem}_{who}_{what}", [xs, ys])
                    xs_all += xs
            nv = nominal.get(factor)
            nx = [] if nv is None else [float(nv) * scale]
            dpg.set_value(f"nt_st_rb_{stem}_nom", [nx])
            xs_all += nx
            if xs_all:
                lo, hi = min(xs_all), max(xs_all)
                pad = max(0.06 * (hi - lo), 0.5)
                dpg.set_axis_limits(f"nt_st_rb_x_{stem}", lo - pad, hi + pad)
            dpg.set_axis_limits(f"nt_st_rb_y_{stem}", -4.0, 104.0)

    # ---- show one level in the Raster tab ---------------------------------

    def _nt_rb_pick_entry(self):
        cur = self._nt_get("nt_st_rb_pick", "")
        return next((e for e in self._nt_rb_entries if e[0] == cur), None)

    def _nt_rb_show_raster(self, *_):
        """Replay one swept level in the Raster tab. False = refused."""
        self._nt_rb_error = None
        r, meta = self._nt_rb_result, self._nt_rb_meta
        if r is None or meta is None:
            return self._nt_rb_refuse("run a sweep first")
        got = self._nt_rb_check()
        if got is None:
            return False
        s, _nominal, _how, key = got
        if (s is not meta["session"] or key != meta["key"]
                or self._nt_stream_trained_s != meta["trained_s"]):
            return self._nt_rb_refuse("the network changed since this sweep "
                                      "(trained further or rebuilt): run the "
                                      "sweep again")
        e = self._nt_rb_pick_entry()
        if e is None:
            return self._nt_rb_refuse("pick a swept level first")
        arch = self._nt_st_arch
        if not arch or arch.get("error") or not self._nt_st_same_net(arch):
            return self._nt_rb_refuse("the network view is not available")
        images = bool(s.given)
        nominal = r["nominal"]
        view_ms = sv.view_window_ms(self._nt_get("nt_st_view_win", 0.0),
                                    images, s.P.pattern_ms,
                                    float(nominal["pattern_frac"]))
        self.trainer_running = True
        self._trainer_stop = False
        self._nt_rb_kind = "raster"
        self._nt_busy(True)
        self._nt_st_set_play(False)
        self._nt_rb_say(f"Raster: running {e[0]} ({r['seconds']:g} s, STDP "
                        "off, the sweep's stream)...", C_AMBER)
        self._nt_status("robustness: raster run", C_AMBER)
        threading.Thread(target=self._nt_rb_raster_worker,
                         args=(s, e[1], e[2], dict(nominal), r["seconds"],
                               r["key"], arch, view_ms), daemon=True).start()
        return True

    def _nt_rb_raster_done(self, out):
        if out["error"]:
            self._nt_rb_error = out["error"]
            self._nt_status("robustness raster failed", C_RED)
            self._nt_rb_say("raster run failed: " + out["error"], C_RED)
            return
        if out.get("stopped"):
            self._nt_status("robustness raster stopped", C_AMBER)
            self._nt_rb_say("raster run stopped", C_AMBER)
            return
        arch = out["arch"]
        self._nt_rb_raster_last = out["post_spikes"]
        self._nt_st_setup_raster(arch)
        self._nt_st_pending = None
        self._nt_st_keep_run(out["view"], out["post"][0], out["post"][1],
                             out["u"], arch, out["share"])
        self._nt_set_control("nt_st_tabs", "nt_st_tab_raster")
        e = next((x for x in self._nt_rb_entries
                  if x[1] == out["factor"] and x[2] == out["level"]), None)
        what = e[0] if e else f"{out['factor']} {out['level']:g}"
        self._nt_status("robustness raster shown", C_GREEN)
        self._nt_rb_say(f"Raster tab: {what}, {out['post_spikes']} output "
                        "spikes (as in the sweep). Train / Test replace it.",
                        C_GREEN)
        self.log(f"[robust] raster of {what}: {out['post_spikes']} output "
                 "spikes, stream time from 0 (a probe run)")

    # ---- agent / bridge ---------------------------------------------------

    def _nt_rb_compact(self):
        return compact(self._nt_rb_result)

    def _nt_rb_context(self):
        """Lines for the agent's context: the last robustness sweep."""
        r = self._nt_rb_result
        if not r:
            return ["robustness sweep (nt_st_tab_robust): none yet."]
        meta = self._nt_rb_meta or {}
        return ([f"robustness sweep (STDP off, threshold held, one fixed "
                 f"stream, whole run counted; {r['seconds']:g} s per level, "
                 f"after {meta.get('trained_s', 0.0):.0f} s of training; "
                 f"nominal = {meta.get('how', '?')}; * = nominal; untrained: "
                 f"{r['untrained_note']}"
                 f"{'; STOPPED, partial' if r.get('stopped') else ''}):"]
                + summary_lines(r))
