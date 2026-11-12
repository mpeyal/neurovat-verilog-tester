"""Centre views of the Trainer's 'unsupervised stream' mode: Raster (live and
replayed input / output spikes and potential), Network (architecture and
weights), Patterns (the hidden patterns), Learning, Weights, Energy.

Render thread only. Live snapshots arrive as ``nt_stream_view`` queue
messages with every array already extracted in the worker
(:class:`vatester.stream_view.RunView`); ``_on_nt_stream_view`` only stores
the latest and ``_nt_tick_stream_view`` (called from ``App.run()``, guarded)
draws it. Plot items are created once and updated, never added per frame.
"""
import math
import time

import dearpygui.dearpygui as dpg
import numpy as np

from . import stream_view as sv
from .ui_constants import C_GREEN, C_MUTED, C_RED, C_TEXT2

MAX_PAT = 8                      # patterns / distractors / neurons drawn
MAX_OUT = 8
PAT_COL = [(80, 160, 255), (255, 160, 60), (90, 210, 120), (220, 110, 220),
           (240, 220, 80), (80, 220, 220), (255, 110, 110), (170, 150, 255)]
OUT_COL = [(255, 80, 80), (255, 225, 70), (110, 255, 140), (255, 120, 255),
           (90, 230, 255), (255, 170, 90), (235, 235, 235), (180, 160, 255)]
BG_COL = (118, 124, 140)
DIS_COL = (176, 140, 98)
DRAW_HZ = 30.0                   # replay redraw cap
SPEED = {"0.25x": 0.25, "1x": 1.0, "4x": 4.0}
DEVICE_LABEL = "selected device"


def _blend(c, a):
    return (c[0], c[1], c[2], a)


class NetworkStreamViewsMixin:
    _nt_st_rec = None            # RunView + spikes of the last finished run
    _nt_st_pending = None        # latest live snapshot, not yet drawn
    _nt_st_playing = False
    _nt_st_play_wall = 0.0
    _nt_st_draw_wall = 0.0
    _nt_st_tick_err = None
    _nt_st_arch = None           # architecture of the network shown
    _nt_st_net_w = None          # (n_out, shown) weights drawn in Network
    _nt_st_net_size = (0, 0)
    _nt_st_th = None

    # ---- themes -------------------------------------------------------

    def _nt_st_theme(self, kind, col, size=2.0):
        """Cached plot themes: 'dot' / 'tri' scatter markers, 'fill' shade,
        'line' line."""
        if self._nt_st_th is None:
            self._nt_st_th = {}
        key = (kind, tuple(col), size)
        th = self._nt_st_th.get(key)
        if th is not None:
            return th
        cat = dpg.mvThemeCat_Plots
        with dpg.theme() as th:
            if kind in ("dot", "tri"):
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, col, category=cat)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, col,
                                        category=cat)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker,
                                        dpg.mvPlotMarker_Circle if kind == "dot"
                                        else dpg.mvPlotMarker_Down, category=cat)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, size,
                                        category=cat)
            elif kind == "fill":
                with dpg.theme_component(dpg.mvShadeSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Fill, col, category=cat)
            else:
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, col, category=cat)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, size,
                                        category=cat)
        self._nt_st_th[key] = th
        return th

    # ---- layout -------------------------------------------------------

    def _nt_stream_views(self):
        """Centre views of the stream mode (inside ``nt_stream_center``)."""
        dpg.add_text("press Train", tag="nt_st_status", color=C_TEXT2, wrap=0)
        with dpg.tab_bar(tag="nt_st_tabs", callback=self._nt_st_on_tab):
            with dpg.tab(label="  Raster  ", tag="nt_st_tab_raster"):
                self._nt_st_raster_tab()
            with dpg.tab(label="  Network  ", tag="nt_st_tab_net"):
                with dpg.child_window(tag="nt_st_net_scroll", width=-1,
                                      height=-1, border=False,
                                      horizontal_scrollbar=True):
                    dpg.add_drawlist(width=900, height=560, tag="nt_st_net_dl")
            with dpg.tab(label="  Patterns  ", tag="nt_st_tab_pat"):
                with dpg.child_window(tag="nt_st_pat_holder", width=-1,
                                      height=-1, border=False,
                                      horizontal_scrollbar=True):
                    pass
            with dpg.tab(label="  Learning  ", tag="nt_st_tab_learn"):
                self._nt_st_learning_tab()
            with dpg.tab(label="  Weights  ", tag="nt_st_tab_weights"):
                self._small("Final weights of every neuron (0 = minimum, "
                            "1 = maximum conductance).", color=C_TEXT2)
                with dpg.plot(width=-1, height=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="weight",
                                      tag="nt_st_wx")
                    dpg.add_plot_axis(dpg.mvYAxis, label="synapses",
                                      tag="nt_st_wy")
            with dpg.tab(label="  Energy  ", tag="nt_st_tab_energy"):
                self._small("Uses the power profile chosen under Results > "
                            "Power in a supervised / unsupervised mode.",
                            color=C_TEXT2)
                dpg.add_input_text(tag="nt_st_energy", multiline=True,
                                   readonly=True, width=-1, height=-1,
                                   default_value="Device synapses only: run "
                                   "Train or Test with Synapses = "
                                   f"'{DEVICE_LABEL}'.")

    def _nt_st_learning_tab(self):
        with dpg.plot(width=-1, height=185, no_menus=True):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="stream time (s)",
                              tag="nt_st_x1")
            with dpg.plot_axis(dpg.mvYAxis, label="hit rate (%)",
                               tag="nt_st_y1"):
                dpg.add_line_series([], [], label="pattern windows answered ("
                                    + sv.hit_rule(False) + ")",
                                    tag="nt_st_s_hit")
        with dpg.plot(width=-1, height=185, no_menus=True):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="stream time (s)",
                              tag="nt_st_x2")
            with dpg.plot_axis(dpg.mvYAxis, label="per second / ms",
                               tag="nt_st_y2"):
                dpg.add_line_series([], [], label="false alarms / s",
                                    tag="nt_st_s_fa")
                dpg.add_line_series([], [], label="latency (ms)",
                                    tag="nt_st_s_lat")
        dpg.add_text("", tag="nt_st_verdict", wrap=0)
        self._small("SPIKES PER 50 ms WINDOW (end of run)", color=C_TEXT2,
                    tag="nt_st_table_cap")
        dpg.add_group(tag="nt_st_table_holder")

    def _nt_st_raster_tab(self):
        with dpg.group(horizontal=True):
            b = dpg.add_button(label=" Play ", tag="nt_st_play", width=70,
                               callback=lambda *_: self._nt_st_set_play(
                                   not self._nt_st_playing))
            with dpg.tooltip(b):
                dpg.add_text("Replay the last run from the slider position "
                             "(the bridge: set nt_st_play 1 / 0).", wrap=300)
            dpg.add_combo(sv.SPEEDS, default_value="1x", tag="nt_st_speed",
                          width=70)
            dpg.add_slider_float(tag="nt_st_view_t", width=-210,
                                 min_value=0.0, max_value=1.0,
                                 default_value=0.0, format="t = %.2f s",
                                 callback=self._nt_st_on_slider)
            with dpg.tooltip("nt_st_view_t"):
                dpg.add_text("Time in the last run (s) at the right edge of "
                             "the window shown.", wrap=300)
            self._small("window s", color=C_TEXT2)
            dpg.add_input_double(tag="nt_st_view_win", default_value=0.0,
                                 width=70, format="%.3g", step=0,
                                 callback=self._nt_st_on_slider)
            with dpg.tooltip("nt_st_view_win"):
                dpg.add_text("Seconds of stream in the window. 0 = auto: 1.5 s "
                             "for Poisson input, six presentations' worth for "
                             "images.", wrap=300)
        dpg.add_text("no run yet - press Train; the raster plays live while "
                     "the stream runs", tag="nt_st_view_status", color=C_TEXT2,
                     wrap=0)
        self._small("", color=C_MUTED, tag="nt_st_u_note")
        with dpg.subplots(2, 1, link_all_x=True, row_ratios=[3.0, 2.0],
                          width=-1, height=-1, tag="nt_st_sub"):
            with dpg.plot(no_menus=True, tag="nt_st_rplot"):
                dpg.add_plot_legend(horizontal=True, outside=True,
                                    location=dpg.mvPlot_Location_North)
                dpg.add_plot_axis(dpg.mvXAxis, label="stream time (s)",
                                  tag="nt_st_rx")
                with dpg.plot_axis(dpg.mvYAxis, label="afferent",
                                   tag="nt_st_ry"):
                    for p in range(MAX_PAT):
                        s = dpg.add_shade_series([], [], y2=[], show=False,
                                                 tag=f"nt_st_r_sh{p}")
                        dpg.bind_item_theme(s, self._nt_st_theme(
                            "fill", _blend(PAT_COL[p], 48)))
                    s = dpg.add_shade_series([], [], y2=[], show=False,
                                             tag="nt_st_r_dsh")
                    dpg.bind_item_theme(s, self._nt_st_theme(
                        "fill", _blend(DIS_COL, 26)))
                    s = dpg.add_scatter_series([], [], label="noise / background",
                                               tag="nt_st_r_bg")
                    dpg.bind_item_theme(s, self._nt_st_theme("dot", BG_COL, 1.6))
                    s = dpg.add_scatter_series([], [], label="distractor",
                                               tag="nt_st_r_dis", show=False)
                    dpg.bind_item_theme(s, self._nt_st_theme("dot", DIS_COL, 2.0))
                    for p in range(MAX_PAT):
                        s = dpg.add_scatter_series([], [], label=f"pattern {p + 1}",
                                                   tag=f"nt_st_r_p{p}", show=False)
                        dpg.bind_item_theme(s, self._nt_st_theme(
                            "dot", PAT_COL[p], 2.4))
                    for j in range(MAX_OUT):
                        s = dpg.add_scatter_series([], [], label=f"neuron {j + 1} spike",
                                                   tag=f"nt_st_r_o{j}", show=False)
                        dpg.bind_item_theme(s, self._nt_st_theme(
                            "tri", OUT_COL[j], 5.0))
            with dpg.plot(no_menus=True, tag="nt_st_uplot"):
                dpg.add_plot_legend(horizontal=True, outside=True,
                                    location=dpg.mvPlot_Location_North)
                dpg.add_plot_axis(dpg.mvXAxis, label="stream time (s)",
                                  tag="nt_st_ux")
                with dpg.plot_axis(dpg.mvYAxis, label="potential / threshold",
                                   tag="nt_st_uy"):
                    for p in range(MAX_PAT):
                        s = dpg.add_shade_series([], [], y2=[], show=False,
                                                 tag=f"nt_st_u_sh{p}")
                        dpg.bind_item_theme(s, self._nt_st_theme(
                            "fill", _blend(PAT_COL[p], 40)))
                    s = dpg.add_inf_line_series([1.0], horizontal=True,
                                                label="threshold",
                                                tag="nt_st_u_thr")
                    dpg.bind_item_theme(s, self._nt_st_theme(
                        "line", (205, 210, 225, 190), 1.0))
                    for j in range(MAX_OUT):
                        s = dpg.add_line_series([], [], label=f"neuron {j + 1}",
                                                tag=f"nt_st_u_l{j}", show=False,
                                                skip_nan=True)
                        dpg.bind_item_theme(s, self._nt_st_theme(
                            "line", OUT_COL[j], 1.2))
                    for j in range(MAX_OUT):
                        s = dpg.add_scatter_series([], [], tag=f"nt_st_u_o{j}",
                                                   show=False)
                        dpg.bind_item_theme(s, self._nt_st_theme(
                            "tri", OUT_COL[j], 5.0))

    # ---- raster drawing -----------------------------------------------

    def _nt_st_axes(self, arch):
        """Fixed y layout of the raster for ``arch``: (afferent rows, y of
        the output strip of neuron j, y limits, shade span)."""
        n_rows = int(arch["n_shown"])
        n_out = int(arch["n_out"])
        if arch["images"]:
            outy = [-1.5 - j for j in range(n_out)]
            lim = (-1.5 - n_out, n_rows - 0.5)
        else:
            outy = [n_rows + 0.5 + j for j in range(n_out)]
            lim = (-0.5, n_rows + n_out + 0.5)
        return outy, lim, (-0.5, n_rows - 0.5)

    @staticmethod
    def _nt_st_bands(starts, ends, lo, hi):
        x, y1, y2 = [], [], []
        for a, b in zip(starts, ends):
            x += [a, a, b, b]
            y1 += [lo, lo, lo, lo]
            y2 += [lo, hi, hi, lo]
        return [x, y1, y2]

    def _nt_st_draw_view(self, snap, arch):
        """Draw one window (live or replay) into the Raster plots."""
        outy, lim, (slo, shi) = self._nt_st_axes(arch)
        n_out = int(arch["n_out"])
        for p in range(MAX_PAT):
            on = p < len(snap["pat"])
            xs, ys = snap["pat"][p] if on else ([], [])
            dpg.configure_item(f"nt_st_r_p{p}", show=on)
            dpg.set_value(f"nt_st_r_p{p}", [xs, ys])
            st, en = snap["win"][p] if on else ([], [])
            for tag, lo, hi in ((f"nt_st_r_sh{p}", slo, shi),
                                (f"nt_st_u_sh{p}", -0.8, 1.3)):
                dpg.configure_item(tag, show=on and bool(st))
                dpg.set_value(tag, self._nt_st_bands(st, en, lo, hi))
        dpg.set_value("nt_st_r_bg", list(snap["bg"]))
        has_dis = bool(snap["dwin"])
        dpg.configure_item("nt_st_r_dis", show=has_dis)
        dpg.set_value("nt_st_r_dis", list(snap["dis"]))
        st = sum((w[0] for w in snap["dwin"]), [])
        en = sum((w[1] for w in snap["dwin"]), [])
        dpg.configure_item("nt_st_r_dsh", show=has_dis and bool(st))
        dpg.set_value("nt_st_r_dsh", self._nt_st_bands(st, en, slo, shi))
        u = snap["u"]
        for j in range(MAX_OUT):
            on = j < n_out
            xs = snap["post"][j] if on else []
            dpg.configure_item(f"nt_st_r_o{j}", show=on)
            dpg.set_value(f"nt_st_r_o{j}", [xs, [outy[j]] * len(xs) if on else []])
            dpg.configure_item(f"nt_st_u_o{j}", show=on and u is not None)
            dpg.set_value(f"nt_st_u_o{j}", [xs, [1.15] * len(xs)])
            line = on and u is not None
            dpg.configure_item(f"nt_st_u_l{j}", show=line)
            dpg.set_value(f"nt_st_u_l{j}", [u[0], u[1][j]] if line else [[], []])
        for ax in ("nt_st_rx", "nt_st_ux"):
            dpg.set_axis_limits(ax, snap["t0"], snap["t1"])
        dpg.set_axis_limits("nt_st_ry", *lim)
        dpg.set_axis_limits("nt_st_uy", -0.8, 1.3)
        names = arch.get("names") or []
        present = ", ".join(names[p] if p < len(names) else f"pattern {p + 1}"
                            for p in snap["present"]) or "none"
        per = ", ".join(f"n{j + 1}: {c}" for j, c in
                        enumerate(snap["post_per_neuron"]))
        mode = "LIVE" if snap["live"] else "replay"
        playing = " (playing)" if self._nt_st_playing and not snap["live"] else ""
        self._nt_st_view_say(
            f"{mode}{playing}: run time {snap['t_run_s']:.2f} of "
            f"{snap['run_s']:.1f} s (stream {snap['t1']:.2f} s) - "
            f"{snap['n_windows']} pattern window(s) in view ({present}) - "
            f"{snap['n_post']} output spike(s) in view ({per})")

    def _nt_st_view_say(self, text, color=C_TEXT2):
        if dpg.does_item_exist("nt_st_view_status"):
            dpg.set_value("nt_st_view_status", text)
            dpg.configure_item("nt_st_view_status", color=color)

    def _nt_st_setup_raster(self, arch):
        """Axis labels / direction for the input kind (on run start / end)."""
        if arch["images"]:
            label = "afferent (pixel row, 0 = top)"
        else:
            label = (f"afferent ({arch['n_shown']} of {arch['n_aff']} "
                     "afferents shown)")
        dpg.configure_item("nt_st_ry", label=label, invert=bool(arch["images"]))
        n = int(arch["n_out"])
        for j in range(MAX_OUT):
            dpg.configure_item(f"nt_st_r_o{j}", show=j < n)
        names = arch.get("names") or []
        for p in range(MAX_PAT):
            lab = names[p] if p < len(names) else f"pattern {p + 1}"
            dpg.configure_item(f"nt_st_r_p{p}", label=lab)

    # ---- live / replay --------------------------------------------------

    def _on_nt_stream_view(self, snap):
        """Queue handler: keep only the latest live snapshot (drawn by the
        tick, so a backlog costs one draw)."""
        self._nt_st_pending = snap

    def _nt_tick_stream_view(self):
        """Per frame, from App.run(): draw a pending live snapshot, or advance
        the replay. Never raises."""
        try:
            snap = self._nt_st_pending
            if snap is not None:
                self._nt_st_pending = None
                arch = snap.get("arch") or self._nt_st_arch
                if arch is not None:
                    self._nt_st_draw_view(snap, arch)
                    dpg.configure_item("nt_st_view_t",
                                       max_value=max(snap["run_s"], 1e-3))
                    dpg.set_value("nt_st_view_t", snap["t_run_s"])
                    if snap.get("weights") is not None:
                        self._nt_st_net_w = snap["weights"]
                        if self._nt_st_tab_is("nt_st_tab_net"):
                            self._nt_st_draw_net()
                return
            self._nt_st_fit_net()
            if not self._nt_st_playing:
                return
            rec = self._nt_st_rec
            if rec is None or self.trainer_running:
                self._nt_st_set_play(False)
                return
            now = time.monotonic()
            dt = now - self._nt_st_play_wall
            self._nt_st_play_wall = now
            speed = SPEED.get(self._nt_get("nt_st_speed", "1x"), 1.0)
            t = float(dpg.get_value("nt_st_view_t")) + dt * speed
            end = rec["view"].duration_ms / 1000.0
            if t >= end:
                t = end
                self._nt_st_set_play(False)
            dpg.set_value("nt_st_view_t", t)
            if now - self._nt_st_draw_wall >= 1.0 / DRAW_HZ or t >= end:
                self._nt_st_draw_wall = now
                self._nt_st_draw_replay(t)
        except Exception as exc:                        # noqa: BLE001
            self._nt_st_playing = False
            msg = f"{exc!r}"
            if msg != self._nt_st_tick_err:
                self._nt_st_tick_err = msg
                self.log(f"[stream] view error: {msg}")
            self._nt_st_view_say(f"view error: {exc}", C_RED)

    def _nt_st_draw_replay(self, t_s):
        rec = self._nt_st_rec
        if rec is None:
            return
        rv = rec["view"]
        rv.win_ms = sv.view_window_ms(self._nt_get("nt_st_view_win", 0.0),
                                      rv.images, rv.pattern_ms, rec["share"])
        snap = rv.snapshot(t_s * 1000.0, rec["post_t"], rec["post_j"],
                           rec["u"])
        self._nt_st_draw_view(snap, rec["arch"])

    def _nt_st_on_slider(self, *_):
        try:
            if self._nt_st_rec is not None and not self.trainer_running:
                self._nt_st_draw_replay(float(dpg.get_value("nt_st_view_t")))
        except Exception as exc:                        # noqa: BLE001
            self._nt_st_view_say(f"view error: {exc}", C_RED)

    def _nt_st_set_play(self, on):
        on = bool(on) and self._nt_st_rec is not None and not self.trainer_running
        if on:
            rec = self._nt_st_rec
            end = rec["view"].duration_ms / 1000.0
            if float(dpg.get_value("nt_st_view_t")) >= end - 1e-6:
                dpg.set_value("nt_st_view_t", 0.0)       # replay from the start
            self._nt_st_play_wall = time.monotonic()
        self._nt_st_playing = on
        if dpg.does_item_exist("nt_st_play"):
            dpg.configure_item("nt_st_play", label=" Pause " if on else " Play ")
        return on

    def _nt_st_keep_run(self, view, post_t, post_j, u, arch, share):
        """Recorder of the LAST run (replaces the previous one)."""
        self._nt_st_rec = {"view": view, "post_t": np.asarray(post_t, float),
                           "post_j": np.asarray(post_j, int), "u": u,
                           "arch": arch, "share": share}
        end = view.duration_ms / 1000.0
        dpg.configure_item("nt_st_view_t", max_value=max(end, 1e-3))
        dpg.set_value("nt_st_view_t", end)
        self._nt_st_draw_replay(end)

    # ---- learning / weights tabs ---------------------------------------

    def _nt_stream_table(self, table):
        dpg.delete_item("nt_st_table_holder", children_only=True)
        if not table:
            return
        n_pat = len(table[0]) - 1
        with dpg.table(parent="nt_st_table_holder", header_row=True,
                       borders_innerH=True, borders_innerV=True,
                       borders_outerH=True, borders_outerV=True,
                       policy=dpg.mvTable_SizingFixedFit):
            dpg.add_table_column(label="")
            for p in range(n_pat):
                dpg.add_table_column(label=f"pattern {p + 1}")
            dpg.add_table_column(label="no pattern")
            for j, row in enumerate(table):
                with dpg.table_row():
                    dpg.add_text(f"neuron {j + 1}", color=C_TEXT2)
                    for p, v in enumerate(row):
                        hot = v >= 0.9 if p < n_pat else False
                        bad = v > 0.05 if p == n_pat else False
                        dpg.add_text(f"{v:.2f}", color=C_GREEN if hot else
                                     C_RED if bad else C_MUTED)

    def _nt_stream_weights(self, W):
        dpg.delete_item("nt_st_wy", children_only=True)
        edges = np.linspace(0.0, 1.0, 41)
        mid = (0.5 * (edges[1:] + edges[:-1])).tolist()
        for j, w in enumerate(W):
            n, _ = np.histogram(np.clip(w, 0.0, 1.0), edges)
            dpg.add_line_series(mid, n.tolist(), label=f"neuron {j + 1}",
                                parent="nt_st_wy")
        dpg.fit_axis_data("nt_st_wx")
        dpg.fit_axis_data("nt_st_wy")

    # ---- network tab ------------------------------------------------------

    def _nt_st_tab_is(self, tag):
        if not dpg.does_item_exist("nt_st_tabs"):
            return False
        cur = dpg.get_value("nt_st_tabs")
        return cur == tag or (dpg.does_item_exist(tag)
                              and cur == dpg.get_alias_id(tag))

    def _nt_st_on_tab(self, *_):
        try:
            if self._nt_st_tab_is("nt_st_tab_net"):
                self._nt_st_draw_net()
            elif self._nt_st_tab_is("nt_st_tab_pat"):
                self._nt_st_draw_patterns()
        except Exception as exc:                        # noqa: BLE001
            self.log(f"[stream] view error: {exc!r}")

    def _nt_st_fit_net(self):
        """Keep the architecture drawlist as large as its scroll window."""
        if not dpg.does_item_exist("nt_st_net_scroll"):
            return
        size = dpg.get_item_rect_size("nt_st_net_scroll")
        if not size or size[0] < 60 or size[1] < 60:
            return
        if tuple(size) != self._nt_st_net_size:
            self._nt_st_net_size = tuple(size)
            self._nt_st_draw_net()

    def _nt_st_draw_net(self):
        dl = "nt_st_net_dl"
        if not dpg.does_item_exist(dl):
            return
        dpg.delete_item(dl, children_only=True)
        arch = self._nt_st_arch
        if arch is None:
            dpg.draw_text((16, 16), "set the network on the left", size=15,
                          color=C_MUTED, parent=dl)
            return
        if arch.get("error"):
            dpg.draw_text((16, 16), "input error: " + arch["error"], size=15,
                          color=C_RED, parent=dl)
            return
        idx = np.asarray(arch["net_idx"], int)
        k, n_out = idx.size, int(arch["n_out"])
        W = self._nt_st_net_w
        if W is None or np.shape(W) != (n_out, k):
            W = np.full((n_out, k), float(arch["w_init"]))
        avail_w, avail_h = self._nt_st_net_size or (900, 560)
        top = 112.0
        pitch = max(min(18.0, (max(avail_h, 420) - top - 70.0) / max(k, 1)), 9.0)
        span = pitch * k
        ax, nx = 70.0, 330.0
        ay = [top + (i + 0.5) * pitch for i in range(k)]
        npitch = span / max(n_out, 1)
        ny = [top + (j + 0.5) * npitch for j in range(n_out)]
        images = arch["images"]
        need_w = nx + 220
        cell = pitch
        if images:
            need_w = nx + 200 + (cell * arch["size"] + 18) * len(arch["vecs"]) \
                + 80 * n_out + 40
        width = int(max(avail_w - 16, need_w, 640))
        height = int(max(avail_h - 16, top + span + 70, 420))
        dpg.configure_item(dl, width=width, height=height)

        # caption
        syn = arch["synapse"]
        rule = arch["rule"]
        lines = [
            f"input: {arch['input']}   -   {arch['n_aff']} afferents -> "
            f"{n_out} output neuron(s), all-to-all, lateral inhibition",
            f"threshold {arch['threshold']:.4g} ({arch['thr_how']})   -   "
            f"synapses: {syn}   -   section {arch['pattern_ms']:g} ms",
            f"STDP: A+ {rule['a_plus']:g}, A-/A+ {rule['a_minus_ratio']:g}, "
            f"tau+ {rule['tau_plus']:g} ms, tau- {rule['tau_minus']:g} ms, "
            f"start weight {rule['w_init']:g}   -   neuron: tau m "
            f"{rule['tau_m']:g} ms, tau s {rule['tau_s']:g} ms",
        ]
        for i, s in enumerate(lines):
            dpg.draw_text((16, 8 + 18 * i), s, size=14,
                          color=(200, 208, 224) if i == 0 else C_TEXT2,
                          parent=dl)
        weights_note = ("weights: live" if self.trainer_running else
                        "weights: end of the last run"
                        if self._nt_stream_last else "weights: starting values")
        dpg.draw_text((16, 8 + 18 * 3), weights_note + "  (line colour and "
                      "width follow the weight)", size=13, color=C_MUTED,
                      parent=dl)

        # synapses
        for j in range(n_out):
            for i in range(k):
                w = float(np.clip(W[j, i], 0.0, 1.0))
                dpg.draw_line((ax + 7, ay[i]), (nx - 12, ny[j]),
                              color=self._cmap_color(w, int(70 + 170 * w)),
                              thickness=0.4 + 2.6 * w, parent=dl)
        # afferents
        for i in range(k):
            dpg.draw_circle((ax, ay[i]), min(5.0, pitch * 0.35),
                            fill=(150, 170, 210, 255), color=(60, 70, 90, 255),
                            parent=dl)
            if images or k <= 30:
                dpg.draw_text((ax - 52, ay[i] - 7),
                              (f"row {int(idx[i])}" if images
                               else f"#{int(idx[i])}"),
                              size=12, color=C_MUTED, parent=dl)
        cap = (f"{arch['n_aff']} pixel rows" if images else
               f"{k} of {arch['n_aff']} afferents drawn ...")
        dpg.draw_text((ax - 52, top + span + 8), cap, size=13, color=C_TEXT2,
                      parent=dl)
        dpg.draw_text((ax - 30, top - 22), "afferents", size=13,
                      color=C_TEXT2, parent=dl)
        # neurons and lateral inhibition
        dpg.draw_text((nx - 24, top - 22), "neurons", size=13, color=C_TEXT2,
                      parent=dl)
        for j in range(n_out):
            for jj in range(j + 1, n_out):
                mid = 0.5 * (ny[j] + ny[jj])
                bend = nx + 34 + 14 * (jj - j)
                dpg.draw_bezier_quadratic((nx + 12, ny[j]), (bend, mid),
                                          (nx + 12, ny[jj]),
                                          color=(255, 110, 110, 150),
                                          thickness=1.2, parent=dl)
        for j in range(n_out):
            col = OUT_COL[j % len(OUT_COL)]
            dpg.draw_circle((nx, ny[j]), 11, fill=_blend(col, 200),
                            color=(30, 30, 40, 255), parent=dl)
            dpg.draw_text((nx - 4, ny[j] - 8), f"{j + 1}", size=14,
                          color=(20, 20, 28, 255), parent=dl)
        if n_out > 1:
            dpg.draw_text((nx + 30, top + span + 8),
                          f"inhibition {rule['inh_alpha']:g} x thr, "
                          f"tau {rule['tau_inh']:g} ms", size=13,
                          color=(255, 140, 140), parent=dl)
        if not images:
            return
        # hidden images aligned with the afferent rows, then per-row weights
        size = int(arch["size"])
        x0 = nx + 200
        for p, vec in enumerate(arch["vecs"]):
            v = np.asarray(vec, float).reshape(size, size)
            for r in range(size):
                for c in range(size):
                    val = float(v[r, c])
                    g = int(30 + 215 * val) if val > 0 else 26
                    dpg.draw_rectangle((x0 + c * cell, top + r * cell),
                                       (x0 + (c + 1) * cell, top + (r + 1) * cell),
                                       fill=(g, g, min(255, g + 20), 255),
                                       color=(0, 0, 0, 0), parent=dl)
            names = arch.get("names") or []
            dpg.draw_text((x0, top - 22), names[p] if p < len(names)
                          else f"pattern {p + 1}", size=13,
                          color=PAT_COL[p % len(PAT_COL)], parent=dl)
            x0 += cell * size + 18
        bw = 70.0
        for j in range(n_out):
            col = OUT_COL[j % len(OUT_COL)]
            dpg.draw_text((x0, top - 22), f"n{j + 1} w", size=13, color=col,
                          parent=dl)
            dpg.draw_line((x0, top), (x0, top + span), color=(80, 88, 110, 255),
                          parent=dl)
            for i in range(k):
                w = float(np.clip(W[j, i], 0.0, 1.0))
                dpg.draw_rectangle((x0, ay[i] - pitch * 0.38),
                                   (x0 + bw * w, ay[i] + pitch * 0.38),
                                   fill=_blend(col, 210), color=(0, 0, 0, 0),
                                   parent=dl)
            x0 += bw + 10
        dpg.draw_text((nx + 200, top + span + 8), "hidden image(s), and each "
                      "neuron's weight per pixel row (bar length = weight, 0 to 1)", size=13,
                      color=C_TEXT2, parent=dl)

    # ---- patterns tab -----------------------------------------------------

    def _nt_st_thumb(self, vec, size, px=112, parent=0):
        dl = dpg.add_drawlist(width=px, height=px, parent=parent)
        v = np.asarray(vec, float).reshape(size, size)
        c = px / size
        for r in range(size):
            for k in range(size):
                val = float(v[r, k])
                g = int(30 + 215 * val) if val > 0 else 26
                dpg.draw_rectangle((k * c, r * c), ((k + 1) * c, (r + 1) * c),
                                   fill=(g, g, min(255, g + 20), 255),
                                   color=(0, 0, 0, 0), parent=dl)
        return dl

    def _nt_st_pat_block(self, parent, title, col, xs, ys, n_rows, t_ms,
                         vec=None, size=0, invert=True, ylabel="pixel row",
                         where=""):
        with dpg.group(parent=parent):
            dpg.add_text(f"{title}  -  {len(xs)} spikes over {t_ms:g} ms"
                         f"{where}", color=col)
            h = int(max(150, 6 * n_rows + 70))
            with dpg.group(horizontal=True):
                if vec is not None:
                    self._nt_st_thumb(vec, size, px=min(h - 20, 180))
                with dpg.plot(width=520, height=h, no_menus=True,
                              no_mouse_pos=True):
                    dpg.add_plot_axis(dpg.mvXAxis, label="ms after onset")
                    dpg.set_axis_limits(dpg.last_item(), 0.0, t_ms)
                    with dpg.plot_axis(dpg.mvYAxis, label=ylabel,
                                       invert=invert) as yax:
                        s = dpg.add_scatter_series(xs, ys)
                        dpg.bind_item_theme(s, self._nt_st_theme("dot", col, 2.6))
                    dpg.set_axis_limits(yax, -0.5, n_rows - 0.5)
            dpg.add_spacer(height=6)

    def _nt_st_draw_patterns(self):
        holder = "nt_st_pat_holder"
        if not dpg.does_item_exist(holder):
            return
        dpg.delete_item(holder, children_only=True)
        arch = self._nt_st_arch
        if arch is None:
            return
        if arch.get("error"):
            dpg.add_text("input error: " + arch["error"], color=C_RED,
                         wrap=0, parent=holder)
            return
        if arch["images"]:
            size = int(arch["size"])
            self._small(f"Column scan: one afferent per pixel row; pixel (r, c) "
                        f"at or above the ink threshold fires afferent r at "
                        f"c x {arch['bin_ms']:g} ms. Section {arch['pattern_ms']:g} "
                        "ms.", color=C_TEXT2, parent=holder, wrap=0)
            groups = [("hidden", arch["patterns"], arch["names"], arch["vecs"]),
                      ("distractor", arch["distractors"], arch["dnames"],
                       arch["dvecs"])]
            for what, evs, names, vecs in groups:
                for p, (a, t) in enumerate(evs):
                    col = (PAT_COL[p % len(PAT_COL)] if what == "hidden"
                           else DIS_COL)
                    self._nt_st_pat_block(
                        holder, f"{what} {p + 1}: {names[p]}", col,
                        np.asarray(t, float).tolist(),
                        np.asarray(a, float).tolist(), size,
                        arch["pattern_ms"], vecs[p], size)
            if not arch["distractors"]:
                self._small("No distractors: between presentations the "
                            f"afferents carry {arch['bg']:g} Hz background "
                            "(plus 'noise Hz' everywhere).", color=C_MUTED,
                            parent=holder, wrap=0)
            return
        sess = self._nt_stream_session
        copies = (getattr(getattr(sess, "stream", None), "copies", None)
                  if sess is not None and self._nt_st_same_net(arch) else None)
        if not copies:
            dpg.add_text("The Poisson patterns are frozen at the first Train: "
                         f"each is a {arch['pattern_ms']:g} ms section of its "
                         "afferents' own background activity, so it has no "
                         "rate cue. Press Train to see them here.",
                         color=C_TEXT2, wrap=0, parent=holder)
            return
        shown = np.asarray(arch["shown"], int)
        self._small(f"Frozen patterns on the {shown.size} of {arch['n_aff']} "
                    "afferents shown in the raster.", color=C_TEXT2,
                    parent=holder, wrap=0)
        for p, cp in enumerate(copies):
            xs, ys = sv.frozen_raster(cp, shown)
            self._nt_st_pat_block(
                holder, f"pattern {p + 1}", PAT_COL[p % len(PAT_COL)], xs, ys,
                shown.size, arch["pattern_ms"], invert=False,
                ylabel="afferent sample", where=f" on the shown afferents "
                f"({len(cp[0])} on all {arch['n_aff']})")
