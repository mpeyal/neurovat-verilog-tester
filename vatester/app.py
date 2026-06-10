"""NeuroVAT - neuromorphic Verilog-A model tester (DearPyGui).

Layout
  toolbar   brand - Run / quick actions / status
  left      .va browser - models - parameter editor - sim settings
  center    Signal Designer | Results | Analysis | Verilog-A Source | Log
  right     Claude agent chat (bubbles, pattern cards, prompt chips)
"""

import contextlib
import dataclasses
import importlib
import json
import os
import queue
import re
import shutil
import threading
import time

import dearpygui.dearpygui as dpg

from ecfet import (Waveform, EcfetV1, V1Params, EcfetV2, V2Params,
                   FeFET, FeFETParams, simulate)
from . import signal_factory as sf
from .agent import ClaudeAgent
from .va_scan import scan as va_scan
from .virtuoso import VirtuosoLink

APP_TITLE = "NeuroVAT - Neuromorphic Verilog-A Tester"
LEFT_W, RIGHT_W = 330, 312
BUBBLE_W = RIGHT_W - 44
BUBBLE_INDENT = 26

# ---------------------------------------------------------------- palette --
C_BG      = (13, 15, 20)
C_SURF    = (22, 25, 32)
C_SURF2   = (30, 34, 43)
C_SURF3   = (39, 44, 56)
C_BORDER  = (47, 53, 67)
C_ACC     = (96, 134, 255)
C_ACC_H   = (118, 152, 255)
C_ACC_A   = (78, 112, 228)
C_TEXT    = (231, 235, 243)
C_TEXT2   = (158, 166, 184)
C_MUTED   = (108, 116, 134)
C_GREEN   = (86, 212, 150)
C_AMBER   = (255, 193, 94)
C_RED     = (255, 109, 116)
C_AGENT   = (255, 178, 102)

C_BUB_USER   = (37, 50, 86)
C_BUB_USER_B = (66, 88, 148)
C_BUB_AGENT  = (30, 34, 43)
C_BUB_AGENT_B = (50, 56, 70)
C_BUB_ERR    = (62, 30, 36)
C_BUB_ERR_B  = (130, 56, 64)

# (display label, CLI/API model id or None for the user's default, blurb)
MODEL_CHOICES = [
    ("Default", None, "your Claude Code default"),
    ("Fable 5", "claude-fable-5", "most powerful"),
    ("Opus 4.8", "claude-opus-4-8", "most capable Opus"),
    ("Sonnet 4.6", "claude-sonnet-4-6", "fast + smart"),
    ("Haiku 4.5", "claude-haiku-4-5", "fastest, cheapest"),
]

QUICK_CHIPS = [
    ("Poisson train",
     "Generate a potentiating Poisson spike train: 50 Hz, 1 s, -100 pA, 2 ms spikes."),
    ("STDP pairs",
     "Create an STDP experiment: 20 pre/post pairs, dt = +20 ms, +-100 pA, 5 ms width."),
    ("FeFET LTP/LTD",
     "Generate a FeFET voltage pattern: 15 potentiating +1.5 V / 10 ms pulses then 15 of -1.5 V."),
    ("Explain .va",
     "Explain the selected Verilog-A file and summarize its parameters."),
    ("Review v2",
     "Review ecfet_v2.va for convergence hazards or sign-convention bugs."),
    ("Linear LTP",
     "Suggest v2 parameter values for a more linear LTP curve (nu_p, nu_d, n_states)."),
]


@dataclasses.dataclass
class ModelSpec:
    key: str
    label: str
    cls: object
    params_cls: object
    input_kind: str


MODEL_SPECS = [
    ModelSpec("v1", "ECFET v1 (Verilog-A port)", EcfetV1, V1Params, "current"),
    ModelSpec("v2", "ECFET v2 (practical ECRAM)", EcfetV2, V2Params, "current"),
    ModelSpec("fefet", "FeFET (Merz/Preisach-lite)", FeFET, FeFETParams, "voltage"),
]
SPEC_BY_KEY = {s.key: s for s in MODEL_SPECS}
GEN_BY_NAME = {g.name: g for g in sf.GENERATORS}

# model_key -> the Python "twin" source the GUI actually simulates
TWIN_FILE = {"v1": "ecfet/model_v1.py", "v2": "ecfet/model_v2.py",
             "fefet": "ecfet/model_fefet.py"}
LABEL_TO_KEY = {s.label: s.key for s in MODEL_SPECS}

# modules to hot-reload when the agent edits a twin, so the GUI re-simulates
# with the new code (key -> (module, class name, params class name))
from ecfet import model_v1 as _m_v1, model_v2 as _m_v2, model_fefet as _m_fefet
RELOAD_MODULES = {
    "v1": (_m_v1, "EcfetV1", "V1Params"),
    "v2": (_m_v2, "EcfetV2", "V2Params"),
    "fefet": (_m_fefet, "FeFET", "FeFETParams"),
}


def _defaults_of(params_cls):
    inst = params_cls()
    return {f.name: getattr(inst, f.name)
            for f in dataclasses.fields(params_cls)}


class App:
    def __init__(self, workdir):
        self.workdir = os.path.abspath(workdir)
        self.q = queue.Queue()
        self.agent = ClaudeAgent(self.workdir)
        self.va_files = []
        self.selected_va = None
        self.editor_path = None
        self.editor_mtime = 0.0
        self.param_values = {s.key: _defaults_of(s.params_cls)
                             for s in MODEL_SPECS}
        self.gen_values = {}
        self.results = []
        self.results_meta = {}
        self.results_unit = "pA"
        self.file_enabled = {}     # va filename -> simulate its model twin?
        self._last_va_click = ("", 0.0)
        self._status_themes = {}   # color tuple -> flat-button theme cache
        self.analysis_metric = "G"  # "G" (conductance) or "R" (resistance)
        self._twin_mtimes = None    # live-watch state for the model twins
        self._prev_mtimes = None
        self._watch_tick = 0
        self._backup_dir = None     # snapshot of editable files before agent
        self._series = []
        self._ana_series = []
        self.sim_running = False
        self.chat_busy = False
        self.virtuoso = VirtuosoLink()
        self.virt_busy = False
        self.agent_model_label = "Default"
        self.agent_model_id = None
        self.fonts = {}
        self.themes = {}
        self._zoom_anim = {}       # axis -> [cur_lo, cur_hi, tgt_lo, tgt_hi]
        self._zoom_release = []    # axes to unlock next frame
        self._probes = {}          # plot -> {"A"/"B": {pt, ann, sid, x, y}}
        self._probe_armed = None   # "A"/"B" while waiting for placement click
        self._stdp_series = []
        self._stdp_ctx = None
        self._stdp_summary = []
        self._tip_frame = -10
        self._hover_ann = {}       # plot -> in-canvas hover bubble annotation

    # =================================================================
    # fonts & themes
    # =================================================================

    def _font(self):
        fdir = r"C:\Windows\Fonts"
        files = {
            "body":  ("segoeui.ttf", 17),
            "small": ("segoeui.ttf", 14),
            "bold":  ("segoeuib.ttf", 17),
            "h2":    ("seguisb.ttf", 21),
            "title": ("segoeuib.ttf", 24),
            "mono":  ("consola.ttf", 15),
        }
        with dpg.font_registry():
            for key, (fn, size) in files.items():
                path = os.path.join(fdir, fn)
                if not os.path.isfile(path):
                    path = os.path.join(fdir, "segoeui.ttf")
                if not os.path.isfile(path):
                    continue
                self.fonts[key] = dpg.add_font(path, size)
        if "body" in self.fonts:
            dpg.bind_font(self.fonts["body"])

    def _mk_theme(self, name, colors=(), styles=(), plot_colors=()):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvAll):
                for c, v in colors:
                    col = getattr(dpg, c, None)
                    if col is not None:
                        dpg.add_theme_color(col, v)
                for s, v in styles:
                    st = getattr(dpg, s, None)
                    if st is not None:
                        if isinstance(v, tuple):
                            dpg.add_theme_style(st, *v)
                        else:
                            dpg.add_theme_style(st, v)
                for c, v in plot_colors:
                    col = getattr(dpg, c, None)
                    if col is not None:
                        dpg.add_theme_color(col, v,
                                            category=dpg.mvThemeCat_Plots)
        self.themes[name] = t
        return t

    def _theme(self):
        g = self._mk_theme(
            "global",
            colors=[
                ("mvThemeCol_WindowBg", C_BG),
                ("mvThemeCol_ChildBg", C_SURF),
                ("mvThemeCol_PopupBg", (27, 30, 39)),
                ("mvThemeCol_MenuBarBg", C_BG),
                ("mvThemeCol_FrameBg", C_SURF2),
                ("mvThemeCol_FrameBgHovered", C_SURF3),
                ("mvThemeCol_FrameBgActive", (45, 51, 65)),
                ("mvThemeCol_Button", C_SURF2),
                ("mvThemeCol_ButtonHovered", C_SURF3),
                ("mvThemeCol_ButtonActive", (49, 56, 72)),
                ("mvThemeCol_Header", (37, 50, 86)),
                ("mvThemeCol_HeaderHovered", (46, 62, 106)),
                ("mvThemeCol_HeaderActive", (52, 70, 118)),
                ("mvThemeCol_Tab", (0, 0, 0, 0)),
                ("mvThemeCol_TabHovered", C_SURF3),
                ("mvThemeCol_TabActive", C_SURF2),
                ("mvThemeCol_TabUnfocused", (0, 0, 0, 0)),
                ("mvThemeCol_TabUnfocusedActive", C_SURF2),
                ("mvThemeCol_CheckMark", C_ACC),
                ("mvThemeCol_SliderGrab", C_ACC),
                ("mvThemeCol_SliderGrabActive", C_ACC_H),
                ("mvThemeCol_Text", C_TEXT),
                ("mvThemeCol_TextDisabled", C_MUTED),
                ("mvThemeCol_Border", C_BORDER),
                ("mvThemeCol_Separator", (44, 50, 63)),
                ("mvThemeCol_ScrollbarBg", (0, 0, 0, 0)),
                ("mvThemeCol_ScrollbarGrab", (58, 64, 80)),
                ("mvThemeCol_ScrollbarGrabHovered", (74, 82, 102)),
                ("mvThemeCol_ScrollbarGrabActive", (90, 100, 124)),
                ("mvThemeCol_PlotHistogram", C_ACC),
                ("mvThemeCol_TextSelectedBg", (52, 70, 118)),
            ],
            styles=[
                ("mvStyleVar_WindowRounding", 0),
                ("mvStyleVar_ChildRounding", 10),
                ("mvStyleVar_FrameRounding", 7),
                ("mvStyleVar_PopupRounding", 8),
                ("mvStyleVar_GrabRounding", 6),
                ("mvStyleVar_TabRounding", 6),
                ("mvStyleVar_ScrollbarRounding", 8),
                ("mvStyleVar_WindowPadding", (14, 12)),
                ("mvStyleVar_FramePadding", (10, 4)),
                ("mvStyleVar_CellPadding", (8, 3)),
                ("mvStyleVar_ItemSpacing", (10, 5)),
                ("mvStyleVar_ItemInnerSpacing", (8, 4)),
                ("mvStyleVar_ScrollbarSize", 11),
                ("mvStyleVar_ChildBorderSize", 1),
                ("mvStyleVar_WindowBorderSize", 0),
            ],
            plot_colors=[
                ("mvPlotCol_FrameBg", (0, 0, 0, 0)),
                ("mvPlotCol_PlotBg", (17, 19, 25)),
                ("mvPlotCol_PlotBorder", (47, 53, 67, 160)),
                ("mvPlotCol_LegendBg", (22, 25, 32, 230)),
                ("mvPlotCol_LegendBorder", (47, 53, 67, 160)),
                ("mvPlotCol_AxisGrid", (255, 255, 255, 18)),
                ("mvPlotCol_AxisText", C_TEXT2),
                ("mvPlotCol_Crosshairs", (158, 166, 184, 140)),
            ])
        dpg.bind_theme(g)

        self._mk_theme("primary", colors=[
            ("mvThemeCol_Button", C_ACC),
            ("mvThemeCol_ButtonHovered", C_ACC_H),
            ("mvThemeCol_ButtonActive", C_ACC_A),
            ("mvThemeCol_Text", (250, 251, 255)),
        ], styles=[("mvStyleVar_FrameRounding", 8),
                   ("mvStyleVar_FramePadding", (16, 7))])

        self._mk_theme("stop", colors=[
            ("mvThemeCol_Button", (150, 52, 58)),
            ("mvThemeCol_ButtonHovered", (182, 64, 70)),
            ("mvThemeCol_ButtonActive", (200, 70, 76)),
            ("mvThemeCol_Text", (250, 250, 250)),
        ], styles=[("mvStyleVar_FrameRounding", 8),
                   ("mvStyleVar_FramePadding", (16, 7))])

        self._mk_theme("chip", colors=[
            ("mvThemeCol_Button", (35, 40, 52)),
            ("mvThemeCol_ButtonHovered", (47, 54, 70)),
            ("mvThemeCol_ButtonActive", (56, 64, 84)),
            ("mvThemeCol_Text", C_TEXT2),
        ], styles=[("mvStyleVar_FrameRounding", 12),
                   ("mvStyleVar_FramePadding", (10, 4))])

        self._mk_theme("card", colors=[
            ("mvThemeCol_ChildBg", C_SURF2),
            ("mvThemeCol_Border", C_BORDER),
        ], styles=[("mvStyleVar_ChildRounding", 9),
                   ("mvStyleVar_WindowPadding", (10, 8))])

        self._mk_theme("bub_user", colors=[
            ("mvThemeCol_ChildBg", C_BUB_USER),
            ("mvThemeCol_Border", C_BUB_USER_B),
        ], styles=[("mvStyleVar_ChildRounding", 12),
                   ("mvStyleVar_WindowPadding", (12, 9))])

        self._mk_theme("bub_agent", colors=[
            ("mvThemeCol_ChildBg", C_BUB_AGENT),
            ("mvThemeCol_Border", C_BUB_AGENT_B),
        ], styles=[("mvStyleVar_ChildRounding", 12),
                   ("mvStyleVar_WindowPadding", (12, 9))])

        self._mk_theme("bub_err", colors=[
            ("mvThemeCol_ChildBg", C_BUB_ERR),
            ("mvThemeCol_Border", C_BUB_ERR_B),
        ], styles=[("mvStyleVar_ChildRounding", 12),
                   ("mvStyleVar_WindowPadding", (12, 9))])

        self._mk_theme("pattern_card", colors=[
            ("mvThemeCol_ChildBg", (26, 36, 58)),
            ("mvThemeCol_Border", (70, 96, 168)),
        ], styles=[("mvStyleVar_ChildRounding", 12),
                   ("mvStyleVar_WindowPadding", (12, 10))])

        self._mk_theme("panel_flat", colors=[
            ("mvThemeCol_ChildBg", (0, 0, 0, 0)),
        ])

        self._mk_theme("table_roomy", styles=[
            ("mvStyleVar_CellPadding", (10, 3)),
        ])

        self._mk_theme("table_list", styles=[
            ("mvStyleVar_CellPadding", (4, 3)),
            ("mvStyleVar_ItemSpacing", (4, 3)),
        ])

        self._mk_theme("ruler_series", plot_colors=[
            ("mvPlotCol_Line", (175, 185, 210, 170)),
        ])

        self._mk_theme("tip", colors=[
            ("mvThemeCol_WindowBg", (18, 21, 28)),
            ("mvThemeCol_Border", (70, 96, 168)),
        ], styles=[("mvStyleVar_WindowRounding", 6),
                   ("mvStyleVar_WindowPadding", (10, 8)),
                   ("mvStyleVar_WindowBorderSize", 1),
                   ("mvStyleVar_ItemSpacing", (8, 3))])

    # ---- tiny helpers ------------------------------------------------

    def _small(self, text, color=C_MUTED, parent=0, wrap=0, tag=0):
        kw = {}
        if parent:
            kw["parent"] = parent
        if tag:
            kw["tag"] = tag
        if wrap and wrap > 0:
            kw["wrap"] = wrap
        t = dpg.add_text(text, color=color, **kw)
        if "small" in self.fonts:
            dpg.bind_item_font(t, self.fonts["small"])
        return t

    def _section(self, label, pad=6):
        dpg.add_spacer(height=pad)
        self._small(label.upper(), color=(126, 150, 220))
        dpg.add_separator()

    def _caption(self, text):
        self._small(text, color=C_MUTED)

    @contextlib.contextmanager
    def _pad(self, left=8, top=2, bottom=4):
        """Inner padding for borderless child windows / section bodies."""
        if top:
            dpg.add_spacer(height=top)
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=left)
            with dpg.group():
                yield
        if bottom:
            dpg.add_spacer(height=bottom)

    # ---- plot zoom / fit controls ------------------------------------
    # Wheel zoom is reimplemented here: instead of ImPlot's hard step per
    # notch, targets are animated each frame (exponential easing) and the
    # zoom is centered on the mouse cursor.

    ZOOM_PLOTS = {
        "preview_plot": (("prev_x",), ("prev_y",)),
        "plot_i": (("ax_i_x", "ax_r_x", "ax_g_x"), ("ax_i_y",)),
        "plot_r": (("ax_i_x", "ax_r_x", "ax_g_x"), ("ax_r_y",)),
        "plot_g": (("ax_i_x", "ax_r_x", "ax_g_x"), ("ax_g_y",)),
        "ana_plot": (("ana_x",), ("ana_y",)),
        "stdp_plot": (("stdp_x",), ("stdp_y",)),
    }

    def _axis_view(self, ax):
        a = self._zoom_anim.get(ax)
        return (a[2], a[3]) if a else tuple(dpg.get_axis_limits(ax))

    def _push_zoom(self, ax, factor, center=None):
        lo, hi = self._axis_view(ax)
        span = hi - lo
        if span <= 0 or (factor < 1.0 and span * factor < 1e-12):
            return
        c = 0.5 * (lo + hi) if center is None else center
        t_lo = c - (c - lo) * factor
        t_hi = c + (hi - c) * factor
        cur = self._zoom_anim.get(ax)
        if cur:                       # retarget mid-animation (compounds)
            cur[2], cur[3] = t_lo, t_hi
        else:
            self._zoom_anim[ax] = [lo, hi, t_lo, t_hi]

    def _tick_zoom_anim(self):
        for ax in self._zoom_release:
            if dpg.does_item_exist(ax) and ax not in self._zoom_anim:
                dpg.set_axis_limits_auto(ax)
        self._zoom_release = []
        if not self._zoom_anim:
            return
        done = []
        for ax, st in self._zoom_anim.items():
            if not dpg.does_item_exist(ax):
                done.append(ax)
                continue
            st[0] += (st[2] - st[0]) * 0.28
            st[1] += (st[3] - st[1]) * 0.28
            span = abs(st[3] - st[2]) or 1.0
            if (abs(st[2] - st[0]) < span * 2e-3
                    and abs(st[3] - st[1]) < span * 2e-3):
                dpg.set_axis_limits(ax, st[2], st[3])
                done.append(ax)
            else:
                dpg.set_axis_limits(ax, st[0], st[1])
        for ax in done:
            self._zoom_anim.pop(ax, None)
            self._zoom_release.append(ax)

    def _on_wheel(self, sender, delta):
        if not delta:
            return
        hovered = next((p for p in self.ZOOM_PLOTS
                        if dpg.does_item_exist(p) and dpg.is_item_hovered(p)),
                       None)
        if not hovered:
            return
        try:
            mx, my = dpg.get_plot_mouse_pos()
        except Exception:
            mx = my = None
        f = 0.85 ** float(delta)        # wheel up -> zoom in, smooth ramp
        xs, ys = self.ZOOM_PLOTS[hovered]
        for ax in xs:
            if dpg.does_item_exist(ax):
                self._push_zoom(ax, f, mx)
        for ay in ys:
            if dpg.does_item_exist(ay):
                self._push_zoom(ay, f, my)

    def _zoom_axes(self, axes, factor):
        for ax in axes:
            if dpg.does_item_exist(ax):
                self._push_zoom(ax, factor)

    def _fit_axes_of(self, axes):
        for ax in axes:
            self._zoom_anim.pop(ax, None)
            if dpg.does_item_exist(ax):
                dpg.set_axis_limits_auto(ax)
                dpg.fit_axis_data(ax)

    # per-plot hover labels: (x_name, x_unit, y_name, y_unit); "{unit}" is
    # resolved from the current signal-unit combo at runtime.
    HOVER_FMT = {
        "preview_plot": ("t", "s", "amp", "{unit}"),
        "plot_i": ("t", "s", "stim", "{unit}"),
        "plot_r": ("t", "s", "R", "ohm"),
        "plot_g": ("t", "s", "G", "uS"),
        "ana_plot": ("pulse", "", "G", "uS"),
        "stdp_plot": ("dt", "ms", "dG", "uS"),
    }

    def _bind_plot_hover_handlers(self):
        """Per-plot hover handlers. get_plot_mouse_pos() is only valid inside
        an input callback (not when polled in the render loop), which is why
        the tooltip must be driven from a hover handler, like the click path.
        """
        for plot in self.HOVER_FMT:
            if not dpg.does_item_exist(plot):
                continue
            reg = f"hh_{plot}"
            if not dpg.does_item_exist(reg):
                with dpg.item_handler_registry(tag=reg):
                    dpg.add_item_hover_handler(callback=self._on_plot_hover,
                                               user_data=plot)
            dpg.bind_item_handler_registry(plot, reg)

    def _nearest_series_point(self, yax, mx, my, xs_span, ys_span):
        """Nearest data point (axis-normalized) across a plot's data series,
        skipping ## helper series. Returns (d2, x, y, label) or None. Robust
        to series that vanish mid-frame."""
        best = None
        for sid in dpg.get_item_children(yax, 1) or []:
            try:
                lbl = dpg.get_item_configuration(sid).get("label", "") or ""
                if lbl.startswith("##"):
                    continue
                val = dpg.get_value(sid)
            except Exception:
                continue
            if not val or len(val) < 2 or not val[0]:
                continue
            for x, y in zip(val[0], val[1]):
                d = ((x - mx) / xs_span) ** 2 + ((y - my) / ys_span) ** 2
                if best is None or d < best[0]:
                    best = (d, float(x), float(y), lbl)
        return best

    def _hide_hover_bubble(self):
        for aid in self._hover_ann.values():
            if dpg.does_item_exist(aid):
                dpg.hide_item(aid)

    def _hide_tip_if_stale(self):
        # the hover handler stops firing once the cursor leaves the plot;
        # hide the in-canvas bubble shortly after.
        if dpg.get_frame_count() - self._tip_frame > 1:
            self._hide_hover_bubble()

    def _on_plot_hover(self, sender, app_data, user_data):
        """Draw an in-canvas bubble at the nearest data point, showing its
        x / y values with units. Fires while the plot is hovered."""
        plot = user_data
        if plot not in self.HOVER_FMT:
            return
        try:
            mx, my = dpg.get_plot_mouse_pos()
        except Exception:
            self._hide_hover_bubble()
            return
        xax, yax = self.PROBE_AXES[plot]
        x0, x1 = dpg.get_axis_limits(xax)
        y0, y1 = dpg.get_axis_limits(yax)
        xs_span = (x1 - x0) or 1.0
        ys_span = (y1 - y0) or 1.0
        best = self._nearest_series_point(yax, mx, my, xs_span, ys_span)
        if best is None or best[0] > 0.05 ** 2:
            self._hide_hover_bubble()
            return
        _, x, y, lbl = best
        xname, xunit, yname, yunit = self.HOVER_FMT[plot]
        unit = dpg.get_value("unit_combo") if dpg.does_item_exist(
            "unit_combo") else ""
        xunit = xunit.replace("{unit}", unit)
        yunit = yunit.replace("{unit}", unit)
        text = (f"{yname} = {y:.4g} {yunit}".rstrip() + "\n"
                + f"{xname} = {x:.4g} {xunit}".rstrip())
        self._show_hover_bubble(plot, x, y, text)
        self._tip_frame = dpg.get_frame_count()

    def _show_hover_bubble(self, plot, x, y, text):
        # only one bubble visible at a time
        for p, aid in self._hover_ann.items():
            if p != plot and dpg.does_item_exist(aid):
                dpg.hide_item(aid)
        aid = self._hover_ann.get(plot)
        if aid and dpg.does_item_exist(aid):
            dpg.configure_item(aid, default_value=(x, y), label=text,
                               show=True)
        else:
            self._hover_ann[plot] = dpg.add_plot_annotation(
                parent=plot, label=text, default_value=(x, y),
                offset=(14, -14), color=(46, 86, 158, 235), clamped=True)

    # ---- A/B probe markers (Virtuoso-style) --------------------------

    PROBE_AXES = {
        "preview_plot": ("prev_x", "prev_y"),
        "plot_i": ("ax_i_x", "ax_i_y"),
        "plot_r": ("ax_r_x", "ax_r_y"),
        "plot_g": ("ax_g_x", "ax_g_y"),
        "ana_plot": ("ana_x", "ana_y"),
        "stdp_plot": ("stdp_x", "stdp_y"),
    }
    PLOT_NAMES = {"plot_i": "stimulus", "plot_r": "R", "plot_g": "G",
                  "preview_plot": "preview", "ana_plot": "G/pulse",
                  "stdp_plot": "STDP"}
    PROBE_READOUT = {"plot_i": "probe_results", "plot_r": "probe_results",
                     "plot_g": "probe_results",
                     "preview_plot": "probe_prev", "ana_plot": "probe_ana",
                     "stdp_plot": "probe_stdp"}
    PROBE_COLORS = {"A": (90, 190, 255, 255), "B": (255, 170, 90, 255)}

    def _nearest_point(self, plot, mx, my):
        """Closest data point on any series of the plot (axis-normalized)."""
        xax, yax = self.PROBE_AXES[plot]
        x0, x1 = dpg.get_axis_limits(xax)
        y0, y1 = dpg.get_axis_limits(yax)
        xs_span = (x1 - x0) or 1.0
        ys_span = (y1 - y0) or 1.0
        best = None
        for sid in dpg.get_item_children(yax, 1) or []:
            val = dpg.get_value(sid)
            if not val or len(val) < 2 or not val[0]:
                continue
            label = dpg.get_item_configuration(sid).get("label", "")
            if label.startswith("##"):       # probe ruler, not data
                continue
            for x, y in zip(val[0], val[1]):
                d = (((x - mx) / xs_span) ** 2
                     + ((y - my) / ys_span) ** 2)
                if best is None or d < best[0]:
                    best = (d, float(x), float(y), label, sid)
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]

    def _set_probe(self, plot, which, x, y, slabel, sid):
        probes = self._probes.setdefault(plot, {})
        color = self.PROBE_COLORS[which]
        m = probes.get(which)
        if m:
            dpg.set_value(m["pt"], (x, y))
            dpg.configure_item(m["ann"], default_value=(x, y))
        else:
            pt = dpg.add_drag_point(parent=plot, default_value=(x, y),
                                    color=color, thickness=2,
                                    callback=self._on_probe_drag,
                                    user_data=(plot, which))
            ann = dpg.add_plot_annotation(parent=plot, label=which,
                                          default_value=(x, y),
                                          offset=(10, -10), color=color,
                                          clamped=True)
            probes[which] = m = {"pt": pt, "ann": ann}
        m.update(x=x, y=y, series=slabel, sid=sid)
        self._update_probe_readout(plot)

    def _on_probe_drag(self, sender, app_data, user_data):
        plot, which = user_data
        m = self._probes.get(plot, {}).get(which)
        if not m:
            return
        val = dpg.get_value(sender)
        x, y = float(val[0]), float(val[1])
        sid = m.get("sid")
        if sid and dpg.does_item_exist(sid):
            data = dpg.get_value(sid)
            if data and len(data) >= 2 and data[0]:
                xs, ys = data[0], data[1]
                i = min(range(len(xs)), key=lambda k: abs(xs[k] - x))
                x, y = float(xs[i]), float(ys[i])
                dpg.set_value(sender, (x, y))
        dpg.configure_item(m["ann"], default_value=(x, y))
        m["x"], m["y"] = x, y
        self._update_probe_readout(plot)

    def _hovered_probe_plot(self):
        return next((p for p in self.PROBE_AXES
                     if dpg.does_item_exist(p) and dpg.is_item_hovered(p)),
                    None)

    def _arm_probe(self, which):
        self._probe_armed = which
        self._set_status(
            f"● probe {which} armed - click on a plot to place (Esc cancels)",
            self.PROBE_COLORS[which][:3])

    def _disarm_probe(self):
        if self._probe_armed is None:
            return
        self._probe_armed = None
        self._set_status("● ready", C_GREEN)

    def _on_probe_key(self, sender, app_data, user_data):
        # don't react while the user is typing somewhere
        for t in ("chat_input", "custom_text", "va_editor"):
            if dpg.does_item_exist(t) and dpg.is_item_active(t):
                return
        if user_data == "ESC":
            self._disarm_probe()
            return
        if user_data == "C":
            plot = self._hovered_probe_plot()
            if plot:
                self._clear_probes((plot,))
            self._disarm_probe()
            return
        # A / B: arm the probe; it is placed on the next mouse click
        self._arm_probe(user_data)

    def _on_probe_click(self, sender, app_data):
        plot = self._hovered_probe_plot()
        if not self._probe_armed:
            # un-armed click on the STDP curve = drill into that timing point
            if plot == "stdp_plot":
                self._stdp_drilldown()
            return
        if not plot:
            return                       # stays armed until a plot is clicked
        try:
            mx, my = dpg.get_plot_mouse_pos()
        except Exception:
            return
        hit = self._nearest_point(plot, mx, my)
        if not hit:
            return
        x, y, slabel, sid = hit
        which = self._probe_armed
        self._set_probe(plot, which, x, y, slabel, sid)
        self._disarm_probe()

    def _clear_probes(self, plots):
        for plot in plots:
            probes = self._probes.pop(plot, None)
            if not probes:
                continue
            for m in probes.values():
                for k in ("pt", "ann", "line"):
                    item = m.get(k)
                    if item and dpg.does_item_exist(item):
                        dpg.delete_item(item)
            tag = self.PROBE_READOUT.get(plot)
            if tag and dpg.does_item_exist(tag):
                dpg.set_value(tag, "")

    def _update_ruler(self, plot):
        """Straight line through A and B with dX / dY / dy/dx annotation."""
        probes = self._probes.get(plot, {})
        a, b = probes.get("A"), probes.get("B")
        ruler = probes.get("ruler")
        if not (a and b):
            if ruler:
                for k in ("line", "ann"):
                    if dpg.does_item_exist(ruler[k]):
                        dpg.delete_item(ruler[k])
                probes.pop("ruler", None)
            return
        xs, ys = [a["x"], b["x"]], [a["y"], b["y"]]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        slope = "inf" if dx == 0 else f"{dy / dx:.5g}"
        label = f"dX={dx:.5g}\ndY={dy:.5g}\ndy/dx={slope}"
        mid = (0.5 * (a["x"] + b["x"]), 0.5 * (a["y"] + b["y"]))
        if ruler:
            dpg.set_value(ruler["line"], [xs, ys])
            dpg.configure_item(ruler["ann"], default_value=mid, label=label)
        else:
            yax = self.PROBE_AXES[plot][1]
            line = dpg.add_line_series(xs, ys, label="##probe_ruler",
                                       parent=yax)
            dpg.bind_item_theme(line, self.themes["ruler_series"])
            ann = dpg.add_plot_annotation(parent=plot, label=label,
                                          default_value=mid,
                                          offset=(14, 14), clamped=True,
                                          color=(38, 44, 60, 230))
            probes["ruler"] = {"line": line, "ann": ann}

    def _update_probe_readout(self, plot):
        self._update_ruler(plot)
        tag = self.PROBE_READOUT.get(plot)
        if not tag or not dpg.does_item_exist(tag):
            return
        probes = self._probes.get(plot, {})
        a, b = probes.get("A"), probes.get("B")
        parts = []
        for which, m in (("A", a), ("B", b)):
            if m:
                s = f" [{m['series']}]" if m.get("series") else ""
                parts.append(f"{which}: x={m['x']:.6g}  y={m['y']:.6g}{s}")
        if a and b:
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            slope = "inf" if dx == 0 else f"{dy / dx:.5g}"
            parts.append(f"dX={dx:.6g}  dY={dy:.6g}  dy/dx={slope}")
        name = self.PLOT_NAMES.get(plot, plot)
        dpg.set_value(tag, (f"[{name}]   " + "    |    ".join(parts))
                      if parts else "")

    def _on_zoom_btn(self, sender, app_data, user_data):
        axes, factor = user_data
        self._zoom_axes(axes, factor)

    def _on_fit_btn(self, sender, app_data, user_data):
        self._fit_axes_of(user_data)

    def _plot_toolbar(self, xaxes, yaxes, probe_tag=None):
        xaxes, yaxes = tuple(xaxes), tuple(yaxes)
        both = xaxes + yaxes
        with dpg.group(horizontal=True):
            dpg.add_button(label=" Zoom + ", small=True,
                           user_data=(both, 0.7),
                           callback=self._on_zoom_btn)
            dpg.add_button(label=" Zoom - ", small=True,
                           user_data=(both, 1.45),
                           callback=self._on_zoom_btn)
            dpg.add_button(label=" Fit ", small=True, user_data=both,
                           callback=self._on_fit_btn)
            dpg.add_button(label=" Fit X ", small=True, user_data=xaxes,
                           callback=self._on_fit_btn)
            dpg.add_button(label=" Fit Y ", small=True, user_data=yaxes,
                           callback=self._on_fit_btn)
            self._small("wheel zoom · drag pan · dbl-click fit · "
                        "A / B then click = place probe · C = clear")
        if probe_tag:
            self._small("", tag=probe_tag, color=(126, 170, 255))

    # =================================================================
    # UI construction
    # =================================================================

    def build(self):
        dpg.create_context()
        self._font()
        self._theme()

        with dpg.window(tag="main"):
            self._menu()
            self._toolbar()
            with dpg.group(horizontal=True):
                self._left_panel()
                self._center_panel()
                self._right_panel()

        with dpg.handler_registry():
            dpg.add_key_press_handler(key=dpg.mvKey_F5, callback=self.on_run)
            dpg.add_mouse_wheel_handler(callback=self._on_wheel)
            dpg.add_key_press_handler(key=dpg.mvKey_A,
                                      callback=self._on_probe_key,
                                      user_data="A")
            dpg.add_key_press_handler(key=dpg.mvKey_B,
                                      callback=self._on_probe_key,
                                      user_data="B")
            dpg.add_key_press_handler(key=dpg.mvKey_C,
                                      callback=self._on_probe_key,
                                      user_data="C")
            dpg.add_key_press_handler(key=dpg.mvKey_Escape,
                                      callback=self._on_probe_key,
                                      user_data="ESC")
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                        callback=self._on_probe_click)

        dpg.add_file_dialog(directory_selector=True, show=False, modal=True,
                            callback=self._on_dir_picked, tag="dir_dialog",
                            width=720, height=420, default_path=self.workdir)

        self._bind_plot_hover_handlers()

        with dpg.window(tag="virt_modal", label="Virtuoso connection",
                        modal=True, show=False, no_resize=True,
                        no_collapse=True, width=470):
            with dpg.group(horizontal=True):
                dpg.add_text("●", tag="virt_modal_dot", color=C_GREEN)
                t = dpg.add_text("", tag="virt_modal_title", color=C_TEXT)
                if "h2" in self.fonts:
                    dpg.bind_item_font(t, self.fonts["h2"])
            dpg.add_text("", tag="virt_modal_text", wrap=440, color=C_TEXT2)
            dpg.add_spacer(height=4)
            b = dpg.add_button(label="OK", width=90, callback=lambda:
                               dpg.configure_item("virt_modal", show=False))
            dpg.bind_item_theme(b, self.themes["primary"])

        with dpg.window(tag="account_modal", label="Claude account",
                        modal=True, show=False, no_resize=True,
                        no_collapse=True, autosize=True):
            t = dpg.add_text("Claude account", color=C_TEXT)
            if "h2" in self.fonts:
                dpg.bind_item_font(t, self.fonts["h2"])
            dpg.add_text("", tag="account_status_txt", wrap=440, color=C_TEXT2)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Refresh", callback=self.on_account_refresh)
                bl = dpg.add_button(label="Log in / switch (browser)",
                                    callback=self.on_account_login)
                dpg.bind_item_theme(bl, self.themes["primary"])
                dpg.add_button(label="Log out", callback=self.on_account_logout)
            dpg.add_separator()
            self._small("Run this app under a specific account (overrides the "
                        "login for THIS APP ONLY; your global Claude Code "
                        "login is untouched):", wrap=440)
            with dpg.group(horizontal=True):
                dpg.add_combo(["API key", "OAuth token"],
                              default_value="API key", tag="account_kind",
                              width=130)
                dpg.add_input_text(tag="account_override_in", password=True,
                                   width=304,
                                   hint="paste sk-ant-...  (or a setup-token)")
            with dpg.group(horizontal=True):
                ba = dpg.add_button(label="Apply", callback=self.on_account_apply)
                dpg.bind_item_theme(ba, self.themes["primary"])
                dpg.add_button(label="Clear override",
                               callback=self.on_account_clear)
            self._small("API key: Anthropic Console key (sk-ant-api...). "
                        "OAuth token: a 1-year token from `claude setup-token`. "
                        "Nothing is written to disk - it lives only in this "
                        "running app.", wrap=440)
            dpg.add_separator()
            dpg.add_button(label="Close", width=90, callback=lambda:
                           dpg.configure_item("account_modal", show=False))

        dpg.create_viewport(title=APP_TITLE, width=1600, height=980,
                            min_width=1160, min_height=720)
        dpg.set_viewport_resize_callback(self._on_resize)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)

        self.rescan_va(startup=True)
        self.rebuild_param_panel()
        self.rebuild_gen_params()
        self.log(f"workspace: {self.workdir}")
        self.log(f"agent backend: {self.agent.backend_label()}")
        self.append_chat("agent",
                         "Hi! I can generate spike patterns (Poisson, STDP, "
                         "bursts...), explain or modify the .va sources, and "
                         "help fit model parameters. Try a chip below, or "
                         "type /model to pick the model I run on.")
        self._on_resize()

    # ---------------- menu / toolbar --------------------------------

    def _menu(self):
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Change workspace...",
                                  callback=lambda: dpg.show_item("dir_dialog"))
                dpg.add_menu_item(label="Rescan Verilog-A files",
                                  callback=lambda: self.rescan_va())
                dpg.add_separator()
                dpg.add_menu_item(label="Export results to CSV",
                                  callback=self.export_csv)
                dpg.add_menu_item(label="Save plots to PNG (matplotlib)",
                                  callback=self.export_png)
                dpg.add_separator()
                dpg.add_menu_item(label="Exit", callback=dpg.stop_dearpygui)
            with dpg.menu(label="Run"):
                dpg.add_menu_item(label="Run simulation  (F5)",
                                  callback=self.on_run)
                dpg.add_menu_item(label="Fit plot axes", callback=self.fit_axes)
                dpg.add_separator()
                dpg.add_text("Analyze quantity:")
                dpg.add_menu_item(label="Conductance  G (uS)", check=True,
                                  default_value=True, tag="ana_menu_G",
                                  callback=lambda: self.on_analysis_metric("G"))
                dpg.add_menu_item(label="Resistance  R_mem (ohm)", check=True,
                                  default_value=False, tag="ana_menu_R",
                                  callback=lambda: self.on_analysis_metric("R"))
            with dpg.menu(label="Virtuoso", tag="menu_virtuoso"):
                dpg.add_menu_item(label="Connect (tunnel + skillbridge)",
                                  callback=self.on_virtuoso_connect)
                dpg.add_menu_item(label="Disconnect",
                                  callback=self.on_virtuoso_disconnect)
                dpg.add_menu_item(label="List libraries",
                                  callback=self.on_virtuoso_libs)
            with dpg.menu(label="Agent"):
                dpg.add_menu_item(label="Reset conversation",
                                  callback=self.on_agent_reset)
                dpg.add_menu_item(label="Backend info",
                                  callback=lambda: self.log(
                                      "agent: " + self.agent.backend_label()))
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="About", callback=self._about)

    def _toolbar(self):
        with dpg.group(horizontal=True):
            t = dpg.add_text("NeuroVAT", color=C_ACC)
            if "h2" in self.fonts:
                dpg.bind_item_font(t, self.fonts["h2"])
            self._small("Verilog-A tester")
            dpg.add_spacer(width=10)
            b = dpg.add_button(label="Run  (F5)", tag="btn_run",
                               callback=self.on_run)
            dpg.bind_item_theme(b, self.themes["primary"])
            with dpg.tooltip(b):
                dpg.add_text("Simulate the selected models with the "
                             "current stimulus")
            bs = dpg.add_button(label="Plot STDP", tag="btn_stdp",
                                callback=self.on_plot_stdp)
            with dpg.tooltip(bs):
                dpg.add_text("Sweep the pre/post spike timing dt and plot "
                             "the synaptic STDP curve dG vs dt.\nUses the "
                             "'STDP pair' generator's amplitudes/width and "
                             "the post-stimulus tail as settle time.")
            dpg.add_button(label="Fit plots", callback=self.fit_axes)
            dpg.add_button(label="Export CSV", callback=self.export_csv)
            dpg.add_button(label="Rescan .va",
                           callback=lambda: self.rescan_va())
            dpg.add_spacer(width=12)
            dpg.add_loading_indicator(tag="busy_ind", show=False, radius=2.0,
                                      style=1, color=C_AMBER)
            # flat (transparent) button so it baseline-aligns with the buttons
            dpg.add_button(label="● ready", tag="run_status")
            self._set_status("● ready", C_GREEN)
        dpg.add_separator()

    def _set_status(self, text, color):
        if not dpg.does_item_exist("run_status"):
            return
        dpg.configure_item("run_status", label=text)
        th = self._status_themes.get(color)
        if th is None:
            with dpg.theme() as th:
                with dpg.theme_component(dpg.mvAll):
                    for c in ("mvThemeCol_Button", "mvThemeCol_ButtonHovered",
                              "mvThemeCol_ButtonActive"):
                        dpg.add_theme_color(getattr(dpg, c), (0, 0, 0, 0))
                    dpg.add_theme_color(dpg.mvThemeCol_Text, color)
            self._status_themes[color] = th
        dpg.bind_item_theme("run_status", th)

    # ---------------- left panel ------------------------------------

    def _left_panel(self):
        with dpg.child_window(width=LEFT_W, tag="left_child", border=False):
            dpg.add_spacer(height=2)

            with dpg.collapsing_header(label="Verilog-A files",
                                       default_open=True) as h:
                if "bold" in self.fonts:
                    dpg.bind_item_font(h, self.fonts["bold"])
                with self._pad():
                    with dpg.child_window(tag="va_cards", auto_resize_y=True,
                                          border=True) as vc:
                        pass
                    dpg.bind_item_theme(vc, self.themes["card"])
                    with dpg.tooltip(vc):
                        dpg.add_text("check = simulate   ·   click = load "
                                     "params   ·   double-click = edit source")

            with dpg.collapsing_header(label="Parameters",
                                       default_open=True) as h:
                if "bold" in self.fonts:
                    dpg.bind_item_font(h, self.fonts["bold"])
                with self._pad():
                    dpg.add_combo([s.label for s in MODEL_SPECS],
                                  default_value=MODEL_SPECS[1].label,
                                  tag="param_model_sel", width=-1,
                                  callback=lambda *_: self.rebuild_param_panel())
                    with dpg.child_window(tag="param_panel", height=250,
                                          border=False):
                        pass
                    dpg.add_button(label="Reset defaults",
                                   callback=self.on_param_defaults)

            with dpg.collapsing_header(label="Simulation",
                                       default_open=True) as h:
                if "bold" in self.fonts:
                    dpg.bind_item_font(h, self.fonts["bold"])
                with self._pad():
                    with dpg.table(header_row=False,
                                   policy=dpg.mvTable_SizingStretchProp) as tbl:
                        dpg.bind_item_theme(tbl, self.themes["table_roomy"])
                        dpg.add_table_column(init_width_or_weight=0.62)
                        dpg.add_table_column(init_width_or_weight=0.38)
                        with dpg.table_row():
                            dpg.add_text("post-stimulus tail (s)",
                                         color=C_TEXT2)
                            dpg.add_input_double(tag="tail_input",
                                                 default_value=1.0,
                                                 width=-1, format="%.4g",
                                                 step=0)
                        with dpg.table_row():
                            dpg.add_checkbox(label="manual t_stop (s)",
                                             tag="tstop_manual")
                            dpg.add_input_double(tag="tstop_input",
                                                 default_value=2.0,
                                                 width=-1, format="%.6g",
                                                 step=0)
                        with dpg.table_row():
                            dpg.add_checkbox(label="auto-save CSV after run",
                                             tag="csv_cb")
                            dpg.add_text("")

            with dpg.collapsing_header(label="Cadence Virtuoso",
                                       default_open=False) as h:
                if "bold" in self.fonts:
                    dpg.bind_item_font(h, self.fonts["bold"])
                with self._pad():
                    with dpg.group(horizontal=True):
                        dpg.add_text("●", tag="virt_dot", color=C_RED)
                        dpg.add_text("not connected", tag="virt_status",
                                     color=C_TEXT2)
                    with dpg.group(horizontal=True):
                        b = dpg.add_button(label="Connect",
                                           tag="btn_virt_connect",
                                           callback=self.on_virtuoso_connect)
                        dpg.bind_item_theme(b, self.themes["primary"])
                        with dpg.tooltip(b):
                            dpg.add_text("Start the SSH tunnel to "
                                         "coen-cassia and open a skillbridge "
                                         "workspace.\nLinux side must have "
                                         "Virtuoso open with the skill server "
                                         "running.")
                        dpg.add_button(label="Disconnect",
                                       callback=self.on_virtuoso_disconnect)
                    self._small("", tag="virt_info", wrap=LEFT_W - 36)

    # ---------------- center panel ----------------------------------

    def _center_panel(self):
        with dpg.child_window(tag="center_child", width=620, border=False):
            with dpg.tab_bar(tag="center_tabs"):
                with dpg.tab(label="  Signal Designer  ", tag="tab_designer"):
                    self._designer_tab()
                with dpg.tab(label="  Results  ", tag="tab_results"):
                    self._results_tab()
                with dpg.tab(label="  Analysis  ", tag="tab_analysis"):
                    self._analysis_tab()
                with dpg.tab(label="  STDP  ", tag="tab_stdp"):
                    self._stdp_tab()
                with dpg.tab(label="  Verilog-A Source  ", tag="tab_source"):
                    self._source_tab()
                with dpg.tab(label="  Log  ", tag="tab_log"):
                    with self._pad(left=8, top=8, bottom=0):
                        with dpg.child_window(tag="console", border=False):
                            pass

    def _designer_tab(self):
        with self._pad(left=8, top=8, bottom=0):
            with dpg.group(horizontal=True):
                with dpg.group():
                    self._caption("GENERATOR")
                    dpg.add_combo([g.name for g in sf.GENERATORS],
                                  default_value=sf.GENERATORS[2].name,
                                  tag="gen_combo", width=230,
                                  callback=lambda *_: self.rebuild_gen_params())
                with dpg.group():
                    self._caption("SIGNAL")
                    dpg.add_combo(["current", "voltage"],
                                  default_value="current",
                                  tag="kind_combo", width=110,
                                  callback=lambda *_: self._sync_unit_combo())
                with dpg.group():
                    self._caption("UNIT")
                    dpg.add_combo(sf.CURRENT_UNITS, default_value="pA",
                                  tag="unit_combo", width=86)
                with dpg.group():
                    self._caption("DELAY (ms)")
                    dpg.add_input_double(tag="delay_ms", default_value=0.0,
                                         width=80, format="%.4g", step=0)
                with dpg.group():
                    dpg.add_spacer(height=26)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Preview",
                                       callback=self.on_preview)
                        b = dpg.add_button(label="Run", callback=self.on_run)
                        dpg.bind_item_theme(b, self.themes["primary"])
            self._small("", tag="gen_desc", color=C_MUTED, wrap=940)
            with dpg.child_window(tag="gen_params", height=176, border=False):
                pass
            self._small("custom rows:  t_start_ms   width_ms   amplitude"
                        "     (# comment)")
            dpg.add_input_text(tag="custom_text", multiline=True, width=-1,
                               height=86, tab_input=True,
                               default_value="10 10 -100\n60 10 -100\n"
                                             "200 20 150\n")
            if "mono" in self.fonts:
                dpg.bind_item_font("custom_text", self.fonts["mono"])
            dpg.add_text("", tag="designer_summary", color=C_GREEN)
            self._plot_toolbar(("prev_x",), ("prev_y",),
                               probe_tag="probe_prev")
            with dpg.plot(height=-1, width=-1, tag="preview_plot"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="time (s)", tag="prev_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="amplitude", tag="prev_y")

    def _results_tab(self):
        with self._pad(left=8, top=8, bottom=0):
            self._plot_toolbar(("ax_i_x", "ax_r_x", "ax_g_x"),
                               ("ax_i_y", "ax_r_y", "ax_g_y"),
                               probe_tag="probe_results")
            with dpg.subplots(3, 1, link_all_x=True, width=-1, height=-1,
                              row_ratios=[0.62, 1.0, 1.0]):
                with dpg.plot(tag="plot_i"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, tag="ax_i_x",
                                      no_tick_labels=True)
                    dpg.add_plot_axis(dpg.mvYAxis, label="stimulus",
                                      tag="ax_i_y")
                with dpg.plot(tag="plot_r"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, tag="ax_r_x",
                                      no_tick_labels=True)
                    dpg.add_plot_axis(dpg.mvYAxis, label="R_mem (ohm)",
                                      tag="ax_r_y")
                with dpg.plot(tag="plot_g"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="time (s)",
                                      tag="ax_g_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="G (uS)",
                                      tag="ax_g_y")

    def _analysis_tab(self):
        with self._pad(left=8, top=8, bottom=0):
            with dpg.group(horizontal=True):
                self._caption("ANALYZE")
                dpg.add_combo(["Conductance G", "Resistance R_mem"],
                              default_value="Conductance G",
                              tag="ana_metric_combo", width=180,
                              callback=lambda s, a: self.on_analysis_metric(
                                  "R" if a == "Resistance R_mem" else "G"))
                self._small("retained value sampled after each pulse",
                            tag="ana_caption")
            dpg.add_text("", tag="ana_text", wrap=940, color=C_TEXT2)
            self._plot_toolbar(("ana_x",), ("ana_y",), probe_tag="probe_ana")
            with dpg.plot(height=-1, width=-1, tag="ana_plot"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="pulse #", tag="ana_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="G (uS)", tag="ana_y")

    def _stdp_tab(self):
        with self._pad(left=8, top=8, bottom=0):
            self._small("spike-timing-dependent plasticity: one pre/post "
                        "spike pair per point, dG = retained G change after "
                        "settling. dt > 0 means post follows pre. |dt| is "
                        "swept from the pulse width (no overlap) to +-1000 ms.")
            with dpg.group(horizontal=True):
                for tag, cap, default, w in (
                        ("stdp_amp_pre", "PRE AMP", -100.0, 84),
                        ("stdp_amp_post", "POST AMP", 100.0, 84),
                        ("stdp_width_ms", "WIDTH (ms)", 5.0, 84),
                        ("stdp_range_ms", "DT RANGE (+-ms)", 1000.0, 96),
                        ("stdp_settle_ms", "SETTLE (ms)", 1000.0, 90)):
                    with dpg.group():
                        self._caption(cap)
                        dpg.add_input_double(tag=tag, default_value=default,
                                             width=w, format="%.4g", step=0)
                with dpg.group():
                    self._caption("AMP UNIT")
                    self._small("", tag="stdp_unit_lbl", color=C_TEXT2)
                with dpg.group():
                    dpg.add_spacer(height=18)
                    b = dpg.add_button(label="Plot STDP",
                                       callback=self.on_plot_stdp)
                    dpg.bind_item_theme(b, self.themes["primary"])
            with dpg.tooltip("stdp_amp_pre"):
                dpg.add_text("PRE/POST AMP: the two spike amplitudes (in the "
                             "Signal Designer's unit + current/voltage mode).\n"
                             "WIDTH: pulse width; the sweep skips any |dt| "
                             "smaller than it.\n"
                             "DT RANGE: sweep dt from -range..+range ms "
                             "(e.g. 200 = -200..+200 ms, 1000 = -1..+1 s).\n"
                             "SETTLE: wait time after the pair before reading "
                             "dG, so the volatile component has relaxed and "
                             "you measure the RETAINED change, not the "
                             "transient overshoot.")
            self._plot_toolbar(("stdp_x",), ("stdp_y",),
                               probe_tag="probe_stdp")
            with dpg.plot(height=-1, width=-1, tag="stdp_plot"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis,
                                  label="dt = t_post - t_pre (ms)",
                                  tag="stdp_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="dG (uS)", tag="stdp_y")

    def _source_tab(self):
        with self._pad(left=8, top=8, bottom=0):
            with dpg.group(horizontal=True):
                dpg.add_combo([], tag="va_edit_sel", width=300,
                              callback=self.on_editor_file_change)
                dpg.add_button(label="Save", callback=self.on_editor_save)
                dpg.add_button(label="Reload", callback=self.on_editor_reload)
                dpg.add_button(label="Ask agent about this file",
                               callback=self.on_ask_about_file)
                self._small("", tag="editor_status")

            # --- open source straight from Cadence Virtuoso -----------
            with dpg.child_window(auto_resize_y=True, border=True) as vbar:
                with dpg.group(horizontal=True):
                    self._small("FROM VIRTUOSO", color=(126, 150, 220))
                    dpg.add_checkbox(label="Verilog views only",
                                     tag="virt_va_only", default_value=True,
                                     callback=lambda *_: self._virt_lib_changed())
                with dpg.group(horizontal=True):
                    self._caption("library")
                    dpg.add_combo([], tag="virt_lib_combo", width=190,
                                  callback=lambda *_: self._virt_lib_changed())
                    self._caption("cell")
                    dpg.add_combo([], tag="virt_cell_combo", width=170,
                                  callback=lambda *_: self._virt_cell_changed())
                    self._caption("view")
                    dpg.add_combo([], tag="virt_view_combo", width=130)
                    b = dpg.add_button(label="↓ Load source",
                                       callback=self.on_virt_load_source)
                    dpg.bind_item_theme(b, self.themes["primary"])
                    dpg.add_button(label="↻", tag="virt_refresh_libs",
                                   callback=self.on_virt_refresh_libs)
                    with dpg.tooltip("virt_refresh_libs"):
                        dpg.add_text("Refresh library list from Virtuoso")
                self._small("connect to Virtuoso first (left panel)",
                            tag="virt_browse_status")
            dpg.bind_item_theme(vbar, self.themes["card"])

            dpg.add_input_text(tag="va_editor", multiline=True, width=-1,
                               height=-1, tab_input=True)
            if "mono" in self.fonts:
                dpg.bind_item_font("va_editor", self.fonts["mono"])

    # ---------------- right panel: agent chat ------------------------

    def _right_panel(self):
        with dpg.child_window(width=RIGHT_W, tag="right_child", border=False):
            with self._pad(left=8, top=6, bottom=2):
                with dpg.group(horizontal=True):
                    online = self.agent.backend != "none"
                    dot = dpg.add_text("●", tag="agent_dot",
                                       color=C_GREEN if online else C_RED)
                    with dpg.tooltip(dot):
                        dpg.add_text("backend: " + self.agent.backend_label())
                    h = dpg.add_text("Claude Agent", color=C_TEXT)
                    if "h2" in self.fonts:
                        dpg.bind_item_font(h, self.fonts["h2"])
                    # lower the chips so they baseline-align with the heading
                    with dpg.group():
                        dpg.add_spacer(height=5)
                        with dpg.group(horizontal=True):
                            badge = dpg.add_button(label=self.agent_model_label,
                                                   tag="model_badge", small=True)
                            dpg.bind_item_theme(badge, self.themes["chip"])
                            if "small" in self.fonts:
                                dpg.bind_item_font(badge, self.fonts["small"])
                            with dpg.tooltip(badge):
                                dpg.add_text("Model used for agent replies - "
                                             "click to switch, or type /model "
                                             "in the chat")
                            with dpg.popup(badge,
                                           mousebutton=dpg.mvMouseButton_Left):
                                self._small("MODEL", color=(126, 150, 220))
                                for label, mid, desc in MODEL_CHOICES:
                                    dpg.add_selectable(
                                        label=f"{label}   ·  {desc}",
                                        user_data=(label, mid),
                                        callback=self._on_model_pick)
                            acc = dpg.add_button(label="Account", small=True,
                                                 callback=self.on_account_open)
                            dpg.bind_item_theme(acc, self.themes["chip"])
                            if "small" in self.fonts:
                                dpg.bind_item_font(acc, self.fonts["small"])
                            with dpg.tooltip(acc):
                                dpg.add_text("Sign in / switch Claude account, "
                                             "or run this app under a specific "
                                             "key/token")
                with dpg.child_window(tag="chat_log", height=-162,
                                      border=False):
                    pass
                with dpg.group(tag="chips_row"):
                    for row in (QUICK_CHIPS[:3], QUICK_CHIPS[3:]):
                        with dpg.group(horizontal=True):
                            for label, prompt in row:
                                b = dpg.add_button(
                                    label=label, user_data=prompt, small=True,
                                    callback=lambda s, a, u: (
                                        dpg.set_value("chat_input", u),
                                        dpg.focus_item("chat_input")))
                                dpg.bind_item_theme(b, self.themes["chip"])
                                if "small" in self.fonts:
                                    dpg.bind_item_font(b, self.fonts["small"])
                with dpg.group(horizontal=True):
                    dpg.add_input_text(tag="chat_input", width=-78,
                                       hint="Message Claude...  (/model, /help)",
                                       on_enter=True, callback=self.on_send)
                    b = dpg.add_button(label="Send", tag="btn_send",
                                       callback=self.on_send)
                    dpg.bind_item_theme(b, self.themes["primary"])
                    bstop = dpg.add_button(label="Stop", tag="btn_stop",
                                           show=False,
                                           callback=self.on_agent_stop)
                    dpg.bind_item_theme(bstop, self.themes["stop"])
                with dpg.group(horizontal=True):
                    cb1 = dpg.add_checkbox(label="file edits", tag="cb_edits")
                    with dpg.tooltip(cb1):
                        dpg.add_text("Allow the agent to modify files in the "
                                     "workspace (.va sources, twins, scripts)")
                    cb2 = dpg.add_checkbox(label="shell", tag="cb_bash")
                    with dpg.tooltip(cb2):
                        dpg.add_text("Allow the agent to run shell commands "
                                     "(simulations, Virtuoso scripts, git)")
                    cb3 = dpg.add_checkbox(label="auto-fix", tag="cb_auto")
                    with dpg.tooltip(cb3):
                        dpg.add_text("Autonomous mode: grants file edits + "
                                     "shell and lets the agent run the full "
                                     "edit -> simulate -> verify -> fix loop\n"
                                     "until the behavior is right (no per-step "
                                     "approval). Up to ~30 min per request.")
                    dpg.add_button(label="Reset", small=True,
                                   callback=self.on_agent_reset)
                with dpg.group(horizontal=True):
                    cb4 = dpg.add_checkbox(label="live plot", tag="cb_liveplot",
                                           default_value=True)
                    with dpg.tooltip(cb4):
                        dpg.add_text("Auto re-simulate and re-plot whenever the "
                                     "agent saves an edit to a model twin, so "
                                     "you see changes in real time.")
                    rb = dpg.add_button(label="Revert agent edits", small=True,
                                        callback=self.on_agent_revert)
                    with dpg.tooltip(rb):
                        dpg.add_text("Restore the model twins + .va files to "
                                     "the snapshot taken before the last agent "
                                     "turn.")
                    self._small("", tag="cost_label")

    # =================================================================
    # chat rendering
    # =================================================================

    def append_chat(self, who, text):
        """who: 'you' | 'agent' | 'err' | 'sys'"""
        indent = BUBBLE_INDENT if who == "you" else 0
        theme = {"you": "bub_user", "agent": "bub_agent",
                 "err": "bub_err", "sys": "bub_agent"}[who]
        role = {"you": ("You", C_ACC_H), "agent": ("Claude", C_AGENT),
                "err": ("Error", C_RED), "sys": ("System", C_TEXT2)}[who]
        with dpg.group(horizontal=True, parent="chat_log"):
            if indent:
                dpg.add_spacer(width=indent)
            with dpg.child_window(width=BUBBLE_W - indent, auto_resize_y=True,
                                  border=True) as bub:
                with dpg.group(horizontal=True):
                    self._small(role[0], color=role[1])
                    self._small(time.strftime("%H:%M"), color=C_MUTED)
                dpg.add_text(text, wrap=BUBBLE_W - indent - 28, color=C_TEXT)
            dpg.bind_item_theme(bub, self.themes[theme])
        dpg.add_spacer(height=5, parent="chat_log")
        self._scroll_bottom("chat_log")

    def _add_pattern_card(self, wf):
        with dpg.group(parent="chat_log"):
            with dpg.child_window(width=BUBBLE_W, auto_resize_y=True,
                                  border=True) as card:
                self._small("WAVEFORM PATTERN", color=(126, 150, 220))
                dpg.add_text(wf["label"], color=C_TEXT,
                             wrap=BUBBLE_W - 28)
                t_end = max(p[0] + p[1] for p in wf["pulses"])
                self._small(f"{len(wf['pulses'])} pulses · {wf['kind']} "
                            f"· {wf['unit']} · ends {t_end:.4g} s",
                            color=C_TEXT2)
                b = dpg.add_button(label="Load into designer",
                                   user_data=wf,
                                   callback=self.on_load_agent_pattern)
                dpg.bind_item_theme(b, self.themes["primary"])
                if "small" in self.fonts:
                    dpg.bind_item_font(b, self.fonts["small"])
            dpg.bind_item_theme(card, self.themes["pattern_card"])
        dpg.add_spacer(height=5, parent="chat_log")
        self._scroll_bottom("chat_log")

    # ---------------- slash commands & model picker -------------------

    def _match_model(self, query):
        q = query.lower().strip()
        q = q.replace("claude-", "")
        for label, mid, _ in MODEL_CHOICES:
            hay = label.lower() + " " + (mid or "")
            if q and q in hay:
                return label, mid
        return None

    def _set_model(self, label, mid, announce=True):
        self.agent_model_label = label
        self.agent_model_id = mid
        if dpg.does_item_exist("model_badge"):
            dpg.configure_item("model_badge", label=label)
        if announce:
            self.append_chat("sys", f"Model set to {label}"
                             + (f"  ({mid})" if mid else " (CLI default)"))
        self.log(f"[agent] model -> {label}" + (f" ({mid})" if mid else ""))

    def _on_model_pick(self, sender, app_data, user_data):
        label, mid = user_data
        self._set_model(label, mid)

    def _add_model_picker(self):
        with dpg.group(parent="chat_log"):
            with dpg.child_window(width=BUBBLE_W, auto_resize_y=True,
                                  border=True) as card:
                self._small("SELECT MODEL", color=(126, 150, 220))
                for label, mid, desc in MODEL_CHOICES:
                    current = "  ●" if label == self.agent_model_label else ""
                    b = dpg.add_button(label=f"{label}  ·  {desc}{current}",
                                       width=-1, user_data=(label, mid),
                                       callback=self._on_model_pick)
                    dpg.bind_item_theme(b, self.themes["chip"])
                    if "small" in self.fonts:
                        dpg.bind_item_font(b, self.fonts["small"])
            dpg.bind_item_theme(card, self.themes["pattern_card"])
        dpg.add_spacer(height=5, parent="chat_log")
        self._scroll_bottom("chat_log")

    def _handle_command(self, text):
        parts = text.lstrip("/").split()
        cmd = parts[0].lower() if parts else ""
        if cmd == "model":
            if len(parts) == 1:
                self._add_model_picker()
            else:
                hit = self._match_model(" ".join(parts[1:]))
                if hit:
                    self._set_model(*hit)
                else:
                    names = " | ".join(l.split()[0].lower()
                                       for l, _, _ in MODEL_CHOICES)
                    self.append_chat("err",
                                     f"No model matches "
                                     f"'{' '.join(parts[1:])}'. "
                                     f"Try /model {names}")
        elif cmd in ("clear", "reset"):
            self.on_agent_reset()
        elif cmd == "help":
            self.append_chat("sys",
                             "Commands:\n"
                             "/model            choose the model\n"
                             "/model <name>     e.g. /model opus, /model haiku\n"
                             "/clear            reset the conversation\n"
                             "/help             this list")
        else:
            self.append_chat("err", f"Unknown command /{cmd} - try /help")

    def _chat_pending(self, show):
        if dpg.does_item_exist("pending_bubble"):
            dpg.delete_item("pending_bubble")
        if show:
            with dpg.group(parent="chat_log", tag="pending_bubble",
                           horizontal=True):
                with dpg.child_window(width=150, auto_resize_y=True,
                                      border=True) as bub:
                    with dpg.group(horizontal=True):
                        dpg.add_loading_indicator(radius=1.5, style=1,
                                                  color=C_AGENT)
                        self._small("Claude is thinking", color=C_TEXT2)
                dpg.bind_item_theme(bub, self.themes["bub_agent"])
            self._scroll_bottom("chat_log")

    def _scroll_bottom(self, tag):
        frame = dpg.get_frame_count() + 2
        try:
            dpg.set_frame_callback(
                frame, lambda: dpg.set_y_scroll(tag, dpg.get_y_scroll_max(tag)))
        except SystemError:
            pass

    # =================================================================
    # dynamic panels
    # =================================================================

    def rebuild_param_panel(self):
        spec = self._param_spec()
        dpg.delete_item("param_panel", children_only=True)
        vals = self.param_values[spec.key]
        with dpg.table(parent="param_panel", header_row=False,
                       policy=dpg.mvTable_SizingStretchProp) as tbl:
            dpg.bind_item_theme(tbl, self.themes["table_roomy"])
            dpg.add_table_column(init_width_or_weight=0.42)
            dpg.add_table_column(init_width_or_weight=0.58)
            for name, val in vals.items():
                with dpg.table_row():
                    dpg.add_text(name, color=C_TEXT2)
                    is_int = isinstance(val, int) and not isinstance(val, bool)
                    tag = f"p_{spec.key}_{name}"
                    if is_int:
                        dpg.add_input_int(tag=tag, default_value=int(val),
                                          width=-1, step=0,
                                          callback=self._on_param_edit,
                                          user_data=(spec.key, name, True))
                    else:
                        dpg.add_input_double(tag=tag, default_value=float(val),
                                             width=-1, format="%.6g", step=0,
                                             callback=self._on_param_edit,
                                             user_data=(spec.key, name, False))

    def _on_param_edit(self, sender, value, user_data):
        key, name, is_int = user_data
        self.param_values[key][name] = int(value) if is_int else float(value)

    def _param_spec(self):
        label = dpg.get_value("param_model_sel")
        for s in MODEL_SPECS:
            if s.label == label:
                return s
        return MODEL_SPECS[0]

    def on_param_defaults(self):
        spec = self._param_spec()
        self.param_values[spec.key] = _defaults_of(spec.params_cls)
        self.rebuild_param_panel()
        self.log(f"[params] {spec.key} reset to defaults")

    def rebuild_gen_params(self):
        gen = GEN_BY_NAME[dpg.get_value("gen_combo")]
        dpg.set_value("gen_desc", gen.desc)
        dpg.delete_item("gen_params", children_only=True)
        saved = self.gen_values.setdefault(gen.name, {})
        if not gen.params:
            return
        with dpg.table(parent="gen_params", header_row=False, width=640,
                       policy=dpg.mvTable_SizingStretchProp) as tbl:
            dpg.bind_item_theme(tbl, self.themes["table_roomy"])
            dpg.add_table_column(init_width_or_weight=0.30)
            dpg.add_table_column(init_width_or_weight=0.22)
            dpg.add_table_column(init_width_or_weight=0.30)
            dpg.add_table_column(init_width_or_weight=0.22)
            items = []
            for p in gen.params:
                items.append(p)
            for i in range(0, len(items), 2):
                with dpg.table_row():
                    for p in items[i:i + 2]:
                        val = saved.get(p.key, p.default)
                        dpg.add_text(p.label, color=C_TEXT2)
                        tag = f"g_{p.key}"
                        if p.is_int:
                            dpg.add_input_int(tag=tag, default_value=int(val),
                                              width=150, step=0,
                                              callback=self._on_gen_edit,
                                              user_data=(gen.name, p.key))
                        else:
                            dpg.add_input_double(tag=tag,
                                                 default_value=float(val),
                                                 width=150, format="%.6g",
                                                 step=0,
                                                 callback=self._on_gen_edit,
                                                 user_data=(gen.name, p.key))
                    if len(items[i:i + 2]) == 1:
                        dpg.add_text("")
                        dpg.add_text("")

    def _on_gen_edit(self, sender, value, user_data):
        gname, key = user_data
        self.gen_values[gname][key] = value

    def _sync_unit_combo(self):
        kind = dpg.get_value("kind_combo")
        units = sf.CURRENT_UNITS if kind == "current" else sf.VOLTAGE_UNITS
        dpg.configure_item("unit_combo", items=units)
        if dpg.get_value("unit_combo") not in units:
            dpg.set_value("unit_combo", units[0])

    # =================================================================
    # waveform building / preview
    # =================================================================

    def build_waveform(self):
        gen = GEN_BY_NAME[dpg.get_value("gen_combo")]
        unit = dpg.get_value("unit_combo")
        kind = dpg.get_value("kind_combo")
        scale = sf.UNIT_SCALE.get(unit, 1.0)

        if gen.build is None:
            pulses_u, errors = sf.parse_custom(dpg.get_value("custom_text"))
            if errors:
                raise ValueError("custom pattern: " + "; ".join(errors[:4]))
            meta = {}
        else:
            vals = {p.key: self.gen_values.get(gen.name, {}).get(p.key,
                                                                 p.default)
                    for p in gen.params}
            pulses_u, meta = gen.build(vals)
        if not pulses_u:
            raise ValueError("waveform has no pulses")
        delay = max(0.0, dpg.get_value("delay_ms") or 0.0) * 1e-3
        pulses = [(t0 + delay, w, a * scale) for t0, w, a in pulses_u]
        return Waveform(pulses), meta, unit, kind, gen.name

    def on_preview(self):
        try:
            wf, meta, unit, kind, label = self.build_waveform()
        except ValueError as e:
            self.log(f"[designer] {e}")
            dpg.set_value("designer_summary", str(e))
            return
        self._clear_probes(("preview_plot",))
        xs, ys = self._stair_points(wf, 1.0 / sf.UNIT_SCALE.get(unit, 1.0))
        dpg.delete_item("prev_y", children_only=True)
        dpg.add_stair_series(xs, ys, parent="prev_y",
                             label=f"{label} ({unit})")
        dpg.configure_item("prev_y", label=f"amplitude ({unit})")
        dpg.fit_axis_data("prev_x")
        dpg.fit_axis_data("prev_y")
        t_end = wf.breakpoints[-1] if wf.breakpoints else 0
        dpg.set_value("designer_summary",
                      f"{len(wf.pulses)} pulses · last edge {t_end:.4g} s")

    @staticmethod
    def _stair_points(wf, yscale, tail=None):
        xs, ys = [0.0], [0.0]
        for e, v in zip(wf.edges, wf.values):
            xs.append(e)
            ys.append(v * yscale)
        if tail:
            xs.append(xs[-1] + tail)
            ys.append(ys[-1])
        return xs, ys

    # =================================================================
    # simulation
    # =================================================================

    def _checked_models(self):
        models = []
        for key in self._enabled_keys():
            s = SPEC_BY_KEY[key]
            try:
                models.append(s.cls(s.params_cls(**self.param_values[key])))
            except Exception as e:
                self.log(f"[params] {key}: {e}")
        return models

    def on_run(self, *_, live=False):
        if self.sim_running:
            return
        try:
            wf, meta, unit, kind, label = self.build_waveform()
        except ValueError as e:
            if not live:
                self.log(f"[run] {e}")
            return
        models = self._checked_models()
        if not models:
            if not live:
                self.log("[run] no model selected")
            return
        if dpg.get_value("tstop_manual"):
            t_stop = max(dpg.get_value("tstop_input"), 1e-3)
        else:
            t_stop = (wf.breakpoints[-1] if wf.breakpoints else 0.0) \
                + max(dpg.get_value("tail_input"), 0.0)
        self.sim_running = True
        if live:
            self._busy(True, "live re-plot (agent edited the code)")
        else:
            self._busy(True, f"simulating  {label}  (t_stop {t_stop:.4g} s)")
            self.log(f"[run] {label}: {len(wf.pulses)} pulses, unit {unit}, "
                     f"kind {kind}, t_stop {t_stop:.4g} s, "
                     f"models: {', '.join(m.name for m in models)}")
        autosave = dpg.get_value("csv_cb") and not live
        threading.Thread(
            target=self._sim_worker,
            args=(models, wf, t_stop, meta, unit, kind, autosave, live),
            daemon=True).start()

    def _sim_worker(self, models, wf, t_stop, meta, unit, kind, autosave,
                    live=False):
        results = []
        t0 = time.perf_counter()
        for m in models:
            mk = getattr(m, "input_kind", "current")
            if mk != kind and not live:
                self.q.put(("log", f"  [warn] {m.name} expects {mk} input; "
                                   f"amplitudes are treated as {mk}"))
            try:
                r = simulate(m, wf, t_stop, label=m.name)
            except Exception as e:
                self.q.put(("log", f"  [sim] {m.name} FAILED: {e}"))
                continue
            results.append(r)
            if not live:
                self.q.put(("log", "  " + r.summary()))
        if not live:
            self.q.put(("log", f"[run] done in "
                               f"{time.perf_counter() - t0:.2f} s"))
        if autosave and results:
            outdir = os.path.join(self.workdir, "results")
            os.makedirs(outdir, exist_ok=True)
            for r in results:
                stem = re.sub(r"[^\w]+", "_", r.label).strip("_")
                p = r.save_csv(os.path.join(outdir, f"gui_{stem}.csv"))
                self.q.put(("log", f"  csv -> {p}"))
        self.q.put(("results", results, meta, unit, kind, live))

    # ---------------- STDP characterization --------------------------

    def on_plot_stdp(self, *_):
        if self.sim_running:
            return
        models = self._checked_models()
        if not models:
            self.log("[stdp] no model selected")
            return
        unit = dpg.get_value("unit_combo")
        kind = dpg.get_value("kind_combo")
        scale = sf.UNIT_SCALE.get(unit, 1.0)
        if dpg.does_item_exist("stdp_unit_lbl"):
            dpg.set_value("stdp_unit_lbl", f"{unit} ({kind})")
        amp_pre = dpg.get_value("stdp_amp_pre") * scale
        amp_post = dpg.get_value("stdp_amp_post") * scale
        width = max(dpg.get_value("stdp_width_ms"), 1e-3) * 1e-3
        tail = max(dpg.get_value("stdp_settle_ms") * 1e-3, 0.02)
        # sweep |dt| from just past the pulse width (a 1% gap so the two
        # pulses never touch) out to the user range: dense near the start,
        # coarse to the edges.
        dt_max = abs(dpg.get_value("stdp_range_ms")) * 1e-3
        dt_min = width * 1.01                      # e.g. width 10 ms -> 10.1 ms
        if dt_min >= dt_max:
            self.sim_running = False
            self._busy(False, "STDP: width >= dt range")
            self.log(f"[stdp] pulse width {width * 1e3:g} ms is too large for "
                     f"the dt range ({dt_max * 1e3:g} ms); widen DT RANGE or "
                     f"reduce WIDTH")
            return
        fine = min(9.0 * width, dt_max)           # dense region near the start
        inner = [fine * k / 24 for k in range(1, 25)]
        outer = [dt_max * k / 20 for k in range(1, 21)]
        cand = [dt_min] + inner + outer           # sample the gap boundary
        pos = sorted({round(d, 9) for d in cand
                      if d >= dt_min - 1e-12})      # |dt| > pulse width
        dts = [-d for d in reversed(pos)] + pos    # mirror; no overlap region
        self.log(f"[stdp] |dt| starts at {dt_min * 1e3:g} ms "
                 f"(> pulse width {width * 1e3:g} ms): no pulse overlap")
        self._stdp_ctx = {
            "models": models, "amp_pre": amp_pre, "amp_post": amp_post,
            "width": width, "tail": tail, "unit": unit, "kind": kind,
            "dts": dts,
        }
        for m in models:
            mk = getattr(m, "input_kind", "current")
            if mk != kind:
                self.log(f"  [warn] {m.name} expects {mk} input; "
                         f"amplitudes are treated as {mk}")
        self.sim_running = True
        self._busy(True, f"STDP sweep: {len(dts)} timings x "
                         f"{len(models)} model(s)")
        self.log(f"[stdp] sweep dt {dts[0] * 1e3:+.4g}..{dts[-1] * 1e3:+.4g} ms"
                 f" ({len(dts)} pts), "
                 f"pre {dpg.get_value('stdp_amp_pre'):+g} {unit}, "
                 f"post {dpg.get_value('stdp_amp_post'):+g} {unit}, "
                 f"width {dpg.get_value('stdp_width_ms'):g} ms, "
                 f"settle {dpg.get_value('stdp_settle_ms'):g} ms")
        threading.Thread(target=self._stdp_worker,
                         args=(models, amp_pre, amp_post, width, dts, tail),
                         daemon=True).start()

    def _stdp_worker(self, models, amp_pre, amp_post, width, dts, tail):
        t0c = 10e-3
        curves = {}
        t_start = time.perf_counter()
        try:
            for m in models:
                ys = []
                for dt in dts:
                    pre_t = t0c + max(0.0, -dt)
                    post_t = pre_t + dt
                    wf = Waveform([(pre_t, width, amp_pre),
                                   (post_t, width, amp_post)])
                    t_stop = max(pre_t, post_t) + width + tail
                    r = simulate(m, wf, t_stop, label=m.name)
                    ys.append(float((r.G[-1] - r.G[0]) * 1e6))
                curves[m.name] = ys
                self.q.put(("log", f"  [stdp] {m.name}: dG "
                                   f"{min(ys):+.4g}..{max(ys):+.4g} uS"))
        except Exception as e:
            self.q.put(("log", f"[stdp] FAILED: {e!r}"))
        self.q.put(("log", f"[stdp] sweep done in "
                           f"{time.perf_counter() - t_start:.1f} s"))
        self.q.put(("stdp", [d * 1e3 for d in dts], curves))

    def _show_stdp(self, dts_ms, curves):
        self.sim_running = False
        self._clear_probes(("stdp_plot",))
        for s in self._stdp_series:
            if dpg.does_item_exist(s):
                dpg.delete_item(s)
        self._stdp_series = []
        if not curves:
            self._busy(False, "STDP sweep failed")
            return
        lines = []
        nan = float("nan")
        for label, ys in curves.items():
            # break the line across the empty overlap gap (no segment joining
            # the -width and +width branches) via a NaN point at dt = 0
            neg = [(d, y) for d, y in zip(dts_ms, ys) if d < 0]
            pos = [(d, y) for d, y in zip(dts_ms, ys) if d > 0]
            line_x = [d for d, _ in neg] + [0.0] + [d for d, _ in pos]
            line_y = [y for _, y in neg] + [nan] + [y for _, y in pos]
            self._stdp_series.append(dpg.add_line_series(
                line_x, line_y, parent="stdp_y", label=label))
            self._stdp_series.append(dpg.add_scatter_series(
                dts_ms, ys, parent="stdp_y", label="##pts_" + label))
            imax = max(range(len(ys)), key=lambda i: abs(ys[i]))
            lines.append(f"{label}:  dG {min(ys):+.4g}..{max(ys):+.4g} uS"
                         f"  |  strongest change at dt = "
                         f"{dts_ms[imax]:+.3g} ms")
        self._stdp_summary = lines
        for ln in lines:
            self.log("  [stdp] " + ln)
        self.log("  [stdp] click any curve point to drill into that timing "
                 "(R/G/spike transient + dT, dG, dR)")
        self._fit_axes_of(("stdp_x", "stdp_y"))
        self._busy(False, f"STDP curve ready - {len(curves)} model(s)")
        dpg.set_value("center_tabs", "tab_stdp")

    def _stdp_drilldown(self):
        """Re-run the single pre/post pair at the clicked dt and show the
        full transient plus dT / dG / dR. Only fires when the click lands ON
        a data point (2D proximity in both dt and dG), not anywhere in the
        empty plot canvas."""
        ctx = self._stdp_ctx
        if not ctx or self.sim_running:
            return
        try:
            mx, my = dpg.get_plot_mouse_pos()
        except Exception:
            return
        x0, x1 = dpg.get_axis_limits("stdp_x")
        y0, y1 = dpg.get_axis_limits("stdp_y")
        xs_span = (x1 - x0) or 1.0
        ys_span = (y1 - y0) or 1.0
        best = self._nearest_series_point("stdp_y", mx, my, xs_span, ys_span)
        # require the click within ~3.5% of the plot of an actual point
        if best is None or best[0] > 0.035 ** 2:
            return
        dt_ms = best[1]
        dts_ms = [d * 1e3 for d in ctx["dts"]]
        i = min(range(len(dts_ms)), key=lambda k: abs(dts_ms[k] - dt_ms))
        self._stdp_show_dt(i)

    def _stdp_show_dt(self, i):
        ctx = self._stdp_ctx
        dt = ctx["dts"][i]
        width, tail = ctx["width"], ctx["tail"]
        pre_t = 10e-3 + max(0.0, -dt)
        post_t = pre_t + dt
        wf = Waveform([(pre_t, width, ctx["amp_pre"]),
                       (post_t, width, ctx["amp_post"])])
        t_stop = max(pre_t, post_t) + width + tail
        results, summary = [], []
        for m in ctx["models"]:
            r = simulate(m, wf, t_stop, label=m.name)
            results.append(r)
            dG = (r.G[-1] - r.G[0]) * 1e6
            dR = r.R[-1] - r.R[0]
            summary.append(f"{m.name}: dG={dG:+.4g} uS  dR={dR:+.4g} ohm")
        order = "post after pre (potentiating side)" if dt > 0 else \
                ("pre after post (depressing side)" if dt < 0
                 else "simultaneous")
        msg = f"dt = {dt * 1e3:+.4g} ms ({order}) | " + "  |  ".join(summary)
        self.log("[stdp] " + msg)
        self._show_results(results, {}, ctx["unit"], ctx["kind"])
        self._set_status("● " + msg, (126, 170, 255))

    # ---------------- plots ------------------------------------------

    def _show_results(self, results, meta, unit, kind, live=False):
        self.results = results
        self.results_meta = meta
        self.results_unit = unit
        self._clear_probes(("plot_i", "plot_r", "plot_g"))
        for s in self._series:
            if dpg.does_item_exist(s):
                dpg.delete_item(s)
        self._series = []
        if not results:
            self._busy(False, "no results")
            return
        scale = 1.0 / sf.UNIT_SCALE.get(unit, 1.0)
        xs, ys = self._stair_points(results[0].waveform, scale,
                                    tail=0.05 * results[0].t[-1])
        self._series.append(dpg.add_stair_series(
            xs, ys, parent="ax_i_y", label="stimulus"))
        dpg.configure_item("ax_i_y",
                           label=("I_gate (%s)" if kind == "current"
                                  else "V_gate (%s)") % unit)
        for r in results:
            self._series.append(dpg.add_line_series(
                r.t.tolist(), r.R.tolist(), parent="ax_r_y", label=r.label))
            self._series.append(dpg.add_line_series(
                r.t.tolist(), (r.G * 1e6).tolist(), parent="ax_g_y",
                label=r.label))
        self.fit_axes()
        self.update_analysis()
        if live:
            self._busy(False, f"live · {len(results)} result(s) "
                              f"(agent edit applied)")
        else:
            self._busy(False, f"done · {len(results)} result(s)")
            dpg.set_value("center_tabs", "tab_results")

    def fit_axes(self):
        for ax in ("ax_i_x", "ax_i_y", "ax_r_x", "ax_r_y",
                   "ax_g_x", "ax_g_y", "ana_x", "ana_y"):
            if dpg.does_item_exist(ax):
                dpg.fit_axis_data(ax)

    def on_analysis_metric(self, metric):
        self.analysis_metric = "R" if metric == "R" else "G"
        if dpg.does_item_exist("ana_menu_G"):
            dpg.set_value("ana_menu_G", self.analysis_metric == "G")
            dpg.set_value("ana_menu_R", self.analysis_metric == "R")
        if dpg.does_item_exist("ana_metric_combo"):
            dpg.set_value("ana_metric_combo",
                          "Resistance R_mem" if self.analysis_metric == "R"
                          else "Conductance G")
        # keep the hover-bubble label in sync with the plotted quantity
        name, unit = self._ana_name_unit()
        App.HOVER_FMT["ana_plot"] = ("pulse", "", name, unit)
        self.update_analysis()
        if self.results:
            dpg.set_value("center_tabs", "tab_analysis")

    def _ana_name_unit(self):
        return ("R_mem", "ohm") if self.analysis_metric == "R" else ("G", "uS")

    def update_analysis(self):
        self._clear_probes(("ana_plot",))
        for s in self._ana_series:
            if dpg.does_item_exist(s):
                dpg.delete_item(s)
        self._ana_series = []
        name, unit = self._ana_name_unit()
        if dpg.does_item_exist("ana_y"):
            dpg.configure_item("ana_y", label=f"{name} ({unit})")
        if dpg.does_item_exist("ana_caption"):
            full = "resistance" if self.analysis_metric == "R" else "conductance"
            dpg.set_value("ana_caption",
                          f"retained {full} sampled after each pulse")
        if not self.results:
            return
        wf = self.results[0].waveform
        wins = wf.pulse_windows()
        if len(wins) < 2:
            dpg.set_value("ana_text", f"need a multi-pulse stimulus for the "
                                      f"{name}-vs-pulse curve")
            return
        ends = [t1 for _, t1 in wins]
        starts = [t0 for t0, _ in wins]
        t_sim = float(self.results[0].t[-1])
        # sample each pulse's retained value relative to ITS OWN gap to the
        # next pulse (not a single global settle), capped at the median gap
        # so every point is read after a comparable relaxation time.
        inter = [starts[k + 1] - ends[k] for k in range(len(ends) - 1)
                 if starts[k + 1] > ends[k]]
        typ = sorted(inter)[len(inter) // 2] if inter else (t_sim - ends[-1])
        cap = max(1e-4, 0.9 * typ)
        sample_t = []
        for k in range(len(ends)):
            gap = (starts[k + 1] - ends[k]) if k + 1 < len(ends) \
                else (t_sim - ends[k])
            s = min(max(0.9 * gap if gap > 0 else cap, 1e-5), cap)
            sample_t.append(min(ends[k] + s, t_sim))
        n_each = self.results_meta.get("n_each")
        lines = []
        for r in self.results:
            R_s, G_s = r.at(sample_t)
            arr = (R_s if self.analysis_metric == "R" else G_s * 1e6)
            vals = arr.tolist()
            n = list(range(1, len(vals) + 1))
            if n_each and 0 < n_each < len(vals):
                self._ana_series.append(dpg.add_line_series(
                    n[:n_each], vals[:n_each], parent="ana_y",
                    label=f"{r.label} LTP"))
                # LTD starts one point early (the last LTP point) so the two
                # branches join with no visual gap at the turnover
                self._ana_series.append(dpg.add_line_series(
                    n[n_each - 1:], vals[n_each - 1:], parent="ana_y",
                    label=f"{r.label} LTD"))
                dl = [b - a for a, b in zip(vals[:n_each - 1], vals[1:n_each])]
                dd = [b - a for a, b in zip(vals[n_each:-1], vals[n_each + 1:])]
                mean = lambda v: sum(v) / len(v) if v else 0.0
                lines.append(
                    f"{r.label}:  {name} {min(vals):.4g}..{max(vals):.4g} "
                    f"{unit} | mean d{name}  LTP {mean(dl):+.4g}  "
                    f"LTD {mean(dd):+.4g} {unit}/pulse")
            else:
                self._ana_series.append(dpg.add_line_series(
                    n, vals, parent="ana_y", label=r.label))
                lines.append(f"{r.label}:  {name} {min(vals):.4g}.."
                             f"{max(vals):.4g} {unit} over {len(vals)} pulses")
        dpg.set_value("ana_text", "\n".join(lines))
        dpg.fit_axis_data("ana_x")
        dpg.fit_axis_data("ana_y")

    # ---------------- export -----------------------------------------

    def export_csv(self):
        if not self.results:
            self.log("[export] no results yet - run a simulation first")
            return
        outdir = os.path.join(self.workdir, "results")
        os.makedirs(outdir, exist_ok=True)
        for r in self.results:
            stem = re.sub(r"[^\w]+", "_", r.label).strip("_")
            p = r.save_csv(os.path.join(outdir, f"gui_{stem}.csv"))
            self.log(f"[export] csv -> {p}")

    def export_png(self):
        if not self.results:
            self.log("[export] no results yet - run a simulation first")
            return
        from ecfet import plotting
        outdir = os.path.join(self.workdir, "results")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "gui_plot.png")
        plotting.plot_transient(self.results, path,
                                title="NeuroVAT run", show_extras=True)
        self.log(f"[export] png -> {path}")

    # =================================================================
    # Verilog-A browsing / editing
    # =================================================================

    def rescan_va(self, startup=False):
        self.va_files = va_scan(self.workdir)
        names = [v.name for v in self.va_files]
        dpg.delete_item("va_cards", children_only=True)
        with dpg.table(parent="va_cards", header_row=False,
                       policy=dpg.mvTable_SizingStretchProp,
                       no_pad_outerX=True, scrollY=False) as tbl:
            dpg.bind_item_theme(tbl, self.themes["table_list"])
            dpg.add_table_column(width_fixed=True, init_width_or_weight=20)
            dpg.add_table_column(width_stretch=True)
            dpg.add_table_column(width_fixed=True, init_width_or_weight=58)
            for v in self.va_files:
                mapped = SPEC_BY_KEY.get(v.model_key)
                with dpg.table_row():
                    if v.model_key:
                        default = self.file_enabled.get(
                            v.name, v.model_key in ("v1", "v2"))
                        self.file_enabled[v.name] = default
                        dpg.add_checkbox(tag=f"cb_file_{v.name}",
                                         default_value=default,
                                         user_data=v.name,
                                         callback=self._on_file_toggle)
                    else:
                        dpg.add_text("")
                    dpg.add_selectable(
                        label=v.name, tag=f"vasel_{v.name}", user_data=v.name,
                        default_value=(self.selected_va is not None
                                       and v.name == self.selected_va.name),
                        callback=lambda s, a, u: self.on_va_selected(u))
                    if mapped:
                        self._small("." + mapped.input_kind, color=C_GREEN)
                    else:
                        dpg.add_text("")
        dpg.configure_item("va_edit_sel", items=names)
        if not startup:
            self.log(f"[va] found {len(names)} file(s): {', '.join(names)}")
        else:
            self.log(f"[va] auto-detected {len(names)} Verilog-A file(s)")
        if names and not self.selected_va:
            self.on_va_selected(names[0], interact=False)

    def _va_by_name(self, name):
        for v in self.va_files:
            if v.name == name:
                return v
        return None

    def _on_file_toggle(self, sender, value, user_data):
        self.file_enabled[user_data] = value

    def _enabled_keys(self):
        """Model twin keys enabled via checked .va files (deduped)."""
        keys = []
        for v in self.va_files:
            if not v.model_key or v.model_key in keys:
                continue
            tag = f"cb_file_{v.name}"
            if dpg.does_item_exist(tag) and dpg.get_value(tag):
                keys.append(v.model_key)
        return keys

    def on_va_selected(self, name, interact=True):
        va = self._va_by_name(name)
        self.selected_va = va
        for v in self.va_files:
            tag = f"vasel_{v.name}"
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, v.name == name)
        if not interact:
            return
        now = time.monotonic()
        prev_name, prev_t = self._last_va_click
        self._last_va_click = (name, now)
        if name == prev_name and now - prev_t < 0.35:
            self.on_open_in_editor()      # double-click -> edit source
        else:
            self.on_apply_va_params()     # single click -> load params

    def on_apply_va_params(self):
        va = self.selected_va
        if not va:
            self.log("[va] select a .va file first")
            return
        spec = SPEC_BY_KEY.get(va.model_key) or self._param_spec()
        vals = self.param_values[spec.key]
        fields = {k.lower(): k for k in vals}
        applied, skipped = [], []
        for name, value in va.params.items():
            k = fields.get(name.lower())
            if k is None:
                skipped.append(name)
                continue
            vals[k] = int(value) if isinstance(vals[k], int) \
                and not isinstance(vals[k], bool) else float(value)
            applied.append(name)
        dpg.set_value("param_model_sel", spec.label)
        self.rebuild_param_panel()
        self.log(f"[va] {va.name} -> {spec.key}: applied {len(applied)} "
                 f"param(s)" + (f"; no match for {len(skipped)}"
                                if skipped else ""))

    def on_open_in_editor(self):
        if not self.selected_va:
            return
        dpg.set_value("va_edit_sel", self.selected_va.name)
        self.on_editor_file_change(None, self.selected_va.name)
        dpg.set_value("center_tabs", "tab_source")

    def on_editor_file_change(self, sender, name):
        va = self._va_by_name(name)
        if not va:
            return
        try:
            with open(va.path, "r", encoding="utf-8", errors="replace") as f:
                dpg.set_value("va_editor", f.read())
            self.editor_path = va.path
            self.editor_mtime = os.path.getmtime(va.path)
            dpg.set_value("editor_status", f"loaded {va.name}")
        except OSError as e:
            dpg.set_value("editor_status", f"error: {e}")

    def on_editor_save(self):
        if not self.editor_path:
            return
        try:
            with open(self.editor_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(dpg.get_value("va_editor"))
            self.editor_mtime = os.path.getmtime(self.editor_path)
            dpg.set_value("editor_status",
                          f"saved {os.path.basename(self.editor_path)}")
            self.log(f"[editor] saved {self.editor_path}")
            self.rescan_va()
        except OSError as e:
            dpg.set_value("editor_status", f"save error: {e}")

    def on_editor_reload(self):
        if self.editor_path:
            self.on_editor_file_change(None, os.path.basename(self.editor_path))

    def on_ask_about_file(self):
        name = dpg.get_value("va_edit_sel") or (
            self.selected_va.name if self.selected_va else "")
        if name:
            dpg.set_value("chat_input",
                          f"Explain {name}: device physics, parameters, and "
                          f"any issues you notice.")
            dpg.focus_item("chat_input")

    def _check_external_edits(self):
        if self.editor_path and os.path.isfile(self.editor_path):
            m = os.path.getmtime(self.editor_path)
            if m > self.editor_mtime + 1e-6:
                self.on_editor_file_change(
                    None, os.path.basename(self.editor_path))
                self.log(f"[agent] modified "
                         f"{os.path.basename(self.editor_path)} - editor "
                         f"reloaded")
        self.rescan_va()

    # =================================================================
    # Virtuoso (skillbridge over SSH tunnel)
    # =================================================================

    def on_virtuoso_connect(self, *_):
        if self.virt_busy:
            return
        if self.virtuoso.connected:
            self.log("[virtuoso] already connected")
            return
        self.virt_busy = True
        dpg.configure_item("btn_virt_connect", enabled=False)
        dpg.configure_item("virt_dot", color=C_AMBER)
        dpg.set_value("virt_status", "connecting...")
        self.log("[virtuoso] connecting (SSH tunnel + skillbridge)...")

        def worker():
            try:
                info = self.virtuoso.connect()
                self.q.put(("virtuoso", True, info))
            except Exception as e:
                self.q.put(("virtuoso", False, str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def on_virtuoso_disconnect(self, *_):
        if not self.virtuoso.connected and self.virtuoso.tunnel is None:
            self.log("[virtuoso] nothing to disconnect")
            return
        self.virtuoso.disconnect()
        dpg.configure_item("virt_dot", color=C_RED)
        dpg.set_value("virt_status", "not connected")
        dpg.set_value("virt_info", "")
        dpg.configure_item("menu_virtuoso", label="Virtuoso")
        for tag in ("virt_lib_combo", "virt_cell_combo", "virt_view_combo"):
            dpg.configure_item(tag, items=[])
            dpg.set_value(tag, "")
        dpg.set_value("virt_browse_status", "connect to Virtuoso first "
                                            "(left panel)")
        self.log("[virtuoso] disconnected")

    def on_virtuoso_libs(self, *_):
        libs = self.virtuoso.info.get("libraries", [])
        if not self.virtuoso.connected:
            self.log("[virtuoso] not connected")
        elif libs:
            self.log(f"[virtuoso] {len(libs)} libraries: {', '.join(libs)}")
        else:
            self.log("[virtuoso] no libraries reported")
        dpg.set_value("center_tabs", "tab_log")

    def _virt_dialog(self, ok, title, body):
        dpg.configure_item("virt_modal_dot", color=C_GREEN if ok else C_RED)
        dpg.set_value("virt_modal_title", title)
        dpg.set_value("virt_modal_text", body)
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()
        dpg.configure_item("virt_modal", show=True,
                           pos=(max(0, (vw - 470) // 2), max(0, vh // 3)))

    def _on_virtuoso_done(self, ok, payload):
        self.virt_busy = False
        dpg.configure_item("btn_virt_connect", enabled=True)
        if ok:
            ver = self.virtuoso.short_version()
            libs = payload.get("libraries", [])
            dpg.configure_item("virt_dot", color=C_GREEN)
            dpg.set_value("virt_status", f"connected · {ver}")
            dpg.configure_item("menu_virtuoso", label="Virtuoso (Connected)")
            head = ", ".join(libs[:5]) + (" ..." if len(libs) > 5 else "")
            dpg.set_value("virt_info",
                          f"{payload['tunnel']} · {len(libs)} libraries"
                          + (f": {head}" if libs else ""))
            self.log(f"[virtuoso] connected ({payload['tunnel']}): "
                     f"{payload['version']}")
            if libs:
                self.log(f"[virtuoso] libraries: {', '.join(libs)}")
            self._virt_dialog(True, "Connected to Virtuoso",
                              f"Virtuoso {ver} ({payload['tunnel']})\n"
                              f"{len(libs)} libraries available"
                              + (f": {head}" if libs else ""))
            if libs:
                dpg.configure_item("virt_lib_combo", items=libs)
                dpg.set_value("virt_browse_status",
                              f"{len(libs)} libraries · pick one to list cells")
        else:
            dpg.configure_item("virt_dot", color=C_RED)
            dpg.set_value("virt_status", "connection failed")
            dpg.set_value("virt_info", payload)
            dpg.configure_item("menu_virtuoso", label="Virtuoso")
            self.log(f"[virtuoso] FAILED: {payload}")
            self._virt_dialog(False, "Connection failed", str(payload))

    # ---------------- library/cell/view browser ----------------------

    def _virt_browse(self, fn, kind):
        """Run a blocking skillbridge browse call off the UI thread."""
        if not self.virtuoso.connected:
            dpg.set_value("virt_browse_status", "not connected to Virtuoso")
            return

        def worker():
            try:
                self.q.put(("virt_browse", kind, fn()))
            except Exception as e:
                self.q.put(("virt_browse", "error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def on_virt_refresh_libs(self, *_):
        dpg.set_value("virt_browse_status", "refreshing libraries...")
        self._virt_browse(self.virtuoso.list_libraries, "libs")

    def _virt_lib_changed(self):
        lib = dpg.get_value("virt_lib_combo")
        for tag in ("virt_cell_combo", "virt_view_combo"):
            dpg.configure_item(tag, items=[])
            dpg.set_value(tag, "")
        if not lib:
            return
        va_only = dpg.get_value("virt_va_only")
        dpg.set_value("virt_browse_status", f"listing cells in {lib}...")
        self._virt_browse(lambda: self.virtuoso.list_cells(lib, va_only),
                          "cells")

    def _virt_cell_changed(self):
        lib = dpg.get_value("virt_lib_combo")
        cell = dpg.get_value("virt_cell_combo")
        dpg.configure_item("virt_view_combo", items=[])
        dpg.set_value("virt_view_combo", "")
        if not (lib and cell):
            return
        dpg.set_value("virt_browse_status", f"listing views of {cell}...")
        self._virt_browse(lambda: self.virtuoso.list_views(lib, cell), "views")

    def on_virt_load_source(self, *_):
        lib = dpg.get_value("virt_lib_combo")
        cell = dpg.get_value("virt_cell_combo")
        view = dpg.get_value("virt_view_combo")
        if not (lib and cell and view):
            dpg.set_value("virt_browse_status",
                          "pick a library, cell, and view first")
            return
        dpg.set_value("virt_browse_status", f"loading {lib}/{cell}/{view}...")
        self._virt_browse(
            lambda: dict(self.virtuoso.read_source(lib, cell, view),
                         lib=lib, cell=cell, view=view), "source")

    def _on_virt_browse(self, kind, payload):
        if kind == "error":
            dpg.set_value("virt_browse_status", f"error: {payload}")
            self.log(f"[virtuoso] browse error: {payload}")
            return
        if kind == "libs":
            dpg.configure_item("virt_lib_combo", items=payload)
            dpg.set_value("virt_browse_status",
                          f"{len(payload)} libraries")
        elif kind == "cells":
            dpg.configure_item("virt_cell_combo", items=payload)
            only = " Verilog" if dpg.get_value("virt_va_only") else ""
            if payload:
                dpg.set_value("virt_browse_status",
                              f"{len(payload)}{only} cell(s) · pick one")
            else:
                hint = (" — uncheck 'Verilog views only' to see all cells"
                        if dpg.get_value("virt_va_only") else "")
                dpg.set_value("virt_browse_status",
                              f"no{only} cells in this library{hint}")
        elif kind == "views":
            dpg.configure_item("virt_view_combo", items=payload)
            if payload:
                pref = next((v for v in payload
                             if v in ("veriloga", "verilogams", "functional",
                                      "verilog", "ahdl")), payload[0])
                dpg.set_value("virt_view_combo", pref)
                dpg.set_value("virt_browse_status",
                              f"{len(payload)} view(s) · Load source")
            else:
                dpg.set_value("virt_browse_status", "cell has no views")
        elif kind == "source":
            self._load_virt_source(payload)

    def _load_virt_source(self, res):
        lib, cell, view = res["lib"], res["cell"], res["view"]
        tag = f"{lib}/{cell}/{view}"
        if not res["ok"]:
            dpg.set_value("virt_browse_status", f"{tag}: {res['note']}")
            self.log(f"[virtuoso] {tag}: {res['note']}"
                     + (f" (files: {', '.join(res['files'])})"
                        if res["files"] else ""))
            self._virt_dialog(False, "No readable source",
                              f"{tag}\n{res['note']}"
                              + (f"\n\nfiles: {', '.join(res['files'])}"
                                 if res["files"] else ""))
            return
        dpg.set_value("va_editor", res["text"])
        # Remote buffer: detach from any local path so Save can't clobber a
        # local file with remote content.
        self.editor_path = None
        self.editor_mtime = 0.0
        dpg.set_value("va_edit_sel", "")
        dpg.set_value("editor_status",
                      f"remote: {tag}  (read-only — {res['path']})")
        dpg.set_value("virt_browse_status",
                      f"loaded {tag} · {len(res['text'])} chars")
        dpg.set_value("center_tabs", "tab_source")
        self.log(f"[virtuoso] loaded {tag} from {res['path']} "
                 f"({len(res['text'])} chars)")

    # =================================================================
    # agent chat logic
    # =================================================================

    def _downsample(self, arr, n=240):
        m = len(arr)
        if m <= n:
            return [round(float(v), 6) for v in arr]
        step = m / n
        return [round(float(arr[min(int(i * step), m - 1)]), 6)
                for i in range(n)]

    def _write_agent_snapshot(self):
        """Dump the current plot/sim state to JSON so the agent can read the
        actual waveform/STDP/probe data with its file tools.  Returns the
        path, or None if there's nothing to show."""
        snap = {"workspace": self.workdir,
                "note": ("The GUI simulates the Python model twins in ecfet/ "
                         "(model_v1.py, model_v2.py, model_fefet.py); the .va "
                         "files are the Verilog-A sources. The plotted curves "
                         "come from the Python twins. To change what the plot "
                         "shows, edit the matching twin; to change the Verilog "
                         "model, edit the .va. Keep the two in sync."),
                "verilog_files": [], "models_enabled": [], "stimulus": None,
                "results": [], "stdp": None, "probes": {}}

        for v in self.va_files:
            key = v.model_key
            snap["verilog_files"].append({
                "va_file": v.name, "module": v.module,
                "python_twin": TWIN_FILE.get(key),
                "params": v.raw_params})

        for key in self._enabled_keys():
            s = SPEC_BY_KEY[key]
            snap["models_enabled"].append({
                "label": s.label, "key": key,
                "python_twin": TWIN_FILE[key],
                "params": self.param_values[key]})

        try:
            wf, meta, unit, kind, label = self.build_waveform()
            snap["stimulus"] = {
                "generator": label, "kind": kind, "unit": unit,
                "n_pulses": len(wf.pulses),
                "pulses_[t_s,width_s,amp]": [[round(p[0], 6), round(p[1], 6),
                                              p[2]] for p in wf.pulses[:400]]}
        except ValueError:
            pass

        for r in self.results:
            snap["results"].append({
                "model": r.label,
                "R_ohm": {"start": float(r.R[0]), "end": float(r.R[-1]),
                          "min": float(r.R.min()), "max": float(r.R.max())},
                "G_uS": {"start": float(r.G[0] * 1e6),
                         "end": float(r.G[-1] * 1e6),
                         "min": float(r.G.min() * 1e6),
                         "max": float(r.G.max() * 1e6)},
                "t_s": self._downsample(r.t),
                "R_ohm_trace": self._downsample(r.R),
                "G_uS_trace": self._downsample(r.G * 1e6)})

        if self._stdp_series and self._stdp_ctx:
            curves = {}
            for sid in self._stdp_series:
                lbl = dpg.get_item_configuration(sid).get("label", "")
                if lbl.startswith("##"):
                    continue
                d = dpg.get_value(sid)
                if d and len(d) >= 2:
                    curves[lbl] = {"dt_ms": [round(float(x), 4) for x in d[0]],
                                   "dG_uS": [round(float(y), 5) for y in d[1]]}
            snap["stdp"] = {"curves": curves}

        for plot, probes in self._probes.items():
            entry = {}
            for which in ("A", "B"):
                m = probes.get(which)
                if m:
                    entry[which] = {"x": m["x"], "y": m["y"],
                                    "series": m.get("series")}
            a, b = probes.get("A"), probes.get("B")
            if a and b:
                dx, dy = b["x"] - a["x"], b["y"] - a["y"]
                entry["delta"] = {"dX": dx, "dY": dy,
                                  "slope_dy_dx": (None if dx == 0 else dy / dx)}
            if entry:
                snap["probes"][self.PLOT_NAMES.get(plot, plot)] = entry

        if not (snap["results"] or snap["stdp"]):
            return None
        path = os.path.join(self.workdir, "results", "agent_snapshot.json")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snap, f, indent=2)
        except OSError as e:
            self.log(f"[agent] snapshot write failed: {e}")
            return None
        return path

    def _agent_context(self):
        lines = [f"Workspace: {self.workdir}",
                 "Verilog-A files: " + ", ".join(
                     f"{v.name} (module {v.module})" for v in self.va_files)]
        if self.selected_va:
            lines.append(f"User-selected .va: {self.selected_va.name}")
        checked = [SPEC_BY_KEY[k].label for k in self._enabled_keys()]
        lines.append("Models enabled: " + (", ".join(checked) or "none"))
        if self.virtuoso.connected:
            libs = self.virtuoso.info.get("libraries", [])
            lines.append(f"Virtuoso: connected via skillbridge "
                         f"({self.virtuoso.short_version()}), "
                         f"libraries: {', '.join(libs) or 'none'}")
        else:
            lines.append("Virtuoso: not connected")
        try:
            wf, meta, unit, kind, label = self.build_waveform()
            t_end = wf.breakpoints[-1] if wf.breakpoints else 0
            lines.append(f"Current stimulus: {label}, {len(wf.pulses)} pulses, "
                         f"{kind} in {unit}, last edge {t_end:.4g} s")
        except ValueError:
            pass

        snap = self._write_agent_snapshot()
        if snap:
            rel = os.path.relpath(snap, self.workdir)
            lines.append(
                f"Current plot/simulation data: {rel}  (READ this file for "
                f"the actual R(t)/G(t) traces, STDP curve, and probe A/B "
                f"readings now on screen). The plotted curves are produced by "
                f"the Python twins in ecfet/ (model_v1.py=ECFET v1, "
                f"model_v2.py=ECFET v2, model_fefet.py=FeFET), NOT directly by "
                f"the .va. So when the user says a part of the plot 'breaks', "
                f"diagnose it in the matching Python twin and fix it there; if "
                f"they want the Verilog model changed too, edit the .va to "
                f"match. The snapshot maps each .va to its twin.")
        if self.agent.backend == "sdk" and self.selected_va:
            try:
                with open(self.selected_va.path, "r", encoding="utf-8",
                          errors="replace") as f:
                    src = f.read()[:6000]
                lines.append(f"--- {self.selected_va.name} ---\n{src}\n---")
            except OSError:
                pass
        return "\n".join(lines)

    def on_send(self, *_):
        if self.chat_busy:
            return
        text = (dpg.get_value("chat_input") or "").strip()
        if not text:
            return
        dpg.set_value("chat_input", "")
        self.append_chat("you", text)
        if text.startswith("/"):
            self._handle_command(text)
            return
        self.chat_busy = True
        dpg.configure_item("btn_send", enabled=False)
        self._chat_pending(True)
        ctx = self._agent_context()
        auto = dpg.get_value("cb_auto")
        edits = dpg.get_value("cb_edits") or auto
        bash = dpg.get_value("cb_bash") or auto
        model = self.agent_model_id or "default"
        if edits:                       # snapshot before the agent can edit
            self._backup_editable()
            dpg.configure_item("btn_send", show=False)
            dpg.configure_item("btn_stop", show=True)

        def worker():
            res = self.agent.send(text, ctx, allow_edits=edits,
                                  allow_bash=bash, model=model,
                                  autonomous=auto)
            self.q.put(("chat", res, edits or bash))

        threading.Thread(target=worker, daemon=True).start()

    def _on_chat_done(self, res, may_have_edited):
        self.chat_busy = False
        dpg.configure_item("btn_send", enabled=True, show=True)
        dpg.configure_item("btn_stop", show=False)
        self._chat_pending(False)
        text = res.get("text") or ""
        wf = self.agent.extract_waveform(text)
        if res.get("ok"):
            shown = text
            if wf:
                shown = re.sub(r"```(?:json)?\s*\{.*?\}\s*```",
                               "", shown, flags=re.S).strip()
                shown = shown or f"Here is the pattern — {wf['label']}."
            self.append_chat("agent", shown or "(empty reply)")
            if wf:
                self._add_pattern_card(wf)
        else:
            self.append_chat("err", res.get("error") or "unknown error")
        if self.agent.total_cost:
            dpg.set_value("cost_label",
                          f"session cost ${self.agent.total_cost:.3f}")
        if may_have_edited:
            self._check_external_edits()

    def on_load_agent_pattern(self, sender=None, app_data=None, user_data=None):
        wf = user_data
        if not wf:
            return
        kind = "voltage" if wf["kind"] == "voltage" else "current"
        dpg.set_value("kind_combo", kind)
        self._sync_unit_combo()
        units = sf.CURRENT_UNITS if kind == "current" else sf.VOLTAGE_UNITS
        unit = wf["unit"] if wf["unit"] in units else units[-1]
        dpg.set_value("unit_combo", unit)
        rows = [f"# {wf['label']} (from agent)"]
        for t0, w, a in wf["pulses"]:
            rows.append(f"{t0 * 1e3:.6g} {w * 1e3:.6g} {a:.6g}")
        dpg.set_value("custom_text", "\n".join(rows) + "\n")
        dpg.set_value("gen_combo", "Custom pattern")
        self.rebuild_gen_params()
        dpg.set_value("center_tabs", "tab_designer")
        self.on_preview()
        self.log(f"[agent] loaded pattern '{wf['label']}' "
                 f"({len(wf['pulses'])} pulses) into the designer")

    def on_agent_reset(self):
        self.agent.reset()
        dpg.delete_item("chat_log", children_only=True)
        self.append_chat("agent", "Conversation reset.")
        self.log("[agent] conversation reset")

    def on_agent_stop(self):
        if self.agent.stop():
            self.log("[agent] stop requested - terminating run")
            self.append_chat("sys", "Stopping the agent...")
        else:
            self.log("[agent] nothing running to stop")

    # ---- live re-plot on code change --------------------------------

    def _reload_models(self):
        """Hot-reload the edited model twins so the GUI simulates new code."""
        for key, (mod, clsname, pname) in RELOAD_MODULES.items():
            importlib.reload(mod)
            spec = SPEC_BY_KEY[key]
            spec.cls = getattr(mod, clsname)
            spec.params_cls = getattr(mod, pname)

    def _twin_mtime_map(self):
        m = {}
        for rel in TWIN_FILE.values():
            p = os.path.join(self.workdir, rel)
            try:
                m[rel] = os.path.getmtime(p)
            except OSError:
                m[rel] = 0.0
        return m

    def _watch_code(self):
        """Poll the model twins; when one changes (and is stable across two
        polls), hot-reload and re-simulate so the plot tracks the agent."""
        if not dpg.does_item_exist("cb_liveplot") \
                or not dpg.get_value("cb_liveplot"):
            return
        self._watch_tick += 1
        if self._watch_tick % 30 != 0:        # ~ every 0.5 s at 60 fps
            return
        cur = self._twin_mtime_map()
        if self._twin_mtimes is None:
            self._twin_mtimes = cur
            self._prev_mtimes = cur
            return
        changed = any(cur[k] != self._twin_mtimes.get(k) for k in cur)
        # require stability across two polls so we don't read a half-written file
        stable = cur == self._prev_mtimes
        self._prev_mtimes, self._twin_mtimes = self._twin_mtimes, cur
        if changed and stable and not self.sim_running:
            self.log("[live] model code changed - reloading + re-plotting")
            try:
                self._reload_models()
            except Exception as e:
                self.log(f"[live] reload failed (syntax error in twin?): {e}")
                return
            self.on_run(live=True)

    # ---- backup / revert agent edits --------------------------------

    def _editable_files(self):
        files = list(TWIN_FILE.values())
        for v in self.va_files:
            rel = os.path.relpath(v.path, self.workdir)
            files.append(rel)
        return files

    def _backup_editable(self):
        """Snapshot the model twins + .va files before an agent edit turn."""
        bdir = os.path.join(self.workdir, "results", ".agent_backup")
        try:
            if os.path.isdir(bdir):
                shutil.rmtree(bdir, ignore_errors=True)
            os.makedirs(bdir, exist_ok=True)
            for rel in self._editable_files():
                src = os.path.join(self.workdir, rel)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(bdir, rel.replace("/", "__").replace("\\",
                                                                        "__"))
                shutil.copy2(src, dst)
            self._backup_dir = bdir
        except OSError as e:
            self.log(f"[agent] backup failed: {e}")
            self._backup_dir = None

    def on_agent_revert(self):
        bdir = self._backup_dir
        if not bdir or not os.path.isdir(bdir):
            self.log("[agent] no backup to revert to")
            self.append_chat("sys", "No backup available to revert.")
            return
        n = 0
        for rel in self._editable_files():
            dst = os.path.join(self.workdir, rel)
            src = os.path.join(bdir, rel.replace("/", "__").replace("\\", "__"))
            if os.path.isfile(src):
                try:
                    shutil.copy2(src, dst)
                    n += 1
                except OSError as e:
                    self.log(f"[agent] revert {rel} failed: {e}")
        self.log(f"[agent] reverted {n} file(s) to pre-agent state")
        self.append_chat("sys", f"Reverted {n} file(s) to the state before the "
                                "last agent turn.")
        try:
            self._reload_models()
            self.rescan_va()
            if self.editor_path:
                self.on_editor_reload()
            if self.results:
                self.on_run(live=True)
        except Exception as e:
            self.log(f"[agent] reload after revert failed: {e}")

    # ---- account management -----------------------------------------

    def on_account_open(self):
        dpg.configure_item("account_modal", show=True)
        self.on_account_refresh()

    def _account_busy(self, msg):
        dpg.set_value("account_status_txt", msg)

    def on_account_refresh(self):
        self._account_busy("checking account...")
        threading.Thread(target=lambda: self.q.put(
            ("account", self.agent.auth_status())), daemon=True).start()

    def on_account_login(self):
        ok, msg = self.agent.login_interactive()
        self._account_busy(msg)
        self.log("[account] " + msg)

    def on_account_logout(self):
        self._account_busy("logging out...")

        def worker():
            ok, msg = self.agent.logout()
            self.q.put(("account", self.agent.auth_status()))
            self.q.put(("log", "[account] logout: " + msg))

        threading.Thread(target=worker, daemon=True).start()

    def on_account_apply(self):
        val = (dpg.get_value("account_override_in") or "").strip()
        if not val:
            self._account_busy("paste a key or token first, then Apply.")
            return
        kind = "api_key" if dpg.get_value("account_kind") == "API key" \
            else "oauth"
        self.agent.set_override(kind, val)
        self._refresh_agent_status()
        self.on_account_refresh()
        self.log(f"[account] app override set ({kind}); "
                 f"global login untouched")

    def on_account_clear(self):
        self.agent.set_override(None, None)
        dpg.set_value("account_override_in", "")
        self._refresh_agent_status()
        self.on_account_refresh()
        self.log("[account] app override cleared")

    def _refresh_agent_status(self):
        online = self.agent.backend != "none"
        if dpg.does_item_exist("agent_dot"):
            dpg.configure_item("agent_dot",
                               color=C_GREEN if online else C_RED)
        if dpg.does_item_exist("agent_status"):
            dpg.set_value("agent_status", self.agent.backend_label())

    def _on_account_status(self, result):
        ok, text = result
        dpg.set_value("account_status_txt",
                      ("[signed in]  " if ok else "[not signed in]  ") + text)

    # =================================================================
    # misc
    # =================================================================

    def _on_dir_picked(self, sender, app_data):
        path = app_data.get("file_path_name") or ""
        if path and os.path.isdir(path):
            self.workdir = os.path.abspath(path)
            self.agent.workdir = self.workdir
            self.selected_va = None
            self.rescan_va()
            self.log(f"[workspace] -> {self.workdir}")

    def _about(self):
        self.log("NeuroVAT - neuromorphic Verilog-A tester. Behavioral Python "
                 "twins of the workspace .va models, neuromorphic stimulus "
                 "designer, and an embedded Claude agent. F5 runs; export "
                 "CSVs for Spectre comparison.")
        dpg.set_value("center_tabs", "tab_log")

    def _busy(self, on, msg):
        dpg.configure_item("busy_ind", show=on)
        dpg.configure_item("btn_run", enabled=not on)
        if dpg.does_item_exist("btn_stdp"):
            dpg.configure_item("btn_stdp", enabled=not on)
        self._set_status(f"● {msg}", C_AMBER if on else C_GREEN)

    def log(self, msg):
        stamp = time.strftime("%H:%M:%S")
        t = dpg.add_text(f"{stamp}  {msg}", parent="console",
                         wrap=1200, color=C_TEXT2)
        if "mono" in self.fonts:
            dpg.bind_item_font(t, self.fonts["mono"])
        kids = dpg.get_item_children("console", 1) or []
        if len(kids) > 500:
            dpg.delete_item(kids[0])
        self._scroll_bottom("console")

    def _on_resize(self):
        vw = dpg.get_viewport_client_width()
        center = max(430, vw - LEFT_W - RIGHT_W - 44)
        if dpg.does_item_exist("center_child"):
            dpg.configure_item("center_child", width=center)

    # =================================================================
    # main loop
    # =================================================================

    def _process_queue(self):
        while True:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                return
            try:
                kind = item[0]
                if kind == "log":
                    self.log(item[1])
                elif kind == "results":
                    self.sim_running = False
                    self._show_results(item[1], item[2], item[3], item[4],
                                       live=(len(item) > 5 and item[5]))
                elif kind == "stdp":
                    self._show_stdp(item[1], item[2])
                elif kind == "chat":
                    self._on_chat_done(item[1], item[2])
                elif kind == "account":
                    self._on_account_status(item[1])
                elif kind == "virtuoso":
                    self._on_virtuoso_done(item[1], item[2])
                elif kind == "virt_browse":
                    self._on_virt_browse(item[1], item[2])
            except Exception as e:
                self.log(f"[ui] error: {e!r}")

    def run(self, smoke_frames=0):
        self.build()
        frame = 0
        while dpg.is_dearpygui_running():
            self._process_queue()
            self._tick_zoom_anim()
            self._hide_tip_if_stale()
            self._watch_code()
            dpg.render_dearpygui_frame()
            frame += 1
            if smoke_frames and frame >= smoke_frames:
                break
        self.virtuoso.disconnect()
        dpg.destroy_context()


def main(workdir=None, smoke_frames=0):
    App(workdir or os.getcwd()).run(smoke_frames=smoke_frames)
