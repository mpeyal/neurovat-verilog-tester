"""NeuroVAT - neuromorphic Verilog-A model tester (DearPyGui).

Application coordinator for the native Device Lab and Network Studio.
Plot interaction, network rendering and data controls live in focused UI
modules. This class owns application state, worker dispatch and lifecycle.
"""

import contextlib
import dataclasses
import glob
import importlib
import json
import math
import os
import queue
import re
import shutil
import sys
import threading
import time

import dearpygui.dearpygui as dpg
import numpy as np

from simulation_core import Waveform, simulate
from . import signal_factory as sf
from . import neuro
from . import models_catalog
from .agent import ClaudeAgent
from .textshape import TextShaper, needs_shaping
from .va_scan import scan as va_scan
from .virtuoso import VirtuosoLink
from . import design_system as ds
from .ui_constants import (
    APP_TITLE, LEFT_W, RIGHT_W, BUBBLE_W, BUBBLE_INDENT,
    C_BG, C_SURF, C_SURF2, C_SURF3, C_BORDER, C_ACC, C_ACC_H, C_ACC_A,
    C_TEXT, C_TEXT2, C_MUTED, C_GREEN, C_AMBER, C_RED, C_AGENT,
    C_BUB_USER, C_BUB_USER_B, C_BUB_AGENT, C_BUB_AGENT_B, C_BUB_ERR, C_BUB_ERR_B,
    UNIT_SETS, UNIT_FACTOR, UNIT_DIM, CANON_UNIT,
    STDP_MODES, POLAR_MEASUREMENTS, POLAR_DESC, CLASS_SETTING_TAGS,
)
# Re-export shared registries for existing scripts importing vatester.app.
from .registry import (
    ModelSpec, MODEL_SPECS, SPEC_BY_KEY, DEVICE_FAMILIES, DEVICE_OF_KEY,
    DEVICE_KIND, TWIN_FILE, LABEL_TO_KEY, RELOAD_MODULES, WATCH_FILES,
    BUILTIN_PATSETS, PATTERN_PLUGINS, register_pattern_plugins,
    register_dynamic_twins, _register_twin, _defaults_of, _reload_module, _m_v1,
)
from .plot_interaction import PlotInteractionMixin
from .network_visualization import NetworkVisualizationMixin
from .network_data import NetworkDataMixin
from .network_stream import NetworkStreamMixin, STREAM_CONTROLS, STREAM_MODE
from . import analysis, stdp_protocol
from .stdp_panel import StdpControls


# (display label, CLI/API model id or None for the user's default, blurb)
# These are only the OFFLINE FALLBACK - the picker is populated at runtime from
# models_catalog.list_models(), which asks each provider what it actually
# offers today, so a model released after this build still shows up. See
# App._refresh_models_async / self.model_choices.
MODEL_CHOICES = models_catalog.BUILTIN["claude"]
OPENAI_MODEL_CHOICES = models_catalog.BUILTIN["openai"]
GEMINI_MODEL_CHOICES = models_catalog.BUILTIN["gemini"]
PROVIDER_MODELS = {"claude": MODEL_CHOICES, "openai": OPENAI_MODEL_CHOICES,
                   "gemini": GEMINI_MODEL_CHOICES}
PROVIDER_TITLES = {"claude": "Claude Agent", "openai": "OpenAI Agent",
                   "gemini": "Gemini Agent"}
PROVIDER_ORDER = ("claude", "openai", "gemini")

# ---- Neuromorphic Trainer agent: the controls it may set + its instructions --
# (tag, one-line meaning) - listed to the agent with current values so it can
# tune the studio via the nt_action "set" block.
NT_CONTROLS = [
    ("nt_gh", "input grid height"), ("nt_gw", "input grid width"),
    ("nt_nout", "output neurons / classes"),
    ("nt_hidden", "hidden layer sizes, comma list e.g. '8' or '8,4' ('' = none)"),
    ("nt_hidden_gain", "extra drive on hidden layers (raise if hidden quiet)"),
    ("nt_mode", "supervised | unsupervised | unsupervised stream - sets the "
                "whole Trainer window; the nt_st_* controls belong to the last"),
    ("nt_patset", "bars|letters|digits|random|nand|xor|<plugins>|custom|dataset"),
    ("nt_learnrule", "STDP (device-local) | Surrogate grad (BPTT)"),
    ("nt_sg_lr", "surrogate learning rate (0.05-0.2)"),
    ("nt_epochs", "training epochs"), ("nt_present", "ms per pattern"),
    ("nt_rate", "peak input rate Hz"), ("nt_seed", "rng seed"),
    ("nt_epsp", "excitatory drive gain (activity-normalised)"),
    ("nt_ipsp", "inhibitory gain"), ("nt_inhib", "lateral WTA strength"),
    ("nt_theta", "homeostatic threshold bump"), ("nt_teacher", "teacher drive"),
    ("nt_tau_m", "membrane tau ms"), ("nt_vth", "spike threshold"),
    ("nt_potamp", "potentiation pulse amp (device unit)"),
    ("nt_depamp", "depression pulse amp"), ("nt_pwidth", "pulse width ms"),
    ("nt_offset", "LTP/LTD pre-trace split"),
    ("nt_encoding", "rate (Poisson) | latency (TTFS) | temporal (phase)"),
    ("nt_bg_rate", "background firing Hz"), ("nt_input_noise", "sensor noise 0-1"),
    ("nt_signal_frac", "fraction of afferents carrying the pattern 0-1"),
    ("nt_jitter", "spike jitter ms"), ("nt_vnoise", "membrane noise sigma"),
    ("nt_pat_ms", "pattern-active ms within the window (rest = noise)"),
    ("nt_wnoise", "device write noise sigma_c2c"),
    ("nt_ds_res", "dataset downsample resolution"),
    ("nt_ds_perclass", "dataset images per class"),
    ("nt_dt", "integration time step ms"),
    ("nt_refrac", "refractory period ms"), ("nt_tausyn", "synaptic time constant ms"),
    ("nt_aplus", "potentiation scale"), ("nt_aminus", "depression scale"),
    ("nt_taupre", "pre-synaptic trace time constant ms"),
] + STREAM_CONTROLS


NT_AGENT_SYSTEM = """
=== NEUROMORPHIC TRAINER (the "Network Studio" workspace in this same app) ===
This app ALSO has a Neuromorphic Trainer: a spiking neural network whose synapses
are the selected device (a crossbar of ECFET/FeFET twins).  These nt_action
blocks are IN ADDITION to the device-plot actions above (run / plot_stdp / sweep
/ waveform) - use whichever the user's request needs.  When the user asks about
TRAINING, the network, accuracy, the crossbar, neurons or learning, help them
build, train and DEBUG it: read the live state below, change controls, run
build/train/evaluate, and explain what is working or failing.

YOU CAN SEE EVERY PLOT'S DATA (in the LIVE TRAINER STATE below and, in full, in
results/neuro_snapshot.json which you may Read): the crossbar weights, the
all-synapse weight trajectories over training ("All weights" tab), the
accuracy-vs-epoch curve, the confusion matrix + per-class precision/recall/F1,
the last presentation's output spikes, and the synapse open in the Cell
inspector. Talk the user through any of them.

YOU CAN CHANGE, with the user's permission (the "Autonomous: edit + run + fix"
toggle - it is reported under WHAT YOU MAY DO below):
  * the NETWORK / NEURONS / any parameter -> via nt_action set/build/train/test;
  * the DEVICE PHYSICS itself -> edit the selected synapse's Verilog-A (.va) or
    its Python twin (twins/ or simulation_core/model_*.py) with your normal file
    tools, then rebuild (nt_action build) so the crossbar uses the new physics.
When Autonomous is OFF you are advise-only: PROPOSE the nt_action blocks / code
edits and explain them, but they are not applied until the user allows it. Never
claim you changed something you were not permitted to change.

Two learning rules: "STDP (device-local)" (local, biologically/hardware-faithful;
weak on deep nets) and "Surrogate grad (BPTT)" (supervised backprop-through-time,
gradient applied THROUGH the device; trains deep/multi-layer nets to high
accuracy). Multiple crossbars: set nt_hidden to a comma list (e.g. "8,4").

TO ACT, put one or more fenced json blocks in your reply (the GUI runs them in
order, top to bottom):
```json
{"type":"nt_action","action":"set","params":{"nt_learnrule":"Surrogate grad (BPTT)","nt_hidden":"8","nt_sg_lr":0.12,"nt_epochs":40}}
```
```json
{"type":"nt_action","action":"build"}
```
```json
{"type":"nt_action","action":"train"}
```
"action" is one of: "set" (with "params": {nt_tag: value, ...}), "build"
(construct the network from the controls), "train" (train + live-evaluate),
"test" (evaluate now). A typical tuning turn: set the controls you want, then
build, then train. After an agent-triggered train finishes the GUI feeds you the
fresh metrics automatically - then diagnose and, if it can be better, change a
control and train again. Stop when it is good or you have no better idea.

DIAGNOSIS GUIDE: low TRAIN accuracy = underfitting (try Surrogate rule, raise
nt_sg_lr/epochs, raise nt_epsp or nt_hidden_gain if neurons are quiet, simplify
the task); high train but low TEST = overfitting (more images/class, add
nt_input_noise, fewer epochs, smaller hidden); one class with low F1 / a hot
off-diagonal cell in the confusion matrix = those classes are confused (more
contrast/among them, more epochs, check encoding); hidden layers silent = raise
nt_hidden_gain.

THE FULL LEVER SET - do NOT just keep raising nt_epochs. Epochs give diminishing
returns once the accuracy-vs-epoch curve has flattened; adding more then is
mostly wasted compute. When accuracy plateaus, change a STRUCTURAL lever and
rotate through them instead of repeating one:
  * ARCHITECTURE: nt_hidden (add/size hidden crossbars) - but hidden layers only
    help under nt_learnrule="Surrogate grad (BPTT)"; with STDP the local rule
    cannot train THROUGH stacked crossbars, so add hidden layers ONLY together
    with the Surrogate rule. nt_hidden_gain drives quiet hidden layers.
  * INPUT SIZE / AFFERENTS: the input is an nt_gh x nt_gw pixel grid = nt_gh*nt_gw
    afferent channels. For a DATASET, nt_ds_res sets that resolution (and the
    grid) - raise it for more detail (more afferents, more compute), lower it to
    cut overfitting / speed up. For built-in patterns set nt_gh/nt_gw directly.
    Changing input size needs a rebuild.
  * LEARNING RULE: STDP (hardware-faithful, shallow) vs Surrogate grad (deep,
    high accuracy) - switching rule is often the biggest single lever.
  * ENCODING / FRONT-END: nt_encoding (rate / latency / temporal), nt_rate,
    nt_present,
    nt_signal_frac, nt_jitter, nt_input_noise.
  * DEVICE PROGRAMMING: nt_potamp / nt_depamp / nt_pwidth (write strength), or
    edit the .va / twin physics itself.
  * DATA: images/class, class balance.
Before each retrain, name WHICH lever you are moving and WHY from the metrics;
if you already raised epochs last round and it plateaued, move a different lever
this round. Keep replies short and concrete; say what you changed and why.
Use exact combo strings for nt_learnrule / nt_mode / nt_encoding / nt_patset.

STREAM MODE (nt_mode = "unsupervised stream"; Masquelier-style learning): one
continuous spike stream, no epochs and no labels. Hidden 50 ms patterns recur
in background activity; STDP acts at every spike and lateral inhibition lets
each neuron take one pattern. Setting nt_mode switches the whole Trainer
window; in this mode the network is set by the nt_st_* controls (not the pixel
grid), "build" starts a new network, "train" runs nt_st_seconds with STDP on,
continuing from the current weights, and "test" plays a new stream around the
same patterns with STDP off ("stream" / "stream_test" do the same and switch
the mode for you). Change nt_st_jitter /
nt_st_noise / nt_st_delete / nt_st_share between a stream and a stream_test to
measure robustness; changing any other nt_st_* control or nt_seed starts a new
network. nt_st_aplus / nt_st_aratio / nt_st_tplus / nt_st_tminus / nt_st_w0
(STDP), nt_st_taum / nt_st_taus / nt_st_refr (neuron) and nt_st_inh /
nt_st_tinh (lateral inhibition) are the rule's constants; their defaults are
the validated paper values, so change them only with a stated reason.
A run is good when every pattern has its own neuron (about one spike
per window of that pattern, none elsewhere) and false alarms are below 1 /s.
With nt_st_synapse = "selected device" the weights are device conductances.
nt_st_pulse = "calibrated" searches the pulse amplitudes so that one pulse of
width nt_pwidth moves a device by one STDP step; "as set" uses nt_potamp /
nt_depamp as they are. Whether a given device learns is an open question:
report what you see. After a stream run you started, its result is sent to you.
"""


GEN_BY_NAME = {g.name: g for g in sf.GENERATORS}


class App(PlotInteractionMixin, NetworkVisualizationMixin, NetworkDataMixin,
          NetworkStreamMixin):
    """Own shared UI state and dispatch queued worker results on the render thread.

    Analysis modules are called through module attributes so hot reload takes
    effect immediately. UI components themselves require an application restart.
    """

    def __init__(self, workdir, bridge=None):
        self.workdir = os.path.abspath(workdir)
        # control bridge (skillbridge-style): let an outside driver operate this
        # running app via a file mailbox. ON by default; toggle live from the
        # Tools menu, or launch with --no-bridge / NVAT_BRIDGE=0 to start off.
        if bridge is None:
            bridge = os.environ.get("NVAT_BRIDGE", "1") != "0"
        self._bridge_enabled = bool(bridge)
        self.control = None
        self._log_lines = []            # plain-text mirror of the console (copyable)
        # register user/agent twins from the separate twins/ folder BEFORE the
        # model registry is consumed below
        self.dynamic_twins, self.twin_errors = register_dynamic_twins(self.workdir)
        # input-pattern plugins (patterns/ folder, hot-watched - see _nt_watch_patterns)
        self.pattern_plugins, self.pattern_errors = register_pattern_plugins(self.workdir)
        self._pat_mtimes = None
        self._pat_prev_mtimes = None
        self.q = queue.Queue()
        self.agent = ClaudeAgent(self.workdir)
        self._agent_turns = 0          # agent replies this session (for /cost)
        # HarfBuzz shaper for complex-script chat (Bangla, etc.); falls back to
        # plain text if uharfbuzz/freetype/a covering font are unavailable.
        # Sized to sit with the smaller chat body font (FreeType rasterises a
        # touch larger than DearPyGui at the same nominal size).
        self.shaper = TextShaper(size=13)
        self._chat_tex = []            # texture tags for shaped chat images
        self._tex_n = 0
        self.va_files = []
        self.selected_va = None
        self.editor_path = None
        self.editor_mtime = 0.0
        self.editor_remote = None   # (lib, cell, view) when the buffer is remote
        self.device_class = "ECFET"  # ECFET (dG, current) | FeFET (dVt, voltage)
        self._untwinned_prompted = None   # last set of .va shown in the prompt
        self.sweep_specs = []      # [{param,type,...}] parametric sweep specs
        self.SWEEP_CAP = 24        # max overlaid curves (Cartesian product)
        self.legends_visible = True   # show/hide all plot legends (toolbar toggle)
        self.legend_outside = False   # legend inside (False) / outside (True) canvas
        self.legend_horizontal = False  # legend layout: vertical (False) / horizontal
        self.legend_location = None   # ImPlot anchor (None = ImPlot default corner)
        self._plot_menus = []         # [{plot,menu,xaxis,yaxis}] right-click menus
        self._open_menu = None        # tag of the currently shown plot context menu
        self._menu_unfocused = 0      # frames the open context menu has lost focus
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
        self._last_compute = "transient"  # "transient" | "stdp" (live re-run)
        self._restart = False       # set to relaunch and apply GUI-code edits
        self._series = []
        self._ana_series = []
        self.sim_running = False
        self.chat_busy = False
        self.virtuoso = VirtuosoLink()
        self.virt_busy = False
        # per-provider model selection: provider -> (label, id-or-None).
        # Claude stays explicitly pinned (the CLI's own default may pick the
        # priciest model); the id is checked against the live list on refresh
        # and falls back to "Default" if the provider no longer offers it, so
        # pinning here can't go stale the way the old hardcoded list did.
        self.model_sel = {p: ("Default", None) for p in PROVIDER_ORDER}
        self.model_sel["claude"] = ("Opus 5", "claude-opus-5")
        # live model lists, provider -> [(label, id, blurb)]. Seeded with the
        # offline fallback and replaced by _refresh_models_async once each
        # provider has been asked what it actually offers.
        self.model_choices = {p: list(PROVIDER_MODELS[p])
                              for p in PROVIDER_ORDER}
        self.model_source = {p: "builtin" for p in PROVIDER_ORDER}
        self.models_busy = False
        self.attachments = []      # absolute paths queued for the next message
        self.fonts = {}
        self.themes = {}
        self._zoom_anim = {}       # axis -> [cur_lo, cur_hi, tgt_lo, tgt_hi]
        self._zoom_release = []    # axes to unlock next frame
        self._probes = {}          # plot -> {"A"/"B": {pt, ann, sid, x, y}}
        self._probe_armed = None   # "A"/"B" while waiting for placement click
        self._dragging = None      # (plot, which) while a probe is dragged
        self._drag_anchor = None   # (gx, gy, mx, my, ux, uy) at grab time
        self._hover_hist = {}      # plot -> [(px, py, mx, my)] for px->unit
        self._pan_disabled = set() # plots with pan parked (cursor on a probe)
        self._stdp_series = []
        self._polar_series = []
        self._stdp_ctx = None
        self._class_settings = {}      # device class -> its saved GUI settings
        self._builtin_settings = None  # the tab defaults as first built
        self._stdp_summary = []
        self._stdp_dts_ms = []     # last sweep's dt points (ms), for copy menu
        self._stdp_curves = {}     # last sweep's {model: [dG_uS]}, for copy menu
        self._tip_frame = -10
        self._hover_ann = {}       # plot -> in-canvas hover bubble annotation
        self._hover_cache = {}     # plot -> [(label, xs, ys)] for fast hover
        self._hover_last = None    # (plot, mx, my) of last computed bubble
        # ---- Neuromorphic Trainer studio (crossbar of device synapses) ----
        self.trainer = None             # neuro.Trainer instance
        self._nt_loaded_patterns = None  # (pats,tgts) injected when loading a model
        self.trainer_running = False
        self._trainer_stop = False
        self._nt_dev = None             # {make, kind, label} synapse factory
        self._nt_loaded_model = False   # the built network holds loaded weights
        self._nt_loading = False        # inside on_nt_load
        self._rf_tex = None             # receptive-field montage raw texture
        self._rf_layout = None          # (n_out, gh, gw, cell, gap, tw, th)
        self._nt_series = {}            # group -> [series tags] to clear
        self._nt_diag_size = (900, 300)
        self._nt_zoom = 1.0             # canvas zoom (1 = fit; >1 scrolls)
        self._last_present = None       # (label, vec, result) for the live views
        self._wevo = {}                 # neuron -> [mean weight uS] learning curve
        self._wevo_epochs = []
        self._cmap_lut_cache = None
        self._nt_win_rect = None        # last (w, h) seen, to detect tab resize
        self._nt_spk_series = {}        # neuron -> training output-spike series
        self._nt_spk_pts = {}           # neuron -> ([pres_idx], [neuron_row])
        self._nt_pres_idx = 0           # training presentation counter
        self._nt_paint = None           # custom-pattern paint grid (gh x gw)
        self._nt_paint_gh = 0
        self._nt_paint_gw = 0
        self._nt_custom = []            # [(label, grid01)] painted patterns
        self._nt_ds = None              # loaded dataset {images, labels, names}
        self._nt_busy_load = False
        self._nt_xbar_sel_idx = 0       # which crossbar the Weights tab shows
        self._nt_xbar_labels = []
        self._nt_wuS_all = None         # last snapshot's per-crossbar weights uS
        self._nt_pending_test = None    # held-out test set staged at build
        self._nt_acc_hist = {"epoch": [], "train": [], "validation": []}
        self._nt_cm_series = None       # confusion-matrix heat series
        self._nt_last_metrics = None    # last eval summary (for the agent)
        self._nt_agent_run = False      # an agent-triggered train is in flight
        self._nt_agent_rounds = 0       # auto-analyze loop guard
        self._nt_best_acc = 0.0         # best accuracy the auto-loop has reached
        self._nt_best_seen = 0.0        # best eval acc for the BEST stat card
        self._nt_no_improve = 0         # rounds since the best improved (patience)
        self._nt_last_patset = "bars"   # last built-in set (for DATA SOURCE cards)
        self._nt_units = {}             # tag -> selected display unit (see _nt_getu)
        self._nt_cell = None            # selected synapse (c, j, i) to inspect
        self._nt_cell_hits = []         # [(c, j, i, x, y)] clickable cells on canvas
        self._nt_anim_t = 0.0           # cell-inspector animation clock (s)
        self._nt_whist = {"epoch": [], "W": []}   # per-epoch weights (cell trace)
        self._nt_last_kind = "current"  # drive kind of the selected synapse device
        self._nt_trace_themes = {}      # (c,j) -> per-neuron line theme

    # =================================================================
    # fonts & themes
    # =================================================================

    def _font(self):
        fdir = r"C:\Windows\Fonts"
        files = {
            "body":  ("segoeui.ttf", 17),
            "small": ("segoeui.ttf", 14),
            "chat_en": ("segoeui.ttf", 16),    # English chat body
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
            # Segoe UI has no Indic glyphs, so non-Latin agent replies (e.g.
            # Bangla) render as boxes. Nirmala UI (ships with Windows) covers
            # Latin + Bengali + Devanagari; load it with those ranges for the
            # chat text so replies in those scripts display correctly.
            self._load_chat_font(fdir)
            # Segoe MDL2 Assets: window-control icons (minimize / maximize /
            # restore / close) for the studio's custom title-bar buttons.
            # DPG 2.x auto-loads the font's full glyph set (incl. the PUA icons).
            icon_path = os.path.join(fdir, "segmdl2.ttf")
            if os.path.isfile(icon_path):
                self.fonts["icons"] = dpg.add_font(icon_path, 15)
        if "body" in self.fonts:
            dpg.bind_font(self.fonts["body"])

    def _load_chat_font(self, fdir):
        """Font for chat-message text that also covers Bengali/Devanagari.
        Nirmala UI (ships as Nirmala.ttc on most Windows) covers Latin + Indic;
        Kalpurush is a Bengali-focused fallback. DearPyGui 2.x auto-loads every
        glyph the file contains, so no explicit Unicode ranges are needed."""
        for fn in ("Nirmala.ttc", "Nirmala.ttf", "NirmalaS.ttf",
                   "kalpurush.ttf"):
            path = os.path.join(fdir, fn)
            if not os.path.isfile(path):
                continue
            try:
                self.fonts["chat"] = dpg.add_font(path, 17)
                return
            except Exception as e:                       # noqa: BLE001
                print(f"[font] {fn} load failed: {e}")

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
                ("mvThemeCol_PopupBg", C_SURF),
                ("mvThemeCol_TitleBg", C_SURF),
                ("mvThemeCol_TitleBgActive", C_SURF2),
                ("mvThemeCol_TitleBgCollapsed", C_SURF),
                ("mvThemeCol_ModalWindowDimBg", (5, 10, 16, 175)),
                ("mvThemeCol_MenuBarBg", C_BG),
                ("mvThemeCol_FrameBg", C_SURF2),
                ("mvThemeCol_FrameBgHovered", C_SURF3),
                ("mvThemeCol_FrameBgActive", (45, 51, 65)),
                ("mvThemeCol_Button", C_SURF2),
                ("mvThemeCol_ButtonHovered", C_SURF3),
                ("mvThemeCol_ButtonActive", (49, 56, 72)),
                # accent-tinted section-header bars (Network / Neuron / ...)
                ("mvThemeCol_Header", C_SURF2),
                ("mvThemeCol_HeaderHovered", C_SURF3),
                ("mvThemeCol_HeaderActive", (38, 77, 74)),
                # the SELECTED tab pops in accent-blue; the rest stay quiet
                ("mvThemeCol_Tab", (0, 0, 0, 0)),
                ("mvThemeCol_TabHovered", C_SURF3),
                ("mvThemeCol_TabActive", (35, 69, 68)),
                ("mvThemeCol_TabUnfocused", (0, 0, 0, 0)),
                ("mvThemeCol_TabUnfocusedActive", (34, 40, 54)),
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
                ("mvThemeCol_TextSelectedBg", (46, 94, 91)),
            ],
            styles=[
                ("mvStyleVar_WindowRounding", 10),
                ("mvStyleVar_ChildRounding", 8),
                ("mvStyleVar_FrameRounding", 5),
                ("mvStyleVar_PopupRounding", 8),
                ("mvStyleVar_GrabRounding", 4),
                ("mvStyleVar_TabRounding", 5),
                ("mvStyleVar_ScrollbarRounding", 8),
                # tighter spacing -> denser / more compact rows
                ("mvStyleVar_WindowPadding", (14, 12)),
                ("mvStyleVar_FramePadding", (10, 5)),
                ("mvStyleVar_CellPadding", (7, 2)),
                ("mvStyleVar_ItemSpacing", (10, 6)),
                ("mvStyleVar_ItemInnerSpacing", (6, 3)),
                ("mvStyleVar_ScrollbarSize", 10),
                ("mvStyleVar_ChildBorderSize", 1),
                ("mvStyleVar_WindowBorderSize", 1),
            ],
            plot_colors=[
                ("mvPlotCol_FrameBg", (0, 0, 0, 0)),
                ("mvPlotCol_PlotBg", C_BG),
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
            ("mvThemeCol_Text", ds.ON_ACCENT),
        ], styles=[("mvStyleVar_FrameRounding", 5),
                   ("mvStyleVar_FramePadding", (14, 5))])

        self._mk_theme("stop", colors=[
            ("mvThemeCol_Button", (66, 40, 47)),
            ("mvThemeCol_ButtonHovered", (85, 47, 55)),
            ("mvThemeCol_ButtonActive", (109, 54, 64)),
            ("mvThemeCol_Text", C_RED),
        ], styles=[("mvStyleVar_FrameRounding", 8),
                   ("mvStyleVar_FramePadding", (14, 5))])

        self._mk_theme("chip", colors=[
            ("mvThemeCol_Button", (35, 40, 52)),
            ("mvThemeCol_ButtonHovered", (47, 54, 70)),
            ("mvThemeCol_ButtonActive", (56, 64, 84)),
            ("mvThemeCol_Text", C_TEXT2),
        ], styles=[("mvStyleVar_FrameRounding", 5),
                   ("mvStyleVar_FramePadding", (10, 4))])

        # flat window-control buttons (minimize / maximize); close goes red
        self._mk_theme("winctrl", colors=[
            ("mvThemeCol_Button", (0, 0, 0, 0)),
            ("mvThemeCol_ButtonHovered", (60, 68, 86)),
            ("mvThemeCol_ButtonActive", (74, 84, 106)),
            ("mvThemeCol_Text", C_TEXT2),
        ], styles=[("mvStyleVar_FrameRounding", 3),
                   ("mvStyleVar_FramePadding", (9, 5))])
        self._mk_theme("winctrl_close", colors=[
            ("mvThemeCol_Button", (0, 0, 0, 0)),
            ("mvThemeCol_ButtonHovered", (200, 70, 76)),
            ("mvThemeCol_ButtonActive", (220, 80, 86)),
            ("mvThemeCol_Text", C_TEXT),
        ], styles=[("mvStyleVar_FrameRounding", 3),
                   ("mvStyleVar_FramePadding", (9, 5))])

        self._mk_theme("card", colors=[
            ("mvThemeCol_ChildBg", C_SURF2),
            ("mvThemeCol_Border", C_BORDER),
        ], styles=[("mvStyleVar_ChildRounding", 9),
                   ("mvStyleVar_WindowPadding", (10, 8))])

        # a DATA SOURCE card that is the selected one: accent border + tint
        self._mk_theme("card_active", colors=[
            ("mvThemeCol_ChildBg", (28, 53, 54)),
            ("mvThemeCol_Border", C_ACC),
        ], styles=[("mvStyleVar_ChildRounding", 9),
                   ("mvStyleVar_WindowPadding", (10, 8)),
                   ("mvStyleVar_ChildBorderSize", 2)])

        # raster preview dots: tiny circles, clean = accent blue, noisy = amber
        for _nm, _col in (("dot_clean", C_ACC), ("dot_noise", C_AMBER)):
            with dpg.theme() as _th:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, _col,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, _col,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker,
                                        dpg.mvPlotMarker_Circle,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 1.6,
                                        category=dpg.mvThemeCat_Plots)
            self.themes[_nm] = _th

        # explicit line-series colours for the neuron probe plots
        for _nm, _col in (("ln_amber", C_AMBER), ("ln_blue", C_ACC),
                          ("ln_green", C_GREEN), ("ln_muted", (150, 160, 180))):
            with dpg.theme() as _th:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, _col,
                                        category=dpg.mvThemeCat_Plots)
            self.themes[_nm] = _th

        # right-click context menu on the STDP plot: rounded dark popup with
        # full-width hover-highlighted rows
        self._mk_theme("stdp_menu", colors=[
            ("mvThemeCol_WindowBg", (24, 27, 36)),
            ("mvThemeCol_Border", (70, 96, 168)),
            ("mvThemeCol_Header", (40, 54, 92)),
            ("mvThemeCol_HeaderHovered", (52, 70, 118)),
            ("mvThemeCol_HeaderActive", (60, 80, 132)),
            ("mvThemeCol_Text", C_TEXT),
        ], styles=[("mvStyleVar_WindowRounding", 10),
                   ("mvStyleVar_WindowBorderSize", 1),
                   ("mvStyleVar_WindowPadding", (10, 10)),
                   ("mvStyleVar_FrameRounding", 6),
                   ("mvStyleVar_ItemSpacing", (6, 5))])

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

        # line-series-with-markers: bound to STDP curves so a single legend
        # entry controls BOTH the connecting line and the point markers (the
        # marker inherits the line's auto-assigned colour).
        with dpg.theme() as _mk:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker,
                                    dpg.mvPlotMarker_Circle,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 3.0,
                                    category=dpg.mvThemeCat_Plots)
        self.themes["markers"] = _mk
        ds.install(self)

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
        self._small(label.upper(), color=C_TEXT2)
        dpg.add_separator()

    def _caption(self, text, tip=None):
        t = self._small(text, color=C_MUTED)
        if tip:                                  # hover the LABEL for help too
            with dpg.tooltip(t):
                dpg.add_text(tip, wrap=340)
        return t

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

    # =================================================================
    # UI construction
    # =================================================================

    def build(self):
        dpg.create_context()
        self._font()
        self._theme()
        from vatester.desktop_layout import DesktopLayout
        self.desktop = DesktopLayout(self)
        from .network_studio import NetworkStudio
        self.network_studio = NetworkStudio(self, MODEL_SPECS)
        dpg.add_texture_registry(tag="chat_tex_reg")   # shaped-chat textures

        with dpg.window(tag="main", no_scrollbar=True, no_scroll_with_mouse=True):
            self._menu()
            self._toolbar()
            with dpg.group(horizontal=True, tag="workspace_body"):
                self._left_panel()
                self._center_panel()
                self._right_panel()

        with dpg.handler_registry():
            dpg.add_key_press_handler(key=dpg.mvKey_F5, callback=self._run_active)
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
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,
                                          callback=self._on_probe_release)
            # ImPlot swallows a plot's own right-click, so we open each plot's
            # context menu explicitly from this global handler.
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right,
                                        callback=self._on_plot_rclick)

        dpg.add_file_dialog(directory_selector=True, show=False, modal=True,
                            callback=self._on_dir_picked, tag="dir_dialog",
                            width=720, height=420, default_path=self.workdir)

        self._bind_plot_hover_handlers()

        with dpg.window(tag="virt_modal", label="Virtuoso connection",
                        modal=True, show=False, no_resize=True,
                        no_collapse=True, no_title_bar=True, width=560, autosize=True):
            with dpg.group(horizontal=True):
                dpg.add_text("●", tag="virt_modal_dot", color=C_GREEN)
                t = dpg.add_text("", tag="virt_modal_title", color=C_TEXT)
                if "h2" in self.fonts:
                    dpg.bind_item_font(t, self.fonts["h2"])
            dpg.add_text("", tag="virt_modal_text", wrap=440, color=C_TEXT2)
            dpg.add_spacer(height=4)
            ds.footer(self, [("Done", lambda: dpg.hide_item("virt_modal"), True, None)])
        dpg.bind_item_theme("virt_modal", self.themes["dialog"])

        with dpg.window(tag="account_modal", label="Claude account",
                        modal=True, show=False, no_resize=True,
                        no_collapse=True, width=620, height=640):
            with dpg.child_window(tag="account_body", height=-100, border=False):
                dpg.bind_item_theme("account_body", self.themes["dialog_body"])
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
                    dpg.add_button(label="Get Claude Code",
                                   callback=self.on_account_install)
                self._small("Sign in with Claude Code for assistant actions. "
                            "An API key provides chat-only access.", wrap=490)
                dpg.add_separator()
                self._small("APP-ONLY CREDENTIALS", color=C_TEXT2)
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
                self._small("Credentials stay in memory for this session. "
                            "Your global login is unchanged.", wrap=490)
                dpg.add_separator()
                self._small("OPENAI API KEY", color=C_TEXT2)
                with dpg.group(horizontal=True):
                    dpg.add_input_text(tag="openai_key_in", password=True,
                                       width=304, hint="paste sk-...")
                    bo = dpg.add_button(label="Apply",
                                        callback=self.on_openai_key_apply)
                    dpg.bind_item_theme(bo, self.themes["primary"])
                    dpg.add_button(label="Clear",
                                   callback=self.on_openai_key_clear)
            ds.footer(self, [("Done", lambda: dpg.hide_item("account_modal"), False, "account_done")])
        dpg.bind_item_theme("account_modal", self.themes["dialog"])

        with dpg.file_dialog(directory_selector=False, show=False, modal=True,
                             width=620, height=420, tag="attach_dialog",
                             callback=self._on_attach_picked, file_count=8,
                             default_path=self.workdir):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".pdf", color=(255, 150, 130, 255))
            dpg.add_file_extension(".csv", color=(150, 220, 160, 255))
            dpg.add_file_extension(".png", color=(150, 180, 255, 255))
            dpg.add_file_extension(".jpg", color=(150, 180, 255, 255))
            dpg.add_file_extension(".jpeg", color=(150, 180, 255, 255))
            dpg.add_file_extension(".txt", color=(200, 200, 200, 255))
            dpg.add_file_extension(".md", color=(200, 200, 200, 255))
            dpg.add_file_extension(".json", color=(220, 200, 140, 255))
            dpg.add_file_extension(".va", color=(220, 170, 255, 255))

        # each plot's right-click menu is built natively next to its plot
        # (see _add_plot_menu, called from each tab)

        # "Save as..." target picker for the Verilog-A editor
        with dpg.file_dialog(directory_selector=False, show=False, modal=True,
                             width=620, height=420, tag="save_va_dialog",
                             callback=self._on_save_va_picked,
                             default_path=self.workdir,
                             default_filename="model.va"):
            dpg.add_file_extension(".va", color=(220, 170, 255, 255))
            dpg.add_file_extension(".*")

        # CSV picker for Tools > Fit LTP/LTD nonlinearity
        with dpg.file_dialog(directory_selector=False, show=False, modal=True,
                             width=620, height=420, tag="fit_csv_dialog",
                             callback=self._on_fit_csv_file,
                             default_path=self.workdir):
            dpg.add_file_extension(".csv", color=(150, 220, 160, 255))
            dpg.add_file_extension(".txt", color=(200, 200, 200, 255))
            dpg.add_file_extension(".*")

        # confirm dialog for writing source back into a Virtuoso cellview
        with dpg.window(label="Write back to Virtuoso", tag="virt_write_modal",
                        modal=True, show=False, no_resize=True, width=560,
                        no_title_bar=True, autosize=True):
            ds.heading(self, "Update Virtuoso source")
            dpg.add_text("", tag="virt_write_msg", wrap=490, color=C_TEXT2)
            ds.footer(self, [("Cancel", lambda: dpg.hide_item("virt_write_modal"), False, None),
                             ("Write to library", self._do_virt_write_back, True, None)])
        dpg.bind_item_theme("virt_write_modal", self.themes["dialog"])

        # auto-prompt: Verilog-A files that have no Python twin yet
        with dpg.window(label="Device simulation setup", tag="untwin_modal",
                        modal=True, show=False, no_resize=True, width=580,
                        no_title_bar=True, autosize=True):
            ds.heading(self, "Set up a device model")
            dpg.add_text("", tag="untwin_msg", wrap=510, color=C_TEXT2)
            self._small("VERILOG-A FILES", color=C_MUTED)
            dpg.add_listbox([], tag="untwin_list", num_items=3, width=-1)
            self._small("The assistant creates a Python twin in your workspace.", wrap=510)
            ds.footer(self, [("Not now", lambda: dpg.hide_item("untwin_modal"), False, "untwin_dismiss"),
                             ("Build selected twin", self._build_selected_twin, True, "untwin_build")])
        dpg.bind_item_theme("untwin_modal", self.themes["dialog"])

        # Parametric Sweep dialog (Virtuoso-style): manage per-variable sweep
        # specs (From/To, Center/Span%, explicit Values) in engineering units
        with dpg.window(label="Parametric Sweep", tag="sweep_dialog",
                        modal=True, show=False, no_resize=False, width=650,
                        height=590, min_size=(650, 510)):
            with dpg.child_window(tag="sweep_body", height=-100, border=False):
                dpg.bind_item_theme("sweep_body", self.themes["dialog_body"])
                self._small("SWEEP VARIABLES", color=C_ACC)
                with dpg.child_window(tag="sweep_spec_list", height=120,
                                      border=True):
                    pass
                dpg.add_separator()
                self._small("ADD / EDIT A VARIABLE", color=C_ACC)
                with dpg.group(horizontal=True):
                    self._caption("Parameter")
                    dpg.add_combo([], tag="sw_param", width=160)
                    self._caption("Sweep type")
                    dpg.add_combo(["From/To", "Center/Span%", "Values"],
                                  default_value="From/To", tag="sw_type", width=130,
                                  callback=self._on_sw_type)
                with dpg.group(tag="sw_range_row"):
                    with dpg.group(horizontal=True):
                        self._caption("From")
                        dpg.add_input_text(tag="sw_from", width=80, hint="4u")
                        self._caption("To")
                        dpg.add_input_text(tag="sw_to", width=80, hint="12u")
                        self._caption("Step type")
                        dpg.add_combo(["Linear", "Log", "Auto"],
                                      default_value="Linear", tag="sw_steptype",
                                      width=80)
                        self._caption("Total steps")
                        dpg.add_input_int(tag="sw_steps", default_value=5, width=70,
                                          step=0, min_value=2, min_clamped=True)
                with dpg.group(tag="sw_center_row", show=False):
                    with dpg.group(horizontal=True):
                        self._caption("Center")
                        dpg.add_input_text(tag="sw_center", width=80, hint="8u")
                        self._caption("Span %")
                        dpg.add_input_text(tag="sw_span", width=70, hint="50")
                        self._caption("Total steps")
                        dpg.add_input_int(tag="sw_csteps", default_value=5, width=70,
                                          step=0, min_value=2, min_clamped=True)
                with dpg.group(tag="sw_values_row", show=False):
                    with dpg.group(horizontal=True):
                        self._caption("Values")
                        dpg.add_input_text(tag="sw_values", width=-1,
                                           hint="4u, 8u, 12u")
                with dpg.group(horizontal=True):
                    ba = dpg.add_button(label="Add / Update",
                                        callback=self._on_sweep_spec_add)
                    dpg.bind_item_theme(ba, self.themes["primary"])
                    dpg.add_button(label="Delete selected",
                                   callback=self._on_sweep_spec_delete)
                self._small("Engineering notation accepted: 4u, 3m, 100k. "
                            "Multiple variables create a sweep grid.", wrap=540)
            ds.footer(self, [("Cancel", self._on_sweep_cancel, False, "sweep_cancel"),
                             ("Apply sweep", self._on_sweep_ok, True, "sweep_apply")])
        dpg.bind_item_theme("sweep_dialog", self.themes["dialog"])

        # Neuromorphic Trainer: its texture registry + dataset dialogs (the UI
        # itself is a center tab built in _center_panel)
        self._build_neuro_dialogs()

        dpg.create_viewport(title=APP_TITLE, width=1600, height=980,
                            min_width=1160, min_height=720)
        dpg.set_viewport_resize_callback(self._on_resize)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)

        self.rescan_va(startup=True)
        self.rebuild_param_panel()
        self.rebuild_gen_params()
        # device class is detected from the default-checked .va files (no radio)
        _cls = next(iter({DEVICE_OF_KEY.get(k) for k in self._enabled_keys()}
                         - {None}), self.device_class)
        self._sync_class_ui(_cls)                      # init labels + tab visibility
        self.log(f"workspace: {self.workdir}")
        self.log(f"agent backend: {self.agent.backend_label()}")
        self.log("chat script shaping (Bangla, etc.): "
                 + ("ON (HarfBuzz)" if (self.shaper and self.shaper.ok)
                    else "OFF - 'pip install uharfbuzz freetype-py' for shaped "
                         "complex scripts (falling back to unshaped glyphs)"))
        self._check_orphan_twins()        # warn about unregistered simulation_core/ twins
        self.append_chat("agent",
                         "Hi! I can generate spike patterns (Poisson, STDP, "
                         "bursts...), explain or modify the .va sources, and "
                         "help fit model parameters. Try a chip below, or "
                         "type /model to pick the model I run on.")
        dpg.set_item_callback("unit_combo", self.desktop.request_preview)
        dpg.set_item_callback("delay_ms", self.desktop.request_preview)
        dpg.set_item_callback("custom_text", self.desktop.request_preview)
        self._on_resize()
        self._on_center_tab()

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
                dpg.add_menu_item(label="Run active page  (F5)",
                                  callback=self._run_active)
                dpg.add_menu_item(label="Network Studio...",
                                  callback=self.on_open_trainer)
                dpg.add_menu_item(label="Parametric Sweep...",
                                  callback=self.on_open_sweep_dialog)
                dpg.add_menu_item(label="Fit plot axes", callback=self.fit_axes)
                dpg.add_separator()
                dpg.add_text("Analyze quantity:")
                dpg.add_menu_item(label="G (uS)", check=True,
                                  default_value=True, tag="ana_menu_0",
                                  user_data=0, callback=self._on_ana_menu)
                dpg.add_menu_item(label="R_mem (ohm)", check=True,
                                  default_value=False, tag="ana_menu_1",
                                  user_data=1, callback=self._on_ana_menu)
            with dpg.menu(label="Virtuoso", tag="menu_virtuoso"):
                dpg.add_menu_item(label="Connect (tunnel + skillbridge)",
                                  callback=self.on_virtuoso_connect)
                dpg.add_menu_item(label="Disconnect",
                                  callback=self.on_virtuoso_disconnect)
                dpg.add_menu_item(label="List libraries",
                                  callback=self.on_virtuoso_libs)
            with dpg.menu(label="Tools"):
                with dpg.menu(label="Control bridge"):
                    with dpg.group(horizontal=True):
                        dpg.add_text("Status:")
                        dpg.add_text("●", tag="bridge_dot", color=C_RED)
                        dpg.add_text("disconnected", tag="bridge_state_txt",
                                     color=C_RED)
                    bi = dpg.add_menu_item(
                        label="Connect bridge", tag="menu_bridge",
                        callback=self._on_toggle_bridge)
                    with dpg.tooltip(bi):
                        dpg.add_text(
                            "Connected: an outside driver - the agent or an\n"
                            "external Claude Code - can operate this running app via\n"
                            "the control/ file mailbox (python tools/nvat_ctl.py ...):\n"
                            "run sims, set fields, screenshot, hot-reload edited twins.\n"
                            "This is the LOCAL bridge, separate from the Cadence\n"
                            "skillbridge in the Virtuoso menu. Disconnected: mailbox closed.")
                dpg.add_separator()
                ci = dpg.add_menu_item(label="Check twin ↔ .va params",
                                       callback=self.on_check_consistency)
                with dpg.tooltip(ci):
                    dpg.add_text("Compare each .va's `parameter` values against\n"
                                 "its Python twin's defaults; flags mismatches so\n"
                                 "the twin and Verilog-A model stay in sync.")
                oi = dpg.add_menu_item(label="Compile .va (OpenVAF)",
                                       callback=self.on_openvaf_compile)
                with dpg.tooltip(oi):
                    dpg.add_text("Compile the enabled .va to OSDI with OpenVAF to\n"
                                 "catch Verilog-A errors before Cadence. Needs\n"
                                 "openvaf on PATH (or set NVAT_OPENVAF); reports\n"
                                 "gracefully if it isn't installed.")
                fi = dpg.add_menu_item(label="Fit LTP/LTD from CSV...",
                                       callback=lambda: dpg.show_item("fit_csv_dialog"))
                with dpg.tooltip(fi):
                    dpg.add_text("Load a measured conductance-per-pulse CSV (1 col\n"
                                 "G, or 2 cols pulse,G) and fit the LTP/LTD\n"
                                 "nonlinearity + dynamic range + asymmetry.")
                # (future tools go here as sibling items / submenus)
            with dpg.menu(label="Agent"):
                dpg.add_menu_item(
                    label="Autonomous: edit + run + fix (auto-approve)",
                    check=True, default_value=True, tag="menu_auto")
                with dpg.tooltip("menu_auto"):
                    dpg.add_text("On: the agent may edit files, run shell "
                                 "commands, and loop edit->simulate->verify->"
                                 "fix without per-step approval.\nOff: "
                                 "read-only - it can look but not change "
                                 "anything.")
                dpg.add_menu_item(label="Live re-plot on code change",
                                  check=True, default_value=True,
                                  tag="menu_liveplot",
                                  callback=lambda s, a:
                                  dpg.set_value("cb_liveplot", a))
                dpg.add_separator()
                dpg.add_menu_item(label="Revert agent edits",
                                  callback=self.on_agent_revert)
                ri = dpg.add_menu_item(
                    label="Restart app (apply GUI-code edits)",
                    callback=self.on_restart_app)
                with dpg.tooltip(ri):
                    dpg.add_text("Model twins (simulation_core/model_*.py) hot-reload "
                                 "live. Edits to the GUI itself "
                                 "(vatester/app.py) need a restart - this "
                                 "relaunches the app to apply them.")
                dpg.add_menu_item(label="Reset conversation",
                                  callback=self.on_agent_reset)
                dpg.add_menu_item(label="Account / sign-in...",
                                  callback=self.on_account_open)
                dpg.add_menu_item(label="Backend info",
                                  callback=lambda: self.log(
                                      "agent: " + self.agent.backend_label()))
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="About", callback=self._about)

    def _toolbar(self):
        with dpg.group(horizontal=True):
            t = dpg.add_text("NeuroVAT", color=C_TEXT)
            if "h2" in self.fonts:
                dpg.bind_item_font(t, self.fonts["h2"])
            self.desktop.header()
            run = dpg.add_button(label="Run simulation", tag="btn_run", callback=self._run_active, height=28, width=155)
            dpg.bind_item_theme(run, self.themes["primary"])
            with dpg.tooltip(run):
                dpg.add_text("Run the active simulation, experiment or network training. Shortcut: F5.")
            dpg.add_spacer(width=160, tag="brand_spacer")
            dpg.add_loading_indicator(tag="busy_ind", show=False, radius=2.0,
                                      style=1, color=C_AMBER)
            dpg.add_button(label="Ready", tag="run_status")
            self._set_status("Ready", C_GREEN)
            dpg.add_button(label="Assistant", tag="assistant_toggle", callback=self.desktop.assistant, height=28)
        dpg.add_separator()

    def _device_actions(self):
        with dpg.group(horizontal=True, tag="device_actions"):
            dpg.add_button(label="STDP experiment", tag="btn_stdp", height=32,
                           callback=lambda: self._open_device_page("tab_stdp"))
            with dpg.group(horizontal=True, tag="device_toolbar"):
                dpg.add_button(label="Fit plots", callback=self.fit_axes, height=32)
                dpg.add_button(label="Export CSV", callback=self.export_csv, height=32)
        dpg.add_spacer(height=4)

    def _run_active(self, *_):
        active = dpg.get_value("center_tabs")
        if active in ("tab_trainer", dpg.get_alias_id("tab_trainer")):
            self.on_nt_train()
        elif active in ("tab_stdp", dpg.get_alias_id("tab_stdp")):
            self.on_plot_stdp()
        elif active in ("tab_polar", dpg.get_alias_id("tab_polar")):
            self.on_plot_polar()
        elif active in ("tab_noise", dpg.get_alias_id("tab_noise")):
            self.noise_panel.refresh()
        elif active in ("tab_compute", dpg.get_alias_id("tab_compute")):
            self.compute_panel.start()
        else:
            self.on_run()

    def _open_device_page(self, tag):
        dpg.show_item(tag)
        dpg.set_value("center_tabs", tag)
        self._on_center_tab()

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
            dpg.bind_item_theme("left_child", self.themes["inspector"])
            dpg.add_spacer(height=2)

            with dpg.collapsing_header(label="Device library",
                                       default_open=True, tag="lh_va") as h:
                with self._pad():
                    with dpg.child_window(tag="va_cards", height=180,
                                          border=True) as vc:
                        pass
                    dpg.bind_item_theme(vc, self.themes["card"])
                    with dpg.tooltip(vc):
                        dpg.add_text("check = simulate   ·   click = load "
                                     "params   ·   double-click = edit source\n"
                                     "one device class at a time - checking a "
                                     "FeFET unchecks ECFET (and vice versa); "
                                     "the GUI auto-switches drive + plots")

            with dpg.collapsing_header(label="Parameter inspector",
                                       default_open=True, tag="lh_params") as h:
                with self._pad():
                    dpg.add_combo([s.label for s in MODEL_SPECS],
                                  default_value=MODEL_SPECS[1].label,
                                  tag="param_model_sel", width=-1,
                                  callback=lambda *_: self.rebuild_param_panel())
                    from vatester.desktop_layout import filter_parameters
                    dpg.add_input_text(tag="param_search", hint="Search name, description or group", width=-1,
                                       callback=lambda *_: filter_parameters(self))
                    dpg.add_checkbox(label="Modified only", tag="param_changed",
                                     callback=lambda *_: filter_parameters(self))
                    self._small("", tag="param_count", wrap=LEFT_W-40)
                    with dpg.child_window(tag="param_panel", height=300,
                                          border=False):
                        pass
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Reset defaults", callback=self.on_param_defaults)
                        dpg.add_button(label="Use in Network Studio", tag="use_device_network",
                                       callback=self.desktop.use_device)

            with dpg.collapsing_header(label="Run settings",
                                       default_open=False, tag="lh_sim") as h:
                with self._pad():
                    with dpg.table(header_row=False,
                                   policy=dpg.mvTable_SizingStretchProp) as tbl:
                        dpg.bind_item_theme(tbl, self.themes["table_roomy"])
                        dpg.add_table_column(init_width_or_weight=0.62)
                        dpg.add_table_column(init_width_or_weight=0.38)
                        with dpg.table_row():
                            dpg.add_text("post-stimulus tail (s)",
                                         color=C_TEXT2)
                            # 60 s so a single pulse's slow (19 s tail) recovery
                            # fully settles to +/-10 in view; lower for quick runs
                            dpg.add_input_double(tag="tail_input",
                                                 default_value=60.0,
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
                                       default_open=False, tag="lh_virt") as h:
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
                                         "the Virtuoso host and open a skillbridge "
                                         "workspace.\nLinux side must have "
                                         "Virtuoso open with the skill server "
                                         "running.")
                        dpg.add_button(label="Disconnect",
                                       callback=self.on_virtuoso_disconnect)
                    self._small("", tag="virt_info", wrap=LEFT_W - 36)

            # Neuro Trainer parameters - shown here (replacing the device-tester
            # sections above) while the Neuro Trainer center tab is active
            with dpg.group(tag="left_trainer", show=False):
                self._nt_controls()

    def _on_center_tab(self, *_):
        """Swap the LEFT panel between the device-tester sections and the Neuro
        Trainer parameters depending on the active center tab."""
        on_tr = (dpg.does_item_exist("center_tabs")
                 and dpg.get_value("center_tabs") == "tab_trainer")
        for t in ("lh_va", "lh_params", "lh_sim", "lh_virt"):
            if dpg.does_item_exist(t):
                dpg.configure_item(t, show=not on_tr)
        if dpg.does_item_exist("left_trainer"):
            dpg.configure_item("left_trainer", show=on_tr)
        if hasattr(self, "desktop"):
            self.desktop.sync()

    # ---------------- center panel ----------------------------------

    def _center_panel(self):
        with dpg.child_window(tag="center_child", width=620, border=False, no_scrollbar=True, no_scroll_with_mouse=True):
            self._device_actions()
            with dpg.tab_bar(tag="center_tabs",
                             callback=self._on_center_tab, tab_list_popup_button=True):
                with dpg.tab(label="Stimulus", tag="tab_designer"):
                    self._designer_tab()
                with dpg.tab(label="Transient", tag="tab_results"):
                    self._results_tab()
                with dpg.tab(label="Pulse response", tag="tab_analysis"):
                    self._analysis_tab()
                with dpg.tab(label="STDP", tag="tab_stdp"):
                    self._stdp_tab()
                with dpg.tab(label="Network Studio", tag="tab_trainer"):
                    self._trainer_tab()
                with dpg.tab(label="Polarization", tag="tab_polar",
                             show=False):
                    self._polar_tab()
                with dpg.tab(label="Noise & retention", tag="tab_noise"):
                    from vatester.noise_panel import NoisePanel
                    self.noise_panel = NoisePanel(self, MODEL_SPECS)
                    self.noise_panel.build()
                with dpg.tab(label="Source", tag="tab_source"):
                    self._source_tab()
                with dpg.tab(label="Compute", tag="tab_compute"):
                    from .compute_panel import ComputePanel
                    self.compute_panel = ComputePanel(self, MODEL_SPECS)
                    self.compute_panel.build()
                with dpg.tab(label="Log", tag="tab_log"):
                    with self._pad(left=8, top=8, bottom=0):
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Copy log",
                                           callback=lambda: self._copy_log())
                            dpg.add_button(label="Save to file",
                                           callback=lambda: self._save_log())
                            dpg.add_button(label="Clear",
                                           callback=lambda: self._clear_log())
                        with dpg.child_window(tag="console", border=False):
                            pass

    def _designer_tab(self):
        with self._pad(left=8, top=8, bottom=0):
            with dpg.group(horizontal=True):
                with dpg.group():
                    self._caption("GENERATOR")
                    # general-purpose generators; a device class may swap in
                    # its own list (e.g. FeCAP adds PUND) on selection
                    dpg.add_combo(sf.generator_names(),
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
            with dpg.tooltip("gen_combo"):
                self._small("", tag="gen_desc", color=C_TEXT2, wrap=480)
            with dpg.child_window(tag="gen_params", height=176, border=False):
                pass
            self._small("custom rows:  t_start_ms   width_ms   amplitude"
                        "     (# comment)", tag="custom_hint")
            dpg.add_input_text(tag="custom_text", multiline=True, width=-1,
                               height=86, tab_input=True,
                               default_value="10 10 -100\n60 10 -100\n"
                                             "200 20 150\n")
            if "mono" in self.fonts:
                dpg.bind_item_font("custom_text", self.fonts["mono"])
            dpg.add_text("", tag="designer_summary", color=C_GREEN)
            self._plot_toolbar(("prev_x",), ("prev_y",),
                               probe_tag="probe_prev")
            with dpg.plot(height=-1, width=-1, tag="preview_plot",
                          no_menus=True):
                dpg.add_plot_legend(tag="preview_plot_leg")
                dpg.add_plot_axis(dpg.mvXAxis, label="time (s)", tag="prev_x",
                                  no_menus=True)
                dpg.add_plot_axis(dpg.mvYAxis, label="amplitude", tag="prev_y",
                                  no_menus=True)
            self._add_plot_menu("preview_plot", "prev_x", "prev_y", x_unit="s")

    def _results_tab(self):
        with self._pad(left=8, top=8, bottom=0):
            self._plot_toolbar(("ax_i_x", "ax_r_x", "ax_g_x"),
                               ("ax_i_y", "ax_r_y", "ax_g_y"),
                               probe_tag="probe_results")
            with dpg.subplots(3, 1, tag="transient_plots", link_all_x=True, width=-1, height=-1,
                              row_ratios=[0.62, 1.0, 1.0]):
                with dpg.plot(tag="plot_i", no_menus=True):
                    dpg.add_plot_legend(tag="plot_i_leg")
                    dpg.add_plot_axis(dpg.mvXAxis, tag="ax_i_x",
                                      no_tick_labels=True, no_menus=True)
                    dpg.add_plot_axis(dpg.mvYAxis, label="stimulus",
                                      tag="ax_i_y", no_menus=True)
                with dpg.plot(tag="plot_r", no_menus=True):
                    dpg.add_plot_legend(tag="plot_r_leg")
                    dpg.add_plot_axis(dpg.mvXAxis, tag="ax_r_x",
                                      no_tick_labels=True, no_menus=True)
                    dpg.add_plot_axis(dpg.mvYAxis, label="R_mem (ohm)",
                                      tag="ax_r_y", no_menus=True)
                with dpg.plot(tag="plot_g", no_menus=True):
                    dpg.add_plot_legend(tag="plot_g_leg")
                    dpg.add_plot_axis(dpg.mvXAxis, label="time (s)",
                                      tag="ax_g_x", no_menus=True)
                    dpg.add_plot_axis(dpg.mvYAxis, label="G (uS)",
                                      tag="ax_g_y", no_menus=True)
            self._add_plot_menu("plot_i", "ax_i_x", "ax_i_y", x_unit="s")
            self._add_plot_menu("plot_r", "ax_r_x", "ax_r_y", x_unit="s")
            self._add_plot_menu("plot_g", "ax_g_x", "ax_g_y", x_unit="s")

    def _analysis_tab(self):
        with self._pad(left=8, top=8, bottom=0):
            with dpg.group(horizontal=True):
                self._caption("ANALYZE")
                dpg.add_combo(["G (uS)", "R_mem (ohm)"],
                              default_value="G (uS)",
                              tag="ana_metric_combo", width=180,
                              callback=lambda s, a: self.on_analysis_metric(a))
                self._small("retained value sampled after each pulse",
                            tag="ana_caption")
            dpg.add_text("", tag="ana_text", wrap=940, color=C_TEXT2)
            self._plot_toolbar(("ana_x",), ("ana_y",), probe_tag="probe_ana")
            with dpg.plot(height=-1, width=-1, tag="ana_plot", no_menus=True):
                dpg.add_plot_legend(tag="ana_plot_leg")
                dpg.add_plot_axis(dpg.mvXAxis, label="pulse #", tag="ana_x",
                                  no_menus=True)
                dpg.add_plot_axis(dpg.mvYAxis, label="G (uS)", tag="ana_y",
                                  no_menus=True)
            self._add_plot_menu("ana_plot", "ana_x", "ana_y")

    def _stdp_tab(self):
        with self._pad(left=8, top=2, bottom=0):
            self.stdp_controls = StdpControls(self)
            with dpg.group(horizontal=True):
                run = dpg.add_button(label="Plot STDP", callback=self.on_plot_stdp)
                dpg.bind_item_theme(run, self.themes["primary"])
                settings = dpg.add_button(label="Experiment settings", tag="stdp_settings_btn")
                dpg.add_button(label="Preview spikes", callback=self.stdp_controls.refresh)
                dpg.add_button(label="Export", callback=self.stdp_controls.export)
                self._small("", tag="stdp_unit_lbl", color=C_TEXT2)
                self._small("", tag="stdp_preview_status", color=C_TEXT2)
            with self.desktop.popup(settings, tag="stdp_settings_popup"):
                with dpg.child_window(width=530, height=420, border=False):
                    self._small("Timing experiment", tag="stdp_intro")
                    with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                        dpg.add_table_column(init_width_or_weight=1)
                        dpg.add_table_column(init_width_or_weight=1)
                        for tag, label, value in (
                            ("stdp_amp_pre", "Pre amplitude", -50.),
                            ("stdp_amp_post", "Post amplitude", 50.),
                            ("stdp_width_ms", "Pulse width (ms)", 10.),
                            ("stdp_range_ms", "Timing range (+/- ms)", 1800.),
                            ("stdp_settle_ms", "Read delay (ms)", 1000.),
                            ("stdp_tail_ratio", "Tail ratio", 1.),
                            ("stdp_tail_tau_ms", "Tail tau (ms)", 20.)):
                            with dpg.table_row():
                                dpg.add_text(label)
                                dpg.add_input_double(tag=tag, default_value=value, width=-1, step=0, format="%.4g",
                                                     callback=self.stdp_controls.refresh)
                        with dpg.table_row():
                            dpg.add_text("Timing samples")
                            dpg.add_input_int(tag="stdp_npts", default_value=80, width=-1,
                                              min_value=6, max_value=600, min_clamped=True, max_clamped=True, step=0)
                        with dpg.table_row():
                            dpg.add_text("Stimulus")
                            dpg.add_combo(STDP_MODES, tag="stdp_mode", default_value=STDP_MODES[0],
                                          width=-1, callback=self.stdp_controls.refresh)
                    with dpg.tooltip("stdp_amp_pre"):
                        dpg.add_text("Signed contributions in the stimulus units. Net drive = pre + post.\nFor voltage programming: pre = -Vpre, post = +Vpost.", wrap=420)
                    self.stdp_controls.build()
            self._plot_toolbar(("stdp_x",), ("stdp_y",), probe_tag="probe_stdp")
            with dpg.subplots(2, 1, tag="stdp_plots", height=-1, width=-1, row_ratios=[.38, .62]):
                self.stdp_controls.build_preview()
                with dpg.plot(tag="stdp_plot", label="Timing-dependent state change", no_menus=True):
                    dpg.add_plot_legend(tag="stdp_plot_leg")
                    dpg.add_plot_axis(dpg.mvXAxis, label="dt = t_post - t_pre (ms)", tag="stdp_x", no_menus=True)
                    dpg.add_plot_axis(dpg.mvYAxis, label="dG (uS)", tag="stdp_y", no_menus=True)
            self._add_plot_menu("stdp_plot", "stdp_x", "stdp_y", copy=True, x_unit="ms")

    # ---------------- device class + polarization --------------------

    def _apply_device_class(self, cls):
        """Programmatic device-class switch: ENABLE that family's models (so
        only one class is checked), then reconfigure drive/labels/tabs. The UI
        has no class selector - users switch by checking a .va; this is for
        startup defaults, the agent, and tests."""
        if cls not in DEVICE_FAMILIES:
            return
        keys = DEVICE_FAMILIES[cls]
        for v in self.va_files:
            tag = f"cb_file_{v.name}"
            if v.model_key and dpg.does_item_exist(tag):
                on = v.model_key in keys
                dpg.set_value(tag, on)
                self.file_enabled[v.name] = on
        self._sync_class_ui(cls, log=True)

    def _sync_class_ui(self, cls, log=False):
        """Reconfigure the drive kind, parameter panel, STDP labels and the
        Polarization tab for a device class WITHOUT touching the model
        checkboxes (so it's safe to call when the user toggles a model)."""
        if cls not in DEVICE_FAMILIES:
            return
        prev = self.device_class
        if self._builtin_settings is None:          # the tab defaults, as built
            self._builtin_settings = self._settings_snapshot()
        if prev != cls:
            self._class_settings[prev] = self._settings_snapshot()
        self.device_class = cls
        keys = DEVICE_FAMILIES[cls]
        kind = DEVICE_KIND.get(cls, "current")
        if dpg.does_item_exist("kind_combo"):
            dpg.set_value("kind_combo", kind)
            self._sync_unit_combo()
        spec = next((s for s in MODEL_SPECS if s.key in keys and s.key in self._enabled_keys()),
                    next((s for s in MODEL_SPECS if s.key in keys), None))
        if spec and dpg.does_item_exist("param_model_sel"):
            dpg.set_value("param_model_sel", spec.label)
            self.rebuild_param_panel()
        has_polar = any("polarization" in getattr(SPEC_BY_KEY[k].cls,
                                                   "ANALYSES", ())
                        for k in keys if k in SPEC_BY_KEY)
        if dpg.does_item_exist("tab_polar"):
            dpg.configure_item("tab_polar", show=has_polar)
        self._refresh_stdp_labels()
        # which tabs, generators and measurements this class offers
        self._apply_class_visibility(cls)
        # settings (drive, generator values, STDP, measurements) belong to a
        # class: restore this class's own if you've used it before, else start
        # from the built-in defaults plus the device twin's GUI_PROFILE - the
        # built-in values are ECFET pA pulses, meaningless for a capacitor
        if prev != cls:
            saved = self._class_settings.get(cls)
            if saved is not None:
                self._settings_restore(saved)
            else:
                self._settings_restore(self._builtin_settings)
                prof = self._class_attr(cls, "GUI_PROFILE")
                if prof:
                    self._apply_gui_profile(prof)
                stim = self._class_attr(cls, "STDP_STIMULUS")
                if not stim and kind == "voltage":
                    defaults = spec.params_cls() if spec else None
                    threshold = min(getattr(defaults, "Vset", .2),
                                    getattr(defaults, "Vreset", .2))
                    stim = {"mode": "overlap", "unit": "V", "amp": .65*threshold,
                            "width_ms": 1., "tail_ratio": 1., "tail_tau_ms": 20.,
                            "range_ms": 100., "settle_ms": 20., "npts": 24}
                if stim:
                    self._apply_stdp_stimulus(stim)
                if dpg.does_item_exist("stdp_include_rule"):
                    dpg.set_value("stdp_include_rule", self._stdp_own_timing(keys))
        if hasattr(self, "stdp_controls"):
            self.stdp_controls.refresh()
        # the per-pulse Analysis metric set also changes (G/R vs Vth/P)
        metrics = self._ana_metrics()
        if self.analysis_metric not in [m[0] for m in metrics]:
            self.analysis_metric = metrics[0][0]
        self._rebuild_ana_metric_combo()
        sp = self._ana_spec()
        App.HOVER_FMT["ana_plot"] = ("pulse", "", sp[1], sp[2])
        if log:
            self.log(f"[device] class -> {cls} "
                     f"(models: {', '.join(keys)}, drive: {kind})")

    # ---- per-device-class GUI settings --------------------------------

    @staticmethod
    def _class_attr(cls, name):
        """First model in a device class that declares `name` (e.g. GUI_PROFILE)."""
        for k in DEVICE_FAMILIES.get(cls, ()):
            if k in SPEC_BY_KEY:
                val = getattr(SPEC_BY_KEY[k].cls, name, None)
                if val:
                    return val
        return None

    def _settings_snapshot(self):
        snap = {"gen_values": {k: dict(v) for k, v in self.gen_values.items()}}
        for tag in CLASS_SETTING_TAGS:
            if dpg.does_item_exist(tag):
                snap[tag] = dpg.get_value(tag)
        return snap

    def _settings_restore(self, snap):
        if not snap:
            return
        self.gen_values = {k: dict(v)
                           for k, v in snap.get("gen_values", {}).items()}
        for tag in CLASS_SETTING_TAGS:
            if tag not in snap or tag == "unit_combo" \
                    or not dpg.does_item_exist(tag):
                continue
            if tag in ("gen_combo", "polar_meas") and snap[tag] not in \
                    dpg.get_item_configuration(tag).get("items", []):
                continue                       # not offered for this class
            dpg.set_value(tag, snap[tag])
        if dpg.does_item_exist("unit_combo"):
            units = (sf.CURRENT_UNITS if dpg.get_value("kind_combo") == "current"
                     else sf.VOLTAGE_UNITS)
            if snap.get("unit_combo") in units:
                dpg.set_value("unit_combo", snap["unit_combo"])
            self._sync_unit_combo()
        if dpg.does_item_exist("gen_combo"):
            self.rebuild_gen_params()
        self._on_polar_meas_change()

    def _apply_gui_profile(self, prof):
        """Load a device twin's GUI_PROFILE: drive unit, tail, generator values,
        default generator and the measurement-tab settings."""
        def put(tag, val):
            if val is not None and dpg.does_item_exist(tag):
                dpg.set_value(tag, val)

        if prof.get("unit") in sf.VOLTAGE_UNITS + sf.CURRENT_UNITS:
            put("unit_combo", prof["unit"])
        put("tail_input", prof.get("tail_s"))
        put("custom_text", prof.get("custom_text"))
        for gname, vals in (prof.get("gen_values") or {}).items():
            self.gen_values[gname] = dict(vals)
        if dpg.does_item_exist("gen_combo") and prof.get("generator") in \
                dpg.get_item_configuration("gen_combo").get("items", []):
            dpg.set_value("gen_combo", prof["generator"])
        loop = prof.get("polar") or {}
        put("polar_vamp", loop.get("vamp"))
        put("polar_period", loop.get("period"))
        put("polar_cycles", loop.get("cycles"))
        put("polar_pts", loop.get("pts"))
        pund = prof.get("pund") or {}
        put("pund_amp", pund.get("amp"))
        put("pund_width_ms", pund.get("width_ms"))
        put("pund_delay_ms", pund.get("delay_ms"))
        kin = prof.get("kinetics") or {}
        put("kin_amps", kin.get("amps"))
        put("kin_wmin_s", kin.get("wmin_s"))
        put("kin_wmax_s", kin.get("wmax_s"))
        put("kin_npts", kin.get("npts"))
        if dpg.does_item_exist("gen_combo"):
            self.rebuild_gen_params()
        self._on_polar_meas_change()

    def _apply_class_visibility(self, cls):
        """Apply Device Lab capabilities; Network Studio has its own synapse selection."""
        prof = self._class_attr(cls, "GUI_PROFILE") or {}
        hide = set(prof.get("hide", ()))
        if dpg.does_item_exist("kind_combo"):
            dpg.configure_item("kind_combo", enabled=not prof.get("lock_kind"))
        if dpg.does_item_exist("gen_combo"):
            names = sf.generator_names(prof.get("generators"))
            dpg.configure_item("gen_combo", items=names)
            if dpg.get_value("gen_combo") not in names:
                dpg.set_value("gen_combo", names[0])
                self.rebuild_gen_params()
        if dpg.does_item_exist("polar_meas"):
            offered = prof.get("measurements") or POLAR_MEASUREMENTS[:1]
            meas = [m for m in POLAR_MEASUREMENTS if m in offered]
            dpg.configure_item("polar_meas", items=meas)
            if dpg.get_value("polar_meas") not in meas:
                dpg.set_value("polar_meas", meas[0])
            self._on_polar_meas_change()
        if dpg.does_item_exist("tab_polar"):
            dpg.configure_item(
                "tab_polar",
                label=f"  {prof.get('polar_tab_label', 'Polarization')}  ")

        if hasattr(self, "desktop"):
            self._on_center_tab()

    @staticmethod
    def _stdp_own_timing(keys):
        """True if a model of this class responds to spike timing by itself
        (a non-zero A_stdp / A_stdp_V in its default parameters), as the
        ECFET does: its STDP is then measured with plain rectangular pulses."""
        for key in keys:
            spec = SPEC_BY_KEY.get(key)
            try:
                p = spec.params_cls() if spec else None
            except Exception:                           # noqa: BLE001
                continue
            if any(getattr(p, name, 0.) for name in ("A_stdp", "A_stdp_V")):
                return True
        return False

    def _apply_stdp_stimulus(self, stim):
        """Load a STDP_STIMULUS dict into the STDP tab widgets."""
        mode = STDP_MODES[1] if stim.get("mode") == "overlap" else STDP_MODES[0]
        amp = abs(float(stim["amp"])) if "amp" in stim else None
        values = {
            "stdp_mode": mode,
            "stdp_amp_pre": -amp if amp is not None else None,
            "stdp_amp_post": amp,                          # opposite signs
            "stdp_width_ms": stim.get("width_ms"),
            "stdp_range_ms": stim.get("range_ms"),
            "stdp_settle_ms": stim.get("settle_ms"),
            "stdp_npts": stim.get("npts"),
            "stdp_tail_ratio": stim.get("tail_ratio"),
            "stdp_tail_tau_ms": stim.get("tail_tau_ms"),
        }
        for tag, val in values.items():
            if val is not None and dpg.does_item_exist(tag):
                dpg.set_value(tag, int(val) if tag == "stdp_npts" else val)
        unit = stim.get("unit")
        if unit and dpg.does_item_exist("unit_combo"):
            dpg.set_value("unit_combo", unit)
        if dpg.does_item_exist("stdp_unit_lbl") and dpg.does_item_exist("unit_combo"):
            dpg.set_value("stdp_unit_lbl", f"{dpg.get_value('unit_combo')} "
                                           f"({dpg.get_value('kind_combo')})")

    # default transient plots = R_mem / G (ECFET); FeFET overrides via RESULT_PLOTS
    DEFAULT_RESULT_PLOTS = (("R", "R_mem", "ohm", 1.0), ("G", "G", "uS", 1e6))

    def _result_plots(self):
        models = self._checked_models()
        if models:
            return getattr(models[0], "RESULT_PLOTS", self.DEFAULT_RESULT_PLOTS)
        return self.DEFAULT_RESULT_PLOTS

    @staticmethod
    def _obs_arr(r, obs):
        """Array for a result observable: 'R'/'G' from the SimResult, else an
        extras key (e.g. 'Vth (V)', 'P (uC/cm2)')."""
        if obs == "R":
            return r.R
        if obs == "G":
            return r.G
        extras = getattr(r, "extras", None) or {}
        return extras.get(obs, r.R)

    def _active_stdp_profile(self):
        models = self._checked_models()
        if models:
            return analysis.state_profile(models[0])
        return ("dG", "uS", 1e6)

    def _refresh_stdp_labels(self):
        label, unit, _ = self._active_stdp_profile()
        if dpg.does_item_exist("stdp_y"):
            dpg.configure_item("stdp_y", label=f"{label} ({unit})")

    def on_plot_polar(self, *_):
        if self.sim_running:
            return
        meas = (dpg.get_value("polar_meas") if dpg.does_item_exist("polar_meas")
                else POLAR_MEASUREMENTS[0])
        models = [m for m in self._sweep_models()       # expands over the sweep
                  if hasattr(m, "P") or getattr(m, "POLAR_OBS", None)]
        if not models:
            self.log("[polar] select a ferroelectric model (FeFET or FeCAP)")
            self._busy(False, "needs a ferroelectric model")
            return
        if meas in POLAR_MEASUREMENTS[:3]:           # the triangular-sweep trio
            v_amp = abs(dpg.get_value("polar_vamp"))
            period = max(dpg.get_value("polar_period"), 1e-3)
            n_pts = max(int(dpg.get_value("polar_pts")), 20)
            n_cycles = max(int(dpg.get_value("polar_cycles")), 1)
            self.sim_running = True
            self._busy(True, f"{meas}: sweep ({n_cycles} cycles)...")
            threading.Thread(target=self._polar_worker,
                             args=(models, v_amp, period, n_pts, n_cycles, meas),
                             daemon=True).start()
        elif meas == "PUND":
            amp = abs(dpg.get_value("pund_amp"))
            width = dpg.get_value("pund_width_ms") * 1e-3
            delay = dpg.get_value("pund_delay_ms") * 1e-3
            if amp <= 0 or width <= 0 or delay < 0:
                self.log("[pund] AMP and WIDTH must be > 0, DELAY >= 0")
                return
            self.sim_running = True
            self._busy(True, f"PUND at {amp:g} V...")
            threading.Thread(target=self._pund_worker,
                             args=(models, amp, width, delay),
                             daemon=True).start()
        else:                                        # switching kinetics
            try:
                amps = [abs(float(x)) for x in
                        dpg.get_value("kin_amps").replace(";", ",").split(",")
                        if x.strip()]
            except ValueError:
                amps = []
            wmin = dpg.get_value("kin_wmin_s")
            wmax = dpg.get_value("kin_wmax_s")
            n = int(dpg.get_value("kin_npts"))
            if not amps or not 0 < wmin < wmax or n < 3:
                self.log("[kinetics] need a comma-separated AMPLITUDES list, "
                         "0 < WIDTH MIN < WIDTH MAX, and POINTS >= 3")
                return
            widths = np.logspace(np.log10(wmin), np.log10(wmax), n).tolist()
            self.sim_running = True
            self._busy(True, f"switching kinetics: {len(amps)} amplitudes x "
                             f"{n} widths...")
            threading.Thread(target=self._kin_worker,
                             args=(models, amps, widths), daemon=True).start()

    # the workers only run the pure analysis layer and ALWAYS post a terminal
    # queue message, so sim_running can never get stuck
    def _polar_worker(self, models, v_amp, period, n_pts, n_cycles,
                      meas=POLAR_MEASUREMENTS[0]):
        try:
            loops = analysis.polarization_loop(models, v_amp=v_amp,
                                               period=period, n_pts=n_pts,
                                               n_cycles=n_cycles)
        except Exception as e:
            loops = {}
            self.q.put(("log", f"[polar] FAILED: {e!r}"))
        self.q.put(("polar", loops, meas))

    def _pund_worker(self, models, amp, width, delay):
        try:
            res = analysis.pund_measure(models, amp, width, delay)
        except Exception as e:
            res = {}
            self.q.put(("log", f"[pund] FAILED: {e!r}"))
        self.q.put(("polar_pund", res))

    def _kin_worker(self, models, amps, widths):
        try:
            res = analysis.switching_kinetics(
                models, amps, widths, preset_amp=max([3.0] + list(amps)))
        except Exception as e:
            res = {}
            self.q.put(("log", f"[kinetics] FAILED: {e!r}"))
        self.q.put(("polar_kin", res))

    def _clear_polar_series(self):
        self.sim_running = False
        for s in self._polar_series:
            if dpg.does_item_exist(s):
                dpg.delete_item(s)
        self._polar_series = []

    def _finish_polar_plot(self, xlabel, ylabel, status):
        dpg.configure_item("polar_x", label=xlabel)
        dpg.configure_item("polar_y", label=ylabel)
        self._fit_axes_of(("polar_x", "polar_y"))
        self._busy(False, status)
        dpg.set_value("center_tabs", "tab_polar")

    def _show_polar(self, loops, meas=POLAR_MEASUREMENTS[0]):
        self._clear_polar_series()
        if not loops:
            self._busy(False, f"{meas}: no data")
            return
        key, ylabel, scale = {
            "P-V loop": ("P", "P ({unit})", 1.0),
            "I-V switching current": ("I", "I (nA)", 1e9),
            "C-V (dQ/dV)": ("C", "dQ/dV (pF)", 1e12),
        }.get(meas, ("P", "P ({unit})", 1.0))
        unit, n = "norm.", 0
        for label, d in loops.items():
            if key not in d:
                self.log(f"  [polar] {label}: no {meas} data from this model")
                continue
            unit = d["unit"]
            ys = [y * scale for y in d[key]]
            self._polar_series.append(dpg.add_line_series(
                d["V"], ys, parent="polar_y", label=label))
            self.log(f"  [polar] {label}: {meas} {key} {min(ys):+.3g}.."
                     f"{max(ys):+.3g} {ylabel.format(unit=unit)}")
            n += 1
        self._finish_polar_plot("voltage (V)", ylabel.format(unit=unit),
                                f"{meas} ready - {n} model(s)")

    def _show_pund(self, res):
        self._clear_polar_series()
        if not res:
            self._busy(False, "PUND: no capacitor model selected")
            return
        status = "PUND ready"
        for label, d in res.items():
            self._polar_series.append(dpg.add_line_series(
                d["t_ms"], d["I_nA"], parent="polar_y", label=label))
            m = d["metrics"]
            self.log(f"  [pund] {label}: P*={m['P*']:+.3f}  U={m['P^']:+.3f}  "
                     f"N*={m['N*']:+.3f}  D={m['N^']:+.3f}  uC/cm^2")
            self.log(f"  [pund] {label}: switched P-U={m['switched+']:+.3f}, "
                     f"N-D={m['switched-']:+.3f} uC/cm^2  ->  "
                     f"Pr = {m['Pr']:.3f} uC/cm^2")
            status = f"PUND: Pr = {m['Pr']:.3f} uC/cm^2 (P-U = {m['switched+']:.3f})"
        self._finish_polar_plot("time (ms)", "I (nA)", status)

    def _show_kinetics(self, res):
        self._clear_polar_series()
        if not res:
            self._busy(False, "switching kinetics: no capacitor model selected")
            return
        many = len(res) > 1
        for label, d in res.items():
            xs = [float(np.log10(w)) for w in d["widths"]]
            for amp, fr in d["curves"].items():
                ys = [f * 100.0 for f in fr]
                name = f"{label} @ {amp:g} V" if many else f"{amp:g} V"
                self._polar_series.append(dpg.add_line_series(
                    xs, ys, parent="polar_y", label=name))
                # t50: pulse width that switches half the polarization
                t50 = None
                for i in range(1, len(fr)):
                    if fr[i - 1] < 0.5 <= fr[i]:
                        f0, f1 = fr[i - 1], fr[i]
                        x = xs[i - 1] + (0.5 - f0) / (f1 - f0) * (xs[i] - xs[i - 1])
                        t50 = 10.0 ** x
                        break
                self.log(f"  [kinetics] {name}: switched {min(ys):.1f}.."
                         f"{max(ys):.1f} %, t50 = "
                         + (f"{t50:.3g} s" if t50 else "not reached in range"))
        self._finish_polar_plot("log10 pulse width (s)", "switched fraction (%)",
                                "switching kinetics ready")

    def _on_polar_meas_change(self):
        if not dpg.does_item_exist("polar_meas"):
            return
        meas = dpg.get_value("polar_meas")
        for tag, show in (("polar_grp_loop", meas in POLAR_MEASUREMENTS[:3]),
                          ("polar_grp_pund", meas == "PUND"),
                          ("polar_grp_kin", meas == "Switching kinetics")):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=show)
        if dpg.does_item_exist("polar_desc"):
            dpg.set_value("polar_desc", POLAR_DESC.get(meas, ""))
        if dpg.does_item_exist("polar_run_btn"):
            dpg.configure_item("polar_run_btn", label=f"Run {meas}")

    def _polar_tab(self):
        with self._pad(left=8, top=8, bottom=0):
            self._small(POLAR_DESC[POLAR_MEASUREMENTS[0]], tag="polar_desc",
                        wrap=940)
            with dpg.group(horizontal=True):
                with dpg.group():
                    self._caption("MEASUREMENT")
                    dpg.add_combo(list(POLAR_MEASUREMENTS[:1]), tag="polar_meas",
                                  default_value=POLAR_MEASUREMENTS[0], width=190,
                                  callback=lambda *_: self._on_polar_meas_change())
                with dpg.group(horizontal=True, tag="polar_grp_loop"):
                    for tag, cap, default in (("polar_vamp", "V AMP (V)", 3.0),
                                              ("polar_period", "SWEEP (s)", 0.3),
                                              ("polar_cycles", "CYCLES", 4),
                                              ("polar_pts", "POINTS/CYC", 300)):
                        with dpg.group():
                            self._caption(cap)
                            if tag in ("polar_cycles", "polar_pts"):
                                dpg.add_input_int(tag=tag,
                                                  default_value=int(default),
                                                  width=84, step=0, min_value=1,
                                                  min_clamped=True)
                            else:
                                dpg.add_input_double(tag=tag,
                                                     default_value=default,
                                                     width=84, format="%.4g",
                                                     step=0)
                with dpg.group(horizontal=True, tag="polar_grp_pund", show=False):
                    for tag, cap, default in (("pund_amp", "AMP (V)", 3.0),
                                              ("pund_width_ms", "WIDTH (ms)", 1.0),
                                              ("pund_delay_ms", "DELAY (ms)", 10.0)):
                        with dpg.group():
                            self._caption(cap)
                            dpg.add_input_double(tag=tag, default_value=default,
                                                 width=84, format="%.4g", step=0)
                with dpg.group(horizontal=True, tag="polar_grp_kin", show=False):
                    with dpg.group():
                        self._caption("AMPLITUDES (V)")
                        dpg.add_input_text(tag="kin_amps", width=150,
                                           default_value="1.0, 1.5, 2.0, 3.0")
                    for tag, cap, default in (("kin_wmin_s", "WIDTH MIN (s)", 1e-9),
                                              ("kin_wmax_s", "WIDTH MAX (s)", 0.1)):
                        with dpg.group():
                            self._caption(cap)
                            dpg.add_input_double(tag=tag, default_value=default,
                                                 width=96, format="%.3g", step=0)
                    with dpg.group():
                        self._caption("POINTS")
                        dpg.add_input_int(tag="kin_npts", default_value=17,
                                          width=72, step=0, min_value=3,
                                          min_clamped=True)
                with dpg.group():
                    dpg.add_spacer(height=18)
                    b = dpg.add_button(label=f"Run {POLAR_MEASUREMENTS[0]}",
                                       tag="polar_run_btn",
                                       callback=self.on_plot_polar)
                    dpg.bind_item_theme(b, self.themes["primary"])
            self._plot_toolbar(("polar_x",), ("polar_y",), probe_tag="probe_polar")
            with dpg.plot(height=-1, width=-1, tag="polar_plot", no_menus=True):
                dpg.add_plot_legend(tag="polar_plot_leg")
                dpg.add_plot_axis(dpg.mvXAxis, label="voltage (V)",
                                  tag="polar_x", no_menus=True)
                dpg.add_plot_axis(dpg.mvYAxis, label="P (uC/cm^2)",
                                  tag="polar_y", no_menus=True)
            self._add_plot_menu("polar_plot", "polar_x", "polar_y", x_unit="V")

    def _source_tab(self):
        with self._pad(left=8, top=8, bottom=0):
            with dpg.group(horizontal=True):
                dpg.add_combo([], tag="va_edit_sel", width=300,
                              callback=self.on_editor_file_change)
                sb = dpg.add_button(label="Save", callback=self.on_editor_save)
                dpg.bind_item_theme(sb, self.themes["primary"])
                dpg.add_button(label="Save as...",
                               callback=self.on_editor_save_as)
                dpg.add_button(label="Reload", callback=self.on_editor_reload)
                dpg.add_button(label="Ask agent about this file",
                               callback=self.on_ask_about_file)
                tb = dpg.add_button(label="Build twin & run",
                                    callback=self.on_agent_build_twin)
                dpg.bind_item_theme(tb, self.themes["primary"])
                with dpg.tooltip(tb):
                    dpg.add_text("Ask the agent to read this Verilog-A, build/"
                                 "update its Python twin, and run a simulation "
                                 "in the GUI (needs Agent > Autonomous on)")
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
                    dpg.add_combo([], tag="virt_lib_combo", width=130,
                                  callback=lambda *_: self._virt_lib_changed())
                    self._caption("cell")
                    dpg.add_combo([], tag="virt_cell_combo", width=120,
                                  callback=lambda *_: self._virt_cell_changed())
                    self._caption("view")
                    dpg.add_combo([], tag="virt_view_combo", width=95)
                    b = dpg.add_button(label="Load source",
                                       callback=self.on_virt_load_source)
                    dpg.bind_item_theme(b, self.themes["primary"])
                    rb = dpg.add_button(label="Refresh",
                                        tag="virt_refresh_libs",
                                        callback=self.on_virt_refresh_libs)
                    dpg.bind_item_theme(rb, self.themes["chip"])
                    with dpg.tooltip("virt_refresh_libs"):
                        dpg.add_text("Refresh library list from Virtuoso")
                    wb = dpg.add_button(label="Write back",
                                        tag="virt_write_back",
                                        callback=self.on_virt_write_back)
                    dpg.bind_item_theme(wb, self.themes["chip"])
                    with dpg.tooltip("virt_write_back"):
                        dpg.add_text("Write the editor's text back into the "
                                     "selected Virtuoso library/cell/view "
                                     "(overwrites the cellview source - "
                                     "recompile in Cadence after)")
                self._small("connect to Virtuoso first (left panel)",
                            tag="virt_browse_status")
            dpg.bind_item_theme(vbar, self.themes["card"])

            dpg.add_input_text(tag="va_editor", multiline=True, width=-1,
                               height=-1, tab_input=True)
            if "mono" in self.fonts:
                dpg.bind_item_font("va_editor", self.fonts["mono"])

    # ---------------- right panel: agent chat ------------------------

    def _right_panel(self):
        with dpg.child_window(width=RIGHT_W, tag="right_child", border=False, show=False):
            with self._pad(left=8, top=6, bottom=2):
                with dpg.group(horizontal=True):
                    online = self.agent.backend != "none"
                    dot = dpg.add_text("●", tag="agent_dot",
                                       color=C_GREEN if online else C_RED)
                    with dpg.tooltip(dot):
                        dpg.add_text("backend: " + self.agent.backend_label())
                    h = dpg.add_text("Claude Agent", tag="agent_title",
                                     color=C_TEXT)
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
                                             "or /provider in the chat")
                            with dpg.popup(badge, tag="model_popup",
                                           mousebutton=dpg.mvMouseButton_Left):
                                pass
                            self._rebuild_model_popup()
                            acc = dpg.add_button(label="Account", small=True,
                                                 callback=self.on_account_open)
                            dpg.bind_item_theme(acc, self.themes["chip"])
                            if "small" in self.fonts:
                                dpg.bind_item_font(acc, self.fonts["small"])
                            with dpg.tooltip(acc):
                                dpg.add_text("Sign in / switch Claude account, "
                                             "or run this app under a specific "
                                             "key/token")
                            # session cost is shown on demand (/cost or /usage)
                            # rather than inline here, where it overflowed the
                            # narrow agent panel
                with dpg.child_window(tag="chat_log", height=-56,
                                      border=False):
                    pass
                with dpg.group(tag="attach_row", horizontal=True, show=False):
                    self._small("", tag="attach_label", color=C_MUTED)
                    bx = dpg.add_button(label="x", small=True,
                                        callback=self.on_attach_clear)
                    dpg.bind_item_theme(bx, self.themes["chip"])
                    with dpg.tooltip(bx):
                        dpg.add_text("Remove attachments")
                with dpg.group(horizontal=True):
                    batt = dpg.add_button(label="+", width=26,
                                          callback=self.on_attach_open)
                    dpg.bind_item_theme(batt, self.themes["chip"])
                    with dpg.tooltip(batt):
                        dpg.add_text("Attach files - pdf, csv, image... The "
                                     "agent reads them and can retune the "
                                     "model / Verilog from their content.")
                    ci = dpg.add_input_text(tag="chat_input", width=-78,
                                            hint="Message the agent...  (/help)",
                                            on_enter=True, callback=self.on_send)
                    if "chat" in self.fonts:    # let the user type Bangla too
                        dpg.bind_item_font(ci, self.fonts["chat"])
                    b = dpg.add_button(label="Send", tag="btn_send",
                                       callback=self.on_send)
                    dpg.bind_item_theme(b, self.themes["primary"])
                    bstop = dpg.add_button(label="Stop", tag="btn_stop",
                                           show=False,
                                           callback=self.on_agent_stop)
                    dpg.bind_item_theme(bstop, self.themes["stop"])
                # the live-plot toggle is mirrored from the Agent menu; the
                # watcher reads cb_liveplot, so keep it (hidden, in sync)
                dpg.add_checkbox(tag="cb_liveplot", default_value=True,
                                 show=False)

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
                    cp = dpg.add_button(label="copy", small=True,
                                        user_data=text,
                                        callback=lambda s, a, u:
                                        dpg.set_clipboard_text(u))
                    dpg.bind_item_theme(cp, self.themes["chip"])
                    if "small" in self.fonts:
                        dpg.bind_item_font(cp, self.fonts["small"])
                    with dpg.tooltip(cp):
                        dpg.add_text("Copy this message to the clipboard")
                msg = self._chat_message(text, BUBBLE_W - indent - 28)
                with dpg.popup(msg):    # right-click the text -> Copy
                    dpg.add_selectable(label="Copy message", user_data=text,
                                       callback=lambda s, a, u:
                                       dpg.set_clipboard_text(u))
            dpg.bind_item_theme(bub, self.themes[theme])
        dpg.add_spacer(height=5, parent="chat_log")
        self._scroll_bottom("chat_log")

    def _chat_message(self, text, wrap_w):
        """Render a chat message body and return its item id. Complex scripts
        (Bangla, Devanagari, ...) are shaped by HarfBuzz and shown as an image
        because ImGui can't shape them; plain Latin uses fast, selectable text."""
        complex_ = needs_shaping(text)
        if complex_ and self.shaper and self.shaper.ok:
            out = self.shaper.render(text, color=C_TEXT[:3], max_width=wrap_w)
            if out:
                flat, tw, th = out
                self._tex_n += 1
                ttag = "chat_tex_%d" % self._tex_n
                try:
                    dpg.add_static_texture(tw, th, flat.tolist(),
                                           parent="chat_tex_reg", tag=ttag)
                    self._chat_tex.append(ttag)
                    return dpg.add_image(ttag)
                except Exception as e:           # noqa: BLE001
                    print(f"[chat] shaped-image failed: {e}")
        msg = dpg.add_text(text, wrap=wrap_w, color=C_TEXT)
        if complex_ and "chat" in self.fonts:
            # shaper unavailable: at least use a font that HAS the glyphs (the
            # Nirmala chat font) so it reads, instead of boxes from Segoe
            dpg.bind_item_font(msg, self.fonts["chat"])
        elif "chat_en" in self.fonts:            # plain Latin: 15px chat body
            dpg.bind_item_font(msg, self.fonts["chat_en"])
        return msg

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

    # ---------------- slash commands, provider & model picker ---------

    @property
    def agent_model_label(self):
        return self.model_sel[self.agent.provider][0]

    @property
    def agent_model_id(self):
        return self.model_sel[self.agent.provider][1]

    def _provider_models(self):
        return self.model_choices.get(self.agent.provider,
                                      PROVIDER_MODELS["claude"])

    def _refresh_models_async(self, force=False, announce=False):
        """Ask every provider for its current model list, off the render thread.

        Network I/O, so it runs in a daemon worker and reports back through
        self.q like every other long job. Always posts its terminal ("models")
        message - even on exception - so models_busy can't stick.
        """
        if self.models_busy:
            return
        self.models_busy = True
        keys = {p: self.agent.provider_key(p) for p in PROVIDER_ORDER}
        wd = self.workdir

        def worker():
            out, src = {}, {}
            try:
                for p in PROVIDER_ORDER:
                    try:
                        rows, s = models_catalog.list_models(
                            p, wd, api_key=keys.get(p), force=force)
                    except Exception:
                        rows, s = list(PROVIDER_MODELS[p]), "builtin"
                    if rows:
                        out[p], src[p] = rows, s
            finally:
                self.q.put(("models", out, src, announce))

        threading.Thread(target=worker, daemon=True).start()

    def _on_models_done(self, choices, sources, announce):
        """Render-thread half of _refresh_models_async."""
        for p, rows in (choices or {}).items():
            self.model_choices[p] = rows
            self.model_source[p] = (sources or {}).get(p, "builtin")
        self.models_busy = False
        # a selected model that the provider no longer lists falls back to
        # Default rather than silently sending an unknown id
        for p in PROVIDER_ORDER:
            label, mid = self.model_sel.get(p, ("Default", None))
            if mid and not any(r[1] == mid for r in self.model_choices[p]):
                self.model_sel[p] = ("Default", None)
        self._rebuild_model_popup()
        if dpg.does_item_exist("model_badge"):
            dpg.configure_item("model_badge", label=self.agent_model_label)
        src = self.model_source.get(self.agent.provider, "builtin")
        n = len(self._provider_models())
        self.log(f"[agent] model list <- {src} ({n} entries)")
        if announce:
            self.append_chat("sys", f"Model list refreshed from {src}: "
                                    f"{n} models for "
                                    f"{PROVIDER_TITLES[self.agent.provider]}.")

    def _rebuild_model_popup(self):
        if not dpg.does_item_exist("model_popup"):
            return
        dpg.delete_item("model_popup", children_only=True)
        src = self.model_source.get(self.agent.provider, "builtin")
        tag = {"live": "live", "cache": "cached",
               "builtin": "offline list"}.get(src, src)
        self._small(f"MODEL  ({tag})", color=(126, 150, 220),
                    parent="model_popup")
        for label, mid, desc in self._provider_models():
            dpg.add_selectable(label=f"{label}   ·  {desc}",
                               user_data=(label, mid), parent="model_popup",
                               callback=self._on_model_pick)
        dpg.add_separator(parent="model_popup")
        # plain ASCII on purpose - the default ImGui font has no arrow glyphs
        dpg.add_selectable(label="Refresh model list",
                           parent="model_popup",
                           callback=lambda s, a, u:
                               self._refresh_models_async(force=True,
                                                          announce=True))
        dpg.add_separator(parent="model_popup")
        for other in PROVIDER_ORDER:
            if other == self.agent.provider:
                continue
            dpg.add_selectable(label=f"Switch to {PROVIDER_TITLES[other]}",
                               user_data=other, parent="model_popup",
                               callback=lambda s, a, u: self._set_provider(u))

    def _set_provider(self, name, announce=True):
        ok, msg = self.agent.set_provider(name)
        if not ok:
            self.append_chat("err", msg)
            return
        prov = self.agent.provider
        if dpg.does_item_exist("agent_title"):
            dpg.set_value("agent_title", PROVIDER_TITLES[prov])
        if dpg.does_item_exist("model_badge"):
            dpg.configure_item("model_badge", label=self.agent_model_label)
        self._rebuild_model_popup()
        self._refresh_agent_status()
        if announce:
            note = ""
            if self.agent.backend == "none":
                note = ("\nNo backend available: " + self.agent.backend_label()
                        + "\nUse /provider claude to switch back.")
            elif prov == "openai" and self.agent.backend == "sdk":
                note = ("\nChat-only: patterns, GUI actions and advice work; "
                        "file edits / autonomous fixing need the codex CLI "
                        "or the Claude provider.")
            self.append_chat("sys", f"Provider: {PROVIDER_TITLES[prov]} "
                                    f"({self.agent.backend_label()}){note}")
        self.log(f"[agent] provider -> {prov} ({self.agent.backend_label()})")

    def _match_model(self, query):
        q = query.lower().strip()
        q = q.replace("claude-", "")
        for label, mid, _ in self._provider_models():
            hay = label.lower() + " " + (mid or "")
            if q and q in hay:
                return label, mid
        # OpenAI: accept any explicit model id verbatim (e.g. a new release)
        if self.agent.provider == "openai" and q and " " not in q:
            raw = query.strip()
            return raw, raw
        return None

    def _set_model(self, label, mid, announce=True):
        self.model_sel[self.agent.provider] = (label, mid)
        if dpg.does_item_exist("model_badge"):
            dpg.configure_item("model_badge", label=label)
        if announce:
            self.append_chat("sys", f"Model set to {label}"
                             + (f"  ({mid})" if mid else " (default)"))
        self.log(f"[agent] model -> {label}" + (f" ({mid})" if mid else ""))

    def _on_model_pick(self, sender, app_data, user_data):
        label, mid = user_data
        self._set_model(label, mid)

    def _add_model_picker(self):
        with dpg.group(parent="chat_log"):
            with dpg.child_window(width=BUBBLE_W, auto_resize_y=True,
                                  border=True) as card:
                self._small(f"SELECT MODEL ({PROVIDER_TITLES[self.agent.provider]})",
                            color=(126, 150, 220))
                for label, mid, desc in self._provider_models():
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
                                       for l, _, _ in self._provider_models())
                    self.append_chat("err",
                                     f"No model matches "
                                     f"'{' '.join(parts[1:])}'. "
                                     f"Try /model {names}")
        elif cmd == "provider":
            if len(parts) == 1:
                self.append_chat(
                    "sys", f"Provider: {PROVIDER_TITLES[self.agent.provider]} "
                           f"({self.agent.backend_label()})\n"
                           "Switch with /provider claude | openai")
            else:
                self._set_provider(parts[1])
        elif cmd in ("clear", "reset"):
            self.on_agent_reset()
        elif cmd in ("cost", "usage"):
            self.append_chat("sys", self._usage_summary())
        elif cmd == "help":
            self.append_chat("sys",
                             "Commands:\n"
                             "/model              choose the model\n"
                             "/model <name>       e.g. /model opus, /model gpt-5.1\n"
                             "/provider           show the active provider\n"
                             "/provider <name>    claude | openai\n"
                             "/cost  or  /usage   show this session's cost/tokens\n"
                             "/clear              reset the conversation\n"
                             "/help               this list")
        else:
            self.append_chat("err", f"Unknown command /{cmd} - try /help")

    def _usage_summary(self):
        """Session cost / token / turn summary shown on /cost or /usage."""
        a = self.agent
        if not self._agent_turns:
            return "No agent calls yet this session."
        lines = [f"Session usage ({self._agent_turns} "
                 f"repl{'y' if self._agent_turns == 1 else 'ies'}):",
                 f"  cost:   ${a.total_cost:.4f}"]
        if getattr(a, "total_in", 0) or getattr(a, "total_out", 0):
            lines.append(f"  tokens: {a.total_in:,} in / {a.total_out:,} out")
        if not a.total_cost:
            lines.append("  (cost not reported by this provider/backend)")
        return "\n".join(lines)

    def _chat_pending(self, show):
        if dpg.does_item_exist("pending_bubble"):
            dpg.delete_item("pending_bubble")
        if show:
            self._chat_t0 = time.time()
            self._chat_status = "Claude is thinking"
            with dpg.group(parent="chat_log", tag="pending_bubble",
                           horizontal=True):
                with dpg.child_window(width=340, auto_resize_y=True,
                                      border=True) as bub:
                    with dpg.group(horizontal=True):
                        dpg.add_loading_indicator(radius=1.5, style=1,
                                                  color=C_AGENT)
                        dpg.add_text("Claude is thinking…", tag="pending_text",
                                     color=C_TEXT2)
                dpg.bind_item_theme(bub, self.themes["bub_agent"])
            self._scroll_bottom("chat_log")

    def _chat_progress(self, label):
        """Update the live 'working…' bubble with the agent's current action."""
        if label:
            self._chat_status = label
        self._chat_pending_refresh()

    def _chat_tick(self):
        """Render-loop tick: keep the pending bubble's elapsed clock moving even
        during a long single step. Must never raise (called every frame)."""
        try:
            if not self.chat_busy:
                return
            now = time.time()
            if now - getattr(self, "_chat_tick_last", 0.0) < 1.0:
                return
            self._chat_tick_last = now
            self._chat_pending_refresh()
        except Exception:
            pass

    def _chat_pending_refresh(self):
        """Repaint the pending bubble text with the latest action + elapsed
        time. Called on each tool event and once per second from the render
        loop so a long single step still shows a ticking clock (not frozen)."""
        if not dpg.does_item_exist("pending_text"):
            return
        el = int(time.time() - getattr(self, "_chat_t0", time.time()))
        status = getattr(self, "_chat_status", "Claude is thinking")
        dpg.set_value("pending_text", f"{status}…  ({el // 60}:{el % 60:02d})")

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
        from vatester.desktop_layout import build_parameters
        build_parameters(self, spec, _defaults_of(spec.params_cls), C_TEXT2)
        self._prune_sweeps()      # drop sweeps on params not in this model

    def _on_param_edit(self, sender, value, user_data):
        key, name, is_int = user_data
        self.param_values[key][name] = int(value) if is_int else float(value)
        from vatester.desktop_layout import filter_parameters
        filter_parameters(self)

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
        custom = gen.name == "Custom pattern"
        dpg.configure_item("custom_text", show=custom)
        dpg.configure_item("custom_hint", show=custom)
        dpg.configure_item("gen_params", show=bool(gen.params), height=max(64, math.ceil(len(gen.params)/2)*30+12))
        self.desktop.request_preview()
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
        self.desktop.request_preview()

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
        self._hover_cache.pop("preview_plot", None)
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
        return self._build_models()

    def _build_models(self, overrides=None, name_suffix=""):
        """Enabled models built from the panel params, optionally with a few
        parameter overrides (for a sweep) and a label suffix on each .name."""
        models = []
        for key in self._enabled_keys():
            s = SPEC_BY_KEY[key]
            pv = dict(self.param_values[key])
            if overrides:
                for p, v in overrides.items():
                    if p in pv:
                        pv[p] = v
            try:
                m = s.cls(s.params_cls(**pv))
                if name_suffix:
                    m.name = f"{m.name} [{name_suffix}]"
                models.append(m)
            except Exception as e:
                self.log(f"[params] {key}: {e}")
        return models

    # ---- parameter sweep (Virtuoso-style specs, engineering units) -------

    _SI_MULT = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
                "": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}

    @classmethod
    def _parse_eng(cls, s):
        """'4u'->4e-6, '3m'->3e-3, '100k'->1e5, '1.5'->1.5, '4e-6'->4e-6.
        None if it doesn't parse.  Case-sensitive (m=milli, M=mega)."""
        s = (s or "").strip().replace("µ", "u")
        m = re.fullmatch(r"([-+]?(?:[0-9]*\.?[0-9]+)(?:[eE][-+]?[0-9]+)?)\s*"
                         r"([fpnumkKMGT]?)", s)
        if not m:
            return None
        try:
            return float(m.group(1)) * cls._SI_MULT[m.group(2)]
        except (ValueError, KeyError):
            return None

    @classmethod
    def _parse_eng_list(cls, text):
        out = []
        for tok in (text or "").replace(",", " ").split():
            v = cls._parse_eng(tok)
            if v is not None:
                out.append(v)
        return out

    def _spec_values(self, spec):
        """Concrete value list for a sweep spec (range / center / explicit)."""
        t = spec.get("type", "values")
        n = max(2, int(spec.get("steps", 5)))
        if t == "range":
            a, b = spec.get("from"), spec.get("to")
            if a is None or b is None:
                return []
            if spec.get("step_type") == "Log" and a > 0 and b > 0:
                return [a * (b / a) ** (k / (n - 1)) for k in range(n)]
            return [a + (b - a) * k / (n - 1) for k in range(n)]
        if t == "center":
            c, sp = spec.get("center"), spec.get("span")
            if c is None or sp is None:
                return []
            a, b = c - abs(c) * sp / 200.0, c + abs(c) * sp / 200.0
            return [a + (b - a) * k / (n - 1) for k in range(n)]
        return list(spec.get("values", []))

    def _spec_summary(self, spec):
        vals = self._spec_values(spec)
        t, e = spec.get("type", "values"), self._eng
        if t == "range":
            return (f"{e(spec.get('from', 0))} -> {e(spec.get('to', 0))}  "
                    f"{len(vals)} {spec.get('step_type', 'Linear')[:3].lower()}")
        if t == "center":
            return (f"{e(spec.get('center', 0))} +/-{spec.get('span', 0):g}%  "
                    f"{len(vals)} pts")
        return ", ".join(e(v) for v in vals)

    def _sweep_combos(self):
        """Cartesian product of the sweep specs -> [(label, {param: value})].
        No sweep -> [("", {})] (a single base run).  Capped at SWEEP_CAP."""
        active = [(s["param"], self._spec_values(s)) for s in self.sweep_specs]
        active = [(p, vs) for p, vs in active if vs]
        if not active:
            return [("", {})]
        import itertools
        combos = []
        for vals in itertools.product(*[vs for _, vs in active]):
            ov = {p: v for (p, _), v in zip(active, vals)}
            label = ", ".join(f"{p}={self._eng(v)}"
                              for (p, _), v in zip(active, vals))
            combos.append((label, ov))
        return combos[:self.SWEEP_CAP]

    def _sweep_models(self):
        """Enabled models expanded over the sweep combos, each distinctly named.
        Identical to _checked_models() when no sweep is defined."""
        out = []
        for label, ov in self._sweep_combos():
            out.extend(self._build_models(ov, name_suffix=label))
        return out

    def _sweep_param_names(self):
        """Sweepable parameter names = the params of the model shown in the
        param panel (falls back to the first enabled model)."""
        label = (dpg.get_value("param_model_sel")
                 if dpg.does_item_exist("param_model_sel") else None)
        key = LABEL_TO_KEY.get(label)
        if not key:
            keys = self._enabled_keys()
            key = keys[0] if keys else MODEL_SPECS[0].key
        return list(_defaults_of(SPEC_BY_KEY[key].params_cls).keys())

    def _prune_sweeps(self):
        """Drop sweep specs whose param isn't in the current model."""
        names = set(self._sweep_param_names())
        self.sweep_specs = [s for s in self.sweep_specs
                            if s.get("param") in names]

    # ---- the Parametric Sweep dialog (Run menu) --------------------------

    def on_open_sweep_dialog(self, *_):
        names = self._sweep_param_names()
        if dpg.does_item_exist("sw_param"):
            dpg.configure_item("sw_param", items=names)
            if names and dpg.get_value("sw_param") not in names:
                dpg.set_value("sw_param", names[0])
        self._sweep_backup = [dict(s) for s in self.sweep_specs]
        self._rebuild_spec_list()
        self._on_sw_type()
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()
        dpg.configure_item("sweep_dialog", show=True,
                           pos=(max(0, (vw - 580) // 2), max(0, (vh - 470) // 4)))

    def _on_sw_type(self, *_):
        t = dpg.get_value("sw_type") if dpg.does_item_exist("sw_type") else "From/To"
        for tag, kind in (("sw_range_row", "From/To"),
                          ("sw_center_row", "Center/Span%"),
                          ("sw_values_row", "Values")):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=(t == kind))

    def _read_spec_editor(self):
        p = dpg.get_value("sw_param")
        if not p:
            return None
        t = dpg.get_value("sw_type")
        if t == "From/To":
            a = self._parse_eng(dpg.get_value("sw_from"))
            b = self._parse_eng(dpg.get_value("sw_to"))
            if a is None or b is None:
                return None
            return {"param": p, "type": "range", "from": a, "to": b,
                    "step_type": dpg.get_value("sw_steptype"),
                    "steps": int(dpg.get_value("sw_steps"))}
        if t == "Center/Span%":
            c = self._parse_eng(dpg.get_value("sw_center"))
            sp = self._parse_eng(dpg.get_value("sw_span"))
            if c is None or sp is None:
                return None
            return {"param": p, "type": "center", "center": c, "span": sp,
                    "steps": int(dpg.get_value("sw_csteps"))}
        vals = self._parse_eng_list(dpg.get_value("sw_values"))
        return {"param": p, "type": "values", "values": vals} if vals else None

    def _load_spec_into_editor(self, spec):
        dpg.set_value("sw_param", spec["param"])
        t = spec.get("type", "values")
        dpg.set_value("sw_type", {"range": "From/To", "center": "Center/Span%",
                                  "values": "Values"}[t])
        self._on_sw_type()
        if t == "range":
            dpg.set_value("sw_from", self._eng(spec["from"]))
            dpg.set_value("sw_to", self._eng(spec["to"]))
            dpg.set_value("sw_steptype", spec.get("step_type", "Linear"))
            dpg.set_value("sw_steps", int(spec.get("steps", 5)))
        elif t == "center":
            dpg.set_value("sw_center", self._eng(spec["center"]))
            dpg.set_value("sw_span", f"{spec['span']:g}")
            dpg.set_value("sw_csteps", int(spec.get("steps", 5)))
        else:
            dpg.set_value("sw_values",
                          ", ".join(self._eng(v) for v in spec.get("values", [])))

    def _on_spec_row_click(self, sender, app_data, user_data):
        spec = next((s for s in self.sweep_specs
                     if s.get("param") == user_data), None)
        if spec:
            self._load_spec_into_editor(spec)

    def _on_sweep_spec_add(self, *_):
        spec = self._read_spec_editor()
        if not spec:
            self.log("[sweep] enter valid values (eng units OK: 4u, 3m, 100k)")
            return
        self.sweep_specs = [s for s in self.sweep_specs
                            if s.get("param") != spec["param"]]
        self.sweep_specs.append(spec)
        self._rebuild_spec_list()

    def _on_sweep_spec_delete(self, sender=None, app_data=None, user_data=None):
        p = user_data or dpg.get_value("sw_param")
        self.sweep_specs = [s for s in self.sweep_specs if s.get("param") != p]
        self._rebuild_spec_list()

    def _rebuild_spec_list(self):
        if not dpg.does_item_exist("sweep_spec_list"):
            return
        dpg.delete_item("sweep_spec_list", children_only=True)
        n = 1
        for s in self.sweep_specs:
            vals = self._spec_values(s)
            n *= max(1, len(vals))
            with dpg.group(parent="sweep_spec_list", horizontal=True):
                bx = dpg.add_button(label="x", small=True, user_data=s["param"],
                                    callback=self._on_sweep_spec_delete)
                dpg.bind_item_theme(bx, self.themes["chip"])
                dpg.add_selectable(
                    label=f"  {s['param']}   {self._spec_summary(s)}   "
                          f"({len(vals)})",
                    user_data=s["param"], callback=self._on_spec_row_click)
        if self.sweep_specs:
            total = min(n, self.SWEEP_CAP)
            self._small(f"= {total} curve(s)"
                        + (" (capped)" if n > self.SWEEP_CAP else ""),
                        parent="sweep_spec_list", color=C_ACC)
        else:
            self._small("no sweep - add a variable above", color=C_MUTED,
                        parent="sweep_spec_list")

    def _on_sweep_ok(self, *_):
        dpg.configure_item("sweep_dialog", show=False)
        if self.sweep_specs:
            self.log("[sweep] " + "; ".join(
                f"{s['param']} {self._spec_summary(s)}"
                for s in self.sweep_specs)
                + f"  ({len(self._sweep_combos())} curves) - run a plot to see it")

    def _on_sweep_cancel(self, *_):
        self.sweep_specs = getattr(self, "_sweep_backup", [])
        dpg.configure_item("sweep_dialog", show=False)

    def on_run(self, *_, live=False):
        if self.sim_running:
            return
        try:
            wf, meta, unit, kind, label = self.build_waveform()
        except ValueError as e:
            if not live:
                self.log(f"[run] {e}")
            return
        models = self._sweep_models()       # expands over the parameter sweep
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
        if not live:
            self._last_compute = "transient"
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
            # never let a save failure kill the worker before the "results"
            # message below - that would strand sim_running=True forever.
            try:
                outdir = os.path.join(self.workdir, "results")
                os.makedirs(outdir, exist_ok=True)
                for r in results:
                    stem = re.sub(r"[^\w]+", "_", r.label).strip("_")
                    p = r.save_csv(os.path.join(outdir, f"gui_{stem}.csv"))
                    self.q.put(("log", f"  csv -> {p}"))
            except Exception as e:
                self.q.put(("log", f"  [save] auto-save CSV failed: {e}"))
        self.q.put(("results", results, meta, unit, kind, live))

    # ---------------- STDP characterization --------------------------

    def on_plot_stdp(self, *_, live=False):
        if self.sim_running:
            return
        models = self._sweep_models()       # expands over the parameter sweep
        if not models:
            if not live:
                self.log("[stdp] no model selected")
            return
        unit = dpg.get_value("unit_combo")
        kind = dpg.get_value("kind_combo")
        scale = sf.UNIT_SCALE.get(unit, 1.0)
        if dpg.does_item_exist("stdp_unit_lbl"):
            dpg.set_value("stdp_unit_lbl", f"{unit} ({kind})")
        amp_pre = dpg.get_value("stdp_amp_pre") * scale
        amp_post = dpg.get_value("stdp_amp_post") * scale
        width = dpg.get_value("stdp_width_ms") * 1e-3
        tail = dpg.get_value("stdp_settle_ms") * 1e-3
        spike = self.stdp_controls.spike()
        measurement = dpg.get_value("stdp_measurement")
        include_rule = dpg.get_value("stdp_include_rule")
        try:
            stdp_protocol.waveforms(0., amp_pre, amp_post, width, tail, spike)
            if not np.isfinite(dpg.get_value("stdp_range_ms")):
                raise ValueError("Timing range must be finite.")
            if any(getattr(m, "input_kind", "current") != kind for m in models):
                raise ValueError("Choose models matching the selected voltage/current drive.")
            models = [stdp_protocol.prepare_model(m, include_rule) for m in models]
        except Exception as exc:
            self.log(f"[stdp] {exc}")
            self._busy(False, "STDP: invalid experiment settings")
            return
        self.stdp_controls.refresh()
        # Rectangular references keep the legacy non-overlap sampling. Shaped
        # protocols also resolve peak overlap and explicitly include dt=0.
        dt_max = abs(dpg.get_value("stdp_range_ms")) * 1e-3
        dt_min = width * (1.01 if spike is None else .05)
        if dt_min >= dt_max:
            self.sim_running = False
            self._busy(False, "STDP: width >= dt range")
            self.log(f"[stdp] pulse width {width * 1e3:g} ms is too large for "
                     f"the dt range ({dt_max * 1e3:g} ms); widen DT RANGE or "
                     f"reduce WIDTH")
            return
        n_pts = int(dpg.get_value("stdp_npts")) \
            if dpg.does_item_exist("stdp_npts") else 80
        per_side = max(3, n_pts // 2)              # mirrored -> ~n_pts total
        ratio = dt_max / dt_min
        pos = [dt_min * ratio ** (k / (per_side - 1)) for k in range(per_side)]
        pos[-1] = dt_max                          # land exactly on the range
        dts = [-d for d in reversed(pos)] + ([] if spike is None else [0.]) + pos
        self._stdp_ctx = {
            "models": models, "amp_pre": amp_pre, "amp_post": amp_post,
            "width": width, "tail": tail, "unit": unit, "kind": kind,
            "dts": dts, "spike": spike,
            "protocol": stdp_protocol.metadata(models, spike, measurement, include_rule),
        }
        if not live:
            if spike is None:
                self.log(f"[stdp] |dt| starts at {dt_min * 1e3:g} ms "
                         f"(> pulse width {width * 1e3:g} ms): no pulse overlap")
            else:
                self.log(f"[stdp] waveform-driven: {spike['shape']}; "
                         f"tau={stdp_protocol.tau(spike)*1e3:g} ms; includes peak overlap and dt=0")
            self.log(f"[stdp] {measurement}; device's own timing response {'ON' if include_rule else 'OFF'}; "
                     "each timing starts from the same initial state")
            for m in models:
                mk = getattr(m, "input_kind", "current")
                if mk != kind:
                    self.log(f"  [warn] {m.name} expects {mk} input; "
                             f"amplitudes are treated as {mk}")
        self.sim_running = True
        if not live:
            self._last_compute = "stdp"
        if live:
            self._busy(True, "live STDP re-sweep (agent edited the code)")
        else:
            self._busy(True, f"STDP sweep: {len(dts)} timings x "
                             f"{len(models)} model(s)")
            self.log(f"[stdp] sweep dt {dts[0]*1e3:+.4g}..{dts[-1]*1e3:+.4g} ms"
                     f" ({len(dts)} pts), "
                     f"pre {dpg.get_value('stdp_amp_pre'):+g} {unit}, "
                     f"post {dpg.get_value('stdp_amp_post'):+g} {unit}, "
                     f"width {dpg.get_value('stdp_width_ms'):g} ms, "
                     f"settle {dpg.get_value('stdp_settle_ms'):g} ms, "
                     f"stimulus: {STDP_MODES[0 if spike is None else 1]}")
        threading.Thread(target=self._stdp_worker,
                         args=(models, amp_pre, amp_post, width, dts, tail,
                               live, spike, measurement),
                         daemon=True).start()

    def _stdp_worker(self, models, amp_pre, amp_post, width, dts, tail,
                     live=False, spike=None, measurement=stdp_protocol.MEASUREMENTS[0]):
        t_start = time.perf_counter()
        curves, diagnostics = {}, {}
        log = None if live else (lambda msg: self.q.put(("log", msg)))
        try:
            curves, diagnostics = stdp_protocol.sweep(
                models, amp_pre, amp_post, width, dts, tail, spike,
                measurement=measurement, log=log)
        except Exception as exc:
            self.q.put(("log", f"[stdp] FAILED: {exc!r}"))
        if not live:
            self.q.put(("log", f"[stdp] sweep done in {time.perf_counter()-t_start:.1f} s"))
        self.q.put(("stdp", [d*1e3 for d in dts], curves, live, diagnostics))

    def _show_stdp(self, dts_ms, curves, live=False, diagnostics=None):
        if self._stdp_ctx is not None:
            self._stdp_ctx["diagnostics"] = diagnostics or {}
        self.sim_running = False
        self._clear_probes(("stdp_plot",))
        self._hover_cache.pop("stdp_plot", None)
        for s in self._stdp_series:
            if dpg.does_item_exist(s):
                dpg.delete_item(s)
        self._stdp_series = []
        if not curves:
            self._busy(False, "STDP sweep failed")
            return
        self._stdp_dts_ms = list(dts_ms)        # remember points for the copy menu
        self._stdp_curves = {k: list(v) for k, v in curves.items()}
        controls = []
        for model in self._stdp_ctx["models"]:
            rows = (diagnostics or {}).get(model.name)
            if rows:
                unit = analysis.state_profile(model)[1]
                single = max(abs(x) for key in ("pre", "post") for x in rows[key])
                idle = max(abs(x) for x in rows["idle"])
                controls.append(f"{model.name}: max single-spike change {single:.4g} {unit}; idle drift {idle:.4g} {unit}")
        if dpg.does_item_exist("stdp_control_summary"):
            dpg.set_value("stdp_control_summary", " | ".join(controls))
        slabel, sunit, _ = analysis.state_profile(self._stdp_ctx["models"][0])
        measurement = self._stdp_ctx.get("protocol", {}).get("measurement", "Pair state change")
        dpg.configure_item("stdp_y", label=f"{slabel} ({sunit}) - {measurement.split(' (')[0]}")
        lines = []
        nan = float("nan")
        for label, ys in curves.items():
            # break the line across the empty overlap gap (no segment joining
            # the -width and +width branches) via a NaN point at dt = 0
            neg = [(d, y) for d, y in zip(dts_ms, ys) if d < 0]
            pos = [(d, y) for d, y in zip(dts_ms, ys) if d > 0]
            line_x = [d for d, _ in neg] + [0.0] + [d for d, _ in pos]
            line_y = [y for _, y in neg] + [nan] + [y for _, y in pos]
            if self._stdp_ctx.get("spike") is not None:
                line_x, line_y = list(dts_ms), list(ys)
            # ONE line-series-with-markers per curve (markers via the bound
            # "markers" theme; the NaN at dt=0 both breaks the line and skips a
            # marker there) so a single legend entry toggles line AND points
            # together.
            sid = dpg.add_line_series(
                line_x, line_y, parent="stdp_y", label=label)
            dpg.bind_item_theme(sid, self.themes["markers"])
            self._stdp_series.append(sid)
            imax = max(range(len(ys)), key=lambda i: abs(ys[i]))
            lines.append(f"{label}:  {slabel} {min(ys):+.4g}..{max(ys):+.4g} "
                         f"{sunit}  |  strongest change at dt = "
                         f"{dts_ms[imax]:+.3g} ms")
        self._stdp_summary = lines
        if not live:
            for ln in lines:
                self.log("  [stdp] " + ln)
            self.log("  [stdp] click any curve point to drill into that timing "
                     "(the pair's full transient + its retained state change)")
        # a fresh sweep fits to the data; the user re-applies a manual Axis
        # range lock if they want one (no auto/sticky lock - they choose)
        self._fit_axes_of(("stdp_x", "stdp_y"))
        if live:
            self._busy(False, f"live STDP · {len(curves)} model(s) "
                              f"(agent edit applied)")
        else:
            self._busy(False, f"STDP curve ready - {len(curves)} model(s)")
            dpg.set_value("center_tabs", "tab_stdp")

    _SI_PREFIX = {-15: "f", -12: "p", -9: "n", -6: "u", -3: "m", 0: "",
                  3: "k", 6: "M", 9: "G", 12: "T"}

    @classmethod
    def _eng(cls, value):
        """Engineering SI notation (Spectre/Cadence style): mantissa in
        [1, 1000) with a metric prefix - 0.2 -> '200m', 1.8 -> '1.8',
        3.8e-6 -> '3.8u'.  Carries the unit via the prefix only (no symbol)."""
        if not math.isfinite(value) or value == 0.0:
            return "0"
        exp3 = (int(math.floor(math.log10(abs(value)))) // 3) * 3
        exp3 = max(-15, min(12, exp3))
        mant = value / (10.0 ** exp3)
        return f"{mant:.4g}{cls._SI_PREFIX.get(exp3, f'e{exp3}')}"

    def _copy_stdp(self, kind):
        """Copy completed STDP data with the measured state's actual units."""
        dts = self._stdp_dts_ms
        curves = self._stdp_curves
        if not dts or not curves:
            self.log("[stdp] nothing to copy yet - run Plot STDP first")
            return
        dt_row = lambda vals: " ".join(self._eng(v * 1e-3) for v in vals)  # ms->s
        model = self._stdp_ctx["models"][0]
        state, display_unit, _ = analysis.state_profile(model)
        unit, factor = {"uS": ("S", 1e-6), "mV": ("V", 1e-3),
                        "uC/cm^2": ("C/m^2", .01)}.get(display_unit, (display_unit, 1.))
        state_row = lambda vals: " ".join(self._eng(v * factor) for v in vals)
        if kind == "dt":
            text = dt_row(dts)
            what = f"{len(dts)} dt values"
        elif kind == "dg":
            if len(curves) == 1:
                text = f"# {state} ({unit})\n" + state_row(next(iter(curves.values())))
            else:
                text = "\n".join(f"# {label}: {state} ({unit})\n{state_row(ys)}"
                                 for label, ys in curves.items())
            what = f"{state} ({unit}) values ({len(curves)} model(s))"
        else:  # pairs CSV (dt in s, dG in S, engineering units)
            labels = list(curves)
            head = ["dt_s"] + [f"{state}_{l.split()[0]} ({unit})" for l in labels]
            out = [",".join(head)]
            for i, d in enumerate(dts):
                out.append(",".join([self._eng(d * 1e-3)]
                                    + [self._eng(curves[l][i] * factor)
                                       for l in labels]))
            text = "\n".join(out)
            what = f"{len(dts)} dt,{state} rows"
        dpg.set_clipboard_text(text)
        self.log(f"[stdp] copied {what} to clipboard")
        self._busy(False, f"copied {what} to clipboard")


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
        waves, t_stop = stdp_protocol.waveforms(
            dt, ctx["amp_pre"], ctx["amp_post"], width, tail, ctx.get("spike"))
        wf = waves["pair"]
        dpg.set_value("stdp_preview_dt", dt*1e3)
        self.stdp_controls.preview(ctx, dt)
        results, summary = [], []
        for m in ctx["models"]:
            r = simulate(m, wf, t_stop, dt_min=min(1e-6, width/50), label=m.name)
            results.append(r)
            # retained change of the device's own STDP state (dG for ECFET,
            # dVt for FeFET, dP for FeCAP) - this is the raw pair, before the
            # single-spike subtraction the sweep applies
            slabel, sunit, sscale = analysis.state_profile(m)
            arr = analysis.state_trace(r, m)
            text = f"{m.name}: {slabel}={(arr[-1] - arr[0]) * sscale:+.4g} {sunit}"
            if getattr(m, "STDP_OBS", "G") in ("G", "R"):
                text += f"  dR={r.R[-1] - r.R[0]:+.4g} ohm"
            summary.append(text)
        order = ("post after pre" if dt > 0
                 else ("pre after post" if dt < 0
                       else "simultaneous"))
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
        for _p in ("plot_i", "plot_r", "plot_g"):
            self._hover_cache.pop(_p, None)
        for s in self._series:
            if dpg.does_item_exist(s):
                dpg.delete_item(s)
        self._series = []
        if not results:
            self._busy(False, "no results")
            return
        if meta and meta.get("pund"):             # PUND generator run
            for r in results:
                wins = r.waveform.pulse_windows()
                if analysis.PUND_OBS in (r.extras or {}) and len(wins) >= 4:
                    m = analysis.pund_metrics(r, wins[-4:])
                    self.log(f"[pund] {r.label}: P-U = {m['switched+']:+.3f}, "
                             f"N-D = {m['switched-']:+.3f} uC/cm^2  ->  "
                             f"Pr = {m['Pr']:.3f} uC/cm^2  (U = {m['P^']:+.3f} "
                             f"dielectric)")
        scale = 1.0 / sf.UNIT_SCALE.get(unit, 1.0)
        xs, ys = self._stair_points(results[0].waveform, scale,
                                    tail=0.05 * results[0].t[-1])
        self._series.append(dpg.add_stair_series(
            xs, ys, parent="ax_i_y", label="stimulus"))
        dpg.configure_item("ax_i_y",
                           label=("I_gate (%s)" if kind == "current"
                                  else "V_gate (%s)") % unit)
        # the two transient plots are device-profile driven: R_mem/G for ECFET,
        # Vth/polarization for FeFET
        (o1, l1, u1, s1), (o2, l2, u2, s2) = self._result_plots()
        self._results_plot_profile = ((o1, l1, u1, s1), (o2, l2, u2, s2))
        self._results_kind = kind
        dpg.configure_item("ax_r_y", label=f"{l1} ({u1})")
        dpg.configure_item("ax_g_y", label=f"{l2} ({u2})")
        self._result_obs = ((o1, l1, u1), (o2, l2, u2))   # for hover labels
        for r in results:
            self._series.append(dpg.add_line_series(
                r.t.tolist(), (self._obs_arr(r, o1) * s1).tolist(),
                parent="ax_r_y", label=r.label))
            self._series.append(dpg.add_line_series(
                r.t.tolist(), (self._obs_arr(r, o2) * s2).tolist(),
                parent="ax_g_y", label=r.label))
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

    # default per-pulse metrics (ECFET); FeFET overrides via ANALYSIS_METRICS
    DEFAULT_ANALYSIS_METRICS = (("G", "G", "uS", 1e6),
                                ("R", "R_mem", "ohm", 1.0))

    def _ana_metrics(self):
        models = self._checked_models()
        if models:
            return getattr(models[0], "ANALYSIS_METRICS",
                           self.DEFAULT_ANALYSIS_METRICS)
        return self.DEFAULT_ANALYSIS_METRICS

    def _ana_spec(self):
        """(obs, label, unit, scale) for the active analysis metric, falling
        back to the profile's first metric when the current one isn't valid
        for this device class."""
        metrics = self._ana_metrics()
        for m in metrics:
            if m[0] == self.analysis_metric:
                return m
        return metrics[0]

    def _rebuild_ana_metric_combo(self):
        metrics = self._ana_metrics()
        labels = [f"{lab} ({u})" for _, lab, u, _ in metrics]
        cur = self._ana_spec()
        if dpg.does_item_exist("ana_metric_combo"):
            dpg.configure_item("ana_metric_combo", items=labels,
                               default_value=f"{cur[1]} ({cur[2]})")
            dpg.set_value("ana_metric_combo", f"{cur[1]} ({cur[2]})")
        # relabel + re-check the Run-menu mirror items
        for i, tag in enumerate(("ana_menu_0", "ana_menu_1")):
            if dpg.does_item_exist(tag):
                if i < len(metrics):
                    dpg.configure_item(tag, label=labels[i], show=True)
                    dpg.set_value(tag, metrics[i][0] == self.analysis_metric)
                else:
                    dpg.configure_item(tag, show=False)

    def _on_ana_menu(self, sender, app_data, user_data):
        metrics = self._ana_metrics()
        idx = user_data if user_data < len(metrics) else 0
        self.on_analysis_metric(metrics[idx][0])

    def on_analysis_metric(self, metric):
        # accept a metric label ("G (uS)" / "Vth (mV)") or an observable key
        metrics = self._ana_metrics()
        obs = None
        for o, lab, u, _ in metrics:
            if metric in (o, lab, f"{lab} ({u})"):
                obs = o
                break
        self.analysis_metric = obs or metrics[0][0]
        spec = self._ana_spec()
        # sync the two Run-menu check items to the active metric (by index)
        for i, tag in enumerate(("ana_menu_0", "ana_menu_1")):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, i < len(metrics)
                              and metrics[i][0] == self.analysis_metric)
        if dpg.does_item_exist("ana_metric_combo"):
            dpg.set_value("ana_metric_combo", f"{spec[1]} ({spec[2]})")
        App.HOVER_FMT["ana_plot"] = ("pulse", "", spec[1], spec[2])
        self.update_analysis()
        if self.results:
            dpg.set_value("center_tabs", "tab_analysis")

    def _ana_name_unit(self):
        spec = self._ana_spec()
        return (spec[1], spec[2])

    def update_analysis(self):
        self._clear_probes(("ana_plot",))
        self._hover_cache.pop("ana_plot", None)
        for s in self._ana_series:
            if dpg.does_item_exist(s):
                dpg.delete_item(s)
        self._ana_series = []
        obs, name, unit, scale = self._ana_spec()
        if dpg.does_item_exist("ana_y"):
            dpg.configure_item("ana_y", label=f"{name} ({unit})")
        if dpg.does_item_exist("ana_caption"):
            dpg.set_value("ana_caption",
                          f"retained {name} sampled after each pulse")
        if not self.results:
            return
        if len(self.results[0].waveform.pulse_windows()) < 2:
            dpg.set_value("ana_text", f"need a multi-pulse stimulus for the "
                                      f"{name}-vs-pulse curve")
            return
        # the per-pulse sampling lives in the hot-reloaded analysis layer;
        # here we only turn its output into plot series.
        n_each = self.results_meta.get("n_each")
        data = analysis.per_pulse_samples(self.results, obs, scale, n_each)
        mean = lambda v: sum(v) / len(v) if v else 0.0
        lines = []
        for d in data:
            if d is None:
                continue
            n, vals, ne = d["n"], d["vals"], d["n_each"]
            if ne:
                self._ana_series.append(dpg.add_line_series(
                    n[:ne], vals[:ne], parent="ana_y",
                    label=f"{d['label']} LTP"))
                # LTD starts one point early (the last LTP point) so the two
                # branches join with no visual gap at the turnover
                self._ana_series.append(dpg.add_line_series(
                    n[ne - 1:], vals[ne - 1:], parent="ana_y",
                    label=f"{d['label']} LTD"))
                lines.append(
                    f"{d['label']}:  {name} {min(vals):.4g}..{max(vals):.4g} "
                    f"{unit} | mean d{name}  LTP {mean(d['dl']):+.4g}  "
                    f"LTD {mean(d['dd']):+.4g} {unit}/pulse")
            else:
                self._ana_series.append(dpg.add_line_series(
                    n, vals, parent="ana_y", label=d["label"]))
                lines.append(f"{d['label']}:  {name} {min(vals):.4g}.."
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
        from .plot_export import result_figure
        outdir = os.path.join(self.workdir, "results")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "gui_plot.png")
        figure = result_figure(self.results, self._results_plot_profile,
                               self.results_unit, self._results_kind)
        figure.savefig(path, dpi=130)
        self.log(f"[export] png -> {path}")

    # =================================================================
    # Verilog-A browsing / editing
    # =================================================================

    def rescan_va(self, startup=False):
        self.va_files = va_scan(self.workdir)
        self.network_studio.refresh_sources()
        self.compute_panel.refresh_sources()
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
                            v.name, v.model_key in ("v1", "v2", "v3"))
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
                    elif v.role == "network_primitive":
                        primitive = self._small("primitive", color=C_TEXT2)
                        with dpg.tooltip(primitive):
                            dpg.add_text("Ideal source asset. Network Studio requires a supported conductance twin; this primitive is not available for training.", wrap=300)
                    else:
                        dpg.add_text("")
        dpg.configure_item("va_edit_sel", items=names)
        if not startup:
            self.log(f"[va] found {len(names)} file(s): {', '.join(names)}")
        else:
            self.log(f"[va] auto-detected {len(names)} Verilog-A file(s)")
        if names and not self.selected_va:
            self.on_va_selected(names[0], interact=False)
        self._check_untwinned_va()      # prompt to build twins for unmapped .va

    def _va_by_name(self, name):
        for v in self.va_files:
            if v.name == name:
                return v
        return None

    def _on_file_toggle(self, sender, value, user_data):
        self.file_enabled[user_data] = value
        if value:
            # ONE device class at a time: checking a model of a different class
            # auto-unchecks the others (no ECFET+FeFET mix).  The checkmarks are
            # the device-class selector - no separate control needed.
            va = self._va_by_name(user_data)
            new_cls = DEVICE_OF_KEY.get(va.model_key) if va else None
            if new_cls:
                dropped = []
                for v in self.va_files:
                    if v.name == user_data or not v.model_key:
                        continue
                    if (self.file_enabled.get(v.name)
                            and DEVICE_OF_KEY.get(v.model_key) != new_cls):
                        tag = f"cb_file_{v.name}"
                        if dpg.does_item_exist(tag):
                            dpg.set_value(tag, False)
                        self.file_enabled[v.name] = False
                        dropped.append(v.name)
                if dropped:
                    self.append_chat("sys", f"Switched to {new_cls} - unchecked "
                                     f"{', '.join(dropped)} (one device class at "
                                     f"a time).")
        # sync drive (current/voltage) + labels/tabs to the now-coherent class
        self._sync_class_from_checked()

    def _sync_class_from_checked(self):
        """Detect the device class from the checked .va files and reconfigure
        the GUI to match (drive kind, STDP/Analysis labels, Polarization tab)."""
        classes = {DEVICE_OF_KEY.get(k) for k in self._enabled_keys()}
        classes.discard(None)
        if len(classes) == 1:
            cls = classes.pop()
            if cls != self.device_class:
                self._sync_class_ui(cls, log=True)

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
            self.editor_remote = None
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

    def on_editor_save_as(self, *_):
        """Save the editor buffer to a local .va file (works for remote buffers
        too - it makes a tracked local copy)."""
        name = "model.va"
        if self.editor_path:
            name = os.path.basename(self.editor_path)
        elif self.editor_remote:
            name = f"{self.editor_remote[1]}.va"     # cell name
        folder = os.path.join(self.workdir, "verilog")
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            dpg.set_value("editor_status", f"save error: {e}")
            return
        dpg.configure_item("save_va_dialog", default_path=folder,
                           default_filename=name, show=True)

    def _on_save_va_picked(self, sender, app_data):
        path = (app_data or {}).get("file_path_name") or ""
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".va"
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(dpg.get_value("va_editor"))
        except OSError as e:
            dpg.set_value("editor_status", f"save error: {e}")
            return
        # if it landed in the workspace, adopt it as the live local buffer
        in_ws = os.path.abspath(path).startswith(os.path.abspath(self.workdir))
        if in_ws:
            self.editor_path = path
            self.editor_mtime = os.path.getmtime(path)
            self.editor_remote = None
            self.rescan_va()
        dpg.set_value("editor_status", f"saved local: {os.path.basename(path)}")
        self.log(f"[editor] saved local copy -> {path}"
                 + ("  (now tracked)" if in_ws else ""))

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

    def _editor_local_path(self):
        """A local .va path for the current editor buffer, saving a remote
        (Virtuoso-loaded) buffer into the workspace first so the agent can read
        it. Returns the path or None."""
        if self.editor_path:
            return self.editor_path
        if self.editor_remote:                       # remote buffer -> save it
            cell = self.editor_remote[1]
            folder = os.path.join(self.workdir, "verilog")
            path = os.path.join(folder, f"{cell}.va")
            try:
                os.makedirs(folder, exist_ok=True)
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(dpg.get_value("va_editor"))
                self.editor_path = path
                self.editor_mtime = os.path.getmtime(path)
                self.editor_remote = None
                self.rescan_va()
                dpg.set_value("editor_status",
                              f"saved local: {cell}.va (so the agent can read it)")
                self.log(f"[editor] saved remote source -> {path} for the agent")
                return path
            except OSError as e:
                self.append_chat("err", f"could not save source: {e}")
                return None
        if self.selected_va:
            return self.selected_va.path
        name = dpg.get_value("va_edit_sel")
        va = self._va_by_name(name) if name else None
        return va.path if va else None

    def on_agent_build_twin(self, *_):
        """One click: have the agent read the current Verilog-A file, build/
        update a Python twin for it, and run a simulation in the GUI."""
        path = self._editor_local_path()
        if not path:
            self.append_chat("err",
                             "Open or load a Verilog-A file first, then "
                             "'Build twin & run'.")
            return
        rel = os.path.relpath(path, self.workdir)
        prompt = (
            f"Read {rel} and explain what this Verilog-A device does (physics "
            f"and key parameters). Then make the GUI able to SIMULATE it by "
            f"writing a NEW Python twin as a file in the twins/ folder - follow "
            f"the contract in twins/README.md and the annotated TWIN_SPEC "
            f"template in vatester/twin_loader.py's module docstring (a model "
            f"class with step(t,dt,drive)/.R/.G/reset()/observables(), a params "
            f"dataclass, and a TWIN_SPEC dict with key/label/device_class/"
            f"input_kind/va_keywords and the profile so the right axes/units "
            f"show). Do NOT edit the GUI "
            f"(vatester/) or core engine (simulation_core/) - twins/ only. The GUI "
            f"auto-loads twins/ at startup, so after you add the file tell me to "
            f"restart, then RUN a transient and show the plot, iterating until "
            f"it's physically sensible. Summarize the device and what you wrote.")
        dpg.set_value("chat_input", prompt)
        auto = (dpg.get_value("menu_auto")
                if dpg.does_item_exist("menu_auto") else True)
        if not auto:
            self.append_chat("sys", "Tip: enable Agent > Autonomous so the agent "
                             "can edit the twin and run the sim itself.")
        self.on_send()

    def _check_orphan_twins(self):
        """Warn (in the log + chat) about simulation_core/model_*.py files that define a
        twin but are NOT registered - so they'd silently never simulate (the
        exact trap v3 hit).  Catches a hand-added core twin where a registration
        step was missed.  Twins in twins/ are auto-registered, so they're fine."""
        import glob
        import importlib
        ecfet_dir = os.path.dirname(_m_v1.__file__)
        registered = set()
        for mod, _c, _p in RELOAD_MODULES.values():
            f = getattr(mod, "__file__", None)
            if f:
                registered.add(os.path.normcase(os.path.abspath(f)))
        for path in sorted(glob.glob(os.path.join(ecfet_dir, "model_*.py"))):
            if os.path.normcase(os.path.abspath(path)) in registered:
                continue
            base = os.path.splitext(os.path.basename(path))[0]
            try:
                mod = importlib.import_module(f"simulation_core.{base}")
            except Exception as e:                       # noqa: BLE001
                self.log(f"[models] simulation_core/{base}.py failed to import: {e!r}")
                continue
            twin = next((n for n in dir(mod)
                         if isinstance(getattr(mod, n), type)
                         and all(hasattr(getattr(mod, n), a)
                                 for a in ("step", "reset", "R"))), None)
            if twin:
                self.log(f"[models] WARNING: simulation_core/{base}.py defines a twin "
                         f"({twin}) but is NOT registered - it will not "
                         f"simulate. Register it (MODEL_SPECS / TWIN_FILE / "
                         f"RELOAD_MODULES + a va_scan keyword) or move it to "
                         f"twins/ with a TWIN_SPEC.")
                self.append_chat("err", f"simulation_core/{base}.py looks like a device "
                                 f"twin but isn't registered, so it won't run. "
                                 f"Register it or put it in twins/.")

    def _check_untwinned_va(self):
        """If any .va has no Python twin, prompt to build one (agent -> twins/)."""
        untw = [v for v in self.va_files if not v.model_key and v.role != "network_primitive"]
        if not untw:
            return
        names = tuple(sorted(v.name for v in untw))
        if names == self._untwinned_prompted:        # don't nag repeatedly
            return
        self._untwinned_prompted = names
        if not dpg.does_item_exist("untwin_modal"):
            return
        backend_ok = self.agent.backend != "none"
        msg = ("These files need a Python device twin before they can be "
               "simulated in Device Lab. Choose a file to set up.")
        if not backend_ok:
            msg += "\nConnect an assistant in Account to enable model generation."
        dpg.set_value("untwin_msg", msg)
        dpg.configure_item("untwin_list", items=[v.name for v in untw], num_items=min(5, max(2, len(untw))))
        dpg.set_value("untwin_list", untw[0].name)
        dpg.configure_item("untwin_build", enabled=backend_ok)
        dpg.show_item("untwin_modal")
        ds.center_dialog("untwin_modal")

    def _build_selected_twin(self, *_):
        self._on_build_twin_for(None, None, dpg.get_value("untwin_list"))

    def _on_build_twin_for(self, sender, app_data, user_data):
        dpg.configure_item("untwin_modal", show=False)
        va = self._va_by_name(user_data)
        if not va:
            return
        dpg.set_value("va_edit_sel", va.name)
        self.on_editor_file_change(None, va.name)
        self.on_agent_build_twin()

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

    def on_virt_write_back(self, *_):
        """Ask to confirm writing the editor buffer back to a Virtuoso cellview."""
        if not self.virtuoso.connected:
            dpg.set_value("virt_browse_status",
                          "connect to Virtuoso first (left panel)")
            return
        lib = dpg.get_value("virt_lib_combo")
        cell = dpg.get_value("virt_cell_combo")
        view = dpg.get_value("virt_view_combo")
        if not (lib and cell and view):
            dpg.set_value("virt_browse_status",
                          "pick the target library, cell, and view first")
            return
        if not (dpg.get_value("va_editor") or "").strip():
            dpg.set_value("virt_browse_status", "editor is empty - nothing to write")
            return
        self._virt_write_target = (lib, cell, view)
        dpg.set_value("virt_write_msg",
                      f"Overwrite the source of\n\n    {lib} / {cell} / {view}\n\n"
                      f"with the {len(dpg.get_value('va_editor'))} characters in "
                      f"the editor?  This replaces the cellview's source file in "
                      f"the Cadence library. Re-netlist / recompile the cell in "
                      f"Cadence afterwards for it to take effect.")
        vw = dpg.get_viewport_client_width()
        dpg.configure_item("virt_write_modal", show=True,
                           pos=(max(0, (vw - 470) // 2), 160))

    def _do_virt_write_back(self, *_):
        dpg.configure_item("virt_write_modal", show=False)
        target = getattr(self, "_virt_write_target", None)
        if not target:
            return
        lib, cell, view = target
        text = dpg.get_value("va_editor")
        dpg.set_value("virt_browse_status", f"writing to {lib}/{cell}/{view}...")
        self.log(f"[virtuoso] writing editor -> {lib}/{cell}/{view} "
                 f"({len(text)} chars)")
        self._virt_browse(
            lambda: dict(self.virtuoso.write_source(lib, cell, view, text),
                         lib=lib, cell=cell, view=view), "write")

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
        elif kind == "write":
            tag = f"{payload['lib']}/{payload['cell']}/{payload['view']}"
            if payload.get("ok"):
                dpg.set_value("virt_browse_status",
                              f"wrote {tag} -> {payload['path']} "
                              f"(recompile in Cadence)")
                self.log(f"[virtuoso] wrote {tag} -> {payload['path']}")
                self._virt_dialog(True, "Written to Virtuoso",
                                  f"Saved the editor's source into\n{tag}\n\n"
                                  f"{payload['path']}\n\nRe-netlist / recompile "
                                  f"the cell in Cadence for the change to apply.")
            else:
                note = payload.get("note", "write failed")
                dpg.set_value("virt_browse_status", f"{tag}: {note}")
                self.log(f"[virtuoso] write FAILED {tag}: {note}")
                self._virt_dialog(False, "Write failed", f"{tag}\n{note}")

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
        # Remote buffer: detach from any local path so the plain Save can't
        # clobber a local file with remote content.  "Write back" targets the
        # cellview, "Save as..." writes a local copy.
        self.editor_path = None
        self.editor_mtime = 0.0
        self.editor_remote = (lib, cell, view)
        dpg.set_value("va_edit_sel", "")
        dpg.set_value("editor_status",
                      f"remote: {tag}  (Write back -> Virtuoso, "
                      f"or Save as... -> local file)")
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
                "note": ("The GUI simulates the Python model twins in simulation_core/ "
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
            # key the values by the device's own STDP state (dG_uS for ECFET,
            # dVt_mV for FeFET, dP_uC/cm^2 for FeCAP) rather than always dG
            slabel, sunit, _ = analysis.state_profile(self._stdp_ctx["models"][0])
            vkey = f"{slabel}_{sunit}"
            for sid in self._stdp_series:
                lbl = dpg.get_item_configuration(sid).get("label", "")
                if lbl.startswith("##"):
                    continue
                d = dpg.get_value(sid)
                if d and len(d) >= 2:
                    # drop the NaN gap-break point (invalid JSON; agent-facing)
                    pts = [(x, y) for x, y in zip(d[0], d[1]) if y == y]
                    curves[lbl] = {"dt_ms": [round(float(x), 4) for x, _ in pts],
                                   vkey: [round(float(y), 5) for _, y in pts]}
            ctx = self._stdp_ctx
            snap["stdp"] = {
                "curves": curves, "state": slabel, "unit": sunit,
                "protocol": ctx.get("protocol"), "diagnostics": ctx.get("diagnostics"),
                "stimulus": (STDP_MODES[0] if ctx.get("spike") is None
                             else STDP_MODES[1]),
                **({} if ctx.get("spike") is None else {
                    "tail_ratio": ctx["spike"]["tail_ratio"],
                    "tail_tau_ms": ctx["spike"]["tail_tau"] * 1e3})}

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
        sweepable = self._sweep_param_names()
        if sweepable:
            lines.append(
                "Sweepable parameters for the active model (use these EXACT "
                "names in the 'sweep' action - they are case-sensitive; e.g. "
                "device width/length are 'w'/'l', not 'W'/'L'): "
                + ", ".join(sweepable))
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
                f"the Python twins in simulation_core/ (model_v1.py=ECFET v1, "
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
        # Neuromorphic Trainer: when its tab is active or a network exists, give
        # the agent the trainer instructions + live state so the SAME chat can
        # build/train/diagnose the spiking crossbar.
        on_trainer = (dpg.does_item_exist("center_tabs")
                      and dpg.get_value("center_tabs") == "tab_trainer")
        if on_trainer or self.trainer is not None:
            lines.append(NT_AGENT_SYSTEM)
            lines.append(self._nt_agent_context())
            self._write_nt_snapshot()
        return "\n".join(lines)

    # ---- attachments --------------------------------------------------

    def on_attach_open(self, *_):
        dpg.configure_item("attach_dialog", show=True)

    def _on_attach_picked(self, sender, app_data):
        picks = list((app_data or {}).get("selections", {}).values())
        for p in picks:
            if os.path.isfile(p) and p not in self.attachments:
                self.attachments.append(p)
        self._update_attach_row()

    def on_attach_clear(self, *_):
        self.attachments = []
        self._update_attach_row()

    def _update_attach_row(self):
        names = ", ".join(os.path.basename(p) for p in self.attachments)
        if dpg.does_item_exist("attach_label"):
            dpg.set_value("attach_label", f"attached: {names}"[:120])
        if dpg.does_item_exist("attach_row"):
            dpg.configure_item("attach_row", show=bool(self.attachments))

    _TEXT_ATTACH_EXT = (".csv", ".txt", ".md", ".json", ".va", ".py", ".log")

    def _attachment_context(self):
        """Context block describing queued attachments. Agentic backends read
        the files themselves; chat-only (SDK) backends get small text files
        inlined since they have no Read tool."""
        if not self.attachments:
            return ""
        lines = ["", "ATTACHED FILES (provided by the user for THIS request):"]
        inline = self.agent.backend == "sdk"
        for p in self.attachments:
            lines.append(f"- {p}")
            ext = os.path.splitext(p)[1].lower()
            if inline:
                if ext in self._TEXT_ATTACH_EXT:
                    try:
                        with open(p, "r", encoding="utf-8",
                                  errors="replace") as f:
                            body = f.read(16000)
                        lines.append(f"--- content of {os.path.basename(p)} "
                                     f"(may be truncated) ---\n{body}\n---")
                    except OSError as e:
                        lines.append(f"  (could not read: {e})")
                else:
                    lines.append("  (binary file - NOT readable on this "
                                 "chat-only backend; tell the user to use the "
                                 "Claude provider for pdf/image analysis)")
        if not inline:
            lines.append("Read them with your Read tool (it understands PDF, "
                         "images, CSV and text). Typical jobs: extract device "
                         "parameters / measured curves from a paper or "
                         "datasheet and retune the matching twin + .va, or "
                         "compare measured CSV data against the simulation.")
        return "\n".join(lines)

    # -------------------------------------------------------------------

    def on_send(self, *_):
        if self.chat_busy:
            return
        text = (dpg.get_value("chat_input") or "").strip()
        if not text:
            return
        dpg.set_value("chat_input", "")
        shown = text
        if self.attachments and not text.startswith("/"):
            shown += "\n[attached: " + ", ".join(
                os.path.basename(p) for p in self.attachments) + "]"
        self.append_chat("you", shown)
        if text.startswith("/"):
            self._handle_command(text)
            return
        self._nt_agent_rounds = 0           # fresh user turn resets the auto-loop
        self._nt_best_acc = 0.0
        self._nt_no_improve = 0
        self._nt_epoch_bumps = 0            # reset the "stop raising epochs" nudge
        self._main_send(text)

    def _main_send(self, text):
        """Send `text` to the agent (the caller already showed the user bubble).
        Also used programmatically, e.g. to feed training metrics back."""
        if self.chat_busy:
            return False
        self.chat_busy = True
        dpg.configure_item("btn_send", enabled=False)
        self._chat_pending(True)
        ctx = self._agent_context() + self._attachment_context()
        self.attachments = []
        self._update_attach_row()
        auto = (dpg.get_value("menu_auto")
                if dpg.does_item_exist("menu_auto") else True)
        edits = bash = auto
        model = self.agent_model_id or "default"
        if edits:                       # snapshot before the agent can edit
            self._backup_editable()
            dpg.configure_item("btn_send", show=False)
            dpg.configure_item("btn_stop", show=True)

        def prog(label):                # live tool-activity feedback
            self.q.put(("chat_progress", label))

        def worker():
            # always emit a terminal ("chat", ...) message, even on an
            # unexpected error, so chat_busy can't stay True until restart.
            try:
                res = self.agent.send(text, ctx, allow_edits=edits,
                                      allow_bash=bash, model=model,
                                      autonomous=auto, on_progress=prog)
            except Exception as e:
                res = {"ok": False, "text": "",
                       "error": f"{type(e).__name__}: {e}"}
            self.q.put(("chat", res, edits or bash))

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _on_chat_done(self, res, may_have_edited):
        self.chat_busy = False
        dpg.configure_item("btn_send", enabled=True, show=True)
        dpg.configure_item("btn_stop", show=False)
        self._chat_pending(False)
        text = res.get("text") or ""
        wf = self.agent.extract_waveform(text)
        action = self.agent.extract_action(text)
        nt_actions = self._extract_nt_actions(text)   # Neuro Trainer actions
        if res.get("ok"):
            shown = text
            if wf or action or nt_actions:   # hide the json control blocks
                shown = re.sub(r"```(?:json)?\s*\{.*?\}\s*```",
                               "", shown, flags=re.S).strip()
                shown = shown or (f"Here is the pattern — {wf['label']}."
                                  if wf else "Running it now.")
            self.append_chat("agent", shown or "(empty reply)")
            if wf:
                self._add_pattern_card(wf)
            if action:
                self._run_agent_action(action)
            if nt_actions:
                self._run_nt_actions(nt_actions)
        else:
            self.append_chat("err", res.get("error") or "unknown error")
        self._agent_turns += 1          # session usage is shown via /cost
        if may_have_edited:
            self._check_external_edits()

    def _run_agent_action(self, action):
        """Execute a GUI action the agent asked for (so it can actually run
        plots in the app instead of scripting them).  `action` is the action
        dict ({"action": "...", ...}); a bare string is also accepted."""
        if isinstance(action, str):
            action = {"action": action}
        name = action.get("action")
        if name == "sweep":
            self._apply_agent_sweep(action)
            return
        actions = {
            "run": lambda: self.on_run(),
            "plot_stdp": lambda: self.on_plot_stdp(),
            "plot_polar": lambda: self.on_plot_polar(),
            "preview": lambda: self.on_preview(),
            "analyze_g": lambda: self.on_analysis_metric("G"),
            "analyze_r": lambda: self.on_analysis_metric("R"),
            "fit": lambda: self.fit_axes(),
            "export_csv": lambda: self.export_csv(),
        }
        fn = actions.get(name)
        if fn:
            self.log(f"[agent] running GUI action: {name}")
            fn()
        else:
            self.log(f"[agent] unknown GUI action: {name}")

    # common synonyms the agent might use for geometry params -> canonical
    _SWEEP_ALIAS = {"length": "l", "len": "l", "channel_length": "l",
                    "width": "w", "wid": "w", "channel_width": "w",
                    "diffusion": "d", "thickness": "tox"}

    def _resolve_sweep_param(self, name, valid):
        """Map an agent-supplied param name to a real model parameter, tolerant
        of case ('L'->'l', 'W'->'w') and a few common synonyms. Returns the
        canonical name, or None if it can't be matched."""
        if not name:
            return None
        name = str(name).strip()
        if name in valid:
            return name
        low = {v.lower(): v for v in valid}      # case-insensitive lookup
        if name.lower() in low:
            return low[name.lower()]
        alias = self._SWEEP_ALIAS.get(name.lower())
        if alias and alias in low:
            return low[alias]
        return None

    def _apply_agent_sweep(self, action):
        """Agent-driven parameter sweep: set the sweep spec(s) then run the plot.
        Accepts {"sweeps":[{"param","values"|"from"/"to"/"steps"}], "mode":..}
        or a single {"param","values",...,"mode"}."""
        sweeps = action.get("sweeps")
        if not sweeps and action.get("param"):
            sweeps = [action]
        valid = list(self._sweep_param_names())
        specs = []
        unknown = []
        for s in sweeps or []:
            p = self._resolve_sweep_param(s.get("param"), valid)
            if not p:
                if s.get("param"):
                    unknown.append(str(s.get("param")))
                continue
            if s.get("from") is not None and s.get("to") is not None:
                specs.append({"param": p, "type": "range",
                              "from": float(s["from"]), "to": float(s["to"]),
                              "step_type": s.get("step_type", "Linear"),
                              "steps": int(s.get("steps", 5))})
                continue
            raw = s.get("values")
            if isinstance(raw, (list, tuple)):
                vals = [float(v) for v in raw if isinstance(v, (int, float))]
            else:
                vals = self._parse_eng_list(str(raw))
            if vals:
                specs.append({"param": p, "type": "values", "values": vals})
        if not specs:
            self.log("[agent] sweep: no valid (param, values) for the active "
                     f"model (params: {sorted(valid)})")
            hint = ""
            if unknown:
                hint = (f" Unknown param(s): {', '.join(unknown)}. "
                        f"This model's parameters are: {', '.join(sorted(valid))}.")
            self.append_chat("err", "Sweep had no valid parameter/values for "
                             "the selected model." + hint)
            return
        self.sweep_specs = specs
        if dpg.does_item_exist("sweep_spec_list"):
            self._rebuild_spec_list()
        mode = (action.get("mode") or "run").lower()
        self.log(f"[agent] sweep {self._sweep_combos().__len__()} curves -> {mode}")
        if mode in ("stdp", "plot_stdp"):
            self.on_plot_stdp()
        elif mode in ("polar", "polarization", "pv", "plot_polar"):
            self.on_plot_polar()
        else:
            self.on_run()

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
        for t in self._chat_tex:                 # free shaped-message textures
            if dpg.does_item_exist(t):
                dpg.delete_item(t)
        self._chat_tex = []
        self.append_chat("agent", "Conversation reset.")
        self.log("[agent] conversation reset")

    def on_restart_app(self):
        self.log("[app] restarting to apply GUI-code changes...")
        self._restart = True
        dpg.stop_dearpygui()

    def on_agent_stop(self):
        self._nt_agent_run = False          # also halt the trainer auto-loop
        self._nt_agent_rounds = 99
        if self.agent.stop():
            self.log("[agent] stop requested - terminating run")
            self.append_chat("sys", "Stopping the agent...")
        else:
            self.log("[agent] nothing running to stop")

    # ---- live re-plot on code change --------------------------------

    def _reload_models(self):
        """Hot-reload the edited model twins so the GUI simulates new code.

        Parameter values are refreshed too: anything still at the OLD class
        default adopts the NEW default (so edited defaults in the twin take
        effect), while values the user explicitly changed in the panel are
        kept.  Without this, sims after a reload silently run new code with
        the stale startup parameter snapshot."""
        for key, (mod, clsname, pname) in RELOAD_MODULES.items():
            try:
                old_defaults = _defaults_of(SPEC_BY_KEY[key].params_cls)
                _reload_module(mod)
                spec = SPEC_BY_KEY[key]
                spec.cls = getattr(mod, clsname)
                spec.params_cls = getattr(mod, pname)
                new_defaults = _defaults_of(spec.params_cls)
                cur = self.param_values.get(key, {})
                self.param_values[key] = {
                    name: (cur[name] if name in cur and name in old_defaults
                           and cur[name] != old_defaults[name] else new_val)
                    for name, new_val in new_defaults.items()}
            except Exception as e:                   # one bad twin must not
                self.log(f"[reload] {key} failed: {e!r}")   # break the others
        importlib.reload(analysis)        # hot-swap the measurement layer too
        importlib.reload(stdp_protocol)
        self.rebuild_param_panel()

    def _twin_mtime_map(self):
        m = {}
        for rel in WATCH_FILES:
            p = os.path.join(self.workdir, rel)
            try:
                m[rel] = os.path.getmtime(p)
            except OSError:
                m[rel] = 0.0
        return m

    def _watch_code(self):
        """Poll the model twins + the analysis layer; when one changes (and is
        stable across two polls), hot-reload and re-plot so the view tracks the
        edit with no restart."""
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
        # _twin_mtimes is the currently-LOADED baseline; _prev_mtimes is only the
        # previous poll. Reload when the folder has been STABLE for a poll
        # (cur == prev, so we don't read a half-written file) AND differs from the
        # loaded baseline. (The old code swapped both every poll, so it only fired
        # on an a->b->a flicker and never on a real, persistent edit.)
        stable = cur == self._prev_mtimes
        changed = cur != self._twin_mtimes
        self._prev_mtimes = cur
        if changed and stable and not self.sim_running:
            self._twin_mtimes = cur     # adopt the new baseline
            self.log("[live] code changed (twin/analysis) - reloading + re-plotting")
            try:
                self._reload_models()
            except Exception as e:
                self.log(f"[live] reload failed (syntax error?): {e}")
                return
            # re-run whatever the user last computed (STDP sweep or transient)
            if self._last_compute == "stdp" and self._stdp_ctx is not None:
                self.on_plot_stdp(live=True)
            else:
                self.on_run(live=True)
            # refresh the per-pulse Analysis curve from existing results so edits
            # to analysis.per_pulse_samples show live without a re-sim
            if self.results:
                self.update_analysis()

    # ---- backup / revert agent edits --------------------------------

    def _editable_files(self):
        files = list(TWIN_FILE.values()) + ["vatester/analysis.py"]
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

    def on_account_install(self):
        """Open the official Claude Code install page (license-respecting - we
        don't redistribute Anthropic's CLI; the user installs it from source)."""
        import webbrowser
        url = "https://claude.com/claude-code"
        try:
            webbrowser.open(url)
            self._account_busy(f"opened {url} - install Claude Code, then click "
                               "'Log in'.")
        except Exception as e:                          # noqa: BLE001
            self._account_busy(f"couldn't open a browser - go to {url}  ({e})")

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

    def on_openai_key_apply(self):
        val = (dpg.get_value("openai_key_in") or "").strip()
        if not val:
            self._account_busy("paste an OpenAI API key first, then Apply.")
            return
        self.agent.set_openai_key(val)
        self._refresh_agent_status()
        self.log("[account] OpenAI key set for this app "
                 f"(...{val[-4:]}); switch with /provider openai")
        self._account_busy(f"OpenAI key applied (...{val[-4:]}). "
                           "Use /provider openai in the chat.")

    def on_openai_key_clear(self):
        self.agent.set_openai_key(None)
        dpg.set_value("openai_key_in", "")
        self._refresh_agent_status()
        self.log("[account] OpenAI key cleared")

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

    # =================================================================
    # Network Studio - device crossbars and LIF neurons, trained through
    # device-local STDP or surrogate gradients.
    # =================================================================


    def _build_neuro_dialogs(self):
        """Texture registry + dataset file pickers for the trainer (created in
        build(); the trainer UI itself lives in the main window's center tab)."""
        dpg.add_texture_registry(tag="nt_tex_reg")
        with dpg.file_dialog(directory_selector=False, show=False, modal=True,
                             width=640, height=420, tag="nt_ds_file_dialog",
                             callback=self._on_nt_ds_file,
                             default_path=self.workdir):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".npz", color=(220, 200, 140, 255))
            dpg.add_file_extension(".csv", color=(150, 220, 160, 255))
            dpg.add_file_extension(".gz", color=(150, 180, 255, 255))
        dpg.add_file_dialog(directory_selector=True, show=False, modal=True,
                            width=640, height=420, tag="nt_ds_dir_dialog",
                            callback=self._on_nt_ds_dir,
                            default_path=self.workdir)
        mdir = os.path.join(self.workdir, "results", "models")
        os.makedirs(mdir, exist_ok=True)
        with dpg.file_dialog(directory_selector=False, show=False, modal=True,
                             width=640, height=420, tag="nt_save_dialog",
                             callback=self._on_nt_save_file, default_path=mdir,
                             default_filename="neuro_model"):
            dpg.add_file_extension(".json", color=(220, 200, 140, 255))
        os.makedirs(self._nt_power_report_dir, exist_ok=True)
        with dpg.file_dialog(directory_selector=False, show=False, modal=True,
                             width=640, height=420, tag="nt_power_save_dialog",
                             callback=self._on_nt_power_save_file,
                             default_path=self._nt_power_report_dir,
                             default_filename="energy_report"):
            dpg.add_file_extension(".txt", color=(150, 220, 160, 255))
        with dpg.file_dialog(directory_selector=False, show=False, modal=True,
                             width=640, height=420, tag="nt_load_dialog",
                             callback=self._on_nt_load_file, default_path=mdir):
            dpg.add_file_extension(".json", color=(220, 200, 140, 255))
            dpg.add_file_extension(".*")

    def _trainer_tab(self):
        """The Neuromorphic Trainer, embedded as a center tab in the main
        window (synapse device = the model checked in the left panel; the
        right-side Claude chat is the agent)."""
        with self._pad(left=8, top=6, bottom=0):
            with dpg.group(horizontal=True):
                b = dpg.add_button(label=" Build network ", tag="nt_btn_build",
                                   callback=self.on_nt_build)
                b = dpg.add_button(label=" Train ", tag="nt_btn_train",
                                   callback=self.on_nt_train)
                dpg.bind_item_theme(b, self.themes["primary"])
                b = dpg.add_button(label=" Stop ", callback=self.on_nt_stop)
                dpg.bind_item_theme(b, self.themes["stop"])
                dpg.add_button(label=" Test ", callback=self.on_nt_test)
                options = dpg.add_button(label="Manage network")
                with self.desktop.popup(options):
                    dpg.add_menu_item(label="Save model...", callback=lambda: dpg.show_item("nt_save_dialog"))
                    dpg.add_menu_item(label="Load model...", callback=lambda: dpg.show_item("nt_load_dialog"))
                    dpg.add_separator()
                    dpg.add_menu_item(label="Reset network", callback=self.on_nt_reset)
                    dpg.add_menu_item(label="Restore default settings", callback=self.on_nt_defaults)
                dpg.add_loading_indicator(tag="nt_busy", show=False, radius=2.0,
                                          style=1, color=C_AMBER)
                dpg.add_button(label="● ready", tag="nt_status")
            dpg.add_text("Select a Verilog-A synapse, configure layers, then build.",
                         tag="nt_devlabel", color=C_TEXT2)
            dpg.add_separator()
            # the visualization fills the center; the controls live in the LEFT
            # panel (swapped in for the device-tester sections on this tab)
            with dpg.child_window(width=-1, border=False, no_scrollbar=True,
                                  tag="nt_canvas_col"):
                # grouped into 7 top-level tabs (related views nested) so the
                # bar no longer overflows / truncates
                # tab_list_popup_button = a dropdown that reaches EVERY tab even
                # when the bar is too narrow to show them all (a narrow center
                # panel used to clip the right-hand tabs, making them vanish)
                with dpg.tab_bar(tag="nt_viz_tabs",
                                 tab_list_popup_button=True):
                    with dpg.tab(label="Architecture", tag="ns_architecture_tab"):
                        self.network_studio.architecture()
                    # Data and network views share the same trainer
                    with dpg.tab(label="Data", tag="nt_tab_data"):
                        self._nt_patterns_tab()
                    with dpg.tab(label="Simulation"):
                        self._nt_canvas_tab()
                    with dpg.tab(label="Neuron", tag="nt_tab_neuron"):
                        self._nt_neuron_tab()
                    with dpg.tab(label="Cell", tag="nt_tab_cell"):
                        self._nt_cell_tab()
                    with dpg.tab(label="Weights", tag="nt_tab_weights"):
                        with dpg.tab_bar():
                            with dpg.tab(label="  Maps  "):
                                self._nt_weights_tab()
                            with dpg.tab(label="  Trajectories  ",
                                         tag="nt_tab_wtrace"):
                                self._nt_wtrace_tab()
                    with dpg.tab(label="Raster", tag="nt_tab_raster"):
                        self._nt_raster_tab()
                    with dpg.tab(label="Activity"):
                        with dpg.tab_bar():
                            with dpg.tab(label="  Membrane  "):
                                self._nt_activity_tab()
                            with dpg.tab(label="  Out spikes  "):
                                self._nt_output_tab()
                    with dpg.tab(label="Results", tag="nt_tab_results"):
                        self._nt_result_summary()
                        with dpg.tab_bar(tag="nt_results_tabs"):
                            with dpg.tab(label="  Learning  "):
                                self._nt_learning_tab()
                            with dpg.tab(label="  Metrics  "):
                                self._nt_metrics_tab()
                            with dpg.tab(label="  Power  ", tag="nt_tab_power"):
                                self._nt_power_tab()
                with dpg.group(tag="nt_stream_center", show=False):
                    self._nt_stream_views()
                self._nt_stream_clear()
                dpg.set_item_callback("nt_viz_tabs", self._on_nt_viz_tab)
        # Data is the default tab - populate its live rasters right away
        self._nt_encoding_preview()

    def _nt_wtrace_tab(self):
        with self._pad(left=6, top=4, bottom=0):
            with dpg.group(horizontal=True):
                b = dpg.add_button(label=" Refresh ",
                                   callback=self._nt_draw_weight_traces)
                dpg.bind_item_theme(b, self.themes["primary"])
                self._small("EVERY synapse's conductance (weight) over training "
                            "- the whole crossbar learning at once. Colour = "
                            "destination neuron.", color=(126, 150, 220))
            dpg.add_text("train to populate", tag="nt_wtrace_info",
                         color=C_TEXT2)
            with dpg.plot(width=-1, height=-1, tag="nt_wtrace_plot",
                          no_menus=True):
                dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag="nt_wtrace_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="weight  G (uS)",
                                  tag="nt_wtrace_y")

    def on_model_builder(self, *_):
        """Old integrations now open the single network workspace."""
        self.on_open_trainer()
        dpg.set_value("nt_viz_tabs", "ns_architecture_tab")

    def _on_nt_viz_tab(self, *_):
        if not dpg.does_item_exist("nt_viz_tabs"):
            return
        sel = dpg.get_value("nt_viz_tabs")
        if sel == "nt_tab_weights":          # outer Weights tab (Maps / Trajectories)
            self._nt_draw_weight_traces()
        elif sel == "nt_tab_neuron":
            self._nt_probe_neuron()
        elif sel == "nt_tab_data":
            self._nt_encoding_preview()
        elif sel == "nt_tab_raster" and self._last_present:
            self._nt_draw_raster(*self._last_present)

    def _nt_neuron_tab(self):
        """A current-injection probe: inject a Step / Pulse / Ramp of current (in
        units of rheobase) into one LIF neuron and watch the membrane integrate
        -> spike -> reset, alongside the f-I curve.  The neuron's parameters live
        in the LEFT panel's 'Neuron (LIF)' section."""
        with self._pad(left=6, top=4, bottom=0):
            with self._card(height=490, tag="nt_neuron_card"):
                # header: title + summary  |  Step / Pulse / Ramp toggle (right)
                with dpg.table(header_row=False, borders_innerV=False,
                               policy=dpg.mvTable_SizingStretchProp,
                               pad_outerX=False):
                    dpg.add_table_column()
                    dpg.add_table_column(width_fixed=True,
                                         init_width_or_weight=210)
                    with dpg.table_row():
                        with dpg.group(horizontal=True):
                            t = dpg.add_text("Neuron probe")
                            if "bold" in self.fonts:
                                dpg.bind_item_font(t, self.fonts["bold"])
                            self._small("LIF output neuron - inject current, read "
                                        "membrane & spikes", tag="nt_np_sub",
                                        color=C_MUTED)
                        with dpg.group(horizontal=True):
                            for m in ("step", "pulse", "ramp"):
                                dpg.add_button(label=f" {m.capitalize()} ",
                                               width=62, tag=f"nt_np_mode_{m}",
                                               callback=(lambda s, a, mm=m:
                                                         self._nt_set_probe_mode(mm)))
                dpg.add_spacer(height=2)
                # inject-current slider (in units of rheobase)
                with dpg.group(horizontal=True):
                    self._small("inject I", color=C_TEXT2)
                    dpg.add_slider_float(tag="nt_inject_i", width=-140,
                                         default_value=1.20, min_value=0.0,
                                         max_value=3.0, format="%.2f",
                                         callback=lambda *_: self._nt_probe_neuron())
                    self._small("x rheobase", color=C_MUTED)
                dpg.add_spacer(height=2)
                # four readout tiles
                with dpg.table(header_row=False, borders_innerV=False,
                               policy=dpg.mvTable_SizingStretchSame,
                               pad_outerX=False):
                    for _ in range(5):
                        dpg.add_table_column()
                    with dpg.table_row():
                        self._stat_card("firing rate", "nt_np_rate", "--",
                                        C_GREEN)
                        self._stat_card("spikes", "nt_np_spikes", "0")
                        self._stat_card("I_inj", "nt_np_iinj", "--", C_ACC)
                        self._stat_card("regime", "nt_np_regime", "--", C_ACC)
                dpg.add_spacer(height=4)
                # V(t) and f-I plots, side by side
                with dpg.table(header_row=False, borders_innerV=False,
                               policy=dpg.mvTable_SizingStretchSame,
                               pad_outerX=False):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                        with dpg.group():
                            with dpg.group(horizontal=True):
                                self._small("Membrane potential V(t)  -  spikes "
                                            "at threshold, reset + refractory",
                                            color=C_TEXT2)
                                self._nt_zoom_btns("nt_nv_x", "nt_nv_y")
                            with dpg.plot(width=-1, height=316, no_menus=True,
                                          tag="nt_nv_plot"):
                                dpg.add_plot_legend(
                                    location=dpg.mvPlot_Location_NorthEast)
                                dpg.add_plot_axis(dpg.mvXAxis, label="time (ms)",
                                                  tag="nt_nv_x")
                                dpg.add_plot_axis(dpg.mvYAxis, label="membrane V",
                                                  tag="nt_nv_y")
                        with dpg.group():
                            with dpg.group(horizontal=True):
                                self._small("F-I curve  -  rate vs drive",
                                            color=C_TEXT2)
                                self._nt_zoom_btns("nt_fi_x", "nt_fi_y")
                            with dpg.plot(width=-1, height=316, no_menus=True,
                                          tag="nt_fi_plot"):
                                dpg.add_plot_legend(
                                    location=dpg.mvPlot_Location_NorthWest)
                                dpg.add_plot_axis(dpg.mvXAxis,
                                                  label="I_inj (x rheobase)",
                                                  tag="nt_fi_x")
                                dpg.add_plot_axis(dpg.mvYAxis, label="rate (Hz)",
                                                  tag="nt_fi_y")
        self._nt_set_probe_mode(getattr(self, "_nt_probe_mode", "step"))

    def _nt_zoom_btns(self, xax, yax):
        for lbl, f in ((" - ", 1.30), (" + ", 0.77)):
            dpg.add_button(label=lbl, width=28,
                           callback=lambda s, a, x=xax, y=yax, ff=f:
                           self._nt_zoom_axes(x, y, ff))
        dpg.add_button(label="Fit", width=38,
                       callback=lambda s, a, x=xax, y=yax:
                       self._nt_fit_axes(x, y))

    def _nt_zoom_axes(self, xax, yax, f):
        for ax in (xax, yax):
            try:
                lo, hi = dpg.get_axis_limits(ax)
                c, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * f
                dpg.set_axis_limits(ax, c - half, c + half)
            except Exception:                               # noqa: BLE001
                pass

    def _nt_fit_axes(self, xax, yax):
        for ax in (xax, yax):
            dpg.set_axis_limits_auto(ax)
            dpg.fit_axis_data(ax)

    def _nt_set_probe_mode(self, mode):
        self._nt_probe_mode = mode
        for m in ("step", "pulse", "ramp"):
            tag = f"nt_np_mode_{m}"
            if dpg.does_item_exist(tag):
                dpg.bind_item_theme(tag, self.themes[
                    "primary" if m == mode else "chip"])
        self._nt_probe_neuron()

    def _nt_probe_neuron(self, *_):
        """Inject a Step / Pulse / Ramp current (x rheobase) into one LIF neuron
        and draw V(t) + the f-I curve with the current operating point marked."""
        if not dpg.does_item_exist("nt_nv_y"):
            return
        N, _S, _cfg = self._nt_read_params()
        pms = 200.0                                         # fixed bench window
        dt = max(self._nt_getu("nt_dt", 1.0), 0.1)
        rheo = max(N.v_threshold - N.v_rest, 1e-6)          # min drive to fire
        imult = max(float(self._nt_get("nt_inject_i", 1.2)), 0.0)
        idrive = imult * rheo
        mode = getattr(self, "_nt_probe_mode", "step")
        n = max(1, int(round(pms / dt)))
        if mode == "pulse":
            drive = np.zeros(n)
            drive[int(n * 0.30):int(n * 0.78)] = idrive
        elif mode == "ramp":
            drive = np.linspace(0.0, idrive, n)
        else:                                               # step
            drive = np.full(n, idrive)
        t, V, TH, outs = neuro.probe_lif_wave(N, drive, dt)
        rate = len(outs) / (pms * 1e-3)
        # draw each output spike as a vertical action-potential in the V(t) line
        Vd = V.copy()
        peak = float(max(TH.max(), V.max())) + 0.4
        for ts in outs:
            Vd[min(int(round(ts / dt)), len(Vd) - 1)] = peak
        self._nt_clear("nv")
        tl = t.tolist()
        s_th = dpg.add_line_series(tl, TH.tolist(), label="V_th",
                                   parent="nt_nv_y")
        s_v = dpg.add_line_series(tl, Vd.tolist(), label="V(t)",
                                  parent="nt_nv_y")
        dpg.bind_item_theme(s_th, self.themes["ln_amber"])
        dpg.bind_item_theme(s_v, self.themes["ln_blue"])
        self._nt_series["nv"] = [s_th, s_v]
        self._nt_fit_axes("nt_nv_x", "nt_nv_y")
        # f-I curve (rate vs drive in x rheobase) + current operating point
        drives, rates = neuro.fi_curve(N, pms, dt, 3.0 * rheo)
        xs, ymax = (drives / rheo).tolist(), float(max(rates.max(), 1.0))
        self._nt_clear("fi")
        s_fi = dpg.add_line_series(xs, rates.tolist(), label="f(I)",
                                   parent="nt_fi_y")
        s_ii = dpg.add_line_series([imult, imult], [0.0, ymax], label="I_inj",
                                   parent="nt_fi_y")
        dpg.bind_item_theme(s_fi, self.themes["ln_green"])
        dpg.bind_item_theme(s_ii, self.themes["ln_blue"])
        self._nt_series["fi"] = [s_fi, s_ii]
        self._nt_fit_axes("nt_fi_x", "nt_fi_y")
        # regime label
        if rate <= 0:
            regime = "subthreshold"
        elif mode == "ramp":
            regime = "adapting"
        elif len(outs) == 1:
            regime = "single"
        else:
            regime = "tonic"
        self._nt_stat("nt_np_rate", f"{rate:.0f} Hz")
        self._nt_stat("nt_np_spikes", str(len(outs)))
        self._nt_stat("nt_np_iinj", f"{imult:.2f}")
        self._nt_stat("nt_np_regime", regime)
        if dpg.does_item_exist("nt_np_sub"):
            dpg.set_value("nt_np_sub",
                          f"LIF - tau_m {N.tau_m:.0f} ms - V_th "
                          f"{N.v_threshold:.2g} - inject current, read membrane "
                          f"& spikes")

    @contextlib.contextmanager
    def _ctl_table(self, label_w=90):
        """A borderless 2-col form table: fixed label column + stretch input."""
        with dpg.table(header_row=False, borders_innerH=False,
                       borders_outerH=False, borders_innerV=False,
                       borders_outerV=False, pad_outerX=False,
                       policy=dpg.mvTable_SizingStretchProp) as t:
            dpg.add_table_column(width_fixed=True, init_width_or_weight=label_w)
            dpg.add_table_column()
            yield t

    def _ctl_row(self, label, tag=None, kind="double", default=0.0, tip=None,
                 items=None, dv=None, callback=None, hint=None, unit=None):
        """One aligned 'label | input' row inside an open _ctl_table.  When
        `unit` (a dimension: current/voltage/time/freq/amp) is given, the input
        shares its cell with a small unit dropdown."""
        with dpg.table_row():
            cap = self._small(label, color=C_TEXT2)
            if kind == "int":
                w = dpg.add_input_int(tag=tag, default_value=int(default),
                                      width=-1, step=0, callback=callback)
            elif kind == "combo":
                w = dpg.add_combo(items or [], default_value=dv, tag=tag,
                                  width=-1, callback=callback)
            elif kind == "text":
                w = dpg.add_input_text(tag=tag, width=-1, hint=hint or "",
                                       callback=callback)
            elif unit:
                with dpg.group(horizontal=True):
                    w = dpg.add_input_double(tag=tag, default_value=float(default),
                                             width=-62, format="%.4g", step=0,
                                             callback=callback)
                    self._nt_unit_combo(tag, unit)
            else:
                w = dpg.add_input_double(tag=tag, default_value=float(default),
                                         width=-1, format="%.4g", step=0,
                                         callback=callback)
            if tip:
                for tg in (cap, w):
                    with dpg.tooltip(tg):
                        dpg.add_text(tip, wrap=300)

    # ---- compact card helpers ---------------------------------------

    @contextlib.contextmanager
    def _card(self, title=None, tag=None, width=-1, height=0, tip=None):
        """A rounded surface panel (web-style card) with an optional title."""
        kw = {}
        if tag:
            kw["tag"] = tag
        with dpg.child_window(width=width, height=height, border=True,
                              autosize_y=(height == 0), **kw) as cw:
            dpg.bind_item_theme(cw, self.themes["card"])
            if title:
                h = self._small(title.upper(), color=(126, 150, 220))
                if tip:
                    with dpg.tooltip(h):
                        dpg.add_text(tip, wrap=320)
                dpg.add_spacer(height=4)
            yield cw

    def _stat_card(self, label, tag, value="--", color=None):
        """A compact metric tile: small caps label over a big value."""
        with dpg.child_window(width=-1, height=82, border=True, no_scrollbar=True) as cw:
            dpg.bind_item_theme(cw, self.themes["card"])
            self._small(label.upper(), color=C_MUTED)
            v = dpg.add_text(value, tag=tag, color=color or C_TEXT)
            if "h2" in self.fonts:
                dpg.bind_item_font(v, self.fonts["h2"])

    def _nt_stat(self, tag, value):
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, value)

    def _nt_source_card(self, kind, title):
        """One clickable DATA SOURCE radio card: a dot + title + live subtitle.
        Child windows can't take a clicked-handler, so a single global mouse
        handler (added once) routes a click to whichever card is hovered."""
        with dpg.child_window(width=-1, height=80, border=True, no_scrollbar=True,
                              tag=f"nt_srccard_{kind}") as cw:
            dpg.bind_item_theme(cw, self.themes["card"])
            with dpg.group(horizontal=True):
                dpg.add_text("O", tag=f"nt_srcdot_{kind}", color=C_MUTED)
                with dpg.group():
                    t = dpg.add_text(title, color=C_TEXT)
                    if "bold" in self.fonts:
                        dpg.bind_item_font(t, self.fonts["bold"])
                    self._small("-", tag=f"nt_srcsub_{kind}", color=C_MUTED)
        if not getattr(self, "_nt_src_click_bound", False):
            self._nt_src_click_bound = True
            with dpg.handler_registry():
                dpg.add_mouse_click_handler(
                    button=dpg.mvMouseButton_Left,
                    callback=self._nt_source_click)

    def _nt_source_click(self, *_):
        if dpg.does_item_exist("nt_source_selector"):
            dpg.set_value("nt_source_selector", {"set": "Pattern set", "custom": "Custom painted", "dataset": "Imported"}[active])
            dpg.configure_item("nt_source_setup", enabled=active != "set")
            dpg.set_value("nt_source_summary", subs[active])
        for kind in ("set", "custom", "dataset"):
            tag = f"nt_srccard_{kind}"
            if dpg.does_item_exist(tag) and dpg.is_item_hovered(tag):
                self._nt_set_source(kind)
                return

    def _nt_set_source(self, kind, *_):
        """DATA SOURCE cards: point the 'patterns' combo at the chosen source."""
        if not dpg.does_item_exist("nt_patset"):
            return
        if kind == "custom":
            dpg.set_value("nt_patset", "custom")
        elif kind == "dataset":
            dpg.set_value("nt_patset", "dataset")
        else:                                   # a built-in pattern set
            cur = dpg.get_value("nt_patset")
            if cur in ("custom", "dataset"):
                dpg.set_value("nt_patset", self._nt_last_patset or "bars")
        self._nt_refresh_source()
        self._nt_encoding_preview()

    def _patset_class_count(self, name, gh, gw):
        """How many classes the named pattern set has (for the card subtitle)."""
        try:
            if name in ("bars", "letters", "digits", "random"):
                return len(neuro.make_patterns(name, gh, gw))
            if name in PATTERN_PLUGINS:
                pats, _ = PATTERN_PLUGINS[name]["make"](self._nt_read_params()[2])
                return len(pats)
        except Exception:                                   # noqa: BLE001
            pass
        return None

    def _nt_refresh_source(self):
        """Highlight the active DATA SOURCE card and refresh every subtitle."""
        if not dpg.does_item_exist("nt_patset"):
            return
        cur = dpg.get_value("nt_patset")
        if cur not in ("custom", "dataset"):
            self._nt_last_patset = cur
        active = ("custom" if cur == "custom"
                  else "dataset" if cur == "dataset" else "set")
        for source, tag in (("custom", "nt_custom_card"), ("dataset", "nt_dataset_card")):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=active == source)
        gh = int(self._nt_get("nt_gh", 5))
        gw = int(self._nt_get("nt_gw", 5))
        n = self._patset_class_count(self._nt_last_patset, gh, gw)
        subs = {
            "set": (f"{self._nt_last_patset}"
                    + (f"  -  {n} classes" if n else "")),
            "custom": f"{len(self._nt_custom)} patterns",
            "dataset": getattr(self, "_nt_ds_label", None) or "none loaded",
        }
        if dpg.does_item_exist("nt_source_selector"):
            dpg.set_value("nt_source_selector", {"set": "Pattern set", "custom": "Custom painted", "dataset": "Imported"}[active])
            dpg.configure_item("nt_source_setup", enabled=active != "set")
            dpg.set_value("nt_source_summary", subs[active])
        for kind in ("set", "custom", "dataset"):
            on = (kind == active)
            if dpg.does_item_exist(f"nt_srccard_{kind}"):
                dpg.bind_item_theme(f"nt_srccard_{kind}",
                                    self.themes["card_active" if on else "card"])
            if dpg.does_item_exist(f"nt_srcdot_{kind}"):
                dpg.set_value(f"nt_srcdot_{kind}", "(*)" if on else "O")
                dpg.configure_item(f"nt_srcdot_{kind}",
                                   color=C_ACC if on else C_MUTED)
            if dpg.does_item_exist(f"nt_srcsub_{kind}"):
                dpg.set_value(f"nt_srcsub_{kind}", subs[kind])

    def _nt_controls(self):
        with dpg.collapsing_header(label="Network", default_open=True):
            self.network_studio.source_controls()
            dpg.add_combo([s.label for s in MODEL_SPECS],
                          default_value=MODEL_SPECS[1].label, tag="nt_device",
                          width=-1, callback=self._on_nt_device_change, show=False)
            with dpg.tooltip("nt_device"):
                dpg.add_text("Which device twin is the synapse at every crossbar "
                             "cross-point (its conductance = the weight). ECFET = "
                             "current-driven, FeFET = voltage-driven.", wrap=300)
            with self._ctl_table():
                self._ctl_row("mode", "nt_mode", "combo",
                              items=["supervised", "unsupervised", STREAM_MODE],
                              dv="supervised", callback=self._on_nt_mode, tip=(
                    "Sets the whole window. supervised: a teacher current "
                    "forces each pattern's neuron to fire. unsupervised: the "
                    "same pattern-by-pattern presentations without a teacher "
                    "(winner-take-all). unsupervised stream: one continuous "
                    "spike stream with hidden recurring patterns and STDP at "
                    "every spike (Masquelier)."))
                self._ctl_row("seed", "nt_seed", "int", 1,
                              "Random seed - fixes spike noise, weight init and "
                              "sampling so a run repeats.")
            with dpg.group(tag="nt_stream_group", show=False):
                self._nt_stream_left()
            with dpg.group(tag="nt_grid_net"), self._ctl_table():
                with dpg.table_row():
                    cap = self._small("input grid", color=C_TEXT2)
                    with dpg.group(horizontal=True):
                        dpg.add_input_int(tag="nt_gh", default_value=5, width=54,
                                          step=0)
                        self._small("x", color=C_MUTED)
                        dpg.add_input_int(tag="nt_gw", default_value=5, width=54,
                                          step=0)
                    with dpg.tooltip(cap):
                        dpg.add_text("Input is an H x W pixel grid; each pixel = "
                                     "one INPUT neuron, so inputs = H x W "
                                     "(5 x 5 = 25). First box H, second W.",
                                     wrap=300)
                self._ctl_row("outputs", "nt_nout", "int", 4,
                              "OUTPUT (classifier) neurons - crossbar columns, "
                              "usually one per class.")
                # Shared architecture value also used by the bridge and agent.
                with dpg.table_row(show=False):
                    dpg.add_input_text(tag="nt_hidden", default_value="", show=False)
                self._ctl_row("hidden gain", "nt_hidden_gain", "double", 1.7,
                              "Extra drive on hidden layers (no teacher). Raise "
                              "if hidden layers stay quiet.")
                self._ctl_row("learning rule", "nt_learnrule", "combo",
                              items=["STDP (device-local)",
                                     "Surrogate grad (BPTT)"],
                              dv="STDP (device-local)",
                              callback=lambda *_: self._nt_sync_rule(), tip=(
                    "STDP: local device-physics plasticity (weak on deep nets). "
                    "Surrogate grad (BPTT): supervised backprop through the "
                    "device - trains deep nets to high accuracy."))
                self._ctl_row("SG rate", "nt_sg_lr", "double", 0.1,
                              "Surrogate-gradient learning rate (BPTT step). "
                              "0.05-0.2 is useful.")
                self._ctl_row("epochs", "nt_epochs", "int", 40,
                              "Passes over the full pattern set.")
                self._ctl_row("present", "nt_present", "double", 120,
                              "Duration each pattern is shown for.", unit="time")
                self._ctl_row("rate", "nt_rate", "double", 180,
                              "Peak Poisson spike rate for a fully-on pixel.",
                              unit="freq")
                self._ctl_row("step", "nt_dt", "double", 1.0,
                              "Integration time-step.", unit="time")
        with dpg.collapsing_header(label="Synapse write pulses and STDP",
                                   default_open=True):
            self._small("the network rule sets the DIRECTION of each write; the "
                        "device twin sets the ACTUAL dG.", color=C_MUTED, wrap=340)
            with self._ctl_table():
                self._ctl_row("pot amp", "nt_potamp", "double", 50,
                              "Amplitude of a full potentiating pulse, in the "
                              "device's drive unit (pA ECFET / V FeFET).",
                              unit="amp")
                self._ctl_row("dep amp", "nt_depamp", "double", 50,
                              "Amplitude of a full depressing pulse.", unit="amp")
                self._ctl_row("pulse", "nt_pwidth", "double", 10,
                              "Programming pulse width.", unit="time")
            with dpg.group(tag="nt_grid_rule"), self._ctl_table():
                self._ctl_row("A+ rate", "nt_aplus", "double", 1.0,
                              "Potentiation learning-rate scale.")
                self._ctl_row("A- rate", "nt_aminus", "double", 1.0,
                              "Heterosynaptic depression rate.")
                self._ctl_row("LTP/LTD split", "nt_offset", "double", 0.25,
                              "Pre-trace split: inputs above this potentiate on a "
                              "post spike, below depress. Carves the RF.")
                self._ctl_row("tau_pre", "nt_taupre", "double", 20,
                              "Pre-synaptic eligibility-trace decay.", unit="time")
        with dpg.collapsing_header(label="Neuron (LIF)", default_open=False,
                                   tag="nt_grid_neuron"):
            self._small("the output/hidden LIF neuron - bench it live in the "
                        "Neuron tab.", color=C_MUTED, wrap=340)
            with self._ctl_table():
                self._ctl_row("tau_m", "nt_tau_m", "double", 20,
                              "Membrane time constant - how fast the potential "
                              "leaks back to rest.", unit="time")
                self._ctl_row("V_th", "nt_vth", "double", 1.0,
                              "Fire when the membrane crosses this (+ the "
                              "adaptive homeostatic theta).")
                self._ctl_row("refrac", "nt_refrac", "double", 5,
                              "Refractory period - dead time after a spike.",
                              unit="time")
                self._ctl_row("EPSP gain", "nt_epsp", "double", 11.0,
                              "How strongly an excitatory input spike (scaled by "
                              "the synapse conductance) depolarises the neuron. "
                              "Activity-normalised, so one value works for 5x5 "
                              "or 28x28.")
                self._ctl_row("IPSP gain", "nt_ipsp", "double", 1.0,
                              "Feed-forward inhibition depth from inhibitory "
                              "inputs (hyperpolarising).")
                self._ctl_row("tau_syn", "nt_tausyn", "double", 8,
                              "Synaptic (PSP) trace decay.", unit="time")
                self._ctl_row("inhib", "nt_inhib", "double", 0.9,
                              "Lateral inhibition: winner-take-all competition "
                              "between output neurons. 0 disables it.")
                self._ctl_row("theta+", "nt_theta", "double", 0.06,
                              "Homeostatic threshold bump per output spike "
                              "(keeps one neuron from hogging every pattern). "
                              "Set 0 for a fixed threshold.")
                self._ctl_row("teacher", "nt_teacher", "double", 1.4,
                              "Supervised teacher drive onto the target neuron.")
        # NOTE: the pattern selector and the spike-encoding + noise controls now
        # live in the Data tab's "Encoding & noise" card (all data setup in one
        # window). Their widget tags are created there.

    def _nt_canvas_tab(self):
        # the "main canvas": just the network diagram, sized to fill the tab in
        # _nt_relayout() so it never needs a scrollbar
        with self._pad(left=6, top=4, bottom=0):
            self._small("single crossbar: input neurons -> wordlines -> ARRAY "
                        "(a 3-terminal device per cross-point; zoom in to see "
                        "source/drain + gate) -> bitlines -> output.   CLICK a "
                        "cell to inspect it in the Cell tab.", color=C_TEXT2)
            with dpg.group(horizontal=True):                  # zoom toolbar
                self._small("zoom", color=C_MUTED)
                for lbl, fn in (("-", lambda: self._nt_set_zoom(self._nt_zoom / 1.25)),
                                ("+", lambda: self._nt_set_zoom(self._nt_zoom * 1.25)),
                                ("Fit", lambda: self._nt_set_zoom(1.0))):
                    dpg.add_button(label=f" {lbl} ", width=34 if lbl != "Fit"
                                   else 44, callback=lambda s, a, f=fn: f())
                dpg.add_text("100%", tag="nt_zoom_lbl", color=C_TEXT2)
                self._small("(or scroll-wheel over the canvas)", color=C_MUTED)
            with dpg.child_window(border=False, tag="nt_canvas_scroll",
                                  horizontal_scrollbar=True):
                Wd, Hd = self._nt_diag_size
                dl = dpg.add_drawlist(width=Wd, height=Hd, tag="nt_diagram")
                dpg.add_draw_layer(parent=dl, tag="nt_diag_layer")
                dpg.draw_text((40, 70), "Press  Build network  to wire the "
                              "crossbar.", size=18, color=(110, 120, 140),
                              parent="nt_diag_layer")
            dpg.add_text("trained on: -", tag="nt_pat_txt", color=C_MUTED)

    def _nt_cell_tab(self):
        with self._pad(left=6, top=4, bottom=0):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Animate read / write", tag="nt_cell_play", default_value=True)
                dpg.add_slider_float(label="Playback speed", tag="nt_cell_speed", default_value=1.,
                                     min_value=.25, max_value=3., width=140, format="%.2fx")
                self._small("Illustration; playback speed does not change the simulation.")
            dpg.add_text("no cell selected", tag="nt_cell_title", color=C_AGENT)
            with dpg.child_window(height=-208, border=False,
                                  tag="nt_cell_holder"):
                cw = dpg.add_drawlist(width=900, height=360, tag="nt_cell_draw")
                dpg.add_draw_layer(parent=cw, tag="nt_cell_layer")
            self._small("THIS synapse's weight (channel conductance) over "
                        "training - the write trajectory from start to end",
                        color=(126, 150, 220))
            with dpg.plot(width=-1, height=-1, tag="nt_cell_plot",
                          no_menus=True):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag="nt_cell_px")
                dpg.add_plot_axis(dpg.mvYAxis, label="G (uS)", tag="nt_cell_py")

    def _nt_weights_tab(self):
        with self._pad(left=6, top=4, bottom=0):
            self._small("RECEPTIVE FIELDS  (each neuron's learned weight map; "
                        "viridis = low->high conductance)", color=(126, 150, 220))
            with dpg.child_window(height=190, border=False,
                                  horizontal_scrollbar=True, tag="nt_rf_holder"):
                pass
            dpg.add_separator()
            with dpg.group(horizontal=True):
                self._small("CROSSBAR CONDUCTANCE MATRIX  (rows = post neurons, "
                            "cols = pre neurons)", color=(126, 150, 220))
                dpg.add_combo([], tag="nt_xbar_sel", width=200, show=False,
                              callback=self._on_nt_xbar_sel)
            with dpg.group(horizontal=True):
                with dpg.plot(width=-92, height=-1, tag="nt_wm_plot",
                              no_menus=True, no_mouse_pos=True):
                    dpg.add_plot_axis(dpg.mvXAxis, label="input pixel",
                                      tag="nt_wm_x", no_menus=True)
                    dpg.add_plot_axis(dpg.mvYAxis, label="neuron",
                                      tag="nt_wm_y", no_menus=True)
                dpg.bind_colormap("nt_wm_plot", dpg.mvPlotColormap_Viridis)
                dpg.add_colormap_scale(min_scale=0.0, max_scale=1.0,
                                       colormap=dpg.mvPlotColormap_Viridis,
                                       width=72, height=-1, tag="nt_wm_cbar",
                                       label="uS")

    def _nt_raster_tab(self):
        """Web-style two-panel spike raster for the last presented pattern:
        INPUT afferent spikes (blue) over the OUTPUT neuron spikes (green),
        both on the presentation time axis, with the winner neuron named."""
        with self._pad(left=6, top=4, bottom=0):
            with self._card(height=352, tag="nt_raster_input_card"):
                with dpg.group(horizontal=True):
                    t = dpg.add_text("Input spike raster")
                    if "bold" in self.fonts:
                        dpg.bind_item_font(t, self.fonts["bold"])
                    self._small("- encoded stimulus with noise + jitter",
                                tag="nt_rc_in_sub", color=C_MUTED)
                with dpg.plot(width=-1, height=300, no_menus=True,
                              tag="nt_rc_in"):
                    dpg.add_plot_axis(dpg.mvXAxis, label="time (ms)",
                                      tag="nt_rc_in_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="afferent #",
                                      tag="nt_rc_in_y")
            dpg.add_spacer(height=6)
            with self._card(height=300, tag="nt_raster_output_card"):
                with dpg.table(header_row=False, borders_innerV=False,
                               policy=dpg.mvTable_SizingStretchProp,
                               pad_outerX=False):
                    dpg.add_table_column()
                    dpg.add_table_column(width_fixed=True,
                                         init_width_or_weight=180)
                    with dpg.table_row():
                        with dpg.group(horizontal=True):
                            t = dpg.add_text("Output spike raster")
                            if "bold" in self.fonts:
                                dpg.bind_item_font(t, self.fonts["bold"])
                            self._small("- winner-take-all", color=C_MUTED)
                        w = dpg.add_text("winner -", tag="nt_rc_win",
                                         color=C_GREEN)
                        if "bold" in self.fonts:
                            dpg.bind_item_font(w, self.fonts["bold"])
                with dpg.plot(width=-1, height=248, no_menus=True,
                              tag="nt_rc_out"):
                    dpg.add_plot_axis(dpg.mvXAxis, label="time (ms)",
                                      tag="nt_rc_out_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="neuron #",
                                      tag="nt_rc_out_y")

    def _nt_draw_raster(self, label, vec, res):
        """Fill the Raster tab from a presentation result (drawn live each epoch
        and after a test).  Each spike is a short vertical tick; one line series
        per panel with NaN gaps between ticks keeps it fast."""
        if not dpg.does_item_exist("nt_rc_in_y") or self.trainer is None:
            return
        dt = res.get("dt_ms", 1.0)
        present_ms = max(res.get("n_steps", 1) * dt, 1.0)
        n_in, n_out = self.trainer.n_in, self.trainer.n_out
        NAN = float("nan")
        # input afferent ticks (blue)
        self._nt_clear("rcin")
        xs, ys = [], []
        for t, i in res.get("in_spikes", []):
            xs += [t, t, NAN]
            ys += [i - 0.42, i + 0.42, NAN]
        s = dpg.add_line_series(xs, ys, label="input", parent="nt_rc_in_y")
        dpg.bind_item_theme(s, self.themes["ln_blue"])
        self._nt_series["rcin"] = [s]
        dpg.set_axis_limits("nt_rc_in_x", 0.0, present_ms)
        dpg.set_axis_limits("nt_rc_in_y", -0.5, max(n_in, 1) - 0.5)
        # output neuron ticks (green)
        self._nt_clear("rcout")
        xo, yo = [], []
        for j, sp in enumerate(res.get("out_spikes", [])):
            for t in sp:
                xo += [t, t, NAN]
                yo += [j - 0.42, j + 0.42, NAN]
        so = dpg.add_line_series(xo, yo, label="output", parent="nt_rc_out_y")
        dpg.bind_item_theme(so, self.themes["ln_green"])
        self._nt_series["rcout"] = [so]
        dpg.set_axis_limits("nt_rc_out_x", 0.0, present_ms)
        dpg.set_axis_limits("nt_rc_out_y", -0.5, max(n_out, 1) - 0.5)
        # captions + winner
        if dpg.does_item_exist("nt_rc_in_sub"):
            dpg.set_value("nt_rc_in_sub",
                          f"{n_in} afferent channels  -  encoded stimulus with "
                          f"noise + jitter")
        if dpg.does_item_exist("nt_rc_win"):
            w = res.get("winner", -1)
            names = getattr(self.trainer, "class_names", [])
            nm = str(names[w]) if 0 <= w < len(names) else ""
            dpg.set_value("nt_rc_win",
                          f"winner {nm}  (N{w:02d})" if w >= 0 else "winner -")

    def _nt_activity_tab(self):
        with self._pad(left=6, top=4, bottom=0):
            dpg.add_text("output-neuron MEMBRANE during the last presentation - "
                         "winner bold, amber = threshold, dots = spikes  (the "
                         "input/output spike rasters are on the Raster tab)",
                         tag="nt_act_title", color=C_TEXT2)
            with dpg.plot(width=-1, height=-1, tag="nt_mem_plot", no_menus=True):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="time (ms)",
                                  tag="nt_mem_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="membrane V (a.u.)",
                                  tag="nt_mem_y")

    def _nt_result_summary(self):
        with dpg.table(tag="nt_result_summary", header_row=False,
                       policy=dpg.mvTable_SizingStretchSame, pad_outerX=False):
            for _ in range(5):
                dpg.add_table_column()
            with dpg.table_row():
                for label, tag, value, color in (
                    ("Epoch", "nt_stat_epoch", "0", C_TEXT),
                    ("Train", "nt_stat_tracc", "--", C_GREEN),
                    ("Validation", "nt_stat_valacc", "--", C_ACC),
                    ("Test", "nt_stat_teacc", "--", C_ACC),
                    ("Best", "nt_stat_best", "--", C_AMBER)):
                    with dpg.group():
                        self._small(label, color=C_MUTED)
                        dpg.add_text(value, tag=tag, color=color)

    def _nt_learning_tab(self):
        with self._pad(left=6, top=4, bottom=0):
            self._small("ACCURACY vs EPOCH  (live; train = solid, held-out test "
                        "where available)", color=(126, 150, 220))
            with dpg.plot(width=-1, height=-230, tag="nt_acc_plot",
                          no_menus=True):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag="nt_acc_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="accuracy (%)",
                                  tag="nt_acc_y")
            self._small("WEIGHT EVOLUTION  (mean synaptic weight per output "
                        "neuron vs epoch)", color=(126, 150, 220))
            with dpg.plot(width=-1, height=-1, tag="nt_wevo_plot",
                          no_menus=True):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag="nt_wevo_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="mean weight (uS)",
                                  tag="nt_wevo_y")

    def _nt_metrics_tab(self):
        with self._pad(left=6, top=4, bottom=0):
            with dpg.group(horizontal=True):
                b = dpg.add_button(label=" Evaluate ", callback=self.on_nt_test)
                dpg.bind_item_theme(b, self.themes["primary"])
                self._small("classification metrics on the trained network "
                            "(train + held-out test for datasets). Run after "
                            "training.", color=C_TEXT2)
            dpg.add_text("run Train, then Evaluate", tag="nt_metrics_summary",
                         wrap=1000, color=C_TEXT)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                with dpg.group():
                    self._small("CONFUSION MATRIX  (rows = true, cols = "
                                "predicted)", color=(126, 150, 220))
                    with dpg.plot(width=430, height=-1, tag="nt_cm_plot",
                                  no_menus=True, no_mouse_pos=True):
                        dpg.add_plot_axis(dpg.mvXAxis, label="predicted",
                                          tag="nt_cm_x", no_menus=True)
                        dpg.add_plot_axis(dpg.mvYAxis, label="true",
                                          tag="nt_cm_y", no_menus=True)
                    dpg.bind_colormap("nt_cm_plot", dpg.mvPlotColormap_Viridis)
                with dpg.group():
                    self._small("PER-CLASS METRICS", color=(126, 150, 220))
                    with dpg.child_window(width=-1, height=-1,
                                          tag="nt_metrics_holder",
                                          border=False):
                        pass

    NT_POWER_HINT = ("build a network, then Train / Evaluate to see its "
                     "energy breakdown")

    def _nt_power_tab(self):
        head = (126, 150, 220)
        with self._pad(left=6, top=4, bottom=0):
            with dpg.group(horizontal=True):
                self._small("NEURON / PERIPHERAL POWER", color=C_TEXT2)
                dpg.add_combo(self._nt_power_choices(), tag="nt_power_profile",
                              default_value=neuro.PROFILE_DEFAULT, width=220,
                              callback=lambda *_: self._on_nt_power_profile())
                dpg.add_button(label="Refresh list",
                               callback=lambda *_: self._nt_refresh_power_choices())
                dpg.add_button(label="Save report (.txt)...",
                               callback=lambda *_: self._on_nt_power_save_click())
            self._small("Default = the selected device's own profile "
                        "(ecfet.json for ECFET, fefet.json for FeFET), else "
                        "power_profiles/default.json if you create one, else "
                        "placeholders. Changing it re-prices the last run.",
                        color=C_TEXT2)
            dpg.add_separator()
            with dpg.child_window(width=-1, height=-1, border=False):
                self._small(self.NT_POWER_HINT, tag="nt_power_note",
                            color=C_TEXT2)
                dpg.add_text("", tag="nt_power_warn", color=C_AMBER, wrap=0,
                             show=False)
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=440, height=300, border=False):
                        self._small("KEY FIGURES", color=head)
                        dpg.add_group(tag="nt_power_keys")
                    with dpg.group():
                        self._small("WHERE THE ENERGY GOES  (share of each "
                                    "phase's total)", color=head)
                        with dpg.plot(width=-1, height=280, tag="nt_power_plot",
                                      no_menus=True, no_mouse_pos=True):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, tag="nt_power_x",
                                              no_menus=True)
                            dpg.add_plot_axis(dpg.mvYAxis, label="share (%)",
                                              tag="nt_power_y", no_menus=True)
                self._small("BREAKDOWN  (energy per component; % of the "
                            "phase total)", color=head)
                dpg.add_group(tag="nt_power_table")
                dpg.add_spacer(height=6)
                with dpg.collapsing_header(label="Full text report",
                                           default_open=False):
                    t = dpg.add_text(self.NT_POWER_HINT, tag="nt_power_txt",
                                     color=C_TEXT)
                    if "mono" in self.fonts:
                        dpg.bind_item_font(t, self.fonts["mono"])

    def _nt_show_power(self, text):
        """Show the last run's energy: text report (kept for Save), key
        figures, component chart and breakdown table. Called on the render
        thread once the trainer is idle."""
        if not text:
            return
        self._nt_power_report = text
        if dpg.does_item_exist("nt_power_txt"):
            dpg.set_value("nt_power_txt", text)
        tr = self.trainer
        try:
            reports = ({} if tr is None or self.trainer_running
                       else neuro.power_reports(tr))
            self._nt_draw_power(reports, getattr(tr, "power_note", None))
        except Exception as ex:                          # noqa: BLE001
            self.log(f"[neuro] energy view error: {ex!r}")

    def _nt_clear_power(self):
        self._nt_power_report = None
        if dpg.does_item_exist("nt_power_txt"):
            dpg.set_value("nt_power_txt", self.NT_POWER_HINT)
        self._nt_draw_power({}, None)

    def _nt_draw_power(self, reports, note):
        if not dpg.does_item_exist("nt_power_keys"):
            return
        for tag in ("nt_power_keys", "nt_power_table", "nt_power_y"):
            dpg.delete_item(tag, children_only=True)
        if note:                     # workspace-relative path, easier to read
            note = note.replace(os.path.join(self.workdir, ""), "")
        dpg.set_value("nt_power_note", f"Power profile: {note}" if note
                      else self.NT_POWER_HINT)
        phases = [ph for ph in neuro.POWER_PHASES if ph in reports]
        placeholders = sorted({n for ph in phases
                               for n in reports[ph]["placeholders"]})
        dpg.set_value("nt_power_warn",
                      "PLACEHOLDER values (not measured): "
                      + ", ".join(placeholders)
                      + ". Any energy that uses them is not a real number."
                      if placeholders else "")
        dpg.configure_item("nt_power_warn", show=bool(placeholders))
        if not phases:
            dpg.set_axis_ticks("nt_power_x", (("", 0.0),))
            return
        si, names = neuro.fmt_si, neuro.PHASE_NAMES
        pj = lambda v, unit: si(None if v is None else v * 1e-12, "J")
        table = dict(header_row=True, borders_innerH=True,
                     borders_outerH=True, borders_innerV=True,
                     borders_outerV=True, policy=dpg.mvTable_SizingStretchProp)

        # key figures: one column per phase
        rows = (
            ("total energy", lambda r: si(r["energy_J"]["total"], "J")),
            ("average power", lambda r: si(r["avg_power_W"], "W")),
            ("energy / presentation", lambda r: si(r["e_inference_J"], "J")),
            ("energy / synaptic op", lambda r: pj(r["e_sop_pJ"], "pJ")),
            ("energy / neuron spike", lambda r: pj(r["e_neuron_spike_pJ"], "pJ")),
            ("energy / write pulse", lambda r: pj(r["e_write_pulse_pJ"], "pJ")),
            ("presentations", lambda r: str(r["counts"]["presentations"])),
            ("simulated time", lambda r: si(r["sim_time_s"], "s")),
        )
        with dpg.table(parent="nt_power_keys", **table):
            dpg.add_table_column(label="", init_width_or_weight=1.4)
            for ph in phases:
                dpg.add_table_column(label=names[ph])
            for label, fn in rows:
                with dpg.table_row():
                    dpg.add_text(label, color=C_TEXT2)
                    for ph in phases:
                        dpg.add_text(fn(reports[ph]))

        # share of each component, grouped bars per phase; components that
        # are zero in every phase are left out
        def share(rep, group, key):
            total = rep["energy_J"]["total"]
            v = neuro.power_component(rep, group, key)
            return 100.0 * v / total if total else 0.0
        parts = [(lbl, g, k) for lbl, g, k in neuro.POWER_COMPONENTS
                 if k not in ("peripherals", "static")
                 and any(neuro.power_component(reports[ph], g, k) > 0
                         for ph in phases)]
        width = 0.8 / len(phases)
        for i, ph in enumerate(phases):
            off = (i - (len(phases) - 1) / 2.0) * width
            dpg.add_bar_series([j + off for j in range(len(parts))],
                               [share(reports[ph], g, k) for _, g, k in parts],
                               weight=width * 0.9, label=names[ph],
                               parent="nt_power_y")
        short = {"neuron spikes": "spikes", "row drivers": "row drive",
                 "column sense": "col sense", "WTA / inhibition": "WTA",
                 "neuron leakage": "leakage", "peripheral static": "periph static",
                 "array read": "read", "array write": "write"}
        dpg.set_axis_ticks("nt_power_x", tuple((short.get(lbl, lbl), float(j))
                                               for j, (lbl, _, _)
                                               in enumerate(parts))
                           or (("", 0.0),))
        dpg.set_axis_limits("nt_power_x", -0.6, max(len(parts), 1) - 0.4)
        dpg.set_axis_limits("nt_power_y", 0.0, 105.0)

        # breakdown: energy and share per phase
        with dpg.table(parent="nt_power_table", **table):
            dpg.add_table_column(label="component", init_width_or_weight=1.4)
            for ph in phases:
                dpg.add_table_column(label=names[ph])
                dpg.add_table_column(label="%", init_width_or_weight=0.5)
            for lbl, g, k in neuro.POWER_COMPONENTS + (("total", None, "total"),):
                sub = g is not None
                with dpg.table_row():
                    dpg.add_text(("    " if sub else "") + lbl,
                                 color=C_TEXT2 if sub else C_TEXT)
                    for ph in phases:
                        v = neuro.power_component(reports[ph], g, k)
                        dpg.add_text(si(v, "J"), color=C_TEXT2 if sub else C_TEXT)
                        dpg.add_text(f"{share(reports[ph], g, k):.1f}",
                                     color=C_TEXT2)

    @property
    def _nt_power_report_dir(self):
        return os.path.join(self.workdir, "results", "energy_reports")

    def _on_nt_power_save_click(self):
        if not getattr(self, "_nt_power_report", None):
            self._nt_status("train or evaluate first, then save the energy "
                            "report", C_AMBER)
            return
        os.makedirs(self._nt_power_report_dir, exist_ok=True)
        dpg.configure_item("nt_power_save_dialog",
                           default_filename="energy_report_"
                           + time.strftime("%Y%m%d_%H%M%S"))
        dpg.show_item("nt_power_save_dialog")

    def _on_nt_power_save_file(self, sender, app_data):
        path = (app_data or {}).get("file_path_name")
        if path:
            self.on_nt_power_save(path)

    def on_nt_power_save(self, path=None):
        """Write the energy report shown in the Power tab to a .txt file
        (default: results/energy_reports/energy_report_<time>.txt). Returns
        the path, or None if there is nothing to save or the write failed."""
        text = getattr(self, "_nt_power_report", None)
        if not text:
            self._nt_status("train or evaluate first, then save the energy "
                            "report", C_AMBER)
            return None
        if not path:
            path = os.path.join(self._nt_power_report_dir, "energy_report_"
                                + time.strftime("%Y%m%d_%H%M%S") + ".txt")
        if path.endswith(".*"):                  # DPG "all files" filter
            path = path[:-2]
        if not path.lower().endswith(".txt"):
            path += ".txt"
        head = ["NeuroVAT energy report",
                "saved:  " + time.strftime("%Y-%m-%d %H:%M:%S")]
        dev = (getattr(self.trainer, "device_binding", None)
               or self._nt_dev or {}).get("label")
        if dev:
            head.append("device: " + str(dev))
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(head) + "\n\n"
                         + text.replace(os.path.join(self.workdir, ""), "")
                         + "\n")
        except OSError as ex:
            self._nt_status(f"could not save the energy report: {ex}", C_RED)
            return None
        shown = os.path.abspath(path)
        if shown.startswith(os.path.join(self.workdir, "")):
            shown = os.path.relpath(shown, self.workdir)
        self._nt_status(f"energy report saved: {shown}", C_GREEN)
        self.log(f"[neuro] energy report saved: {path}")
        return path

    @property
    def _nt_power_dir(self):
        # the workspace's folder, not the process cwd (VS Code's Run button,
        # shortcuts and the exe all start elsewhere)
        return os.path.join(self.workdir, neuro.POWER_DIR)

    def _nt_power_choices(self):
        return ([neuro.PROFILE_DEFAULT]
                + neuro.list_power_profiles(self._nt_power_dir)
                + [neuro.PROFILE_NONE])

    def _nt_refresh_power_choices(self):
        if not dpg.does_item_exist("nt_power_profile"):
            return
        items = self._nt_power_choices()
        dpg.configure_item("nt_power_profile", items=items)
        if dpg.get_value("nt_power_profile") not in items:
            dpg.set_value("nt_power_profile", neuro.PROFILE_DEFAULT)

    def _nt_power_choice(self):
        if dpg.does_item_exist("nt_power_profile"):
            return dpg.get_value("nt_power_profile") or neuro.PROFILE_DEFAULT
        return neuro.PROFILE_DEFAULT

    def _nt_load_power(self):
        family = (self._nt_dev or {}).get("family")
        return neuro.load_power_params(family, folder=self._nt_power_dir,
                                       choice=self._nt_power_choice())

    def _on_nt_power_profile(self):
        """Profile changed: re-price the last run (the meter keeps counts, not
        energies), or just take effect at the next Build."""
        tr = self.trainer
        if tr is None or self.trainer_running:
            self.log(f"[neuro] power profile '{self._nt_power_choice()}' "
                     "applies at the next Build / Train")
            return
        power, note = self._nt_load_power()
        tr.energy.params = power
        tr.power_note = note
        self.log(f"[neuro] power model: {note}")
        if tr.energy.tallies:
            self._nt_show_power(neuro.power_text(
                tr, tuple(ph for ph in ("init", "train", "eval")
                          if ph in tr.energy.tallies)))

    # ---- embedded trainer agent -------------------------------------


    @staticmethod
    def _extract_nt_actions(text):
        out = []
        for blob in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "",
                               re.S):
            try:
                d = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and d.get("type") == "nt_action" \
                    and d.get("action"):
                out.append(d)
        return out

    def _nt_set_control(self, tag, val):
        if not dpg.does_item_exist(tag):
            return False
        if tag == "nt_learnrule":
            val = ("Surrogate grad (BPTT)" if str(val).lower().startswith("sur")
                   else "STDP (device-local)")
        elif tag == "nt_encoding":
            _s = str(val).lower()
            val = ("latency (TTFS)" if _s.startswith("lat")
                   else "temporal (phase)" if _s.startswith("temp")
                   else "rate (Poisson)")
        elif tag == "nt_mode":
            if self.trainer_running:
                return False
            _s = str(val).lower()
            val = (STREAM_MODE if "stream" in _s else
                   "unsupervised" if _s.startswith("uns") else "supervised")
        cur = dpg.get_value(tag)
        # tab bars store an int id but are switched by the tab's string alias
        if (isinstance(cur, int) and isinstance(val, str)
                and dpg.does_item_exist(val)):
            dpg.set_value(tag, val)
            return True
        try:
            if isinstance(cur, bool):
                val = str(val).lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int):
                val = int(float(val))
            elif isinstance(cur, float):
                val = float(val)
            else:
                val = str(val)
        except (TypeError, ValueError):
            return False
        if tag == "nt_power_profile":
            self._nt_refresh_power_choices()
        if tag in ("nt_device", "nt_va_source", "nt_power_profile") and val not in dpg.get_item_configuration(tag).get("items", []):
            return False
        dpg.set_value(tag, val)
        if tag == "nt_learnrule":
            self._nt_sync_rule()
        elif tag == "nt_mode":
            self._on_nt_mode()
        elif tag == "nt_device":
            self._on_nt_device_change()
        elif tag == "nt_va_source":
            self.network_studio.select_source()
        elif tag == "nt_power_profile":
            self._on_nt_power_profile()
        return True

    def _run_nt_actions(self, actions):
        """Execute Neuro Trainer actions the agent asked for (set / build /
        train / test), reporting each into the main chat as a system note.
        Applying changes needs the user's permission - the 'Autonomous' toggle;
        when it is OFF the agent only PROPOSES (advise-only)."""
        if not actions:
            return
        auto = (dpg.get_value("menu_auto")
                if dpg.does_item_exist("menu_auto") else True)
        if not auto:
            props = []
            for a in actions:
                nm = str(a.get("action", "")).lower()
                if nm == "set":
                    props.append("set " + ", ".join(
                        f"{k}={v}" for k, v in (a.get("params") or {}).items()))
                elif nm:
                    props.append(nm)
            self.append_chat(
                "sys", "proposed (NOT applied - enable 'Autonomous: edit + run "
                "+ fix' to let me apply, or do it yourself): "
                + "  |  ".join(p for p in props if p))
            return
        if dpg.does_item_exist("center_tabs") and dpg.does_item_exist(
                "tab_trainer"):
            dpg.set_value("center_tabs", "tab_trainer")   # show what it's doing
        for a in actions:
            name = str(a.get("action", "")).lower()
            if name == "set":
                applied = []
                for tag, val in (a.get("params") or {}).items():
                    old = self._nt_get(tag, None)
                    if self._nt_set_control(tag, val):
                        applied.append(f"{tag}={dpg.get_value(tag)}")
                        if tag == "nt_epochs":     # count agent epoch bumps
                            try:
                                if float(dpg.get_value(tag)) > float(old or 0):
                                    self._nt_epoch_bumps = getattr(
                                        self, "_nt_epoch_bumps", 0) + 1
                            except (TypeError, ValueError):
                                pass
                self.append_chat("sys", "set " + ", ".join(applied)
                                 if applied else "set: nothing applied")
            elif name == "build":
                self.on_nt_build()
                self.append_chat(
                    "sys", "built · " + dpg.get_value("nt_devlabel")[:140])
            elif name in ("train", "test") and not self._nt_is_stream():
                if not self.trainer_running:
                    self._nt_agent_run = True
                    if name == "train":
                        self.on_nt_train()
                        self.append_chat("sys", "training started...")
                    else:
                        self.on_nt_test()
            elif name in ("train", "test", "stream", "stream_test"):
                if not self.trainer_running:
                    self._nt_set_control("nt_mode", STREAM_MODE)
                    self._nt_stream_start(name in ("train", "stream"))
                    self._nt_stream_agent = self.trainer_running
                    self.append_chat("sys", self._nt_stream_error
                                     or f"{name} started...")

    # ---- agent context + snapshot -----------------------------------

    def _nt_metrics_text(self):
        m = self._nt_last_metrics
        if not m:
            return "no evaluation yet (train + Evaluate first)."
        out = [f"train acc {100*m['tr_acc']:.1f}%"]
        if m["te_acc"] is not None:
            out.append(f"TEST acc {100*m['te_acc']:.1f}%")
        out.append(f"macro-F1 {m['macro_f1']:.3f}")
        out.append(f"weighted-F1 {m['weighted_f1']:.3f}")
        lines = ["  " + ", ".join(out)]
        lines.append("  per-class F1: " + ", ".join(
            f"{n}={f:.2f}" for n, f in zip(m["names"], m["f1"])))
        cm = m["cm"]
        lines.append("  confusion (rows=true, cols=pred): "
                     + "; ".join(str(list(int(x) for x in row)) for row in cm))
        return "\n".join(lines)

    def _nt_agent_context(self):
        L = ["", "=== LIVE TRAINER STATE ==="]
        tr = self.trainer
        if tr is None:
            L.append("No network built yet. Set controls, then build + train.")
        else:
            cfg = tr.cfg
            dev = (self._nt_dev or {}).get("label", "?")
            kind = (self._nt_dev or {}).get("kind", "?")
            L.append(f"synapse device: {dev} ({kind}-driven)")
            L.append("layers: " + " -> ".join(map(str, tr.layer_sizes))
                     + f"  ({tr.n_layers} crossbar(s))")
            L.append(f"mode={cfg.mode}  learn_rule={cfg.learn_rule}  "
                     f"encoding={cfg.encoding}")
            L.append(f"patterns: {len(tr.patterns)} train"
                     + (f" + {len(tr.test_patterns)} held-out test"
                        if tr.test_patterns else "")
                     + f"; classes={list(tr.class_names)}")
            for c in range(tr.n_layers):
                w = tr.weights_uS(c)
                L.append(f"  crossbar {c} ({tr.layer_sizes[c]}x"
                         f"{tr.layer_sizes[c+1]}): {w.min():.0f}..{w.max():.0f} "
                         f"uS (mean {w.mean():.0f})")
            h = self._nt_acc_hist
            if h["epoch"]:
                tail = h["train"][-6:]
                L.append("train accuracy by epoch (last few): "
                         + ", ".join(f"{v:.0f}%" for v in tail))
            # all-synapse weight trajectories (the 'All weights' plot)
            wh = self._nt_whist
            if wh["epoch"]:
                L.append(f"weight trajectories over {len(wh['epoch'])} epochs "
                         "(every synapse; 'All weights' tab):")
                for c in range(tr.n_layers):
                    try:
                        W0 = np.asarray(wh["W"][0][c], float)
                        W1 = np.asarray(wh["W"][-1][c], float)
                    except Exception:       # noqa: BLE001
                        continue
                    d = W1 - W0
                    L.append(f"  crossbar {c}: mean {W0.mean():.0f}->{W1.mean():.0f}"
                             f" uS, end-spread {W1.std():.0f}; "
                             f"{int((d > 1).sum())} synapses potentiated, "
                             f"{int((d < -1).sum())} depressed")
            # the synapse currently open in the Cell inspector
            if self._nt_cell is not None and self._nt_cell[0] < tr.n_layers:
                cc, jj, ii = self._nt_cell
                pot = tr.last_write.get((cc, jj, ii), (None, 0))[0]
                L.append(f"inspected synapse (Cell tab): crossbar {cc}, input {ii}"
                         f" -> neuron {jj}: G={tr.weights_uS(cc)[jj][ii]:.0f} uS, "
                         f"last write={'potentiate' if pot else 'depress' if pot is not None else 'n/a'}")
            # the most recent presentation's output spikes
            if self._last_present and isinstance(self._last_present[2], dict):
                sp = self._last_present[2].get("n_out_spikes")
                if sp is not None:
                    L.append("last presentation output spikes/neuron: "
                             + str([int(x) for x in sp]))
            L.append("last evaluation:")
            L.append(self._nt_metrics_text())
        L.extend(self._nt_stream_context())
        L.append("")
        L.append("=== CONTROLS (current values) ===")
        for tag, desc in NT_CONTROLS:
            if dpg.does_item_exist(tag):
                L.append(f"  {tag} = {dpg.get_value(tag)!r}   # {desc}")
        L.append("  nt_patset valid values: " + ", ".join(self._patset_items()))
        L.append("(full state - every plot's data - is also in "
                 "results/neuro_snapshot.json; you may Read it.)")
        # permission gate the user controls
        auto = (dpg.get_value("menu_auto")
                if dpg.does_item_exist("menu_auto") else True)
        L.append("")
        L.append("=== WHAT YOU MAY DO (user's permission) ===")
        if auto:
            L.append("Autonomous is ON - the user ALLOWS you to act. You MAY apply "
                     "nt_action (set/build/train/test) AND edit the synapse "
                     "device's Verilog-A (.va) / Python twin to change its physics "
                     "(then rebuild). Make focused changes and explain them.")
        else:
            L.append("Autonomous is OFF - READ + ADVISE only. You may inspect all "
                     "data and PROPOSE nt_action blocks or code edits, but nothing "
                     "is applied until the user enables Autonomous or does it "
                     "manually. Do NOT claim you changed anything.")
        return "\n".join(L)

    def _write_nt_snapshot(self):
        try:
            tr = self.trainer
            snap = {"controls": {t: dpg.get_value(t)
                                 for t, _ in NT_CONTROLS if dpg.does_item_exist(t)}}
            if tr is not None:
                snap["layers"] = list(tr.layer_sizes)
                snap["mode"] = tr.cfg.mode
                snap["learn_rule"] = tr.cfg.learn_rule
                snap["classes"] = list(tr.class_names)
                snap["weights_uS"] = {
                    f"crossbar_{c}": {"min": float(tr.weights_uS(c).min()),
                                      "max": float(tr.weights_uS(c).max()),
                                      "mean": float(tr.weights_uS(c).mean())}
                    for c in range(tr.n_layers)}
                snap["accuracy_curve"] = self._nt_acc_hist
                # full weight trajectories (the 'All weights' plot data)
                wh = self._nt_whist
                if wh["epoch"]:
                    wt = {"epochs": list(wh["epoch"])}
                    for c in range(tr.n_layers):
                        try:
                            st = np.stack([np.asarray(wh["W"][e][c], float)
                                           for e in range(len(wh["epoch"]))])
                        except Exception:           # noqa: BLE001
                            continue
                        wt[f"crossbar_{c}_overall_mean_uS"] = \
                            [round(float(v), 1) for v in st.mean(axis=(1, 2))]
                        wt[f"crossbar_{c}_mean_per_neuron_uS"] = \
                            np.round(st.mean(axis=2), 1).tolist()   # (epochs, n_post)
                    snap["weight_trajectories"] = wt
                if self._nt_cell is not None and self._nt_cell[0] < tr.n_layers:
                    cc, jj, ii = self._nt_cell
                    snap["inspected_cell"] = {
                        "crossbar": cc, "input": int(ii), "neuron": jj,
                        "G_uS": float(tr.weights_uS(cc)[jj][ii])}
                if self._last_present and isinstance(self._last_present[2], dict):
                    sp = self._last_present[2].get("n_out_spikes")
                    if sp is not None:
                        snap["last_presentation_spikes"] = [int(x) for x in sp]
            snap["autonomous_allowed"] = bool(
                dpg.get_value("menu_auto")
                if dpg.does_item_exist("menu_auto") else True)
            if self._nt_last_metrics:
                m = dict(self._nt_last_metrics)
                m["f1"] = [float(x) for x in m["f1"]]
                m["cm"] = [[int(x) for x in row] for row in m["cm"]]
                snap["last_metrics"] = m
            outdir = os.path.join(self.workdir, "results")
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, "neuro_snapshot.json"), "w",
                      encoding="utf-8") as f:
                json.dump(snap, f, indent=2, default=str)
        except Exception as e:                          # noqa: BLE001
            self.log(f"[neuro] snapshot write failed: {e!r}")

    def _nt_output_tab(self):
        with self._pad(left=6, top=4, bottom=0):
            self._small("TRAINING OUTPUT SPIKES  (live) - a dot each time an "
                        "output neuron fires during training; watch neurons "
                        "specialise to patterns as the synapses learn",
                        color=(126, 150, 220))
            dpg.add_text("totals: -", tag="nt_spk_totals", color=C_TEXT2)
            with dpg.plot(width=-1, height=-1, tag="nt_spk_plot",
                          no_menus=True):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="training presentation #",
                                  tag="nt_spk_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="output neuron",
                                  tag="nt_spk_y")

    def on_nt_input_spikes(self, *_):
        """Compatibility callback for callers of the former Inputs tab."""
        self._nt_encoding_preview()


    # ---- training output-spike stream -------------------------------

    def _nt_reset_output(self):
        self._nt_pres_idx = 0
        self._nt_spk_pts = {}
        for s in self._nt_spk_series.values():
            if dpg.does_item_exist(s):
                dpg.delete_item(s)
        self._nt_spk_series = {}
        if dpg.does_item_exist("nt_spk_totals"):
            dpg.set_value("nt_spk_totals", "totals: -")

    def _on_nt_spk(self, counts, winner):
        idx = self._nt_pres_idx
        self._nt_pres_idx += 1
        if not dpg.does_item_exist("nt_spk_y"):
            return
        for j, c in enumerate(counts):
            if c <= 0:
                continue
            xs, ys = self._nt_spk_pts.setdefault(j, ([], []))
            xs.append(idx)
            ys.append(j)
            if len(xs) > 2000:               # sliding window (bounds redraw cost)
                del xs[0]
                del ys[0]
            s = self._nt_spk_series.get(j)
            if s and dpg.does_item_exist(s):
                dpg.set_value(s, [xs, ys])
            else:
                self._nt_spk_series[j] = dpg.add_scatter_series(
                    xs, ys, label=f"N{j}", parent="nt_spk_y")
        tot = self.trainer.spike_count if self.trainer is not None else []
        dpg.set_value("nt_spk_totals", "totals:  " + "   ".join(
            f"N{j}={int(t)}" for j, t in enumerate(tot)))
        if idx % 8 == 0:
            dpg.fit_axis_data("nt_spk_x")
            dpg.fit_axis_data("nt_spk_y")

    # ---- open / status ----------------------------------------------

    def on_open_trainer(self, *_):
        """Switch the main window's center area to the Neuro Trainer tab (and
        swap the left panel to the trainer parameters)."""
        if dpg.does_item_exist("center_tabs") and dpg.does_item_exist(
                "tab_trainer"):
            dpg.show_item("tab_trainer")
            dpg.set_value("center_tabs", "tab_trainer")
            self._on_center_tab()           # set_value doesn't fire the callback
        if self._nt_paint is None:          # init the custom-pattern canvas
            self._nt_paint_new()
        if self.trainer is None:
            self._nt_status("press 'Build network' to start", C_TEXT2)

    def _nt_tick_layout(self):
        """Per-frame: keep the network-diagram drawlist sized to its canvas
        column (in the center tab), so it tracks the window height with no
        scrollbar. Cheap - only reconfigures + redraws when the size changes,
        and a no-op while the Trainer tab is not the visible one."""
        if not dpg.does_item_exist("nt_canvas_scroll"):
            return
        size = dpg.get_item_rect_size("nt_canvas_scroll")
        if not size or size[0] < 60 or size[1] < 60:
            return                                   # tab not visible
        z = self._nt_zoom
        base_w = max(360, int(size[0]) - 18)
        base_h = max(220, int(size[1]) - 18)
        w, h = int(base_w * z), int(base_h * z)      # zoom grows the drawlist
        if (w, h) == self._nt_win_rect:
            return
        self._nt_win_rect = (w, h)
        self._nt_diag_size = (w, h)
        if dpg.does_item_exist("nt_diagram"):
            dpg.configure_item("nt_diagram", width=w, height=h)
            # redraw at the new size, but not while a worker is mutating the
            # devices (the per-epoch snapshot redraws live during training)
            if self.trainer is not None and not self.trainer_running:
                Wn_all = [self.trainer.weights_norm(c)
                          for c in range(self.trainer.n_layers)]
                self._nt_draw_diagram(Wn_all, self._last_present)

    def _nt_set_zoom(self, z):
        """Set the crossbar-canvas zoom (1.0 = fit). Larger zoom grows the
        drawlist inside its scroll window and reveals the 3-terminal cell
        detail; the layout tick repaints on the next frame."""
        z = min(max(float(z), 0.6), 4.0)
        if abs(z - self._nt_zoom) < 1e-3:
            return
        self._nt_zoom = z
        self._nt_win_rect = None                     # force a relayout + redraw
        if dpg.does_item_exist("nt_zoom_lbl"):
            dpg.set_value("nt_zoom_lbl", f"{z * 100:.0f}%")

    def _nt_status(self, text, color):
        if dpg.does_item_exist("nt_status"):
            dpg.configure_item("nt_status", label=f"● {text}")
            th = self._status_themes.get(color)
            if th is None:                          # a colored status "pill"
                r, g, b = color[:3]
                th = self._mk_theme(f"ntstat_{color}", colors=[
                    ("mvThemeCol_Button", (r, g, b, 28)),
                    ("mvThemeCol_ButtonHovered", (r, g, b, 40)),
                    ("mvThemeCol_ButtonActive", (r, g, b, 40)),
                    ("mvThemeCol_Border", (r, g, b, 130)),
                    ("mvThemeCol_Text", color)],
                    styles=[("mvStyleVar_FrameRounding", 11),
                            ("mvStyleVar_FrameBorderSize", 1),
                            ("mvStyleVar_FramePadding", (12, 3))])
                self._status_themes[color] = th
            dpg.bind_item_theme("nt_status", th)

    def _nt_busy(self, on):
        if dpg.does_item_exist("nt_busy"):
            dpg.configure_item("nt_busy", show=on)

    def _nt_sync_rule(self):
        """Surrogate gradient is supervised - force the mode + hint it."""
        if self._nt_is_stream():
            return
        if self._nt_get("nt_learnrule", "STDP").startswith("Surrogate"):
            if dpg.does_item_exist("nt_mode"):
                dpg.set_value("nt_mode", "supervised")
                self._on_nt_mode()
            self._nt_status("surrogate gradient (BPTT) - supervised, in-situ; "
                            "trains deep nets", C_ACC)

    # ---- parameter reading ------------------------------------------

    def _nt_get(self, tag, default):
        return dpg.get_value(tag) if dpg.does_item_exist(tag) else default

    def _nt_getu(self, tag, default):
        """Read a unit-bearing field in its CANONICAL unit (pA / V / ms / Hz).
        For a field with no unit combo this is identical to _nt_get()."""
        v = self._nt_get(tag, default)
        u = self._nt_units.get(tag)
        try:
            return v * UNIT_FACTOR[u] if u else v
        except (KeyError, TypeError):
            return v

    def _nt_unit_combo(self, tag, dim):
        """A small unit dropdown bound to `tag`.  The canonical unit (what the
        engine expects: pA / V / ms / Hz) is the default selection."""
        if dim == "amp":                         # canonical depends on device kind
            kind = (self._nt_dev or {}).get("kind", "current")
            d = "voltage" if kind == "voltage" else "current"
        else:
            d = dim
        items, canon = UNIT_SETS[d], CANON_UNIT[d]
        self._nt_units[tag] = canon
        return dpg.add_combo(items, default_value=canon, tag=f"{tag}__u",
                             width=58, callback=lambda s, a, tg=tag:
                             self._on_unit_change(tg, a))

    def _on_unit_change(self, tag, new_unit):
        """Rescale the field so the physical value is unchanged when the unit
        changes (50 pA -> 0.05 nA)."""
        old = self._nt_units.get(tag)
        self._nt_units[tag] = new_unit
        if old and old != new_unit and old in UNIT_FACTOR \
                and new_unit in UNIT_FACTOR:
            canonical = self._nt_get(tag, 0.0) * UNIT_FACTOR[old]
            dpg.set_value(tag, canonical / UNIT_FACTOR[new_unit])
        self._nt_encoding_preview()              # keep the Data-tab rasters live

    def _nt_set_amp_units(self, kind):
        """Point the pot/dep-amp unit combos at current (pA) or voltage (V)."""
        items = UNIT_SETS["voltage" if kind == "voltage" else "current"]
        canon = "V" if kind == "voltage" else "pA"
        for tag in ("nt_potamp", "nt_depamp"):
            cb = f"{tag}__u"
            if dpg.does_item_exist(cb):
                dpg.configure_item(cb, items=items, default_value=canon)
                dpg.set_value(cb, canon)
            self._nt_units[tag] = canon

    def _nt_reset_units(self):
        """Reset every unit combo to its canonical unit WITHOUT rescaling (used
        by Defaults, which already writes canonical-unit values)."""
        for tag, cur in list(self._nt_units.items()):
            canon = CANON_UNIT.get(UNIT_DIM.get(cur, ""), cur)
            cb = f"{tag}__u"
            if dpg.does_item_exist(cb):
                dpg.set_value(cb, canon)
            self._nt_units[tag] = canon

    def _on_nt_device_change(self, *_):
        """When the synapse device's DRIVE KIND changes (current<->voltage),
        switch the programming-pulse amplitude/unit/EPSP to suit it."""
        self.network_studio.sync_device()
        device = self._nt_device_factory()
        kind = (device or {}).get("kind", "current")
        if kind == getattr(self, "_nt_last_kind", None):
            self._nt_source_changed()
            return
        self._nt_last_kind = kind
        fields = (dict(nt_potamp=2.0, nt_depamp=2.0, nt_pwidth=1.0, nt_epsp=6.0)
                  if kind == "voltage" else
                  dict(nt_potamp=50.0, nt_depamp=50.0, nt_pwidth=10.0,
                       nt_epsp=11.0))
        for tag, val in fields.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, val)
        self._nt_set_amp_units(kind)
        self._nt_source_changed()

    def _nt_source_changed(self):
        """The synapse source was changed while a network is built. The views
        (Cell, Weights, ...) show the BUILT network, so they would keep
        showing the old device: rebuild at once if nothing was learnt yet,
        otherwise say clearly that the views are those of the old device."""
        tr = self.trainer
        if (tr is None or self.trainer_running or self._nt_loading
                or self._nt_is_stream()):
            return
        try:
            now = self.network_studio.binding()
        except (ValueError, OSError):
            return
        built = getattr(tr, "source_binding", None) or {}
        label = dpg.get_value("nt_devlabel") if dpg.does_item_exist("nt_devlabel") else ""
        note = "NOT REBUILT  |  "
        if now == built:
            if label.startswith(note):
                dpg.set_value("nt_devlabel", label[len(note):])
                self._nt_status("built", C_GREEN)
            return
        if tr.write_count == 0 and not self._nt_loaded_model:
            self.log("[neuro] synapse source changed - network rebuilt")
            self.on_nt_build()
            return
        old, new = (os.path.basename(b.get("path", "?")) for b in (built, now))
        if not label.startswith(note):
            dpg.set_value("nt_devlabel", note + label)
        self._nt_status(f"synapse changed to {new} - the views still show the "
                        f"{old} network; press Build network", C_AMBER)

    def _nt_device_factory(self):
        # the SYNAPSE DEVICE selector in the trainer's left-panel controls picks
        # the device twin; fall back to a checked .va, then a built-in default
        key = LABEL_TO_KEY.get(self._nt_get("nt_device", ""))
        if not key:
            keys = self._enabled_keys()
            key = keys[0] if keys else ("v2" if "v2" in SPEC_BY_KEY
                                        else MODEL_SPECS[0].key)
        spec = SPEC_BY_KEY.get(key)
        if spec is None:
            return None
        pv = dict(self.param_values[key])
        # silence the device's intrinsic STDP lock-in: in the crossbar the
        # NETWORK rule decides plasticity direction, the device supplies the
        # analog weight store (write nonlinearity / bounds / retention).
        for silent in ("A_stdp", "A_stdp_V"):
            if silent in pv:
                pv[silent] = 0.0
        # device cycle-to-cycle write noise (only models that model it, e.g. v2)
        wnoise = float(self._nt_get("nt_wnoise", 0.0))
        if wnoise > 0 and "sigma_c2c" in pv:
            pv["sigma_c2c"] = wnoise
        has_seed = "seed" in pv
        counter = {"n": 0}

        def make(_cls=spec.cls, _pc=spec.params_cls, _pv=pv,
                 _seed=has_seed, _c=counter):
            p = dict(_pv)
            if _seed:                    # unique seed per synapse -> independent
                p["seed"] = 1000 + _c["n"]            # write-noise streams
                _c["n"] += 1
            return _cls(_pc(**p))

        # ECFET (current gate) + FeFET (voltage gate) are 3-terminal; RRAM /
        # memristor (and any non-FET twin) is 2-terminal - written across its
        # read lines. A twin may override via a TERMINALS class attribute.
        terms = getattr(spec.cls, "TERMINALS", None)
        if terms is None:
            terms = 3 if DEVICE_OF_KEY.get(key) in ("ECFET", "FeFET") else 2
        return {"make": make, "kind": spec.input_kind, "label": spec.label,
                "key": key, "three_terminal": terms >= 3,
                "family": DEVICE_OF_KEY.get(key) or key}

    def _nt_read_params(self):
        N = neuro.NeuronParams(
            tau_m=self._nt_getu("nt_tau_m", 20.0),
            v_threshold=self._nt_get("nt_vth", 1.0),
            t_refractory=self._nt_getu("nt_refrac", 5.0),
            tau_syn=self._nt_getu("nt_tausyn", 8.0),
            epsp_gain=self._nt_get("nt_epsp", 11.0),
            ipsp_gain=self._nt_get("nt_ipsp", 1.0),
            inhibition=self._nt_get("nt_inhib", 0.9),
            theta_plus=self._nt_get("nt_theta", 0.06),
            teacher=self._nt_get("nt_teacher", 1.4),
            v_noise=max(self._nt_get("nt_vnoise", 0.0), 0.0),
            hidden_gain=max(self._nt_get("nt_hidden_gain", 1.7), 0.1))
        kind = (self._nt_dev or {}).get("kind", "current")
        ascale = 1e-12 if kind == "current" else 1.0   # pA -> A, or V
        S = neuro.STDPParams(
            a_plus=self._nt_get("nt_aplus", 1.0),
            a_minus=self._nt_get("nt_aminus", 1.0),
            offset=self._nt_get("nt_offset", 0.25),
            tau_pre=self._nt_getu("nt_taupre", 20.0),
            pot_amp=self._nt_getu("nt_potamp", 50.0) * ascale,
            dep_amp=self._nt_getu("nt_depamp", 50.0) * ascale,
            pulse_width=max(self._nt_getu("nt_pwidth", 10.0), 0.01) * 1e-3,
            sg_lr=max(self._nt_get("nt_sg_lr", 0.1), 1e-4))
        mode = self._nt_get("nt_mode", "supervised")
        from .network_studio import hidden_sizes
        hidden = tuple(hidden_sizes(self._nt_get("nt_hidden", "") or ""))
        cfg = neuro.NetConfig(
            grid_h=min(max(int(self._nt_get("nt_gh", 5)), 2), 14),
            grid_w=min(max(int(self._nt_get("nt_gw", 5)), 2), 14),
            n_out=min(max(int(self._nt_get("nt_nout", 4)), 1), 16),
            hidden_layers=hidden,
            mode="supervised" if mode.startswith("super") else "unsupervised",
            present_ms=max(self._nt_getu("nt_present", 120.0), 10.0),
            dt_ms=max(self._nt_getu("nt_dt", 1.0), 0.1),
            max_rate_hz=max(self._nt_getu("nt_rate", 180.0), 1.0),
            seed=int(self._nt_get("nt_seed", 1)),
            pattern_set=self._nt_get("nt_patset", "bars"),
            learn_rule=("surrogate" if self._nt_get(
                "nt_learnrule", "STDP").startswith("Surrogate") else "stdp"),
            encoding=(lambda e: "latency" if e.startswith("lat")
                      else "temporal" if e.startswith("temp") else "rate")(
                      self._nt_get("nt_encoding", "rate").lower()),
            bg_rate_hz=max(self._nt_getu("nt_bg_rate", 0.0), 0.0),
            input_noise=min(max(self._nt_get("nt_input_noise", 0.0), 0.0), 1.0),
            signal_frac=min(max(self._nt_get("nt_signal_frac", 1.0), 0.0), 1.0),
            jitter_ms=max(self._nt_get("nt_jitter", 0.0), 0.0),
            pattern_ms=max(self._nt_getu("nt_pat_ms", 0.0), 0.0))
        return N, S, cfg

    def on_nt_defaults(self, *_):
        """Fill neuron / learning fields with values that suit the currently
        selected device's drive kind (ECFET current vs FeFET voltage)."""
        self._nt_dev = self._nt_device_factory()
        kind = (self._nt_dev or {}).get("kind", "current")
        common = {"nt_tau_m": 20, "nt_vth": 1.0, "nt_refrac": 5,
                  "nt_tausyn": 8, "nt_ipsp": 1.0, "nt_inhib": 0.9,
                  "nt_theta": 0.06, "nt_teacher": 1.4, "nt_pwidth": 10,
                  "nt_aplus": 1.0, "nt_aminus": 1.0, "nt_offset": 0.25,
                  "nt_taupre": 20}
        if kind == "voltage":          # FeFET: gate-voltage programming
            common.update(nt_epsp=6.0, nt_potamp=2.0, nt_depamp=2.0,
                          nt_pwidth=1.0)
        else:                          # ECFET: gate-current programming (pA)
            common.update(nt_epsp=11.0, nt_potamp=50, nt_depamp=50)
        for tag, val in common.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, val)
        self._nt_reset_units()          # values above are in canonical units
        self._nt_set_amp_units(kind)
        self._nt_status(f"defaults loaded for {kind}-driven synapse", C_GREEN)

    # ---- save / load a trained model --------------------------------

    def _on_nt_save_file(self, sender, app_data):
        path = (app_data or {}).get("file_path_name")
        if path:
            self.on_nt_save(path)

    def on_nt_save(self, path=None):
        """Save the trained model - network + ALL weights + params + the
        training patterns - to a self-contained JSON that Load fully restores."""
        if self.trainer_running:
            self._nt_status("Stop training/evaluation before saving", C_AMBER)
            return
        if self._nt_is_stream():
            self._nt_status("stream networks cannot be saved yet - Save is "
                            "for the supervised / unsupervised modes", C_AMBER)
            return
        tr = self.trainer
        if tr is None:
            self._nt_status("build + train first, then Save", C_AMBER)
            return
        if not path:
            path = os.path.join(self.workdir, "results", "models",
                                "neuro_model.json")
        if not path.lower().endswith(".json"):
            path += ".json"
        di = getattr(self.trainer, "device_binding", self._nt_dev or {})
        model = {
            "format": "neurovat-model/2",
            "trainer_state": tr.capture_state(),
            "synapse_source": tr.source_binding,
            "device_params": dataclasses.asdict(tr.crossbars[0][0][0].p),
            "device": {"key": di.get("key"), "label": di.get("label"),
                       "kind": di.get("kind"),
                       "three_terminal": di.get("three_terminal")},
            "layer_sizes": list(tr.layer_sizes),
            "class_names": list(tr.class_names),
            "pot_sign": float(tr.pot_sign),
            "cfg": dataclasses.asdict(tr.cfg),
            "neuron": dataclasses.asdict(tr.N),
            "stdp": dataclasses.asdict(tr.S),
            "controls": dict(tr.gui_controls),
            "units": dict(tr.gui_units),
            "weights_S": [tr.weights_S(c).tolist() for c in range(tr.n_layers)],
            "patterns": [[lbl, np.asarray(v, float).tolist()]
                         for lbl, v in tr.patterns],
            "targets": [int(t) for t in tr.target_of],
        }
        try:                              # restore the exact RNG to resume cleanly
            model["rng_state"] = tr.rng.bit_generator.state
        except Exception:                 # noqa: BLE001
            pass
        if tr.validation_patterns:
            model["validation_patterns"] = [[lbl, np.asarray(v, float).tolist()] for lbl, v in tr.validation_patterns]
            model["validation_targets"] = list(tr.validation_targets)
        if getattr(tr, "test_patterns", None):
            model["test_patterns"] = [[lbl, np.asarray(v, float).tolist()]
                                      for lbl, v in tr.test_patterns]
            model["test_targets"] = [int(t) for t in tr.test_targets]
        if self._nt_last_metrics:
            m = self._nt_last_metrics
            model["metrics"] = {"train_acc": m.get("tr_acc"),
                                "test_acc": m.get("te_acc"),
                                "macro_f1": m.get("macro_f1")}
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(model, f, indent=1, default=float)
            kb = os.path.getsize(path) / 1024.0
            self._nt_status(f"saved -> {os.path.basename(path)} ({kb:.0f} KB)",
                            C_GREEN)
            self.log(f"[neuro] model saved: {path}")
        except Exception as e:                          # noqa: BLE001
            self._nt_status(f"save failed: {e}", C_AMBER)
            self.log(f"[neuro] save failed: {e!r}")

    def _on_nt_load_file(self, sender, app_data):
        path = (app_data or {}).get("file_path_name")
        if path and os.path.isfile(path):
            self.on_nt_load(path)

    def on_nt_load(self, path):
        """Restore a saved model: set the controls, rebuild the same network
        from the saved patterns, then write the saved weights into the devices."""
        if self.trainer_running:
            self._nt_status("Stop training/evaluation before loading", C_AMBER)
            return
        try:
            with open(path, encoding="utf-8") as f:
                model = json.load(f)
        except Exception as e:                          # noqa: BLE001
            self._nt_status(f"load failed: {e}", C_AMBER)
            return
        try:
            self.network_studio.restore_binding(model)
        except (ValueError, OSError) as exc:
            self._nt_status(f"load failed: {exc}", C_RED)
            return
        self._nt_loading = True         # the load rebuilds below, once
        try:
            self._on_nt_device_change()
        finally:
            self._nt_loading = False
        # 1. restore every control so the rebuild matches the saved setup
        for tag, val in (model.get("controls") or {}).items():
            if dpg.does_item_exist(tag):
                try:
                    dpg.set_value(tag, val)
                except Exception:                       # noqa: BLE001
                    pass
        # 1b. restore the saved display units (after the device sync, which resets
        #     the amp unit) so the restored numbers read back to the same value
        for tag, u in (model.get("units") or {}).items():
            if u in UNIT_FACTOR:
                self._nt_units[tag] = u
                if dpg.does_item_exist(f"{tag}__u"):
                    dpg.set_value(f"{tag}__u", u)
        # 2. inject the saved patterns so the rebuild is self-contained (no need
        #    for the original dataset / custom paint), then rebuild the network
        pats = [(lbl, np.asarray(v, np.float32))
                for lbl, v in (model.get("patterns") or [])]
        tgts = [int(t) for t in (model.get("targets") or [])]
        key = model.get("device", {}).get("key")
        if key in self.param_values and model.get("device_params"):
            self.param_values[key].update(model["device_params"])
        self._nt_loaded_patterns = (pats, tgts) if pats else None
        if self._nt_is_stream():        # a saved model is a grid network
            dpg.set_value("nt_mode", "supervised")
        self._on_nt_mode()
        self.on_nt_build()
        self._nt_loaded_patterns = None
        self._nt_loaded_model = self.trainer is not None
        tr = self.trainer
        if tr is None:
            self._nt_status("load: rebuild failed", C_AMBER)
            return
        # 3. restore class names, drive polarity, held-out test set
        if model.get("class_names"):
            tr.class_names = list(model["class_names"])
        if "pot_sign" in model:
            tr.pot_sign = float(model["pot_sign"])
        if model.get("test_patterns"):
            tr.test_patterns = [(lbl, np.asarray(v, np.float32))
                                for lbl, v in model["test_patterns"]]
            tr.test_targets = [int(t) for t in model.get("test_targets", [])]
        if model.get("validation_patterns"):
            tr.validation_patterns = [(lbl, np.asarray(v, float)) for lbl, v in model["validation_patterns"]]
            tr.validation_targets = list(model["validation_targets"])
        if model.get("rng_state"):
            try:
                tr.rng.bit_generator.state = model["rng_state"]
            except Exception:                       # noqa: BLE001
                pass
        # 4. write the saved weights into the rebuilt devices
        W = model.get("weights_S") or []
        ok = 0
        try:
            if model.get("trainer_state"):
                tr.restore_state(model["trainer_state"])
                ok = tr.n_layers
            else:
                for c, weights in enumerate(W):
                    tr.set_weights(c, weights)
                    ok += 1
        except (ValueError, TypeError, KeyError) as exc:
            self._nt_status(f"load failed: {exc}", C_RED)
            self.log(f"[neuro] load failed: {exc}")
            return
        self._apply_nt_snapshot(self._nt_snapshot(0, 0))      # redraw restored
        self._nt_status(
            f"loaded {os.path.basename(path)} - {ok}/{tr.n_layers} crossbars",
            C_GREEN)
        self.log(f"[neuro] model loaded: {path} ({ok} crossbars)")

    # ---- build ------------------------------------------------------

    def on_nt_build(self, *_):
        if self.trainer_running:
            return
        if self._nt_is_stream() and not self._nt_loading:
            self.on_nt_stream_reset()
            return
        try:
            binding = self.network_studio.binding()
            device = self._nt_device_factory()
            if not device:
                raise ValueError("The selected source has no conductance twin.")
            self._nt_cell = None
            self._nt_dev = device
            self._nt_loaded_model = False
            N, S, cfg = self._nt_read_params()
            # custom / dataset sources supply explicit patterns (and may resize
            # the grid + n_out); built-in sources return (None, None)
            self._nt_pending_test = None
            patterns, targets = self._nt_build_patterns(cfg)
            self._nt_refresh_power_choices()
            power, power_note = self._nt_load_power()
            self.trainer = neuro.Trainer(self._nt_dev["make"],
                                         self._nt_dev["kind"], N, S, cfg,
                                         patterns=patterns, targets=targets,
                                         power=power)
            self.trainer.power_note = power_note
        except Exception as e:                          # noqa: BLE001
            self._nt_status(f"build failed: {e}", C_RED)
            self.log(f"[neuro] build failed: {e!r}")
            self.trainer = None
            return
        self.trainer.source_binding = binding
        self.trainer.device_binding = {k: v for k, v in device.items() if k != "make"}
        # held-out test set + class names (datasets only)
        if self._nt_pending_test:
            te_p, te_t, names = self._nt_pending_test
            self.trainer.test_patterns = te_p
            self.trainer.test_targets = te_t
            self.trainer.class_names = names
            if not self._nt_loaded_patterns:
                self.trainer.split_validation()
        # reflect any grid / n_out the source chose back into the controls
        dpg.set_value("nt_gh", cfg.grid_h)
        dpg.set_value("nt_gw", cfg.grid_w)
        dpg.set_value("nt_nout", cfg.n_out)
        self.network_studio.refresh_layers()
        self.trainer.gui_controls = {t: dpg.get_value(t) for t, _ in NT_CONTROLS if dpg.does_item_exist(t)}
        self.trainer.gui_units = dict(self._nt_units)
        self._wevo = {j: [] for j in range(cfg.n_out)}
        self._wevo_epochs = []
        self._nt_acc_hist = {"epoch": [], "train": [], "validation": []}
        self._nt_whist = {"epoch": [], "W": []}    # per-synapse weight trajectory
        self._last_present = None
        self._nt_best_seen = 0.0
        for tg, v in (("nt_stat_epoch", "0"), ("nt_stat_tracc", "--"),
                      ("nt_stat_teacc", "--"), ("nt_stat_valacc", "--"), ("nt_stat_best", "--")):
            self._nt_stat(tg, v)
        self._nt_reset_output()
        self._nt_clear_power()
        self.log(f"[neuro] power model: {power_note}")
        # the receptive-field montage shows crossbar 0's post-layer (first
        # hidden layer, or the output for a single crossbar)
        self._nt_setup_rf_texture(self.trainer.layer_sizes[1], cfg.grid_h,
                                  cfg.grid_w)
        # crossbar selector for the Weights tab
        sizes = self.trainer.layer_sizes
        self._nt_xbar_labels = [f"crossbar {c}: {sizes[c]}x{sizes[c + 1]}"
                                for c in range(self.trainer.n_layers)]
        if dpg.does_item_exist("nt_xbar_sel"):
            dpg.configure_item("nt_xbar_sel", items=self._nt_xbar_labels,
                               default_value=self._nt_xbar_labels[0],
                               show=self.trainer.n_layers > 1)
        self._nt_xbar_sel_idx = 0
        pol = "+" if self.trainer.pot_sign > 0 else "-"
        n_sig = self.trainer.n_signal
        n_noise = self.trainer.n_in - n_sig
        aff = (f"   |   {n_sig} signal + {n_noise} noise afferents"
               if n_noise else "")
        n_dev = sum(sizes[c] * sizes[c + 1] for c in range(self.trainer.n_layers))
        stack = " -> ".join(str(s) for s in sizes)
        dpg.set_value("nt_devlabel",
                      f"synapse: {os.path.basename(binding['path'])} / Python twin   |   layers "
                      f"{stack}  ({self.trainer.n_layers} crossbar"
                      f"{'s' if self.trainer.n_layers > 1 else ''})   |   "
                      f"{n_dev} device synapses   |   "
                      f"{cfg.encoding} coding{aff}   |   "
                      f"drive: {pol}   |   {cfg.mode} / "
                      f"{'surrogate-grad' if cfg.learn_rule == 'surrogate' else 'STDP'}")
        labels = [lbl for lbl, _ in self.trainer.patterns]
        shown = ", ".join(dict.fromkeys(labels[:24]))    # unique, first few
        dpg.set_value("nt_pat_txt", f"{len(labels)} patterns: {shown}"
                      + (" ..." if len(labels) > 24 else ""))
        self._apply_nt_snapshot(self._nt_snapshot(0, 0))
        self._nt_status(f"built - {self.trainer.n_in} inputs x {cfg.n_out} "
                        f"neurons", C_GREEN)
        self.log(f"[neuro] built {self._nt_dev['label']} crossbar: "
                 f"{cfg.grid_h}x{cfg.grid_w} -> {cfg.n_out}, mode {cfg.mode}, "
                 f"{len(self.trainer.patterns)} patterns")


    # ---- train / test / stop / reset --------------------------------

    def on_nt_train(self, *_):
        if self._nt_is_stream():
            self._nt_stream_start(True)
            return
        try:
            self.network_studio.require_current()
        except (ValueError, OSError) as exc:
            self._nt_status(str(exc), C_RED)
            return
        if self.trainer_running:
            return
        if self.trainer is None:
            self.on_nt_build()
            if self.trainer is None:
                return
        epochs = max(1, int(self._nt_get("nt_epochs", 20)))
        self.trainer_running = True
        self._trainer_stop = False
        self._nt_training_error = None
        self._nt_busy(True)
        self._nt_status(f"training - {epochs} epochs", C_AMBER)
        self.log(f"[neuro] training {epochs} epochs on "
                 f"{len(self.trainer.patterns)} patterns")
        threading.Thread(target=self._nt_train_worker, args=(epochs,),
                         daemon=True).start()

    def _nt_train_worker(self, epochs):
        tr = self.trainer
        pats = tr.patterns
        n = len(pats)
        rng = np.random.default_rng(tr.cfg.seed + 777)
        done = 0
        surrogate = (tr.cfg.learn_rule == "surrogate")
        # Select on validation only (or explicitly training accuracy for tiny
        # demo sets without a validation partition). Never inspect test here.
        best_acc, best_loss, best_w, best_ep = -1.0, float("inf"), None, 0
        error = None
        output_spikes, weight_change, power_txt = 0, None, None
        try:
            initial_weights = tr.capture_weights()
            for e in range(epochs):
                if self._trainer_stop:
                    break
                if surrogate:
                    frac = e / max(epochs - 1, 1)
                    tr.lr_scale = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * frac))
                ep_loss = 0.0
                for k in rng.permutation(n):
                    if self._trainer_stop:
                        break
                    k = int(k)
                    # surrogate is supervised; STDP uses targets only in
                    # supervised mode
                    tgt = (tr.target_of[k] if (tr.cfg.mode == "supervised"
                           or surrogate) else None)
                    res = tr.train_step(pats[k][1], tgt)
                    ep_loss += float(res.get("loss", 0.0) or 0.0)
                    output_spikes += int(np.sum(res.get("n_out_spikes", [])))
                    self._last_present = (pats[k][0], pats[k][1], res)
                    self.q.put(("nt_spk", res["n_out_spikes"], res["winner"]))
                done = e + 1
                self.q.put(("nt_snap", self._nt_snapshot(done, epochs)))
                # accuracy curve: cheap sampled eval each epoch (train + test)
                tr_acc = tr.eval_accuracy(use_test=False, max_n=40)
                te_acc = (tr.eval_accuracy(split="validation", max_n=40)
                          if tr.validation_patterns else None)
                self.q.put(("nt_acc", done, tr_acc, te_acc))
                # best-so-far checkpoint (every rule, even a partial final epoch)
                score = te_acc if te_acc is not None else tr_acc
                if (score > best_acc + 1e-9 or (abs(score - best_acc) <= 1e-9
                                                and ep_loss < best_loss - 1e-9)):
                    best_acc, best_loss, best_ep = score, ep_loss, done
                    best_w = tr.capture_state()
            # restore the peak checkpoint - on a clean finish OR a forced stop
            if best_w is not None:
                tr.restore_state(best_w)
                self.q.put(("nt_snap", self._nt_snapshot(done, epochs)))
                self.q.put(("log",
                            f"[neuro] kept best weights: {100*best_acc:.0f}% "
                            f"(epoch {best_ep}/{done}; selection: {'validation' if tr.validation_patterns else 'training'})"))
            weight_change = max(float(np.max(np.abs(a-b))) for a, b in
                                zip(tr.capture_weights(), initial_weights))
            power_txt = neuro.power_text(tr, ("init", "train"))
        except Exception as ex:                          # noqa: BLE001
            error = str(ex)
            self.q.put(("log", f"[neuro] training error: {ex!r}"))
        self.q.put(("nt_done", {"epochs": done, "stopped": self._trainer_stop,
                                "best_acc": best_acc, "best_ep": best_ep, "error": error,
                                "output_spikes": output_spikes, "max_weight_change_S": weight_change,
                                "power": power_txt}))

    def on_nt_stop(self, *_):
        if self.trainer_running:
            self._trainer_stop = True
            self._nt_status("stopping...", C_AMBER)

    def on_nt_reset(self, *_):
        if self.trainer_running:
            return
        self.on_nt_build()

    def on_nt_test(self, *_):
        if self._nt_is_stream():
            self._nt_stream_start(False)
            return
        try:
            self.network_studio.require_current()
        except (ValueError, OSError) as exc:
            self._nt_status(str(exc), C_RED)
            return
        if self.trainer is None or self.trainer_running:
            if self.trainer is None:
                self._nt_status("build + train a network first", C_RED)
            return
        self.trainer_running = True
        self._nt_busy(True)
        self._nt_eval_error = None
        self._nt_status("evaluating...", C_AMBER)
        threading.Thread(target=self._nt_test_worker, daemon=True).start()

    def _nt_test_worker(self):
        tr = self.trainer
        try:
            tr.energy.reset("eval")          # energy of THIS evaluation only
            tr_yt, tr_yp = tr.evaluate(use_test=False)
            has_test = bool(tr.test_patterns)
            te_yt, te_yp = (tr.evaluate(use_test=True) if has_test else ([], []))
            power_txt = neuro.power_text(tr, ("init", "train", "eval"))
            # The raster preview is also evaluation: infer advances device time
            # and neuron/RNG state. Keep Evaluate from altering a later resume.
            preview_state = tr.capture_state()
            try:
                res = tr.infer(tr.patterns[0][1])
            finally:
                tr.restore_state(preview_state)
            self._last_present = (tr.patterns[0][0] + " (test)",
                                  tr.patterns[0][1], res)
            snap = self._nt_snapshot(len(self._wevo_epochs), 0)
            self.q.put(("nt_metrics", {
                "train": (tr_yt, tr_yp), "test": (te_yt, te_yp),
                "n": tr.n_out, "names": list(tr.class_names),
                "mode": tr.cfg.mode, "power": power_txt}, snap))
        except Exception as ex:                          # noqa: BLE001
            self.q.put(("log", f"[neuro] eval error: {ex!r}"))
            self.q.put(("nt_metrics", None, None))

    def _on_nt_done(self, info):
        self.trainer_running = False
        self._nt_busy(False)
        self._nt_training_error = info.get("error")
        self._nt_last_training = dict(info)
        self._nt_show_power(info.get("power"))
        if self._nt_training_error:
            self._nt_status(f"training failed: {self._nt_training_error}", C_RED)
            return
        msg = (f"stopped at epoch {info['epochs']}" if info["stopped"]
               else f"trained {info['epochs']} epochs")
        # the trainer now holds its best-accuracy checkpoint (kept even on a
        # forced stop) - tell the user so they know it's safe to Save/Test/resume
        best = info.get("best_acc")
        if best is not None and best >= 0:
            kept = (f"  -  kept best {100*best:.0f}% (epoch "
                    f"{info.get('best_ep', '?')}); Save / Test / Train to resume")
            msg += kept
        unchanged = info.get("max_weight_change_S") == 0
        if unchanged:
            msg += " | No retained weight change. Check spikes, programming drive and device bounds."
        self._nt_status(msg, C_AMBER if unchanged else C_GREEN)
        self.log(f"[neuro] {msg}")
        self._nt_draw_weight_traces()        # refresh the all-synapse trajectory
        # agent-driven run: evaluate, then the metrics get fed back to the agent
        if self._nt_agent_run and not info["stopped"]:
            self.on_nt_test()

    # ---- accuracy curve (streamed during training) ------------------

    def _on_nt_acc(self, epoch, train_acc, validation_acc):
        h = self._nt_acc_hist
        h["epoch"].append(epoch)
        h["train"].append(100.0 * train_acc)
        h["validation"].append(100.0 * validation_acc if validation_acc is not None else None)
        # live metric tiles on the Data tab
        cur = validation_acc if validation_acc is not None else train_acc
        self._nt_best_seen = max(self._nt_best_seen, cur)
        self._nt_stat("nt_stat_epoch", str(epoch))
        self._nt_stat("nt_stat_tracc", f"{100*train_acc:.0f}%")
        self._nt_stat("nt_stat_valacc",
                      f"{100*validation_acc:.0f}%" if validation_acc is not None else "--")
        self._nt_stat("nt_stat_best", f"{100*self._nt_best_seen:.0f}%")
        if not dpg.does_item_exist("nt_acc_y"):
            return
        self._nt_clear("acc")
        ser = [dpg.add_line_series(h["epoch"], h["train"], label="train",
                                   parent="nt_acc_y")]
        if any(v is not None for v in h["validation"]):
            te = [v if v is not None else float("nan") for v in h["validation"]]
            ser.append(dpg.add_line_series(h["epoch"], te, label="validation",
                                           parent="nt_acc_y"))
        self._nt_series["acc"] = ser
        dpg.fit_axis_data("nt_acc_x")
        dpg.fit_axis_data("nt_acc_y")

    # ---- metrics: confusion matrix + per-class P/R/F1 ----------------

    def _on_nt_metrics(self, data, snap):
        self.trainer_running = False
        self._nt_busy(False)
        self._nt_eval_error = None if data else "evaluation failed"
        if snap is not None:
            self._apply_nt_snapshot(snap)
        if not data:
            self._nt_status("evaluation failed", C_RED)
            return
        self._nt_show_power(data.get("power"))
        if data.get("power"):
            self.log("[neuro] " + data["power"])
        n, names = data["n"], data["names"]
        tr_yt, tr_yp = data["train"]
        te_yt, te_yp = data["test"]
        has_test = bool(te_yt)
        # metrics computed on the held-out test set when available, else train
        yt, yp = (te_yt, te_yp) if has_test else (tr_yt, tr_yp)
        cm = neuro.confusion(yt, yp, n)
        m = neuro.prf1(cm)
        tr_acc = (sum(1 for a, b in zip(tr_yt, tr_yp) if a == b) /
                  max(len(tr_yt), 1))
        te_acc = (sum(1 for a, b in zip(te_yt, te_yp) if a == b) /
                  max(len(te_yt), 1)) if has_test else None
        # live metric tiles on the Data tab
        self._nt_stat("nt_stat_tracc", f"{100*tr_acc:.0f}%")
        self._nt_stat("nt_stat_teacc",
                      f"{100*te_acc:.0f}%" if has_test else "--")
        self._nt_stat("nt_stat_best", f"{100*self._nt_best_seen:.0f}%")
        # remember for the agent's context + the auto-analyze loop
        self._nt_last_metrics = {
            "tr_acc": tr_acc, "te_acc": te_acc, "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"], "f1": list(m["f1"]),
            "names": list(names), "cm": cm.tolist()}
        # summary line
        parts = [f"train accuracy {100*tr_acc:.1f}%  ({len(tr_yt)} samples)"]
        if has_test:
            parts.append(f"TEST accuracy {100*te_acc:.1f}%  "
                         f"({len(te_yt)} held-out)")
        parts.append(f"macro-F1 {m['macro_f1']:.3f}")
        parts.append(f"weighted-F1 {m['weighted_f1']:.3f}")
        evalset = "held-out test set" if has_test else "training set"
        dpg.set_value("nt_metrics_summary",
                      "   |   ".join(parts) +
                      f"\n(confusion matrix + per-class metrics below are on the "
                      f"{evalset})")
        self._nt_draw_confusion(cm, names)
        self._nt_build_metric_table(m, names)
        self._nt_status(f"evaluated - "
                        f"{'test' if has_test else 'train'} acc "
                        f"{100*(te_acc if has_test else tr_acc):.0f}%, "
                        f"macro-F1 {m['macro_f1']:.2f}", C_GREEN)
        self.log(f"[neuro] eval: train {100*tr_acc:.1f}%"
                 + (f", test {100*te_acc:.1f}%" if has_test else "")
                 + f", macro-F1 {m['macro_f1']:.3f}, "
                 f"weighted-F1 {m['weighted_f1']:.3f}")
        # close the agent loop: keep tuning until it reaches the best accuracy
        # (stops on a target, on a plateau, or at a generous safety cap - the
        # loop only continues while the agent itself chooses to train again)
        if self._nt_agent_run:
            self._nt_agent_run = False
            acc = te_acc if has_test else tr_acc
            target = min(max(self._nt_get("nt_auto_target", 98.0), 1.0),
                         100.0) / 100.0
            PATIENCE, SAFETY = 5, 20
            if acc > self._nt_best_acc + 1e-3:      # genuine improvement
                self._nt_best_acc = acc
                self._nt_no_improve = 0
            else:
                self._nt_no_improve += 1
            self._nt_agent_rounds += 1
            reached = acc >= target
            plateaued = self._nt_no_improve >= PATIENCE
            capped = self._nt_agent_rounds >= SAFETY
            if not (reached or plateaued or capped) and not self.chat_busy:
                epoch_bumps = getattr(self, "_nt_epoch_bumps", 0)
                nudge = ("  You have already raised nt_epochs "
                         f"{epoch_bumps}x this session - do NOT raise epochs "
                         "again; move a STRUCTURAL lever (learning rule / hidden "
                         "layers / input resolution+afferents / encoding / device "
                         "programming) instead." if epoch_bumps >= 2 else
                         "  If epochs already plateaued the accuracy curve, "
                         "change a STRUCTURAL lever this round (rule / hidden / "
                         "resolution / encoding), not epochs.")
                msg = ("The training + evaluation you started just finished.\n"
                       + self._nt_metrics_text()
                       + f"\n\nBest accuracy so far this session: "
                       f"{100*self._nt_best_acc:.1f}% (target "
                       f"{100*target:.0f}%). KEEP IMPROVING toward the best "
                       "accuracy: diagnose what limited it, change one or two "
                       "controls, and train again (emit nt_action blocks)." + nudge
                       + " Only if you genuinely cannot do better, explain why "
                       "and stop WITHOUT another train action.")
                self.append_chat("sys", f"round #{self._nt_agent_rounds} done "
                                 f"(best {100*self._nt_best_acc:.0f}%) — "
                                 "auto-tuning toward best accuracy")
                self.append_chat("you", "[auto] keep improving the accuracy")
                self._main_send(msg)
            else:
                why = ("target reached" if reached else
                       f"plateaued ({PATIENCE} rounds no gain)" if plateaued
                       else f"safety cap ({SAFETY} rounds)")
                self.append_chat(
                    "sys", f"auto-tune finished — {why}. Best accuracy "
                    f"{100*self._nt_best_acc:.0f}% over {self._nt_agent_rounds} "
                    "rounds. Ask the agent to continue for more.")

    def _nt_draw_confusion(self, cm, names):
        if not dpg.does_item_exist("nt_cm_y"):
            return
        if self._nt_cm_series and dpg.does_item_exist(self._nt_cm_series):
            dpg.delete_item(self._nt_cm_series)
        n, columns = cm.shape
        hi = float(cm.max()) or 1.0
        # ImPlot heatmaps already place the first row at the top. Match the
        # descending Y tick positions below without reversing the data again.
        vals = cm.reshape(-1).astype(float).tolist()
        self._nt_cm_series = dpg.add_heat_series(
            vals, n, columns, scale_min=0.0, scale_max=hi,
            bounds_min=(0.0, 0.0), bounds_max=(float(columns), float(n)),
            format="%.0f", parent="nt_cm_y")
        ticks = tuple((str(names[i]) if i < len(names) else str(i), i + 0.5)
                      for i in range(n))
        dpg.set_axis_ticks("nt_cm_x", ticks + (("No prediction", n + .5),) if columns > n else ticks)
        dpg.set_axis_ticks("nt_cm_y", tuple((lbl, n - p) for lbl, p in ticks))
        dpg.fit_axis_data("nt_cm_x")
        dpg.fit_axis_data("nt_cm_y")

    def _nt_build_metric_table(self, m, names):
        holder = "nt_metrics_holder"
        if not dpg.does_item_exist(holder):
            return
        dpg.delete_item(holder, children_only=True)
        with dpg.table(parent=holder, header_row=True, resizable=True,
                       borders_innerH=True, borders_outerH=True,
                       borders_innerV=True, borders_outerV=True,
                       scrollY=True):
            for col in ("class", "precision", "recall", "F1", "support"):
                dpg.add_table_column(label=col)
            for i in range(len(m["f1"])):
                with dpg.table_row():
                    dpg.add_text(str(names[i]) if i < len(names) else str(i))
                    dpg.add_text(f"{m['prec'][i]:.3f}")
                    dpg.add_text(f"{m['rec'][i]:.3f}")
                    dpg.add_text(f"{m['f1'][i]:.3f}")
                    dpg.add_text(str(int(m["support"][i])))
            with dpg.table_row():
                dpg.add_text("macro avg")
                dpg.add_text("")
                dpg.add_text("")
                dpg.add_text(f"{m['macro_f1']:.3f}")
                dpg.add_text(str(int(m["support"].sum())))
            with dpg.table_row():
                dpg.add_text("weighted avg")
                dpg.add_text("")
                dpg.add_text("")
                dpg.add_text(f"{m['weighted_f1']:.3f}")
                dpg.add_text(str(int(m["support"].sum())))

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
        line = f"{stamp}  {msg}"
        self._log_lines.append(line)            # copyable/readable plain-text mirror
        if len(self._log_lines) > 500:
            self._log_lines = self._log_lines[-500:]
        t = dpg.add_text(line, parent="console", wrap=1200, color=C_TEXT2)
        if "mono" in self.fonts:
            dpg.bind_item_font(t, self.fonts["mono"])
        kids = dpg.get_item_children("console", 1) or []
        if len(kids) > 500:
            dpg.delete_item(kids[0])
        self._scroll_bottom("console")

    def _copy_log(self):
        """Copy the whole console to the clipboard (add_text isn't selectable)."""
        dpg.set_clipboard_text("\n".join(self._log_lines))
        self.log(f"[log] copied {len(self._log_lines)} lines to clipboard")

    def _save_log(self):
        path = os.path.join(self.workdir, "results", "session_log.txt")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self._log_lines) + "\n")
            self.log(f"[log] saved -> {path}")
        except OSError as e:
            self.log(f"[log] save failed: {e}")

    def _clear_log(self):
        self._log_lines = []
        for k in (dpg.get_item_children("console", 1) or []):
            dpg.delete_item(k)

    def _on_resize(self):
        vw = dpg.get_viewport_client_width()
        if dpg.does_item_exist("brand_spacer"):
            dpg.configure_item("brand_spacer", width=max(10, vw-1040))
        drawer = RIGHT_W + 8 if dpg.does_item_exist("right_child") and dpg.get_item_configuration("right_child")["show"] else 0
        center = max(430, vw - LEFT_W - drawer - 42)
        if dpg.does_item_exist("center_child"):
            dpg.configure_item("center_child", width=center)
        if dpg.does_item_exist("param_panel"):
            dpg.configure_item("param_panel", height=max(190, dpg.get_viewport_client_height()-580))
    # =================================================================
    # main loop
    # =================================================================

    def _process_queue(self):
        self.network_studio.tick()
        while True:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                return
            try:
                kind = item[0]
                if kind == "log":
                    self.log(item[1])
                elif kind == "compute":
                    self.compute_panel.finished(item[1], item[2])
                elif kind == "results":
                    self.sim_running = False
                    self._show_results(item[1], item[2], item[3], item[4],
                                       live=(len(item) > 5 and item[5]))
                elif kind == "stdp":
                    self._show_stdp(item[1], item[2],
                                    live=(len(item) > 3 and item[3]),
                                    diagnostics=item[4] if len(item) > 4 else None)
                elif kind == "polar":
                    self._show_polar(item[1], item[2] if len(item) > 2
                                     else POLAR_MEASUREMENTS[0])
                elif kind == "polar_pund":
                    self._show_pund(item[1])
                elif kind == "polar_kin":
                    self._show_kinetics(item[1])
                elif kind == "models":
                    self._on_models_done(item[1], item[2], item[3])
                elif kind == "chat":
                    self._on_chat_done(item[1], item[2])
                elif kind == "chat_progress":
                    self._chat_progress(item[1])
                elif kind == "account":
                    self._on_account_status(item[1])
                elif kind == "virtuoso":
                    self._on_virtuoso_done(item[1], item[2])
                elif kind == "virt_browse":
                    self._on_virt_browse(item[1], item[2])
                elif kind == "nt_snap":
                    self._apply_nt_snapshot(item[1])
                elif kind == "nt_done":
                    self._on_nt_done(item[1])
                elif kind == "nt_metrics":
                    self._on_nt_metrics(item[1], item[2])
                elif kind == "nt_stream":
                    self._on_nt_stream(item[1])
                elif kind == "nt_stream_done":
                    self._on_nt_stream_done(item[1])
                elif kind == "nt_acc":
                    self._on_nt_acc(item[1], item[2], item[3])
                elif kind == "nt_spk":
                    self._on_nt_spk(item[1], item[2])
                elif kind == "nt_dataset":
                    self._on_nt_dataset(item[1], item[2])
                elif kind == "nt_dataset_loaded":
                    self._on_nt_dataset_loaded(item[1], item[2])
            except Exception as e:
                self.log(f"[ui] error: {e!r}")

    # ---------------- Tools menu: model utilities --------------------
    def _twin_defaults(self):
        """{model_key: {param_name: numeric default}} from the twin dataclasses."""
        out = {}
        for key, spec in SPEC_BY_KEY.items():
            d = _defaults_of(spec.params_cls)
            out[key] = {k: v for k, v in d.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return out

    def check_va_consistency(self):
        """Tools: compare every mapped .va's params to its twin. Returns a report."""
        from vatester import tools_ext as T
        res = T.check_consistency(self.va_files, self._twin_defaults(), TWIN_FILE)
        head, lines = T.format_consistency(res)
        self.log("[tools] " + head)
        for ln in lines:
            self.log(ln)
        return res

    def _goto_log(self):
        """Surface the console so Tools output is visible (matches Virtuoso menu)."""
        if dpg.does_item_exist("center_tabs"):
            dpg.set_value("center_tabs", "tab_log")

    def on_check_consistency(self, *_):
        self._goto_log()
        self.check_va_consistency()

    def openvaf_compile_va(self, path=None):
        """Tools: compile .va(s) with OpenVAF. path=None -> enabled (else all)."""
        from vatester import tools_ext as T
        if path:
            targets = [path]
        else:
            en = set(self._enabled_keys())
            targets = [v.path for v in self.va_files if v.model_key in en] \
                or [v.path for v in self.va_files]
        if not targets:
            self.log("[openvaf] no .va files to compile")
            return {"results": []}
        results = []
        for p in targets:
            r = T.openvaf_compile(p, workdir=self.workdir)
            r["va"] = os.path.basename(p)
            results.append(r)
            if not r["available"]:
                self.log("[openvaf] " + r["output"])
                break
            tag = "OK" if r["ok"] else f"FAILED (rc={r.get('returncode')})"
            msg = f"[openvaf] {r['va']}: {tag}"
            if r.get("osdi"):
                msg += f" -> {os.path.basename(r['osdi'])}"
            self.log(msg)
            if not r["ok"] and r.get("output"):
                for ln in r["output"].splitlines()[-8:]:
                    self.log("  " + ln)
        return {"results": results}

    def on_openvaf_compile(self, *_):
        self._goto_log()
        self.openvaf_compile_va()

    def fit_curve_csv(self, path):
        """Tools: fit the LTP/LTD nonlinearity of a measured G-per-pulse CSV."""
        from vatester import tools_ext as T
        try:
            g = T.read_curve_csv(path)
            res = T.fit_ltp_ltd(g)
        except Exception as e:
            self.log(f"[fit] failed: {e}")
            return {"ok": False, "error": str(e)}
        self.log(f"[fit] {os.path.basename(path)}")
        for ln in T.format_fit(res):
            self.log(ln)
        res["ok"] = True
        return res

    def _on_fit_csv_file(self, sender, app_data):
        sel = app_data.get("file_path_name") or ""
        paths = list((app_data.get("selections") or {}).values())
        path = paths[0] if paths else sel
        if path:
            self._goto_log()
            self.fit_curve_csv(path)

    def _on_toggle_bridge(self, *_):
        """Tools menu: connect/disconnect the control bridge at runtime."""
        want = not self._bridge_enabled    # flip current state
        self._bridge_enabled = want
        if self.control is not None:        # None only pre-run; flag suffices then
            if want:
                self.control.start()
            else:
                self.control.stop()
            want = self.control.active      # reflect what actually happened
            self._bridge_enabled = want
        self._set_bridge_menu(want)

    def _set_bridge_menu(self, connected):
        """Refresh the Tools-menu status line + action label to match state."""
        if not dpg.does_item_exist("bridge_dot"):
            return
        col = C_GREEN if connected else C_RED
        dpg.configure_item("bridge_dot", color=col)
        dpg.set_value("bridge_state_txt",
                      "connected" if connected else "disconnected")
        dpg.configure_item("bridge_state_txt", color=col)
        dpg.set_item_label("menu_bridge",
                           "Disconnect bridge" if connected else "Connect bridge")

    def run(self, smoke_frames=0):
        self.build()
        # ask the providers what models they offer now (cached for a day), so
        # a model released after this build is still pickable. Off-thread;
        # the picker shows the offline list until the answer lands.
        self._refresh_models_async()
        # always instantiate so the Tools-menu toggle can flip it live; it only
        # opens the mailbox when active (default = self._bridge_enabled).
        from vatester.control_bridge import ControlBridge
        self.control = ControlBridge(self, active=self._bridge_enabled)
        self._set_bridge_menu(self.control.active)      # sync the Tools status
        frame = 0
        while dpg.is_dearpygui_running():
            self._process_queue()
            self.desktop.tick()
            self._tick_zoom_anim()
            self._tick_probe_drag()
            self._hide_tip_if_stale()
            self._tick_menu_dismiss()
            self._nt_tick_layout()
            self._nt_tick_paint()
            self._nt_tick_cell_anim()
            self._chat_tick()
            self.noise_panel.tick()
            self._watch_code()
            self._nt_watch_patterns()
            if self.control is not None:        # process one bridge command/frame
                self.control.poll()
            dpg.render_dearpygui_frame()
            frame += 1
            if smoke_frames and frame >= smoke_frames:
                break
        self.virtuoso.disconnect()
        dpg.destroy_context()
        if self._restart:               # relaunch to apply GUI-code edits
            os.execv(sys.executable, [sys.executable] + sys.argv)


def main(workdir=None, smoke_frames=0, bridge=None):
    # keep the smoke self-test clean: never open the mailbox during it
    if smoke_frames:
        bridge = False
    App(workdir or os.getcwd(), bridge=bridge).run(smoke_frames=smoke_frames)
