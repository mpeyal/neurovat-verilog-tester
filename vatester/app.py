"""NeuroVAT - neuromorphic Verilog-A model tester (DearPyGui).

Layout
  toolbar   brand - Run / quick actions / status
  left      .va browser - models - parameter editor - sim settings
  center    Signal Designer | Results | Analysis | Verilog-A Source | Log
  right     Claude agent chat (bubbles, pattern cards, prompt chips)
"""

import bisect
import contextlib
import dataclasses
import importlib
import glob
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

from ecfet import (Waveform, EcfetV1, V1Params, EcfetV2, V2Params,
                   EcfetV3, V3Params, FeFET, FeFETParams, simulate)
from . import signal_factory as sf
from . import neuro
from .agent import ClaudeAgent
from .textshape import TextShaper, needs_shaping
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
OPENAI_MODEL_CHOICES = [
    ("Default", None, "gpt-5.1"),
    ("GPT-5.1", "gpt-5.1", "flagship"),
    ("GPT-5.1 mini", "gpt-5.1-mini", "fast, cheap"),
]
PROVIDER_MODELS = {"claude": MODEL_CHOICES, "openai": OPENAI_MODEL_CHOICES}
PROVIDER_TITLES = {"claude": "Claude Agent", "openai": "OpenAI Agent"}

# ---- Neuromorphic Trainer agent: the controls it may set + its instructions --
# (tag, one-line meaning) - listed to the agent with current values so it can
# tune the studio via the nt_action "set" block.
NT_CONTROLS = [
    ("nt_gh", "input grid height"), ("nt_gw", "input grid width"),
    ("nt_nout", "output neurons / classes"),
    ("nt_hidden", "hidden layer sizes, comma list e.g. '8' or '8,4' ('' = none)"),
    ("nt_hidden_gain", "extra drive on hidden layers (raise if hidden quiet)"),
    ("nt_mode", "supervised | unsupervised"),
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
    ("nt_encoding", "rate (Poisson) | latency (TTFS)"),
    ("nt_bg_rate", "background firing Hz"), ("nt_input_noise", "sensor noise 0-1"),
    ("nt_signal_frac", "fraction of afferents carrying the pattern 0-1"),
    ("nt_jitter", "spike jitter ms"), ("nt_vnoise", "membrane noise sigma"),
    ("nt_wnoise", "device write noise sigma_c2c"),
    ("nt_ds_res", "dataset downsample resolution"),
    ("nt_ds_perclass", "dataset images per class"),
]

NT_AGENT_SYSTEM = """
=== NEUROMORPHIC TRAINER (the "Neuro Trainer" tab in this same app) ===
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
    its Python twin (twins/ or vatester/ecfet*.py / models) with your normal file
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
nt_hidden_gain. Keep replies short and concrete; say what you changed and why.
Use exact combo strings for nt_learnrule / nt_mode / nt_encoding / nt_patset.
"""


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
    ModelSpec("v3", "ECFET v3 (paper-faithful)", EcfetV3, V3Params, "current"),
    ModelSpec("fefet", "FeFET (Merz/Preisach-lite)", FeFET, FeFETParams, "voltage"),
]
SPEC_BY_KEY = {s.key: s for s in MODEL_SPECS}
GEN_BY_NAME = {g.name: g for g in sf.GENERATORS}

# device classes group the model keys; selecting a class reconfigures the GUI
# (which models are enabled, current vs voltage drive, ΔG vs ΔVt, polarization)
DEVICE_FAMILIES = {"ECFET": ("v1", "v2", "v3"), "FeFET": ("fefet",)}
DEVICE_OF_KEY = {k: dev for dev, keys in DEVICE_FAMILIES.items() for k in keys}
DEVICE_KIND = {"ECFET": "current", "FeFET": "voltage"}   # default drive

# model_key -> the Python "twin" source the GUI actually simulates
TWIN_FILE = {"v1": "ecfet/model_v1.py", "v2": "ecfet/model_v2.py",
             "v3": "ecfet/model_v3.py", "fefet": "ecfet/model_fefet.py"}
LABEL_TO_KEY = {s.label: s.key for s in MODEL_SPECS}

# modules to hot-reload when the agent edits a twin, so the GUI re-simulates
# with the new code (key -> (module, class name, params class name))
from ecfet import (model_v1 as _m_v1, model_v2 as _m_v2,
                   model_v3 as _m_v3, model_fefet as _m_fefet)
RELOAD_MODULES = {
    "v1": (_m_v1, "EcfetV1", "V1Params"),
    "v2": (_m_v2, "EcfetV2", "V2Params"),
    "v3": (_m_v3, "EcfetV3", "V3Params"),
    "fefet": (_m_fefet, "FeFET", "FeFETParams"),
}

# The MEASUREMENT layer (STDP sweep, per-pulse sampling) lives in
# vatester/analysis.py and is hot-reloaded LIVE like the twins, so editing the
# analysis math takes effect with no restart.  app.py always calls it as
# `analysis.func(...)` (attribute lookup) so importlib.reload swaps in new code.
from vatester import analysis
# every file the live watcher polls + reloads (twins + the analysis layer)
WATCH_FILES = list(TWIN_FILE.values()) + ["vatester/analysis.py"]

# ---- dynamic device twins (the user/agent-extensible `twins/` folder) --------
# New devices live OUTSIDE the app source in a top-level `twins/` directory and
# are registered at startup, so users never edit the GUI/core to add a model.
from . import va_scan as _va_scan_mod
from . import twin_loader
from . import pattern_loader

# user/agent input-PATTERN plugins (the `patterns/` folder, hot-watched). Built
# in the GUI; plugins ADD to it without editing the app or restarting.
BUILTIN_PATSETS = ["bars", "letters", "digits", "random", "nand", "xor"]
PATTERN_PLUGINS = {}            # key -> PATTERN_SPEC dict (make(cfg)->pats,tgts)


def register_pattern_plugins(workdir):
    """(Re)load every patterns/*.py into PATTERN_PLUGINS. Returns (keys,errs)."""
    results = pattern_loader.load_patterns(os.path.join(workdir, "patterns"))
    reg, errs = {}, []
    for path, payload, err in results:
        if err:
            errs.append((os.path.basename(path), err))
            continue
        _mod, ps = payload
        reg[ps["key"]] = ps
    PATTERN_PLUGINS.clear()
    PATTERN_PLUGINS.update(reg)
    return list(reg), errs


def _register_twin(mod, ts):
    """Register one dynamic twin (its TWIN_SPEC) into the model registry."""
    key = ts["key"]
    if key in SPEC_BY_KEY:
        return key                                  # built-in or already loaded
    mcls, pcls = ts["model_class"], ts["params_class"]
    kind = ts.get("input_kind", "current")
    spec = ModelSpec(key, ts["label"], mcls, pcls, kind)
    MODEL_SPECS.append(spec)
    SPEC_BY_KEY[key] = spec
    LABEL_TO_KEY[spec.label] = key
    RELOAD_MODULES[key] = (mod, mcls.__name__, pcls.__name__)
    if getattr(mod, "__file__", None):
        TWIN_FILE[key] = mod.__file__
        WATCH_FILES.append(mod.__file__)
    dev = ts.get("device_class", "ECFET")
    DEVICE_FAMILIES[dev] = tuple(DEVICE_FAMILIES.get(dev, ())) + (key,)
    DEVICE_OF_KEY[key] = dev
    DEVICE_KIND.setdefault(dev, kind)
    _va_scan_mod.MODEL_HINTS.insert(0, (key, tuple(ts.get("va_keywords", (key,)))))
    # optional profile -> set as class attributes the GUI already reads
    p = ts.get("stdp")
    if p:
        mcls.STDP_OBS = p.get("obs", "G")
        mcls.STDP_LABEL = p.get("label", "dG")
        mcls.STDP_UNIT = p.get("unit", "uS")
        mcls.STDP_SCALE = p.get("scale", 1e6)
    if ts.get("result_plots"):
        mcls.RESULT_PLOTS = tuple(tuple(x) for x in ts["result_plots"])
    if ts.get("analysis_metrics"):
        mcls.ANALYSIS_METRICS = tuple(tuple(x) for x in ts["analysis_metrics"])
    if ts.get("polar_obs"):
        mcls.POLAR_OBS = ts["polar_obs"]
    if ts.get("analyses"):
        mcls.ANALYSES = tuple(ts["analyses"])
    return key


def register_dynamic_twins(workdir):
    """Load + register every twin under <workdir>/twins. (loaded, errors)."""
    loaded, errors = [], []
    for path, payload, err in twin_loader.load_twins(
            os.path.join(workdir, "twins")):
        if err:
            errors.append((os.path.basename(path), err))
            continue
        mod, ts = payload
        try:
            loaded.append(_register_twin(mod, ts))
        except Exception as e:                       # noqa: BLE001
            errors.append((os.path.basename(path), f"register failed: {e}"))
    return loaded, errors


def _defaults_of(params_cls):
    inst = params_cls()
    return {f.name: getattr(inst, f.name)
            for f in dataclasses.fields(params_cls)}


class App:
    def __init__(self, workdir):
        self.workdir = os.path.abspath(workdir)
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
        # Default Claude to Opus 4.8 explicitly (otherwise the CLI's own default
        # may pick Fable 5).
        self.model_sel = {"claude": ("Opus 4.8", "claude-opus-4-8"),
                          "openai": ("Default", None)}
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
        self._nt_acc_hist = {"epoch": [], "train": [], "test": []}
        self._nt_cm_series = None       # confusion-matrix heat series
        self._nt_last_metrics = None    # last eval summary (for the agent)
        self._nt_agent_run = False      # an agent-triggered train is in flight
        self._nt_agent_rounds = 0       # auto-analyze loop guard
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
        if (dpg.does_item_exist("nt_canvas_scroll")
                and dpg.is_item_hovered("nt_canvas_scroll")):
            self._nt_set_zoom(self._nt_zoom * (1.12 ** float(delta)))
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
                if y != y:                       # NaN gap-break point
                    continue
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

    def _get_hover_data(self, plot):
        """Cached (label, xs, ys) lists for a plot's data series, so the hover
        handler doesn't copy big arrays every frame. Rebuilt lazily after a
        re-plot (the cache is cleared when data changes)."""
        data = self._hover_cache.get(plot)
        if data is not None:
            return data
        yax = self.PROBE_AXES[plot][1]
        data = []
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
            data.append((lbl, list(val[0]), list(val[1])))
        self._hover_cache[plot] = data
        return data

    @staticmethod
    def _nearest_on_series(xs, ys, mx, my, xspan, yspan):
        """Nearest point on a single x-sorted series via binary search +
        outward scan with early termination (x-distance is monotonic). Returns
        (d2, x, y) or None. Skips NaN y (gap breaks)."""
        n = len(xs)
        if n == 0:
            return None
        i = bisect.bisect_left(xs, mx)
        best = None
        lo, hi = i - 1, i
        while lo >= 0 or hi < n:
            for j in (lo, hi):
                if 0 <= j < n:
                    dxn = (xs[j] - mx) / xspan
                    if best is not None and dxn * dxn >= best[0]:
                        continue
                    yv = ys[j]
                    if yv != yv:                 # NaN
                        continue
                    d = dxn * dxn + ((yv - my) / yspan) ** 2
                    if best is None or d < best[0]:
                        best = (d, float(xs[j]), float(yv))
            ldx = ((xs[lo] - mx) / xspan) ** 2 if lo >= 0 else float("inf")
            hdx = ((xs[hi] - mx) / xspan) ** 2 if hi < n else float("inf")
            if best is not None and ldx >= best[0] and hdx >= best[0]:
                break
            lo -= 1
            hi += 1
        return best

    def _probe_near(self, plot, mx, my, px_radius=16.0):
        """'A'/'B' if the cursor is within px_radius pixels of that probe."""
        probes = self._probes.get(plot)
        if not probes:
            return None
        xax, yax = self.PROBE_AXES[plot]
        x0, x1 = dpg.get_axis_limits(xax)
        y0, y1 = dpg.get_axis_limits(yax)
        w, h = dpg.get_item_rect_size(plot) or (1, 1)
        pxx = (w or 1) / ((x1 - x0) or 1.0)
        pxy = (h or 1) / ((y1 - y0) or 1.0)
        best = None
        for which in ("A", "B"):
            m = probes.get(which)
            if not m:
                continue
            d2 = ((m["x"] - mx) * pxx) ** 2 + ((m["y"] - my) * pxy) ** 2
            if d2 <= px_radius ** 2 and (best is None or d2 < best[0]):
                best = (d2, which)
        return best[1] if best else None

    def _set_plot_pan_disabled(self, plot, disabled):
        """Park the plot's pan on an unused button while the cursor is on a
        probe, so pressing the mouse grabs the probe instead of panning."""
        if (plot in self._pan_disabled) == bool(disabled):
            return
        try:
            dpg.configure_item(plot, pan_button=(dpg.mvMouseButton_X2
                                                 if disabled else
                                                 dpg.mvMouseButton_Left))
        except Exception:
            return
        if disabled:
            self._pan_disabled.add(plot)
        else:
            self._pan_disabled.discard(plot)

    def _drag_probe_to(self, plot, which, mx, my):
        m = self._probes.get(plot, {}).get(which)
        if not m:
            return
        dpg.set_value(m["pt"], (mx, my))
        if m.get("ann"):
            dpg.configure_item(m["ann"], default_value=(mx, my))
        m["x"], m["y"] = mx, my
        self._update_probe_readout(plot)

    def _units_per_px(self, plot, gx, gy, mx, my):
        """Plot-units per screen pixel at grab time. Baseline: axis span over
        the item rect minus typical axis-label margins (sign always right).
        Refined by a least-squares fit over consecutive hover samples when
        the history is consistent with that estimate."""
        xax, yax = self.PROBE_AXES[plot]
        x0, x1 = dpg.get_axis_limits(xax)
        y0, y1 = dpg.get_axis_limits(yax)
        w, h = dpg.get_item_rect_size(plot) or (1, 1)
        ux = (x1 - x0) / max((w or 1) - 62.0, 1.0)
        uy = -(y1 - y0) / max((h or 1) - 30.0, 1.0)  # screen y grows downward
        hist = self._hover_hist.get(plot, [])
        sxx = sxm = syy = sym = 0.0
        for (g1x, g1y, m1x, m1y), (g2x, g2y, m2x, m2y) in zip(hist, hist[1:]):
            dgx, dgy = g2x - g1x, g2y - g1y
            sxx += dgx * dgx
            sxm += dgx * (m2x - m1x)
            syy += dgy * dgy
            sym += dgy * (m2y - m1y)
        if sxx > 25.0:
            cand = sxm / sxx
            if cand * ux > 0 and 0.4 < abs(cand / ux) < 2.5:
                ux = cand
        if syy > 25.0:
            cand = sym / syy
            if cand * uy > 0 and 0.4 < abs(cand / uy) < 2.5:
                uy = cand
        return ux, uy

    def _tick_probe_drag(self):
        """Render-loop drag driver: while a probe is grabbed, move it with
        the GLOBAL mouse position (which stays live during the hold, unlike
        the plot-space mouse position)."""
        if not self._dragging or not self._drag_anchor:
            return
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            return                       # the release handler finishes up
        plot, which = self._dragging
        gx0, gy0, mx0, my0, ux, uy = self._drag_anchor
        gx, gy = dpg.get_mouse_pos(local=False)
        self._drag_probe_to(plot, which,
                            mx0 + (gx - gx0) * ux,
                            my0 + (gy - gy0) * uy)

    def _on_plot_hover(self, sender, app_data, user_data):
        """Draw an in-canvas bubble at the nearest data point, showing its
        x / y values with units. Fires while the plot is hovered. Also drives
        the manual probe drag (independent of ImPlot's native grab)."""
        plot = user_data
        if plot not in self.HOVER_FMT:
            return
        try:
            mx, my = dpg.get_plot_mouse_pos()
        except Exception:
            self._hide_hover_bubble()
            return
        down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        if down:
            # drag motion is driven by _tick_probe_drag (the plot-space mouse
            # pos FREEZES while the button is held, so it is useless here);
            # just keep the bubble out of the way
            self._hide_hover_bubble()
            self._hover_last = None
            return
        # button up: remember px<->plot-unit samples for the drag mapping
        gx, gy = dpg.get_mouse_pos(local=False)
        hist = self._hover_hist.setdefault(plot, [])
        if not hist or (abs(gx - hist[-1][0]) + abs(gy - hist[-1][1])) >= 3:
            hist.append((gx, gy, mx, my))
            if len(hist) > 24:
                del hist[0]
        # when the cursor sits on a probe, park the plot's pan so the next
        # press grabs the probe (not the canvas)
        near = self._probe_near(plot, mx, my)
        self._set_plot_pan_disabled(plot, near is not None)
        if near:
            self._hide_hover_bubble()
            self._hover_last = None
            self._tip_frame = dpg.get_frame_count()
            return
        xax, yax = self.PROBE_AXES[plot]
        x0, x1 = dpg.get_axis_limits(xax)
        y0, y1 = dpg.get_axis_limits(yax)
        xs_span = (x1 - x0) or 1.0
        ys_span = (y1 - y0) or 1.0
        # throttle: if the cursor barely moved since the last computed bubble,
        # keep the bubble alive but skip the (potentially heavy) search
        last = self._hover_last
        if (last and last[0] == plot
                and abs(mx - last[1]) < xs_span * 0.004
                and abs(my - last[2]) < ys_span * 0.004):
            self._tip_frame = dpg.get_frame_count()
            return
        self._hover_last = (plot, mx, my)
        best = None
        labels = set()
        for lbl, xs, ys in self._get_hover_data(plot):
            cl = self._clean_label(lbl)
            if cl:
                labels.add(cl)
            cand = self._nearest_on_series(xs, ys, mx, my, xs_span, ys_span)
            if cand is not None and (best is None or cand[0] < best[0]):
                best = (cand[0], cand[1], cand[2], lbl)
        if best is None or best[0] > 0.05 ** 2:
            self._hide_hover_bubble()
            return
        _, x, y, lbl = best
        xname, xunit, yname, yunit = self.HOVER_FMT[plot]
        # the two transient plots relabel per device profile (R/G vs Vth/P)
        ro = getattr(self, "_result_obs", None)
        if ro and plot in ("plot_r", "plot_g"):
            _, yname, yunit = ro[0 if plot == "plot_r" else 1]
        unit = dpg.get_value("unit_combo") if dpg.does_item_exist(
            "unit_combo") else ""
        xunit = xunit.replace("{unit}", unit)
        yunit = yunit.replace("{unit}", unit)
        # identify the curve (its legend label) when there are several or the
        # legend is hidden - so a hidden legend still tells you which is which
        cl = self._clean_label(lbl)
        head = (cl + "\n") if (cl and (not self.legends_visible
                                       or len(labels) > 1)) else ""
        text = (head + f"{yname} = {y:.4g} {yunit}".rstrip() + "\n"
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
            if m.get("ann"):
                dpg.configure_item(m["ann"], default_value=(x, y))
        else:
            # no_inputs: the native ImPlot grab is disabled on purpose - it
            # takes ActiveId on press, which FREEZES the plot's mouse-position
            # updates and breaks the drag. The marker is visual-only; all
            # drag motion is driven manually in _on_plot_hover with live
            # get_plot_mouse_pos coordinates.
            pt = dpg.add_drag_point(parent=plot, default_value=(x, y),
                                    color=color, thickness=4,
                                    no_inputs=True)
            # the A / B label bubble that rides next to the marker
            ann = dpg.add_plot_annotation(parent=plot, label=which,
                                          default_value=(x, y),
                                          offset=(12, -12), color=color,
                                          clamped=True)
            probes[which] = m = {"pt": pt, "ann": ann}
        m.update(x=x, y=y, series=slabel, sid=sid)
        self._update_probe_readout(plot)

    def _on_probe_release(self, sender, app_data):
        """On mouse-up after a drag, snap the moved probe onto the nearest
        curve point."""
        d = self._dragging
        self._dragging = None
        self._drag_anchor = None
        if not d:
            return
        plot, which = d
        self._set_plot_pan_disabled(plot, False)
        m = self._probes.get(plot, {}).get(which)
        if not m:
            return
        xax, yax = self.PROBE_AXES[plot]
        x0, x1 = dpg.get_axis_limits(xax)
        y0, y1 = dpg.get_axis_limits(yax)
        xs_span = (x1 - x0) or 1.0
        ys_span = (y1 - y0) or 1.0
        best, best_lbl = None, None
        for lbl, xs, ys in self._get_hover_data(plot):
            cand = self._nearest_on_series(xs, ys, m["x"], m["y"],
                                           xs_span, ys_span)
            if cand is not None and (best is None or cand[0] < best[0]):
                best, best_lbl = cand, lbl
        if best is not None:
            _, x, y = best
            dpg.set_value(m["pt"], (x, y))
            if m.get("ann"):
                dpg.configure_item(m["ann"], default_value=(x, y))
            m["x"], m["y"], m["series"] = x, y, best_lbl
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
        # click a synapse on the trainer crossbar canvas -> inspect it
        if (dpg.does_item_exist("nt_diagram")
                and dpg.is_item_hovered("nt_diagram") and self._nt_cell_hits):
            self._nt_pick_cell()
            return
        plot = self._hovered_probe_plot()
        if not self._probe_armed:
            if not plot:
                return
            try:
                mx, my = dpg.get_plot_mouse_pos()
            except Exception:
                return
            # click on an existing probe = grab it (manual drag starts)
            which = self._probe_near(plot, mx, my)
            if which:
                gx, gy = dpg.get_mouse_pos(local=False)
                ux, uy = self._units_per_px(plot, gx, gy, mx, my)
                self._dragging = (plot, which)
                self._drag_anchor = (gx, gy, mx, my, ux, uy)
                return
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
            lb = dpg.add_button(label=" Legend ", small=True,
                                callback=self._toggle_legends)
            with dpg.tooltip(lb):
                dpg.add_text("Show/hide the legend (when hidden, hover a curve "
                             "to see its label). Click a legend entry to "
                             "hide/show that single curve.")
            ob = dpg.add_button(label=" Leg out ", small=True,
                                callback=self._toggle_legend_outside)
            with dpg.tooltip(ob):
                dpg.add_text("Move the legend outside / inside the plot canvas "
                             "(a big legend can cover the curves).")
            self._small("wheel zoom · drag pan · dbl-click fit · "
                        "A / B then click = place probe · C = clear")
        if probe_tag:
            self._small("", tag=probe_tag, color=(126, 170, 255))

    LEGEND_TAGS = ("preview_plot_leg", "plot_i_leg", "plot_r_leg", "plot_g_leg",
                   "ana_plot_leg", "stdp_plot_leg", "polar_plot_leg")

    def _toggle_legends(self, *_):
        self.legends_visible = not self.legends_visible
        for t in self.LEGEND_TAGS:
            if dpg.does_item_exist(t):
                dpg.configure_item(t, show=self.legends_visible)
        self._sync_leg_checks()                   # keep menu checkboxes in sync
        self.log(f"[plot] legends {'shown' if self.legends_visible else 'hidden'}"
                 + ("" if self.legends_visible else
                    " - hover a curve to see its label"))

    def _toggle_legend_outside(self, *_):
        self.legend_outside = not self.legend_outside
        for t in self.LEGEND_TAGS:
            if dpg.does_item_exist(t):
                dpg.configure_item(t, outside=self.legend_outside)
        self._sync_leg_checks()
        self.log(f"[plot] legend moved {'outside' if self.legend_outside else 'inside'}"
                 " the plot canvas")

    @staticmethod
    def _clean_label(lbl):
        """Strip the '##'/'##pts_' prefixes used to hide helper series."""
        s = (lbl or "").lstrip("#")
        if s.startswith("pts_"):
            s = s[4:]
        return s.strip()

    # =================================================================
    # UI construction
    # =================================================================

    def build(self):
        dpg.create_context()
        self._font()
        self._theme()
        dpg.add_texture_registry(tag="chat_tex_reg")   # shaped-chat textures

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
            self._small("OpenAI provider (/provider openai): API key for the "
                        "OpenAI SDK backend (also read from OPENAI_API_KEY):",
                        wrap=440)
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="openai_key_in", password=True,
                                   width=304, hint="paste sk-...")
                bo = dpg.add_button(label="Apply",
                                    callback=self.on_openai_key_apply)
                dpg.bind_item_theme(bo, self.themes["primary"])
                dpg.add_button(label="Clear",
                               callback=self.on_openai_key_clear)
            dpg.add_separator()
            dpg.add_button(label="Close", width=90, callback=lambda:
                           dpg.configure_item("account_modal", show=False))

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

        # confirm dialog for writing source back into a Virtuoso cellview
        with dpg.window(label="Write back to Virtuoso", tag="virt_write_modal",
                        modal=True, show=False, no_resize=True, width=470,
                        autosize=True):
            dpg.add_text("", tag="virt_write_msg", wrap=440)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                cb = dpg.add_button(label="Write to library",
                                    callback=self._do_virt_write_back)
                dpg.bind_item_theme(cb, self.themes["primary"])
                dpg.add_button(label="Cancel", callback=lambda: dpg.configure_item(
                    "virt_write_modal", show=False))

        # auto-prompt: Verilog-A files that have no Python twin yet
        with dpg.window(label="Verilog-A without a twin", tag="untwin_modal",
                        modal=True, show=False, no_resize=True, width=540,
                        autosize=True):
            dpg.add_text("", tag="untwin_msg", wrap=510)
            with dpg.child_window(tag="untwin_list", auto_resize_y=True,
                                  border=False):
                pass
            dpg.add_separator()
            dpg.add_button(label="Not now", callback=lambda: dpg.configure_item(
                "untwin_modal", show=False))

        # Parametric Sweep dialog (Virtuoso-style): manage per-variable sweep
        # specs (From/To, Center/Span%, explicit Values) in engineering units
        with dpg.window(label="Parametric Sweep", tag="sweep_dialog",
                        modal=True, show=False, no_resize=False, width=580,
                        height=460):
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
            self._small("engineering units OK:  4u = 4e-6,  3m = 3e-3,  "
                        "100k = 1e5.   Run / Plot STDP / Plot P-V overlay one "
                        "curve per value (2+ variables = a grid).", wrap=540)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                bo = dpg.add_button(label="OK", width=80,
                                    callback=self._on_sweep_ok)
                dpg.bind_item_theme(bo, self.themes["primary"])
                dpg.add_button(label="Cancel", width=80,
                               callback=self._on_sweep_cancel)

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
        self._check_orphan_twins()        # warn about unregistered ecfet/ twins
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
                dpg.add_menu_item(label="Neuromorphic Trainer studio...",
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
                    dpg.add_text("Model twins (ecfet/model_*.py) hot-reload "
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
            bn = dpg.add_button(label="Neuro Trainer", tag="btn_trainer",
                                callback=self.on_open_trainer)
            dpg.bind_item_theme(bn, self.themes["primary"])
            with dpg.tooltip(bn):
                dpg.add_text("Open the Neuromorphic Trainer studio: wire the "
                             "selected device into a crossbar of synapses + "
                             "spiking neurons, train it with STDP and watch "
                             "the weights (device conductances) update live.")
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
                                       default_open=True, tag="lh_va") as h:
                if "bold" in self.fonts:
                    dpg.bind_item_font(h, self.fonts["bold"])
                with self._pad():
                    with dpg.child_window(tag="va_cards", auto_resize_y=True,
                                          border=True) as vc:
                        pass
                    dpg.bind_item_theme(vc, self.themes["card"])
                    with dpg.tooltip(vc):
                        dpg.add_text("check = simulate   ·   click = load "
                                     "params   ·   double-click = edit source\n"
                                     "one device class at a time - checking a "
                                     "FeFET unchecks ECFET (and vice versa); "
                                     "the GUI auto-switches drive + plots")

            with dpg.collapsing_header(label="Parameters",
                                       default_open=True, tag="lh_params") as h:
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
                                       default_open=True, tag="lh_sim") as h:
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

    # ---------------- center panel ----------------------------------

    def _center_panel(self):
        with dpg.child_window(tag="center_child", width=620, border=False):
            with dpg.tab_bar(tag="center_tabs",
                             callback=self._on_center_tab):
                with dpg.tab(label="  Signal Designer  ", tag="tab_designer"):
                    self._designer_tab()
                with dpg.tab(label="  Results  ", tag="tab_results"):
                    self._results_tab()
                with dpg.tab(label="  Analysis  ", tag="tab_analysis"):
                    self._analysis_tab()
                with dpg.tab(label="  STDP  ", tag="tab_stdp"):
                    self._stdp_tab()
                with dpg.tab(label="  Neuro Trainer  ", tag="tab_trainer"):
                    self._trainer_tab()
                with dpg.tab(label="  Polarization  ", tag="tab_polar",
                             show=False):
                    self._polar_tab()
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
            with dpg.subplots(3, 1, link_all_x=True, width=-1, height=-1,
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
        with self._pad(left=8, top=8, bottom=0):
            self._small("spike-timing-dependent plasticity: one pre/post pulse "
                        "pair per point. ANTI-SYMMETRIC - dt > 0 (post after "
                        "pre) potentiates (+dG), dt < 0 depresses (-dG). The "
                        "POSITIVE side follows the paper Fig.4b 3-exp "
                        "(tau=22ms/315ms/19s; the 19s tail holds |dG|~0.7 uS "
                        "out to ~1800 ms). Use OPPOSITE-polarity PRE/POST "
                        "(swap signs to flip LTP side). Window height = A_stdp, "
                        "AMPLITUDE-INDEPENDENT, so use SMALL probes (~20 pA); "
                        "set DT RANGE ~1800 ms to see the full curve.")
            with dpg.group(horizontal=True):
                # anti-symmetric STDP needs OPPOSITE-polarity pulses so each
                # meets the other's surviving 3-exp trace (the A_stdp lock-in);
                # with pre=-/post=+, dt>0 (causal pre->post) potentiates.  SMALL
                # probe amplitude keeps the +/- window symmetric (the lock-in
                # is per-spike, so amplitude doesn't change the window height).
                for tag, cap, default, w in (
                        ("stdp_amp_pre", "PRE AMP", -50.0, 84),
                        ("stdp_amp_post", "POST AMP", 50.0, 84),
                        ("stdp_width_ms", "WIDTH (ms)", 10.0, 84),
                        ("stdp_range_ms", "DT RANGE (+-ms)", 1800.0, 96),
                        ("stdp_settle_ms", "SETTLE (ms)", 1000.0, 90)):
                    with dpg.group():
                        self._caption(cap)
                        dpg.add_input_double(tag=tag, default_value=default,
                                             width=w, format="%.4g", step=0)
                with dpg.group():
                    self._caption("POINTS")
                    dpg.add_input_int(tag="stdp_npts", default_value=80,
                                      width=72, min_value=6, min_clamped=True,
                                      max_value=600, max_clamped=True, step=0)
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
                             "Signal Designer's unit + current/voltage mode). "
                             "Use OPPOSITE signs - the anti-symmetric window "
                             "comes from each pulse meeting the other's "
                             "surviving trace; swap the signs to flip which "
                             "timing side is LTP.\n"
                             "WIDTH: pulse width; the sweep skips any |dt| "
                             "smaller than it.\n"
                             "DT RANGE: sweep dt from -range..+range ms "
                             "(e.g. 200 = -200..+200 ms, 1000 = -1..+1 s).\n"
                             "POINTS: total dt points to simulate (both signs); "
                             "log-spaced. Fewer = faster, more = smoother.\n"
                             "SETTLE: tail after the pair before reading dG, so "
                             "the volatile part has relaxed and you measure the "
                             "RETAINED change locked in by the timing.")
            self._plot_toolbar(("stdp_x",), ("stdp_y",),
                               probe_tag="probe_stdp")
            # axis bounds + legend live in the right-click menu (Axis range /
            # Legend submenus)
            with dpg.plot(height=-1, width=-1, tag="stdp_plot", no_menus=True):
                dpg.add_plot_legend(tag="stdp_plot_leg")
                dpg.add_plot_axis(dpg.mvXAxis,
                                  label="dt = t_post - t_pre (ms)",
                                  tag="stdp_x", no_menus=True)
                dpg.add_plot_axis(dpg.mvYAxis, label="dG (uS)", tag="stdp_y",
                                  no_menus=True)
            # native right-click menu (copy points + Fit/Zoom/Axis/Legend)
            self._add_plot_menu("stdp_plot", "stdp_x", "stdp_y",
                                copy=True, x_unit="ms")

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
        self.device_class = cls
        keys = DEVICE_FAMILIES[cls]
        kind = DEVICE_KIND.get(cls, "current")
        if dpg.does_item_exist("kind_combo"):
            dpg.set_value("kind_combo", kind)
            self._sync_unit_combo()
        spec = next((s for s in MODEL_SPECS if s.key in keys), None)
        if spec and dpg.does_item_exist("param_model_sel"):
            dpg.set_value("param_model_sel", spec.label)
            self.rebuild_param_panel()
        has_polar = any("polarization" in getattr(SPEC_BY_KEY[k].cls,
                                                   "ANALYSES", ())
                        for k in keys if k in SPEC_BY_KEY)
        if dpg.does_item_exist("tab_polar"):
            dpg.configure_item("tab_polar", show=has_polar)
        self._refresh_stdp_labels()
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
        models = [m for m in self._sweep_models()       # expands over the sweep
                  if hasattr(m, "P") or getattr(m, "POLAR_OBS", None)]
        if not models:
            self.log("[polar] select a FeFET model (no polarization observable)")
            self._busy(False, "Polarization needs a FeFET model")
            return
        v_amp = abs(dpg.get_value("polar_vamp"))
        period = max(dpg.get_value("polar_period"), 1e-3)
        n_pts = max(int(dpg.get_value("polar_pts")), 20)
        n_cycles = max(int(dpg.get_value("polar_cycles")), 1)
        self.sim_running = True
        self._busy(True, f"polarization sweep ({n_cycles} cycles)...")
        threading.Thread(target=self._polar_worker,
                         args=(models, v_amp, period, n_pts, n_cycles),
                         daemon=True).start()

    def _polar_worker(self, models, v_amp, period, n_pts, n_cycles):
        try:
            loops = analysis.polarization_loop(models, v_amp=v_amp,
                                               period=period, n_pts=n_pts,
                                               n_cycles=n_cycles)
        except Exception as e:
            loops = {}
            self.q.put(("log", f"[polar] FAILED: {e!r}"))
        self.q.put(("polar", loops))

    def _show_polar(self, loops):
        self.sim_running = False
        for s in self._polar_series:
            if dpg.does_item_exist(s):
                dpg.delete_item(s)
        self._polar_series = []
        if not loops:
            self._busy(False, "polarization: no data")
            return
        unit = "norm."
        for label, d in loops.items():
            unit = d["unit"]
            self._polar_series.append(dpg.add_line_series(
                d["V"], d["P"], parent="polar_y", label=label))
            self.log(f"  [polar] {label}: P {min(d['P']):+.3g}.."
                     f"{max(d['P']):+.3g} {unit}")
        dpg.configure_item("polar_y", label=f"P ({unit})")
        self._fit_axes_of(("polar_x", "polar_y"))
        self._busy(False, f"P-V loop ready - {len(loops)} model(s)")
        dpg.set_value("center_tabs", "tab_polar")

    def _polar_tab(self):
        with self._pad(left=8, top=8, bottom=0):
            self._small("ferroelectric polarization-voltage hysteresis loop "
                        "(FeFET): a triangular gate sweep switches the domains; "
                        "the loop area is the remanent / coercive signature.")
            with dpg.group(horizontal=True):
                for tag, cap, default in (("polar_vamp", "V AMP (V)", 3.0),
                                          ("polar_period", "SWEEP (s)", 0.3),
                                          ("polar_cycles", "CYCLES", 4),
                                          ("polar_pts", "POINTS/CYC", 300)):
                    with dpg.group():
                        self._caption(cap)
                        if tag in ("polar_cycles", "polar_pts"):
                            dpg.add_input_int(tag=tag, default_value=int(default),
                                              width=84, step=0, min_value=1,
                                              min_clamped=True)
                        else:
                            dpg.add_input_double(tag=tag, default_value=default,
                                                 width=84, format="%.4g", step=0)
                with dpg.group():
                    dpg.add_spacer(height=18)
                    b = dpg.add_button(label="Plot P-V loop",
                                       callback=self.on_plot_polar)
                    dpg.bind_item_theme(b, self.themes["primary"])
            self._plot_toolbar(("polar_x",), ("polar_y",), probe_tag="probe_polar")
            with dpg.plot(height=-1, width=-1, tag="polar_plot", no_menus=True):
                dpg.add_plot_legend(tag="polar_plot_leg")
                dpg.add_plot_axis(dpg.mvXAxis, label="gate voltage (V)",
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
        with dpg.child_window(width=RIGHT_W, tag="right_child", border=False):
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
        return PROVIDER_MODELS[self.agent.provider]

    def _rebuild_model_popup(self):
        if not dpg.does_item_exist("model_popup"):
            return
        dpg.delete_item("model_popup", children_only=True)
        self._small("MODEL", color=(126, 150, 220), parent="model_popup")
        for label, mid, desc in self._provider_models():
            dpg.add_selectable(label=f"{label}   ·  {desc}",
                               user_data=(label, mid), parent="model_popup",
                               callback=self._on_model_pick)
        dpg.add_separator(parent="model_popup")
        other = "openai" if self.agent.provider == "claude" else "claude"
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
        self._prune_sweeps()      # drop sweeps on params not in this model

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
            outdir = os.path.join(self.workdir, "results")
            os.makedirs(outdir, exist_ok=True)
            for r in results:
                stem = re.sub(r"[^\w]+", "_", r.label).strip("_")
                p = r.save_csv(os.path.join(outdir, f"gui_{stem}.csv"))
                self.q.put(("log", f"  csv -> {p}"))
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
        width = max(dpg.get_value("stdp_width_ms"), 1e-3) * 1e-3
        tail = max(dpg.get_value("stdp_settle_ms") * 1e-3, 0.02)
        # sweep |dt| from just past the pulse width (a 1% gap so the two pulses
        # never touch) out to the user range. POINTS sets the total count; the
        # spacing is LOGARITHMIC so the 3-exp window (22 ms / 315 ms / 19 s) is
        # sampled evenly per decade - dense near the fast rise, enough out on
        # the slow tail.
        dt_max = abs(dpg.get_value("stdp_range_ms")) * 1e-3
        dt_min = width * 1.01                      # e.g. width 10 ms -> 10.1 ms
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
        dts = [-d for d in reversed(pos)] + pos    # mirror; no overlap region
        self._stdp_ctx = {
            "models": models, "amp_pre": amp_pre, "amp_post": amp_post,
            "width": width, "tail": tail, "unit": unit, "kind": kind,
            "dts": dts,
        }
        if not live:
            self.log(f"[stdp] |dt| starts at {dt_min * 1e3:g} ms "
                     f"(> pulse width {width * 1e3:g} ms): no pulse overlap")
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
                     f"settle {dpg.get_value('stdp_settle_ms'):g} ms")
        threading.Thread(target=self._stdp_worker,
                         args=(models, amp_pre, amp_post, width, dts, tail,
                               live),
                         daemon=True).start()

    def _stdp_worker(self, models, amp_pre, amp_post, width, dts, tail,
                     live=False):
        # thread + queue plumbing only; the measurement is in the hot-reloaded
        # analysis layer (analysis.stdp_sweep), so it can be edited live.
        t_start = time.perf_counter()
        curves = {}
        try:
            curves = analysis.stdp_sweep(
                models, amp_pre, amp_post, width, dts, tail,
                log=None if live else (lambda msg: self.q.put(("log", msg))))
        except Exception as e:
            self.q.put(("log", f"[stdp] FAILED: {e!r}"))
        if not live:
            self.q.put(("log", f"[stdp] sweep done in "
                               f"{time.perf_counter() - t_start:.1f} s"))
        self.q.put(("stdp", [d * 1e3 for d in dts], curves, live))

    def _show_stdp(self, dts_ms, curves, live=False):
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
        slabel, sunit, _ = self._active_stdp_profile()   # dG/uS or dVt/mV
        dpg.configure_item("stdp_y", label=f"{slabel} ({sunit})")
        lines = []
        nan = float("nan")
        for label, ys in curves.items():
            # break the line across the empty overlap gap (no segment joining
            # the -width and +width branches) via a NaN point at dt = 0
            neg = [(d, y) for d, y in zip(dts_ms, ys) if d < 0]
            pos = [(d, y) for d, y in zip(dts_ms, ys) if d > 0]
            line_x = [d for d, _ in neg] + [0.0] + [d for d, _ in pos]
            line_y = [y for _, y in neg] + [nan] + [y for _, y in pos]
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
                     "(R/G/spike transient + dT, dG, dR)")
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
        """Copy the swept STDP points to the clipboard (right-click menu) in
        Spectre/Cadence engineering units - dt as seconds (10.1 ms -> '10.1m')
        and dG as siemens (3.8 uS -> '3.8u').
        kind: 'dt' -> dt values, 'dg' -> dG values per model, 'pairs' -> CSV."""
        dts = self._stdp_dts_ms
        curves = self._stdp_curves
        if not dts or not curves:
            self.log("[stdp] nothing to copy yet - run Plot STDP first")
            return
        dt_row = lambda vals: " ".join(self._eng(v * 1e-3) for v in vals)  # ms->s
        dg_row = lambda vals: " ".join(self._eng(v * 1e-6) for v in vals)  # uS->S
        if kind == "dt":
            text = dt_row(dts)
            what = f"{len(dts)} dt values"
        elif kind == "dg":
            if len(curves) == 1:
                text = dg_row(next(iter(curves.values())))
            else:
                text = "\n".join(f"# {label}\n{dg_row(ys)}"
                                 for label, ys in curves.items())
            what = f"dG values ({len(curves)} model(s))"
        else:  # pairs CSV (dt in s, dG in S, engineering units)
            labels = list(curves)
            head = ["dt"] + [f"dG_{l.split()[0]}" for l in labels]
            out = [",".join(head)]
            for i, d in enumerate(dts):
                out.append(",".join([self._eng(d * 1e-3)]
                                    + [self._eng(curves[l][i] * 1e-6)
                                       for l in labels]))
            text = "\n".join(out)
            what = f"{len(dts)} dt,dG rows"
        dpg.set_clipboard_text(text)
        self.log(f"[stdp] copied {what} to clipboard")
        self._busy(False, f"copied {what} to clipboard")


    # short id -> ImPlot legend location constant / readable label
    _LEGEND_GRID = (
        ("NW", "mvPlot_Location_NorthWest"), ("N", "mvPlot_Location_North"),
        ("NE", "mvPlot_Location_NorthEast"),
        ("W", "mvPlot_Location_West"), ("C", "mvPlot_Location_Center"),
        ("E", "mvPlot_Location_East"),
        ("SW", "mvPlot_Location_SouthWest"), ("S", "mvPlot_Location_South"),
        ("SE", "mvPlot_Location_SouthEast"))
    _LOC_NAME = {"NW": "Top-left", "N": "Top", "NE": "Top-right",
                 "W": "Left", "C": "Center", "E": "Right",
                 "SW": "Bottom-left", "S": "Bottom", "SE": "Bottom-right"}

    def _add_plot_menu(self, plot, xaxis, yaxis, copy=False, x_unit=""):
        """Attach a native right-click context menu to `plot`. A plain WINDOW
        (reliable open/close, no auto-dismiss) of nested dpg.menu submenus that
        cascade on hover and keep the parent open. Opened by _on_plot_rclick,
        closed by _tick_menu_dismiss. copy=True adds the STDP copy-points items.
        Registering many plots gives every plot the same Fit/Zoom/Axis/Legend."""
        menu = plot + "_menu"
        with dpg.window(tag=menu, show=False, no_title_bar=True, no_resize=True,
                        no_move=True, no_collapse=True, no_scrollbar=True,
                        autosize=True):
            if copy:
                self._small("COPY POINTS", color=C_ACC)
                dpg.add_menu_item(label="dt  (delta-T) values", callback=lambda:
                                  self._menu_act(menu, lambda: self._copy_stdp("dt")))
                dpg.add_menu_item(label="dG  values", callback=lambda:
                                  self._menu_act(menu, lambda: self._copy_stdp("dg")))
                dpg.add_menu_item(label="dt, dG pairs  (CSV)", callback=lambda:
                                  self._menu_act(menu, lambda: self._copy_stdp("pairs")))
                dpg.add_separator()
            self._small("VIEW", color=C_ACC)
            dpg.add_menu_item(label="Fit all", callback=lambda: self._menu_act(
                menu, lambda: self._fit_axes_of((xaxis, yaxis))))
            dpg.add_menu_item(label="Fit X", callback=lambda: self._menu_act(
                menu, lambda: self._fit_axes_of((xaxis,))))
            dpg.add_menu_item(label="Fit Y", callback=lambda: self._menu_act(
                menu, lambda: self._fit_axes_of((yaxis,))))
            dpg.add_menu_item(label="Zoom in", callback=lambda: self._menu_act(
                menu, lambda: self._zoom_axes((xaxis, yaxis), 0.7)))
            dpg.add_menu_item(label="Zoom out", callback=lambda: self._menu_act(
                menu, lambda: self._zoom_axes((xaxis, yaxis), 1.45)))
            dpg.add_separator()
            with dpg.menu(label="Axis range"):
                self._build_axis_menu(menu, xaxis, yaxis, x_unit)
            with dpg.menu(label="Legend"):
                self._build_legend_menu(menu)
            dpg.add_separator()
            dpg.add_menu_item(label="Clear A / B probes", callback=lambda:
                              self._menu_act(menu, lambda: self._clear_probes((plot,))))
        if "stdp_menu" in self.themes:
            dpg.bind_item_theme(menu, self.themes["stdp_menu"])
        self._plot_menus.append({"plot": plot, "menu": menu,
                                 "xaxis": xaxis, "yaxis": yaxis})

    def _menu_act(self, menu, fn):
        """Run a one-shot menu action, then close that menu."""
        fn()
        if dpg.does_item_exist(menu):
            dpg.configure_item(menu, show=False)
        if self._open_menu == menu:
            self._open_menu = None

    def _on_plot_rclick(self, sender, app_data):
        """Right-click over any registered plot opens THAT plot's context menu at
        the cursor (ImPlot swallows the plot's own right-click). Only the plot on
        the active tab is hovered, so this naturally targets the right one."""
        for m in self._plot_menus:
            menu, plot = m["menu"], m["plot"]
            if dpg.does_item_exist(menu) and dpg.is_item_hovered(plot):
                self._prefill_axis(menu, m["xaxis"], m["yaxis"])
                mx, my = dpg.get_mouse_pos(local=False)
                vw = dpg.get_viewport_client_width()
                vh = dpg.get_viewport_client_height()
                px = max(0, min(int(mx), vw - 230))
                py = max(0, min(int(my), vh - 340))
                dpg.configure_item(menu, show=True, pos=[px, py])
                dpg.focus_item(menu)
                self._open_menu = menu
                self._menu_unfocused = 0
                return

    @staticmethod
    def _item_visible(tag):
        """is_item_visible that can't raise: a never-rendered item has no
        'visible' state key (is_item_visible would KeyError)."""
        if not dpg.does_item_exist(tag):
            return False
        try:
            return bool(dpg.get_item_state(tag).get("visible"))
        except Exception:
            return False

    def _tick_menu_dismiss(self):
        """Close the open context menu once neither it NOR an open submenu of it
        is focused / hovered (a short grace tolerates focus transitions). A
        submenu is detected by one of its rendered child widgets - the axis-range
        input and the legend checkbox - reporting 'visible'."""
        menu = self._open_menu
        if not menu or not dpg.does_item_exist(menu) \
                or not dpg.is_item_shown(menu):
            return
        submenu_open = (self._item_visible(menu + "_xmin")
                        or self._item_visible(menu + "_leg_show"))
        if (submenu_open or dpg.is_item_focused(menu)
                or dpg.is_item_hovered(menu)):
            self._menu_unfocused = 0
            return
        self._menu_unfocused += 1
        if self._menu_unfocused >= 4:
            dpg.configure_item(menu, show=False)
            self._open_menu = None

    def _prefill_axis(self, menu, xaxis, yaxis):
        """Seed the Axis-range inputs with the plot's CURRENT view, so 'Lock
        view' locks what's on screen (and the user edits from there)."""
        for axis, lo, hi in ((xaxis, "_xmin", "_xmax"), (yaxis, "_ymin", "_ymax")):
            try:
                lim = dpg.get_axis_limits(axis)
            except Exception:
                continue
            if lim and len(lim) == 2 and dpg.does_item_exist(menu + lo):
                dpg.set_value(menu + lo, round(float(lim[0]), 6))
                dpg.set_value(menu + hi, round(float(lim[1]), 6))

    def _build_axis_menu(self, menu, xaxis, yaxis, x_unit=""):
        """'Axis range' submenu: edit a range + Lock view (inputs/buttons don't
        close the submenu, so the user can set X and Y then lock)."""
        self._small("Lock the view to a range; Auto releases it.", color=C_MUTED)
        def _lock(*_):
            self._apply_view(menu, xaxis, yaxis)
        with dpg.group(horizontal=True):
            self._small("X")
            dpg.add_input_float(tag=menu + "_xmin", width=72, step=0,
                                format="%.4g", on_enter=True, callback=_lock)
            self._small("to")
            dpg.add_input_float(tag=menu + "_xmax", width=72, step=0,
                                format="%.4g", on_enter=True, callback=_lock)
            self._small(x_unit)
        with dpg.group(horizontal=True):
            self._small("Y")
            dpg.add_input_float(tag=menu + "_ymin", width=72, step=0,
                                format="%.4g", on_enter=True, callback=_lock)
            self._small("to")
            dpg.add_input_float(tag=menu + "_ymax", width=72, step=0,
                                format="%.4g", on_enter=True, callback=_lock)
            self._small("(0 to 0 = auto Y)", color=C_MUTED)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Lock view", callback=_lock)
            dpg.add_button(label="Auto", callback=lambda *_:
                           self._auto_view(menu, xaxis, yaxis))

    def _apply_view(self, menu, xaxis, yaxis):
        """Lock the plot to the X/Y range in `menu`'s Axis-range inputs."""
        try:
            xlo = float(dpg.get_value(menu + "_xmin"))
            xhi = float(dpg.get_value(menu + "_xmax"))
            ylo = float(dpg.get_value(menu + "_ymin"))
            yhi = float(dpg.get_value(menu + "_ymax"))
        except (TypeError, ValueError):
            return
        if not xhi > xlo:
            self.log("[plot] Axis range: X 'to' must be greater than X 'from'")
            return
        self._zoom_anim.pop(xaxis, None)
        self._zoom_anim.pop(yaxis, None)
        dpg.set_axis_limits(xaxis, xlo, xhi)
        if yhi > ylo:
            dpg.set_axis_limits(yaxis, ylo, yhi)
        else:
            dpg.set_axis_limits_auto(yaxis)

    def _auto_view(self, menu, xaxis, yaxis):
        """Release the lock and fit `menu`'s plot to its data."""
        self._fit_axes_of((xaxis, yaxis))

    def _build_legend_menu(self, menu):
        """'Legend' submenu. Uses checkboxes (not check menu_items) so (a) they
        report a 'visible' state - which is how _tick_menu_dismiss knows this
        submenu is open - and (b) toggling one doesn't close the submenu, so the
        user can flip several at once. Position is a nested anchor submenu."""
        dpg.add_checkbox(label="Show legend", default_value=self.legends_visible,
                         tag=menu + "_leg_show", callback=self._toggle_legends)
        dpg.add_checkbox(label="Outside canvas", default_value=self.legend_outside,
                         tag=menu + "_leg_out", callback=self._toggle_legend_outside)
        dpg.add_checkbox(label="Horizontal layout",
                         default_value=self.legend_horizontal,
                         tag=menu + "_leg_horiz",
                         callback=self._toggle_legend_horizontal)
        with dpg.menu(label="Position"):
            for short, const in self._LEGEND_GRID:
                loc = getattr(dpg, const, 0)
                dpg.add_menu_item(label=self._LOC_NAME[short], user_data=loc,
                                  callback=self._on_legend_pos)

    def _sync_leg_checks(self):
        """Mirror the current legend state into every plot menu's checkboxes."""
        for m in self._plot_menus:
            for suf, val in ((("_leg_show"), self.legends_visible),
                             (("_leg_out"), self.legend_outside),
                             (("_leg_horiz"), self.legend_horizontal)):
                t = m["menu"] + suf
                if dpg.does_item_exist(t):
                    dpg.set_value(t, val)

    def _on_legend_pos(self, sender, app_data, loc):
        """Snap every legend to the picked anchor."""
        self.legend_location = loc
        for t in self.LEGEND_TAGS:
            if dpg.does_item_exist(t):
                dpg.configure_item(t, location=loc)

    def _toggle_legend_horizontal(self, *_):
        self.legend_horizontal = not self.legend_horizontal
        for t in self.LEGEND_TAGS:
            if dpg.does_item_exist(t):
                dpg.configure_item(t, horizontal=self.legend_horizontal)
        self._sync_leg_checks()
        self.log("[plot] legend layout "
                 + ("horizontal" if self.legend_horizontal else "vertical"))

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
            dG = (r.G[-1] - r.G[0]) * 1e6      # retained STDP weight change
            dR = r.R[-1] - r.R[0]
            summary.append(f"{m.name}: dG={dG:+.4g} uS  dR={dR:+.4g} ohm")
        order = ("post after pre -> potentiation" if dt > 0
                 else ("pre after post -> depression" if dt < 0
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
        dpg.configure_item("save_va_dialog", default_filename=name, show=True)

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
            path = os.path.join(self.workdir, f"{cell}.va")
            try:
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
            f"the contract in twins/README.md and the worked example "
            f"twins/example_rram.py (a model class with step(t,dt,drive)/.R/.G/"
            f"reset()/observables(), a params dataclass, and a TWIN_SPEC dict "
            f"with key/label/device_class/input_kind/va_keywords and the "
            f"profile so the right axes/units show). Do NOT edit the GUI "
            f"(vatester/) or core engine (ecfet/) - twins/ only. The GUI "
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
        """Warn (in the log + chat) about ecfet/model_*.py files that define a
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
                mod = importlib.import_module(f"ecfet.{base}")
            except Exception as e:                       # noqa: BLE001
                self.log(f"[models] ecfet/{base}.py failed to import: {e!r}")
                continue
            twin = next((n for n in dir(mod)
                         if isinstance(getattr(mod, n), type)
                         and all(hasattr(getattr(mod, n), a)
                                 for a in ("step", "reset", "R"))), None)
            if twin:
                self.log(f"[models] WARNING: ecfet/{base}.py defines a twin "
                         f"({twin}) but is NOT registered - it will not "
                         f"simulate. Register it (MODEL_SPECS / TWIN_FILE / "
                         f"RELOAD_MODULES + a va_scan keyword) or move it to "
                         f"twins/ with a TWIN_SPEC.")
                self.append_chat("err", f"ecfet/{base}.py looks like a device "
                                 f"twin but isn't registered, so it won't run. "
                                 f"Register it or put it in twins/.")

    def _check_untwinned_va(self):
        """If any .va has no Python twin, prompt to build one (agent -> twins/)."""
        untw = [v for v in self.va_files if not v.model_key]
        if not untw:
            return
        names = tuple(sorted(v.name for v in untw))
        if names == self._untwinned_prompted:        # don't nag repeatedly
            return
        self._untwinned_prompted = names
        if not dpg.does_item_exist("untwin_modal"):
            return
        backend_ok = self.agent.backend != "none"
        dpg.delete_item("untwin_list", children_only=True)
        msg = (f"{len(untw)} Verilog-A file(s) have no Python twin yet, so the "
               f"GUI can't simulate them. Build a twin and the agent writes a "
               f"new model into the twins/ folder (not the app source):")
        if not backend_ok:
            msg += ("\n\n(no agent backend available - set up Claude/OpenAI in "
                    "Account to enable one-click building)")
        dpg.set_value("untwin_msg", msg)
        for v in untw:
            with dpg.group(parent="untwin_list", horizontal=True):
                b = dpg.add_button(label=f"Build twin:  {v.name}",
                                   user_data=v.name,
                                   callback=self._on_build_twin_for,
                                   enabled=backend_ok)
                if backend_ok:
                    dpg.bind_item_theme(b, self.themes["primary"])
        vw = dpg.get_viewport_client_width()
        dpg.configure_item("untwin_modal", show=True,
                           pos=(max(0, (vw - 540) // 2), 150))

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
                    # drop the NaN gap-break point (invalid JSON; agent-facing)
                    pts = [(x, y) for x, y in zip(d[0], d[1]) if y == y]
                    curves[lbl] = {"dt_ms": [round(float(x), 4) for x, _ in pts],
                                   "dG_uS": [round(float(y), 5) for _, y in pts]}
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

        def worker():
            res = self.agent.send(text, ctx, allow_edits=edits,
                                  allow_bash=bash, model=model,
                                  autonomous=auto)
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
                importlib.reload(mod)
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
        changed = any(cur[k] != self._twin_mtimes.get(k) for k in cur)
        # require stability across two polls so we don't read a half-written file
        stable = cur == self._prev_mtimes
        self._prev_mtimes, self._twin_mtimes = self._twin_mtimes, cur
        if changed and stable and not self.sim_running:
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
    # Neuromorphic Trainer studio - a crossbar of device synapses driving
    # spiking (LIF) neurons, trained by device-physics STDP.  A pop-out
    # "canvas" (think Simulink) that visualises the weight updates live.
    # =================================================================

    # viridis-style colour ramp, expanded once to a 256-row [0,1] rgb LUT
    _CMAP_STOPS = [(0.0, (68, 1, 84)), (0.22, (65, 68, 135)),
                   (0.45, (42, 120, 142)), (0.66, (34, 168, 132)),
                   (0.84, (122, 209, 81)), (1.0, (253, 231, 37))]

    def _cmap_lut(self):
        if self._cmap_lut_cache is None:
            xs = np.array([s for s, _ in self._CMAP_STOPS])
            cols = np.array([c for _, c in self._CMAP_STOPS], float) / 255.0
            g = np.linspace(0, 1, 256)
            self._cmap_lut_cache = np.stack(
                [np.interp(g, xs, cols[:, k]) for k in range(3)], axis=1)
        return self._cmap_lut_cache

    def _cmap_color(self, v, alpha=255):
        lut = self._cmap_lut()
        r, g, b = lut[int(min(max(v, 0.0), 1.0) * 255)] * 255.0
        return (int(r), int(g), int(b), alpha)

    # ---- window construction ----------------------------------------

    NT_CHAT_W = 388                 # agent side-panel width

    def _nt_num(self, tag, label, default, width=78, is_int=False, tip=None):
        with dpg.group():
            self._caption(label)
            if is_int:
                dpg.add_input_int(tag=tag, default_value=int(default),
                                  width=width, step=0)
            else:
                dpg.add_input_double(tag=tag, default_value=float(default),
                                     width=width, format="%.4g", step=0)
        if tip:
            with dpg.tooltip(tag):
                dpg.add_text(tip, wrap=320)

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
                b = dpg.add_button(label=" Build network ",
                                   callback=self.on_nt_build)
                dpg.bind_item_theme(b, self.themes["primary"])
                b = dpg.add_button(label=" Train ", tag="nt_btn_train",
                                   callback=self.on_nt_train)
                dpg.bind_item_theme(b, self.themes["primary"])
                b = dpg.add_button(label=" Stop ", callback=self.on_nt_stop)
                dpg.bind_item_theme(b, self.themes["stop"])
                dpg.add_button(label=" Test ", callback=self.on_nt_test)
                dpg.add_button(label=" Reset ", callback=self.on_nt_reset)
                dpg.add_button(label=" Defaults ", callback=self.on_nt_defaults)
                dpg.add_button(label=" Save ",
                               callback=lambda: dpg.show_item("nt_save_dialog"))
                dpg.add_button(label=" Load ",
                               callback=lambda: dpg.show_item("nt_load_dialog"))
                dpg.add_loading_indicator(tag="nt_busy", show=False, radius=2.0,
                                          style=1, color=C_AMBER)
                dpg.add_button(label="● ready", tag="nt_status")
            dpg.add_text("synapse device + all parameters are in the LEFT panel "
                         "- press 'Build network'", tag="nt_devlabel",
                         color=C_TEXT2)
            dpg.add_separator()
            # the visualization fills the center; the controls live in the LEFT
            # panel (swapped in for the device-tester sections on this tab)
            with dpg.child_window(width=-1, border=False, no_scrollbar=True,
                                  tag="nt_canvas_col"):
                with dpg.tab_bar(tag="nt_viz_tabs"):
                    with dpg.tab(label="  Canvas  "):
                        self._nt_canvas_tab()
                    with dpg.tab(label="  Cell  ", tag="nt_tab_cell"):
                        self._nt_cell_tab()
                    with dpg.tab(label="  Weights  "):
                        self._nt_weights_tab()
                    with dpg.tab(label="  Activity  "):
                        self._nt_activity_tab()
                    with dpg.tab(label=" Out spikes "):
                        self._nt_output_tab()
                    with dpg.tab(label=" In spikes "):
                        self._nt_inspikes_tab()
                    with dpg.tab(label=" Dataset "):
                        self._nt_patterns_tab()
                    with dpg.tab(label="  Learning  "):
                        self._nt_learning_tab()
                    with dpg.tab(label=" All weights ", tag="nt_tab_wtrace"):
                        self._nt_wtrace_tab()
                    with dpg.tab(label="  Metrics  "):
                        self._nt_metrics_tab()
                dpg.set_item_callback("nt_viz_tabs", self._on_nt_viz_tab)

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

    def _on_nt_viz_tab(self, *_):
        if (dpg.does_item_exist("nt_viz_tabs")
                and dpg.get_value("nt_viz_tabs") == "nt_tab_wtrace"):
            self._nt_draw_weight_traces()

    def _nt_controls(self):
        with dpg.collapsing_header(label="Network", default_open=True):
            self._caption("SYNAPSE DEVICE")
            dpg.add_combo([s.label for s in MODEL_SPECS],
                          default_value=MODEL_SPECS[1].label, tag="nt_device",
                          width=-1, callback=self._on_nt_device_change)
            with dpg.tooltip("nt_device"):
                dpg.add_text("Which device twin is the synapse at every "
                             "crossbar cross-point (its conductance = the "
                             "weight). ECFET = current-driven, FeFET = "
                             "voltage-driven.", wrap=320)
            with dpg.group(horizontal=True):
                self._nt_num("nt_gh", "GRID H", 5, 60, True,
                             "Input pixel grid height (one spiking input "
                             "neuron per pixel).")
                self._nt_num("nt_gw", "GRID W", 5, 60, True,
                             "Input pixel grid width.")
                self._nt_num("nt_nout", "NEURONS", 4, 64, True,
                             "Number of output LIF neurons (crossbar columns).")
            with dpg.group(horizontal=True):
                with dpg.group():
                    self._caption("HIDDEN LAYERS")
                    dpg.add_input_text(tag="nt_hidden", width=150,
                                       hint="e.g.  8   or   8,4")
                self._nt_num("nt_hidden_gain", "HIDDEN GAIN", 1.7, 78, False,
                             "Extra drive on hidden layers (they have no teacher "
                             "and must fire on their own to pass a code to the "
                             "next crossbar). Raise if hidden layers stay quiet.")
            with dpg.tooltip("nt_hidden"):
                dpg.add_text("Stack MULTIPLE crossbars: list the hidden LIF "
                             "layer sizes between input and output.\n"
                             "  empty -> single crossbar (input -> output)\n"
                             "  8     -> input -> [crossbar] -> 8 -> [crossbar] "
                             "-> output\n"
                             "  8,4   -> two hidden layers, three crossbars.\n"
                             "Each layer pair gets its own device array; every "
                             "crossbar learns LOCALLY (no backprop), so deep "
                             "classification is experimental - single layer is "
                             "best for accuracy.", wrap=330)
            with dpg.group(horizontal=True):
                with dpg.group():
                    self._caption("MODE")
                    dpg.add_combo(["supervised", "unsupervised"],
                                  default_value="supervised", tag="nt_mode",
                                  width=130)
                with dpg.group():
                    self._caption("PATTERNS")
                    dpg.add_combo(self._patset_items(),
                                  default_value="bars", tag="nt_patset",
                                  width=110)
            with dpg.tooltip("nt_mode"):
                dpg.add_text("supervised: a teacher current forces each "
                             "pattern's assigned neuron to fire, so its "
                             "receptive field grows into that pattern.\n"
                             "unsupervised: pure winner-take-all competition - "
                             "neurons self-organise to tile the patterns.",
                             wrap=320)
            with dpg.group(horizontal=True):
                with dpg.group():
                    self._caption("LEARNING RULE")
                    dpg.add_combo(["STDP (device-local)",
                                   "Surrogate grad (BPTT)"],
                                  default_value="STDP (device-local)",
                                  tag="nt_learnrule", width=200,
                                  callback=lambda *_: self._nt_sync_rule())
                self._nt_num("nt_sg_lr", "SG rate", 0.1, 64, False,
                             "Surrogate-gradient learning rate (the BPTT step "
                             "size). 0.05-0.2 is a useful range.")
            with dpg.tooltip("nt_learnrule"):
                dpg.add_text("STDP: local, device-physics plasticity (each "
                             "crossbar learns on its own pre/post activity). "
                             "Best matches real in-situ STDP; deep nets are "
                             "weak.\n"
                             "Surrogate gradient: supervised backprop-through-"
                             "time with a smooth spike surrogate. The gradient "
                             "is applied THROUGH the device (in-situ / hardware-"
                             "aware), so the device nonlinearity stays in the "
                             "loop. Trains deep (multi-layer) nets to high "
                             "accuracy; needs supervised mode.", wrap=330)
            with dpg.group(horizontal=True):
                self._nt_num("nt_epochs", "EPOCHS", 40, 60, True,
                             "Passes over the full pattern set. (A 50 pA gate "
                             "pulse moves G gently, so it wants more passes.)")
                self._nt_num("nt_present", "PRESENT ms", 120, 72, False,
                             "Duration each pattern is shown for.")
                self._nt_num("nt_seed", "SEED", 1, 56, True)
            with dpg.group(horizontal=True):
                self._nt_num("nt_rate", "RATE Hz", 180, 70, False,
                             "Peak Poisson spike rate for a fully-on pixel.")
                self._nt_num("nt_dt", "STEP ms", 1.0, 60, False,
                             "Integration time-step.")
        with dpg.collapsing_header(label="Neuron (LIF + PSP)", default_open=True):
            with dpg.group(horizontal=True):
                self._nt_num("nt_tau_m", "tau_m ms", 20, 66, False,
                             "Membrane time constant - how fast the potential "
                             "leaks back to rest.")
                self._nt_num("nt_vth", "V threshold", 1.0, 70, False,
                             "Fire when the membrane potential crosses this "
                             "(+ the adaptive homeostatic theta).")
                self._nt_num("nt_refrac", "refrac ms", 5, 64, False,
                             "Dead time after a spike.")
            with dpg.group(horizontal=True):
                self._nt_num("nt_epsp", "EPSP gain", 11.0, 70, False,
                             "Excitatory post-synaptic potential weight - how "
                             "strongly an excitatory input spike (scaled by the "
                             "synapse conductance) depolarises the neuron. The "
                             "drive is activity-normalised, so the same value "
                             "works for a 5x5 grid or 28x28 MNIST.")
                self._nt_num("nt_ipsp", "IPSP gain", 1.0, 70, False,
                             "Inhibitory post-synaptic potential depth - "
                             "feed-forward inhibition from inhibitory inputs "
                             "(hyperpolarising).")
                self._nt_num("nt_tausyn", "tau_syn ms", 8, 66, False,
                             "Synaptic (PSP) trace decay.")
            with dpg.group(horizontal=True):
                self._nt_num("nt_inhib", "lateral inhib", 0.9, 78, False,
                             "Winner-take-all lateral inhibition between output "
                             "neurons (competition). 0 disables it.")
                self._nt_num("nt_theta", "theta+ homeo", 0.06, 78, False,
                             "Homeostatic threshold bump per output spike "
                             "(keeps one neuron from hogging every pattern).")
                self._nt_num("nt_teacher", "teacher", 1.4, 60, False,
                             "Supervised teacher drive onto the target neuron.")
        with dpg.collapsing_header(label="Synapse learning (STDP)",
                                   default_open=True):
            self._small("the network rule sets the DIRECTION of each write; the "
                        "device twin sets the ACTUAL dG (its nonlinearity, "
                        "soft bounds, retention).", color=C_MUTED)
            with dpg.group(horizontal=True):
                self._nt_num("nt_potamp", "pot amp", 50, 70, False,
                             "Amplitude of a full potentiating programming "
                             "pulse, in the device's drive unit.")
                self._nt_num("nt_depamp", "dep amp", 50, 70, False,
                             "Amplitude of a full depressing pulse.")
                with dpg.group():
                    self._caption("UNIT")
                    dpg.add_text("pA", tag="nt_amp_unit", color=C_TEXT2)
            with dpg.group(horizontal=True):
                self._nt_num("nt_pwidth", "pulse ms", 10, 64, False,
                             "Programming pulse width.")
                self._nt_num("nt_aplus", "A+ rate", 1.0, 60, False,
                             "Potentiation learning-rate scale.")
                self._nt_num("nt_aminus", "A- rate", 1.0, 60, False,
                             "Heterosynaptic depression rate.")
            with dpg.group(horizontal=True):
                self._nt_num("nt_offset", "LTP/LTD split", 0.25, 80, False,
                             "Pre-trace split: inputs above this are "
                             "potentiated on a post spike, those below are "
                             "depressed. Carves the receptive field.")
                self._nt_num("nt_taupre", "tau_pre ms", 20, 70, False,
                             "Pre-synaptic eligibility-trace decay.")
        with dpg.collapsing_header(label="Spike encoding + noise",
                                   default_open=False):
            self._small("how pixels become spikes, plus front-end / device "
                        "noise for a practical design. All noise defaults to 0 "
                        "(off).", color=C_MUTED)
            with dpg.group(horizontal=True):
                with dpg.group():
                    self._caption("ENCODING")
                    dpg.add_combo(["rate (Poisson)", "latency (TTFS)"],
                                  default_value="rate (Poisson)",
                                  tag="nt_encoding", width=150)
                self._nt_num("nt_bg_rate", "bg rate Hz", 0.0, 70, False,
                             "Spontaneous background firing on EVERY input "
                             "(sensor dark-count / cortical background). Adds "
                             "noise spikes everywhere, even on 'off' pixels.")
            with dpg.tooltip("nt_encoding"):
                dpg.add_text("rate: pixel intensity -> Poisson firing rate "
                             "(0..max rate); information is in the RATE.\n"
                             "latency / time-to-first-spike: brighter pixels "
                             "fire EARLIER - the spike onset ORDER carries the "
                             "pattern (a temporal/spatiotemporal code).",
                             wrap=320)
            with dpg.group(horizontal=True):
                self._nt_num("nt_signal_frac", "signal afferents", 1.0, 90,
                             False,
                             "Fraction of inputs that carry the REAL pattern "
                             "(0..1). The rest are a FIXED random subset that "
                             "fires ONLY the background rate - the classic "
                             "'pattern embedded in noise' paradigm. Set 'bg "
                             "rate' > 0 so the noise afferents actually fire.")
                self._nt_num("nt_jitter", "jitter ms", 0.0, 72, False,
                             "Gaussian temporal jitter (sigma, ms) applied to "
                             "every spike - blurs spike timing. Most meaningful "
                             "with latency encoding / tight patterns.")
            with dpg.group(horizontal=True):
                self._nt_num("nt_input_noise", "input noise", 0.0, 78, False,
                             "Per-presentation sensor noise on the pixels "
                             "(Gaussian + salt-and-pepper), 0..1. Different "
                             "every presentation - trains for robustness.")
                self._nt_num("nt_vnoise", "membrane noise", 0.0, 84, False,
                             "Gaussian noise added to each neuron's membrane "
                             "potential every step (thermal / channel noise). "
                             "Threshold is 1.0, so 0.02-0.08 is a useful range.")
                self._nt_num("nt_wnoise", "write noise", 0.0, 76, False,
                             "Device cycle-to-cycle write noise (relative std "
                             "dev of each programming step; sets the synapse's "
                             "sigma_c2c). Needs a model that has it (ECFET v2).")

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
            self._small("SYNAPSE CELL - how one 3-terminal device reads + writes. "
                        "READ: input spikes ride the wordline through the channel "
                        "(conductance = weight) -> bitline current -> neuron.  "
                        "WRITE: a gate pulse (current for ECFET, voltage for "
                        "FeFET) nudges the channel conductance. Pick a cell on "
                        "the Canvas.", color=C_TEXT2)
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

    def _nt_activity_tab(self):
        with self._pad(left=6, top=4, bottom=0):
            dpg.add_text("present a pattern (Train/Test) to see the dynamics - "
                         "raster: grey = input spikes, orange = neuron spikes",
                         tag="nt_act_title", color=C_TEXT2)
            # linked, auto-filling stack so it scales to the window height
            with dpg.subplots(2, 1, link_all_x=True, width=-1, height=-1,
                              row_ratios=[1.0, 0.85]):
                with dpg.plot(tag="nt_rast_plot", no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="time (ms)",
                                      tag="nt_rast_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="unit (inputs | "
                                      "neurons)", tag="nt_rast_y")
                with dpg.plot(tag="nt_mem_plot", no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="time (ms)",
                                      tag="nt_mem_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="membrane V (a.u.)",
                                      tag="nt_mem_y")

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
            val = ("latency (TTFS)" if str(val).lower().startswith("lat")
                   else "rate (Poisson)")
        elif tag == "nt_mode":
            val = ("unsupervised" if str(val).lower().startswith("uns")
                   else "supervised")
        cur = dpg.get_value(tag)
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
        dpg.set_value(tag, val)
        if tag == "nt_learnrule":
            self._nt_sync_rule()
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
                    if self._nt_set_control(tag, val):
                        applied.append(f"{tag}={dpg.get_value(tag)}")
                self.append_chat("sys", "set " + ", ".join(applied)
                                 if applied else "set: nothing applied")
            elif name == "build":
                self.on_nt_build()
                self.append_chat(
                    "sys", "built · " + dpg.get_value("nt_devlabel")[:140])
            elif name == "train":
                if not self.trainer_running:
                    self._nt_agent_run = True
                    self.on_nt_train()
                    self.append_chat("sys", "training started...")
            elif name == "test":
                if not self.trainer_running:
                    self._nt_agent_run = True
                    self.on_nt_test()

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

    def _nt_inspikes_tab(self):
        with self._pad(left=6, top=4, bottom=0):
            self._small("REAL vs NOISY INPUT SPIKES - the same pattern encoded "
                        "clean (ground truth) and through the practical "
                        "front-end (sensor noise + signal/noise afferent split "
                        "+ background + jitter). For verification / paper "
                        "figures.", color=(126, 150, 220))
            with dpg.group(horizontal=True):
                self._nt_num("nt_is_pidx", "PATTERN #", 0, 70, True,
                             "Which stored pattern to encode (0-based).")
                with dpg.group():
                    dpg.add_spacer(height=16)
                    b = dpg.add_button(label=" Generate ",
                                       callback=self.on_nt_input_spikes)
                    dpg.bind_item_theme(b, self.themes["primary"])
                with dpg.group():
                    dpg.add_spacer(height=16)
                    dpg.add_text("build + Generate to view", tag="nt_is_title",
                                 color=C_TEXT2)
            with dpg.subplots(2, 1, link_all_x=True, width=-1, height=-1,
                              row_ratios=[1.0, 1.0]):
                with dpg.plot(tag="nt_isclean_plot", no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="time (ms)",
                                      tag="nt_isclean_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="afferent  (CLEAN)",
                                      tag="nt_isclean_y")
                with dpg.plot(tag="nt_isnoisy_plot", no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="time (ms)",
                                      tag="nt_isnoisy_x")
                    dpg.add_plot_axis(dpg.mvYAxis,
                                      label="afferent  (NOISY: blue=signal, "
                                      "orange=noise)", tag="nt_isnoisy_y")

    def on_nt_input_spikes(self, *_):
        if self.trainer is None:
            self._nt_status("build a network first", C_RED)
            return
        pats = self.trainer.patterns
        k = min(max(int(self._nt_get("nt_is_pidx", 0)), 0), len(pats) - 1)
        label, vec = pats[k]
        clean, noisy = self.trainer.compare_spikes(vec)
        dt = self.trainer.cfg.dt_ms
        sig = self.trainer.signal_mask
        self._nt_clear("isclean")
        self._nt_clear("isnoisy")
        # clean raster (ground-truth pattern on all afferents)
        tc, ic = np.nonzero(clean)
        s1 = dpg.add_scatter_series((tc * dt).tolist(), ic.tolist(),
                                    label="pattern", parent="nt_isclean_y")
        self._nt_series["isclean"] = [s1]
        # noisy raster, split into signal- vs noise-afferent spikes
        tn, ino = np.nonzero(noisy)
        is_sig = sig[ino] > 0
        ser = []
        if is_sig.any():
            ser.append(dpg.add_scatter_series(
                (tn[is_sig] * dt).tolist(), ino[is_sig].tolist(),
                label="signal", parent="nt_isnoisy_y"))
        if (~is_sig).any():
            ser.append(dpg.add_scatter_series(
                (tn[~is_sig] * dt).tolist(), ino[~is_sig].tolist(),
                label="noise", parent="nt_isnoisy_y"))
        self._nt_series["isnoisy"] = ser
        for ax in ("nt_isclean_x", "nt_isclean_y", "nt_isnoisy_x",
                   "nt_isnoisy_y"):
            dpg.fit_axis_data(ax)
        cfg = self.trainer.cfg
        dpg.set_value("nt_is_title",
                      f"pattern '{label}'  |  clean {int(clean.sum())} spikes "
                      f"-> noisy {int(noisy.sum())}  |  {self.trainer.n_signal}"
                      f"/{self.trainer.n_in} signal afferents  |  jitter "
                      f"{cfg.jitter_ms:g} ms, bg {cfg.bg_rate_hz:g} Hz, "
                      f"input noise {cfg.input_noise:g}")
        self._nt_status(f"input spikes: pattern '{label}'", C_GREEN)

    def _nt_patterns_tab(self):
        with self._pad(left=6, top=4, bottom=0):
            with dpg.group(horizontal=True):
                # --- custom paint editor ---
                with dpg.group():
                    self._small("CUSTOM INPUT PATTERN  (click cells to paint; "
                                "set PATTERNS = custom to train on these)",
                                color=(126, 150, 220))
                    dl = dpg.add_drawlist(width=300, height=300,
                                          tag="nt_pat_editor")
                    dpg.add_draw_layer(parent=dl, tag="nt_pat_layer")
                    with dpg.group(horizontal=True):
                        dpg.add_button(label=" New grid ",
                                       callback=self._nt_paint_new)
                        dpg.add_button(label=" Clear ",
                                       callback=self._nt_paint_clear)
                        b = dpg.add_button(label=" Add pattern ",
                                           callback=self._nt_paint_add)
                        dpg.bind_item_theme(b, self.themes["primary"])
                        dpg.add_button(label=" Drop last ",
                                       callback=self._nt_paint_drop)
                    self._small("left-click / drag paints ON, right-click / "
                                "drag erases; 'Add pattern' stores the grid",
                                color=C_MUTED)
                    dpg.add_text("custom patterns: 0", tag="nt_custom_txt",
                                 color=C_TEXT2)
                # --- dataset loader ---
                with dpg.group():
                    self._small("DATASET INPUT  (MNIST / images)  - set "
                                "PATTERNS = dataset to train on it",
                                color=(126, 150, 220))
                    with dpg.group(horizontal=True):
                        b = dpg.add_button(label=" Download MNIST ",
                                           callback=self.on_nt_download_mnist)
                        dpg.bind_item_theme(b, self.themes["primary"])
                        dpg.add_button(label=" Load file... ",
                                       callback=lambda: dpg.show_item(
                                           "nt_ds_file_dialog"))
                        dpg.add_button(label=" Load folder... ",
                                       callback=lambda: dpg.show_item(
                                           "nt_ds_dir_dialog"))
                    self._small("file: MNIST .idx/.gz, keras .npz, a CSV "
                                "(label,pixels), or one image.  folder: the 4 "
                                "MNIST ubyte files, or images in per-class "
                                "subfolders.", color=C_MUTED)
                    with dpg.group(horizontal=True):
                        self._nt_num("nt_ds_res", "RESOLUTION", 14, 64, True,
                                     "Square grid the images are downsampled to "
                                     "(also sets GRID H/W on build). Smaller = "
                                     "fewer synapses = faster.")
                        self._nt_num("nt_ds_perclass", "IMG/CLASS", 12, 64, True,
                                     "How many images per class to sample for "
                                     "the training set.")
                        with dpg.group():
                            self._caption("INVERT")
                            dpg.add_checkbox(tag="nt_ds_invert",
                                             default_value=False)
                    dpg.add_text("no dataset loaded", tag="nt_ds_txt",
                                 wrap=520, color=C_TEXT2)
                    with dpg.child_window(height=210, border=False,
                                          horizontal_scrollbar=True,
                                          tag="nt_ds_holder"):
                        pass

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

    # ---- custom pattern paint editor --------------------------------

    def _nt_paint_grid_dims(self):
        gh = min(max(int(self._nt_get("nt_gh", 5)), 2), 14)
        gw = min(max(int(self._nt_get("nt_gw", 5)), 2), 14)
        return gh, gw

    def _nt_paint_new(self, *_):
        gh, gw = self._nt_paint_grid_dims()
        if self._nt_custom and (gh, gw) != (self._nt_paint_gh, self._nt_paint_gw):
            self._nt_custom = []            # grid changed -> old ones incompatible
            if dpg.does_item_exist("nt_custom_txt"):
                dpg.set_value("nt_custom_txt", "custom patterns: 0")
        self._nt_paint = np.zeros((gh, gw), np.float32)
        self._nt_paint_gh, self._nt_paint_gw = gh, gw
        self._nt_paint_redraw()

    def _nt_paint_clear(self, *_):
        if self._nt_paint is not None:
            self._nt_paint[:] = 0.0
            self._nt_paint_redraw()

    def _nt_paint_add(self, *_):
        if self._nt_paint is None or self._nt_paint.sum() <= 0:
            self._nt_status("paint some pixels first", C_AMBER)
            return
        n = len(self._nt_custom)
        self._nt_custom.append((f"custom {n}", self._nt_paint.copy()))
        if dpg.does_item_exist("nt_custom_txt"):
            dpg.set_value("nt_custom_txt",
                          f"custom patterns: {len(self._nt_custom)}")
        self._nt_paint_clear()
        self._nt_status(f"added custom pattern ({len(self._nt_custom)} total)",
                        C_GREEN)

    def _nt_paint_drop(self, *_):
        if self._nt_custom:
            self._nt_custom.pop()
            dpg.set_value("nt_custom_txt",
                          f"custom patterns: {len(self._nt_custom)}")

    def _nt_paint_redraw(self):
        layer = "nt_pat_layer"
        if not dpg.does_item_exist(layer) or self._nt_paint is None:
            return
        dpg.delete_item(layer, children_only=True)
        gh, gw = self._nt_paint_gh, self._nt_paint_gw
        W = H = 300
        cw, ch = W / gw, H / gh
        for r in range(gh):
            for c in range(gw):
                v = float(self._nt_paint[r, c])
                col = self._cmap_color(v) if v > 0 else (26, 30, 40, 255)
                dpg.draw_rectangle((c * cw, r * ch), ((c + 1) * cw, (r + 1) * ch),
                                   fill=col, color=(50, 56, 70, 255),
                                   parent=layer)

    def _nt_tick_paint(self):
        """Drag-paint the custom-pattern editor: left held = set, right = erase.
        Driven from the render loop so click-and-drag works smoothly."""
        if (self._nt_paint is None or not dpg.does_item_exist("nt_pat_editor")
                or not dpg.is_item_hovered("nt_pat_editor")):
            return
        left = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        right = dpg.is_mouse_button_down(dpg.mvMouseButton_Right)
        if not (left or right):
            return
        gx, gy = dpg.get_mouse_pos(local=False)
        rmin = dpg.get_item_rect_min("nt_pat_editor")
        lx, ly = gx - rmin[0], gy - rmin[1]
        gh, gw = self._nt_paint_gh, self._nt_paint_gw
        c = int(lx / (300.0 / gw))
        r = int(ly / (300.0 / gh))
        if 0 <= r < gh and 0 <= c < gw:
            nv = 1.0 if left else 0.0
            if self._nt_paint[r, c] != nv:
                self._nt_paint[r, c] = nv
                self._nt_paint_redraw()

    # ---- dataset loading --------------------------------------------

    def on_nt_download_mnist(self, *_):
        if self._nt_busy_load:
            return
        self._nt_busy_load = True
        self._nt_status("downloading MNIST (~11 MB)...", C_AMBER)
        self.log("[neuro] downloading MNIST from the keras mirror...")
        threading.Thread(target=self._nt_download_worker, daemon=True).start()

    def _nt_download_worker(self):
        try:
            from . import datasets as nds
            path = nds.download_mnist(os.path.join(self.workdir, "datasets"))
            self.q.put(("nt_dataset", path, None))
        except Exception as e:                          # noqa: BLE001
            self.q.put(("nt_dataset", None, f"download failed: {e}"))

    def _on_nt_ds_file(self, sender, app_data):
        sel = app_data.get("file_path_name") or ""
        paths = list((app_data.get("selections") or {}).values())
        path = paths[0] if paths else sel
        if path:
            self._nt_load_dataset(path)

    def _on_nt_ds_dir(self, sender, app_data):
        path = app_data.get("file_path_name") or ""
        if path:
            self._nt_load_dataset(path)

    def _nt_load_dataset(self, path):
        if self._nt_busy_load:
            return
        self._nt_busy_load = True
        self._nt_status("loading dataset...", C_AMBER)
        threading.Thread(target=self._nt_load_worker, args=(path,),
                         daemon=True).start()

    def _nt_load_worker(self, path):
        try:
            from . import datasets as nds
            imgs, labels, names = nds.load_any(path)
            n = len(imgs)
            classes = sorted(set((labels if labels is not None
                                  else np.zeros(n, int)).tolist()))
            self.q.put(("nt_dataset_loaded",
                        {"images": imgs, "labels": labels, "names": names,
                         "path": path, "n": n, "classes": classes}, None))
        except Exception as e:                          # noqa: BLE001
            self.q.put(("nt_dataset", None, f"load failed: {e!r}"))

    def _on_nt_dataset(self, path, err):
        self._nt_busy_load = False
        if err:
            self._nt_status(err, C_RED)
            self.log(f"[neuro] {err}")
            return
        if path:                                        # downloaded -> now load
            self._nt_load_dataset(path)

    def _on_nt_dataset_loaded(self, ds, err):
        self._nt_busy_load = False
        if err or not ds:
            self._nt_status(err or "dataset load failed", C_RED)
            return
        self._nt_ds = ds
        nclass = len(ds["classes"])
        dpg.set_value("nt_ds_txt",
                      f"loaded {ds['n']} images, {nclass} classes "
                      f"({os.path.basename(ds['path'])}). Set PATTERNS = "
                      f"dataset and Build.")
        dpg.set_value("nt_patset", "dataset")
        if nclass > 1:
            dpg.set_value("nt_nout", nclass)            # one neuron per class
        self._nt_dataset_preview(ds)
        self._nt_status(f"dataset ready: {ds['n']} imgs / {nclass} classes",
                        C_GREEN)
        self.log(f"[neuro] dataset loaded: {ds['n']} images, {nclass} classes")

    def _nt_dataset_preview(self, ds):
        """Montage of one example image per class into the preview strip."""
        if not dpg.does_item_exist("nt_ds_holder"):
            return
        for tag in ("nt_ds_prev_img", "nt_ds_prev_tex"):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        from . import datasets as nds
        imgs, labels, classes = ds["images"], ds["labels"], ds["classes"]
        labels = (labels if labels is not None
                  else np.zeros(len(imgs), int))
        cell = 40
        cols = min(len(classes), 10)
        lut = self._cmap_lut()
        mont = np.full((cell, cols * cell, 4), 0.08, np.float32)
        mont[..., 3] = 1.0
        for k, c in enumerate(classes[:cols]):
            idx = np.where(np.asarray(labels) == c)[0]
            if not len(idx):
                continue
            g = nds._resize(imgs[idx[0]], cell, cell).astype(np.float32)
            g = g - g.min()
            g = g / (g.max() or 1.0)
            mont[:, k * cell:(k + 1) * cell, :3] = lut[
                np.clip((g * 255).astype(int), 0, 255)]
        dpg.add_raw_texture(cols * cell, cell, mont.reshape(-1),
                            format=dpg.mvFormat_Float_rgba, parent="nt_tex_reg",
                            tag="nt_ds_prev_tex")
        dpg.add_image("nt_ds_prev_tex", tag="nt_ds_prev_img",
                      parent="nt_ds_holder", width=cols * cell * 2,
                      height=cell * 2)

    # ---- pattern-source dispatch (for on_nt_build) ------------------

    def _patset_items(self):
        """PATTERNS dropdown items: built-ins + hot-loaded plugins + the two
        special sources (custom paint / loaded dataset)."""
        return BUILTIN_PATSETS + list(PATTERN_PLUGINS) + ["custom", "dataset"]

    def _nt_watch_patterns(self):
        """Poll the patterns/ folder; when a plugin is added / edited / removed
        (stable for one poll), re-register and refresh the PATTERNS dropdown
        LIVE - so new gates appear with no restart."""
        self._pat_tick = getattr(self, "_pat_tick", 0) + 1
        if self._pat_tick % 30 != 0:              # ~ every 0.5 s at 60 fps
            return
        pdir = os.path.join(self.workdir, "patterns")
        cur = {}
        try:
            for p in glob.glob(os.path.join(pdir, "*.py")):
                if not os.path.basename(p).startswith("_"):
                    cur[p] = os.path.getmtime(p)
        except OSError:
            return
        if self._pat_mtimes is None:
            self._pat_mtimes = self._pat_prev_mtimes = cur
            return
        # reload when the folder has been STABLE for a poll AND differs from the
        # currently-registered set (a new / edited / removed plugin file)
        if cur == self._pat_prev_mtimes and cur != self._pat_mtimes:
            keys, errs = register_pattern_plugins(self.workdir)
            for name, err in errs:
                self.log(f"[patterns] {name}: {err}")
            if dpg.does_item_exist("nt_patset"):
                sel = dpg.get_value("nt_patset")
                items = self._patset_items()
                dpg.configure_item("nt_patset", items=items)
                if sel not in items:              # its file was removed
                    dpg.set_value("nt_patset", "bars")
            self.log(f"[patterns] reloaded - {len(keys)} plugin(s): {keys}")
            self._pat_mtimes = cur
        self._pat_prev_mtimes = cur

    def _nt_build_patterns(self, cfg):
        """Return (patterns, targets) for the selected PATTERNS source, or
        (None, None) to use the built-in bank.  May adjust cfg in place
        (grid + n_out for a dataset)."""
        if self._nt_loaded_patterns is not None:  # restoring a saved model
            pats, tgts = self._nt_loaded_patterns
            self._nt_loaded_patterns = None
            return pats, tgts
        src = cfg.pattern_set
        if src in PATTERN_PLUGINS:               # a patterns/ plugin (hot-loaded)
            pats, targets = PATTERN_PLUGINS[src]["make"](cfg)
            return pats, targets
        if src == "nand":
            # 2-input NAND logic gate -> output neuron 0 ("0") / 1 ("1").
            # COMPLEMENTARY coding: each input becomes a pixel pair [A, ~A] so the
            # afferent grid is [A, ~A, B, ~B] on a 2x2 field.  The softmax/argmax
            # readout has no per-neuron bias, so a shared "always-on" afferent
            # cancels out; the complement pixels instead give each class a wide,
            # bias-free margin (class "0" reads A&B, class "1" reads ~A|~B), and
            # a single crossbar of positive conductances learns it reliably.
            # NAND is 3:1 imbalanced, so the (1,1)->0 minority is oversampled 3x
            # and interleaved with the majority -> a balanced 3-vs-3 train set.
            cfg.grid_h, cfg.grid_w, cfg.n_out = 2, 2, 2
            def _v(a, b):
                return np.array([float(a), 1.0 - a, float(b), 1.0 - b], np.float32)
            table = [((0, 0), 1), ((1, 1), 0), ((0, 1), 1),
                     ((1, 1), 0), ((1, 0), 1), ((1, 1), 0)]
            pats = [(f"A={a} B={b} -> {y}", _v(a, b)) for (a, b), y in table]
            targets = [y for _, y in table]
            return pats, targets
        if src == "xor":
            # 2-input XOR -> output neuron 0 ("0") / 1 ("1"). XOR is NOT linearly
            # separable, so a SINGLE crossbar can never learn it: build with a
            # hidden layer (nt_hidden, e.g. "8") and the Surrogate-grad (BPTT)
            # rule so the gradient flows through the hidden crossbar. Same
            # complementary [A, ~A, B, ~B] 2x2 coding as nand; XOR is already
            # 2-vs-2 balanced, repeated 2x for a fuller train set.
            cfg.grid_h, cfg.grid_w, cfg.n_out = 2, 2, 2
            def _v(a, b):
                return np.array([float(a), 1.0 - a, float(b), 1.0 - b], np.float32)
            table = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0),
                     ((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
            pats = [(f"A={a} B={b} -> {y}", _v(a, b)) for (a, b), y in table]
            targets = [y for _, y in table]
            return pats, targets
        if src == "custom":
            if not self._nt_custom:
                raise ValueError("no custom patterns - paint some in the "
                                 "Inputs/Dataset tab (Add pattern)")
            gh, gw = self._nt_paint_gh or cfg.grid_h, self._nt_paint_gw or cfg.grid_w
            cfg.grid_h, cfg.grid_w = gh, gw
            pats = [(lbl, g.reshape(-1).astype(np.float32))
                    for lbl, g in self._nt_custom]
            targets = [k % cfg.n_out for k in range(len(pats))]
            return pats, targets
        if src == "dataset":
            if not self._nt_ds:
                raise ValueError("no dataset loaded - use Download MNIST / Load "
                                 "in the Inputs/Dataset tab")
            res = min(max(int(self._nt_get("nt_ds_res", 14)), 6), 28)
            per = min(max(int(self._nt_get("nt_ds_perclass", 12)), 2), 60)
            cfg.grid_h = cfg.grid_w = res
            from . import datasets as nds
            tr_p, tr_t, te_p, te_t, names = nds.to_patterns_split(
                self._nt_ds["images"], self._nt_ds["labels"], res, res,
                train_per_class=per, test_per_class=max(4, per // 2),
                seed=cfg.seed,
                invert=bool(self._nt_get("nt_ds_invert", False)))
            cfg.n_out = max(cfg.n_out, len(names))
            self._nt_pending_test = (te_p, te_t, names)   # held-out test set
            return tr_p, tr_t
        return None, None

    # ---- open / status ----------------------------------------------

    def on_open_trainer(self, *_):
        """Switch the main window's center area to the Neuro Trainer tab (and
        swap the left panel to the trainer parameters)."""
        if dpg.does_item_exist("center_tabs") and dpg.does_item_exist(
                "tab_trainer"):
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
            if th is None:
                th = self._mk_theme(f"ntstat_{color}", colors=[
                    ("mvThemeCol_Button", (0, 0, 0, 0)),
                    ("mvThemeCol_ButtonHovered", (0, 0, 0, 0)),
                    ("mvThemeCol_ButtonActive", (0, 0, 0, 0)),
                    ("mvThemeCol_Text", color)])
                self._status_themes[color] = th
            dpg.bind_item_theme("nt_status", th)

    def _nt_busy(self, on):
        if dpg.does_item_exist("nt_busy"):
            dpg.configure_item("nt_busy", show=on)

    def _nt_sync_rule(self):
        """Surrogate gradient is supervised - force the mode + hint it."""
        if self._nt_get("nt_learnrule", "STDP").startswith("Surrogate"):
            if dpg.does_item_exist("nt_mode"):
                dpg.set_value("nt_mode", "supervised")
            self._nt_status("surrogate gradient (BPTT) - supervised, in-situ; "
                            "trains deep nets", C_ACC)

    # ---- parameter reading ------------------------------------------

    def _nt_get(self, tag, default):
        return dpg.get_value(tag) if dpg.does_item_exist(tag) else default

    def _on_nt_device_change(self, *_):
        """When the synapse device's DRIVE KIND changes (current<->voltage),
        switch the programming-pulse amplitude/unit/EPSP to suit it."""
        self._nt_dev = self._nt_device_factory()
        kind = (self._nt_dev or {}).get("kind", "current")
        if kind == getattr(self, "_nt_last_kind", None):
            return
        self._nt_last_kind = kind
        fields = (dict(nt_potamp=2.0, nt_depamp=2.0, nt_pwidth=1.0, nt_epsp=6.0)
                  if kind == "voltage" else
                  dict(nt_potamp=50.0, nt_depamp=50.0, nt_pwidth=10.0,
                       nt_epsp=11.0))
        for tag, val in fields.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, val)
        if dpg.does_item_exist("nt_amp_unit"):
            dpg.set_value("nt_amp_unit", "V" if kind == "voltage" else "pA")

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
                "key": key, "three_terminal": terms >= 3}

    def _nt_read_params(self):
        N = neuro.NeuronParams(
            tau_m=self._nt_get("nt_tau_m", 20.0),
            v_threshold=self._nt_get("nt_vth", 1.0),
            t_refractory=self._nt_get("nt_refrac", 5.0),
            tau_syn=self._nt_get("nt_tausyn", 8.0),
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
            tau_pre=self._nt_get("nt_taupre", 20.0),
            pot_amp=self._nt_get("nt_potamp", 50.0) * ascale,
            dep_amp=self._nt_get("nt_depamp", 50.0) * ascale,
            pulse_width=max(self._nt_get("nt_pwidth", 10.0), 0.01) * 1e-3,
            sg_lr=max(self._nt_get("nt_sg_lr", 0.1), 1e-4))
        mode = self._nt_get("nt_mode", "supervised")
        hidden = tuple(min(max(int(x), 1), 64) for x in
                       re.findall(r"\d+", self._nt_get("nt_hidden", "") or "")[:4])
        cfg = neuro.NetConfig(
            grid_h=min(max(int(self._nt_get("nt_gh", 5)), 2), 14),
            grid_w=min(max(int(self._nt_get("nt_gw", 5)), 2), 14),
            n_out=min(max(int(self._nt_get("nt_nout", 4)), 1), 16),
            hidden_layers=hidden,
            mode="supervised" if mode.startswith("super") else "unsupervised",
            present_ms=max(self._nt_get("nt_present", 120.0), 10.0),
            dt_ms=max(self._nt_get("nt_dt", 1.0), 0.1),
            max_rate_hz=max(self._nt_get("nt_rate", 180.0), 1.0),
            seed=int(self._nt_get("nt_seed", 1)),
            pattern_set=self._nt_get("nt_patset", "bars"),
            learn_rule=("surrogate" if self._nt_get(
                "nt_learnrule", "STDP").startswith("Surrogate") else "stdp"),
            encoding=("latency" if self._nt_get("nt_encoding", "rate")
                      .startswith("lat") else "rate"),
            bg_rate_hz=max(self._nt_get("nt_bg_rate", 0.0), 0.0),
            input_noise=min(max(self._nt_get("nt_input_noise", 0.0), 0.0), 1.0),
            signal_frac=min(max(self._nt_get("nt_signal_frac", 1.0), 0.0), 1.0),
            jitter_ms=max(self._nt_get("nt_jitter", 0.0), 0.0))
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
            dpg.set_value("nt_amp_unit", "V")
        else:                          # ECFET: gate-current programming (pA)
            common.update(nt_epsp=11.0, nt_potamp=50, nt_depamp=50)
            dpg.set_value("nt_amp_unit", "pA")
        for tag, val in common.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, val)
        self._nt_status(f"defaults loaded for {kind}-driven synapse", C_GREEN)

    # ---- save / load a trained model --------------------------------

    def _on_nt_save_file(self, sender, app_data):
        path = (app_data or {}).get("file_path_name")
        if path:
            self.on_nt_save(path)

    def on_nt_save(self, path=None):
        """Save the trained model - network + ALL weights + params + the
        training patterns - to a self-contained JSON that Load fully restores."""
        tr = self.trainer
        if tr is None:
            self._nt_status("build + train first, then Save", C_AMBER)
            return
        if not path:
            path = os.path.join(self.workdir, "results", "models",
                                "neuro_model.json")
        if not path.lower().endswith(".json"):
            path += ".json"
        di = self._nt_dev or {}
        model = {
            "format": "neurovat-model/1",
            "device": {"key": di.get("key"), "label": di.get("label"),
                       "kind": di.get("kind"),
                       "three_terminal": di.get("three_terminal")},
            "layer_sizes": list(tr.layer_sizes),
            "class_names": list(tr.class_names),
            "pot_sign": float(tr.pot_sign),
            "cfg": dataclasses.asdict(tr.cfg),
            "neuron": dataclasses.asdict(tr.N),
            "stdp": dataclasses.asdict(tr.S),
            "controls": {t: dpg.get_value(t) for t, _ in NT_CONTROLS
                         if dpg.does_item_exist(t)},
            "weights_S": [tr.weights_S(c).tolist() for c in range(tr.n_layers)],
            "patterns": [[lbl, np.asarray(v, float).tolist()]
                         for lbl, v in tr.patterns],
            "targets": [int(t) for t in tr.target_of],
        }
        try:                              # restore the exact RNG to resume cleanly
            model["rng_state"] = tr.rng.bit_generator.state
        except Exception:                 # noqa: BLE001
            pass
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
            self._nt_status("stop training before loading", C_AMBER)
            return
        try:
            with open(path, encoding="utf-8") as f:
                model = json.load(f)
        except Exception as e:                          # noqa: BLE001
            self._nt_status(f"load failed: {e}", C_AMBER)
            return
        # 1. restore every control so the rebuild matches the saved setup
        for tag, val in (model.get("controls") or {}).items():
            if dpg.does_item_exist(tag):
                try:
                    dpg.set_value(tag, val)
                except Exception:                       # noqa: BLE001
                    pass
        self._on_nt_device_change()                     # sync amp unit if needed
        # 2. inject the saved patterns so the rebuild is self-contained (no need
        #    for the original dataset / custom paint), then rebuild the network
        pats = [(lbl, np.asarray(v, np.float32))
                for lbl, v in (model.get("patterns") or [])]
        tgts = [int(t) for t in (model.get("targets") or [])]
        self._nt_loaded_patterns = (pats, tgts) if pats else None
        self.on_nt_build()
        self._nt_loaded_patterns = None
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
        if model.get("rng_state"):
            try:
                tr.rng.bit_generator.state = model["rng_state"]
            except Exception:                       # noqa: BLE001
                pass
        # 4. write the saved weights into the rebuilt devices
        W = model.get("weights_S") or []
        ok = 0
        for c in range(min(tr.n_layers, len(W))):
            arr = np.asarray(W[c], float)
            if arr.shape == (tr.layer_sizes[c + 1], tr.layer_sizes[c]):
                tr.set_weights(c, arr)
                ok += 1
        self._apply_nt_snapshot(self._nt_snapshot(0, 0))      # redraw restored
        self._nt_status(
            f"loaded {os.path.basename(path)} - {ok}/{tr.n_layers} crossbars",
            C_GREEN)
        self.log(f"[neuro] model loaded: {path} ({ok} crossbars)")

    # ---- build ------------------------------------------------------

    def on_nt_build(self, *_):
        if self.trainer_running:
            return
        self._nt_dev = self._nt_device_factory()
        if not self._nt_dev:
            self._nt_status("check a model (.va) in the main window first", C_RED)
            self.log("[neuro] no device selected - enable a model first")
            return
        dpg.set_value("nt_amp_unit",
                      "V" if self._nt_dev["kind"] == "voltage" else "pA")
        N, S, cfg = self._nt_read_params()
        try:
            # custom / dataset sources supply explicit patterns (and may resize
            # the grid + n_out); built-in sources return (None, None)
            self._nt_pending_test = None
            patterns, targets = self._nt_build_patterns(cfg)
            self.trainer = neuro.Trainer(self._nt_dev["make"],
                                         self._nt_dev["kind"], N, S, cfg,
                                         patterns=patterns, targets=targets)
        except Exception as e:                          # noqa: BLE001
            self._nt_status(f"build failed: {e}", C_RED)
            self.log(f"[neuro] build failed: {e!r}")
            self.trainer = None
            return
        # held-out test set + class names (datasets only)
        if self._nt_pending_test:
            te_p, te_t, names = self._nt_pending_test
            self.trainer.test_patterns = te_p
            self.trainer.test_targets = te_t
            self.trainer.class_names = names
        # reflect any grid / n_out the source chose back into the controls
        dpg.set_value("nt_gh", cfg.grid_h)
        dpg.set_value("nt_gw", cfg.grid_w)
        dpg.set_value("nt_nout", cfg.n_out)
        self._wevo = {j: [] for j in range(cfg.n_out)}
        self._wevo_epochs = []
        self._nt_acc_hist = {"epoch": [], "train": [], "test": []}
        self._nt_whist = {"epoch": [], "W": []}    # per-synapse weight trajectory
        self._last_present = None
        self._nt_reset_output()
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
                      f"synapse: {self._nt_dev['label']}   |   layers "
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

    def _nt_setup_rf_texture(self, n_out, gh, gw):
        cell = max(8, min(30, 130 // max(gw, 1)))
        gap = 6
        tw = n_out * gw * cell + (n_out - 1) * gap
        th = gh * cell
        if self._rf_tex and dpg.does_item_exist(self._rf_tex):
            dpg.delete_item(self._rf_tex)
        blank = self._blank_rgba(tw, th)
        self._rf_tex = dpg.add_raw_texture(
            tw, th, blank, format=dpg.mvFormat_Float_rgba, parent="nt_tex_reg")
        self._rf_layout = (n_out, gh, gw, cell, gap, tw, th)
        if dpg.does_item_exist("nt_rf_image"):
            dpg.delete_item("nt_rf_image")
        dpg.add_image(self._rf_tex, tag="nt_rf_image", parent="nt_rf_holder",
                      width=tw, height=th)

    @staticmethod
    def _blank_rgba(tw, th):
        img = np.empty((th * tw, 4), np.float32)
        img[:] = (0.08, 0.09, 0.12, 1.0)
        return img.reshape(-1)

    def _nt_montage(self, Wn):
        n_out, gh, gw, cell, gap, tw, th = self._rf_layout
        lut = self._cmap_lut()
        img = np.empty((th, tw, 4), np.float32)
        img[:] = (0.08, 0.09, 0.12, 1.0)
        for j in range(min(n_out, Wn.shape[0])):
            rf = Wn[j].reshape(gh, gw)
            rgb = lut[np.clip((rf * 255).astype(int), 0, 255)]    # (gh,gw,3)
            block = np.repeat(np.repeat(rgb, cell, 0), cell, 1)
            x0 = j * (gw * cell + gap)
            img[:, x0:x0 + gw * cell, :3] = block
        return img.reshape(-1)

    # ---- snapshot (pure data, safe to build off-thread) -------------

    def _nt_snapshot(self, epoch, epochs):
        tr = self.trainer
        K = tr.n_layers
        Wn_all = [tr.weights_norm(c) for c in range(K)]
        W_uS_all = [tr.weights_uS(c) for c in range(K)]
        last = W_uS_all[-1]                 # incoming weights of the output layer
        return {
            "epoch": epoch, "epochs": epochs,
            "Wn0": Wn_all[0], "Wn_all": Wn_all, "W_uS_all": W_uS_all,
            "rf_rgba": self._nt_montage(Wn_all[0]),
            "mean_w": [float(last[j].mean()) for j in range(tr.n_out)],
            "present": self._last_present,
            "spikes": int(tr.spike_count.sum()),
        }

    # ---- snapshot rendering (main thread) ---------------------------

    def _apply_nt_snapshot(self, snap):
        if (self._rf_tex and dpg.does_item_exist(self._rf_tex)
                and snap.get("rf_rgba") is not None):
            dpg.set_value(self._rf_tex, snap["rf_rgba"])
        self._nt_wuS_all = snap["W_uS_all"]            # for the crossbar selector
        c = min(int(self._nt_get("nt_xbar_sel_idx", 0)), len(snap["W_uS_all"]) - 1)
        self._nt_draw_weight_matrix(snap["W_uS_all"][c])
        self._nt_draw_diagram(snap["Wn_all"], snap.get("present"))
        if snap["epoch"] > 0:
            self._wevo_epochs.append(snap["epoch"])
            for j, mw in enumerate(snap["mean_w"]):
                self._wevo.setdefault(j, []).append(mw)
            self._nt_draw_wevo()
            # record this epoch's full weights for the per-synapse cell trace
            self._nt_whist["epoch"].append(snap["epoch"])
            self._nt_whist["W"].append([np.asarray(w).copy()
                                        for w in snap["W_uS_all"]])
            if len(self._nt_whist["epoch"]) > 400:     # bound memory
                self._nt_whist["epoch"].pop(0)
                self._nt_whist["W"].pop(0)
            self._nt_draw_cell_trace()
        if snap.get("present"):
            self._nt_draw_activity(*snap["present"])

    def _nt_clear(self, group):
        for s in self._nt_series.get(group, []):
            if dpg.does_item_exist(s):
                dpg.delete_item(s)
        self._nt_series[group] = []

    def _on_nt_xbar_sel(self, sender, value):
        labels = getattr(self, "_nt_xbar_labels", [])
        idx = labels.index(value) if value in labels else 0
        self._nt_xbar_sel_idx = idx
        wall = getattr(self, "_nt_wuS_all", None)
        if wall:
            self._nt_draw_weight_matrix(wall[min(idx, len(wall) - 1)])

    def _nt_draw_weight_matrix(self, W_uS):
        if not dpg.does_item_exist("nt_wm_y"):
            return
        self._nt_clear("wm")
        n_out, n_in = W_uS.shape
        lo, hi = float(W_uS.min()), float(W_uS.max())
        if hi - lo < 1e-6:
            hi = lo + 1.0
        s = dpg.add_heat_series(
            W_uS.reshape(-1).tolist(), n_out, n_in, scale_min=lo, scale_max=hi,
            bounds_min=(0.0, 0.0), bounds_max=(float(n_in), float(n_out)),
            format="", parent="nt_wm_y")
        self._nt_series["wm"] = [s]
        dpg.configure_item("nt_wm_cbar", min_scale=lo, max_scale=hi)
        dpg.fit_axis_data("nt_wm_x")
        dpg.fit_axis_data("nt_wm_y")

    def _nt_draw_diagram(self, Wn_all, present):
        """Dispatch: one crossbar -> the detailed array schematic; a stack of
        crossbars -> the layered multi-crossbar network."""
        layer = "nt_diag_layer"
        if not dpg.does_item_exist(layer) or self.trainer is None:
            return
        dpg.delete_item(layer, children_only=True)
        if self.trainer.n_layers > 1:
            self._nt_draw_layered(Wn_all, present)
        else:
            self._nt_draw_xbar_array(Wn_all[0], present)

    def _nt_draw_xbar_array(self, Wn, present):
        """A single crossbar: input neurons drive horizontal wordlines that
        cross vertical bitlines; a synapse device sits at every crossing
        (colour = conductance/weight); each bitline is summed by an output
        neuron at the bottom. Input spikes light their wordline, output spikes
        light their neuron, the winner is ringed."""
        layer = "nt_diag_layer"
        self._nt_cell_hits = []                       # clickable device cells
        n_out, gh, gw = self._rf_layout[:3]
        n_in = gh * gw
        Wd, Hd = self._nt_diag_size
        vec = present[1] if present else None
        counts = present[2]["n_out_spikes"] if present else None
        winner = present[2]["winner"] if present else -1
        sig = self.trainer.signal_mask if self.trainer is not None else None
        inh = self.trainer.inh0 if self.trainer is not None else None
        di = self._nt_dev or {}
        kind = di.get("kind", "current")            # current=ECFET, voltage=FeFET
        tt = di.get("three_terminal", True)         # 3-terminal (gate) vs 2-term
        dname = "ECFET" if kind == "current" else ("FeFET" if tt else "memristor")

        # cap the number of wordlines drawn on a big grid (sample evenly)
        N_show = min(n_in, 56)
        rows = (np.linspace(0, n_in - 1, N_show).round().astype(int)
                if n_in > N_show else np.arange(n_in))
        nr = len(rows)
        ow = max(2, len(str(max(n_out - 1, 0))))      # zero-pad widths for IDs
        iw = max(2, len(str(max(n_in - 1, 0))))
        label_inputs = nr <= 30                       # skip on big grids (clutter)

        thumb = float(min(120.0, Hd * 0.30))
        ix = 18 + thumb + 28                       # input-neuron column x
        ax0 = ix + 50                              # array left (first bitline)
        ax1 = Wd - 182                             # array right (last bitline)
        ay0 = 52.0                                 # array top (room for legend)
        ay1 = Hd - 84.0                            # array bottom
        oy = Hd - 42.0                             # output-neuron row y
        row_y = lambda r: ay0 + (r + 0.5) * (ay1 - ay0) / max(nr, 1)
        col_x = lambda j: ax0 + (j + 0.5) * (ax1 - ax0) / max(n_out, 1)
        dev = max(2.5, min((ax1 - ax0) / max(n_out, 1) * 0.20,
                           (ay1 - ay0) / max(nr, 1) * 0.42))
        detail = dev >= 9.0                        # big cells -> 3-terminal view
        pitch_r = (ay1 - ay0) / max(nr, 1)         # row spacing (pre drivers)

        # input pattern thumbnail (the 2-D image driving the wordlines)
        if vec is not None and gw and gh:
            cw, ch = thumb / gw, thumb / gh
            for r in range(gh):
                for c in range(gw):
                    v = float(vec[r * gw + c])
                    g = int(40 + 205 * v)
                    dpg.draw_rectangle((18 + c * cw, 30 + r * ch),
                                       (18 + (c + 1) * cw, 30 + (r + 1) * ch),
                                       fill=(g, g, min(255, g + 25), 255),
                                       color=(40, 46, 60, 110), parent=layer)
        dpg.draw_text((18, 12), "input pattern", size=13,
                      color=(140, 150, 172), parent=layer)

        # wordlines + input neurons (one per shown afferent), coloured by type:
        # EXCITATORY afferents add to the membrane (EPSP +), INHIBITORY ones
        # subtract (IPSP -); ~1/6 of inputs are inhibitory when ipsp_gain > 0
        C_EXC, C_INH = (95, 215, 165), (245, 115, 115)
        C_PRE = (180, 150, 225)                       # PRE program / driver
        prw = min(22.0, 26.0)
        prh = min(10.0, pitch_r * 0.5)
        for r, i in enumerate(rows):
            i = int(i)
            y = row_y(r)
            inten = float(vec[i]) if vec is not None else 0.18
            active = inten > 0.25
            is_inh = inh is not None and i < len(inh) and inh[i] > 0
            ring = (C_INH if is_inh else C_EXC) + (255,)
            g = int(55 + 190 * inten)
            dpg.draw_circle((ix, y), 4.6, fill=(g, g, min(255, g + 30), 255),
                            color=ring, thickness=1.7, parent=layer)
            if label_inputs:                        # neuron id, e.g. x00, x01
                dpg.draw_text((ix - 16 - 6 * iw, y - 6), f"x{i:0{iw}d}",
                              size=10, color=(124, 136, 156), parent=layer)
            if is_inh:                              # inhibitory wordline -> red
                wl = (240, 120, 120, 220) if active else (120, 70, 70, 150)
            else:
                wl = (150, 185, 255, 210) if active else (66, 76, 96, 150)
            dpg.draw_line((ix + 5, y), (col_x(n_out - 1) + dev + 4, y),
                          color=wl, thickness=1.8 if active else 0.8,
                          parent=layer)
            # PRE write DRIVER, SIDE-BY-SIDE with the input neuron (mirrors the
            # post driver): it taps the pre spike and drives THIS row's gate /
            # program line. The wordline READ still rides straight into the array.
            if label_inputs:
                pry = y - (dev * 1.5 if (tt and detail) else 4.0)
                dpg.draw_line((ix + 5, y), (ix + 9, pry), color=C_PRE + (215,),
                              thickness=1.3, parent=layer)      # neuron -> driver
                dpg.draw_rectangle((ix + 9, pry - prh / 2),
                                   (ix + 9 + prw, pry + prh / 2),
                                   fill=(38, 32, 54, 240), color=C_PRE + (235,),
                                   thickness=1.1, rounding=2, parent=layer)
                if tt:                 # 3-terminal: drives a separate GATE line
                    dpg.draw_line((ix + 9 + prw, pry),
                                  (col_x(n_out - 1) + dev, pry),
                                  color=C_PRE + (150,), thickness=1.0, parent=layer)
                else:                  # 2-terminal: the write rides the WORDLINE
                    dpg.draw_line((ix + 9 + prw, pry), (ix + 9 + prw + 8, y),
                                  color=C_PRE + (150,), thickness=1.0, parent=layer)
            # signal-vs-noise (Masquelier) kept as a small orange side dot
            if sig is not None and sig[i] == 0:
                dpg.draw_circle((ix - 9, y), 2.0, fill=(255, 170, 90, 235),
                                parent=layer)

        # bitlines + device cells + output neurons.  When the cells are big
        # enough (small grid or zoomed in) each synapse is drawn as a real
        # 3-TERMINAL device: a source-drain channel (colour = conductance) with
        # a separate GATE (write) terminal tapping a per-row gate/program line.
        cmax = max(counts) if counts else 1
        for j in range(n_out):
            x = col_x(j)
            dpg.draw_line((x, ay0 - 6), (x, oy - 13), color=(82, 94, 122, 170),
                          thickness=1.3, parent=layer)
            for r, i in enumerate(rows):
                w = float(Wn[j][int(i)])
                y = row_y(r)
                sel = self._nt_cell == (0, j, int(i))
                ecol = (255, 220, 120, 255) if sel else (18, 22, 30, 160)
                if detail:
                    bw, bh = dev * 1.25, dev * 0.78
                    # source-drain channel (the stored weight)
                    dpg.draw_rectangle((x - bw, y - bh), (x + bw, y + bh),
                                       fill=self._cmap_color(w, 255), color=ecol,
                                       thickness=2.6 if sel else 1.2,
                                       rounding=2.5, parent=layer)
                    # source contact taps the wordline (left), drain the bitline
                    dpg.draw_circle((x - bw, y), max(1.4, dev * 0.13),
                                    fill=(220, 224, 236, 230), parent=layer)
                    dpg.draw_circle((x, y - bh), max(1.4, dev * 0.13),
                                    fill=(150, 170, 210, 230), parent=layer)
                    if tt:        # 3-terminal only: a GATE stub up to its rail
                        gy = y - dev * 1.5
                        dpg.draw_line((x, y - bh), (x, gy),
                                      color=(170, 150, 200, 200), thickness=1.4,
                                      parent=layer)
                        dpg.draw_line((x - bw * 0.5, gy + 2),
                                      (x + bw * 0.5, gy + 2),
                                      color=(196, 168, 226, 230), thickness=2.0,
                                      parent=layer)
                else:
                    dpg.draw_rectangle((x - dev, y - dev), (x + dev, y + dev),
                                       fill=self._cmap_color(w, 255), color=ecol,
                                       thickness=2.5 if sel else 1.0,
                                       rounding=1.5, parent=layer)
                self._nt_cell_hits.append((0, j, int(i), x, y))
            act = (counts[j] / cmax) if (counts and cmax) else 0.0
            fill = self._cmap_color(0.15 + 0.85 * act, 255)
            oring = (255, 178, 90, 255) if j == winner else (90, 100, 124, 230)
            dpg.draw_circle((x, oy), 12.0, fill=fill, color=oring,
                            thickness=2.5, parent=layer)
            lbl = f"N{j:0{ow}d}" + (f" {counts[j]}sp" if counts else "")
            dpg.draw_text((x - 4 * len(lbl), oy + 15), lbl, size=12,
                          color=(200, 208, 224), parent=layer)

        # ---- DIRECT, LOCAL STDP wiring (NO central driver): every synapse is
        # programmed by ITS OWN pre + post neuron. The PRE runs along its ROW
        # (the purple program rail, fed by input neuron i); the POST feeds back
        # UP its COLUMN (amber, from output neuron Nj). They meet at the
        # cross-point and that coincidence sets that one device's gate pulse. ---
        C_FB = (240, 220, 70)                         # POST feedback (yellow)
        pitch_c = (ax1 - ax0) / max(n_out, 1)
        dw = min(30.0, max(12.0, pitch_c * 0.5))      # driver width (fits pitch)
        for j in range(n_out):
            nx = col_x(j)
            fx = nx + 5                               # post line, just off bitline
            # per-neuron WRITE DRIVER, SIDE-BY-SIDE with the neuron: it only TAPS
            # the post spike. The drain-source READ path (bitline) shorts STRAIGHT
            # into the neuron - it does NOT pass through the driver.
            dx0 = nx + 13
            dpg.draw_line((nx + 12, oy), (dx0, oy), color=C_FB + (225,),
                          thickness=1.6, parent=layer)         # neuron -> driver
            dpg.draw_rectangle((dx0, oy - 8), (dx0 + dw, oy + 8),
                               fill=(50, 46, 16, 242), color=C_FB + (240,),
                               thickness=1.4, rounding=3, parent=layer)
            if dw >= 24:
                dpg.draw_text((dx0 + 4, oy - 7), "drv", size=11,
                              color=C_FB + (255,), parent=layer)
            if tt:        # 3-terminal: drives a separate GATE line up the column
                dpg.draw_line((dx0, oy - 8), (fx, oy - 18), color=C_FB + (230,),
                              thickness=1.5, parent=layer)
                dpg.draw_line((fx, oy - 18), (fx, ay0 - 4), color=C_FB + (185,),
                              thickness=1.4, parent=layer)
                dpg.draw_circle((fx, ay0 - 4), 2.4, fill=C_FB + (240,),
                                parent=layer)
                if detail:                            # tap each gate in the column
                    for r in range(nr):
                        yr = row_y(r) - dev * 1.5
                        dpg.draw_line((nx, yr), (fx, yr), color=C_FB + (175,),
                                      thickness=1.2, parent=layer)
            else:         # 2-terminal: the write rides the BITLINE itself
                dpg.draw_line((dx0, oy - 8), (nx, oy - 16), color=C_FB + (210,),
                              thickness=1.4, parent=layer)
        # ---- top: one-line explainer + a single horizontal legend strip ------
        if tt:
            wq = "current" if kind == "current" else "voltage"
            explain = (f"3-TERMINAL {dname} - GATE {wq} pulse programs each cell "
                       "(read on source-drain); PRE drives the ROW gate, POST "
                       "the COLUMN gate.")
        else:
            explain = ("2-TERMINAL memristor - SET/RESET voltage across the SAME "
                       "word & bit lines (no gate); PRE drives the ROW, POST the "
                       "COLUMN.")
        dpg.draw_text((ix - 30, 6), explain,
                      size=11, color=(206, 196, 150), parent=layer)
        leg = [(C_EXC, "exc"), (C_INH, "inh"), (C_PRE, "PRE drv"),
               (C_FB, "POST drv"), ((240, 120, 120), "WTA")]
        lxp = ix - 30
        for col, name in leg:
            dpg.draw_circle((lxp + 5, 27), 4.2, fill=col + (255,), parent=layer)
            dpg.draw_text((lxp + 13, 20), name, size=12, color=col + (255,),
                          parent=layer)
            lxp += 30 + 7 * len(name)

        # region labels
        dpg.draw_text((ix - 30, ay0 - 16), "input neurons", size=13,
                      color=(140, 150, 172), parent=layer)
        dpg.draw_text((0.5 * (ax0 + ax1) - 70, ay0 - 16),
                      (f"crossbar array  ({'3' if tt else '2'}-terminal "
                       f"{dname} per cross-point)" if detail else
                       f"crossbar array  (zoom in for the {dname} cell view)"),
                      size=13, color=(150, 165, 210), parent=layer)
        dpg.draw_text((col_x(0) - 16, oy + 30), "output neurons", size=13,
                      color=(140, 150, 172), parent=layer)
        if n_in > N_show:
            dpg.draw_text((ix - 30, ay1 + 8),
                          f"(showing {N_show} of {n_in} wordlines)", size=12,
                          color=C_MUTED, parent=layer)
        # lateral inhibition (winner-take-all) between the output neurons
        if (n_out > 1 and self.trainer is not None
                and self.trainer.N.inhibition > 0):
            ybar = ay1 + 10                        # just below the array
            dpg.draw_line((col_x(0), ybar), (col_x(n_out - 1), ybar),
                          color=(240, 120, 120, 170), thickness=1.4, parent=layer)
            for j in range(n_out):                 # drop-ticks toward each neuron
                dpg.draw_line((col_x(j), ybar), (col_x(j), oy - 12),
                              color=(240, 120, 120, 130), thickness=1.2,
                              parent=layer)
            dpg.draw_text((col_x(0) - 4, ybar - 15),
                          "lateral inhibition (WTA)", size=11,
                          color=(245, 140, 140, 230), parent=layer)
        if present and winner >= 0:
            dpg.draw_text((Wd - 150, oy - 6), f"output -> N{winner}", size=15,
                          color=(255, 190, 110, 255), parent=layer)

    def _nt_draw_layered(self, Wn_all, present):
        """A STACK of crossbars: each layer is a column of LIF neurons, joined
        to the next by its own crossbar (weighted connections, colour =
        conductance).  Signal flows input -> crossbar -> hidden -> ... ->
        output.  The detailed conductance array of each crossbar is on the
        Weights tab (use the crossbar selector)."""
        layer = "nt_diag_layer"
        L = self.trainer.layer_sizes
        nL = len(L)
        Wd, Hd = self._nt_diag_size
        vec = present[1] if present else None
        counts = present[2]["n_out_spikes"] if present else None
        winner = present[2]["winner"] if present else -1
        lspk = present[2].get("layer_spikes") if present else None

        padx, ptop, pbot = 64, 40, 56
        NS = 28                                # max nodes shown per layer
        xs = [padx + s * (Wd - 2 * padx) / max(nL - 1, 1) for s in range(nL)]
        pos, shown = [], []
        for sz in L:
            ns = min(sz, NS)
            idx = (np.linspace(0, sz - 1, ns).round().astype(int)
                   if sz > NS else np.arange(sz))
            ys = [ptop + (r + 0.5) * (Hd - ptop - pbot) / max(ns, 1)
                  for r in range(ns)]
            pos.append(list(zip([xs[len(pos)]] * ns, ys)))
            shown.append(idx)

        # crossbar connections (top-K per post neuron, colour = weight)
        for c in range(nL - 1):
            W = Wn_all[c]
            src, dst = shown[c], shown[c + 1]
            kk = min(len(src), 10)
            for dj, j in enumerate(dst):
                row = W[int(j)][src]
                for oi in np.argsort(row)[::-1][:kk]:
                    w = float(row[oi])
                    if w < 0.06:
                        continue
                    dpg.draw_line(pos[c][oi], pos[c + 1][dj],
                                  color=self._cmap_color(w, int(40 + 175 * w)),
                                  thickness=0.5 + 2.0 * w, parent=layer)
            mx = 0.5 * (xs[c] + xs[c + 1])
            dpg.draw_text((mx - 28, ptop - 38), f"crossbar {c}", size=12,
                          color=(150, 165, 210), parent=layer)
            dpg.draw_text((mx - 22, Hd - pbot + 24), f"{L[c]}x{L[c + 1]}",
                          size=12, color=C_MUTED, parent=layer)

        # neuron nodes per layer
        for s in range(nL):
            sz, idx = L[s], shown[s]
            cmax = (max(counts) or 1) if counts else 1
            smax = (max((len(x) for x in lspk[s]), default=1) or 1) \
                if (lspk and s < len(lspk)) else 1
            rad = 5.5 if sz > NS else max(5.0, min(13.0, 95.0 / max(sz, 1)))
            lw = max(2, len(str(max(sz - 1, 0))))     # id zero-pad width
            if s == 0:                                 # input -> x00, x01 ...
                pre, w = "x", lw
            elif s == nL - 1:                          # output -> N00, N01 ...
                pre, w = "N", lw
            else:                                      # hidden: encode the layer
                pre = f"h{s - 1}"                      # 1st hidden h0.., 2nd h1..
                w = max(1, len(str(max(sz - 1, 0))))
            label_nodes = len(idx) <= NS              # skip when too crowded
            for r, i in enumerate(idx):
                i = int(i)
                x, y = pos[s][r]
                if s == 0:
                    a = float(vec[i]) if vec is not None else 0.18
                    g = int(45 + 200 * a)
                    inh = self.trainer.inh0
                    is_inh = inh is not None and i < len(inh) and inh[i] > 0
                    fill = (g, g, min(255, g + 25), 255)
                    ring = (245, 115, 115, 255) if is_inh else (95, 215, 165, 255)
                else:
                    if s == nL - 1 and counts:
                        a = counts[i] / cmax
                    elif lspk and s < len(lspk):
                        a = len(lspk[s][i]) / smax
                    else:
                        a = 0.2
                    fill = self._cmap_color(0.15 + 0.85 * a, 255)
                    ring = ((255, 178, 90, 255) if (s == nL - 1 and i == winner)
                            else (90, 100, 124, 220))
                dpg.draw_circle((x, y), rad, fill=fill, color=ring,
                                thickness=1.6, parent=layer)
                if label_nodes:                     # neuron id (x / h<L> / N)
                    tag = f"{pre}{i:0{w}d}"
                    tx = (x - rad - 4 - 6 * len(tag)) if s == 0 else (x + rad + 4)
                    dpg.draw_text((tx, y - 6), tag, size=10,
                                  color=(150, 160, 182), parent=layer)
            name = ("input" if s == 0 else
                    "output" if s == nL - 1 else f"hidden {s}")
            dpg.draw_text((xs[s] - 16, ptop - 22), name, size=13,
                          color=(140, 150, 172), parent=layer)
            if sz > NS:
                dpg.draw_text((xs[s] - 22, Hd - pbot + 8),
                              f"{sz} (showing {len(idx)})", size=11,
                              color=C_MUTED, parent=layer)
        # E/I legend for the input layer
        lx, ly = 12, 10
        dpg.draw_circle((lx + 5, ly + 5), 4.4, fill=(0, 0, 0, 0),
                        color=(95, 215, 165, 255), thickness=1.6, parent=layer)
        dpg.draw_text((lx + 14, ly - 2), "excitatory in", size=11,
                      color=(95, 215, 165, 255), parent=layer)
        dpg.draw_circle((lx + 5, ly + 22), 4.4, fill=(0, 0, 0, 0),
                        color=(245, 115, 115, 255), thickness=1.6, parent=layer)
        dpg.draw_text((lx + 14, ly + 15), "inhibitory in", size=11,
                      color=(245, 115, 115, 255), parent=layer)
        if present and winner >= 0:
            dpg.draw_text((Wd - 150, Hd - pbot + 24), f"output -> N{winner}",
                          size=14, color=(255, 190, 110, 255), parent=layer)

    # ---- synapse cell inspector (animated read/write) ----------------

    def _nt_pick_cell(self):
        """Select the crossbar cell nearest the click on the Canvas; jump to
        the Cell tab."""
        gx, gy = dpg.get_mouse_pos(local=False)
        rmin = dpg.get_item_rect_min("nt_diagram")
        lx, ly = gx - rmin[0], gy - rmin[1]
        best = None
        for c, j, i, x, y in self._nt_cell_hits:
            d2 = (lx - x) ** 2 + (ly - y) ** 2
            if best is None or d2 < best[0]:
                best = (d2, (c, j, i))
        if best and best[0] < 26 ** 2:
            self._nt_cell = best[1]
            self._nt_anim_t = 0.0
            self._nt_draw_cell_trace()
            if dpg.does_item_exist("nt_viz_tabs"):
                dpg.set_value("nt_viz_tabs", "nt_tab_cell")

    def _nt_tick_cell_anim(self):
        """Advance + redraw the cell-inspector animation while its tab shows."""
        if (self.trainer is None or not dpg.does_item_exist("nt_cell_draw")
                or not dpg.is_item_visible("nt_cell_draw")):
            return
        size = dpg.get_item_rect_size("nt_cell_holder")
        if size and size[0] > 60 and (
                int(size[0]), int(size[1])) != getattr(self, "_nt_cell_sz", None):
            self._nt_cell_sz = (int(size[0]), int(size[1]))
            dpg.configure_item("nt_cell_draw", width=max(360, int(size[0]) - 8),
                               height=max(220, int(size[1]) - 8))
        self._nt_anim_t += 0.02
        self._nt_draw_cell()

    CELL_CYCLE = 5.0                 # seconds: READ (0..0.62) then WRITE (..1)

    def _nt_draw_cell(self):
        layer = "nt_cell_layer"
        if not dpg.does_item_exist(layer) or self.trainer is None:
            return
        tr = self.trainer
        # default to the strongest synapse of neuron 0 if nothing picked
        if (self._nt_cell is None
                or self._nt_cell[0] >= tr.n_layers):
            W0 = tr.weights_norm(0)
            self._nt_cell = (0, 0, int(np.argmax(W0[0])))
        c, j, i = self._nt_cell
        Wn = tr.weights_norm(c)
        g = float(Wn[j][int(i)])
        g_uS = float(tr.weights_uS(c)[j][int(i)])
        kind = (self._nt_dev or {}).get("kind", "current")
        unit = "pA gate current" if kind == "current" else "V gate voltage"
        pre_name = "input pixel" if c == 0 else f"layer-{c} neuron"
        vec = self._last_present[1] if self._last_present else None
        inten = (float(vec[int(i)]) if (vec is not None and c == 0
                 and int(i) < len(vec)) else 0.7)
        pot, strg = tr.last_write.get((c, j, i), (g >= 0.5, 0.0))

        dpg.delete_item(layer, children_only=True)
        sz = getattr(self, "_nt_cell_sz", None)
        W, H = (max(360, sz[0] - 8), max(220, sz[1] - 8)) if sz else (900, 360)
        L = layer

        # ---- PLAYBACK: replay this synapse's REAL per-epoch history (the gate
        # pulses that arrived + the channel conductance changing, start -> now)-
        h = self._nt_whist
        traj = []
        for ep, Wt in zip(h["epoch"], h["W"]):
            if c < len(Wt) and j < Wt[c].shape[0] and int(i) < Wt[c].shape[1]:
                traj.append((int(ep), float(Wt[c][j][int(i)])))
        playback = len(traj) >= 2
        Gmin_uS, Gmax_uS = tr.g_min * 1e6, tr.g_max * 1e6
        spanS = (Gmax_uS - Gmin_uS) or 1.0
        _norm = lambda u: min(max((u - Gmin_uS) / spanS, 0.0), 1.0)
        cur_ep = None
        if playback:
            E = len(traj)
            BEAT = 0.95                          # seconds of animation per epoch
            pos = (self._nt_anim_t % (E * BEAT)) / (E * BEAT)
            fe = pos * E
            e = min(int(fe), E - 1)
            beat_p = fe - e
            cur_ep, G_cur = traj[e]
            G_nxt = traj[e + 1][1] if e + 1 < E else G_cur
            split = 0.55
            reading = beat_p < split
            wfrac = 0.0 if reading else min((beat_p - split) / (1 - split), 1.0)
            g_uS = G_cur + (G_nxt - G_cur) * wfrac        # eased displayed G
            g = _norm(g_uS)
            pot = (G_nxt - G_cur) >= 0
            p = beat_p
        else:
            split = 0.62
            p = (self._nt_anim_t % self.CELL_CYCLE) / self.CELL_CYCLE
            reading = p < split
        read_prog = (p / split) if reading else 0.0
        write_prog = 0.0 if reading else (p - split) / (1 - split)

        # ---- fixed geometry (3-terminal FET: GATE top, SOURCE/DRAIN sides);
        # the top strip is reserved for the gate pulse-train timeline ----------
        ymid = H * 0.54
        ix = W * 0.085                     # input (pre) neuron
        cx = W * 0.45                       # device channel centre
        cw, ch = 92.0, 66.0
        sx = cx - cw / 2 - 22               # SOURCE pin (wordline side)
        dx = cx + cw / 2 + 22               # DRAIN pin (bitline side)
        gate_top = max(ymid - ch / 2 - 64, 86)   # GATE pin (write electrode)
        ox = W * 0.85                       # output neuron
        oy = H * 0.86

        # real numbers: signed gate pulse amplitude, channel R, read current
        amp_si = tr.S.pot_amp if pot else tr.S.dep_amp
        direction = tr.pot_sign if pot else -tr.pot_sign
        gsign = "+" if direction > 0 else "-"
        if kind == "current":
            amp_str = f"{gsign}{abs(amp_si) / 1e-12:.0f} pA"
            gate_unit = "gate current"
        else:
            amp_str = f"{gsign}{abs(amp_si):.2g} V"
            gate_unit = "gate voltage"
        R_ohm = 1.0e6 / max(g_uS, 1e-6)     # uS -> ohm
        R_str = f"{R_ohm/1000:.1f} kohm" if R_ohm >= 1000 else f"{R_ohm:.0f} ohm"
        Vread = 0.1
        I_read = g_uS * Vread               # G(uS)*Vread -> uA

        # phase banner (bottom - the top is the pulse-train timeline)
        dpg.draw_text((W * 0.04, H - 18),
                      ("READ  -  read voltage on SOURCE -> current through the "
                       "channel (set by G) -> DRAIN -> bitline -> neuron"
                       if reading else
                       "WRITE  -  a GATE pulse programs the channel conductance "
                       "(the stored weight)"),
                      size=14, color=((120, 200, 255) if reading
                                      else (255, 190, 110)), parent=L)

        # ---- gate pulse-train timeline (playback): one tick per epoch, up =
        # potentiate / down = depress, height ~ |dG|; a playhead scrubs it -----
        if playback:
            sy = 52.0
            sx0, sx1 = W * 0.07, W * 0.93
            deltas = [traj[k + 1][1] - traj[k][1] for k in range(E - 1)]
            dmax = max((abs(d) for d in deltas), default=1.0) or 1.0
            dpg.draw_text((sx0, 8),
                          "gate pulse history (green up = potentiate, red down "
                          "= depress)", size=12, color=(170, 180, 200), parent=L)
            dpg.draw_text((sx1 - 92, 8), f"epoch {cur_ep} / {traj[-1][0]}",
                          size=13, color=(235, 205, 120), parent=L)
            dpg.draw_line((sx0, sy), (sx1, sy), color=(70, 80, 100, 160),
                          thickness=1.0, parent=L)
            for k, d in enumerate(deltas):
                xk = sx0 + (k + 0.5) * (sx1 - sx0) / max(E - 1, 1)
                hh = 2.0 + 15.0 * min(abs(d) / dmax, 1.0)
                up = d >= 0
                col = (120, 235, 150, 235) if up else (255, 130, 130, 235)
                y2 = sy - hh if up else sy + hh
                dpg.draw_line((xk, sy), (xk, y2), color=col, thickness=2.2,
                              parent=L)
                dpg.draw_circle((xk, y2), 2.0, fill=col, parent=L)
            phx = sx0 + min(e + beat_p, E - 1) * (sx1 - sx0) / max(E - 1, 1)
            dpg.draw_line((phx, sy - 24), (phx, sy + 24),
                          color=(255, 220, 120, 235), thickness=1.6, parent=L)

        # ---- GATE pin (write electrode) - always shows its signed amplitude ----
        gate_on = (not reading)
        gcol = ((120, 235, 150) if pot else (255, 130, 130)) if gate_on \
            else (110, 120, 150)
        dpg.draw_line((cx, gate_top), (cx, ymid - ch / 2), color=gcol,
                      thickness=3.0, parent=L)
        dpg.draw_circle((cx, gate_top), 9, fill=(40, 46, 60, 255), color=gcol,
                        thickness=2.0, parent=L)
        dpg.draw_text((cx + 16, gate_top - 9), "G  (gate)", size=14,
                      color=(180, 190, 210), parent=L)
        dpg.draw_text((cx + 16, gate_top + 11), f"pulse = {amp_str}",
                      size=14, color=gcol if gate_on else (160, 170, 190),
                      parent=L)
        dpg.draw_text((cx + 16, gate_top + 30),
                      f"({gate_unit}; {'POTENTIATE - G up' if pot else 'DEPRESS - G down'})",
                      size=12, color=(150, 160, 180), parent=L)
        if gate_on:                        # the write pulse travelling down
            yp = gate_top + min(write_prog / 0.5, 1.0) * (ymid - ch / 2 - gate_top)
            dpg.draw_circle((cx, yp), 7, fill=gcol, parent=L)

        # ---- channel (the weight): g is already eased through the real
        # trajectory in playback; in non-playback it nudges to show the change --
        gd = g
        if gate_on and not playback:
            gd = min(max(g + (0.12 if pot else -0.12)
                         * max(write_prog - 0.5, 0) * 2, 0), 1)
        dpg.draw_rectangle((cx - cw / 2, ymid - ch / 2), (cx + cw / 2, ymid + ch / 2),
                           fill=self._cmap_color(gd, 255),
                           color=(230, 235, 245, 210), rounding=8, thickness=2.0,
                           parent=L)
        dpg.draw_text((cx - cw / 2 - 6, ymid + ch / 2 + 10),
                      f"channel:  G = {g_uS:.0f} uS    R = {R_str}    "
                      f"I_read = {I_read:.1f} uA  (Vread {Vread:g} V)",
                      size=14, color=(210, 218, 232), parent=L)

        # ---- SOURCE pin (left) + DRAIN pin (right) ----
        for px, lbl, full in ((sx, "S", "source"), (dx, "D", "drain")):
            edge = cx - cw / 2 if px < cx else cx + cw / 2
            dpg.draw_line((edge, ymid), (px, ymid), color=(215, 220, 235, 230),
                          thickness=2.6, parent=L)
            dpg.draw_circle((px, ymid), 5, fill=(215, 220, 235, 255), parent=L)
            dpg.draw_text((px - 4, ymid - 28), lbl, size=15,
                          color=(235, 205, 120), parent=L)
            dpg.draw_text((px - 16, ymid - 46), full, size=11,
                          color=(150, 160, 180), parent=L)

        # ---- input (pre) neuron + wordline to the SOURCE pin ----
        dpg.draw_circle((ix, ymid), 16, fill=(50, 60, 84, 255),
                        color=(120, 160, 255, 255), thickness=2.0, parent=L)
        dpg.draw_text((ix - 28, ymid + 22), f"{pre_name} {i}", size=13,
                      color=(170, 180, 200), parent=L)
        dpg.draw_line((ix + 16, ymid), (sx, ymid),
                      color=(150, 185, 255, 210) if reading else (70, 80, 100, 150),
                      thickness=2.4, parent=L)
        dpg.draw_text(((ix + sx) / 2 - 36, ymid - 24), "wordline (read V)",
                      size=12, color=(130, 150, 190), parent=L)

        # ---- bitline: from the DRAIN pin down to the output neuron ----
        dpg.draw_line((dx, ymid), (dx, oy), color=(82, 94, 122, 190),
                      thickness=1.9, parent=L)
        dpg.draw_line((dx, oy), (ox - 18, oy), color=(82, 94, 122, 190),
                      thickness=1.9, parent=L)
        dpg.draw_text((dx + 8, (ymid + oy) / 2 - 12),
                      f"bitline\nI_read = {I_read:.1f} uA", size=12,
                      color=(130, 150, 190), parent=L)

        # ---- READ animation: dots input -> SOURCE -> channel -> DRAIN ->
        # bitline -> neuron (the read current path) ----
        if reading:
            rp = read_prog
            pts = [(ix + 16, ymid), (sx, ymid), (dx, ymid), (dx, oy),
                   (ox - 18, oy)]
            seglen = [((pts[k + 1][0] - pts[k][0]) ** 2
                       + (pts[k + 1][1] - pts[k][1]) ** 2) ** 0.5
                      for k in range(len(pts) - 1)]
            total = sum(seglen) or 1.0
            ndots = max(2, int(2 + inten * 5))
            for d in range(ndots):
                frac = (rp + d / ndots) % 1.0
                dist = frac * total
                k = 0
                while k < len(seglen) and dist > seglen[k]:
                    dist -= seglen[k]; k += 1
                k = min(k, len(seglen) - 1)
                t = dist / (seglen[k] or 1.0)
                px = pts[k][0] + (pts[k + 1][0] - pts[k][0]) * t
                py = pts[k][1] + (pts[k + 1][1] - pts[k][1]) * t
                # past the DRAIN the read current is scaled by the conductance G
                after = (px > dx - 1)
                a = int(110 + 140 * (g if after else 1.0))
                dpg.draw_circle((px, py), 4.5,
                                fill=(120, 200, 255, a), parent=L)

        # ---- output neuron with a membrane fill bar ----
        counts = (self._last_present[2]["n_out_spikes"]
                  if self._last_present else None)
        # neuron j here is the post of crossbar c; membrane illustrative
        mem = (0.2 + 0.8 * read_prog * g) if reading else 0.15
        fired = reading and mem > 0.85
        nfill = self._cmap_color(0.15 + 0.85 * (1.0 if fired else mem), 255)
        ring = (255, 178, 90, 255) if fired else (90, 110, 140, 230)
        dpg.draw_circle((ox, oy), 22, fill=(36, 42, 56, 255), color=ring,
                        thickness=2.5, parent=L)
        dpg.draw_circle((ox, oy), 22 * (0.35 + 0.6 * mem), fill=nfill, parent=L)
        post = "output neuron" if c == tr.n_layers - 1 else f"hidden neuron"
        lbl = f"{post} {j}"
        if counts and c == tr.n_layers - 1 and j < len(counts):
            lbl += f"  ({counts[j]} sp)"
        dpg.draw_text((ox - 34, oy + 28), lbl, size=13, color=(200, 208, 224),
                      parent=L)
        if fired:
            dpg.draw_text((ox + 26, oy - 8), "spike!", size=14,
                          color=(255, 200, 120, 255), parent=L)

        if dpg.does_item_exist("nt_cell_title"):
            scrub = (f"replaying epoch {cur_ep}/{traj[-1][0]}   |   "
                     if playback else "")
            dpg.set_value("nt_cell_title",
                          f"crossbar {c} · {pre_name} {i} -> neuron {j}   |   "
                          + scrub +
                          f"G = {g_uS:.0f} uS  /  R = {R_str}  "
                          f"({g*100:.0f}% of range)   |   gate pulse "
                          f"{amp_str}  ({'potentiate' if pot else 'depress'})")

    def _nt_draw_cell_trace(self):
        """Transient plot of the selected synapse's conductance (its weight)
        across training epochs - the start-to-end write trajectory."""
        if (not dpg.does_item_exist("nt_cell_py") or self.trainer is None
                or self._nt_cell is None):
            return
        c, j, i = self._nt_cell
        h = self._nt_whist
        ys = []
        for W in h["W"]:
            if (c < len(W) and j < W[c].shape[0] and i < W[c].shape[1]):
                ys.append(float(W[c][j][i]))
        self._nt_clear("celltr")
        if len(ys) >= 1:
            xs = h["epoch"][:len(ys)]
            s = dpg.add_line_series(xs, ys, label=f"input {i} -> N{j}",
                                    parent="nt_cell_py")
            dpg.bind_item_theme(s, self.themes["markers"])
            self._nt_series["celltr"] = [s]
            dpg.fit_axis_data("nt_cell_px")
            dpg.fit_axis_data("nt_cell_py")

    def _nt_trace_theme(self, c, j, n_post):
        """A thin, semi-transparent line theme coloured by the destination
        neuron j (cached per (crossbar, neuron))."""
        key = (c, j)
        th = self._nt_trace_themes.get(key)
        if th is None or not dpg.does_item_exist(th):
            col = self._cmap_color(j / max(n_post - 1, 1), 70)
            with dpg.theme() as th:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, col,
                                        category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1.3,
                                        category=dpg.mvThemeCat_Plots)
            self._nt_trace_themes[key] = th
        return th

    def _nt_draw_weight_traces(self, *_):
        """Plot EVERY synapse's conductance over training in one figure - the
        whole crossbar's learning trajectory, coloured by destination neuron."""
        if not dpg.does_item_exist("nt_wtrace_y") or self.trainer is None:
            return
        self._nt_clear("wtrace")
        h = self._nt_whist
        if not h["epoch"]:
            if dpg.does_item_exist("nt_wtrace_info"):
                dpg.set_value("nt_wtrace_info", "train to populate")
            return
        xs = list(h["epoch"])
        E = len(xs)
        CAP = 700                                     # max lines actually drawn
        cols = []                                     # (c, j, n_post, ys)
        for c in range(self.trainer.n_layers):
            try:
                stack = np.stack([h["W"][e][c] for e in range(E)])   # (E,post,pre)
            except Exception:
                continue
            n_post, n_pre = stack.shape[1], stack.shape[2]
            for j in range(n_post):
                for i in range(n_pre):
                    cols.append((c, j, n_post, stack[:, j, i]))
        n_total = len(cols)
        if n_total > CAP:                             # subsample evenly
            idx = np.linspace(0, n_total - 1, CAP).astype(int)
            cols = [cols[k] for k in idx]
        series = []
        for c, j, n_post, ys in cols:
            s = dpg.add_line_series(xs, ys.tolist(), label="",
                                    parent="nt_wtrace_y")
            dpg.bind_item_theme(s, self._nt_trace_theme(c, j, n_post))
            series.append(s)
        self._nt_series["wtrace"] = series
        dpg.fit_axis_data("nt_wtrace_x")
        dpg.fit_axis_data("nt_wtrace_y")
        nx = self.trainer.n_layers
        info = (f"{n_total} synapses"
                + (f"  (showing {len(cols)} sampled)" if len(cols) < n_total
                   else "")
                + f"  ·  {E} epochs"
                + (f"  ·  {nx} crossbars" if nx > 1 else "")
                + "  ·  colour = destination neuron")
        if dpg.does_item_exist("nt_wtrace_info"):
            dpg.set_value("nt_wtrace_info", info)

    def _nt_draw_wevo(self):
        if not dpg.does_item_exist("nt_wevo_y"):
            return
        self._nt_clear("wevo")
        xs = self._wevo_epochs
        series = []
        for j in sorted(self._wevo):
            ys = self._wevo[j]
            if not ys:
                continue
            series.append(dpg.add_line_series(
                xs[:len(ys)], ys, label=f"neuron {j}", parent="nt_wevo_y"))
        self._nt_series["wevo"] = series
        dpg.fit_axis_data("nt_wevo_x")
        dpg.fit_axis_data("nt_wevo_y")

    def _nt_draw_activity(self, label, vec, res):
        n_in = self.trainer.n_in if self.trainer else 0
        if dpg.does_item_exist("nt_act_title"):
            dpg.set_value("nt_act_title",
                          f"pattern '{label}'  ->  winner neuron "
                          f"{res['winner']}   (output spikes: "
                          f"{res['n_out_spikes']})")
        # raster
        if dpg.does_item_exist("nt_rast_y"):
            self._nt_clear("rast")
            ser = []
            in_sp = res["in_spikes"]
            if in_sp:
                xs = [t for t, _ in in_sp]
                ys = [i for _, i in in_sp]
                ser.append(dpg.add_scatter_series(xs, ys, label="inputs",
                                                  parent="nt_rast_y"))
            xs2, ys2 = [], []
            for j, sp in enumerate(res["out_spikes"]):
                xs2 += list(sp)
                ys2 += [n_in + j] * len(sp)
            if xs2:
                ser.append(dpg.add_scatter_series(
                    xs2, ys2, label="neuron spikes", parent="nt_rast_y"))
            self._nt_series["rast"] = ser
            dpg.fit_axis_data("nt_rast_x")
            dpg.fit_axis_data("nt_rast_y")
        # membrane
        if dpg.does_item_exist("nt_mem_y"):
            self._nt_clear("mem")
            v = res["v_trace"]
            dt = res["dt_ms"]
            tarr = [k * dt for k in range(res["n_steps"])]
            ser = []
            for j in range(v.shape[1]):
                ser.append(dpg.add_line_series(
                    tarr, v[:, j].tolist(), label=f"N{j}", parent="nt_mem_y"))
            vth = self.trainer.N.v_threshold if self.trainer else 1.0
            ser.append(dpg.add_line_series(
                [tarr[0], tarr[-1]] if tarr else [0, 1], [vth, vth],
                label="threshold", parent="nt_mem_y"))
            self._nt_series["mem"] = ser
            dpg.fit_axis_data("nt_mem_x")
            dpg.fit_axis_data("nt_mem_y")

    # ---- train / test / stop / reset --------------------------------

    def on_nt_train(self, *_):
        if self.trainer_running:
            return
        if self.trainer is None:
            self.on_nt_build()
            if self.trainer is None:
                return
        epochs = max(1, int(self._nt_get("nt_epochs", 20)))
        self.trainer_running = True
        self._trainer_stop = False
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
        try:
            for e in range(epochs):
                if self._trainer_stop:
                    break
                for k in rng.permutation(n):
                    if self._trainer_stop:
                        break
                    k = int(k)
                    # surrogate is supervised; STDP uses targets only in
                    # supervised mode
                    tgt = (tr.target_of[k] if (tr.cfg.mode == "supervised"
                           or tr.cfg.learn_rule == "surrogate") else None)
                    res = tr.train_step(pats[k][1], tgt)
                    self._last_present = (pats[k][0], pats[k][1], res)
                    self.q.put(("nt_spk", res["n_out_spikes"], res["winner"]))
                done = e + 1
                self.q.put(("nt_snap", self._nt_snapshot(done, epochs)))
                # accuracy curve: cheap sampled eval each epoch (train + test)
                tr_acc = tr.eval_accuracy(use_test=False, max_n=40)
                te_acc = (tr.eval_accuracy(use_test=True, max_n=40)
                          if tr.test_patterns else None)
                self.q.put(("nt_acc", done, tr_acc, te_acc))
        except Exception as ex:                          # noqa: BLE001
            self.q.put(("log", f"[neuro] training error: {ex!r}"))
        self.q.put(("nt_done", {"epochs": done, "stopped": self._trainer_stop}))

    def on_nt_stop(self, *_):
        if self.trainer_running:
            self._trainer_stop = True
            self._nt_status("stopping...", C_AMBER)

    def on_nt_reset(self, *_):
        if self.trainer_running:
            return
        self.on_nt_build()

    def on_nt_test(self, *_):
        if self.trainer is None or self.trainer_running:
            if self.trainer is None:
                self._nt_status("build + train a network first", C_RED)
            return
        self.trainer_running = True
        self._nt_busy(True)
        self._nt_status("evaluating...", C_AMBER)
        threading.Thread(target=self._nt_test_worker, daemon=True).start()

    def _nt_test_worker(self):
        tr = self.trainer
        try:
            tr_yt, tr_yp = tr.evaluate(use_test=False)
            has_test = bool(tr.test_patterns)
            te_yt, te_yp = (tr.evaluate(use_test=True) if has_test else ([], []))
            res = tr.infer(tr.patterns[0][1])
            self._last_present = (tr.patterns[0][0] + " (test)",
                                  tr.patterns[0][1], res)
            snap = self._nt_snapshot(len(self._wevo_epochs), 0)
            self.q.put(("nt_metrics", {
                "train": (tr_yt, tr_yp), "test": (te_yt, te_yp),
                "n": tr.n_out, "names": list(tr.class_names),
                "mode": tr.cfg.mode}, snap))
        except Exception as ex:                          # noqa: BLE001
            self.q.put(("log", f"[neuro] eval error: {ex!r}"))
            self.q.put(("nt_metrics", None, None))

    def _on_nt_done(self, info):
        self.trainer_running = False
        self._nt_busy(False)
        msg = (f"stopped at epoch {info['epochs']}" if info["stopped"]
               else f"trained {info['epochs']} epochs")
        self._nt_status(msg, C_GREEN)
        self.log(f"[neuro] {msg}")
        self._nt_draw_weight_traces()        # refresh the all-synapse trajectory
        # agent-driven run: evaluate, then the metrics get fed back to the agent
        if self._nt_agent_run and not info["stopped"]:
            self.on_nt_test()

    # ---- accuracy curve (streamed during training) ------------------

    def _on_nt_acc(self, epoch, train_acc, test_acc):
        h = self._nt_acc_hist
        h["epoch"].append(epoch)
        h["train"].append(100.0 * train_acc)
        h["test"].append(100.0 * test_acc if test_acc is not None else None)
        if not dpg.does_item_exist("nt_acc_y"):
            return
        self._nt_clear("acc")
        ser = [dpg.add_line_series(h["epoch"], h["train"], label="train",
                                   parent="nt_acc_y")]
        if any(v is not None for v in h["test"]):
            te = [v if v is not None else float("nan") for v in h["test"]]
            ser.append(dpg.add_line_series(h["epoch"], te, label="test",
                                           parent="nt_acc_y"))
        self._nt_series["acc"] = ser
        dpg.fit_axis_data("nt_acc_x")
        dpg.fit_axis_data("nt_acc_y")

    # ---- metrics: confusion matrix + per-class P/R/F1 ----------------

    def _on_nt_metrics(self, data, snap):
        self.trainer_running = False
        self._nt_busy(False)
        if snap is not None:
            self._apply_nt_snapshot(snap)
        if not data:
            self._nt_status("evaluation failed", C_RED)
            return
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
        # close the agent loop: feed the fresh metrics back to the MAIN chat
        if self._nt_agent_run:
            self._nt_agent_run = False
            if self._nt_agent_rounds < 5 and not self.chat_busy:
                self._nt_agent_rounds += 1
                msg = ("The training + evaluation you started just finished. "
                       "Metrics:\n" + self._nt_metrics_text() +
                       "\n\nDiagnose what worked and what failed. If it can be "
                       "improved, change one or two controls and train again "
                       "(emit the nt_action blocks). If it is already good, say "
                       "so and stop.")
                self.append_chat("sys", f"run #{self._nt_agent_rounds} "
                                 "finished — handing the metrics to the agent")
                self.append_chat("you", "[auto] analyse the training result")
                self._main_send(msg)
            else:
                self.append_chat("sys", "auto-tune loop paused (cap reached). "
                                 "Ask the agent to continue if you want more.")

    def _nt_draw_confusion(self, cm, names):
        if not dpg.does_item_exist("nt_cm_y"):
            return
        if self._nt_cm_series and dpg.does_item_exist(self._nt_cm_series):
            dpg.delete_item(self._nt_cm_series)
        n = cm.shape[0]
        hi = float(cm.max()) or 1.0
        # row-major, but flip rows so true-class 0 sits at the TOP
        vals = cm[::-1].reshape(-1).astype(float).tolist()
        self._nt_cm_series = dpg.add_heat_series(
            vals, n, n, scale_min=0.0, scale_max=hi,
            bounds_min=(0.0, 0.0), bounds_max=(float(n), float(n)),
            format="%.0f", parent="nt_cm_y")
        ticks = tuple((str(names[i]) if i < len(names) else str(i), i + 0.5)
                      for i in range(n))
        dpg.set_axis_ticks("nt_cm_x", ticks)
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
                    self._show_stdp(item[1], item[2],
                                    live=(len(item) > 3 and item[3]))
                elif kind == "polar":
                    self._show_polar(item[1])
                elif kind == "chat":
                    self._on_chat_done(item[1], item[2])
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

    def run(self, smoke_frames=0):
        self.build()
        frame = 0
        while dpg.is_dearpygui_running():
            self._process_queue()
            self._tick_zoom_anim()
            self._tick_probe_drag()
            self._hide_tip_if_stale()
            self._tick_menu_dismiss()
            self._nt_tick_layout()
            self._nt_tick_paint()
            self._nt_tick_cell_anim()
            self._watch_code()
            self._nt_watch_patterns()
            dpg.render_dearpygui_frame()
            frame += 1
            if smoke_frames and frame >= smoke_frames:
                break
        self.virtuoso.disconnect()
        dpg.destroy_context()
        if self._restart:               # relaunch to apply GUI-code edits
            os.execv(sys.executable, [sys.executable] + sys.argv)


def main(workdir=None, smoke_frames=0):
    App(workdir or os.getcwd()).run(smoke_frames=smoke_frames)
