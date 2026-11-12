"""Trainer mode 'unsupervised stream': continuous-stream STDP (Masquelier et
al. 2008/2009). Selecting the mode swaps the Trainer's left-panel controls and
centre views for the ones built here and in ``network_stream_views``; Build /
Train / Test / Stop act on it.

One uninterrupted spike stream instead of epochs: hidden patterns recur in
background activity, STDP acts at every spike, and lateral inhibition lets
each neuron take one pattern. Two inputs: the Poisson patterns of the papers
(50 ms sections of frozen background on a set of afferents), or scanned
images (spatio-temporal: one afferent per pixel row, one time bin per pixel
column; the section is columns x bin ms) shown between background noise or
distractor images. Train runs with STDP on; Test freezes the weights and
plays a fresh stream around the same patterns. The network is its own (the
pixel-grid crossbar is too small for it) but uses the selected synapse
device and power profile.

App supplies widget helpers, the device factory and the event queue. The
worker only posts queue messages (``nt_stream`` learning points,
``nt_stream_view`` raster snapshots, ``log``) and always ends with
``nt_stream_done``.
"""
import math
import threading
import time

import dearpygui.dearpygui as dpg
import numpy as np

from . import neuro
from . import stream_view as sv
from .network_stream_views import NetworkStreamViewsMixin
from .ui_constants import C_AMBER, C_GREEN, C_MUTED, C_RED, C_TEXT2

IDEAL, DEVICE = "ideal weights", "selected device"
STREAM_MODE = "unsupervised stream"    # value of the nt_mode combo
CALIBRATED, AS_SET = "calibrated", "as set"
CHUNK_S = 2.0                       # live-plot resolution
VIEW_HZ = 12.0                      # live raster snapshots per wall second

# Learning-rule, neuron and inhibition constants of the stream network:
# (tag, label, MasquelierParams field, default, min, max, meaning). The
# defaults are the values of Masquelier et al. 2008 that the benchmark is
# validated with; the two inhibition values are choices made here.
RULE = [
    ("nt_st_aplus", "A+ (LTP step)", "a_plus", 0.03125, 1e-5, 0.5,
     "weight gained by a synapse whose spike just preceded the output spike"),
    ("nt_st_aratio", "A- / A+", "a_minus_ratio", 0.85, 0.0, 5.0,
     "depression step as a multiple of A+"),
    ("nt_st_tplus", "tau+ ms", "tau_plus", 16.8, 0.5, 200.0,
     "LTP window time constant"),
    ("nt_st_tminus", "tau- ms", "tau_minus", 33.7, 0.5, 200.0,
     "LTD window time constant"),
    ("nt_st_w0", "Start weight", "w_init", 0.475, 0.0, 1.0,
     "initial weight of every synapse, 0-1 of the range"),
    ("nt_st_taum", "tau m ms", "tau_m", 10.0, 0.5, 200.0,
     "membrane time constant"),
    ("nt_st_taus", "tau s ms", "tau_s", 2.5, 0.1, 100.0,
     "synapse time constant (must stay below tau m)"),
    ("nt_st_refr", "Refractory ms", "t_ref_ms", 1.0, 0.0, 50.0,
     "time after a spike in which the neuron cannot fire"),
    ("nt_st_inh", "Inhibition", "inh_alpha", 0.4, 0.0, 5.0,
     "a neuron that fires holds the others back by this x threshold"),
    ("nt_st_tinh", "tau inh ms", "tau_inh", 20.0, 0.5, 500.0,
     "decay of that inhibition"),
]

# (tag, one-line meaning) - added to the controls the agent / bridge may set
STREAM_CONTROLS = [
    ("nt_st_input", f"stream: input, {sv.POISSON} | {sv.IMAGES}"),
    ("nt_st_aff", "stream (Poisson input): afferents (100-4000)"),
    ("nt_st_neurons", "stream: output neurons (1-8)"),
    ("nt_st_patterns", "stream (Poisson input): hidden patterns (1-8)"),
    ("nt_st_in_pattern", "stream (Poisson input): fraction of afferents in "
                         "each pattern"),
    ("nt_st_imgset", "stream (image input): digits | letters | bars | dataset "
                     "(one image per class of the loaded dataset)"),
    ("nt_st_img_size", "stream (image input): image rows = columns = "
                       "afferents (5-28)"),
    ("nt_st_img_pick", "stream (image input): indices of the hidden images, "
                       "e.g. '0' or '0,3'"),
    ("nt_st_img_distract", "stream (image input): indices of distractor images "
                           "shown between presentations ('' = none, 'all')"),
    ("nt_st_bin", "stream (image input): ms per pixel column"),
    ("nt_st_ink", "stream (image input): ink threshold 0-1"),
    ("nt_st_bg", "stream (image input): background Hz between presentations "
                 "(unused with distractors)"),
    ("nt_st_seconds", "stream: seconds simulated per Train / Test"),
    ("nt_st_share", "stream: share of sections holding a pattern 0-0.5"),
    ("nt_st_spacing", "stream: random | even (every k-th section)"),
    ("nt_st_jitter", "stream: spike jitter ms"),
    ("nt_st_noise", "stream: extra noise spikes Hz on every afferent, inside "
                    "the patterns too (switching input sets 10 Poisson / 0 "
                    "images if untouched)"),
    ("nt_st_delete", "stream: fraction of pattern spikes deleted 0-1"),
    ("nt_st_thr", "stream: neuron threshold (0 = auto: 0.25 x afferents for "
                  "Poisson, 0.85 x weakest clean image peak for images)"),
    ("nt_st_synapse", f"stream: {IDEAL} | {DEVICE}"),
    ("nt_st_window", "stream: device write window ms (one pulse per synapse per window)"),
    ("nt_st_pulse", f"stream: device pulse amplitude, {CALIBRATED} (one pulse = "
                    f"one STDP step) | {AS_SET} (nt_potamp / nt_depamp)"),
    ("nt_st_tabs", "stream view tab: nt_st_tab_raster | nt_st_tab_net | "
                   "nt_st_tab_pat | nt_st_tab_learn | nt_st_tab_weights | "
                   "nt_st_tab_energy"),
    ("nt_st_view_win", "stream raster: seconds shown (0 = auto)"),
    ("nt_st_view_t", "stream raster: replay position, s into the last run"),
    ("nt_st_speed", "stream raster: replay speed 0.25x | 1x | 4x"),
    ("nt_st_play", "stream raster: 1 = play the replay, 0 = pause"),
] + [(tag, f"stream: {tip} ({lo:g}-{hi:g}, default {default:g})")
     for tag, _label, _field, default, lo, hi, tip in RULE]

POISSON_TAGS = ("nt_st_aff", "nt_st_patterns", "nt_st_in_pattern")
IMAGE_TAGS = ("nt_st_imgset", "nt_st_img_size", "nt_st_img_pick",
              "nt_st_img_distract", "nt_st_bin", "nt_st_ink", "nt_st_bg")
# controls whose change redraws the Patterns / Network previews
PREVIEW_TAGS = set(POISSON_TAGS + IMAGE_TAGS) | {
    "nt_st_input", "nt_st_neurons", "nt_st_thr", "nt_st_synapse", "nt_st_share"
} | {r[0] for r in RULE}


def _clean(x):
    """JSON-safe copy: numpy -> python, NaN -> None, dict keys -> str."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(float(x)) else float(x)
    return x


class NetworkStreamMixin(NetworkStreamViewsMixin):
    _nt_stream_session = None       # neurovat.masquelier.Session
    _nt_stream_key = None           # settings the session was built for
    _nt_stream_last = None          # JSON-safe result of the last run
    _nt_stream_error = None
    _nt_stream_agent = False        # the agent started the run in flight
    _nt_stream_rounds = 0           # agent-started runs since the last pause
    _nt_mode_shown = "supervised"   # the mode the window currently shows
    _nt_devlabel_grid = None        # the grid network's label, while hidden
    _nt_st_input_shown = sv.POISSON

    # ---- layout -----------------------------------------------------

    def _nt_is_stream(self):
        return self._nt_get("nt_mode", "") == STREAM_MODE

    def _nt_st_is_images(self):
        return sv.input_kind(self._nt_get("nt_st_input", sv.POISSON)) == sv.IMAGES

    def _nt_stream_left(self):
        """Left-panel controls of the stream mode (inside ``nt_stream_group``)."""
        self._small("One continuous spike stream, no epochs and no labels: "
                    "hidden patterns recur in background activity. "
                    "Train = STDP on, continuing from the weights learnt so "
                    "far. Test = STDP off, a new stream around the same "
                    "patterns. New network = start again.",
                    color=C_MUTED, wrap=340)
        changed = self._nt_st_on_control
        with self._ctl_table():
            self._ctl_row("input", "nt_st_input", "combo", items=sv.INPUTS,
                          dv=sv.POISSON, callback=changed,
                          tip="Poisson patterns: the papers' input - 50 ms "
                              "sections of frozen variable-rate Poisson "
                              "activity hidden on a set of afferents. Scanned "
                              "images: spatio-temporal code, one afferent per "
                              "pixel row, pixel column c fires at c x bin ms; "
                              "a presentation is columns x bin ms long and "
                              "sits between background noise or distractor "
                              "images.")
            self._ctl_row("afferents", "nt_st_aff", "int", 1000,
                          "Input lines (100-4000).")
            self._ctl_row("neurons", "nt_st_neurons", "int", 3,
                          "Output neurons (1-8); lateral inhibition lets each "
                          "take one pattern.")
            self._ctl_row("patterns", "nt_st_patterns", "int", 3,
                          "Hidden patterns (1-8).")
            self._ctl_row("in pattern", "nt_st_in_pattern", default=0.5,
                          tip="Fraction of the afferents that take part in "
                              "each pattern. With several patterns each "
                              "gets its own random set.")
            self._ctl_row("image set", "nt_st_imgset", "combo",
                          items=sv.IMAGE_SETS, dv="digits", callback=changed,
                          tip="Built-in digits / letters / bars, or 'dataset': "
                              "one image per class of the loaded dataset "
                              "(the first of each class, in class order; "
                              "Invert from the Data tab applies).")
            self._ctl_row("image size", "nt_st_img_size", "int", 14,
                          "Rows = columns (5-28). Rows are the afferents.",
                          callback=changed)
            self._ctl_row("hidden", "nt_st_img_pick", "text", hint="0",
                          callback=changed,
                          tip="Indices of the hidden images, comma-separated "
                              "(0 = the first of the set). Each is one "
                              "pattern. Several images all drive the same "
                              "rows, so one neuron tends to take them all.")
            self._ctl_row("distractors", "nt_st_img_distract", "text",
                          hint="none", callback=changed,
                          tip="Indices of other images shown, unlabelled, in "
                              "every section that holds no hidden image. "
                              "Empty = none (background noise instead), "
                              "'all' = every image not hidden.")
            self._ctl_row("ms / column", "nt_st_bin", default=10.0,
                          callback=changed,
                          tip="Scan time per pixel column (min 1). The "
                              "section is ceil(columns x this) ms.")
            self._ctl_row("ink", "nt_st_ink", default=0.5, callback=changed,
                          tip="A pixel at or above this (0-1) fires.")
            self._ctl_row("background Hz", "nt_st_bg", default=10.0,
                          callback=changed,
                          tip="Poisson rate per afferent in the sections "
                              "that show no image. Unused when there are "
                              "distractors.")
            self._ctl_row("threshold", "nt_st_thr", default=0.0,
                          callback=changed,
                          tip="Neuron threshold in units of weight. 0 = auto: "
                              "Poisson input 0.25 x afferents (the 2008 paper: "
                              "500 for 2000 afferents); image input 0.85 x "
                              "the lowest peak potential one clean "
                              "presentation of a hidden image drives at the "
                              "starting weights. 0.85 is a free choice made "
                              "here, not a paper value. The value used is "
                              "logged.")
            self._ctl_row("synapses", "nt_st_synapse", "combo",
                          items=[IDEAL, DEVICE], dv=IDEAL, callback=changed,
                          tip="Ideal weights: the additive rule of the "
                              "paper. Selected device: the weight is the "
                              "conductance of the Verilog-A synapse chosen "
                              "above, written with its potentiation / "
                              "depression pulses.")
            self._ctl_row("write window", "nt_st_window", default=50.0,
                          tip="ms. Device synapses only. The STDP updates of "
                              "one window are merged into one pulse per "
                              "synapse. 0 = one pulse per update (slow).")
            self._ctl_row("pulse amp", "nt_st_pulse", "combo",
                          items=[CALIBRATED, AS_SET], dv=CALIBRATED,
                          tip="Device synapses only. Calibrated: the "
                              "amplitudes are searched so that one pulse "
                              "on a device at the starting weight moves "
                              "it by one STDP step (pulse width from "
                              "'Synapse write pulses'); the values found are "
                              "logged. As set: the pot / dep amplitude "
                              "below stands for one STDP step, whatever "
                              "the device does with it.")
        if dpg.does_item_exist("nt_st_img_pick"):
            dpg.set_value("nt_st_img_pick", "0")
        with dpg.group(tag="nt_st_img_extra", show=False):
            dpg.add_button(label="Load MNIST", tag="nt_st_load_mnist", width=-1,
                           callback=self.on_nt_download_mnist)
            with dpg.tooltip("nt_st_load_mnist"):
                dpg.add_text("Loads datasets/mnist.npz of the workspace (the "
                             "existing Data-tab loader; it downloads only when "
                             "the file is missing). For image set 'dataset'.",
                             wrap=300)
            self._small("Known limit: several hidden images are not split "
                        "over several neurons (every image drives the same "
                        "rows, one neuron takes them all), and a neuron "
                        "trained on one image also answers other images shown "
                        "as distractors. Hence the default: one image.",
                        color=C_MUTED, wrap=340)
        with dpg.collapsing_header(label="Stream input", default_open=True):
            with self._ctl_table():
                self._ctl_row("seconds", "nt_st_seconds", default=60.0,
                              tip="Simulated time of one Train or Test run.")
                self._ctl_row("pattern share", "nt_st_share", default=0.4,
                              callback=changed,
                              tip="Share of the sections that hold a pattern "
                                  "(0.25 in the 2008 paper). Patterns never "
                                  "follow one another directly.")
                self._ctl_row("spacing", "nt_st_spacing", "combo",
                              items=sv.SPACINGS, dv="random",
                              tip="random: pattern sections drawn at random "
                                  "(never two in a row). even: every k-th "
                                  "section, k = max(2, round(1 / share)).")
                self._ctl_row("jitter ms", "nt_st_jitter", default=1.0)
                self._ctl_row("noise Hz", "nt_st_noise", default=10.0,
                              tip="Extra random spikes on every afferent, "
                                  "everywhere, inside the patterns too. "
                                  "Switching the input sets it to 10 (Poisson) "
                                  "or 0 (images) unless you changed it.")
                self._ctl_row("spike deletion", "nt_st_delete", default=0.0,
                              tip="Fraction of the pattern's spikes dropped "
                                  "at each presentation.")
            self._small("Share, spacing, jitter, noise, deletion and "
                        "background Hz may be changed between runs (test "
                        "under more jitter, noise or deletion). Changing "
                        "anything else, or the seed, starts a new network.",
                        color=C_MUTED, wrap=340)
        with dpg.collapsing_header(label="Learning rule and neuron (stream)",
                                   default_open=False):
            with self._ctl_table(label_w=110):
                for tag, label, _field, default, lo, hi, tip in RULE:
                    self._ctl_row(label, tag, default=default, callback=changed,
                                  tip=f"{tip[0].upper()}{tip[1:]}. Default "
                                      f"{default:g}, allowed {lo:g} to {hi:g}.")
            dpg.add_button(label="Paper defaults", width=-1,
                           callback=self.on_nt_stream_rule_defaults)
            self._small("Defaults: Masquelier et al. 2008, the values the "
                        "benchmark is validated with. The two inhibition "
                        "values are choices made here: a neuron that fires "
                        "resets the others and holds them back.",
                        color=C_MUTED, wrap=340)
        self._nt_st_show_input_rows()

    def _nt_st_show_input_rows(self):
        images = self._nt_st_is_images()
        for tags, show in ((POISSON_TAGS, not images), (IMAGE_TAGS, images)):
            for tag in tags:
                if dpg.does_item_exist(tag):
                    row = dpg.get_item_parent(tag)
                    if "TableRow" in dpg.get_item_type(row):
                        dpg.configure_item(row, show=show)
        if dpg.does_item_exist("nt_st_img_extra"):
            dpg.configure_item("nt_st_img_extra", show=images)
        if dpg.does_item_exist("nt_st_load_mnist"):
            dpg.configure_item("nt_st_load_mnist", enabled=images and self._nt_get(
                "nt_st_imgset", "digits") == "dataset")

    def _nt_st_on_control(self, sender=None, *_):
        self._nt_st_changed(sender if isinstance(sender, str) else "")

    def _nt_st_changed(self, tag):
        """A stream control changed (widget callback, bridge or agent set):
        input-dependent rows, the noise default, and the previews."""
        try:
            if tag == "nt_st_input":
                kind = sv.input_kind(self._nt_get("nt_st_input", sv.POISSON))
                old = self._nt_st_input_shown
                if kind != old:
                    self._nt_st_input_shown = kind
                    noise = float(self._nt_get("nt_st_noise", 10.0))
                    if noise == sv.NOISE_DEFAULT[old] and dpg.does_item_exist(
                            "nt_st_noise"):
                        dpg.set_value("nt_st_noise", sv.NOISE_DEFAULT[kind])
                        self.log(f"[stream] input: {kind}; noise Hz set to its "
                                 f"default for this input, "
                                 f"{sv.NOISE_DEFAULT[kind]:g}")
            if tag in ("nt_st_input", "nt_st_imgset"):
                self._nt_st_show_input_rows()
            if tag in PREVIEW_TAGS or not tag:
                self._nt_st_refresh_previews()
        except Exception as exc:                        # noqa: BLE001
            self.log(f"[stream] preview error: {exc!r}")

    def _nt_st_refresh_previews(self):
        """Rebuild the architecture from the controls (no run) and redraw
        the Network / Patterns tabs. Errors are shown in place."""
        if self.trainer_running:
            return
        try:
            inp = self._nt_st_input_cfg()
            self._nt_st_arch = self._nt_st_make_arch(inp, None)
        except (ValueError, TypeError) as exc:
            self._nt_st_arch = {"error": str(exc)}
        if not self._nt_st_same_net(self._nt_st_arch):
            self._nt_st_net_w = None            # starting weights
        elif self._nt_stream_session is not None:
            w = self._nt_stream_session.weights
            if w is not None:
                self._nt_st_net_w = np.atleast_2d(w)[:, self._nt_st_arch["net_idx"]]
        self._nt_st_draw_net()
        self._nt_st_draw_patterns()

    def _nt_st_same_net(self, arch):
        """Is ``arch`` the network of the current session (its key a prefix
        of the session key)?"""
        k, cur = (arch or {}).get("key"), self._nt_stream_key
        return bool(k and cur and tuple(cur[:len(k)]) == tuple(k))

    def _nt_apply_mode(self):
        """Show the controls and views of the selected learning mode. Called
        on every change of ``nt_mode`` (set_value fires no callback)."""
        stream = self._nt_is_stream()
        for tag, show in (("nt_grid_net", not stream), ("nt_grid_rule", not stream),
                          ("nt_grid_neuron", not stream), ("nt_viz_tabs", not stream),
                          ("nt_stream_group", stream), ("nt_stream_center", stream)):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=show)
        if dpg.does_item_exist("nt_devlabel"):
            if stream:
                if self._nt_devlabel_grid is None:
                    self._nt_devlabel_grid = dpg.get_value("nt_devlabel")
                dpg.set_value("nt_devlabel", "Unsupervised stream: set the "
                              "network on the left, then Train. Power profile "
                              "and saved models belong to the other modes.")
            elif getattr(self, "_nt_devlabel_grid", None) is not None:
                dpg.set_value("nt_devlabel", self._nt_devlabel_grid)
                self._nt_devlabel_grid = None
        if dpg.does_item_exist("nt_btn_build"):
            dpg.configure_item("nt_btn_build", label=" New network " if stream
                               else " Build network ")
        if stream:
            self._nt_st_changed("")
        else:
            self._nt_st_set_play(False)
            try:
                self._on_nt_viz_tab()
            except Exception:                           # noqa: BLE001
                pass

    def _on_nt_mode(self, *_):
        """The mode combo changed. Not while a run is in flight."""
        if self.trainer_running:
            dpg.set_value("nt_mode", self._nt_mode_shown)
            self._nt_status("stop the run before changing the mode", C_AMBER)
            return
        self._nt_mode_shown = dpg.get_value("nt_mode")
        self._nt_apply_mode()

    def on_nt_stream_rule_defaults(self, *_):
        for tag, _label, _field, default, *_rest in RULE:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, default)
        self._nt_st_changed("")

    def _nt_st_set_combo(self, tag, val):
        """Normalise a bridge / agent value of a stream combo; None = reject."""
        s = str(val).strip().lower()
        if tag == "nt_st_input":
            return sv.input_kind(val)
        if tag == "nt_st_spacing":
            return "even" if s.startswith("e") else "random" if s.startswith(
                "r") else None
        if tag == "nt_st_speed":
            try:
                f = float(s.rstrip("x"))
            except ValueError:
                return None
            return next((v for v in sv.SPEEDS if float(v[:-1]) == f), None)
        if tag == "nt_st_imgset":
            for v in sv.IMAGE_SETS:
                if s.startswith(v[:3]):
                    return v
            return None
        if tag == "nt_st_synapse":
            return DEVICE if s.startswith(("sel", "dev")) else IDEAL if \
                s.startswith("ide") else None
        if tag == "nt_st_pulse":
            return CALIBRATED if s.startswith("cal") else AS_SET if \
                s.startswith("as") else None
        return val

    # ---- start / stop -----------------------------------------------

    def _nt_st_rule(self):
        g = self._nt_get
        rule = {field: min(max(float(g(tag, default)), lo), hi)
                for tag, _label, field, default, lo, hi, _tip in RULE}
        if rule["tau_s"] >= rule["tau_m"]:
            raise ValueError("tau s must be smaller than tau m")
        return rule

    def _nt_st_input_cfg(self):
        """The input part of the settings (render thread, no device): what
        the network sees and its threshold. Raises ValueError."""
        g = self._nt_get
        rule = self._nt_st_rule()
        n_out = min(max(int(g("nt_st_neurons", 3)), 1), 8)
        thr = float(g("nt_st_thr", 0.0))
        share = min(max(float(g("nt_st_share", 0.4)), 0.01), 0.5)
        inp = {"input": sv.input_kind(g("nt_st_input", sv.POISSON)),
               "n_out": n_out, "rule": rule, "share": share, "image": None}
        if inp["input"] == sv.POISSON:
            aff = min(max(int(g("nt_st_aff", 1000)), 100), 4000)
            frac = min(max(float(g("nt_st_in_pattern", 0.5)), 0.05), 1.0)
            inp.update(aff=aff, n_pat=min(max(int(g("nt_st_patterns", 3)), 1), 8),
                       n_pattern=max(int(round(frac * aff)), 1),
                       pattern_ms=50.0,
                       threshold=thr if thr > 0 else 0.25 * aff,
                       thr_how="as set" if thr > 0 else "auto: 0.25 x afferents")
            return inp
        name = str(g("nt_st_imgset", "digits"))
        size = min(max(int(g("nt_st_img_size", 14)), 5), 28)
        bin_ms = max(float(g("nt_st_bin", 10.0)), 1.0)
        ink = min(max(float(g("nt_st_ink", 0.5)), 0.0), 1.0)
        invert = bool(g("nt_ds_invert", False)) if name == "dataset" else False
        ds = getattr(self, "_nt_ds", None) if name == "dataset" else None
        bank = sv.image_bank(name, size, ds, invert)
        img = sv.build_image_input(bank, g("nt_st_img_pick", "0"),
                                   g("nt_st_img_distract", ""), size, bin_ms, ink)
        if len(img["patterns"]) > 8:
            raise ValueError("at most 8 hidden images")
        if thr > 0:
            how = "as set"
        else:
            from neurovat.masquelier import MasquelierParams
            P = MasquelierParams(n_afferents=size, n_pattern=size, threshold=1.0,
                                 pattern_ms=img["pattern_ms"], n_out=n_out,
                                 **rule)
            thr = sv.auto_threshold(img["patterns"], P)
            how = (f"auto: {sv.AUTO_THR:g} x weakest clean image peak at the "
                   "starting weights")
        ds_id = ((ds.get("path"), ds.get("n")) if ds else None)
        img.update(set=name, invert=invert, ink=ink, ds_id=ds_id,
                   bg=max(float(g("nt_st_bg", 10.0)), 0.0))
        inp.update(aff=size, n_pat=len(img["patterns"]), n_pattern=size,
                   pattern_ms=img["pattern_ms"], threshold=thr, thr_how=how,
                   image=img)
        return inp

    def _nt_st_key(self, inp, seed):
        rule = [inp["rule"][field] for _t, _l, field, *_rest in RULE]
        if inp["image"] is None:            # the Poisson key, as before
            return [inp[k] for k in ("aff", "n_out", "n_pat", "n_pattern",
                                     "threshold")] + [seed] + rule
        img = inp["image"]
        return ["images", img["set"], img["size"], tuple(img["picks"]),
                tuple(img["distract"]), img["bin_ms"], img["ink"],
                img["invert"], img["ds_id"],
                sv.events_digest(img["patterns"] + img["distractors"]),
                inp["n_out"], inp["threshold"], seed] + rule

    def _nt_stream_settings(self):
        """Read the controls (render thread). Returns (key, cfg)."""
        g = self._nt_get
        inp = self._nt_st_input_cfg()
        cfg = {
            "aff": inp["aff"], "n_out": inp["n_out"], "n_pat": inp["n_pat"],
            "n_pattern": inp["n_pattern"], "threshold": inp["threshold"],
            "thr_how": inp["thr_how"], "input": inp["input"],
            "image": inp["image"], "pattern_ms": inp["pattern_ms"],
            "seed": int(g("nt_seed", 1)),
            "seconds": min(max(float(g("nt_st_seconds", 60.0)), 1.0), 3600.0),
            "share": inp["share"],
            "spacing": self._nt_st_set_combo("nt_st_spacing",
                                             g("nt_st_spacing", "random"))
            or "random",
            "jitter": max(float(g("nt_st_jitter", 1.0)), 0.0),
            "noise": max(float(g("nt_st_noise", 10.0)), 0.0),
            "delete": min(max(float(g("nt_st_delete", 0.0)), 0.0), 0.95),
            "bg": inp["image"]["bg"] if inp["image"] else 0.0,
            "device": None,
            "rule": inp["rule"],
        }
        key = self._nt_st_key(inp, cfg["seed"])
        if g("nt_st_synapse", IDEAL) == DEVICE:
            dev = self._nt_device_factory()
            if dev is None:
                raise ValueError("no synapse device selected")
            scale = 1e-12 if dev["kind"] == "current" else 1.0
            power, note = neuro.load_power_params(
                dev.get("family"), folder=self._nt_power_dir,
                choice=self._nt_power_choice())
            cfg["device"] = {
                "make": dev["make"], "kind": dev["kind"], "label": dev["label"],
                "pot": self._nt_getu("nt_potamp", 50.0) * scale,
                "dep": self._nt_getu("nt_depamp", 50.0) * scale,
                "width": max(self._nt_getu("nt_pwidth", 10.0), 0.01) * 1e-3,
                "window": max(float(g("nt_st_window", 50.0)), 0.0),
                "calibrate": g("nt_st_pulse", CALIBRATED) == CALIBRATED,
                "power": power, "power_note": note}
            d = cfg["device"]
            key += [dev["key"], d["pot"], d["dep"], d["width"], d["window"],
                    d["calibrate"]]
        return tuple(key), cfg

    def _nt_st_make_arch(self, inp, cfg, key=None):
        """What the views need to know about the network (self-describing:
        the draw code never reads controls)."""
        images = inp["image"] is not None
        aff = inp["aff"]
        shown = (np.arange(aff) if images
                 else sv.shown_afferents(aff, sv.RASTER_SHOWN))
        net_idx = (np.arange(aff) if images
                   else sv.shown_afferents(aff, sv.NET_SHOWN))
        dev = cfg["device"] if cfg else None
        syn = (dev["label"] if dev else
               IDEAL if self._nt_get("nt_st_synapse", IDEAL) == IDEAL
               else f"{DEVICE} ({self._nt_get('nt_device', '')})")
        if key is None:                 # the input part: a prefix of the key
            key = tuple(self._nt_st_key(inp, int(self._nt_get("nt_seed", 1))))
        arch = {"images": images, "input": inp["input"], "n_aff": aff,
                "n_out": inp["n_out"], "n_shown": int(shown.size),
                "shown": shown, "net_idx": net_idx,
                "w_init": inp["rule"]["w_init"], "threshold": inp["threshold"],
                "thr_how": inp["thr_how"], "synapse": syn, "rule": inp["rule"],
                "pattern_ms": inp["pattern_ms"], "key": key,
                "names": [f"pattern {p + 1}" for p in range(inp["n_pat"])]}
        if images:
            img = inp["image"]
            arch.update(size=img["size"], bin_ms=img["bin_ms"], bg=img["bg"],
                        names=[f"image {i} '{lab}'" for i, lab in
                               zip(img["picks"], img["labels"])],
                        dnames=[f"image {i} '{lab}'" for i, lab in
                                zip(img["distract"], img["dlabels"])],
                        vecs=img["vecs"],
                        dvecs=img["dvecs"], patterns=img["patterns"],
                        distractors=img["distractors"])
        return arch

    def on_nt_stream(self, *_):
        self._nt_stream_start(True)

    def on_nt_stream_test(self, *_):
        self._nt_stream_start(False)

    def on_nt_stream_reset(self, *_):
        if self.trainer_running:
            return
        self._nt_stream_session = self._nt_stream_key = None
        self._nt_stream_last = None
        self._nt_stream_clear()
        self._nt_st_rec = None
        self._nt_st_set_play(False)
        self._nt_stream_say("new network - press Train", C_TEXT2)
        self._nt_status("stream network reset", C_GREEN)
        self._nt_st_changed("")

    def _nt_stream_say(self, text, color=C_TEXT2):
        if dpg.does_item_exist("nt_st_status"):
            dpg.set_value("nt_st_status", text)
            dpg.configure_item("nt_st_status", color=color)

    def _nt_stream_clear(self):
        self._nt_stream_series = {"t": [], "hit": [], "fa": [], "lat": []}
        self._nt_stream_t0 = 0.0
        for tag in ("nt_st_s_hit", "nt_st_s_fa", "nt_st_s_lat"):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, [[], []])

    def _nt_stream_start(self, learn):
        if self.trainer_running:
            return
        self._nt_stream_error = None
        try:
            key, cfg = self._nt_stream_settings()
        except (ValueError, TypeError) as exc:
            self._nt_stream_error = str(exc)
            self._nt_stream_say(str(exc), C_RED)
            return
        fresh = self._nt_stream_session is None or key != self._nt_stream_key
        if fresh and not learn:
            self._nt_stream_error = "nothing to test: Train first"
            self._nt_stream_say(self._nt_stream_error + " (a network setting "
                                "changed, or no network yet)", C_RED)
            return
        if fresh:
            self._nt_stream_session = None
            self._nt_stream_key = key
            self._nt_stream_clear()
        self._nt_st_set_play(False)
        images = cfg["image"] is not None
        cfg["t0_s"] = self._nt_stream_t0
        cfg["record_u"] = cfg["seconds"] * cfg["n_out"] <= sv.U_BUDGET
        cfg["view_ms"] = sv.view_window_ms(self._nt_get("nt_st_view_win", 0.0),
                                           images, cfg["pattern_ms"],
                                           cfg["share"])
        arch = self._nt_st_make_arch(
            {k: cfg[k] for k in ("aff", "n_out", "n_pat", "threshold",
                                 "thr_how", "input", "image", "pattern_ms",
                                 "rule", "share")}, cfg, key)
        cfg["arch"] = self._nt_st_arch = arch
        if fresh:
            self._nt_st_net_w = None
        self._nt_st_setup_raster(arch)
        if dpg.does_item_exist("nt_st_u_note"):
            dpg.set_value("nt_st_u_note", "" if cfg["record_u"] else
                          f"potential not recorded: seconds x neurons above "
                          f"{sv.U_BUDGET:g} (this run {cfg['seconds'] * cfg['n_out']:g})")
        rule_txt = sv.hit_rule(images)
        if dpg.does_item_exist("nt_st_s_hit"):
            dpg.configure_item("nt_st_s_hit", label=f"pattern windows answered "
                               f"({rule_txt})")
        self.trainer_running = True
        self._trainer_stop = False
        self._nt_busy(True)
        what = "training (STDP on)" if learn else "testing (STDP off)"
        self._nt_status(f"stream {what}", C_AMBER)
        self._nt_stream_say(f"{what} - {cfg['seconds']:g} s of stream...",
                            C_AMBER)
        syn = cfg["device"]["label"] if cfg["device"] else IDEAL
        if images:
            img = cfg["image"]
            src = (f"images {img['picks']} of '{img['set']}' ({img['size']} x "
                   f"{img['size']}, {img['bin_ms']:g} ms / column, section "
                   f"{cfg['pattern_ms']:g} ms), distractors "
                   f"{img['distract'] or 'none'}")
        else:
            src = f"{cfg['aff']} afferents, {cfg['n_pat']} pattern(s)"
        self.log(f"[stream] {what}: {src}, {cfg['n_out']} neuron(s), "
                 f"{cfg['seconds']:g} s, synapses: {syn}, threshold "
                 f"{cfg['threshold']:.4g} ({cfg['thr_how']}), spacing "
                 f"{cfg['spacing']}")
        threading.Thread(target=self._nt_stream_worker,
                         args=(cfg, learn, self._nt_stream_session),
                         daemon=True).start()

    # ---- worker (no DPG access) -------------------------------------

    def _nt_stream_worker(self, cfg, learn, session):
        out = {"learn": learn, "error": None, "session": session}
        try:
            from neurovat.masquelier import (MasquelierParams, Session,
                                             calibrate_write, pulse_step)
            from neurovat.power import EnergyMeter, format_report
            d = cfg["device"]
            img = cfg["image"]
            given = ({"patterns": img["patterns"],
                      "distractors": img["distractors"]} if img else {})
            if session is None:
                extra = {"pattern_ms": cfg["pattern_ms"]} if img else {}
                P = MasquelierParams(
                    n_afferents=cfg["aff"], n_pattern=cfg["n_pattern"],
                    threshold=cfg["threshold"], n_out=cfg["n_out"],
                    seed=cfg["seed"], **extra, **cfg["rule"])
                if d is None:
                    session = Session(P, cfg["n_pat"], **given)
                else:
                    array = neuro.twin_factory(d["make"])(cfg["n_out"],
                                                          cfg["aff"])
                    energy = EnergyMeter([(cfg["n_out"], cfg["aff"])],
                                         [d["kind"]], d["power"], 1e-3)
                    a_minus = P.a_minus_ratio * P.a_plus
                    if d["calibrate"]:
                        w = calibrate_write(array, P, d["width"], d["pot"])
                    else:
                        sign = array.potentiation_sign(d["pot"], d["width"])
                        w = {"pot_sign": sign, "pulse_width": d["width"],
                             "write_gain": d["pot"] / P.a_plus,
                             "dep_gain": d["dep"] / a_minus if a_minus > 0 else 0.0,
                             "pot_amp": d["pot"], "dep_amp": d["dep"],
                             "pot_step": pulse_step(array, sign * d["pot"],
                                                    d["width"], P.w_init),
                             "dep_step": -pulse_step(array, -sign * d["dep"],
                                                     d["width"], P.w_init)}
                    unit = "A" if d["kind"] == "current" else "V"
                    how = CALIBRATED if d["calibrate"] else AS_SET
                    self.q.put(("log", (
                        f"[stream] write pulses ({how}): potentiation "
                        f"{w['pot_amp']:.3g} {unit} -> +{w['pot_step']:.4f} "
                        f"of the range (STDP step {P.a_plus:.4f}); depression "
                        f"{w['dep_amp']:.3g} {unit} -> -{w['dep_step']:.4f} "
                        f"(STDP step {a_minus:.4f}); width {d['width']:.3g} s")))
                    session = Session(
                        P, cfg["n_pat"], array, write_window_ms=d["window"],
                        energy=energy, **given, **{k: w[k] for k in (
                            "write_gain", "dep_gain", "pot_sign",
                            "pulse_width")})
                    session.pulse = {k: w[k] for k in (
                        "pot_amp", "dep_amp", "pot_step", "dep_step")}
                    session.pulse["calibrated"] = d["calibrate"]
                out["session"] = session
            images = img is not None
            pms = session.P.pattern_ms
            arch = cfg["arch"]
            acc = {"n": 0, "pat": 0, "hit": 0, "fa": 0, "lat": []}
            per = max(int(round(CHUNK_S * 1000.0 / pms)), 1)
            live = {"view": None, "t": [], "j": [], "wall": 0.0}
            net_idx = np.asarray(arch["net_idx"], int)

            def snapshot(t_ms, info):
                if live["view"] is None:
                    live["view"] = sv.RunView(
                        info["stream"], pms, cfg["n_out"], cfg["threshold"],
                        cfg["t0_s"], arch["shown"], cfg["aff"], cfg["view_ms"],
                        images)
                snap = live["view"].snapshot(t_ms, live["t"], live["j"],
                                             info["u_ms"], live=True)
                snap["weights"] = np.atleast_2d(
                    np.array(info["weights"], float))[:, net_idx].copy()
                snap["arch"] = arch
                self.q.put(("nt_stream_view", snap))

            def on_section(t_ms, info):
                base = info["section"] * pms
                for rel, j in info["spikes"]:
                    live["t"].append(base + rel)
                    live["j"].append(j)
                now = time.monotonic()
                if now - live["wall"] >= 1.0 / VIEW_HZ:
                    live["wall"] = now
                    try:
                        snapshot(t_ms, info)
                    except Exception as exc:            # noqa: BLE001
                        live["wall"] = now + 3600.0     # stop trying
                        self.q.put(("log", f"[stream] live view error: {exc!r}"))
                first = min((s[0] for s in info["spikes"]), default=None)
                acc["n"] += 1
                if info["label"] >= 0:
                    acc["pat"] += 1
                    if sv.section_hit(info["spikes"], images):
                        acc["hit"] += 1
                        acc["lat"].append(first)
                else:
                    acc["fa"] += len(info["spikes"])
                if acc["n"] >= per:
                    span = acc["n"] * pms / 1000.0
                    self.q.put(("nt_stream", {
                        "t_s": t_ms / 1000.0,
                        "hit": 100.0 * acc["hit"] / acc["pat"] if acc["pat"] else None,
                        "fa": acc["fa"] / span,
                        "lat": float(np.mean(acc["lat"])) if acc["lat"] else None,
                        "spikes": info["post_spikes"]}))
                    acc.update(n=0, pat=0, hit=0, fa=0, lat=[])
                return not self._trainer_stop

            changes = {"pattern_frac": cfg["share"], "jitter_ms": cfg["jitter"],
                       "noise_hz": cfg["noise"], "spike_deletion": cfg["delete"],
                       "spacing": cfg["spacing"]}
            if images:
                changes["bg_hz"] = cfg["bg"]
            res = session.run(cfg["seconds"], learn=learn, on_section=on_section,
                              record_u=cfg["record_u"], **changes)
            owner = res["pattern_owner"]
            lats = [res["per_neuron"][j]["mean_latency_ms"]
                    for j in owner.values()]
            out["weights"] = np.atleast_2d(np.array(res["weights"], float))
            out["view"] = sv.RunView(session.stream, pms, cfg["n_out"],
                                     cfg["threshold"], cfg["t0_s"],
                                     arch["shown"], cfg["aff"],
                                     cfg["view_ms"], images)
            out["view"].duration_ms = min(out["view"].duration_ms,
                                          res["simulated_s"] * 1000.0)
            out["post"] = (np.array(res["post_t_ms"], float),
                           np.array(res["post_neuron"], int))
            out["u"] = res.get("u_ms")
            out["arch"] = arch
            out["share"] = cfg["share"]
            out["result"] = _clean({
                "phase": "train" if learn else "test",
                "stopped": bool(self._trainer_stop),
                "input": cfg["input"],
                "images": ({k: img[k] for k in ("set", "size", "picks",
                                                "distract", "labels",
                                                "bin_ms", "ink", "bg")}
                           if img else None),
                "synapses": d["label"] if d else IDEAL,
                "write_pulse": getattr(session, "pulse", None),
                "afferents": cfg["aff"], "neurons": cfg["n_out"],
                "patterns": cfg["n_pat"], "threshold": cfg["threshold"],
                "threshold_rule": cfg["thr_how"],
                "pattern_ms": pms, "hit_rule": sv.hit_rule(images),
                "rule": cfg["rule"],
                "stream": {k: cfg[k] for k in ("share", "spacing", "jitter",
                                               "noise", "delete", "bg")},
                "simulated_s": res["simulated_s"],
                "score_window_s": res["score_window_s"],
                "pattern_occurrences": res["pattern_occurrences"],
                "post_spikes": res["post_spikes"], "writes": res["writes"],
                "patterns_learnt": len(owner), "pattern_owner": owner,
                "false_alarms_per_s": res["false_alarms_per_s"],
                "mean_latency_ms": float(np.mean(lats)) if lats else None,
                "spikes_per_window": res["spikes_per_window"],
                "per_neuron": res["per_neuron"]})
            if d is not None:
                phase = "train" if learn else "eval"
                out["energy"] = (
                    f"power model: {d['power_note']}\n\n" + format_report(
                        session.kw["energy"].report(phase),
                        f"Stream {'training' if learn else 'test'} - "
                        f"{d['label']}"))
        except Exception as ex:                          # noqa: BLE001
            out["error"] = str(ex) or repr(ex)
            self.q.put(("log", f"[stream] error: {ex!r}"))
        self.q.put(("nt_stream_done", out))

    # ---- queue handlers (render thread) -----------------------------

    def _on_nt_stream(self, d):
        s = self._nt_stream_series
        s["t"].append(self._nt_stream_t0 + d["t_s"])
        for k in ("hit", "fa", "lat"):
            s[k].append(d[k])
        for k in ("hit", "fa", "lat"):
            pts = [(t, v) for t, v in zip(s["t"], s[k]) if v is not None]
            dpg.set_value(f"nt_st_s_{k}", [[p[0] for p in pts],
                                           [p[1] for p in pts]])
        for ax in ("nt_st_x1", "nt_st_y2", "nt_st_x2"):
            dpg.fit_axis_data(ax)
        dpg.set_axis_limits("nt_st_y1", -3.0, 106.0)
        hit = "-" if d["hit"] is None else f"{d['hit']:.0f} %"
        self._nt_stream_say(f"{s['t'][-1]:.0f} s of stream - hit rate {hit}, "
                            f"false alarms {d['fa']:.1f} /s, "
                            f"{d['spikes']} output spikes this run", C_AMBER)

    @staticmethod
    def _nt_st_window_lines(r):
        """Window counts per neuron that owns a pattern (or every neuron)."""
        per = r["per_neuron"]
        owners = sorted(set(int(j) for j in r["pattern_owner"].values()))
        rows = owners or list(range(len(per)))
        out = []
        for j in rows:
            w = per[j]["windows"]
            ba = w["balanced_accuracy"]
            out.append(f"neuron {j + 1} vs pattern {per[j]['best_pattern'] + 1}: "
                       f"TP {w['tp']} FN {w['fn']} FP {w['fp']} TN {w['tn']}, "
                       f"balanced accuracy "
                       f"{'-' if ba is None else f'{ba:.2f}'}")
        return out

    def _on_nt_stream_done(self, out):
        self.trainer_running = False
        self._nt_busy(False)
        if out.get("session") is not None:
            self._nt_stream_session = out["session"]
        agent, self._nt_stream_agent = self._nt_stream_agent, False
        if out["error"]:
            self._nt_stream_error = out["error"]
            self._nt_stream_session = self._nt_stream_key = None
            self._nt_status("stream failed", C_RED)
            self._nt_stream_say("failed: " + out["error"], C_RED)
            if agent:
                self.append_chat("sys", "stream run failed: " + out["error"])
            return
        r = self._nt_stream_last = out["result"]
        self._nt_stream_t0 += r["simulated_s"]
        images = r["input"] == sv.IMAGES
        own = ", ".join(f"pattern {int(p) + 1} -> neuron {j + 1}"
                        for p, j in sorted(r["pattern_owner"].items()))
        lat = ("-" if r["mean_latency_ms"] is None
               else f"{r['mean_latency_ms']:.1f} ms")
        wins = self._nt_st_window_lines(r)
        if images:
            per = r["per_neuron"]
            covered = {q["best_pattern"] for q in per
                       if (q["windows"]["balanced_accuracy"] or 0) >= 0.9}
            good = len(covered) == r["patterns"]
        else:
            good = (r["patterns_learnt"] == r["patterns"]
                    and r["false_alarms_per_s"] < 1.0)
        text = (f"{'Training' if out['learn'] else 'Test'}"
                f"{' (stopped)' if r['stopped'] else ''}: "
                f"hit = {r['hit_rule']} ({r['pattern_ms']:g} ms windows). "
                f"{r['patterns_learnt']} of {r['patterns']} pattern(s) "
                f"learnt: answered in at least 90 % of their windows by a "
                f"neuron that stays quiet elsewhere ({own or 'none'}); "
                f"{r['false_alarms_per_s']:.2f} false alarms /s; latency {lat}; "
                f"{r['post_spikes']} output spikes, {r['writes']:,} writes. "
                f"Windows: " + "; ".join(wins) + ". "
                f"Scored on the last {r['score_window_s']:.0f} s of "
                f"{r['simulated_s']:.0f} s.")
        dpg.set_value("nt_st_verdict", text)
        dpg.configure_item("nt_st_verdict", color=C_GREEN if good else C_AMBER)
        self._nt_status("stream done", C_GREEN if good else C_AMBER)
        self._nt_stream_say(f"done - {self._nt_stream_t0:.0f} s of stream in "
                            "total. Train continues from here; Test checks "
                            "without learning.", C_TEXT2)
        self.log("[stream] " + text)
        if dpg.does_item_exist("nt_st_table_cap"):
            dpg.set_value("nt_st_table_cap", f"SPIKES PER {r['pattern_ms']:g} ms "
                          "WINDOW (end of run)")
        self._nt_stream_table(r["spikes_per_window"])
        self._nt_stream_weights(out["weights"])
        if out.get("energy"):
            dpg.set_value("nt_st_energy", out["energy"])
        try:
            arch = out["arch"]
            self._nt_st_arch = arch
            self._nt_st_net_w = out["weights"][:, np.asarray(arch["net_idx"], int)]
            self._nt_st_pending = None
            self._nt_st_keep_run(out["view"], out["post"][0], out["post"][1],
                                 out["u"], arch, out["share"])
            self._nt_st_draw_net()
            self._nt_st_draw_patterns()
        except Exception as exc:                        # noqa: BLE001
            self.log(f"[stream] view error: {exc!r}")
        if agent and not r["stopped"]:
            self._nt_stream_feedback(text)

    def _nt_stream_feedback(self, text, limit=8):
        """Hand the result of an agent-started run back to the agent, which
        may start another; pause after ``limit`` runs in a row."""
        self._nt_stream_rounds += 1
        if self._nt_stream_rounds >= limit or self.chat_busy:
            self._nt_stream_rounds = 0
            self.append_chat("sys", "stream run done: " + text
                             + " (ask the agent to continue)")
            return
        self.append_chat("sys", "stream run done - result sent to the agent")
        self._main_send("The stream run you started just finished.\n"
                        + "\n".join(self._nt_stream_context())
                        + "\nSay what it shows. Start another stream / "
                        "stream_test only if a specific change is worth "
                        "testing; otherwise stop.")

    def _nt_stream_context(self):
        """Lines for the agent's context: the last stream run."""
        r = self._nt_stream_last
        if not r:
            return ["stream mode: no run yet."]
        src = (f"images {r['images']['picks']} of '{r['images']['set']}' "
               f"({r['images']['size']} rows), distractors "
               f"{r['images']['distract'] or 'none'}"
               if r.get("images") else f"{r['afferents']} afferents, "
               f"{r['patterns']} Poisson pattern(s)")
        return [
            f"stream mode, last run ({r['phase']}, {r['synapses']}): {src}, "
            f"{r['neurons']} neuron(s), {r['simulated_s']:.0f} s, threshold "
            f"{r['threshold']:.4g} ({r['threshold_rule']}), section "
            f"{r['pattern_ms']:g} ms, stream {r['stream']}",
            f"  hit rule: {r['hit_rule']}; patterns learnt "
            f"{r['patterns_learnt']}/{r['patterns']} "
            f"(pattern -> neuron, 0-based: {r['pattern_owner']}), false alarms "
            f"{r['false_alarms_per_s']:.2f}/s, latency {r['mean_latency_ms']} "
            f"ms, writes {r['writes']}",
            "  windows: " + "; ".join(self._nt_st_window_lines(r)),
            f"  spikes per {r['pattern_ms']:g} ms window, neuron x (patterns..., "
            "no pattern): "
            + str([[round(v, 2) for v in row] for row in r["spikes_per_window"]]),
        ]
