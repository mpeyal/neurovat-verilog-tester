"""Trainer 'Stream' tab: continuous-stream STDP (Masquelier et al. 2008/2009).

One uninterrupted spike stream instead of epochs: hidden 50 ms patterns recur
in background activity, STDP acts at every spike, and lateral inhibition lets
each neuron take one pattern. Train runs with STDP on; Test freezes the
weights and plays a fresh stream around the same patterns. The network is its
own (the pixel-grid crossbar is too small for it) but uses the selected
synapse device and power profile.

App supplies widget helpers, the device factory and the event queue. The
worker only posts queue messages and always ends with ``nt_stream_done``.
"""
import math
import threading

import dearpygui.dearpygui as dpg
import numpy as np

from . import neuro
from .ui_constants import C_AMBER, C_GREEN, C_MUTED, C_RED, C_TEXT2

IDEAL, DEVICE = "ideal weights", "selected device"
CALIBRATED, AS_SET = "calibrated", "as set"
CHUNK_S = 2.0                       # live-plot resolution

# (tag, one-line meaning) - added to the controls the agent / bridge may set
STREAM_CONTROLS = [
    ("nt_st_aff", "stream: afferents (100-4000)"),
    ("nt_st_neurons", "stream: output neurons (1-8)"),
    ("nt_st_patterns", "stream: hidden patterns (1-8)"),
    ("nt_st_seconds", "stream: seconds simulated per Train / Test"),
    ("nt_st_share", "stream: share of time holding a pattern 0-0.5"),
    ("nt_st_in_pattern", "stream: fraction of afferents in each pattern"),
    ("nt_st_jitter", "stream: spike jitter ms"),
    ("nt_st_noise", "stream: extra noise spikes Hz"),
    ("nt_st_delete", "stream: fraction of pattern spikes deleted 0-1"),
    ("nt_st_thr", "stream: neuron threshold (0 = 0.25 x afferents)"),
    ("nt_st_synapse", f"stream: {IDEAL} | {DEVICE}"),
    ("nt_st_window", "stream: device write window ms (one pulse per synapse per window)"),
    ("nt_st_pulse", f"stream: device pulse amplitude, {CALIBRATED} (one pulse = "
                    f"one STDP step) | {AS_SET} (nt_potamp / nt_depamp)"),
]


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


class NetworkStreamMixin:
    _nt_stream_session = None       # neurovat.masquelier.Session
    _nt_stream_key = None           # settings the session was built for
    _nt_stream_last = None          # JSON-safe result of the last run
    _nt_stream_error = None
    _nt_stream_agent = False        # the agent started the run in flight
    _nt_stream_rounds = 0           # agent-started runs since the last pause

    # ---- layout -----------------------------------------------------

    def _nt_stream_tab(self):
        with self._pad(left=6, top=4, bottom=0):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=310, border=False):
                    self._nt_stream_controls()
                with dpg.child_window(width=-1, border=False):
                    self._nt_stream_views()
        self._nt_stream_clear()

    def _nt_stream_controls(self):
        with dpg.group(horizontal=True):
            b = dpg.add_button(label=" Train ", callback=self.on_nt_stream)
            dpg.bind_item_theme(b, self.themes["primary"])
            dpg.add_button(label=" Test ", callback=self.on_nt_stream_test)
            b = dpg.add_button(label=" Stop ", callback=self.on_nt_stop)
            dpg.bind_item_theme(b, self.themes["stop"])
            dpg.add_button(label=" Reset ", callback=self.on_nt_stream_reset)
        self._small("Train: STDP on, the stream continues from the weights "
                    "learnt so far. Test: STDP off, a new stream around the "
                    "same patterns.", color=C_TEXT2, wrap=296)
        dpg.add_spacer(height=4)
        with self._card("Network", height=304):
            with self._ctl_table(label_w=120):
                self._ctl_row("Afferents", "nt_st_aff", "int", 1000)
                self._ctl_row("Neurons", "nt_st_neurons", "int", 3)
                self._ctl_row("Patterns", "nt_st_patterns", "int", 3)
                self._ctl_row("In pattern", "nt_st_in_pattern", default=0.5,
                              tip="Fraction of the afferents that take part in "
                                  "each pattern. With several patterns each "
                                  "gets its own random set.")
                self._ctl_row("Threshold", "nt_st_thr", default=0.0,
                              tip="Neuron threshold in units of weight. 0 = "
                                  "0.25 x afferents (the 2008 paper: 500 for "
                                  "2000 afferents).")
                self._ctl_row("Synapses", "nt_st_synapse", "combo",
                              items=[IDEAL, DEVICE], dv=IDEAL,
                              tip="Ideal weights: the additive rule of the "
                                  "paper. Selected device: the weight is the "
                                  "conductance of the synapse device chosen "
                                  "in the left panel, written with its "
                                  "potentiation / depression pulses.")
                self._ctl_row("Write window ms", "nt_st_window", default=50.0,
                              tip="Device synapses only. The STDP updates of "
                                  "one window are merged into one pulse per "
                                  "synapse. 0 = one pulse per update (slow).")
                self._ctl_row("Pulse amplitude", "nt_st_pulse", "combo",
                              items=[CALIBRATED, AS_SET], dv=CALIBRATED,
                              tip="Device synapses only. Calibrated: the "
                                  "amplitudes are searched so that one pulse "
                                  "on a device at the starting weight moves "
                                  "it by one STDP step (pulse width from the "
                                  "left panel); the values found are logged. "
                                  "As set: the left panel's pot / dep "
                                  "amplitude stands for one STDP step, "
                                  "whatever the device does with it.")
        with self._card("Stream", height=212):
            with self._ctl_table(label_w=120):
                self._ctl_row("Seconds", "nt_st_seconds", default=60.0,
                              tip="Simulated time of one Train or Test run.")
                self._ctl_row("Pattern share", "nt_st_share", default=0.4,
                              tip="Share of the time that holds a pattern "
                                  "(0.25 in the 2008 paper). Patterns never "
                                  "follow one another directly.")
                self._ctl_row("Jitter ms", "nt_st_jitter", default=1.0)
                self._ctl_row("Noise Hz", "nt_st_noise", default=10.0,
                              tip="Extra random spikes on every afferent.")
                self._ctl_row("Spike deletion", "nt_st_delete", default=0.0,
                              tip="Fraction of the pattern's spikes dropped "
                                  "at each presentation.")
        self._small("The four stream settings may be changed between runs "
                    "(test under more jitter, noise or deletion). Changing a "
                    "network setting or the seed starts a new network. "
                    "Inhibition: a neuron that fires resets the others and "
                    "holds them back by 0.4 x threshold, decaying over 20 ms.",
                    color=C_MUTED, wrap=296)

    def _nt_stream_views(self):
        dpg.add_text("press Train", tag="nt_st_status", color=C_TEXT2, wrap=0)
        with dpg.tab_bar():
            with dpg.tab(label="  Learning  "):
                with dpg.plot(width=-1, height=185, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="stream time (s)",
                                      tag="nt_st_x1")
                    with dpg.plot_axis(dpg.mvYAxis, label="hit rate (%)",
                                       tag="nt_st_y1"):
                        dpg.add_line_series([], [], label="pattern windows "
                                            "answered by exactly one spike "
                                            "of a neuron", tag="nt_st_s_hit")
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
                self._small("SPIKES PER 50 ms WINDOW (end of run)",
                            color=C_TEXT2)
                dpg.add_group(tag="nt_st_table_holder")
            with dpg.tab(label="  Weights  "):
                self._small("Final weights of every neuron (0 = minimum, "
                            "1 = maximum conductance).", color=C_TEXT2)
                with dpg.plot(width=-1, height=-1, no_menus=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="weight",
                                      tag="nt_st_wx")
                    dpg.add_plot_axis(dpg.mvYAxis, label="synapses",
                                      tag="nt_st_wy")
            with dpg.tab(label="  Energy  "):
                dpg.add_input_text(tag="nt_st_energy", multiline=True,
                                   readonly=True, width=-1, height=-1,
                                   default_value="Device synapses only: run "
                                   "Train or Test with Synapses = "
                                   f"'{DEVICE}'.")

    # ---- start / stop -----------------------------------------------

    def _nt_stream_settings(self):
        """Read the controls (render thread). Returns (key, cfg)."""
        g = self._nt_get
        aff = min(max(int(g("nt_st_aff", 1000)), 100), 4000)
        frac = min(max(float(g("nt_st_in_pattern", 0.5)), 0.05), 1.0)
        thr = float(g("nt_st_thr", 0.0))
        cfg = {
            "aff": aff,
            "n_out": min(max(int(g("nt_st_neurons", 3)), 1), 8),
            "n_pat": min(max(int(g("nt_st_patterns", 3)), 1), 8),
            "n_pattern": max(int(round(frac * aff)), 1),
            "threshold": thr if thr > 0 else 0.25 * aff,
            "seed": int(g("nt_seed", 1)),
            "seconds": min(max(float(g("nt_st_seconds", 60.0)), 1.0), 3600.0),
            "share": min(max(float(g("nt_st_share", 0.4)), 0.01), 0.5),
            "jitter": max(float(g("nt_st_jitter", 1.0)), 0.0),
            "noise": max(float(g("nt_st_noise", 10.0)), 0.0),
            "delete": min(max(float(g("nt_st_delete", 0.0)), 0.0), 0.95),
            "device": None,
        }
        key = [cfg[k] for k in ("aff", "n_out", "n_pat", "n_pattern",
                                "threshold", "seed")]
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
        self._nt_stream_say("network reset - press Train", C_TEXT2)

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
        self.trainer_running = True
        self._trainer_stop = False
        self._nt_busy(True)
        what = "training (STDP on)" if learn else "testing (STDP off)"
        self._nt_status(f"stream {what}", C_AMBER)
        self._nt_stream_say(f"{what} - {cfg['seconds']:g} s of stream...",
                            C_AMBER)
        syn = cfg["device"]["label"] if cfg["device"] else IDEAL
        self.log(f"[stream] {what}: {cfg['aff']} afferents, {cfg['n_out']} "
                 f"neuron(s), {cfg['n_pat']} pattern(s), {cfg['seconds']:g} s, "
                 f"synapses: {syn}")
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
            if session is None:
                P = MasquelierParams(
                    n_afferents=cfg["aff"], n_pattern=cfg["n_pattern"],
                    threshold=cfg["threshold"], n_out=cfg["n_out"],
                    seed=cfg["seed"])
                if d is None:
                    session = Session(P, cfg["n_pat"])
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
                             "dep_gain": d["dep"] / a_minus,
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
                        energy=energy, **{k: w[k] for k in (
                            "write_gain", "dep_gain", "pot_sign",
                            "pulse_width")})
                    session.pulse = {k: w[k] for k in (
                        "pot_amp", "dep_amp", "pot_step", "dep_step")}
                    session.pulse["calibrated"] = d["calibrate"]
                out["session"] = session
            acc = {"n": 0, "pat": 0, "hit": 0, "fa": 0, "lat": []}
            per = max(int(round(CHUNK_S * 1000.0 / session.P.pattern_ms)), 1)

            def on_section(t_ms, info):
                first = min((s[0] for s in info["spikes"]), default=None)
                count = (np.bincount([s[1] for s in info["spikes"]])
                         if info["spikes"] else [])
                acc["n"] += 1
                if info["label"] >= 0:
                    acc["pat"] += 1
                    if 1 in count:          # some neuron fired exactly once
                        acc["hit"] += 1
                        acc["lat"].append(first)
                else:
                    acc["fa"] += len(info["spikes"])
                if acc["n"] >= per:
                    span = acc["n"] * session.P.pattern_ms / 1000.0
                    self.q.put(("nt_stream", {
                        "t_s": t_ms / 1000.0,
                        "hit": 100.0 * acc["hit"] / acc["pat"] if acc["pat"] else None,
                        "fa": acc["fa"] / span,
                        "lat": float(np.mean(acc["lat"])) if acc["lat"] else None,
                        "spikes": info["post_spikes"]}))
                    acc.update(n=0, pat=0, hit=0, fa=0, lat=[])
                return not self._trainer_stop

            res = session.run(cfg["seconds"], learn=learn, on_section=on_section,
                              pattern_frac=cfg["share"], jitter_ms=cfg["jitter"],
                              noise_hz=cfg["noise"],
                              spike_deletion=cfg["delete"])
            owner = res["pattern_owner"]
            lats = [res["per_neuron"][j]["mean_latency_ms"]
                    for j in owner.values()]
            out["weights"] = np.atleast_2d(np.array(res["weights"], float))
            out["result"] = _clean({
                "phase": "train" if learn else "test",
                "stopped": bool(self._trainer_stop),
                "synapses": d["label"] if d else IDEAL,
                "write_pulse": getattr(session, "pulse", None),
                "afferents": cfg["aff"], "neurons": cfg["n_out"],
                "patterns": cfg["n_pat"], "threshold": cfg["threshold"],
                "stream": {k: cfg[k] for k in ("share", "jitter", "noise",
                                               "delete")},
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
        dpg.set_axis_limits("nt_st_y1", -2.0, 102.0)
        hit = "-" if d["hit"] is None else f"{d['hit']:.0f} %"
        self._nt_stream_say(f"{s['t'][-1]:.0f} s of stream - hit rate {hit}, "
                            f"false alarms {d['fa']:.1f} /s, "
                            f"{d['spikes']} output spikes this run", C_AMBER)

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
        own = ", ".join(f"pattern {int(p) + 1} -> neuron {j + 1}"
                        for p, j in sorted(r["pattern_owner"].items()))
        lat = ("-" if r["mean_latency_ms"] is None
               else f"{r['mean_latency_ms']:.1f} ms")
        good = (r["patterns_learnt"] == r["patterns"]
                and r["false_alarms_per_s"] < 1.0)
        text = (f"{'Training' if out['learn'] else 'Test'}"
                f"{' (stopped)' if r['stopped'] else ''}: "
                f"{r['patterns_learnt']} of {r['patterns']} pattern(s) "
                f"learnt: answered in at least 90 % of their windows by a "
                f"neuron that stays quiet elsewhere ({own or 'none'}); "
                f"{r['false_alarms_per_s']:.2f} false alarms /s; latency {lat}; "
                f"{r['post_spikes']} output spikes, {r['writes']:,} writes. "
                f"Scored on the last {r['score_window_s']:.0f} s of "
                f"{r['simulated_s']:.0f} s.")
        dpg.set_value("nt_st_verdict", text)
        dpg.configure_item("nt_st_verdict", color=C_GREEN if good else C_AMBER)
        self._nt_status("stream done", C_GREEN if good else C_AMBER)
        self._nt_stream_say(f"done - {self._nt_stream_t0:.0f} s of stream in "
                            "total. Train continues from here; Test checks "
                            "without learning.", C_TEXT2)
        self.log("[stream] " + text)
        self._nt_stream_table(r["spikes_per_window"])
        self._nt_stream_weights(out["weights"])
        if out.get("energy"):
            dpg.set_value("nt_st_energy", out["energy"])
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

    def _nt_stream_context(self):
        """Lines for the agent's context: the last stream run."""
        r = self._nt_stream_last
        if not r:
            return ["stream mode: no run yet."]
        return [
            f"stream mode, last run ({r['phase']}, {r['synapses']}): "
            f"{r['afferents']} afferents, {r['neurons']} neuron(s), "
            f"{r['patterns']} pattern(s), {r['simulated_s']:.0f} s",
            f"  patterns learnt {r['patterns_learnt']}/{r['patterns']} "
            f"(pattern -> neuron, 0-based: {r['pattern_owner']}), false alarms "
            f"{r['false_alarms_per_s']:.2f}/s, latency {r['mean_latency_ms']} "
            f"ms, writes {r['writes']}",
            "  spikes per 50 ms window, neuron x (patterns..., no pattern): "
            + str([[round(v, 2) for v in row] for row in r["spikes_per_window"]]),
        ]
