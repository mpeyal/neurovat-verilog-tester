"""Desktop controls and preview for waveform-driven STDP."""
import csv
import json
import os

import dearpygui.dearpygui as dpg

from vatester import signal_factory as sf
from vatester import stdp_protocol as protocol

SETTING_TAGS = ("stdp_shape", "stdp_r_kohm", "stdp_c_nf", "stdp_preview_dt",
                "stdp_measurement", "stdp_include_rule")


class StdpControls:
    def __init__(self, app):
        self.app = app

    def build(self):
        def row(tag, label, value):
            with dpg.table_row():
                dpg.add_text(label)
                dpg.add_input_double(tag=tag, default_value=value, width=-1, step=0,
                                     format="%.4g", callback=self.refresh)

        with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column()
            dpg.add_table_column()
            row("stdp_preview_dt", "Preview dt (ms)", 5.)
            with dpg.table_row():
                dpg.add_text("Measurement")
                dpg.add_combo(list(protocol.MEASUREMENTS), tag="stdp_measurement", width=-1,
                              default_value=protocol.MEASUREMENTS[0])
        # ON for a device that has its own timing response (ECFET); the app
        # sets it per device class, see App._stdp_own_timing
        dpg.add_checkbox(label="Device's own timing response (A_stdp)", tag="stdp_include_rule",
                         default_value=True, callback=self.refresh)
        with dpg.tooltip("stdp_include_rule"):
            dpg.add_text("ON: the device model responds to spike timing by itself, so plain "
                         "rectangular pulses give its STDP window and no spike shaper is needed "
                         "(ECFET). OFF: that response is switched off and only the overlap of "
                         "shaped spikes can produce a timing dependence (FeFET, RRAM).", wrap=420)
        dpg.add_text("", tag="stdp_shaper_note", wrap=490)
        # only shaped stimuli use the shaper; hidden for rectangular pulses
        with dpg.collapsing_header(label="Spike shaper", tag="stdp_shaper_header", default_open=True):
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column()
                dpg.add_table_column()
                with dpg.table_row():
                    dpg.add_text("Shape")
                    dpg.add_combo(list(protocol.SHAPES), tag="stdp_shape", width=-1,
                                  default_value=protocol.SHAPES[0], callback=self.refresh)
                row("stdp_r_kohm", "R (kOhm)", 1000.)
                row("stdp_c_nf", "C (nF)", 20.)
        dpg.add_text("Protocol assumptions (hover)", tag="stdp_assumptions")
        with dpg.tooltip("stdp_assumptions"):
            dpg.add_text("", tag="stdp_mechanism", wrap=460)
        dpg.add_text("Baseline controls: pre only / post only / idle", tag="stdp_control_summary", wrap=490)

    def build_preview(self):
        with dpg.plot(label="Spike contributions and net drive", tag="stdp_drive_plot"):
            dpg.add_plot_legend(horizontal=True, location=dpg.mvPlot_Location_NorthEast)
            dpg.add_plot_axis(dpg.mvXAxis, label="time (ms)", tag="stdp_drive_x")
            dpg.add_plot_axis(dpg.mvYAxis, label="drive", tag="stdp_drive_y")

    def spike(self):
        if dpg.get_value("stdp_mode") == "rectangular pulses":
            return None
        return {"shape": dpg.get_value("stdp_shape"),
                "tail_ratio": dpg.get_value("stdp_tail_ratio"),
                "tail_tau": dpg.get_value("stdp_tail_tau_ms")*1e-3,
                "r_ohm": dpg.get_value("stdp_r_kohm")*1e3,
                "c_f": dpg.get_value("stdp_c_nf")*1e-9}

    def refresh(self, *_):
        if not dpg.does_item_exist("stdp_shape"):
            return
        panel_width = dpg.get_item_rect_size("center_child")[0]
        if dpg.does_item_exist("stdp_intro"):
            dpg.configure_item("stdp_intro", wrap=max(300, panel_width-40) if panel_width else 650)
        spike = self.spike()
        rc = spike is not None and spike["shape"] == protocol.SHAPES[2]
        for tag in ("stdp_r_kohm", "stdp_c_nf"):
            dpg.configure_item(tag, enabled=rc)
        for tag in ("stdp_tail_ratio", "stdp_tail_tau_ms"):
            dpg.configure_item(tag, enabled=spike is not None and not rc)
        dpg.configure_item("stdp_shape", enabled=spike is not None)
        own = dpg.get_value("stdp_include_rule")
        dpg.configure_item("stdp_shaper_header", show=spike is not None)
        dpg.set_value("stdp_shaper_note", (
            "Rectangular pulses, no spike shaper: the device's own timing response sets the window."
            if spike is None and own else
            "Rectangular pulses with the device's own timing response OFF: expect no timing "
            "window. Turn it on, or choose a shaped stimulus." if spike is None else
            "Shaped spikes and the device's own timing response are both active."
            if own else ""))
        text = ("RC shaper: tau = R*C. Buffered, unloaded high-pass response to a rectangular input; "
                "device loading is not simulated. " if rc else
                "Ideal programming driver. Exponential tails end at 6 tau; linear tails end at TAIL TAU. ")
        text += ("The device's own timing response is ON (A_stdp/A_stdp_V from the model parameters). "
                 if dpg.get_value("stdp_include_rule") else
                 "The device's own timing response is OFF; switching/retention dynamics remain. ")
        text += "Signed drive = pre + post. For voltage programming enter -Vpre and +Vpost. Negative dVt means potentiation."
        if self.app.device_class == "FeFET":
            text += " FeFET uses the simplified Python Merz model; equivalence to the loaded Verilog-A has not been established."
        dpg.set_value("stdp_mechanism", text)
        self.preview()

    def preview(self, experiment=None, dt=None):
        """Draw controls, or the exact completed experiment selected by a click."""
        dpg.delete_item("stdp_drive_y", children_only=True)
        try:
            unit = experiment["unit"] if experiment else dpg.get_value("unit_combo")
            scale = sf.UNIT_SCALE[unit]
            spike = experiment.get("spike") if experiment else self.spike()
            width = experiment["width"] if experiment else dpg.get_value("stdp_width_ms")*1e-3
            waves, stop = protocol.waveforms(
                dt if dt is not None else dpg.get_value("stdp_preview_dt")*1e-3,
                experiment["amp_pre"] if experiment else dpg.get_value("stdp_amp_pre")*scale,
                experiment["amp_post"] if experiment else dpg.get_value("stdp_amp_post")*scale,
                width, 0., spike)
            for key, label in (("pre", "Pre contribution"), ("post", "Post contribution"),
                               ("pair", "Net device drive")):
                x, y = self.app._stair_points(waves[key], 1/scale, tail=0.)
                dpg.add_stair_series([t*1e3 for t in x], y, label=label, parent="stdp_drive_y")
            dpg.configure_item("stdp_drive_y", label=unit)
            for axis in ("stdp_drive_x", "stdp_drive_y"):
                dpg.fit_axis_data(axis)
            first = min(waves["pair"].edges)
            dpg.set_axis_limits("stdp_drive_x", max(0., first-.5*width)*1e3,
                                (stop+.5*width)*1e3)
            info = f"tau = {protocol.tau(spike)*1e3:.4g} ms" if spike else "Rectangular pulses: no external timing tail"
            if experiment:
                info = "Last-run waveform: " + info
            dpg.set_value("stdp_preview_status", info)
        except Exception as exc:
            dpg.set_value("stdp_preview_status", str(exc))

    def export(self, *_):
        try:
            ctx = self.app._stdp_ctx
            if not ctx or not ctx.get("diagnostics") or self.app.sim_running:
                raise ValueError("Complete an STDP sweep before exporting.")
            folder = os.path.join(self.app.workdir, "results")
            os.makedirs(folder, exist_ok=True)
            base = os.path.join(folder, "stdp_experiment")
            data = {**ctx["protocol"], "dt_s": ctx["dts"],
                    "amp_pre_si": ctx["amp_pre"], "amp_post_si": ctx["amp_post"],
                    "width_s": ctx["width"], "read_delay_s": ctx["tail"],
                    "drive_kind": ctx["kind"], "display_unit": ctx["unit"],
                    "diagnostics": ctx["diagnostics"]}
            with open(base+".json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, allow_nan=False)
            with open(base+".csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "dt_s", "state_unit", "pair", "pre", "post", "idle", "interaction"])
                for model in ctx["models"]:
                    rows = ctx["diagnostics"][model.name]
                    for i, dt in enumerate(ctx["dts"]):
                        writer.writerow([model.name, dt, protocol.analysis.state_profile(model)[1]]+
                                        [rows[k][i] for k in ("pair", "pre", "post", "idle", "interaction")])
            self.app.log("[stdp] exported "+base+".json and .csv")
        except Exception as exc:
            self.app.log(f"[stdp] export: {exc}")
