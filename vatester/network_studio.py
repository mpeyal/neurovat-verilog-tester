"""One desktop network workflow, backed by explicit Verilog-A/twin bindings."""
import hashlib
from pathlib import Path
import re

import dearpygui.dearpygui as dpg


def source_record(path, key):
    path = Path(path).resolve(strict=True)
    if path.suffix.lower() != ".va":
        raise ValueError("Select a Verilog-A (.va) synapse source.")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "twin_key": key, "execution": "Python behavioral twin"}


def hidden_sizes(text):
    if not text.strip():
        return []
    if not re.fullmatch(r"\s*\d+\s*(,\s*\d+\s*)*", text):
        raise ValueError("Hidden layers must be comma-separated neuron counts.")
    sizes = [int(v) for v in text.split(",")]
    if len(sizes) > 4 or any(not 1 <= v <= 64 for v in sizes):
        raise ValueError("Use up to four hidden layers, with 1-64 neurons each.")
    return sizes


class NetworkStudio:
    def __init__(self, app, specs):
        self.app = app
        self.specs = {s.key: s for s in specs}
        self.sources = {}
        self._architecture_values = None

    def source_controls(self):
        dpg.add_text("Verilog-A synapse")
        dpg.add_combo([], tag="nt_va_source", width=-1, callback=self.select_source)
        dpg.add_text("", tag="nt_va_description", wrap=310)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Parameters / source", callback=self.inspect)
            dpg.add_button(label="Rescan", callback=lambda: self.app.rescan_va())
        with dpg.tooltip("nt_va_source"):
            dpg.add_text("Only sources mapped to a conductance twin are supported. Mapping is not proof of equation equivalence. Training runs the Python twin, not a Verilog-A solver.", wrap=330)

    def refresh_sources(self):
        if not dpg.does_item_exist("nt_va_source"):
            return
        self.sources = {str(Path(v.path).relative_to(self.app.workdir)): v
                        for v in self.app.va_files if v.model_key in self.specs
                        and v.role == "device" and not v.error
                        and "trainer" not in getattr(self.specs[v.model_key].cls, "GUI_PROFILE", {}).get("hide", ())}
        current = dpg.get_value("nt_va_source")
        key = next((k for k, s in self.specs.items() if s.label == dpg.get_value("nt_device")), None)
        if current not in self.sources:
            current = next((name for name, v in self.sources.items() if v.model_key == key), "")
        dpg.configure_item("nt_va_source", items=list(self.sources), default_value=current)
        self.describe()

    def sync_device(self):
        key = next((k for k, s in self.specs.items() if s.label == dpg.get_value("nt_device")), None)
        current = self.sources.get(dpg.get_value("nt_va_source"))
        if current is None or current.model_key != key:
            dpg.set_value("nt_va_source", next((n for n, v in self.sources.items() if v.model_key == key), ""))
        self.describe()

    def select_source(self, *_):
        va = self.sources.get(dpg.get_value("nt_va_source"))
        if va:
            dpg.set_value("nt_device", self.specs[va.model_key].label)
            self.app._on_nt_device_change()
        self.describe()

    def describe(self):
        va = self.sources.get(dpg.get_value("nt_va_source"))
        text = "No supported .va source. Add a source and its conductance twin."
        if va:
            text = f"Python twin: {self.specs[va.model_key].label}"
            if va.model_key == "fefet":
                text += "\nApproximate model; source equivalence unvalidated."
        dpg.set_value("nt_va_description", text)

    def binding(self):
        va = self.sources.get(dpg.get_value("nt_va_source"))
        if va is None:
            raise ValueError("Choose a supported Verilog-A synapse source first.")
        if dpg.get_value("nt_device") != self.specs[va.model_key].label:
            raise ValueError("Source and twin differ. Select the synapse source again.")
        return source_record(va.path, va.model_key)

    def require_current(self):
        if self.app.trainer is not None:
            if self.binding() != getattr(self.app.trainer, "source_binding", None):
                raise ValueError("Synapse source changed. Build the network before training or evaluation.")

    def restore_binding(self, model):
        saved = model.get("synapse_source")
        key = model.get("device", {}).get("key")
        candidates = [(name, v) for name, v in self.sources.items() if v.model_key == key]
        if saved:
            candidates = [(name, v) for name, v in candidates
                          if source_record(v.path, key)["sha256"] == saved.get("sha256")]
        if not candidates:
            raise ValueError("Saved synapse source is missing or changed. Restore its .va file before loading.")
        name, va = candidates[0]
        dpg.set_value("nt_device", self.specs[va.model_key].label)
        dpg.set_value("nt_va_source", name)
        self.describe()

    def inspect(self, *_):
        va = self.sources.get(dpg.get_value("nt_va_source"))
        if va:
            self.app.desktop.device()
            dpg.set_value("param_model_sel", self.specs[va.model_key].label)
            self.app.rebuild_param_panel()
            self.app.selected_va = va
            self.app.on_open_in_editor()

    def architecture(self):
        dpg.add_text("Dense crossbars + LIF neurons")
        dpg.add_text("Each connection uses the selected Verilog-A device's Python twin.")
        dpg.add_spacer(height=8)
        with dpg.group(horizontal=True):
            dpg.add_listbox([], tag="ns_layers", width=300, num_items=6, callback=self.select_layer)
            with dpg.group():
                dpg.add_input_int(label="Neurons", tag="ns_units", default_value=8,
                                  min_value=1, max_value=64, min_clamped=True, max_clamped=True, width=110)
                dpg.add_button(label="Add hidden layer", callback=lambda: self.edit("add"))
                dpg.add_button(label="Apply size", callback=lambda: self.edit("apply"))
                dpg.add_button(label="Remove layer", callback=lambda: self.edit("remove"))
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Move up", callback=lambda: self.edit("up"))
                    dpg.add_button(label="Move down", callback=lambda: self.edit("down"))
        dpg.add_text("", tag="ns_shape", wrap=800)
        dpg.add_text("Architecture edits take effect when you build the network.")
        self.refresh_layers()

    def refresh_layers(self):
        if not dpg.does_item_exist("ns_layers"):
            return
        try:
            sizes = hidden_sizes(dpg.get_value("nt_hidden"))
        except ValueError as exc:
            dpg.set_value("ns_shape", str(exc))
            return
        items = [f"{i+1}. Dense {n} / LIF" for i, n in enumerate(sizes)]
        old = dpg.get_value("ns_layers")
        dpg.configure_item("ns_layers", items=items, default_value=old if old in items else next(iter(items), ""))
        shape = [dpg.get_value("nt_gh")*dpg.get_value("nt_gw"), *sizes, dpg.get_value("nt_nout")]
        count = sum(a*b for a, b in zip(shape, shape[1:]))
        dpg.set_value("ns_shape", " -> ".join(map(str, shape))+f"   |   {count:,} device synapses")

    def tick(self):
        values = tuple(dpg.get_value(tag) for tag in ("nt_hidden", "nt_gh", "nt_gw", "nt_nout"))
        if values != self._architecture_values:
            self._architecture_values = values
            self.refresh_layers()

    def select_layer(self, *_):
        value = dpg.get_value("ns_layers")
        if value:
            dpg.set_value("ns_units", hidden_sizes(dpg.get_value("nt_hidden"))[int(value.split(".")[0])-1])

    def edit(self, action):
        if self.app.trainer_running:
            return
        try:
            sizes = hidden_sizes(dpg.get_value("nt_hidden"))
            value = dpg.get_value("ns_layers")
            index = int(value.split(".")[0])-1 if value else -1
            if action == "add":
                sizes.append(dpg.get_value("ns_units"))
            elif 0 <= index < len(sizes):
                if action == "apply":
                    sizes[index] = dpg.get_value("ns_units")
                elif action == "remove":
                    sizes.pop(index)
                else:
                    target = index + (-1 if action == "up" else 1)
                    if 0 <= target < len(sizes):
                        sizes[index], sizes[target] = sizes[target], sizes[index]
            text = ",".join(map(str, sizes))
            hidden_sizes(text)
            dpg.set_value("nt_hidden", text)
            self.refresh_layers()
        except ValueError as exc:
            dpg.set_value("ns_shape", str(exc))
            self.app.log(f"[network] {exc}")
