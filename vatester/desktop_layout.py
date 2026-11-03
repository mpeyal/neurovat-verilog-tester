"""Native workspace navigation and parameter presentation (no simulation logic)."""
import contextlib
import inspect
import re
import time

import dearpygui.dearpygui as dpg


DEVICE_TABS = ("tab_designer", "tab_results", "tab_analysis", "tab_stdp",
               "tab_polar", "tab_noise", "tab_source", "tab_compute", "tab_log")

PARAMETER_LABELS = {
    "Rmin": "Min. R (ohm)", "Rmax": "Max. R (ohm)", "Rinit": "Initial R (ohm)",
    "P_init": "Initial polarization", "Vc": "Coercive V (V)",
    "Vread": "Read bias (V)", "Vth0": "Initial threshold (V)",
    "tau_ret": "Retention (s)", "tau_retention": "Retention (s)",
    "n_states": "State count", "polarity": "Write polarity",
    "seed": "Random seed", "A_stdp": "Added timing gain", "A_stdp_V": "Added timing (V)",
}


class DesktopLayout:
    def __init__(self, app):
        self.app = app
        self._dialog_visible = set()
        self.last_device_tab = "tab_designer"
        self.last_tab = None
        self.pending_frames = 0
        self.preview_at = None
        self.popups = {}

    @contextlib.contextmanager
    def popup(self, button, tag=None):
        """Button menus open at their anchor, not the window's default origin."""
        args = {"tag": tag} if tag else {}
        with dpg.window(popup=True, show=False, autosize=True, no_move=True, **args) as popup:
            yield popup
        self.popups[popup] = button
        dpg.set_item_callback(button, lambda *_: self.open_popup(popup))

    def position_popup(self, popup):
        button = self.popups[popup]
        state = dpg.get_item_state(button)
        if "rect_min" not in state:
            return
        x, top = state["rect_min"]
        y = state["rect_max"][1]+4
        width, height = dpg.get_item_rect_size(popup)
        width, height = width or 280, height or 160
        vw, vh = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        x = max(8, min(x, vw-width-8))
        if y+height > vh-8:
            y = max(8, top-height-4)
        dpg.set_item_pos(popup, (x, y))

    def open_popup(self, popup):
        self.position_popup(popup)
        dpg.show_item(popup)

    def request_preview(self, *_):
        self.preview_at = time.monotonic() + .2

    def header(self):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Device Lab", tag="workspace_device",
                           height=28, width=130, callback=self.device)
            dpg.add_button(label="Network Studio", tag="btn_trainer",
                           height=28, width=130, callback=self.app.on_open_trainer)
            with dpg.tooltip("btn_trainer"):
                dpg.add_text("Build, train and inspect a network using Verilog-A device twins.", wrap=350)
            from .design_system import TEXT_MUTED
            dpg.add_text("", tag="workspace_description", color=TEXT_MUTED, show=False)

    def device(self, *_):
        dpg.show_item(self.last_device_tab)
        dpg.set_value("center_tabs", self.last_device_tab)
        self.app._on_center_tab()

    def assistant(self, *_):
        visible = not dpg.get_item_configuration("right_child")["show"]
        dpg.configure_item("right_child", show=visible)
        dpg.configure_item("assistant_toggle", label="Close assistant" if visible else "Assistant")
        self.app._on_resize()

    def use_device(self, *_):
        spec = self.app._param_spec()
        if "trainer" in getattr(spec.cls, "GUI_PROFILE", {}).get("hide", ()):
            self.app.log("[trainer] This device does not provide a conductance weight.")
            return
        dpg.set_value("nt_device", spec.label)
        self.app._on_nt_device_change()
        self.app.on_open_trainer()

    def sync(self):
        if not dpg.does_item_exist("center_tabs"):
            return
        active = dpg.get_value("center_tabs")
        if active != self.last_tab:
            self.pending_frames = 3
        network = active in ("tab_trainer", dpg.get_alias_id("tab_trainer"))
        if not network and active in DEVICE_TABS:
            self.last_device_tab = active
        profile = self.app._class_attr(self.app.device_class, "GUI_PROFILE") or {}
        allowed = "trainer" not in profile.get("hide", ())
        for tag in DEVICE_TABS:
            visible = not network
            if tag == "tab_polar":
                visible &= "polarization" in (self.app._class_attr(self.app.device_class, "ANALYSES") or ())
            dpg.configure_item(tag, show=visible)
        dpg.configure_item("tab_trainer", show=network)
        dpg.configure_item("btn_trainer", show=True, enabled=True)
        dpg.configure_item("use_device_network", enabled=allowed)
        for tag in ("device_actions", "btn_stdp", "device_toolbar"):
            dpg.configure_item(tag, show=not network)
        # These pages provide their own experiment actions.
        own_actions = any(active in (tag, dpg.get_alias_id(tag)) for tag in ("tab_stdp", "tab_polar", "tab_noise", "tab_compute"))
        dpg.configure_item("device_actions", show=not network and not own_actions)
        label = "Run training" if network else "Run simulation"
        for tab, action in (("tab_stdp", "Run STDP"), ("tab_polar", "Run measurement"), ("tab_noise", "Run noise analysis"), ("tab_compute", "Run compute")):
            if active in (tab, dpg.get_alias_id(tab)):
                label = action
        dpg.configure_item("btn_run", show=True, label=label)
        dpg.set_value("workspace_description", "Verilog-A synapses / architecture / training" if network else "Device characterization")
        for tag, selected in (("workspace_device", not network), ("btn_trainer", network)):
            dpg.bind_item_theme(tag, self.app.themes["nav_active" if selected else "nav"])
        self.last_tab = active

    def tick(self):
        for popup in self.popups:
            if dpg.is_item_shown(popup):
                self.position_popup(popup)
        try:
            self.fit_content()
        except (KeyError, SystemError, TypeError) as exc:
            # Item bounds can be unavailable while a native surface is rebuilt.
            if not getattr(self, "_layout_error", False):
                self.app.log(f"[layout] Waiting for rendered bounds: {exc}")
                self._layout_error = True
        else:
            self._layout_error = False
        # Center native dialogs after their content has acquired a real size.
        from .design_system import center_dialog
        dialogs = ("untwin_modal", "account_modal", "virt_modal", "virt_write_modal", "sweep_dialog",
                   "dir_dialog", "attach_dialog", "save_va_dialog", "fit_csv_dialog",
                   "nt_ds_file_dialog", "nt_save_dialog", "nt_load_dialog")
        visible = {tag for tag in dialogs if dpg.does_item_exist(tag) and dpg.is_item_shown(tag)}
        for tag in visible-self._dialog_visible:
            center_dialog(tag)
        self._dialog_visible = visible
        # Programmatic navigation (worker results, tools, bridge) also updates
        # the inspector; DearPyGui's set_value does not fire tab callbacks.
        if self.pending_frames:
            dpg.set_value("center_tabs", self.last_tab)
            self.pending_frames -= 1
        elif dpg.get_value("center_tabs") != self.last_tab:
            self.app._on_center_tab()
        if self.preview_at is not None and time.monotonic() >= self.preview_at and not self.app.sim_running:
            self.preview_at = None
            self.app.on_preview()

    def fit_content(self):
        """Size visible plot canvases from rendered bounds, including DPI scaling.

        Negative heights inside nested groups can use the wrong available region.
        Give plots explicit heights from their screen position to the panel bottom.
        Settings retain their own scroll regions; the central plots never need one.
        """
        if not dpg.does_item_exist("center_child"):
            return
        vh = dpg.get_viewport_client_height()
        top = dpg.get_item_pos("center_child")[1]
        if top < 1:
            return
        height = max(100, int(vh-top-12))
        for tag in ("left_child", "center_child", "right_child"):
            if dpg.does_item_exist(tag):
                self._height(tag, height)
        bottom = top+height-12

        def visible(tag):
            if not dpg.does_item_exist(tag):
                return False
            item = dpg.get_alias_id(tag)
            while item:
                item = dpg.get_alias_id(item) if isinstance(item, str) else item
                if dpg.get_item_configuration(item).get("show") is False:
                    return False
                info = dpg.get_item_info(item)
                parent = info.get("parent")
                if info.get("type", "").endswith("::mvTab"):
                    selected = dpg.get_value(parent)
                    selected = dpg.get_alias_id(selected) if isinstance(selected, str) else selected
                    if selected != item:
                        return False
                item = parent
            return True

        def remaining(tag, margin=0):
            return max(80, int(bottom-self._position(tag)[1]-margin))

        if visible("nt_data_scroll"):
            self._height("nt_data_scroll", remaining("nt_data_scroll"))
            self._height("nt_data_plots", remaining("nt_enc_clean", 60))
            width = max(180, dpg.get_item_rect_size("center_child")[0]-60)
            dpg.configure_item("nt_enc_status", wrap=width)
            dpg.set_y_scroll("nt_data_scroll", 0)

        if visible("compute_plots"):
            width = max(180, dpg.get_item_rect_size("center_child")[0]-35)
            for tag in ("compute_scope", "compute_status"):
                dpg.configure_item(tag, wrap=width)

        # Multi-plot layouts share one bounded canvas, preserving both axes.
        for tag, first in (("stdp_plots", "stdp_drive_plot"), ("transient_plots", "plot_i"),
                           ("noise_plots", "nplot_psd"),
                           ("compute_plots", "compute_plot_output")):
            if visible(tag):
                # DPG subplots expose no rendered bounds; their first plot does.
                self._height(tag, remaining(first))
        for tag in ("preview_plot", "ana_plot", "polar_plot", "nt_mem_plot",
                    "nt_spk_plot", "nt_wtrace_plot", "nt_cm_plot", "nt_cell_plot", "nt_wm_plot"):
            if visible(tag):
                self._height(tag, remaining(tag))
        if visible("nt_acc_plot"):
            available = remaining("nt_acc_plot", 36)
            self._height("nt_acc_plot", int(available*.53))
            self._height("nt_wevo_plot", int(available*.47))
        if visible("nt_raster_input_card"):
            half = int(remaining("nt_raster_input_card", 18)/2)
            for card, plot in (("nt_raster_input_card", "nt_rc_in"), ("nt_raster_output_card", "nt_rc_out")):
                self._height(card, half)
                self._height(plot, max(80, half-55))
        if visible("nt_neuron_card"):
            self._height("nt_neuron_card", remaining("nt_neuron_card"))
            for plot in ("nt_nv_plot", "nt_fi_plot"):
                if visible(plot):
                    self._height(plot, remaining(plot, 12))
        # Reset any scroll position left over from a previously taller view.
        dpg.set_y_scroll("center_child", 0)

    @staticmethod
    def _position(tag):
        state = dpg.get_item_state(tag)
        if "rect_min" in state:
            return state["rect_min"]
        pos = dpg.get_item_pos(tag)
        parent = dpg.get_item_info(tag).get("parent")
        while parent:
            info = dpg.get_item_info(parent)
            if info.get("type", "").endswith(("::mvChildWindow", "::mvWindowAppItem")):
                origin = DesktopLayout._position(parent)
                return [pos[0]+origin[0], pos[1]+origin[1]]
            parent = info.get("parent")
        return pos

    @staticmethod
    def _height(tag, height):
        height = max(1, int(height))
        if dpg.get_item_configuration(tag).get("height") != height:
            dpg.configure_item(tag, height=height)


def parameter_descriptions(params_cls):
    """Use the model author's comments; never infer physical units from names."""
    descriptions = {}
    for cls in reversed(params_cls.__mro__):
        try:
            source = inspect.getsource(cls)
        except (OSError, TypeError):
            continue
        for line in source.splitlines():
            match = re.match(r"\s+(\w+)\s*:\s*[^=]+=[^#]+#\s*(.+)", line)
            if match:
                descriptions[match[1]] = match[2].strip()
    return descriptions


def parameter_group(name):
    lower = name.lower()
    if "stdp" in lower or "trace" in lower:
        return "Timing rule"
    if any(x in lower for x in ("noise", "sigma", "seed", "random")):
        return "Variability"
    if any(x in lower for x in ("tau", "tao", "retention", "kappa")):
        return "Dynamics & retention"
    if any(x in lower for x in ("init", "min", "max", "read", "th0")):
        return "State & readout"
    return "Switching & device"


def build_parameters(app, spec, defaults, text_color):
    descriptions = parameter_descriptions(spec.params_cls)
    app._parameter_rows = []
    app._parameter_groups = []
    vals = app.param_values[spec.key]
    for group in ("State & readout", "Switching & device", "Dynamics & retention", "Timing rule", "Variability"):
        names = [name for name in vals if parameter_group(name) == group]
        if not names:
            continue
        with dpg.collapsing_header(label=group, parent="param_panel", default_open=True) as header:
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp) as table:
                dpg.bind_item_theme(table, app.themes["table_roomy"])
                dpg.add_table_column(init_width_or_weight=.47)
                dpg.add_table_column(init_width_or_weight=.53)
                rows = []
                for name in names:
                    val = vals[name]
                    with dpg.table_row() as row:
                        rows.append(row)
                        friendly = PARAMETER_LABELS.get(name, name)
                        label = dpg.add_text(friendly, color=text_color)
                        description = descriptions.get(name, "Model parameter; see the device source for its definition.")
                        with dpg.tooltip(label):
                            dpg.add_text(name)
                            dpg.add_text(description, wrap=290)
                            dpg.add_text(f"Default: {defaults.get(name, val):.6g}")
                        is_int = isinstance(val, int) and not isinstance(val, bool)
                        widget = dpg.add_input_int if is_int else dpg.add_input_double
                        options = {} if is_int else {"format": "%.6g"}
                        widget(tag=f"p_{spec.key}_{name}", default_value=val, width=-1,
                               step=0, callback=app._on_param_edit,
                               user_data=(spec.key, name, is_int), **options)
                        app._parameter_rows.append((row, name, (name+" "+friendly+" "+description+" "+group).lower()))
            app._parameter_groups.append((header, rows))
    filter_parameters(app)


def filter_parameters(app, *_):
    query = dpg.get_value("param_search").strip().lower()
    changed = dpg.get_value("param_changed")
    spec = app._param_spec()
    defaults = vars(spec.params_cls())
    visible = set()
    for row, name, search in getattr(app, "_parameter_rows", []):
        show = all(word in search for word in query.split()) and (
            not changed or app.param_values[spec.key][name] != defaults.get(name))
        dpg.configure_item(row, show=show)
        if show:
            visible.add(row)
    for header, rows in getattr(app, "_parameter_groups", []):
        dpg.configure_item(header, show=bool(visible.intersection(rows)))
        if query or changed:
            dpg.set_value(header, True)
    dpg.set_value("param_count", f"{len(visible)} / {len(getattr(app, '_parameter_rows', []))} parameters   |   hover names for details")
