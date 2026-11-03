"""Native design/navigation/dialog QA in an isolated workspace; no agent calls."""
from pathlib import Path
import shutil
import tempfile

import dearpygui.dearpygui as dpg

from vatester.app import App
from vatester.va_scan import parse_va


def frames(app, count=8):
    for _ in range(count):
        app._process_queue()
        app.desktop.tick()
        dpg.render_dearpygui_frame()


def capture(app, name):
    frames(app)
    dpg.output_frame_buffer(str(Path(__file__).parent/"results/design_system"/f"{name}.png"))
    frames(app)


def shown(tag):
    return dpg.get_item_configuration(tag)["show"]


def main():
    repo = Path(__file__).resolve().parent
    (repo/"results/design_system").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="design_qa_") as tmp:
        root = Path(tmp)
        shutil.copytree(repo/"verilog", root/"verilog")
        shutil.copytree(repo/"twins", root/"twins", ignore=shutil.ignore_patterns("__pycache__"))
        assert parse_va(root/"verilog/ideal_signed_synapse.va").role == "network_primitive"
        app = App(tmp, bridge=False)
        try:
            app.build(); frames(app)
            assert not shown("untwin_modal"), "An ideal network primitive must not trigger a missing-device-twin dialog"
            capture(app, "device_lab")
            for name in ("custom_synapse", "custom_capacitor"):
                (root/f"{name}.va").write_text(f"module {name}(a,b); inout a,b; endmodule")
            app.rescan_va(); frames(app)
            assert shown("untwin_modal")
            assert set(dpg.get_item_configuration("untwin_list")["items"]) == {"custom_synapse.va", "custom_capacitor.va"}
            primary, cancel = dpg.get_item_rect_min("untwin_build"), dpg.get_item_rect_min("untwin_dismiss")
            assert primary[0] > cancel[0] and abs(primary[1]-cancel[1]) < 2
            assert dpg.get_item_rect_max("untwin_build")[0] < dpg.get_item_pos("untwin_modal")[0]+dpg.get_item_rect_size("untwin_modal")[0]
            capture(app, "device_setup_dialog")
            # Verify selection routing without invoking the actual code-writing agent.
            selected = []
            app._on_build_twin_for = lambda sender, value, name: selected.append(name)
            dpg.set_value("untwin_list", "custom_capacitor.va")
            app._build_selected_twin()
            assert selected == ["custom_capacitor.va"]
            dpg.hide_item("untwin_modal")
            app.on_model_builder(); frames(app)
            assert not dpg.does_item_exist("btn_model_builder")
            assert not dpg.does_item_exist("workspace_builder_host")
            dispatched = []
            start = app.on_nt_train
            app.on_nt_train = lambda: dispatched.append("network")
            app._run_active()
            app.on_nt_train = start
            assert dispatched == ["network"], "F5 must use the sole network trainer"
            dpg.set_value("ns_units", 16)
            app.network_studio.edit("add")
            app._on_resize()
            assert dpg.get_value("nt_hidden") == "16"
            capture(app, "network_studio")
            dpg.set_value("nt_viz_tabs", "nt_tab_data")
            capture(app, "network_data")
            assert not dpg.is_item_visible("nt_stat_best")
            for source, control in (("custom", "nt_pat_editor"), ("dataset", "nt_ds_url")):
                app._nt_set_source(source); frames(app)
                dpg.get_item_callback("nt_source_setup")()
                frames(app)
                assert dpg.is_item_visible(control)
                capture(app, f"data_{source}_settings")
                dpg.hide_item("nt_source_popup"); frames(app)
            app._nt_set_source("set")
            dpg.get_item_callback("nt_noise_setup")(); frames(app)
            assert dpg.is_item_visible("nt_input_noise")
            dpg.set_value("nt_input_noise", .2)
            app._nt_encoding_preview()
            assert "noise 20%" in dpg.get_value("nt_enc_status")
            dpg.hide_item("nt_noise_popup"); frames(app)
            dpg.set_value("nt_viz_tabs", "nt_tab_results"); frames(app)
            assert dpg.get_item_rect_size("nt_stat_best")[0] > 0
            app.desktop.device(); frames(app)
            assert shown("device_actions")
            app.on_open_trainer(); frames(app)
            dpg.configure_viewport(0, width=1160, height=760); frames(app); app._on_resize()
            dpg.set_value("nt_viz_tabs", "ns_architecture_tab")
            capture(app, "network_compact")
            assert dpg.get_item_rect_size("ns_layers")[0] > 0
            app.desktop.device()
            dpg.show_item("sweep_dialog")
            capture(app, "sweep_dialog")
            assert dpg.get_item_rect_max("sweep_apply")[1] < dpg.get_item_pos("sweep_dialog")[1]+dpg.get_item_rect_size("sweep_dialog")[1]
            dpg.hide_item("sweep_dialog")
            frames(app)
            dpg.show_item("account_modal")
            capture(app, "account_dialog")
            assert dpg.is_item_visible("account_done"), "Account dialog did not render"
            assert dpg.get_item_rect_max("account_done")[1] < dpg.get_item_pos("account_modal")[1]+dpg.get_item_rect_size("account_modal")[1]
            dpg.hide_item("account_modal")
            assert not any("[ui] error:" in line for line in app._log_lines), app._log_lines
            print("PASS: shared native styling, twin prompt filtering/selection/footer, single network navigation, preserved architecture, and compact layout")
        finally:
            (repo/"results/design_system/session_log.txt").write_text("\n".join(app._log_lines), encoding="utf-8")
            app.virtuoso.disconnect()
            dpg.destroy_context()


if __name__ == "__main__":
    main()
