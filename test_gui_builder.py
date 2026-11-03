"""Single Network Studio GUI regression; no MNIST download or benchmark."""
import json
from pathlib import Path
import shutil
import tempfile
import dearpygui.dearpygui as dpg
from vatester.app import App
from vatester.network_studio import hidden_sizes
from gui_test_helpers import pump


def main():
    repo = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="network_studio_") as temp:
        root = Path(temp)
        shutil.copytree(repo/"verilog", root/"verilog")
        shutil.copytree(repo/"twins", root/"twins", ignore=shutil.ignore_patterns("__pycache__"))
        app = App(temp, bridge=False)
        try:
            app.build()
            assert not dpg.does_item_exist("btn_model_builder")
            assert not dpg.does_item_exist("sb_window")
            app.on_model_builder()
            assert dpg.get_value("center_tabs") == "tab_trainer"
            assert dpg.get_value("nt_viz_tabs") == "ns_architecture_tab"
            studio = app.network_studio
            assert studio.binding()["path"].endswith(".va")
            assert all("fecap" not in name.lower() for name in studio.sources)
            dpg.set_value("ns_units", 8); studio.edit("add")
            dpg.set_value("ns_units", 4); studio.edit("add")
            assert hidden_sizes(dpg.get_value("nt_hidden")) == [8, 4]
            dpg.set_value("ns_layers", "2. Dense 4 / LIF"); studio.edit("up")
            assert hidden_sizes(dpg.get_value("nt_hidden")) == [4, 8]
            dpg.set_value("ns_layers", "1. Dense 4 / LIF"); studio.edit("remove")
            dpg.set_value("ns_units", 6); studio.edit("apply")
            assert hidden_sizes(dpg.get_value("nt_hidden")) == [6]
            for tag, value in (("nt_gh", 2), ("nt_gw", 2), ("nt_nout", 2),
                               ("nt_present", 20.), ("nt_epochs", 1),
                               ("nt_learnrule", "Surrogate grad (BPTT)")):
                dpg.set_value(tag, value)
            app.on_nt_build()
            assert app.trainer is not None
            assert list(app.trainer.layer_sizes) == [4, 6, 2]
            binding = dict(app.trainer.source_binding)
            saved_dt = dpg.get_value("nt_dt")
            app.on_nt_train(); pump(app)
            assert len(app._nt_acc_hist["epoch"]) == 1
            app.on_nt_test(); pump(app)
            path = root/"saved.json"
            app.on_nt_save(str(path))
            assert json.loads(path.read_text())["synapse_source"] == binding
            state = app.trainer.capture_state()
            app.on_nt_load(str(path))
            assert app.trainer.capture_state() == state
            source = Path(binding["path"])
            original = source.read_bytes()
            source.write_bytes(original+b"\n// changed source\n")
            current = app.trainer
            app.on_nt_train()
            assert not app.trainer_running
            assert "source changed" in dpg.get_item_configuration("nt_status")["label"].lower()
            app.on_nt_load(str(path))
            assert app.trainer is current
            assert "load failed" in dpg.get_item_configuration("nt_status")["label"]
            source.write_bytes(original)
            alternate = next(name for name, va in studio.sources.items() if va.model_key != binding["twin_key"])
            dpg.set_value("nt_va_source", alternate); studio.select_source()
            dpg.set_value("nt_hidden", "16,8")
            dpg.set_value("nt_dt", 2.)
            app.on_nt_save(str(path))
            assert json.loads(path.read_text())["synapse_source"] == binding
            assert json.loads(path.read_text())["device"]["key"] == binding["twin_key"]
            app.on_nt_load(str(path))
            assert list(app.trainer.layer_sizes) == [4, 6, 2]
            assert dpg.get_value("nt_dt") == saved_dt
            assert app.trainer.capture_state() == state
            dpg.set_value("nt_va_source", "")
            app.on_nt_build()
            assert app.trainer is None
            assert "Verilog-A" in dpg.get_item_configuration("nt_status")["label"]
            app.desktop.device(); app._apply_device_class("FeCAP")
            app.on_open_trainer()
            assert dpg.get_value("center_tabs") == "tab_trainer"
            assert dpg.get_item_configuration("btn_trainer")["enabled"]
            assert not any("[ui] error:" in line for line in app._log_lines)
            print("PASS: one studio, layers, VA binding, surrogate train/test, save/load, stale-source guards, no fallback, navigation")
        finally:
            app.virtuoso.disconnect()
            dpg.destroy_context()


if __name__ == "__main__":
    main()
