"""Load the MNIST checkpoint into the real desktop app and run its Test worker."""
import dataclasses
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import dearpygui.dearpygui as dpg
from vatester.app import App, SPEC_BY_KEY
from tools.mnist_ecfet_v2 import OUT, raster


def main():
    repo = Path(__file__).resolve().parents[1]
    expected = json.loads((OUT/"comparison.json").read_text())
    settings = expected["settings"]
    with np.load(OUT/"prepared_data.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(OUT/"evaluation_spikes.npz") as archive:
        banks = {key: archive[key] for key in archive.files}
    lookup = {np.asarray(vec, np.float32).tobytes(): bank[i]
              for split, bank in banks.items() for i, vec in enumerate(data[f"{split}_x"])}
    with tempfile.TemporaryDirectory(prefix="mnist_desktop_") as temp:
        root = Path(temp)
        shutil.copytree(repo/"verilog", root/"verilog")
        shutil.copytree(repo/"twins", root/"twins", ignore=shutil.ignore_patterns("__pycache__"))
        app = App(temp, bridge=False)
        try:
            app.build()
            app.param_values["v2"].update(settings["device"])
            for tag, value in (("nt_device", SPEC_BY_KEY["v2"].label), ("nt_gh", 14),
                               ("nt_gw", 14), ("nt_nout", 10), ("nt_present", 60.),
                               ("nt_epochs", 8), ("nt_seed", 17), ("nt_rate", 160.),
                               ("nt_hidden", ""), ("nt_learnrule", "Surrogate grad (BPTT)")):
                dpg.set_value(tag, value)
            app._nt_loaded_patterns = ([(str(y), x) for x, y in zip(data["train_x"], data["train_y"])], data["train_y"].tolist())
            app.on_nt_build()
            tr = app.trainer
            assert tr is not None, app._log_lines[-5:]
            for obj, key in ((tr.cfg, "cfg"), (tr.N, "neuron"), (tr.S, "stdp")):
                assert json.loads(json.dumps(dataclasses.asdict(obj))) == settings[key], (key, dataclasses.asdict(obj))
            for split in ("validation", "test"):
                setattr(tr, f"{split}_patterns", [(str(y), x) for x, y in zip(data[f"{split}_x"], data[f"{split}_y"])])
                setattr(tr, f"{split}_targets", data[f"{split}_y"].tolist())
            tr.restore_state(json.loads((OUT/"selected_app_state.json").read_text()))
            # The backend received explicit MNIST patterns, so its unused preset
            # label is not the data source. Label the exported model correctly.
            tr.cfg.pattern_set = "dataset"
            dpg.set_value("nt_patset", "dataset")
            dpg.set_value("nt_ds_res", 14)
            dpg.set_value("nt_ds_perclass", 100)
            state = tr.capture_state()
            model_path = str(OUT/"mnist_ecfet_v2_model.json")
            app.on_nt_save(model_path)
            assert Path(model_path).is_file()
            app.on_nt_load(model_path)
            tr = app.trainer
            assert tr.capture_state() == state, "Saved checkpoint did not round-trip"
            original_encode = tr.encode_spikes
            tr.encode_spikes = lambda vec, *a, **k: lookup[np.asarray(vec, np.float32).tobytes()]
            app.on_nt_test()
            deadline = time.monotonic()+180
            while app.trainer_running and time.monotonic() < deadline:
                app._process_queue(); app.desktop.tick(); dpg.render_dearpygui_frame()
                time.sleep(.01)
            app._process_queue()
            assert not app.trainer_running, "Desktop evaluation timed out"
            metrics = app._nt_last_metrics
            assert metrics["tr_acc"] == expected["final"]["train"]["app_accuracy"], metrics
            assert metrics["te_acc"] == expected["final"]["test"]["app_accuracy"], metrics
            mismatches = 0
            for example in expected["final"]["test"]["examples"].values():
                result = tr.infer(data["test_x"][example["split_index"]])
                mismatches += int(np.count_nonzero(raster(result) != np.asarray(example["app_spikes"])))
            assert mismatches == 0
            tr.encode_spikes = original_encode
            app.on_open_trainer()
            node = dpg.get_item_parent("nt_metrics_summary")
            while node:
                if dpg.get_item_type(node).endswith("::mvTab"):
                    dpg.set_value(dpg.get_item_parent(node), node)
                node = dpg.get_item_parent(node)
            for _ in range(8):
                app.desktop.tick(); app._nt_tick_layout(); dpg.render_dearpygui_frame()
            dpg.output_frame_buffer(str(OUT/"desktop_evaluation.png"))
            for _ in range(8):
                app.desktop.tick(); dpg.render_dearpygui_frame()
            summary = {"passed": True, "scope": "Actual DearPyGui build, save, reload and asynchronous Test worker; saved replay spikes injected for exact comparison. Full training ran through app backend, not GUI Train button.",
                       "train_accuracy": metrics["tr_acc"], "test_accuracy": metrics["te_acc"],
                       "checkpoint_round_trip": True, "example_raster_mismatches": mismatches,
                       "model": Path(model_path).name,
                       "note": "The saved model uses normal app encoding when opened manually. For these exact metrics use the replay-bank verification script; a new Poisson realization can change accuracy."}
            (OUT/"desktop_verification.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary, indent=2))
        finally:
            app.virtuoso.disconnect()
            dpg.destroy_context()


if __name__ == "__main__":
    main()
