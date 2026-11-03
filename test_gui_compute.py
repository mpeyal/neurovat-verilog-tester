"""Exercise real native controls, worker delivery, exports and compact bounds."""
import json
from pathlib import Path
import shutil
import tempfile
import time

import dearpygui.dearpygui as dpg
from vatester.app import App
from gui_test_helpers import frames, activate, check


def run(app):
    app._run_active()
    deadline = time.monotonic()+15
    while app.compute_panel.busy and time.monotonic() < deadline:
        frames(app, 2)
    assert not app.compute_panel.busy, 'Worker did not finish'
    assert dpg.get_item_configuration('compute_configure')['enabled']


def main():
    repo = Path(__file__).resolve().parent
    output = repo/'results/compute_gui'
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='compute_gui_') as tmp:
        root = Path(tmp)
        shutil.copytree(repo/"verilog", root/"verilog")
        shutil.copytree(repo/'twins', root/'twins', ignore=shutil.ignore_patterns('__pycache__'))
        app = App(tmp, bridge=False)
        try:
            app.build(); frames(app)
            activate(app, 'tab_compute')
            assert dpg.get_item_configuration('btn_run')['label'] == 'Run compute'
            panel = app.compute_panel
            assert panel.sources
            dpg.get_item_callback('compute_configure')()
            frames(app)
            assert dpg.is_item_visible('compute_a')
            dpg.set_value('compute_a', '1,-0.5;0.25,1')
            dpg.set_value('compute_x', '1,0;0,1')
            dpg.set_value('compute_levels', 5)
            run(app)
            assert panel.result['actual'] == [[1., -.5], [.25, 1.]]
            assert panel.result['source_binding']['twin_key'] == 'v2'
            # Inspecting the selected source must not import defaults over edits.
            old_rmax = app.param_values['v2']['Rmax']
            app.param_values['v2']['Rmax'] = 9000.
            panel.inspect()
            assert app.param_values['v2']['Rmax'] == 9000.
            assert dpg.get_value('param_model_sel') == panel.specs['v2'].label
            app.param_values['v2']['Rmax'] = old_rmax
            app.rebuild_param_panel()
            stem = panel.export()
            saved = json.loads(stem.with_suffix('.json').read_text())
            assert saved['device_params'] == panel.result['device_params']
            assert len(stem.with_suffix('.csv').read_text().splitlines()) == 5
            # Both sizes, assistant open and closed: neither plot falls below viewport.
            for width, height in ((1600, 980), (1160, 720)):
                dpg.configure_viewport(0, width=width, height=height)
                frames(app); app._on_resize(); frames(app)
                for drawer in (False, True):
                    if dpg.get_item_configuration('right_child')['show'] != drawer:
                        app.desktop.assistant()
                    frames(app)
                    check(app, ['compute_plot_output', 'compute_plot_diagnostic'])
                    assert dpg.is_item_visible('btn_run')
                dpg.output_frame_buffer(str(output/f'matrix_{width}.png')); frames(app)
            # Cancel a larger device mapping and recover without replacing results.
            previous = panel.result
            dpg.set_value('compute_a', '\n'.join([','.join(['1']*64)]*64))
            dpg.set_value('compute_x', '\n'.join(['1']*64))
            panel.start(); panel.stop.set()
            deadline = time.monotonic()+15
            while panel.busy and time.monotonic() < deadline:
                frames(app, 2)
            assert not panel.busy and panel.result is previous
            assert 'stopped' in dpg.get_value('compute_status')
            # Bad input preserves prior completed result, unlocks controls.
            previous = panel.result
            dpg.set_value('compute_a', '1,2;3')
            run(app)
            assert panel.result is previous
            assert 'Cannot run' in dpg.get_value('compute_status')
            # Exercise functional ALU through GUI with carry-out, and async validation failure.
            dpg.set_value('compute_mode', panel.MODES[1]); panel.mode_changed()
            dpg.set_value('compute_levels', 256)
            dpg.set_value('compute_bits', 4)
            run(app)
            assert panel.result['passed'] == panel.result['cases'] == 512
            stem = panel.export()
            assert len(stem.with_suffix('.csv').read_text().splitlines()) == 513
            dpg.set_value('compute_bits', 0)
            previous = panel.result
            run(app)
            assert panel.result is previous and '1-4' in dpg.get_value('compute_status')
            # Integration is a separate input mode, and exported trace retains provenance.
            dpg.set_value('compute_mode', panel.MODES[2]); panel.mode_changed()
            dpg.set_value('compute_trace', 't_s,v_1,i_1\n0,2,3\n1,2,3')
            run(app)
            assert panel.result['net_energy_J'] == 6
            assert 'input_trace_csv' in panel.result
            stem = panel.export()
            assert len(stem.with_suffix('.csv').read_text().splitlines()) == 3
            assert not any('[ui] error:' in line for line in app._log_lines)
            print('PASS: real GUI matrix / 512-case ALU / terminal energy, exports, invalid-input recovery, two viewport sizes with both assistant states')
        finally:
            if app.compute_panel.busy:
                app.compute_panel.stop.set()
                app.compute_panel.worker.join(timeout=5)
            app.virtuoso.disconnect()
            dpg.destroy_context()


if __name__ == '__main__':
    main()
