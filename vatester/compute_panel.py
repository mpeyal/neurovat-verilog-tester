"""Native, bounded compute experiment workspace. Workers never access DPG."""
import csv
import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

from . import compute
from .network_studio import source_record


class ComputePanel:
    MODES = ['Matrix multiply', 'Logic / ALU', 'Terminal energy']

    def __init__(self, app, specs):
        self.app = app
        self.specs = {s.key: s for s in specs}
        self.sources = {}
        self.busy = False
        self.result = None
        self.stop = threading.Event()

    def build(self):
        with dpg.group(horizontal=True):
            dpg.add_combo(self.MODES, default_value=self.MODES[0], width=160,
                          tag='compute_mode', callback=self.mode_changed)
            dpg.add_button(label='Configure', tag='compute_configure')
            dpg.add_button(label='Stop', tag='compute_stop', enabled=False,
                           callback=lambda: self.stop.set())
        with dpg.group(horizontal=True):
            dpg.add_button(label='Export results', tag='compute_export', enabled=False, callback=self.export)
            dpg.add_button(label='Analysis / assumptions', tag='compute_details')
        dpg.add_text('Configure inputs, then Run compute (F5).', tag='compute_status', wrap=330)
        dpg.add_text('Device read estimate | ideal state initialization', tag='compute_scope')
        with self.app.desktop.popup('compute_configure', 'compute_config_popup'):
            with dpg.child_window(width=540, height=410):
                with dpg.group(tag='compute_device_settings'):
                    dpg.add_text('Verilog-A source / Python conductance twin')
                    dpg.add_combo([], tag='compute_source', width=-1)
                    dpg.add_button(label='Inspect device parameters', callback=self.inspect)
                    dpg.add_text('Uses the current parameter inspector values for this twin.', wrap=480)
                with dpg.group(tag='compute_matrix_settings'):
                    dpg.add_text('A: weights [outputs x inputs]')
                    dpg.add_input_text(tag='compute_a', multiline=True, height=80, width=-1,
                                       default_value='1, -0.5, 0.25\n-0.25, 0.5, 1')
                    dpg.add_text('X: input vectors as columns [inputs x vectors]')
                    dpg.add_input_text(tag='compute_x', multiline=True, height=80, width=-1,
                                       default_value='1, 0\n0, 1\n1, -1')
                with dpg.group(tag='compute_logic_settings', show=False):
                    dpg.add_combo(['NAND', 'AND', 'OR', 'XOR', 'NOT', 'ADD'], default_value='ADD',
                                  label='Operation', tag='compute_op', width=180)
                    dpg.add_input_int(label='Adder bits (1-4)', tag='compute_bits', default_value=2, width=180)
                    dpg.add_text('All input combinations, including carry-in. ADD includes carry-out.', wrap=480)
                with dpg.group(tag='compute_read_settings'):
                    labels = {'read_voltage': 'Max read voltage (V)', 'read_ns': 'Read duration (ns)',
                              'levels': 'Conductance levels', 'mismatch': 'Mismatch std / G span',
                              'seed': 'Mismatch seed', 'banks': 'Replicated parallel banks',
                              'overhead_ns': 'Per-read overhead (ns)', 'peripheral_pj': 'Peripheral energy (pJ / read)'}
                    for key, value in asdict(compute.Settings()).items():
                        widget = dpg.add_input_int if type(value) is int else dpg.add_input_double
                        widget(label=labels[key], tag='compute_'+key, default_value=value, width=180, step=0)
                    dpg.add_text('One read = one matrix vector or one logic gate evaluation. Banks replicate arrays; they do not benchmark this PC.', wrap=480)
                with dpg.group(tag='compute_trace_settings', show=False):
                    dpg.add_text('CSV: seconds, volts, amps; current positive into each terminal.', wrap=480)
                    dpg.add_input_text(tag='compute_trace', multiline=True, height=180, width=-1,
                                       default_value='t_s,v_1,i_1\n0,0.1,0.001\n1e-7,0.1,0.001\n2e-7,0.1,0.001')
                    dpg.add_input_text(label='CSV path', tag='compute_trace_path', width=360)
                    dpg.add_button(label='Load CSV', callback=self.load_trace)
                    dpg.add_text('Add matched v_2,i_2 columns for more terminals. The initial values are an editable example.', wrap=480)
        with self.app.desktop.popup('compute_details', 'compute_details_popup'):
            with dpg.child_window(width=530, height=360):
                dpg.add_text('Run an experiment to view analysis.', tag='compute_analysis', wrap=490)
                dpg.add_separator()
                dpg.add_text('\n\n'.join(compute.ASSUMPTIONS), tag='compute_assumptions', wrap=490)
        with dpg.subplots(2, 1, tag='compute_plots', width=-1, height=400):
            for suffix, label in [('output', 'Expected vs device result'), ('diagnostic', 'Output error')]:
                with dpg.plot(tag='compute_plot_'+suffix, label=label):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label='Flattened output index', tag='compute_x_'+suffix)
                    dpg.add_plot_axis(dpg.mvYAxis, tag='compute_y_'+suffix)

    def mode_changed(self, *_):
        mode = dpg.get_value('compute_mode')
        for tag, show in [('matrix', mode == self.MODES[0]), ('logic', mode == self.MODES[1]),
                          ('trace', mode == self.MODES[2]), ('device', mode != self.MODES[2]),
                          ('read', mode != self.MODES[2])]:
            dpg.configure_item('compute_'+tag+'_settings', show=show)
        # Keep completed results and their provenance; changing setup does not relabel them.
        dpg.set_value('compute_status', 'Setup changed. Run compute to update the plotted result.')

    def refresh_sources(self):
        self.sources = {str(Path(v.path).relative_to(self.app.workdir)): v for v in self.app.va_files
                        if v.model_key in ('v1', 'v2', 'v3', 'fefet') and not v.error and v.role == 'device'}
        current = dpg.get_value('compute_source')
        if current not in self.sources:
            current = next((n for n, v in self.sources.items() if v.model_key == 'v2'), next(iter(self.sources), ''))
        dpg.configure_item('compute_source', items=list(self.sources), default_value=current)

    def inspect(self, *_):
        va = self.sources.get(dpg.get_value('compute_source'))
        if va:
            self.app.on_va_selected(va.name, interact=False)
            dpg.set_value('param_model_sel', self.specs[va.model_key].label)
            self.app.rebuild_param_panel()
            dpg.hide_item('compute_config_popup')

    def load_trace(self, *_):
        try:
            path = Path(dpg.get_value('compute_trace_path')).expanduser()
            if path.stat().st_size > 4_000_000:
                raise ValueError('CSV is limited to 4 MB.')
            text = path.read_text(encoding='utf-8-sig')
            compute.terminal_energy(text)
            dpg.set_value('compute_trace', text)
        except Exception as exc:
            self.status(f'CSV: {exc}')

    def status(self, text):
        dpg.set_value('compute_status', text)
        self.app.log('[compute] '+text)

    def start(self, *_):
        if self.busy:
            return
        try:
            mode = dpg.get_value('compute_mode')
            provenance = {}
            if mode == self.MODES[2]:
                trace = dpg.get_value('compute_trace')
                job = lambda: compute.terminal_energy(trace)
                provenance = {'input_trace_csv': trace}
            else:
                va = self.sources.get(dpg.get_value('compute_source'))
                if va is None:
                    raise ValueError('Select a supported Verilog-A source in Configure.')
                spec = self.specs[va.model_key]
                params = dict(self.app.param_values[spec.key])
                # Hold the classes as well as values: hot reload must not switch
                # implementation halfway through constructing an array.
                model_cls, params_cls = spec.cls, spec.params_cls
                make = lambda: model_cls(params_cls(**params))
                provenance = {'source_binding': source_record(va.path, va.model_key), 'device_params': params}
                cfg = compute.Settings(**{k: dpg.get_value('compute_'+k) for k in asdict(compute.Settings())})
                cfg.validate()
                if mode == self.MODES[0]:
                    a, x = compute.parse_matrix(dpg.get_value('compute_a')), compute.parse_matrix(dpg.get_value('compute_x'))
                    job = lambda: compute.matrix(make, a, x, cfg, self.stop)
                else:
                    operation, bits = dpg.get_value('compute_op'), dpg.get_value('compute_bits')
                    job = lambda: compute.logic(make, operation, bits, cfg, self.stop)
        except Exception as exc:
            self.status(f'Cannot run: {exc}')
            return
        self.busy = True
        self.stop.clear()
        dpg.configure_item('compute_stop', enabled=True)
        dpg.configure_item('compute_configure', enabled=False)
        dpg.configure_item('compute_mode', enabled=False)
        dpg.hide_item('compute_config_popup')
        self.status('Running '+mode.lower()+'...')

        def worker():
            try:
                result = job()
                if self.stop.is_set():
                    raise InterruptedError('Experiment stopped.')
                result.update(provenance)
                result['created_utc'] = datetime.now(timezone.utc).isoformat()
                if provenance.get('source_binding', {}).get('twin_key') == 'fefet':
                    result['assumptions'].append('Built-in FeFET is approximate; equivalence to this Verilog-A source is unvalidated.')
                self.app.q.put(('compute', result, None))
            except Exception as exc:
                self.app.q.put(('compute', None, str(exc)))
        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def finished(self, result, error):
        self.busy = False
        dpg.configure_item('compute_stop', enabled=False)
        dpg.configure_item('compute_configure', enabled=True)
        dpg.configure_item('compute_mode', enabled=True)
        if error:
            self.status(error+' Previous completed result retained.' if self.result else error)
            return
        self.result = result
        dpg.configure_item('compute_export', enabled=True)
        dpg.set_value('compute_assumptions', '\n\n'.join(result['assumptions']))
        if result['kind'] == 'terminal_energy':
            analysis = '\n'.join(f'Terminal {name}: {energy:.6g} J' for name, energy in result['terminal_energy_J'].items())
            analysis += f"\nDuration: {result['duration_s']:.6g} s\nPeak signed power: {result['peak_power_W']:.6g} W"
        else:
            rate = result.get('vectors_per_second', result.get('cases_per_second'))
            analysis = (f"Array read energy: {result['array_energy_J']:.6g} J\n"
                        f"Estimated peripherals: {result['peripheral_energy_J']:.6g} J\n"
                        f"Accounted total energy: {result['accounted_energy_J']:.6g} J\n"
                        f"Average accounted power: {result['average_accounted_power_W']:.6g} W\n"
                        f"Replicated banks used: {result['effective_banks']}\n"
                        f"Modeled throughput: {rate:.6g} vectors or cases/s\n"
                        f"Scheduling speedup: {result['scheduling_speedup']:.6g}x\n")
            if result['kind'] == 'logic':
                analysis += f"NAND gates: {result['gate_count']} | Depth: {result['depth']}\nMinimum absolute comparator margin: {result['min_comparator_margin']:.6g}\n"
            else:
                analysis += f"Max absolute error: {result['max_abs_error']:.6g}\n"
            analysis += f"Source: {result['source_binding']['path']}\nSHA256: {result['source_binding']['sha256']}"
        dpg.set_value('compute_analysis', analysis)
        for suffix in ('output', 'diagnostic'):
            dpg.delete_item('compute_y_'+suffix, children_only=True)
        if result['kind'] == 'terminal_energy':
            t = result['time_s']
            self.plot('output', 'Terminal power', 'Time (s)', 'Power (W)', [(t, result['power_W'], 'Sum V*I')])
            self.plot('diagnostic', 'Net terminal energy', 'Time (s)', 'Energy (J)', [(t, result['cumulative_energy_J'], 'Integral V*I dt')])
            summary = f"Net energy {result['net_energy_J']:.4g} J | Average power {result['average_power_W']:.4g} W"
            scope = 'Supplied terminal trace | net energy, including storage / return'
        else:
            actual, expected = np.array(result['actual']).ravel(), np.array(result['expected']).ravel()
            index = list(range(len(actual)))
            xlabel = 'Output index (row-major)' if result['kind'] == 'matrix' else 'Truth table case index'
            self.plot('output', 'Expected vs device result', xlabel, 'Decoded output',
                      [(index, expected.tolist(), 'Expected'), (index, actual.tolist(), 'Device result')])
            self.plot('diagnostic', 'Output error', xlabel, 'Device - expected', [(index, (actual-expected).tolist(), 'Error')])
            score = f"RMSE {result['rmse']:.4g}" if result['kind'] == 'matrix' else f"{result['passed']} / {result['cases']} cases passed"
            summary = f"{score} | Array {result['array_energy_J']*1e12:.4g} pJ\nEstimated batch latency {result['estimated_latency_s']*1e9:.4g} ns | {result['physical_devices']} devices"
            scope = f"{result['kind'].title()} / {Path(result['source_binding']['path']).name} | ideal read estimate"
        dpg.set_value('compute_scope', scope)
        self.status(summary)

    @staticmethod
    def plot(suffix, title, xlabel, ylabel, series):
        dpg.configure_item('compute_plot_'+suffix, label=title)
        dpg.configure_item('compute_x_'+suffix, label=xlabel)
        dpg.configure_item('compute_y_'+suffix, label=ylabel)
        for x, y, label in series:
            widget = dpg.add_scatter_series if label in ('Device result', 'Error') else dpg.add_line_series
            widget(x, y, label=label, parent='compute_y_'+suffix)
        dpg.fit_axis_data('compute_x_'+suffix)
        dpg.fit_axis_data('compute_y_'+suffix)

    def export(self, *_):
        if self.result is None:
            return
        try:
            r = self.result
            folder = Path(self.app.workdir)/'results'/'compute'
            folder.mkdir(parents=True, exist_ok=True)
            stem = folder/(r['kind']+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            stem.with_suffix('.json').write_text(json.dumps(r, indent=2, allow_nan=False), encoding='utf-8')
            with stem.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                if r['kind'] == 'terminal_energy':
                    writer.writerow(['t_s', 'power_W', 'net_energy_J'])
                    writer.writerows(zip(r['time_s'], r['power_W'], r['cumulative_energy_J']))
                elif r['kind'] == 'matrix':
                    writer.writerow(['output_row', 'vector_column', 'expected', 'actual', 'error', 'current_A'])
                    for i, row in enumerate(r['actual']):
                        for j, value in enumerate(row):
                            writer.writerow([i, j, r['expected'][i][j], value, r['error'][i][j], r['current_A'][i][j]])
                else:
                    writer.writerow(['a', 'b', 'carry_in', 'expected', 'actual', 'pass', 'array_energy_J'])
                    for inputs, expected, actual, energy in zip(r['inputs'], r['expected'], r['actual'], r['array_energy_per_case_J']):
                        writer.writerow([*inputs, expected, actual, expected == actual, energy])
            self.status('Exported JSON + CSV to results/compute. Full paths in Log.')
            self.app.log(str(stem))
            return stem
        except Exception as exc:
            self.status(f'Export failed: {exc}')
