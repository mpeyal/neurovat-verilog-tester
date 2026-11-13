"""Fixed-bias FeCAP noise/DR canvas. All UI calls stay on the render thread."""
import csv
import dataclasses
import json
import math
import os

import dearpygui.dearpygui as dpg
import numpy as np


def noise_curves(model, voltage, state, points=240):
    """Numerical data shared by the canvas, export and local inspector."""
    p = model.p
    if not all(math.isfinite(x) for x in (voltage, state, p.f_lo, p.f_hi)):
        raise ValueError('Read conditions must be finite.')
    if not -1 <= state <= 1:
        raise ValueError('Polarization state must be between -1 and +1.')
    if not 0 < p.f_lo < p.f_hi:
        raise ValueError('Require 0 < low frequency < high frequency.')
    if not p.dr_u_lo < p.dr_u_hi:
        raise ValueError('Achievable lower endpoint must be below upper endpoint.')
    freq = np.geomspace(p.f_lo, p.f_hi, points)
    spectrum = model.noise_psd(freq, voltage=voltage, ubar=state)
    # Avoid the zero-bandwidth endpoint: its DR is mathematically undefined.
    cutoffs = np.geomspace(p.f_lo * (p.f_hi / p.f_lo)**(1/(points-1)), p.f_hi, points)
    dr, span, total, rms = [], [], [], []
    for upper in cutoffs:
        p.f_hi = float(upper)
        b = model.noise_budget(voltage=voltage, ubar=state)
        dr.append(b['dr_db'] if b['dr_valid'] else None)
        span.append(b['dr_span_db'] if b['dr_valid'] else None)
        total.append(b['dr_total_db'] if b['dr_total_valid'] else None)
        rms.append(b['pnoise_rms'])
    p.f_hi = float(freq[-1])
    budget = model.noise_budget(voltage=voltage, ubar=state)
    return dict(frequency_hz=freq.tolist(),
                psd={k: np.asarray(v).tolist() for k, v in spectrum.items()},
                upper_cutoff_hz=cutoffs.tolist(), dr_db=dr, dr_span_db=span,
                dr_total_db=total, noise_rms=rms, budget=budget,
                voltage_v=float(voltage), state=float(state),
                parameters=dataclasses.asdict(p),
                interpretation='Fixed-bias ideal charge sensing; one-sided polarization-equivalent PSD.')


class NoisePanel:
    def __init__(self, app, specs):
        self.app = app
        self.specs = {s.key: s for s in specs if hasattr(s.cls, 'noise_budget')
                      and hasattr(s.cls, 'noise_psd')}
        self.data = None
        self._signature = None

    def build(self):
        with dpg.group():
            dpg.add_text('Polarization noise & dynamic range')
            dpg.add_text('Source: checked model in the left panel', tag='nplot_model_label', wrap=560)
            with dpg.group(horizontal=True):
                self._field('nplot_vread', 'Read bias (V)', 0.)
                dpg.add_button(label='Plot / refresh', tag='nplot_run', callback=self.refresh)
                dpg.add_button(label='Edit model noise parameters', callback=self.edit_parameters)
                dpg.add_button(label='Export CSV + JSON', callback=self.export)
            dpg.add_text('Check a model with noise support in the left panel.', tag='nplot_status', wrap=560)
            for name in ('psd', 'dr'):
                dpg.add_text('', tag='nplot_'+name+'_notice', wrap=650,
                             color=(240, 190, 100), show=False)
            with dpg.subplots(2, 1, tag='noise_plots', width=-1, height=-1):
                for name, label, ylabel in (
                    ('psd', 'Noise spectrum', 'P-equivalent PSD [(uC/cm^2)^2/Hz]'),
                    ('dr', 'Dynamic range vs upper read-band cutoff', 'Dynamic range (dB)')):
                    with dpg.plot(label=label, tag='nplot_'+name):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, tag='nplot_'+name+'_x',
                                          label='Frequency (Hz)' if name == 'psd' else 'Upper cutoff (Hz); lower cutoff fixed',
                                          scale=dpg.mvPlotScale_Log10)
                        dpg.add_plot_axis(dpg.mvYAxis, tag='nplot_'+name+'_y', label=ylabel,
                                          scale=dpg.mvPlotScale_Log10 if name == 'psd' else dpg.mvPlotScale_Linear)
            with dpg.tooltip('nplot_run'):
                dpg.add_text('Noise coefficients, frequency band and Pinit come from model parameters.\nZero noise uses a linear axis; undefined DR has no finite curve.', wrap=420)

    def _field(self, tag, label, default):
        with dpg.group():
            dpg.add_text(label)
            dpg.add_input_double(tag=tag, default_value=default, width=125,
                                 format='%.5g', step=0, callback=self.refresh)

    def edit_parameters(self, *_):
        try:
            spec = self._spec()
            dpg.set_value('param_model_sel', spec.label)
            self.app.rebuild_param_panel()
            dpg.set_y_scroll('param_panel', dpg.get_y_scroll_max('param_panel'))
        except Exception as exc:
            self._error(exc)

    def _notice(self, name, text):
        tag = 'nplot_'+name+'_notice'
        dpg.set_value(tag, text)
        dpg.configure_item(tag, show=bool(text))

    def _spec(self):
        enabled = list(self.app._enabled_keys())
        supported = [key for key in enabled if key in self.specs]
        if not supported:
            raise ValueError('The checked model has no noise model. Check a model whose twin defines one in the left panel.'
                             if enabled else 'Check a model in the left panel to plot noise and dynamic range.')
        if len(supported) > 1:
            raise ValueError('Check one noise-capable model at a time to plot its noise budget.')
        return self.specs[supported[0]]

    def tick(self):
        if dpg.get_value('center_tabs') != 'tab_noise':
            return
        keys = tuple(self.app._enabled_keys())
        signature = (keys, tuple((k, tuple(self.app.param_values[k].items()))
                                  for k in keys if k in self.specs), dpg.get_value('nplot_vread'))
        if signature != self._signature:
            self._signature = signature
            self.refresh()

    def calculate(self):
        self.data = None
        for axis in ('nplot_psd_y', 'nplot_dr_y'):
            dpg.delete_item(axis, children_only=True)
        dpg.set_value('nplot_model_label', 'Source: checked model in the left panel')
        s = self._spec()
        values = dict(self.app.param_values[s.key])
        m = s.cls(s.params_cls(**values))
        self.data = noise_curves(m, dpg.get_value('nplot_vread'), values.get('Pinit', 0.))
        self.data.update(model_key=s.key)
        skipped = [key for key in self.app._enabled_keys() if key not in self.specs]
        source = 'Source: '+s.label
        if skipped:
            source += ' | Other checked models have no noise support.'
        dpg.set_value('nplot_model_label', source)
        self._draw()
        return self.data

    def refresh(self, *_):
        try:
            self.calculate()
        except Exception as exc:
            self._error(exc)

    def _error(self, exc):
        self.data = None
        for axis in ('nplot_psd_y', 'nplot_dr_y'):
            dpg.delete_item(axis, children_only=True)
        dpg.set_value('nplot_status', str(exc))
        self._notice('psd', str(exc))
        self._notice('dr', str(exc))
        self.app.log(f'[noise] {exc}')

    def _draw(self):
        data = self.data
        zero_noise = not any(y > 0 for y in data['psd']['total'])
        dpg.configure_item('nplot_psd_y', scale=dpg.mvPlotScale_Linear if zero_noise else dpg.mvPlotScale_Log10)
        for ax in ('psd_x', 'psd_y', 'dr_x', 'dr_y'):
            dpg.set_axis_limits_auto('nplot_'+ax)
        self._notice('psd', 'Model predicts zero noise with the current parameters.' if zero_noise else '')
        self._notice('dr', 'No finite device DR: the model noise floor is zero.\n'
                     'Edit the model noise parameters (p_lor_sigma, si_white, kf_i or loss_ratio) using measured values.'
                     if not data['budget']['dr_valid'] else '')
        if zero_noise:
            dpg.add_line_series(data['frequency_hz'], data['psd']['total'],
                                label='Total noise = 0', parent='nplot_psd_y', tag='nplot_series_total')
        for key, label in [('total', 'Total'), ('current', 'Leakage / shunt'),
                           ('dielectric', 'Dielectric thermal'), ('polarization', 'Polarization excess')]:
            pairs = [(x,y) for x,y in zip(data['frequency_hz'], data['psd'][key]) if y > 0 and math.isfinite(y)]
            if pairs:
                x,y = zip(*pairs)
                dpg.add_line_series(list(x), list(y), label=label, parent='nplot_psd_y', tag='nplot_series_'+key)
        for key, label in [('dr_db', 'Max P / noise'), ('dr_span_db', 'Full span / noise'),
                           ('dr_total_db', 'Max P / total uncertainty')]:
            pairs = [(x,y) for x,y in zip(data['upper_cutoff_hz'], data[key]) if y is not None and math.isfinite(y)]
            if pairs:
                x,y = zip(*pairs)
                dpg.add_line_series(list(x), list(y), label=label, parent='nplot_dr_y', tag='nplot_series_'+key)
        for ax in ('psd_x', 'psd_y', 'dr_x', 'dr_y'):
            dpg.fit_axis_data('nplot_'+ax)
        if not data['budget']['dr_total_valid']:
            dpg.set_axis_limits('nplot_dr_x', data['frequency_hz'][0], data['frequency_hz'][-1])
            dpg.set_axis_limits('nplot_dr_y', 0, 1)
        b = data['budget']
        if not b['dr_valid']:
            text = 'No device noise floor configured: device DR is undefined.'
        else:
            text = f"Read noise: {b['pnoise_rms']:.5g} uC/cm^2 | Max P: {b['pmax_mon']:.4g} | DR: {b['dr_db']:.2f} dB"
        note = 'User-marked calibrated parameters; verify this read condition' if b['dr_ready'] else 'Model parameters: not marked calibrated'
        dpg.set_value('nplot_status', note+'\n'+text)

    def export_data(self):
        self.calculate()
        if self.data is None:
            raise ValueError('Plot valid noise data before exporting.')
        folder = os.path.join(self.app.workdir, 'results')
        os.makedirs(folder, exist_ok=True)
        base = os.path.join(folder, 'fecap_noise_dr')
        with open(base+'.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, allow_nan=False)
        with open(base+'_psd.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            keys = ['total', 'current', 'dielectric', 'polarization']
            writer.writerow(['frequency_Hz']+[k+'_P_PSD_(uC_cm2)^2_Hz' for k in keys])
            writer.writerows(zip(self.data['frequency_hz'], *(self.data['psd'][k] for k in keys)))
        with open(base+'_dr.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            keys = ['upper_cutoff_hz', 'dr_db', 'dr_span_db', 'dr_total_db', 'noise_rms']
            writer.writerow(keys)
            writer.writerows(zip(*(self.data[k] for k in keys)))
        return {'json': base+'.json', 'psd_csv': base+'_psd.csv', 'dr_csv': base+'_dr.csv'}

    def export(self, *_):
        try:
            paths = self.export_data()
            self.app.log('[noise] exported '+paths['json'])
            dpg.set_value('nplot_status', dpg.get_value('nplot_status')+'\nExported to results/fecap_noise_dr*.')
        except Exception as exc:
            self.app.log(f'[noise] export: {exc}')
