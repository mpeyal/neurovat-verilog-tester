"""File-only plotting of captured GUI results, without a second GUI toolkit."""
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .signal_factory import UNIT_SCALE


def result_figure(results, profiles, unit, kind):
    if not results:
        raise ValueError('No completed results to export.')
    figure = Figure(figsize=(11, 2.8*(1+len(profiles))), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(1+len(profiles), 1, sharex=True, squeeze=False)[:, 0]
    figure.suptitle('NeuroVAT - completed device simulation')
    first = results[0]
    axes[0].step(first.t, first.i_gate/UNIT_SCALE[unit], where='post', label='Stimulus')
    axes[0].set_ylabel(f'{"I_gate" if kind == "current" else "V_gate"} ({unit})')
    for axis, (observable, label, units, scale) in zip(axes[1:], profiles):
        for result in results:
            values = result.R if observable == 'R' else result.G if observable == 'G' else result.extras[observable]
            axis.plot(result.t, values*scale, label=result.label)
        axis.set_ylabel(f'{label} ({units})')
        axis.legend(fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=.25)
    axes[-1].set_xlabel('Time (s)')
    return figure
