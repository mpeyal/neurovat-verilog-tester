"""Transient and synaptic-characteristic plotting."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

# When False (default), figures are saved to file and closed immediately.
# Call enable_show() before plotting to keep them open, then show() to
# display all of them in interactive matplotlib windows.
SHOW = False


def enable_show():
    global SHOW
    SHOW = True


def show():
    if SHOW:
        plt.show()


def _shade_pulses(ax, waveform):
    if waveform is None:
        return
    for t0, t1 in waveform.pulse_windows():
        ax.axvspan(t0, t1, color="0.85", zorder=0)


def plot_transient(results, path, title="ECFET transient", show_extras=False,
                   logx=False, t_unit="s"):
    """results: single SimResult or list of SimResult (overlaid)."""
    if not isinstance(results, (list, tuple)):
        results = [results]

    scale = {"s": 1.0, "ms": 1e3, "us": 1e6}[t_unit]
    n_rows = 3 + (1 if show_extras else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.6 * n_rows),
                             sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    ax_i, ax_r, ax_g = axes[0], axes[1], axes[2]
    wf = results[0].waveform

    for ax in axes:
        _shade_pulses(ax, wf)
        ax.grid(True, alpha=0.3)

    ax_i.step(results[0].t * scale, results[0].i_gate * 1e12,
              where="post", color="k", lw=1.0)
    ax_i.set_ylabel("I_gate (pA)")

    for r, c in zip(results, COLORS):
        ax_r.plot(r.t * scale, r.R, color=c, lw=1.4, label=r.label)
        ax_g.plot(r.t * scale, r.G * 1e6, color=c, lw=1.4, label=r.label)
    ax_r.set_ylabel("R_mem (Ω)")
    ax_g.set_ylabel("G (µS)")
    if len(results) > 1:
        ax_r.legend(loc="best", fontsize=9)

    if show_extras:
        ax_e = axes[3]
        for r, c in zip(results, COLORS):
            for j, (key, arr) in enumerate(r.extras.items()):
                if "drift" in key or "diffusing" in key:
                    continue
                ax_e.plot(r.t * scale, arr,
                          lw=1.1, ls=["-", "--", ":"][j % 3],
                          label=f"{r.label}: {key}")
        ax_e.set_ylabel("internals")
        ax_e.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel(f"time ({t_unit})")
    if logx:
        for ax in axes:
            ax.set_xscale("log")

    fig.savefig(path, dpi=130)
    if not SHOW:
        plt.close(fig)
    return path


def plot_ltp_ltd(result, pulse_ends, n_each, path, settle=None,
                 title="LTP / LTD characteristic"):
    """Conductance vs pulse number, sampled `settle` seconds after each pulse
    (defaults to just before the next pulse, i.e. retained value)."""
    pulse_ends = np.asarray(pulse_ends)
    if settle is None:
        # sample just before the next pulse starts (retained value)
        gap = np.min(np.diff(pulse_ends)) if len(pulse_ends) > 1 else 1e-3
        settle = 0.5 * gap
    sample_t = pulse_ends + settle
    _, G = result.at(sample_t)
    n = np.arange(1, len(G) + 1)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(n[:n_each], G[:n_each] * 1e6, "o-", color=COLORS[0],
            label="LTP (G rising)")
    ax.plot(n[n_each:], G[n_each:] * 1e6, "s-", color=COLORS[1],
            label="LTD (G falling)")
    ax.set_xlabel("pulse #")
    ax.set_ylabel("G (µS)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=130)
    if not SHOW:
        plt.close(fig)

    # simple linearity metrics
    dG_ltp = np.diff(G[:n_each])
    dG_ltd = np.diff(G[n_each:])
    info = {
        "G_range_uS": (float(G.min() * 1e6), float(G.max() * 1e6)),
        "mean_dG_ltp_uS": float(np.mean(dG_ltp) * 1e6) if len(dG_ltp) else 0.0,
        "mean_dG_ltd_uS": float(np.mean(dG_ltd) * 1e6) if len(dG_ltd) else 0.0,
        "asymmetry": float(abs(np.mean(dG_ltp) + np.mean(dG_ltd)) /
                           (abs(np.mean(dG_ltp)) + 1e-30)) if len(dG_ltp) and len(dG_ltd) else 0.0,
    }
    return path, info
