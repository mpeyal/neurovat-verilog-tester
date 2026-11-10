"""Neuromorphic network engine - spiking neurons trained on a memristive
crossbar whose synapses ARE the device twins (ECFET / FeFET / ...).

This is the simulation behind the GUI's *Neuromorphic Trainer* canvas. The
engine itself lives in the GUI-free ``neurovat`` package (``src/neurovat``);
this module keeps the names the app, the tools and the tests import, and binds
the engine to the GUI's live, hot-reloadable scalar device twins: every
cross-point (i, j) is an INDEPENDENT instance of the selected twin, so the
device's write nonlinearity, soft bounds, retention and cycle-to-cycle noise
shape the weights.
"""
import os
import sys

try:
    import neurovat
except ModuleNotFoundError as e:          # running from a checkout, not installed
    if e.name != "neurovat":
        raise
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src"))
    import neurovat

from neurovat.config import NeuronParams, STDPParams, NetConfig
from neurovat.patterns import (make_patterns, _fit, _letter_glyphs,
                               _digit_glyphs, _bits)
from neurovat.metrics import confusion, prf1
from neurovat.devices import (ScalarTwinArray, device_weight,
                              calibrate_pot_sign, twin_factory)
from neurovat.lif import (probe_lif, probe_lif_wave, fi_curve, probe_lif_psp,
                          fi_curve_psp)
from neurovat.network import CrossbarSNN
from neurovat.power import PowerParams, format_report

POWER_FILE = "power_params.json"      # optional, in the workspace (data only)


def load_power_params(path=POWER_FILE):
    """Energy-model parameters for the Trainer: the values in the workspace
    file ``power_params.json`` (a flat ``name -> value`` table of PowerParams
    fields, SI units) when it exists; everything else stays a flagged
    placeholder. Returns (params, note)."""
    if not os.path.isfile(path):
        return PowerParams(), f"no {path}: all placeholder values"
    try:
        params = PowerParams().load_file(path)
    except (OSError, ValueError) as exc:
        return PowerParams(), f"{path} ignored ({exc}): all placeholder values"
    return params, f"{path}: {len(params.sources)} value(s) loaded"


def power_text(trainer, phases=("train", "eval")):
    """Energy breakdown tables of a trainer, as text."""
    names = {"init": "Energy - weight initialisation",
             "train": "Energy - training", "eval": "Energy - evaluation"}
    return "\n\n".join(format_report(trainer.energy.report(ph), names[ph])
                       for ph in phases if ph in trainer.energy.tallies)


class Trainer(CrossbarSNN):
    """Crossbar(s) of scalar device twins + LIF layers.  Build once, then call
    train_step() per stimulus; weights_uS() reads the live conductances."""

    def __init__(self, make_device, input_kind, neuron, stdp, cfg,
                 patterns=None, targets=None, power=None):
        self.make_device = make_device
        self.input_kind = input_kind          # "current" | "voltage"
        super().__init__(twin_factory(make_device), neuron, stdp, cfg,
                         patterns=patterns, targets=targets, power=power)

    @property
    def crossbars(self):
        """crossbar c as an (size[c+1] x size[c]) grid of device twins."""
        return [a.devices for a in self.arrays]

    @property
    def syn(self):
        return self.arrays[0].devices         # back-compat alias (first array)
