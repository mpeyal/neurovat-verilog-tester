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

POWER_DIR = "power_profiles"          # optional, in the workspace (data only)


def power_profile_path(family, folder=POWER_DIR):
    """``power_profiles/<family>.json`` for a device family ("ECFET" ->
    ``power_profiles/ecfet.json``), or None for an unnamed family."""
    name = "".join(ch for ch in str(family or "").lower()
                   if ch.isalnum() or ch in "-_")
    return os.path.join(folder, name + ".json") if name else None


def load_power_params(family, folder=POWER_DIR):
    """Energy-model parameters for the Trainer from the selected device
    family's profile ``power_profiles/<family>.json`` (a flat ``name -> value``
    table of PowerParams fields in SI units, plus optional source / corner /
    temperature / notes). There is no shared fallback file, so one family
    never runs on another family's numbers: anything the profile does not set
    stays a flagged placeholder. Returns (params, note)."""
    path = power_profile_path(family, folder)
    if path is None or not os.path.isfile(path):
        where = path or f"{folder}/<family>.json"
        return PowerParams(), f"no {where}: all placeholder values"
    try:
        params = PowerParams().load_file(path)
    except (OSError, ValueError) as exc:
        return PowerParams(), f"{path} ignored ({exc}): all placeholder values"
    meta = params.metadata
    corner = " ".join(str(meta[k]) for k in ("corner", "temperature") if k in meta)
    return params, (f"{path}" + (f" ({corner})" if corner else "")
                    + f": {len(params.sources)} value(s) loaded")


def power_text(trainer, phases=("train", "eval")):
    """Energy breakdown tables of a trainer, as text, headed by the power
    profile they were computed with."""
    names = {"init": "Energy - weight initialisation",
             "train": "Energy - training", "eval": "Energy - evaluation"}
    tables = [format_report(trainer.energy.report(ph), names[ph])
              for ph in phases if ph in trainer.energy.tallies]
    note = getattr(trainer, "power_note", None)
    return "\n\n".join(([f"Power profile: {note}"] if note else []) + tables)


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
