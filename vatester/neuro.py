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
from neurovat.encoding import input_size
from neurovat.power import PowerParams, format_report, _eng as fmt_si

POWER_DIR = "power_profiles"          # optional, in the workspace (data only)


def power_profile_path(family, folder=POWER_DIR):
    """``power_profiles/<family>.json`` for a device family ("ECFET" ->
    ``power_profiles/ecfet.json``), or None for an unnamed family."""
    name = "".join(ch for ch in str(family or "").lower()
                   if ch.isalnum() or ch in "-_")
    return os.path.join(folder, name + ".json") if name else None


PROFILE_DEFAULT = "Default (match device)"
PROFILE_NONE = "Placeholders only"
DEFAULT_PROFILE = "default"            # power_profiles/default.json, optional


def list_power_profiles(folder=POWER_DIR):
    """Names of the profiles in ``power_profiles/`` (``ecfet.json`` ->
    "ecfet"), sorted."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(n[:-5] for n in names
                  if n.lower().endswith(".json")
                  and os.path.isfile(os.path.join(folder, n)))


def load_power_params(family, folder=POWER_DIR, choice=None):
    """Energy-model parameters for the Trainer. Returns (params, note).

    ``choice`` is a profile name from :func:`list_power_profiles` (use that
    file whatever the device), :data:`PROFILE_NONE` (no file: every value a
    flagged placeholder), or None / :data:`PROFILE_DEFAULT`: the selected
    device family's own profile ``power_profiles/<family>.json``, else the
    user's ``power_profiles/default.json``, else placeholders. A profile is a
    flat ``name -> value`` table of PowerParams fields in SI units plus
    optional source / corner / temperature / notes. Anything a profile does
    not set stays a flagged placeholder."""
    if choice == PROFILE_NONE:
        return PowerParams(), "placeholders only (no profile): all placeholder values"
    if choice and choice != PROFILE_DEFAULT:
        path = power_profile_path(choice, folder)
        if path is None or not os.path.isfile(path):
            return PowerParams(), f"no {path}: all placeholder values"
    else:
        path = power_profile_path(family, folder)
        if path is None or not os.path.isfile(path):
            path = power_profile_path(DEFAULT_PROFILE, folder)
            if not os.path.isfile(path):
                where = power_profile_path(family, folder) or f"{folder}/<family>.json"
                return PowerParams(), (f"no {where} and no {path}: "
                                       "all placeholder values")
    try:
        params = PowerParams().load_file(path)
    except (OSError, ValueError) as exc:
        return PowerParams(), f"{path} ignored ({exc}): all placeholder values"
    meta = params.metadata
    corner = " ".join(str(meta[k]) for k in ("corner", "temperature") if k in meta)
    return params, (f"{path}" + (f" ({corner})" if corner else "")
                    + f": {len(params.sources)} value(s) loaded")


POWER_PHASES = ("init", "train", "eval")
PHASE_NAMES = {"init": "Initialisation", "train": "Training",
               "eval": "Evaluation"}
# (label, group, key) of every energy component, in report order; group None
# = top-level line, else the line it is part of
POWER_COMPONENTS = (
    ("array read", None, "array_read"),
    ("array write", None, "array_write"),
    ("neuron spikes", None, "neurons"),
    ("peripherals", None, "peripherals"),
    ("row drivers", "peripherals", "row_driver"),
    ("column sense", "peripherals", "column_sense"),
    ("WTA / inhibition", "peripherals", "wta"),
    ("static", None, "static"),
    ("neuron leakage", "static", "neuron_leak"),
    ("peripheral static", "static", "peripheral_static"),
)


def power_component(rep, group, key):
    """Energy (J) of one POWER_COMPONENTS entry in a report."""
    if group is None:
        return rep["energy_J"][key]
    return rep[group + "_J"][key]


def power_reports(trainer, phases=POWER_PHASES):
    """{phase: energy report} for the phases the trainer has run."""
    return {ph: trainer.energy.report(ph, parameters=False)
            for ph in phases if ph in trainer.energy.tallies}


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
