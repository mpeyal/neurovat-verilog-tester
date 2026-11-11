"""Component-level energy and power model of a crossbar spiking network.

Energy is accounted per component, per device and per phase ("init", "train",
"eval") while the network runs:

  array read    V_read^2 * G_ji * t_read for every device on a row, each time
                that row's input neuron spikes (one synaptic operation, SOP)
  array write   per programming pulse, from the pulse amplitude and width and
                the current through the device (see :meth:`EnergyMeter.write_energy`)
  neurons       e_spike per LIF spike
  peripherals   row driver (per row-drive), column sense / ADC / integrator
                (per column, per time step with an active row), WTA /
                inhibition (per inhibition event) - reported separately
  static        neuron leakage power and peripheral static power x time

The meter stores parameter-independent sums (conductance x spikes, pulse
amplitude x width, spike and operation counts), so the energies can be
re-evaluated after changing a :class:`PowerParams` value - e.g. after loading
neuron values extracted from a circuit simulation - without re-simulating.

No parameter has a silent default: every value that was not set in the config
or loaded from a file falls back to :data:`PLACEHOLDERS`, and every report
lists those names under ``"placeholders"``.
"""
import csv
import json
import math
import os
from dataclasses import dataclass, field, fields

import numpy as np

#: name -> (placeholder value, SI unit, meaning). Illustrative order-of-
#: magnitude values, NOT measurements; used only when a value is not supplied,
#: and always flagged in the report.
PLACEHOLDERS = {
    "v_read": (0.1, "V", "read voltage across a device on an active row"),
    "t_read": (100e-9, "s", "read pulse width per input spike"),
    "v_write": (1.0, "V", "compliance voltage of a current-driven write pulse"),
    "e_spike": (1e-12, "J", "neuron energy per output spike"),
    "p_leak": (1e-9, "W", "static leakage power per neuron"),
    "e_row_driver": (0.1e-12, "J", "row driver energy per row-drive"),
    "e_col_sense": (1e-12, "J", "column sense / ADC / integrator energy per "
                                "column per active time step"),
    "e_wta": (0.5e-12, "J", "WTA / lateral-inhibition energy per event"),
    "p_peripheral_static": (0.0, "W", "static power of all peripherals"),
}
#: optional values with a derived (not placeholder) fallback
DERIVED = {
    "write_current": ("A", "fixed current of a voltage-driven write pulse",
                      "device model"),
    "timestep_s": ("s", "hardware duration of one simulation step",
                   "network.dt_ms"),
}
NEURON_FIELDS = ("e_spike", "p_leak")
_ALIASES = {"e_spike_j": "e_spike", "energy_per_spike": "e_spike",
            "p_leak_w": "p_leak", "leakage_power": "p_leak",
            "p_static": "p_leak"}
_METADATA = ("source", "notes", "corner", "vdd", "temperature", "cell", "date")


def read_values(path):
    """Read ``name -> value`` from a .json, .toml or .csv data file.

    CSV rows are ``name,value[,anything]``; blank lines, ``#`` comments and a
    header row are skipped. The file is parsed as data only."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    elif ext == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:            # Python < 3.11
            import tomli as tomllib
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    elif ext == ".csv":
        raw = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 2 or row[0].strip().startswith("#"):
                    continue
                key, text = row[0].strip(), row[1].strip()
                if key.lower() in ("name", "parameter"):
                    continue
                try:
                    raw[key] = float(text)
                except ValueError:
                    raw[key] = text
    else:
        raise ValueError(f"Unsupported power-parameter file type {ext!r} "
                         "(use .json, .toml or .csv)")
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a table of name -> value")
    return raw


@dataclass
class PowerParams:
    """Energy-model parameters in SI units. ``None`` means "not supplied":
    the value in :data:`PLACEHOLDERS` is used and flagged in every report."""
    v_read: float = None                # V
    t_read: float = None                # s
    v_write: float = None               # V, current-driven devices only
    write_current: float = None         # A; None = from the device model
    e_spike: float = None               # J per spike
    p_leak: float = None                # W per neuron
    e_row_driver: float = None          # J per row-drive
    e_col_sense: float = None           # J per column per active step
    e_wta: float = None                 # J per inhibition event
    p_peripheral_static: float = None   # W
    timestep_s: float = None            # s; None = the simulation step
    #: name -> where a supplied value came from ("config", "file:<name>")
    sources: dict = field(default_factory=dict, repr=False, compare=False)
    #: free-text metadata of loaded files (corner, vdd, ...)
    metadata: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)
                if f.name not in ("sources", "metadata")]

    def __post_init__(self):
        for name in self.names():
            value = getattr(self, name)
            if value is not None:
                self._set(name, value, self.sources.get(name, "config"))

    def _set(self, name, value, source):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"power.{name} must be a number") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"power.{name} must be finite and >= 0")
        setattr(self, name, value)
        self.sources[name] = source

    @classmethod
    def from_config(cls, section):
        """From a config ``[power]`` table: parameter values plus an optional
        ``neuron_file`` holding circuit-extracted neuron values."""
        section = dict(section or {})
        neuron_file = section.pop("neuron_file", "")
        unknown = set(section) - set(cls.names())
        if unknown:
            raise ValueError(f"Unknown power field(s): {sorted(unknown)}")
        p = cls(**{k: v for k, v in section.items() if v is not None})
        if neuron_file:
            p.load_neuron_file(neuron_file)
        return p

    def load_file(self, path, allowed=None):
        """Set parameters from a data file (see :func:`read_values`). A value
        already supplied elsewhere is a conflict, not silently overridden."""
        allowed = tuple(allowed or self.names())
        label = "file:" + os.path.basename(str(path))
        for key, value in read_values(path).items():
            name = _ALIASES.get(key.lower(), key)
            if key.lower() in _METADATA:
                self.metadata[key.lower()] = value
                continue
            if name not in allowed:
                raise ValueError(f"{path}: unknown parameter {key!r} "
                                 f"(expected one of {list(allowed)})")
            if getattr(self, name) is not None:
                raise ValueError(f"power.{name} is set both in "
                                 f"{self.sources[name]} and in {path}")
            self._set(name, value, label)
        return self

    def load_neuron_file(self, path):
        """Hook for neuron values extracted from a circuit simulation (e.g. a
        Cadence LIF testbench): a .json/.toml/.csv with ``e_spike`` (J per
        spike) and/or ``p_leak`` (W per neuron), plus optional metadata
        (source, corner, vdd, temperature, cell, date, notes)."""
        return self.load_file(path, NEURON_FIELDS)

    def get(self, name):
        """The value used by the model: supplied, else the placeholder."""
        value = getattr(self, name)
        return PLACEHOLDERS[name][0] if value is None else value

    def placeholders(self):
        """Names still running on a placeholder value."""
        return [n for n in PLACEHOLDERS if getattr(self, n) is None]

    def table(self, dt_s=None):
        """name -> {value, unit, source, meaning} for a report."""
        out = {}
        for name, (value, unit, meaning) in PLACEHOLDERS.items():
            supplied = getattr(self, name)
            out[name] = {"value": value if supplied is None else supplied,
                         "unit": unit, "meaning": meaning,
                         "source": ("PLACEHOLDER" if supplied is None
                                    else self.sources.get(name, "config"))}
        for name, (unit, meaning, fallback) in DERIVED.items():
            supplied = getattr(self, name)
            value = supplied
            if supplied is None and name == "timestep_s":
                value = dt_s
            out[name] = {"value": value, "unit": unit, "meaning": meaning,
                         "source": (fallback if supplied is None
                                    else self.sources.get(name, "config"))}
        return out


class _Tally:
    """Parameter-independent sums of one phase."""

    def __init__(self, shapes):
        z = lambda: [np.zeros(s) for s in shapes]
        self.gs = z()            # sum of G x pre spikes per device (S)
        self.w_ohm = z()         # sum of amp^2 x mean G x width (J)
        self.w_aw = z()          # sum of |amp| x width (V s or A s)
        self.w_dev = z()         # sum of |amp| x device current x width (J)
        self.w_n = [np.zeros(s, dtype=np.int64) for s in shapes]   # pulses
        self.rows = [np.zeros(s[1]) for s in shapes]      # row-drives per input
        self.spikes = [np.zeros(s[0]) for s in shapes]    # spikes per neuron
        self.col_steps = [0] * len(shapes)   # steps with >= 1 active row
        self.wta = 0
        self.steps = 0
        self.presentations = 0


class EnergyMeter:
    """Accumulates the activity of a network and turns it into energy.

    shapes   (n_post, n_pre) of every crossbar
    kinds    "voltage" | "current" write drive of every crossbar
    params   :class:`PowerParams`
    dt_s     simulation time step (s)
    """

    def __init__(self, shapes, kinds, params=None, dt_s=1e-3):
        self.shapes = [tuple(s) for s in shapes]
        self.kinds = list(kinds)
        self.p = params if params is not None else PowerParams()
        self.dt_s = float(dt_s)
        self.device_current = [False] * len(self.shapes)
        self.tallies = {}

    @property
    def params(self):
        """The :class:`PowerParams` energies are priced with. The tallies are
        parameter-independent, so assigning new params re-prices every phase
        already recorded without re-simulating."""
        return self.p

    @params.setter
    def params(self, value):
        self.p = value if value is not None else PowerParams()

    def tally(self, phase):
        if phase not in self.tallies:
            self.tallies[phase] = _Tally(self.shapes)
        return self.tallies[phase]

    def reset(self, phase=None):
        """Clear one phase, or everything."""
        if phase is None:
            self.tallies = {}
        else:
            self.tallies.pop(phase, None)

    # ---- recording --------------------------------------------------------

    def record_forward(self, phase, G, rasters, wta_ops=0):
        """One forward pass. ``G[c]`` is crossbar c's conductance (S) during
        the pass; ``rasters[c]`` the 0/1 spikes of layer c, shaped
        (T, n) or batched (B, T, n), with layer 0 the inputs."""
        t = self.tally(phase)
        first = np.asarray(rasters[0])
        t.steps += int(np.prod(first.shape[:-1]))
        t.presentations += int(np.prod(first.shape[:-2]))
        t.wta += int(wta_ops)
        for c in range(len(self.shapes)):
            pre = np.asarray(rasters[c])
            n_spk = pre.reshape(-1, pre.shape[-1]).sum(0, dtype=float)
            t.gs[c] += np.asarray(G[c], float) * n_spk[None, :]
            t.rows[c] += n_spk
            t.col_steps[c] += int(np.count_nonzero(pre.any(-1)))
            post = np.asarray(rasters[c + 1])
            t.spikes[c] += post.reshape(-1, post.shape[-1]).sum(0, dtype=float)

    def record_write(self, phase, c, rows, cols, amps, width, g_before,
                     g_after, current=None):
        """Programming pulses of signed amplitude ``amps`` and duration
        ``width`` on devices (rows, cols) of crossbar c. ``g_before`` /
        ``g_after`` are their conductances around the pulse; ``current`` the
        device-model write current (A) when the model provides one."""
        t = self.tally(phase)
        a = np.abs(np.asarray(amps, float))
        t.w_n[c][rows, cols] += 1
        t.w_aw[c][rows, cols] += a * width
        g_mean = 0.5 * (np.asarray(g_before, float) + np.asarray(g_after, float))
        t.w_ohm[c][rows, cols] += a * a * g_mean * width
        if current is not None:
            self.device_current[c] = True
            t.w_dev[c][rows, cols] += a * np.abs(np.asarray(current, float)) * width

    # ---- energies ---------------------------------------------------------

    def timestep(self):
        return self.dt_s if self.p.timestep_s is None else self.p.timestep_s

    def read_energy(self, phase, c):
        """Per-device read energy of crossbar c (J): V_read^2 G t_read summed
        over every read of the device."""
        return self.p.get("v_read") ** 2 * self.p.get("t_read") * self.tally(phase).gs[c]

    def write_source(self, c):
        """How the write current of crossbar c is obtained."""
        if self.kinds[c] == "current":
            return "current-driven pulse x v_write"
        if self.p.write_current is not None:
            return "power.write_current"
        if self.device_current[c]:
            return "device model write_current()"
        return "ohmic: amplitude x mean conductance"

    def write_energy(self, phase, c):
        """Per-device write energy of crossbar c (J), amplitude x current x
        width per pulse. Current-driven devices: |I| x v_write x width.
        Voltage-driven devices: |V| x I x width with I, in order of
        preference, ``power.write_current``, the device model's
        ``write_current()``, or the ohmic V x (mean of the conductance before
        and after the pulse)."""
        t = self.tally(phase)
        if self.kinds[c] == "current":
            return t.w_aw[c] * self.p.get("v_write")
        if self.p.write_current is not None:
            return t.w_aw[c] * self.p.write_current
        if self.device_current[c]:
            return t.w_dev[c].copy()
        return t.w_ohm[c].copy()

    def neuron_energy(self, phase, layer):
        """Per-neuron energy (J) of LIF layer ``layer`` (0 = the neurons after
        crossbar 0): spikes x e_spike + leakage over the simulated time."""
        t = self.tally(phase)
        leak = self.p.get("p_leak") * t.steps * self.timestep()
        return self.p.get("e_spike") * t.spikes[layer] + leak

    def row_driver_energy(self, phase, c):
        """Row-driver energy per input line of crossbar c (J)."""
        return self.p.get("e_row_driver") * self.tally(phase).rows[c]

    def maps(self, phase):
        """Per-synapse and per-neuron energy arrays of one phase (J)."""
        out = {}
        for c in range(len(self.shapes)):
            out[f"read_J_c{c}"] = self.read_energy(phase, c)
            out[f"write_J_c{c}"] = self.write_energy(phase, c)
            out[f"write_pulses_c{c}"] = self.tally(phase).w_n[c].copy()
            out[f"row_driver_J_c{c}"] = self.row_driver_energy(phase, c)
            out[f"neuron_J_l{c + 1}"] = self.neuron_energy(phase, c)
        return out

    def report(self, phase, parameters=True):
        """JSON-safe energy report of one phase."""
        p, t, K = self.p, self.tally(phase), len(self.shapes)
        read = [float(self.read_energy(phase, c).sum()) for c in range(K)]
        write = [float(self.write_energy(phase, c).sum()) for c in range(K)]
        n_spikes = float(sum(s.sum() for s in t.spikes))
        n_neurons = sum(s[0] for s in self.shapes)
        sim_time = t.steps * self.timestep()
        row_ops = float(sum(r.sum() for r in t.rows))
        col_ops = float(sum(t.col_steps[c] * self.shapes[c][0] for c in range(K)))
        sops = float(sum(t.rows[c].sum() * self.shapes[c][0] for c in range(K)))
        peripherals = {"row_driver": p.get("e_row_driver") * row_ops,
                       "column_sense": p.get("e_col_sense") * col_ops,
                       "wta": p.get("e_wta") * t.wta}
        static = {"neuron_leak": p.get("p_leak") * n_neurons * sim_time,
                  "peripheral_static": p.get("p_peripheral_static") * sim_time}
        energy = {"array_read": sum(read), "array_write": sum(write),
                  "neurons": p.get("e_spike") * n_spikes,
                  "peripherals": sum(peripherals.values()),
                  "static": sum(static.values())}
        total = sum(energy.values())
        energy["total"] = total
        per = lambda num, den, k=1.0: (k * num / den) if den else None
        rep = {
            "phase": phase,
            "energy_J": energy,
            "peripherals_J": peripherals,
            "static_J": static,
            "array_read_J": read, "array_write_J": write,
            "counts": {"presentations": t.presentations, "steps": t.steps,
                       "sops": sops, "neuron_spikes": n_spikes,
                       "write_pulses": int(sum(n.sum() for n in t.w_n)),
                       "row_ops": row_ops, "col_ops": col_ops,
                       "wta_ops": t.wta},
            "sim_time_s": sim_time,
            "timestep_s": self.timestep(),
            # energy per synaptic operation: whole system, and the array alone
            "e_sop_pJ": per(total, sops, 1e12),
            "e_sop_array_pJ": per(energy["array_read"], sops, 1e12),
            # neuron energy (spikes + leakage) per emitted spike
            "e_neuron_spike_pJ": per(energy["neurons"] + static["neuron_leak"],
                                     n_spikes, 1e12),
            "e_write_pulse_pJ": per(energy["array_write"],
                                    sum(n.sum() for n in t.w_n), 1e12),
            "e_inference_J": per(total, t.presentations),
            "avg_power_W": per(total, sim_time),
            "placeholders": p.placeholders(),
            "write_current": [self.write_source(c) for c in range(K)],
        }
        if parameters:
            rep["parameters"] = p.table(self.dt_s)
        return rep


def _eng(value, unit):
    """1.23e-11, "J" -> "12.3 pJ"."""
    if value is None:
        return "n/a"
    if value == 0:
        return f"0 {unit}"
    exp = int(math.floor(math.log10(abs(value)) / 3.0)) * 3
    exp = min(max(exp, -18), 9)
    prefix = {-18: "a", -15: "f", -12: "p", -9: "n", -6: "u", -3: "m", 0: "",
              3: "k", 6: "M", 9: "G"}[exp]
    return f"{value / 10.0 ** exp:.3g} {prefix}{unit}"


def format_report(rep, title=None):
    """A report as a plain-text breakdown table."""
    e, c = rep["energy_J"], rep["counts"]
    total = e["total"]
    share = lambda v: f"{100.0 * v / total:5.1f}%" if total else "    - "
    line = lambda name, v, indent="": (f"  {indent}{name:<{26 - len(indent)}}"
                                       f"{_eng(v, 'J'):>12}  {share(v)}")
    per = lambda key, unit: ("n/a" if rep[key] is None
                             else f"{rep[key]:.4g} {unit}")
    out = [title or f"Energy - {rep['phase']}",
           f"  {c['presentations']} presentations, {c['steps']} steps of "
           f"{_eng(rep['timestep_s'], 's')} = {_eng(rep['sim_time_s'], 's')}",
           line("array read", e["array_read"]),
           line("array write", e["array_write"]),
           line("neurons (spikes)", e["neurons"]),
           line("peripherals", e["peripherals"]),
           line("row drivers", rep["peripherals_J"]["row_driver"], "  "),
           line("column sense", rep["peripherals_J"]["column_sense"], "  "),
           line("WTA / inhibition", rep["peripherals_J"]["wta"], "  "),
           line("static", e["static"]),
           line("neuron leakage", rep["static_J"]["neuron_leak"], "  "),
           line("peripheral static", rep["static_J"]["peripheral_static"], "  "),
           line("total", total),
           f"  energy / synaptic op      {per('e_sop_pJ', 'pJ/SOP')} "
           f"(array read only: {per('e_sop_array_pJ', 'pJ/SOP')}; "
           f"{c['sops']:.0f} SOPs)",
           f"  energy / neuron spike     {per('e_neuron_spike_pJ', 'pJ')} "
           f"({c['neuron_spikes']:.0f} spikes, incl. leakage)",
           f"  energy / write pulse      {per('e_write_pulse_pJ', 'pJ')} "
           f"({c['write_pulses']} pulses)",
           f"  energy / presentation     {_eng(rep['e_inference_J'], 'J')}",
           f"  average power             {_eng(rep['avg_power_W'], 'W')}",
           "  write current: " + "; ".join(dict.fromkeys(rep["write_current"]))]
    if rep["placeholders"]:
        out.append("  PLACEHOLDER values (not measured): "
                   + ", ".join(rep["placeholders"]))
    else:
        out.append("  all parameters supplied (no placeholders)")
    return "\n".join(out)


def save_maps(meter, path, phases=None):
    """Write every phase's per-synapse / per-neuron energy arrays to one
    ``.npz`` (keys ``<phase>_read_J_c0``, ``<phase>_neuron_J_l1``, ...)."""
    arrays = {}
    for phase in (phases or list(meter.tallies)):
        for key, value in meter.maps(phase).items():
            arrays[f"{phase}_{key}"] = value
    np.savez_compressed(path, **arrays)
    return path


def plot_maps(meter, path, phase, title=None):
    """Heatmaps of one phase: per crossbar the read and write energy of every
    synapse, and the energy of every neuron. Needs matplotlib; returns the
    path, or None when matplotlib is not installed."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        return None
    K = len(meter.shapes)
    fig = Figure(figsize=(13, 3.2 * K), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = np.atleast_2d(fig.subplots(K, 3, squeeze=False))
    for c in range(K):
        panels = ((meter.read_energy(phase, c), f"crossbar {c}: read energy"),
                  (meter.write_energy(phase, c), f"crossbar {c}: write energy"))
        for ax, (data, name) in zip(axes[c], panels):
            im = ax.imshow(data * 1e12, aspect="auto", interpolation="nearest",
                           cmap="viridis")
            ax.set_title(name)
            ax.set_xlabel("input (pre)")
            ax.set_ylabel("neuron (post)")
            fig.colorbar(im, ax=ax, label="pJ")
        ax = axes[c][2]
        e_n = meter.neuron_energy(phase, c) * 1e12
        ax.bar(np.arange(e_n.size), e_n, color="tab:blue")
        ax.set_title(f"layer {c + 1}: neuron energy")
        ax.set_xlabel("neuron")
        ax.set_ylabel("pJ")
    names = meter.p.placeholders()
    fig.suptitle((title or f"Energy maps - {phase}")
                 + (f"   [PLACEHOLDER: {', '.join(names)}]" if names else ""),
                 fontsize=10)
    fig.savefig(path, dpi=130)
    return path
