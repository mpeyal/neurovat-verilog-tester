"""Device-twin compute experiments. No GUI, circuit-solver or timing claims."""
import csv
import io
import math
from dataclasses import asdict, dataclass

import numpy as np

from .neuro import device_weight


@dataclass
class Settings:
    read_voltage: float = .1
    read_ns: float = 100.
    levels: int = 256
    mismatch: float = 0.
    seed: int = 17
    banks: int = 1
    overhead_ns: float = 20.
    peripheral_pj: float = 0.

    def validate(self):
        for name, low, high in (("read_voltage", 1e-6, 10), ("read_ns", .001, 1e9),
                                ("mismatch", 0, 1), ("overhead_ns", 0, 1e9),
                                ("peripheral_pj", 0, 1e12)):
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}.")
        for name, low, high in (("levels", 2, 65536), ("banks", 1, 256), ("seed", 0, 2**32-1)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}.")


def parse_matrix(text):
    """Rows separated by newlines/semicolons, numbers by commas or spaces."""
    if len(text) > 200000:
        raise ValueError("Matrix text is too large.")
    rows = [[float(x) for x in row.replace(',', ' ').split()]
            for row in text.replace(';', '\n').splitlines() if row.strip()]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Enter a rectangular numeric matrix.")
    value = np.array(rows)
    if not np.isfinite(value).all() or np.max(np.abs(value)) > 1e9:
        raise ValueError("Matrix entries must be finite and at most 1e9 in magnitude.")
    return value


def bounds(make):
    dev = make()
    if hasattr(dev, 'P') and hasattr(dev.p, 'dVth_max'):
        dev.P = -1.; low = device_weight(dev)
        dev.P = 1.; high = device_weight(dev)
        low, high = sorted((low, high))
    elif hasattr(dev.p, 'Gmin') and hasattr(dev.p, 'Gmax'):
        low, high = dev.p.Gmin, dev.p.Gmax
    elif hasattr(dev.p, 'Rmin') and hasattr(dev.p, 'Rmax'):
        low, high = 1/dev.p.Rmax, 1/dev.p.Rmin
    else:
        raise ValueError("This twin has no supported conductance-state mapping.")
    if not np.isfinite([low, high]).all() or not 0 <= low < high:
        raise ValueError("Device conductance bounds must be finite and ordered.")
    return float(low), float(high)


def initialize(make, target):
    """Explicit ideal state initialization; NOT a programming-pulse simulation."""
    dev = make()
    dev.reset()
    if hasattr(dev, 'P') and hasattr(dev.p, 'dVth_max'):
        p = dev.p
        fraction = np.clip((target-p.Gmin)/(p.Gmax-p.Gmin), 1e-15, 1-1e-15)
        vth = p.Vread-p.SS*np.log(fraction/(1-fraction))
        dev.P = float(np.clip(2*(p.Vth0-vth)/p.dVth_max, -1, 1))
    elif hasattr(dev, 'G_nv'):
        dev.G_nv = float(target)
        if hasattr(dev, 'x') and hasattr(dev.p, 'x_of_G'):
            dev.x = dev.p.x_of_G(target)
    elif hasattr(dev, 'M') and hasattr(dev.p, 'Rdrift1c'):
        dev.M = dev.M_before_diffusion = 1/target
        dev.Vcap = dev.M/dev.p.Rdrift1c
    else:
        raise ValueError(f"No state initializer for {type(dev).__name__}.")
    actual = device_weight(dev)
    if not np.isclose(actual, target, rtol=1e-7, atol=1e-12):
        raise ValueError("Requested conductance is not reachable by this twin.")
    return actual


ASSUMPTIONS = [
    "Python behavioral twin; Verilog-A source mapping is not proof of circuit equivalence.",
    "Ideal state initialization, followed by quantization and optional imposed mismatch; programming energy excluded.",
    "Linear, non-disturbing read I=GV at retained conductance; no wire resistance, loading, ADC clipping or read-bias nonlinearity.",
    "A twin's readout parameters define retained conductance. Experiment voltage scales I=GV; it does not solve a new transistor bias point.",
    "Signed matrix uses two devices per weight and ideal differential sensing; negative inputs require bipolar drivers.",
    "Array read energy includes both differential branches. Peripheral energy is a user estimate; zero means excluded.",
    "Parallel banks are a scheduling/resource estimate, not host CPU/GPU benchmarking or measured hardware speedup.",
    "Replicated banks share this representative conductance/mismatch map; bank-to-bank variation and idle-bank leakage are excluded.",
]


def matrix(make, weights, inputs, settings=None, stop=None):
    cfg = settings or Settings(); cfg.validate()
    a, x = np.asarray(weights, float), np.asarray(inputs, float)
    if a.ndim != 2 or x.ndim != 2 or a.shape[1] != x.shape[0]:
        raise ValueError("A [outputs, inputs] and X [inputs, vectors] must have matching inner dimensions.")
    if not all(1 <= v <= 64 for v in a.shape) or not 1 <= x.shape[1] <= 512:
        raise ValueError("Use up to 64 x 64 weights and 512 input vectors.")
    if not np.isfinite(a).all() or not np.isfinite(x).all() or max(np.abs(a).max(), np.abs(x).max()) > 1e9:
        raise ValueError("Inputs must be finite with magnitude <= 1e9.")
    low, high = bounds(make); span = high-low
    scale_a = float(np.abs(a).max()) or 1.
    scale_x = float(np.abs(x).max()) or 1.
    normalized = a/scale_a
    rng = np.random.default_rng(cfg.seed)
    gs = []
    for side in (np.maximum(normalized, 0), np.maximum(-normalized, 0)):
        fraction = np.round(side*(cfg.levels-1))/(cfg.levels-1)
        fraction = np.clip(fraction+rng.normal(0, cfg.mismatch, a.shape), 0, 1)
        g = np.empty(a.shape)
        for index in np.ndindex(a.shape):
            if stop is not None and stop.is_set():
                raise InterruptedError("Experiment stopped.")
            g[index] = initialize(make, low+span*fraction[index])
        gs.append(g)
    voltage = cfg.read_voltage*x/scale_x
    current = (gs[0]-gs[1]) @ voltage
    actual = current*scale_a*scale_x/(span*cfg.read_voltage)
    expected = a @ x
    error = actual-expected
    # Sum dissipation in both branches, including the baseline Gmin currents.
    energy = (voltage**2 * (gs[0]+gs[1]).sum(axis=0)[:, None]).sum(axis=0)*cfg.read_ns*1e-9
    vectors = x.shape[1]
    banks = min(cfg.banks, vectors)
    cycles = math.ceil(vectors/banks)
    period = (cfg.read_ns+cfg.overhead_ns)*1e-9
    latency = cycles*period
    peripheral = vectors*cfg.peripheral_pj*1e-12
    return dict(kind='matrix', settings=asdict(cfg), weights=a.tolist(), inputs=x.tolist(),
                expected=expected.tolist(), actual=actual.tolist(), error=error.tolist(),
                current_A=current.tolist(), g_plus_S=gs[0].tolist(), g_minus_S=gs[1].tolist(),
                conductance_range_S=[low, high], array_energy_per_vector_J=energy.tolist(),
                array_energy_J=float(energy.sum()), peripheral_energy_J=peripheral,
                accounted_energy_J=float(energy.sum()+peripheral),
                average_accounted_power_W=float((energy.sum()+peripheral)/latency),
                max_abs_error=float(np.abs(error).max()), rmse=float(np.sqrt(np.mean(error**2))),
                relative_l2_error=float(np.linalg.norm(error)/np.linalg.norm(expected)) if np.linalg.norm(expected) else None,
                effective_banks=banks, read_cycles=cycles, estimated_latency_s=latency,
                serial_latency_s=vectors*period, scheduling_speedup=vectors/cycles,
                vectors_per_second=vectors/latency, physical_devices=2*a.size*banks,
                assumptions=list(ASSUMPTIONS))


def logic(make, operation='ADD', bits=2, settings=None, stop=None):
    """NAND-based combinational networks with ideal threshold comparators.

    Each gate is an independently mapped differential device array. All truth
    table rows run through the gate network; the expected answer is independent.
    """
    if operation not in ('NAND', 'AND', 'OR', 'XOR', 'NOT', 'ADD'):
        raise ValueError("Choose NAND, AND, OR, XOR, NOT or ADD.")
    if type(bits) is not int or not 1 <= bits <= 4:
        raise ValueError("ALU width must be 1-4 bits for exhaustive testing.")
    cfg = settings or Settings(); cfg.validate()
    if operation == 'ADD':
        rows = [(a, b, c) for a in range(2**bits) for b in range(2**bits) for c in (0, 1)]
    else:
        rows = [(a, b, 0) for a in (0, 1) for b in ((0,) if operation == 'NOT' else (0, 1))]
    a, b, carry = np.array(rows).T
    energy = np.zeros(len(rows)); gate_count = 0; margins = []; max_depth = 0

    def nand(left, right):
        nonlocal gate_count, max_depth
        values = np.vstack([left[0], right[0], np.ones(len(rows))])
        local = Settings(**asdict(cfg)); local.seed = (cfg.seed+gate_count) % (2**32)
        local.banks = 1
        r = matrix(make, [[-1., -1., 1.5]], values, local, stop)
        gate_count += 1
        score = np.array(r['actual'])[0]
        margins.append(float(np.abs(score).min()))
        energy[:] += r['array_energy_per_vector_J']
        depth = max(left[1], right[1])+1
        max_depth = max(max_depth, depth)
        return (score >= 0).astype(int), depth

    def xor(left, right):
        first = nand(left, right)
        return nand(nand(left, first), nand(right, first)), first

    left, right = (a, 0), (b, 0)
    if operation == 'ADD':
        result = np.zeros(len(rows), dtype=int); c = (carry, 0)
        for i in range(bits):
            ab, n1 = xor(((a >> i) & 1, 0), ((b >> i) & 1, 0))
            summation, n2 = xor(ab, c)
            result |= summation[0] << i
            c = nand(n1, n2)
        actual = result | (c[0] << bits)
        expected = a+b+carry
    elif operation == 'XOR':
        actual = xor(left, right)[0][0]; expected = a ^ b
    elif operation == 'NAND':
        actual = nand(left, right)[0]; expected = 1-(a & b)
    elif operation == 'NOT':
        actual = nand(left, left)[0]; expected = 1-a
    elif operation == 'AND':
        n = nand(left, right); actual = nand(n, n)[0]; expected = a & b
    else:
        actual = nand(nand(left, left), nand(right, right))[0]; expected = a | b
    assumptions = list(ASSUMPTIONS)+[
        "NAND scores use weights [-1,-1,1.5] including a constant-bias input, followed by an ideal zero-threshold comparator.",
        "Functional combinational verification; no transistor-level ALU, carry setup/hold, hazards or switching-power validation.",
        "Each truth row is independent. Logic scheduling assumes no overlap between transactions and depth times the per-gate read period.",
    ]
    cycles = math.ceil(len(rows)/min(cfg.banks, len(rows)))
    latency = cycles*max_depth*(cfg.read_ns+cfg.overhead_ns)*1e-9
    periph = len(rows)*gate_count*cfg.peripheral_pj*1e-12
    return dict(kind='logic', operation=operation, bits=bits if operation == 'ADD' else 1,
                settings=asdict(cfg), inputs=rows, expected=expected.tolist(), actual=actual.tolist(),
                passed=int((actual == expected).sum()), cases=len(rows),
                gate_count=gate_count, depth=max_depth, min_comparator_margin=min(margins),
                physical_devices=6*gate_count*min(cfg.banks, len(rows)),
                effective_banks=min(cfg.banks, len(rows)), batch_cycles=cycles,
                cases_per_second=len(rows)/latency, scheduling_speedup=len(rows)/cycles,
                array_energy_per_case_J=energy.tolist(), array_energy_J=float(energy.sum()),
                peripheral_energy_J=periph, accounted_energy_J=float(energy.sum()+periph),
                estimated_latency_s=latency, average_accounted_power_W=float((energy.sum()+periph)/latency),
                assumptions=assumptions)


def terminal_energy(text):
    """Integrate sampled terminal power with passive-sign currents into device."""
    if len(text) > 4_000_000:
        raise ValueError("Trace CSV is limited to 4 MB.")
    reader = csv.DictReader(io.StringIO(text.strip()))
    fields = reader.fieldnames or []
    suffixes = [s[2:] for s in fields if s.startswith('v_')]
    if len(fields) != len(set(fields)) or 't_s' not in fields or not suffixes or set(fields) != {'t_s', *('v_'+s for s in suffixes), *('i_'+s for s in suffixes)}:
        raise ValueError("Use t_s and matched terminal columns v_1,i_1 (optionally v_2,i_2, ...), in seconds, volts and amps.")
    rows = list(reader)
    if not 2 <= len(rows) <= 50000:
        raise ValueError("Supply 2-50,000 trace samples.")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Each trace row must match the CSV header column count.")
    try:
        values = np.array([[float(row[k]) for k in fields] for row in rows])
    except (ValueError, TypeError):
        raise ValueError("Trace cells must all be numeric.") from None
    if not np.isfinite(values).all():
        raise ValueError("Trace samples must be finite.")
    t = values[:, fields.index('t_s')]
    if np.any(np.diff(t) <= 0):
        raise ValueError("Trace timestamps must be strictly increasing.")
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            powers = {s: values[:, fields.index('v_'+s)]*values[:, fields.index('i_'+s)] for s in suffixes}
            power = sum(powers.values())
            steps = np.diff(t)*(power[1:]+power[:-1])/2
            energies = {s: float(np.sum(np.diff(t)*(p[1:]+p[:-1])/2)) for s, p in powers.items()}
            cumulative = np.r_[0, np.cumsum(steps)]
            average = float(steps.sum()/(t[-1]-t[0]))
    except FloatingPointError:
        raise ValueError("Trace power overflows numeric range.") from None
    return dict(kind='terminal_energy', time_s=t.tolist(), power_W=power.tolist(),
                cumulative_energy_J=cumulative.tolist(), terminal_energy_J=energies,
                net_energy_J=float(steps.sum()), average_power_W=average,
                peak_power_W=float(power.max()), duration_s=float(t[-1]-t[0]),
                assumptions=["User-supplied terminal trace, not generated by this experiment.",
                             "SI units; currents positive INTO the device; trapezoidal integration of sampled V*I.",
                             "Net terminal energy can include stored/released energy and is not necessarily dissipated heat.",
                             "Include all supply, programming and read terminals for a complete device energy balance."])
