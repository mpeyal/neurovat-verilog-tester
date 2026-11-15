# Device twins (drop new devices here)

This folder is the **extensible** place to add device models ("twins"). The GUI
auto-loads every `*.py` here at startup and registers it as a selectable model —
**you never edit the GUI (`vatester/`) or core engine (`simulation_core/`) to add a
device.**

A twin is one self-contained Python file that defines:

- a **model class** with `step(t, dt, drive)`, `.R`, `.G`, `reset()`, and
  `observables()` (a dict of named scalar traces),
- a **params dataclass**, and
- a module-level **`TWIN_SPEC`** dict telling the GUI how to register/drive it.

See `vatester/twin_loader.py` for the full `TWIN_SPEC` reference (its module
docstring is an annotated, copy-paste template).

## How a `.va` gets a twin
- `.va` files are matched to a twin by **keyword** (`va_keywords` in the spec,
  checked against the filename/module). A new `.va` whose name contains a twin's
  keyword reuses that twin automatically.
- A `.va` that matches **no** twin shows up but can't be simulated — the GUI
  will **prompt you to build a twin** (the agent writes a new file *here*).

## Bundled third-party twin: Stanford-PKU RRAM v1.0.0
`stanford_pku_rram.py` is a Python restatement of the Stanford-PKU RRAM
Verilog-A model. The model was developed by the **Stanford Nanoelectronics
Research Group of Prof. H.-S. Philip Wong**. Please cite:

- Z. Jiang et al., IEEE TED 63(5), 2016;
- Z. Jiang and H.-S. P. Wong, nanoHUB, 2014, doi:10.4231/D37H1DN48.

Licence terms (quoted in full in the file header):

- you may copy, modify and redistribute it free of charge;
- you must keep its copyright notices;
- you must not charge for the Stanford code.

The `.va` file itself is not in git. To get it, download "Verilog-A" from
https://nanohub.org/publications/19 (the file is `rram_v_1_0_0.va`, 6762
bytes, sha256
`ab0eebd3d8a04176f06a4f0a791444aa9f78786f79f1f3e1c54bb1f56c63d1ab`) and save
it, unmodified, as `verilog/stanford_pku_rram.va`; `verilog/.gitignore` keeps
it out of git. The scanner maps both this name and the original
`rram_v_1_0_0` name to the `stanford_pku` twin.

## VTEAM memristor (MemTorch variant)
`vteam.py` is the twin of `verilog/vteam.va`, NeuroVAT's own Spectre form of
the VTEAM model (S. Kvatinsky, M. Ramadan, E. G. Friedman, A. Kolodny, IEEE
TCAS-II 62(8), 2015) in the variant MemTorch 1.1.6 implements: paper eq. (6)
with the ideal rectangular window, linear I-V eq. (7), MemTorch's parameter
names and defaults. The equations are restated, no MemTorch (GPL-3) code is
copied. The twin integrates in closed form (exact, no step size).

Deviations of the MemTorch variant from the paper:

- Window: the paper's ideal rectangular window; the STATE is clipped to
  [0, d] after each step, so there is no wind-up (no Joglekar/Biolek window).
- I-V: eq. (7) with x_on = 0 and x_off - x_on replaced by d:
  R(x) = r_off x/d + r_on (1 - x/d). The exponential eq. (8) is not used.
- x_on is only the initial state and x_off is unused; the bounds are [0, d]
  (the same as the paper for the defaults x_on = 0, x_off = d).
- Defaults: paper Table II, column "TEAM [9]" (r_on 50, r_off 1000 ohm,
  d 3 nm, k_off 5e-4, k_on -10 m/s, alpha_off 1, alpha_on 3, v_off 0.02,
  v_on -0.2 V).
- Threshold tests are inclusive (v >= v_off); the rate is zero there anyway.

## ⚠ Security
Importing a twin **executes its Python** — there is no sandbox. This folder
keeps device code out of the app source and constrains the agent to write only
here, but it is **organisational safety, not a security boundary**. Only place
twins you trust in this folder.
