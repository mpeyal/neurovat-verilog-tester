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

The `.va` file itself is not in git; `validation/README.md` explains where to
get it.

## ⚠ Security
Importing a twin **executes its Python** — there is no sandbox. This folder
keeps device code out of the app source and constrains the agent to write only
here, but it is **organisational safety, not a security boundary**. Only place
twins you trust in this folder.
