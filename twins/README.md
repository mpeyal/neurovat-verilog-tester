# Device twins (drop new devices here)

This folder is the **extensible** place to add device models ("twins"). The GUI
auto-loads every `*.py` here at startup and registers it as a selectable model —
**you never edit the GUI (`vatester/`) or core engine (`ecfet/`) to add a
device.**

A twin is one self-contained Python file that defines:

- a **model class** with `step(t, dt, drive)`, `.R`, `.G`, `reset()`, and
  `observables()` (a dict of named scalar traces),
- a **params dataclass**, and
- a module-level **`TWIN_SPEC`** dict telling the GUI how to register/drive it.

See `example_rram.py` for a complete, working example, and
`vatester/twin_loader.py` for the full `TWIN_SPEC` reference.

## How a `.va` gets a twin
- `.va` files are matched to a twin by **keyword** (`va_keywords` in the spec,
  checked against the filename/module). A new `.va` whose name contains a twin's
  keyword reuses that twin automatically.
- A `.va` that matches **no** twin shows up but can't be simulated — the GUI
  will **prompt you to build a twin** (the agent writes a new file *here*).

## ⚠ Security
Importing a twin **executes its Python** — there is no sandbox. This folder
keeps device code out of the app source and constrains the agent to write only
here, but it is **organisational safety, not a security boundary**. Only place
twins you trust in this folder.
