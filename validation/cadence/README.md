# Virtuoso schematic + Maestro testbenches (Track B)

SKILL that builds the user-friendly versions of the batch decks in a library
`NVAT_VAL`, driven through skillbridge. Load with `load("nvat_tb.il")` and
`load("nvat_mae.il")` in the CIW (or via skillbridge `evalstring`).

- `nvat_tb.il`: Verilog-A cell symbols (`nvatMakeSymbols`) and wired testbench
  schematics `tb_ecfet_v3`, `tb_rram` (`nvatTbEcfet`, `nvatTbRram`). The PWL
  sources read the design variable `pwlfile` (`file_as_var`).
- `nvat_mae.il`: Maestro setups with the deck's exact options and tran
  settings (`nvatMaeEcfet`, `nvatMaeRram`), the OCEAN CSV dump `nvatDump`,
  and image helpers.

Gotchas found on IC25.1 / Spectre 24.1:
- Run `ahdlUpdateViewInfo` on each veriloga view, or netlisting fails (OSSHNL-381: no netlist.oa).
- Symbol pin directions must match the `.va` (IGsc input, Rmem output).
- Maestro copies `spectre.turboOpts uniMode` ("Spectre X" by default) into each test when it is created. Create the tests with the session default set to "Spectre", then restore the default.
- Never call `maeRunSimulation(?waitUntilDone t)` through skillbridge: it blocks Virtuoso's event loop and deadlocks. Start the run and poll the result directory instead.
- Per-test `maeSetVar` values lose their quotes. Set the global `pwlfile` and enable one test per run.

Result: Maestro vs batch Spectre, same deck, max |diff| ≤ 3e-6 of range on
all six runs (validation/data/maestro, compare in validation/plot_maestro_vs_twin.py).
