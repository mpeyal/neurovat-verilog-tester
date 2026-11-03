#!/usr/bin/env python3
"""Read-only: dump instance terminal -> net connectivity of the failing
testbench so we can confirm the I52/IGsc dangling-wire diagnosis on the live
schematic (not just the generated netlist).

    python tools/probe_tb_wiring.py
"""
from skillbridge import Workspace

LIB, CELL, VIEW = "Sumedha_Li_MEM_model", "basic_v1_test_schemetic", "schematic"

ws = Workspace.open()

# push iteration to the SKILL side: one round trip, returns inst|term|net lines
ws["evalstring"](
    '(defun tbWiring (lib cell view)\n'
    '  (let ((cv (dbOpenCellViewByType lib cell view "" nil "r")) (s ""))\n'
    '    (when cv\n'
    '      (foreach inst cv->instances\n'
    '        (foreach it inst->instTerms\n'
    '          (setq s (strcat s inst->name "|" inst->cellName "|" it->name "|"\n'
    '                          (if it->net it->net->name "<FLOAT>") "\\n"))))\n'
    '      (dbClose cv))\n'
    '    s))'
)
out = ws["tbWiring"](LIB, CELL, VIEW)

rows = [ln.split("|") for ln in out.splitlines() if ln.strip()]
by_inst, net_users = {}, {}
for inst, cell, term, net in rows:
    by_inst.setdefault(inst, (cell, []))[1].append((term, net))
    net_users.setdefault(net, set()).add(inst)

print(f"=== {LIB}/{CELL}/{VIEW} : instance (master) terminal -> net ===")
for inst in sorted(by_inst):
    cell, pins = by_inst[inst]
    pinstr = ", ".join(f"{t}->{n}" for t, n in pins)
    print(f"  {inst:6s} [{cell:10s}]: {pinstr}")

print("\n=== nets touched by only ONE instance (floating / single-connection) ===")
for net in sorted(net_users):
    if net not in ("gnd!",) and len(net_users[net]) == 1:
        print(f"  {net:8s} <- only {sorted(net_users[net])[0]}")
