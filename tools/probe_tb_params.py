#!/usr/bin/env python3
"""Read-only: dump CDF parameter values of the stimulus sources + the v2
synapse instance in the failing testbench, so we can see what current is
actually injected into I54's IGsc (net10) and with what timing.

    python tools/probe_tb_params.py
"""
from skillbridge import Workspace

LIB, CELL, VIEW = "Sumedha_Li_MEM_model", "basic_v1_test_schemetic", "schematic"

ws = Workspace.open()
ws["evalstring"](
    '(defun tbParams (lib cell view)\n'
    '  (let ((cv (dbOpenCellViewByType lib cell view "" nil "r")) (s ""))\n'
    '    (when cv\n'
    '      (foreach inst cv->instances\n'
    '        (when (member inst->cellName\n'
    '                 (list "ipulse" "ipwl" "ipwlf" "vpulse" "ECFET_Synapse"))\n'
    '          (setq s (strcat s "### " inst->name " [" inst->cellName "]\\n"))\n'
    '          (let ((cdf (cdfGetInstCDF inst)))\n'
    '            (when cdf\n'
    '              (foreach p cdf->parameters\n'
    '                (when p->value\n'
    '                  (setq s (strcat s "  " p->name " = "\n'
    '                                  (sprintf nil "%A" p->value) "\\n"))))))))\n'
    '      (dbClose cv))\n'
    '    s))'
)
print(ws["tbParams"](LIB, CELL, VIEW))
