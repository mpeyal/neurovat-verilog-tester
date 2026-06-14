#!/usr/bin/env python3
"""Read-only: what drives net10 (I54's IGsc) now, and the i2 of the pulse
sources, after the user re-wired to use I53 only."""
from skillbridge import Workspace

LIB, CELL, VIEW = "Sumedha_Li_MEM_model", "basic_v1_test_schemetic", "schematic"
ws = Workspace.open()
ws["evalstring"](
    '(defun net10drv (lib cell view)\n'
    '  (let ((cv (dbOpenCellViewByType lib cell view "" nil "r")) (s ""))\n'
    '    (when cv\n'
    '      (setq s (strcat s "net10 drivers: "))\n'
    '      (foreach inst cv->instances\n'
    '        (foreach it inst->instTerms\n'
    '          (when (and it->net (equal it->net->name "net10"))\n'
    '            (setq s (strcat s inst->name "." it->name " ")))))\n'
    '      (setq s (strcat s "\\npulse i2 values: "))\n'
    '      (foreach inst cv->instances\n'
    '        (when (member inst->name (list "I47" "I52" "I53"))\n'
    '          (let ((cdf (cdfGetInstCDF inst)))\n'
    '            (when cdf\n'
    '              (setq s (strcat s inst->name "=" cdf->i2->value " "))))))\n'
    '      (dbClose cv))\n'
    '    s))'
)
print(ws["net10drv"](LIB, CELL, VIEW))
