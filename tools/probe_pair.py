#!/usr/bin/env python3
"""Read-only: confirm the I52/I53 pre-post pair on I54's IGsc - signs (i2),
amplitudes, timing (td) and edge rates (tr/tf, which @(cross) needs)."""
from skillbridge import Workspace

LIB, CELL, VIEW = "Sumedha_Li_MEM_model", "basic_v1_test_schemetic", "schematic"
ws = Workspace.open()
ws["evalstring"](
    '(defun pairInfo (lib cell view)\n'
    '  (let ((cv (dbOpenCellViewByType lib cell view "" nil "r")) (s ""))\n'
    '    (when cv\n'
    '      (setq s (strcat s "IGsc(I54) net = "))\n'
    '      (foreach inst cv->instances\n'
    '        (when (equal inst->name "I54")\n'
    '          (foreach it inst->instTerms\n'
    '            (when (equal it->name "IGsc")\n'
    '              (setq s (strcat s (if it->net it->net->name "FLOAT")))))))\n'
    '      (setq s (strcat s "\\n"))\n'
    '      (foreach inst cv->instances\n'
    '        (when (member inst->name (list "I52" "I53"))\n'
    '          (let ((cdf (cdfGetInstCDF inst)) (net ""))\n'
    '            (foreach it inst->instTerms\n'
    '              (when (equal it->name "MINUS")\n'
    '                (setq net (if it->net it->net->name "FLOAT"))))\n'
    '            (when cdf\n'
    '              (setq s (strcat s inst->name " ->" net\n'
    '                " : i1=" cdf->i1->value " i2=" cdf->i2->value\n'
    '                " td=" cdf->td->value " pw=" cdf->pw->value\n'
    '                " tr=" cdf->tr->value " tf=" cdf->tf->value "\\n"))))))\n'
    '      (dbClose cv))\n'
    '    s))'
)
print(ws["pairInfo"](LIB, CELL, VIEW))
