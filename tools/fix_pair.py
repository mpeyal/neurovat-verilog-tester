#!/usr/bin/env python3
"""Apply STDP-pair schematic fixes to I52/I53: equal small width (pw=1m) and
finite edges (tr=tf=1u) so @(cross) fires cleanly. Signs (i2) left as-is
(I53=posAmp +, I52=negAmp -). Runs on the server (local socket, no tunnel)."""
from skillbridge import Workspace

LIB, CELL, VIEW = "Sumedha_Li_MEM_model", "basic_v1_test_schemetic", "schematic"
ws = Workspace.open()
ws["evalstring"](
    '(defun fixPair (lib cell view)\n'
    '  (let ((cv (dbOpenCellViewByType lib cell view "" nil "a")) (s ""))\n'
    '    (when cv\n'
    '      (foreach inst cv->instances\n'
    '        (when (member inst->name (list "I52" "I53"))\n'
    '          (let ((cdf (cdfGetInstCDF inst)))\n'
    '            (when cdf\n'
    '              (cdf->pw->value = "1m")\n'
    '              (cdf->tr->value = "1u")\n'
    '              (cdf->tf->value = "1u")\n'
    '              (setq s (strcat s inst->name\n'
    '                " i2=" cdf->i2->value " pw=" cdf->pw->value\n'
    '                " tr=" cdf->tr->value " tf=" cdf->tf->value "; "))))))\n'
    '      (dbSave cv)\n'
    '      (dbClose cv))\n'
    '    s))'
)
print("applied:", ws["fixPair"](LIB, CELL, VIEW))
