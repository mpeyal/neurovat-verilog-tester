#!/usr/bin/env python3
"""Neutralize the redundant stimulus sources I52 & I53 on the v2 synapse's
IGsc (net10) by setting their pulse amplitude i2 = 0, leaving I47 (the ipwl
STDP pair that matches the v1 synapse's I40) as the sole driver.

Reversible: only the i2 CDF parameter is changed; instances stay in place.

    python tools/fix_tb_stimulus.py
"""
from skillbridge import Workspace

LIB, CELL, VIEW = "Sumedha_Li_MEM_model", "basic_v1_test_schemetic", "schematic"

ws = Workspace.open()
ws["evalstring"](
    '(defun tbZero (lib cell view)\n'
    '  (let ((cv (dbOpenCellViewByType lib cell view "" nil "a")) (s ""))\n'
    '    (when cv\n'
    '      (foreach inst cv->instances\n'
    '        (when (member inst->name (list "I52" "I53"))\n'
    '          (let ((cdf (cdfGetInstCDF inst)))\n'
    '            (when cdf\n'
    '              (setq s (strcat s inst->name ": i2 " cdf->i2->value " -> "))\n'
    '              (cdf->i2->value = "0")\n'
    '              (setq s (strcat s cdf->i2->value "; "))))))\n'
    '      (dbSave cv)\n'
    '      (dbClose cv))\n'
    '    s))'
)
print("change:", ws["tbZero"](LIB, CELL, VIEW))

# read back to confirm
ws["evalstring"](
    '(defun tbCheck (lib cell view)\n'
    '  (let ((cv (dbOpenCellViewByType lib cell view "" nil "r")) (s ""))\n'
    '    (when cv\n'
    '      (foreach inst cv->instances\n'
    '        (when (member inst->name (list "I47" "I52" "I53"))\n'
    '          (let ((cdf (cdfGetInstCDF inst)))\n'
    '            (setq s (strcat s inst->name " i2/i2-pwl="\n'
    '                            (if (equal inst->cellName "ipulse") cdf->i2->value cdf->i2->value)\n'
    '                            " ")))))\n'
    '      (dbClose cv))\n'
    '    s))'
)
print("verify:", ws["tbCheck"](LIB, CELL, VIEW))
