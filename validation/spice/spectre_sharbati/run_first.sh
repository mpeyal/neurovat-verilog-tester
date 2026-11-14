#!/bin/sh
# GENERATED: gate-orientation check. R must FALL at t = 5 s (about 4400 -> 4370 ohm).
# If it rises, regenerate with --flip-gate before running run_all.sh.
set -e
cd "$(dirname "$0")/v3"
spectre +escchars -format psfascii -raw sp_pot.raw +log sp_pot.log sp_pot.scs
