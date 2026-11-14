#!/bin/sh
# GENERATED: smoke test. Checks the .va compiles in Spectre and the
# current sign: during the -1.3 V pulse X1:TE must be NEGATIVE (about -35 mA at
# t = 1 ns, falling in magnitude as the gap opens), and the 0.1 V read at
# ~170 ns must give about +6.6 uA (G_read ~ 66 uS).
set -e
cd "$(dirname "$0")"
spectre +escchars -format psfascii -raw pulse_reset_ref.raw +log pulse_reset_ref.log pulse_reset_ref.scs
