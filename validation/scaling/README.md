# C3: NeuroVAT twin vs Spectre wall time (same machine)

Every timing here was taken on ONE machine: the Spectre host (AMD EPYC 9554,
128 logical CPUs, Spectre 24.1, Python 3.9 + numpy 2.0.2; `machine.json` in
`validation/data/scaling/`). Both sides are single-threaded: Python with
`OMP/OPENBLAS/MKL_NUM_THREADS=1`, Spectre with `+mt=1`; the log's transient
CPU / elapsed ratio is ~1.0 for every run. Each point is run 3 times, one job
at a time, and reported as the median with the min-max spread. The machine
is shared and multi-user, and the 1-minute load average is recorded with
every run.

| script | what it times |
|---|---|
| `v3_array.py` | N independent ECFET v3 devices under the C5 tier-1 write schedule (device k = C5 synapse k mod 16) |
| `network.py time` | the Masquelier stream network (`masquelier.run`), ideal weights and v3 physical time, against afferent count |
| `network.py replay` | an open-loop Spectre replay of a network run's write pulses (devices only) |
| `rram_array.py` | N independent Stanford-PKU RRAM / VTEAM devices under one validation stimulus |
| `collect.py` | fetched JSONs -> `validation/data/scaling/*.csv`, `machine.json`, `validation/figures/scaling/*.png` |
| `spectre_run.py`, `machine.py` | the Spectre runner (process-group timeout, log parsing) and the machine record |

Run them on the timing machine, from a copy of the repo:

    . <cadence env>                       # Spectre 24.1 on PATH
    python3 validation/scaling/v3_array.py bench --n 1 4 16 64 256 1024 4096 \
        --scalar-max 1024 --reps 3 --out <dir>/v3
    python3 validation/scaling/network.py time   --n 64 256 1000 2000 --out <dir>/net
    python3 validation/scaling/network.py replay --n 64 256 1000      --out <dir>/net
    python3 validation/scaling/rram_array.py --model stanford --stimulus pulse_set_ref --step 1e-11 --out <dir>/rram
    python3 validation/scaling/rram_array.py --model stanford --stimulus pulse_set_ref --step 1e-12 --no-twin --out <dir>/rram
    python3 validation/scaling/rram_array.py --model vteam --stimulus on_train --step 1e-10 --n 1 4 16 64 256 --out <dir>/rram
    # copy <dir> back, then
    python validation/scaling/collect.py <copied dir>

The Stanford deck includes the git-ignored `verilog/stanford_pku_rram_spectre.va`
(see `validation/stanford/make_decks.py --patched`).

The results and caveats are in `RESULTS.md`.
