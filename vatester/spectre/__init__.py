"""Validate a Python twin against Cadence Spectre (GUI-free).

- psf.py       read PSF-ASCII transient files and CSV exports
- stimulus.py  one PWL stimulus for both Spectre and the twin
- deck.py      Spectre decks: two-terminal device, ECFET, training replay
- remote.py    run Spectre on the Cadence host over SSH; read results via skillbridge
- twinrun.py   drive a built-in (v1/v2/v3) or twins/ model with the same PWL
- compare.py   NRMSE / maximum error / per-read conductance and the pass rule
- layer1.py    layer 1: twin against one Spectre result
- converge.py  layer 2: Spectre at several steps against the finest; twin refined
- replay.py    layer 3: replay a training run's write schedule in Spectre
- cli.py       `python -m vatester.spectre run|compare|converge|replay|import|deck`
"""

from .compare import read_metrics, verdict, waveform_metrics  # noqa: F401
from .deck import make_deck  # noqa: F401
from .psf import PSFError, find_signal, load_waveform, parse_waveform, read_psf_ascii  # noqa: F401
from .stimulus import Stimulus  # noqa: F401
