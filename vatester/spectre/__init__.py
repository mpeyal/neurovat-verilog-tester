"""Validate a Python twin against Cadence Spectre (GUI-free).

- psf.py       read PSF-ASCII transient files and CSV exports
- stimulus.py  one PWL stimulus for both Spectre and the twin
- deck.py      a Spectre transient deck for a two-terminal Verilog-A device
- remote.py    run Spectre on the Cadence host over SSH; read results via skillbridge
- twinrun.py   drive a twins/ model with the same PWL
- compare.py   NRMSE / maximum error / per-read conductance and the pass rule
- cli.py       `python -m vatester.spectre run|compare|deck|import`
"""

from .compare import read_metrics, verdict, waveform_metrics  # noqa: F401
from .deck import make_deck  # noqa: F401
from .psf import PSFError, find_signal, load_waveform, parse_waveform, read_psf_ascii  # noqa: F401
from .stimulus import Stimulus  # noqa: F401
