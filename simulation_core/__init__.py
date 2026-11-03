"""Shared simulation engine: device models, waveforms and transient analysis.

Built-in ECFET and FeFET behavioral models share this engine with device
plugins in twins/. Verilog-A source files live separately in verilog/.
"""

from .signals import Waveform
from .model_v1 import EcfetV1, V1Params
from .model_v2 import EcfetV2, V2Params
from .model_v3 import EcfetV3, V3Params
try:                                  # private FeFET model; absent in public builds
    from .model_fefet import FeFET, FeFETParams
except ModuleNotFoundError as e:
    if e.name != __name__ + ".model_fefet":
        raise                         # a real import error inside the model
    FeFET = FeFETParams = None
from .simulator import simulate, SimResult
from . import plotting

__all__ = [
    "Waveform",
    "EcfetV1", "V1Params",
    "EcfetV2", "V2Params",
    "EcfetV3", "V3Params",
    "FeFET", "FeFETParams",
    "simulate", "SimResult",
    "plotting",
]
