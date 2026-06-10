"""ECFET test bench: Python port of the Verilog-A diffusion/drift memristor
model (v1) plus an upgraded practical ECFET/ECRAM model (v2), a transient
simulator with pA-pulse waveform generation, and plotting utilities."""

from .signals import Waveform
from .model_v1 import EcfetV1, V1Params
from .model_v2 import EcfetV2, V2Params
from .model_fefet import FeFET, FeFETParams
from .simulator import simulate, SimResult
from . import plotting

__all__ = [
    "Waveform",
    "EcfetV1", "V1Params",
    "EcfetV2", "V2Params",
    "FeFET", "FeFETParams",
    "simulate", "SimResult",
    "plotting",
]
