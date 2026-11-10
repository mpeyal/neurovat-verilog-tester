"""NeuroVAT core: crossbar + LIF spiking-network simulation on device models.

Pure NumPy, no GUI imports. The synaptic device sits behind
:class:`neurovat.devices.DeviceArray`, so a FeFET, FeCAP, MFMIS FeFET or FTJ
model plugs in without touching the network code.
"""

from .config import NeuronParams, STDPParams, NetConfig
from .network import CrossbarSNN
from .devices import (DeviceArray, ScalarTwinArray, SoftBoundArray,
                      register_device, device_factory, available_devices)
from .metrics import confusion, prf1
from .power import PowerParams, EnergyMeter

__version__ = "0.1.0"

__all__ = [
    "NeuronParams", "STDPParams", "NetConfig", "CrossbarSNN",
    "DeviceArray", "ScalarTwinArray", "SoftBoundArray",
    "register_device", "device_factory", "available_devices",
    "confusion", "prf1", "PowerParams", "EnergyMeter", "__version__",
]
