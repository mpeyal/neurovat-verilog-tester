"""Pluggable synaptic device models.

    register_device("ftj", FTJArray)          # a DeviceArray subclass
    net = CrossbarSNN(device_factory("ftj", params={...}, variation={...}), ...)

A device class is constructed as
``cls(n_post, n_pre, params=None, variation=None, write_noise=0.0, seed=None)``.
"""
from .base import DeviceArray, spread
from .scalar import ScalarTwinArray, device_weight, calibrate_pot_sign
from .softbound import SoftBoundArray, SoftBoundParams
from .ecfet_v3 import EcfetV3Array

_REGISTRY = {}


def register_device(name, cls):
    """Make ``cls`` selectable by ``name`` in configs and ``device_factory``."""
    _REGISTRY[str(name).lower()] = cls


def available_devices():
    return sorted(_REGISTRY)


def device_class(name):
    try:
        return _REGISTRY[str(name).lower()]
    except KeyError:
        raise KeyError(f"Unknown device model {name!r}; available: "
                       f"{', '.join(available_devices())}") from None


def device_factory(name, params=None, variation=None, write_noise=0.0):
    """``factory(n_post, n_pre, seed)`` for a registered model - what
    :class:`neurovat.CrossbarSNN` takes as its ``device`` argument."""
    cls = device_class(name)

    def factory(n_post, n_pre, seed=None):
        return cls(n_post, n_pre, params=params, variation=variation,
                   write_noise=write_noise, seed=seed)
    return factory


def twin_factory(make_device, timing="serial", read="nv"):
    """Factory wrapping a scalar device twin (see :class:`ScalarTwinArray`)."""
    def factory(n_post, n_pre, seed=None):
        return ScalarTwinArray(make_device, n_post, n_pre, timing, read)
    return factory


def vector_twin_factory(make_device, timing="physical", read="full"):
    """Like :func:`twin_factory`, but an unmodified ECFET v3 twin without
    cycle-to-cycle noise runs as the vectorised :class:`EcfetV3Array` (same
    results to machine tolerance, checked once per factory against the
    scalar twin); any other twin keeps the scalar adapter. The factory's
    ``kind`` attribute says which one it builds ("vector" | "scalar")."""
    from .ecfet_v3 import EcfetV3Array, is_canonical_v3, self_check
    probe = make_device()
    vec = (is_canonical_v3(probe)
           and not getattr(probe.p, "sigma_c2c", 0.0)
           and self_check(make_device))

    def factory(n_post, n_pre, seed=None):
        if vec:
            return EcfetV3Array(probe.p, n_post, n_pre, timing, read)
        return ScalarTwinArray(make_device, n_post, n_pre, timing, read)
    factory.kind = "vector" if vec else "scalar"
    return factory


register_device("softbound", SoftBoundArray)
try:                                  # private FeFET model; absent in public builds
    from .fefet import FeFETArray, FeFETParams
    register_device("fefet", FeFETArray)
except ModuleNotFoundError as e:
    if e.name != __name__ + ".fefet":
        raise                         # a real import error inside the model
    FeFETArray = FeFETParams = None

__all__ = [
    "DeviceArray", "spread", "ScalarTwinArray", "SoftBoundArray",
    "SoftBoundParams", "EcfetV3Array", "FeFETArray", "FeFETParams", "device_weight",
    "calibrate_pot_sign", "register_device", "available_devices",
    "device_class", "device_factory", "twin_factory", "vector_twin_factory",
]
