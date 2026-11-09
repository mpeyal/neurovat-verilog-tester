"""JSON-safe numerical checkpoints; no executable objects in saved files."""
import random
import numpy as np


def pack(value):
    if isinstance(value, random.Random):
        return {"__random__": pack(value.getstate())}
    if isinstance(value, np.ndarray):
        return {"__array__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [pack(v) for v in value]}
    if isinstance(value, list):
        return [pack(v) for v in value]
    if isinstance(value, dict):
        return {str(k): pack(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported checkpoint state: {type(value).__name__}")


def unpack(value):
    if isinstance(value, list):
        return [unpack(v) for v in value]
    if isinstance(value, dict):
        if "__random__" in value:
            rng = random.Random()
            rng.setstate(unpack(value["__random__"]))
            return rng
        if "__array__" in value:
            return np.asarray(value["__array__"], dtype=value["dtype"])
        if "__tuple__" in value:
            return tuple(unpack(v) for v in value["__tuple__"])
        return {k: unpack(v) for k, v in value.items()}
    return value
