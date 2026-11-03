"""Data-only SNN projects and shape validation; safe to use without PyTorch."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


LAYER_DEFAULTS = {
    "Conv2d": {"channels": 12, "kernel": 5, "padding": 0, "bias": True},
    "MaxPool2d": {"kernel": 2},
    "Flatten": {},
    "Dense": {"units": 10, "bias": True},
    "LIF": {"beta": .5, "threshold": 1., "reset": "subtract"},
}


def layer(kind, **values):
    return {"type": kind, **deepcopy(LAYER_DEFAULTS[kind]), **values}


def positive_int(value, name, maximum=1000000):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum:,}.")
    return value


def finite_range(value, name, low, high, include_low=True):
    if type(value) not in (int, float) or not math.isfinite(value) or not (low <= value <= high if include_low else low < value <= high):
        raise ValueError(f"{name} must be finite and {'at least' if include_low else 'above'} {low}, and at most {high}.")


@dataclass
class Project:
    version: int = 1
    name: str = "MNIST MLP"
    input_shape: list = field(default_factory=lambda: [1, 28, 28])
    classes: int = 10
    layers: list = field(default_factory=lambda: [layer("Flatten"), layer("Dense", units=128), layer("LIF", beta=.9), layer("Dense"), layer("LIF", beta=.9)])
    dataset: str = "MNIST"
    dataset_path: str = ""
    train_limit: int = 1280
    test_limit: int = 500
    validation_fraction: float = .1
    steps: int = 25
    encoding: str = "Static current"
    input_gain: float = 1.
    surrogate: str = "Fast sigmoid"
    slope: float = 25.
    loss: str = "Spike rate"
    optimizer: str = "Adam"
    lr: float = .001
    batch_size: int = 64
    epochs: int = 5
    seed: int = 17
    gradient_clip: float = 1.
    threads: int = 4

    def validate(self):
        if self.version != 1:
            raise ValueError("Unsupported project version.")
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 200:
            raise ValueError("Give the project a name (1-200 characters).")
        if not isinstance(self.input_shape, list) or len(self.input_shape) != 3:
            raise ValueError("Input shape must be [channels, height, width].")
        for size in self.input_shape:
            positive_int(size, "Input dimension", 256)
        positive_int(self.classes, "Class count", 1000)
        for name, maximum in (("steps", 200), ("epochs", 1000), ("batch_size", 1024), ("threads", 32), ("train_limit", 1000000), ("test_limit", 1000000)):
            positive_int(getattr(self, name), name.replace('_', ' ').title(), maximum)
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer from 0 to 4,294,967,295.")
        for name, bounds in {"validation_fraction": (.01, .5), "input_gain": (0., 10.), "slope": (0., 1000.), "lr": (0., 1.), "gradient_clip": (0., 100.)}.items():
            finite_range(getattr(self, name), name, *bounds, include_low=name in ("validation_fraction", "gradient_clip"))
        choices = {"dataset": ("MNIST", "Demo patterns", "NPZ file"), "encoding": ("Static current", "Poisson rate", "Latency"), "surrogate": ("Fast sigmoid", "Arctangent"), "loss": ("Spike rate", "Spike count", "Membrane"), "optimizer": ("Adam", "SGD")}
        for name, options in choices.items():
            if getattr(self, name) not in options:
                raise ValueError(f"Unknown {name}: {getattr(self, name)}")
        if not isinstance(self.dataset_path, str):
            raise ValueError("Dataset path must be text.")
        if self.dataset == "MNIST" and (self.input_shape != [1, 28, 28] or self.classes != 10):
            raise ValueError("MNIST requires a 1 x 28 x 28 input and 10 classes.")
        if self.dataset == "Demo patterns" and (self.input_shape != [1, 8, 8] or self.classes != 4):
            raise ValueError("Demo patterns require a 1 x 8 x 8 input and 4 classes.")
        return infer_shapes(self.input_shape, self.layers, self.classes)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Project must be a JSON object.")
        result = cls(**value)
        result.validate()
        return result

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 1000000:
            raise ValueError("Project JSON is too large (maximum 1 MB).")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def infer_shapes(input_shape, layers, classes=None):
    if not isinstance(layers, list) or not 1 <= len(layers) <= 32:
        raise ValueError("Use 1-32 sequential layers.")
    shape, rows, total = tuple(input_shape), [], 0
    for i, spec in enumerate(layers):
        try:
            if not isinstance(spec, dict) or spec.get("type") not in LAYER_DEFAULTS:
                raise ValueError("Choose Conv2d, MaxPool2d, Flatten, Dense or LIF.")
            kind = spec["type"]
            if set(spec) != {"type", *LAYER_DEFAULTS[kind]}:
                raise ValueError(f"{kind} has missing or unknown parameters.")
            before, count = shape, 0
            if "bias" in spec and type(spec["bias"]) is not bool:
                raise ValueError("Bias must be true or false.")
            if kind == "Conv2d":
                if len(shape) != 3:
                    raise ValueError("Conv2d needs an image-shaped input, before Flatten.")
                channels = positive_int(spec["channels"], "Channels", 512)
                kernel = positive_int(spec["kernel"], "Kernel", 15)
                padding = spec["padding"]
                if type(padding) is not int or not 0 <= padding <= 7:
                    raise ValueError("Padding must be an integer from 0 to 7.")
                count = channels*(shape[0]*kernel*kernel+int(spec["bias"]))
                shape = (channels, shape[1]+2*padding-kernel+1, shape[2]+2*padding-kernel+1)
            elif kind == "MaxPool2d":
                if len(shape) != 3:
                    raise ValueError("MaxPool2d needs image-shaped input.")
                kernel = positive_int(spec["kernel"], "Pool size", 8)
                shape = (shape[0], shape[1]//kernel, shape[2]//kernel)
            elif kind == "Flatten":
                shape = (math.prod(shape),)
            elif kind == "Dense":
                if len(shape) != 1:
                    raise ValueError("Add Flatten before a Dense layer.")
                units = positive_int(spec["units"], "Units", 4096)
                count = units*(shape[0]+int(spec["bias"]))
                shape = (units,)
            else:
                finite_range(spec["beta"], "LIF decay beta", 0, 1)
                finite_range(spec["threshold"], "LIF threshold", 0, 100, include_low=False)
                if spec["reset"] not in ("subtract", "zero"):
                    raise ValueError("Reset must be subtract or zero.")
            if min(shape) < 1:
                raise ValueError("Kernel/pooling size collapses the spatial dimensions.")
            total += count
            if total > 5000000 or math.prod(shape) > 1000000:
                raise ValueError("Network exceeds the desktop limit (5M weights / 1M values per layer).")
            rows.append({"index": i, "type": kind, "input": before, "output": shape, "parameters": count})
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Layer {i+1}: {exc}") from exc
    if classes is not None and (shape != (classes,) or layers[-1]["type"] != "LIF"):
        raise ValueError(f"End the network with Dense({classes}) then LIF; current output is {shape}.")
    if not total:
        raise ValueError("Add at least one trainable Dense or Conv2d layer.")
    return rows


PRESETS = ("MNIST MLP", "MNIST CNN (Kaggle)", "MNIST compact CNN", "Offline demo")


def preset(name):
    if name not in PRESETS:
        raise ValueError("Unknown preset.")
    result = Project(name=name)
    if name == "MNIST CNN (Kaggle)":
        result.layers = [layer("Conv2d"), layer("MaxPool2d"), layer("LIF"), layer("Conv2d", channels=64), layer("MaxPool2d"), layer("LIF"), layer("Flatten"), layer("Dense"), layer("LIF")]
        result.steps, result.lr, result.batch_size, result.gradient_clip = 50, .01, 128, 0.
    elif name == "MNIST compact CNN":
        result.layers = [layer("Conv2d", channels=8, kernel=3), layer("MaxPool2d"), layer("LIF"), layer("Flatten"), layer("Dense"), layer("LIF")]
    elif name == "Offline demo":
        result.dataset, result.input_shape, result.classes = "Demo patterns", [1, 8, 8], 4
        result.train_limit, result.test_limit, result.batch_size = 320, 80, 32
        result.steps, result.epochs, result.lr = 12, 5, .005
        result.slope = 5.
        result.layers = [layer("Flatten"), layer("Dense", units=32), layer("LIF", beta=.8, threshold=.5), layer("Dense", units=4), layer("LIF", beta=.8, threshold=.5)]
    result.validate()
    return result
