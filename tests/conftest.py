import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))       # legacy_neuro (oracle)

from neurovat import CrossbarSNN, NetConfig, NeuronParams, STDPParams


class ToyTwin:
    """Minimal scalar device twin: conductance relaxes toward Gmax (positive
    drive) or Gmin (negative drive) at a rate set by |drive|."""
    input_kind = "voltage"

    def __init__(self):
        self.p = SimpleNamespace(Gmin=1e-6, Gmax=1e-4, tau=20e-3)
        self.reset()

    def reset(self):
        self.G = 0.5 * (self.p.Gmin + self.p.Gmax)

    def step(self, t, dt, drive):
        if drive:
            target = self.p.Gmax if drive > 0 else self.p.Gmin
            self.G = target + (self.G - target) * np.exp(-dt * abs(drive) / self.p.tau)


STDP_AMPS = dict(pot_amp=0.05, dep_amp=0.05)       # volts, for the toy devices


@pytest.fixture
def make_net():
    def make(device="softbound", rule="stdp", grid=(4, 4), n_out=4, seed=3,
             neuron=None, stdp=None, **cfg):
        return CrossbarSNN(
            device, NeuronParams(**(neuron or {})),
            STDPParams(**dict(STDP_AMPS, **(stdp or {}))),
            NetConfig(grid_h=grid[0], grid_w=grid[1], n_out=n_out,
                      learn_rule=rule, seed=seed, present_ms=60.0, **cfg))
    return make


def train(net, epochs=1):
    for _ in range(epochs):
        for k, (_label, vec) in enumerate(net.patterns):
            net.train_step(vec, net.target_of[k])
    return net
