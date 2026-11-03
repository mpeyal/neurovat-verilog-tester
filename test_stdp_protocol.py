"""Independent checks of waveform-driven timing, controls and RC shaping."""
from dataclasses import dataclass
import math
import unittest

import numpy as np

from simulation_core import FeFET, FeFETParams, EcfetV2
from vatester import stdp_protocol as p


@dataclass
class Params:
    threshold: float = 1.


class ThresholdMemory:
    """Nonvolatile device with no clock, trace, or retention variable."""
    name = "threshold memory"
    input_kind = "voltage"
    STDP_SCALE = 1.

    def __init__(self):
        self.p = Params()
        self.reset()

    def reset(self):
        self.G = 10.

    @property
    def R(self):
        return 1/self.G

    def observables(self):
        return {}

    def step(self, t, dt, voltage):
        self.G += dt*math.copysign(max(abs(voltage)-self.p.threshold, 0.), voltage)


class DriftingLinearMemory(ThresholdMemory):
    def step(self, t, dt, voltage):
        self.G += dt*(1+voltage)


def shape(kind=p.SHAPES[0], tc=.02):
    return dict(shape=kind, tail_tau=tc, tail_ratio=1., r_ohm=1e6, c_f=tc/1e6)


class ProtocolChecks(unittest.TestCase):
    def test_rectangles_have_no_timing_window_without_timing_state(self):
        model = ThresholdMemory()
        vals = [p.run_pair(model, dt, -.7, .7, .001, .001)["pair"]
                for dt in (-.05, -.005, .005, .05)]
        np.testing.assert_allclose(vals, 0., atol=1e-12)

    def test_overlap_creates_window_without_internal_trace(self):
        model = ThresholdMemory()
        near = p.run_pair(model, .002, -.7, .7, .001, .001, shape())
        far = p.run_pair(model, .2, -.7, .7, .001, .001, shape())
        reverse = p.run_pair(model, -.002, -.7, .7, .001, .001, shape())
        self.assertGreater(near["pair"], 0.)
        self.assertAlmostEqual(near["pair"], -reverse["pair"], places=10)
        self.assertEqual(near["pre"], 0.)
        self.assertEqual(near["post"], 0.)
        self.assertAlmostEqual(far["pair"], 0., places=12)
        self.assertEqual(p.run_pair(model, 0., -.7, .7, .001, .001, shape())["pair"], 0.)

    def test_external_tail_changes_timing_window(self):
        model = ThresholdMemory()
        short = p.run_pair(model, .015, -.7, .7, .001, 0., shape(tc=.005))["pair"]
        long = p.run_pair(model, .015, -.7, .7, .001, 0., shape(tc=.05))["pair"]
        self.assertGreater(long, short+1e-5)

    def test_rc_matches_analytical_high_pass_and_product_scaling(self):
        cfg = shape(p.SHAPES[2], .002)
        pieces = p.pieces(0., 1., .001, cfg)
        for start, width, value in pieces:
            mid = start+width/2
            expected = (math.exp(-mid/.002) if mid < .001 else
                        -(-math.expm1(-.001/.002))*math.exp(-(mid-.001)/.002))
            self.assertAlmostEqual(value, expected, places=12)
        other = dict(cfg, r_ohm=cfg["r_ohm"]*2, c_f=cfg["c_f"]/2)
        self.assertEqual(pieces, p.pieces(0., 1., .001, other))

    def test_linear_tail_and_companion_grids(self):
        cfg = shape(p.SHAPES[1])
        pieces = p.pieces(0., 2., .001, cfg)
        self.assertAlmostEqual(pieces[-1][0]+pieces[-1][1], .021)
        tails = [v for _, _, v in pieces[1:]]
        self.assertTrue(all(a < b < 0 for a, b in zip(tails, tails[1:])))
        waves, _ = p.waveforms(.003, -.7, .7, .001, .02, cfg)
        self.assertTrue(all(w.breakpoints == waves["pair"].breakpoints for w in waves.values()))

    def test_idle_baseline_prevents_false_interaction_from_drift(self):
        row = p.run_pair(DriftingLinearMemory(), .003, -.7, .7, .001, .02, shape())
        self.assertGreater(row["idle"], 0.)
        self.assertAlmostEqual(row["interaction"], 0., places=10)
        self.assertGreater(abs(row["pair"]-row["pre"]-row["post"]), .01)

    def test_added_rules_disabled_without_mutating_original(self):
        cases = [(EcfetV2(), "A_stdp")]
        if FeFET is not None:             # private FeFET model; absent in public builds
            cases.insert(0, (FeFET(FeFETParams(A_stdp_V=.04)), "A_stdp_V"))
        for original, parameter in cases:
            value = getattr(original.p, parameter)
            model = p.prepare_model(original)
            self.assertEqual(getattr(model.p, parameter), 0.)
            self.assertEqual(getattr(original.p, parameter), value)
            self.assertEqual(getattr(p.prepare_model(original, True).p, parameter), value)
        if FeFET is not None:
            self.assertEqual(FeFETParams().A_stdp_V, 0.)
            self.assertEqual(FeFET.STDP_OBS, "Vth (V)")

    @unittest.skipIf(FeFET is None, "private FeFET model not installed")
    def test_fefet_switching_is_measured_without_synthetic_rule(self):
        model = p.prepare_model(FeFET())
        row = p.run_pair(model, .0005, -.4, .4, .0005, .005, shape(tc=.0005))
        self.assertLess(row["pair"], -10.)  # lower Vth, higher read conductance
        self.assertEqual(model.vt_stdp, 0.)
        self.assertLess(abs(row["pre"]), abs(row["pair"])*.1)

    def test_invalid_settings_are_rejected(self):
        for width, cfg in ((0., None), (math.nan, None), (.001, shape(tc=0.)),
                           (.001, dict(shape(), tail_ratio=-1)),
                           (.001, dict(shape(p.SHAPES[2]), r_ohm=math.inf))):
            with self.assertRaises(ValueError):
                p.waveforms(.001, -.7, .7, width, .01, cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
