"""Regression checks for honest metrics, complete checkpoints and held-out data."""
import json
import unittest
import queue
from types import SimpleNamespace
import numpy as np
from simulation_core import EcfetV1, EcfetV2, EcfetV3, FeFET
from vatester.neuro import Trainer, NeuronParams, STDPParams, NetConfig, confusion, prf1

# FeFET is a private model, absent in public builds
DEVICES = tuple(c for c in (EcfetV1, EcfetV2, EcfetV3, FeFET) if c is not None)


def make(cls=FeFET or EcfetV2, rule="stdp"):
    voltage = cls is FeFET
    return Trainer(cls, "voltage" if voltage else "current", NeuronParams(),
                   STDPParams(pot_amp=1. if voltage else 50e-12,
                              dep_amp=1. if voltage else 50e-12),
                   NetConfig(grid_h=2, grid_w=2, n_out=2, present_ms=30, learn_rule=rule),
                   patterns=[("a", np.array([1., 0., 1., 0.])),
                             ("b", np.array([0., 1., 0., 1.]))], targets=[0, 1])


class Integrity(unittest.TestCase):
    def test_rejections_count_as_errors(self):
        cm = confusion([0, 1], [0, -1], 2)
        self.assertEqual(cm.tolist(), [[1, 0, 0], [0, 0, 1]])
        m = prf1(cm)
        self.assertEqual(m["acc"], .5)
        self.assertEqual(m["support"].tolist(), [1, 1])
        self.assertEqual(m["rec"].tolist(), [1., 0.])
        self.assertEqual(prf1(confusion([0, 1], [-1, -1], 2))["acc"], 0.)

    def test_full_state_and_resume(self):
        for cls in DEVICES:
            with self.subTest(device=cls.__name__):
                tr = make(cls)
                tr.train_step(tr.patterns[0][1], 0)
                snapshot = json.loads(json.dumps(tr.capture_state(), allow_nan=False))
                tr.train_step(tr.patterns[1][1], 1)
                expected = tr.capture_state()
                tr.restore_state(snapshot)
                self.assertEqual(tr.capture_state(), snapshot)
                tr.train_step(tr.patterns[1][1], 1)
                self.assertEqual(tr.capture_state(), expected)

    def test_legacy_weights(self):
        for cls in DEVICES:
            tr = make(cls)
            tr.train_step(tr.patterns[0][1], 0)
            saved = tr.capture_weights()
            tr.train_step(tr.patterns[1][1], 1)
            tr.restore_weights(saved)
            np.testing.assert_allclose(tr.capture_weights(), saved, atol=1e-12)

    def test_evaluation_does_not_change_training_state(self):
        tr = make()
        tr.N.v_noise = .05
        before = tr.capture_state()
        a = tr.evaluate()
        self.assertEqual(tr.capture_state(), before)
        self.assertEqual(tr.evaluate(), a)
        self.assertEqual(tr.capture_state(), before)

    def test_validation_partition(self):
        tr = make()
        tr.patterns = [(str(i), np.full(4, i/20)) for i in range(20)]
        tr.target_of = [i % 2 for i in range(20)]
        tr.test_patterns = [("test", np.ones(4))]
        tr.test_targets = [0]
        tr.split_validation()
        train = {p[0] for p in tr.patterns}
        val = {p[0] for p in tr.validation_patterns}
        self.assertEqual(len(train), 16)
        self.assertEqual(len(val), 4)
        self.assertFalse(train & val)
        self.assertEqual(tr.test_patterns[0][0], "test")
        tr.split_validation()
        self.assertEqual(len(tr.patterns), 16)

    def test_missing_test_is_not_training_accuracy(self):
        with self.assertRaises(ValueError):
            make().eval_accuracy(use_test=True)

    def test_worker_never_uses_test_for_selection(self):
        from vatester.app import App
        tr = make()
        tr.validation_patterns = list(tr.patterns)
        tr.validation_targets = list(tr.target_of)
        tr.test_patterns = [("untouched", np.ones(4))]
        tr.test_targets = [1]
        calls = []
        evaluate = tr.eval_accuracy
        def checked(*args, **kwargs):
            self.assertFalse(kwargs.get("use_test", False))
            calls.append(kwargs.get("split", "train"))
            return evaluate(*args, **kwargs)
        tr.eval_accuracy = checked
        host = SimpleNamespace(trainer=tr, _trainer_stop=False, q=queue.Queue(),
                               _nt_snapshot=lambda *args: {})
        App._nt_train_worker(host, 2)
        self.assertEqual(calls, ["train", "validation", "train", "validation"])
        messages = list(host.q.queue)
        self.assertFalse(any(m[0] == "log" and "error" in m[1] for m in messages), messages)

    def test_extreme_logits_have_uncapped_finite_loss(self):
        tr = make(rule="surrogate")
        tr.N.epsp_gain = 1000.
        tr.weights_norm = lambda c: np.array([[.9, .1, .9, .1], [.1, .9, .1, .9]])
        result = tr.train_step(tr.patterns[0][1], 1)
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertGreater(result["loss"], 30.)


if __name__ == "__main__":
    unittest.main()
