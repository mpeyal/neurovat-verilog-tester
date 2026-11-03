"""Small offline correctness checks; does not run the Kaggle/MNIST benchmark."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
import torch

from vatester.model_spec import Project, PRESETS, layer, preset
from vatester.snn_builder import SequentialSNN, SurrogateSpike, evaluate, load_data, load_weights, loss_and_scores, save_weights, train

torch.set_num_threads(2)


class BuilderTests(unittest.TestCase):
    def test_presets_and_errors(self):
        for name in PRESETS:
            p = preset(name)
            p.validate()
        p = preset("MNIST CNN (Kaggle)")
        self.assertEqual(sum(r["parameters"] for r in p.validate()), 29826)
        p.layers.pop(6)
        with self.assertRaisesRegex(ValueError, "Flatten"):
            p.validate()
        p = preset("Offline demo")
        p.layers[-2]["units"] = 5
        with self.assertRaisesRegex(ValueError, "End the network"):
            p.validate()
        for field, value in (("lr", float("nan")), ("steps", 0), ("batch_size", True), ("seed", -1)):
            p = preset("Offline demo")
            setattr(p, field, value)
            with self.assertRaises(ValueError):
                p.validate()

    def test_analytic_lif_and_gradients(self):
        p = preset("Offline demo")
        p.steps = 4
        p.layers = [layer("Flatten"), layer("Dense", units=4, bias=False), layer("LIF", beta=.5)]
        net = SequentialSNN(p)
        with torch.no_grad():
            net.ops[1].weight.fill_(.75/64)
        image = torch.ones(1, 1, 8, 8)
        out = net(image, record=True)
        torch.testing.assert_close(out["membrane"][:, 0, 0], torch.tensor([.75, 1.125, .3125, .90625]))
        self.assertEqual(out["spikes"][:, 0, 0].tolist(), [0, 1, 0, 0])
        torch.testing.assert_close(out["spikes"], net(image)["spikes"])
        for kind in ("Fast sigmoid", "Arctangent"):
            u = torch.tensor([-.2, 0., .3], requires_grad=True)
            SurrogateSpike.apply(u, 5., kind).sum().backward()
            expected = 1/(1+5*u.detach().abs())**2 if kind == "Fast sigmoid" else 2.5/(1+(torch.pi*5*u.detach()/2)**2)
            torch.testing.assert_close(u.grad, expected)
        loss, _ = loss_and_scores(out, torch.tensor([2]), "Spike rate")
        loss.backward()
        self.assertGreater(float(net.ops[1].weight.grad.abs().sum()), 0)
        p.layers[-1]["reset"] = "zero"
        zero = SequentialSNN(p)
        with torch.no_grad():
            zero.ops[1].weight.copy_(net.ops[1].weight)
        torch.testing.assert_close(zero(image)["membrane"][:, 0, 0], torch.tensor([.75, 1.125, 0., .75]))

    def test_encodings_and_readouts(self):
        p = preset("Offline demo"); p.steps = 4
        images = torch.zeros(2, 1, 8, 8)
        images[:, 0, 0, :3] = torch.tensor([0., .5, 1.])
        p.encoding = "Latency"
        net = SequentialSNN(p)
        spikes = net.encode(images)
        self.assertEqual(spikes[:, 0, 0, 0, 0].sum().item(), 0)
        self.assertEqual(spikes[:, 0, 0, 0, 2].tolist(), [1, 0, 0, 0])
        p.encoding = "Poisson rate"
        net = SequentialSNN(p)
        together = net.encode(images, sample_ids=[0, 1])
        separately = torch.cat([net.encode(images[i:i+1], sample_ids=[i]) for i in range(2)], dim=1)
        torch.testing.assert_close(together, separately)
        out = {"spikes": torch.tensor([[[1., 0.]], [[0., 1.]], [[1., 0.]]]), "membrane": torch.tensor([[[0., 2.]]]*3)}
        target = torch.tensor([0])
        loss, scores = loss_and_scores(out, target, "Spike rate")
        manual = sum(torch.nn.functional.cross_entropy(step, target) for step in out["spikes"])/3
        torch.testing.assert_close(loss, manual)
        self.assertEqual(scores.argmax().item(), 0)
        self.assertEqual(loss_and_scores(out, target, "Membrane")[1].argmax().item(), 1)

    def test_generic_cnn_matches_existing_notebook_engine(self):
        # Small random input, no download and no notebook training.
        from vatester.conv_snn import ConvConfig, ConvSNN
        p = preset("MNIST CNN (Kaggle)"); p.steps = 3
        generic = SequentialSNN(p)
        fixed = ConvSNN(ConvConfig(steps=3))
        with torch.no_grad():
            for source, destination in ((fixed.conv1, generic.ops[0]), (fixed.conv2, generic.ops[3]), (fixed.fc1, generic.ops[7])):
                destination.weight.copy_(source.weight); destination.bias.copy_(source.bias)
        x = torch.rand(2, 1, 28, 28)
        expected, actual = fixed(x, record=True), generic(x)
        torch.testing.assert_close(actual["spikes"], expected["spikes"], rtol=0, atol=0)
        torch.testing.assert_close(actual["membrane"], expected["membrane"], rtol=0, atol=0)
        y = torch.tensor([1, 2])
        loss_and_scores(actual, y, p.loss)[0].backward()
        loss_and_scores(expected, y, p.loss)[0].backward()
        for source, destination in ((fixed.conv1, generic.ops[0]), (fixed.conv2, generic.ops[3]), (fixed.fc1, generic.ops[7])):
            torch.testing.assert_close(source.weight.grad, destination.weight.grad, rtol=1e-5, atol=1e-7)

    def test_checkpoint_transaction_and_project_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = preset("Offline demo")
            path = Path(tmp)
            p.save(path/"project.json")
            self.assertEqual(Project.load(path/"project.json"), p)
            net, other = SequentialSNN(p), SequentialSNN(p)
            save_weights(net, path/"weights.npz")
            load_weights(other, path/"weights.npz")
            for a, b in zip(net.parameters(), other.parameters()):
                torch.testing.assert_close(a, b)
            before = {n: t.clone() for n, t in other.named_parameters()}
            arrays = {n: np.zeros_like(t.detach().numpy()) for n, t in net.named_parameters()}
            arrays[list(arrays)[-1]][0] = np.nan
            np.savez(path/"bad.npz", **arrays)
            with self.assertRaises(ValueError):
                load_weights(other, path/"bad.npz")
            for n, t in other.named_parameters():
                torch.testing.assert_close(t, before[n])

    def test_training_data_separation_stop_and_real_updates(self):
        p = preset("Offline demo")
        p.train_limit, p.test_limit, p.epochs, p.steps, p.batch_size = 80, 20, 2, 4, 13
        with tempfile.TemporaryDirectory() as tmp:
            data = load_data(tmp, p)
            self.assertEqual(sum(len(data[s][1]) for s in ("train", "validation")), 80)
            train_rows = {row.numpy().tobytes() for row in data["train"][0]}
            self.assertTrue(all(row.numpy().tobytes() not in train_rows for row in data["validation"][0]))
            self.assertTrue(all(row.numpy().tobytes() not in train_rows for row in data["test"][0]))
            events = []
            net, _, result = train(tmp, Path(tmp)/"run", p, events.append)
            self.assertEqual([e["kind"] for e in events].count("epoch"), 2)
            self.assertEqual(result["metrics"]["train"]["count"], 72)  # Includes final partial batch.
            self.assertEqual(result["metrics"]["test"]["count"], 20)
            self.assertEqual(json.loads((Path(tmp)/"run/status.json").read_text())["state"], "complete")
            torch.manual_seed(p.seed)
            initial = SequentialSNN(p)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial.parameters(), net.parameters())))
            clone = SequentialSNN(p); load_weights(clone, Path(tmp)/"run/weights.npz")
            self.assertEqual(evaluate(clone, data["test"]), result["metrics"]["test"])
            # The loaded model is chosen on validation, not test accuracy.
            self.assertEqual(result["best_epoch"], max(result["history"], key=lambda r: (r["validation_accuracy"], -r["validation_loss"]))["epoch"])
            event = threading.Event(); event.set()
            _, _, stopped = train(tmp, Path(tmp)/"stopped", p, stop=event)
            self.assertIsNone(stopped)
            self.assertEqual(json.loads((Path(tmp)/"stopped/status.json").read_text())["state"], "stopped")

    def test_invalid_custom_data(self):
        p = preset("Offline demo"); p.dataset = "NPZ file"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"data.npz"; p.dataset_path = str(path)
            np.savez(path, x_train=np.zeros((40, 8, 8)), y_train=np.zeros(40, dtype=np.int64), x_test=np.zeros((20, 8, 8)), y_test=np.zeros(20, dtype=np.int64))
            p.train_limit, p.test_limit = 40, 20
            with self.assertRaisesRegex(ValueError, "Class 1"):
                load_data(tmp, p)


if __name__ == "__main__":
    unittest.main()
