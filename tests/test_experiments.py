"""Config-driven experiments, datasets and the command line."""
import csv
import json

import numpy as np
import pytest

from neurovat import data, experiments
from neurovat.cli import main


@pytest.fixture
def npz(tmp_path):
    """Four separable classes of 8x8 images: one bright quadrant each."""
    rng = np.random.default_rng(0)

    def images(n):
        y = np.arange(n) % 4
        x = rng.integers(0, 40, (n, 8, 8))
        for k, c in enumerate(y):
            r, q = divmod(int(c), 2)
            x[k, 4 * r:4 * r + 4, 4 * q:4 * q + 4] += 200
        return x.astype(np.uint8), y
    (xt, yt), (xs, ys) = images(80), images(40)
    path = tmp_path / "quadrants.npz"
    np.savez(path, x_train=xt, y_train=yt, x_test=xs, y_test=ys)
    return str(path)


def config(npz, **sweep):
    return {
        "name": "t",
        "data": {"path": npz, "train_per_class": 8, "val_per_class": 3,
                 "test_per_class": 5},
        "network": {"grid_h": 4, "grid_w": 4, "n_out": 4, "present_ms": 40.0,
                    "learn_rule": "surrogate"},
        "stdp": {"pot_amp": 0.02, "dep_amp": 0.02, "sg_lr": 0.05},
        "train": {"epochs": 2},
        "sweep": sweep,
    }


def stable(result):
    """A result without wall-clock fields."""
    drop = ("train_s", "train_s_mean", "train_s_std")
    def strip(row):
        row = {k: v for k, v in row.items() if k not in drop}
        if "history" in row:
            row["history"] = [{k: v for k, v in h.items() if k != "train_s"}
                              for h in row["history"]]
        return row
    return ([strip(r) for r in result["runs"]],
            [strip(r) for r in result["summary"]])


# ---- configs -------------------------------------------------------------

def test_defaults_are_filled_and_unknown_keys_rejected():
    cfg = experiments.resolve({"network": {"n_out": 3}})
    assert cfg["device"]["model"] == "softbound" and cfg["network"] == {"n_out": 3}
    assert experiments.DEFAULTS["network"] == {}            # defaults untouched
    for bad in ({"nope": {}}, {"train": {"epoch": 1}}, {"network": {"jiter_ms": 1}},
                {"stdp": {"lr": 1}}, {"sweep": {"apply": "sometimes"}}):
        with pytest.raises(ValueError):
            experiments.resolve(bad)


def test_set_path_validates_the_target():
    cfg = experiments.resolve({})
    experiments.set_path(cfg, "network.jitter_ms", 2.0)
    experiments.set_path(cfg, "device.variation.tau_w", 0.1)
    experiments.set_path(cfg, "device.write_noise", 0.05)
    experiments.set_path(cfg, "name", "x")
    assert cfg["network"]["jitter_ms"] == 2.0 and cfg["name"] == "x"
    assert cfg["device"]["variation"] == {"tau_w": 0.1}
    for bad in ("network.jiter_ms", "device.modle", "nothing.here", "nope"):
        with pytest.raises(ValueError):
            experiments.set_path(cfg, bad, 1)


def test_toml_and_json_configs_load_the_same(tmp_path):
    (tmp_path / "c.toml").write_text(
        'name = "s"\n[network]\nn_out = 4\n[device.variation]\ntau_w = 0.1\n'
        '[sweep]\nparameter = "network.jitter_ms"\nvalues = [0, 2]\n')
    (tmp_path / "c.json").write_text(json.dumps(
        {"name": "s", "network": {"n_out": 4},
         "device": {"variation": {"tau_w": 0.1}},
         "sweep": {"parameter": "network.jitter_ms", "values": [0, 2]}}))
    assert (experiments.load_config(tmp_path / "c.toml")
            == experiments.load_config(tmp_path / "c.json"))


# ---- sweeps --------------------------------------------------------------

def test_sweep_writes_reproducible_results(npz, tmp_path):
    cfg = config(npz, parameter="device.variation.tau_w", values=[0.0, 0.3],
                 seeds=[1, 2])
    a = experiments.run_experiment(cfg, tmp_path / "a", log=lambda *_: None)
    b = experiments.run_experiment(cfg, tmp_path / "b", log=lambda *_: None)
    assert stable(a) == stable(b)
    assert [(r["value"], r["seed"]) for r in a["runs"]] == [
        (0.0, 1), (0.3, 1), (0.0, 2), (0.3, 2)]
    assert [s["n_seeds"] for s in a["summary"]] == [2, 2]
    for r in a["runs"]:
        assert {"test_acc", "test_acc_noisy", "val_acc", "train_acc", "train_s",
                "writes", "write_effort", "history"} <= r.keys()
        assert r["writes"] > 0 and r["write_effort"] > 0
    assert a["summary"][0]["test_acc_mean"] > 0.6            # it learned

    saved = json.loads((tmp_path / "a" / "results.json").read_text())
    assert saved["runs"] == json.loads(json.dumps(a["runs"]))
    prov = saved["provenance"]
    assert prov["data"][0]["sha256"] == data.sha256(npz) and prov["config_sha256"]
    # the saved config alone regenerates the results
    again = experiments.run_experiment(
        experiments.load_config(tmp_path / "a" / "config.json"), tmp_path / "c",
        log=lambda *_: None)
    assert stable(again) == stable(a)
    with open(tmp_path / "a" / "summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [float(r["value"]) for r in rows] == [0.0, 0.3]
    assert float(rows[0]["test_acc_mean"]) == pytest.approx(
        a["summary"][0]["test_acc_mean"])


def test_variation_sweep_changes_the_devices_not_the_data(npz):
    cfg = experiments.resolve(config(npz))
    loaded = experiments.load_data(cfg)
    plain = experiments.build(cfg, 1, loaded)
    experiments.set_path(cfg, "device.variation.tau_w", 0.3)
    varied = experiments.build(cfg, 1, loaded)
    assert plain.rng.bit_generator.state == varied.rng.bit_generator.state
    for (_la, va), (_lb, vb) in zip(plain.patterns, varied.patterns):
        np.testing.assert_array_equal(va, vb)
    assert len(plain.test_patterns) == 20 and len(plain.validation_patterns) == 12
    step = lambda net: (net.train_step(net.patterns[0][1], net.target_of[0]),
                        net.weights_S())[1]
    assert not np.array_equal(step(plain), step(varied))


def test_test_time_sweep_trains_once_per_seed(npz, tmp_path):
    cfg = config(npz, parameter="network.input_noise", values=[0.0, 3.0],
                 seeds=[1], apply="test")
    res = experiments.run_experiment(cfg, tmp_path / "o", log=lambda *_: None)
    quiet, jittered = res["runs"]
    assert quiet["writes"] == jittered["writes"]              # same trained net
    assert quiet["history"] == jittered["history"]
    assert quiet["test_acc"] == jittered["test_acc"]          # clean score
    assert jittered["test_acc_noisy"] < quiet["test_acc_noisy"]
    with pytest.raises(ValueError):
        experiments.run_experiment(
            config(npz, parameter="device.write_noise", values=[0.1], apply="test"),
            tmp_path / "x", log=lambda *_: None)


def test_builtin_patterns_need_no_dataset(tmp_path):
    res = experiments.run_experiment(
        {"data": {"builtin": "bars"}, "train": {"epochs": 2},
         "network": {"grid_h": 4, "grid_w": 4, "n_out": 4, "present_ms": 40.0},
         "stdp": {"pot_amp": 0.05, "dep_amp": 0.05}},
        tmp_path / "o", log=lambda *_: None)
    (run,) = res["runs"]
    assert run["parameter"] is None and "val_acc" not in run
    assert run["test_acc"] == run["train_acc"]                # no test split


def test_cli_runs_a_config_with_overrides(npz, tmp_path, capsys):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(config(npz, parameter="network.input_noise",
                                      values=[0.0, 0.2])))
    out = tmp_path / "out"
    assert main(["run", str(path), "--out", str(out), "--set", "train.epochs=1",
                 "--set", "sweep.seeds=[4]"]) == 0
    saved = json.loads((out / "results.json").read_text())
    assert saved["config"]["train"]["epochs"] == 1
    assert [r["seed"] for r in saved["runs"]] == [4, 4]
    lines = capsys.readouterr().out.strip().splitlines()
    assert json.loads(lines[-1])["value"] == 0.2
    assert main(["devices"]) == 0
    assert "softbound" in capsys.readouterr().out.split()


# ---- datasets ------------------------------------------------------------

def test_images_are_area_averaged_and_normalised():
    img = np.zeros((2, 8, 8))
    img[0, :4, :4] = 10.0
    img[1] = np.arange(64).reshape(8, 8)
    v = data.to_vectors(img, 2, 2)
    np.testing.assert_allclose(v[0], [1, 0, 0, 0])
    np.testing.assert_allclose(v[1], [0, 4 / 36, 32 / 36, 1], rtol=1e-6)
    np.testing.assert_allclose(data.to_vectors(img, 2, 2, invert=True), 1 - v)
    assert data.to_vectors(np.zeros((1, 28, 28)), 4, 4).max() == 0     # blank
    assert data.to_vectors(np.zeros((0, 28, 28)), 4, 4).shape == (0, 16)
    # non-integer ratio: every source pixel still carries equal total weight
    w = data._area_weights(28, 5)
    np.testing.assert_allclose(w.sum(1), 1.0)
    np.testing.assert_allclose(w.sum(0), 5 / 28)


def test_colour_images_become_grayscale():
    rgb = np.random.default_rng(0).random((3, 8, 8, 3)).astype(np.float32)
    gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    np.testing.assert_allclose(data.to_gray(rgb), gray)
    np.testing.assert_allclose(data.to_gray(np.moveaxis(rgb, -1, 1)), gray)
    assert data.to_vectors(rgb, 4, 4).shape == (3, 16)


def test_dataset_splits_are_balanced_disjoint_and_seeded(npz, tmp_path):
    ds = data.load_dataset(npz, 4, 4, 8, 3, 5, seed=1)
    for split, per in ((ds.train, 8), (ds.validation, 3), (ds.test, 5)):
        assert np.bincount(split[1]).tolist() == [per] * 4
        assert split[0].shape[1] == 16 and split[0].dtype == np.float32
    again = data.load_dataset(npz, 4, 4, 8, 3, 5, seed=1)
    np.testing.assert_array_equal(ds.train[0], again.train[0])
    other = data.load_dataset(npz, 4, 4, 8, 3, 5, seed=2)
    assert not np.array_equal(ds.train[0], other.train[0])
    assert ds.meta["sha256"] == data.sha256(npz)
    assert data.load_dataset(npz, 4, 4, 8, 3, 5, n_classes=2).class_names == ["0", "1"]

    # single x/y archive: the test split is held out of the same images
    with np.load(npz) as z:
        np.savez(tmp_path / "xy.npz", x=np.arange(80)[:, None, None]
                 * np.ones((1, 8, 8)), y=z["y_train"])
    one = data.load_dataset(tmp_path / "xy.npz", 1, 1, 8, 3, 5)
    assert [len(s[1]) for s in (one.train, one.validation, one.test)] == [32, 12, 20]


def test_pickled_archives_are_refused(tmp_path):
    path = tmp_path / "evil.npz"
    np.savez(path, x_train=np.array([{"a": 1}], dtype=object), y_train=np.zeros(1))
    with pytest.raises(ValueError):
        data.load_npz(path)


def test_mnist_archive_is_hash_pinned(tmp_path):
    (tmp_path / "mnist.npz").write_bytes(b"not mnist")
    with pytest.raises(ValueError, match="SHA-256"):
        data.fetch_mnist(tmp_path)
