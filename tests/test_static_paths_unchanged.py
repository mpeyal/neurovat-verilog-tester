"""The physical-time engine fix (pulse splitting at reads, exact in-pulse
EDL decay) must not touch the static paths: ideal weights, the NeuroSim
fitted-curve arm and v3_static (serial timing, G_nv read). Their results
for fixed seeds are compared BIT FOR BIT with values recorded on master
ac92294 before the change (``data/static_paths_master.json``).

Re-record (only on a commit whose static paths are known good):
    python tests/test_static_paths_unchanged.py --record
"""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data" / "static_paths_master.json"
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).parent)]

CASES = [(arm, seed, k) for arm in ("ideal", "neurosim", "v3_static")
         for seed, k in ((1, 2), (2, 10))] + [("v3_static_scalar", 3, 2)]
AFFERENTS, PRESENTATIONS = 200, 12
SMALL = 40          # afferents of the (slow) scalar-adapter case


def _make(arm, n_pre, fit):
    import static_arms as ra
    if arm == "v3_static_scalar":
        # the legacy adapter over the scalar twin: serial timing, G_nv read
        from neurovat.devices import ScalarTwinArray
        from simulation_core.model_v3 import EcfetV3
        return ScalarTwinArray(lambda: EcfetV3(ra.v3_params()), 1, n_pre,
                               "serial", "nv")
    return ra.make_array(arm, 1, n_pre, fit)


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(a, np.float64)).tobytes()).hexdigest()


def fingerprint(arm, seed, k, fit):
    import argparse
    import static_arms as ra
    from neurovat.masquelier import calibrate_write, run
    n = SMALL if arm == "v3_static_scalar" else AFFERENTS
    P = replace(ra.base_params(argparse.Namespace(afferents=n)),
                seed=seed, pattern_frac=1.0 / k,
                base_s=PRESENTATIONS * k * 0.05)
    kw, dev = {}, None
    if arm != "ideal":
        dev = _make(arm, P.n_afferents, fit)
        cal = calibrate_write(dev, P, ra.WIDTH, 30 * ra.I_UNIT)
        lo, hi = dev.g_min, dev.g_max
        dev.set_conductance(np.full(dev.shape, lo + P.w_init * (hi - lo)))
        kw = dict(write_window_ms=ra.WINDOW_MS, read_ms=1.0,
                  **{q: cal[q] for q in ("write_gain", "dep_gain", "pot_sign",
                                         "pulse_width")})
    r = run(P, device=dev, **kw)
    out = {"weights": _h(r["weights"]), "post_t": _h(r["post_t_ms"]),
           "post_spikes": int(r["post_spikes"]), "writes": int(r["writes"])}
    if dev is not None:
        out["retained"] = _h(r["retained_weights"])
        out["cal"] = [float(cal[q]).hex() for q in ("write_gain", "dep_gain")]
    return out


@pytest.fixture(scope="module")
def fit():
    import static_arms as ra
    from static_arms import fit_v3
    return fit_v3(ra.v3_params())


@pytest.mark.parametrize("arm,seed,k", CASES)
def test_static_paths_bit_identical_to_master(arm, seed, k, fit):
    want = json.loads(DATA.read_text(encoding="utf-8"))[f"{arm}/{seed}/{k}"]
    assert fingerprint(arm, seed, k, fit) == want


if __name__ == "__main__" and "--record" in sys.argv:
    import static_arms as ra
    from static_arms import fit_v3
    f = fit_v3(ra.v3_params())
    rec = {f"{a}/{s}/{k}": fingerprint(a, s, k, f) for a, s, k in CASES}
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1))
