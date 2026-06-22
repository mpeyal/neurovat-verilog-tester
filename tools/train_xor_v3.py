"""Headless XOR training + test on the ECFET v3 (paper-faithful) synapse.

Mirrors how the GUI's Neuromorphic Trainer builds/trains/tests a network, but
with no dearpygui: build a v3 device factory (A_stdp silenced - in a crossbar
the NETWORK rule drives plasticity, the device is the analog weight store),
train the 2-input XOR with a hidden layer + surrogate-grad (BPTT), then TEST on
the 4 canonical logic rows (deterministic clean infer) and the full noisy set.

XOR is not linearly separable -> needs a hidden layer and the surrogate rule;
the device-local STDP rule cannot carve the XOR partition.

Run:  python tools/train_xor_v3.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecfet import EcfetV3, V3Params               # noqa: E402
from vatester import neuro                          # noqa: E402


# ---- XOR data: complementary [A,~A,B,~B] coding on a 2x2 grid -------------
def _base(a, b):
    return np.array([float(a), 1.0 - a, float(b), 1.0 - b], np.float32)


CORNERS = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]   # a, b, a^b


def xor_train_set(per_corner=24, noise=0.12, seed=1234):
    """Many jittered copies of each XOR corner (the xor_aug recipe) so the
    constant predictor costs real loss and BPTT converges reliably."""
    rng = np.random.default_rng(seed)
    pats, tgts = [], []
    for k in range(per_corner):
        for (a, b, y) in CORNERS:
            v = np.clip(_base(a, b) + rng.normal(0, noise, 4).astype(np.float32),
                        0.0, 1.0)
            pats.append((f"A={a} B={b}->{y} #{k}", v))
            tgts.append(y)
    return pats, tgts


def xor_logic_rows():
    """The 4 noise-free canonical rows - the real XOR truth table to test on."""
    return ([(f"A={a} B={b}->{y}", _base(a, b)) for (a, b, y) in CORNERS],
            [y for (_, _, y) in CORNERS])


def v3_factory():
    """v3 device twin with the device's own STDP lock-in silenced (network-mode
    convention from app._nt_device_factory)."""
    counter = {"n": 0}

    def make():
        p = V3Params.paper(A_stdp=0.0, seed=1000 + counter["n"])
        counter["n"] += 1
        return EcfetV3(p)
    return make


def build_trainer(hidden=(8,), seed=1, present_ms=120.0, sg_lr=0.1):
    N = neuro.NeuronParams()                       # GUI defaults
    S = neuro.STDPParams(pot_amp=50e-12, dep_amp=50e-12,
                         pulse_width=10e-3, sg_lr=sg_lr)
    cfg = neuro.NetConfig(grid_h=2, grid_w=2, n_out=2, hidden_layers=hidden,
                          mode="supervised", learn_rule="surrogate",
                          present_ms=present_ms, dt_ms=1.0, seed=seed)
    pats, tgts = xor_train_set()
    tr = neuro.Trainer(v3_factory(), "current", N, S, cfg,
                       patterns=pats, targets=tgts)
    te_p, te_t = xor_logic_rows()
    tr.test_patterns, tr.test_targets = te_p, te_t   # held-out = the 4 logic rows
    tr.class_names = ["0", "1"]
    return tr


def train(tr, epochs=40):
    pats = tr.patterns
    n = len(pats)
    rng = np.random.default_rng(tr.cfg.seed + 777)
    for e in range(epochs):
        for k in rng.permutation(n):
            k = int(k)
            tr.train_step(pats[k][1], tr.target_of[k])
    return tr


def report(tr, tag):
    # canonical XOR truth table (deterministic clean infer)
    te_p, te_t = xor_logic_rows()
    print(f"\n=== {tag} : XOR truth table (clean infer) ===")
    correct = 0
    for (lbl, vec), y in zip(te_p, te_t):
        res = tr.infer(vec)
        w = res["winner"]
        ok = (w == y)
        correct += ok
        print(f"  {lbl:14s} pred={w} counts={res['n_out_spikes']} "
              f"{'OK' if ok else 'X'}")
    print(f"  logic-row accuracy: {correct}/4 = {correct/4:.0%}")

    # full noisy set metrics
    yt, yp = tr.evaluate(use_test=False)
    cm = neuro.confusion(yt, yp, 2)
    m = neuro.prf1(cm)
    print(f"  noisy train-set acc={m['acc']:.3f}  macro-F1={m['macro_f1']:.3f}  "
          f"confusion(rows=true)=\n   {cm.tolist()}")
    return correct == 4


def logic_acc(tr):
    te_p, te_t = xor_logic_rows()
    return sum(int(tr.infer(v)["winner"] == y)
               for (_, v), y in zip(te_p, te_t))


if __name__ == "__main__":
    # XOR is fiddly: sweep seeds, train long, keep the first seed that nails all
    # 4 logic rows (and, as a fallback, the best seen).
    HIDDEN, LR, EPOCHS = (12,), 0.2, 90
    best = (-1, None)
    solved = None
    for seed in range(24):
        tr = build_trainer(hidden=HIDDEN, seed=seed, sg_lr=LR)
        train(tr, epochs=EPOCHS)
        acc = logic_acc(tr)
        print(f"seed={seed:2d}  logic-rows={acc}/4")
        if acc > best[0]:
            best = (acc, seed)
        if acc == 4:
            solved = seed
            break
    print("\n" + ("=" * 50))
    use_seed = solved if solved is not None else best[1]
    tr = build_trainer(hidden=HIDDEN, seed=use_seed, sg_lr=LR)
    train(tr, epochs=EPOCHS)
    report(tr, f"FINAL hidden={HIDDEN} seed={use_seed} lr={LR} epochs={EPOCHS}")
    print(f"\n{'SOLVED' if solved is not None else 'BEST'} at seed={use_seed}")
