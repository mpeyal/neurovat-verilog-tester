"""Pre-registered analysis of the C6 ablation re-run (see PREREG.md).

Reads runs.jsonl files written by run_ablation.py and reports, per (gap, arm):
learnt / n (learnt = last-third hit rate > 0.9; a non-finite hit rate is not
learnt), strict success / n (hit > 98 %, no false alarm, latency < 10 ms, as
run_ablation.py records it), and the paired exact sign tests (McNemar) fixed
in PREREG.md.

    python validation/ablation/prereg_analysis.py \
        --main validation/ablation/out/prereg/runs.jsonl \
        --extra validation/ablation/out/prereg/attribution_gap450/runs.jsonl \
                validation/ablation/out/prereg/gain_control_gap450/runs.jsonl \
        --out validation/ablation/out/prereg/analysis.json
"""
import argparse
import json
import math
from pathlib import Path

LEARNT_HIT = 0.9
ALPHA = 0.05
GAP_PRIMARY = 450

# (label, arm A, arm B, gap_ms); the test is two-sided, A vs B paired by seed
PRIMARY = ("v3_time vs v3_static", "v3_time", "v3_static", GAP_PRIMARY)
SECONDARY = (
    ("v3_time vs gain control", "v3_time", "v3_static_gain", GAP_PRIMARY),
    ("v3_time vs ideal", "v3_time", "ideal", GAP_PRIMARY),
    ("tau3-only vs G_nv-only", "v3_time_p3", "v3_time_nv", GAP_PRIMARY),
)


def learnt(r):
    h = r.get("hit_rate")
    return h is not None and math.isfinite(h) and h > LEARNT_HIT


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def sign_test(b, c):
    """Exact two-sided sign test on the discordant pairs (b: A only, c: B only)."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = sum(math.comb(n, i) for i in range(m + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def paired(rows, a, b, gap, outcome=learnt):
    A = {r["seed"]: outcome(r) for r in rows if r["arm"] == a and r["gap_ms"] == gap}
    B = {r["seed"]: outcome(r) for r in rows if r["arm"] == b and r["gap_ms"] == gap}
    seeds = sorted(set(A) & set(B))
    n10 = sum(A[s] and not B[s] for s in seeds)
    n01 = sum(B[s] and not A[s] for s in seeds)
    return {"arm_a": a, "arm_b": b, "gap_ms": gap, "n_pairs": len(seeds),
            "missing": sorted(set(A) ^ set(B)),
            "a_count": sum(A[s] for s in seeds), "b_count": sum(B[s] for s in seeds),
            "a_only": n10, "b_only": n01, "p": sign_test(n10, n01)}


def counts(rows):
    out = {}
    for r in rows:
        d = out.setdefault((r["gap_ms"], r["arm"]), {"n": 0, "learnt": 0, "success": 0})
        d["n"] += 1
        d["learnt"] += learnt(r)
        d["success"] += bool(r["success"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", nargs="+", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows = load(a.main + a.extra)
    c = counts(rows)
    for (gap, arm), d in sorted(c.items()):
        print(f"gap {gap:g} ms  {arm:15s} learnt {d['learnt']}/{d['n']}  "
              f"success {d['success']}/{d['n']}")
    res = {"learnt_hit": LEARNT_HIT, "alpha": ALPHA,
           "counts": [{"gap_ms": g, "arm": arm, **d} for (g, arm), d in sorted(c.items())]}
    tests = []
    for i, (label, x, y, gap) in enumerate((PRIMARY,) + SECONDARY):
        if not any(r["arm"] == x for r in rows) or not any(r["arm"] == y for r in rows):
            continue
        t = {"label": label, "role": "primary" if i == 0 else "secondary",
             "outcome": "learnt", **paired(rows, x, y, gap)}
        tests.append(t)
        if i == 0:
            ts = {"label": label + " (strict success)", "role": "secondary",
                  "outcome": "success",
                  **paired(rows, x, y, gap, outcome=lambda r: bool(r["success"]))}
            tests.append(ts)
    for t in tests:
        print(f"[{t['role']}] {t['label']} ({t['outcome']}, gap {t['gap_ms']:g}): "
              f"{t['a_count']}/{t['n_pairs']} vs {t['b_count']}/{t['n_pairs']}, "
              f"discordant {t['a_only']}:{t['b_only']}, exact p = {t['p']:.4g}")
    res["tests"] = tests
    prim = next((t for t in tests if t["role"] == "primary"), None)
    if prim is not None:
        ok = prim["p"] < ALPHA and prim["a_only"] > prim["b_only"]
        res["c6_confirmed"] = bool(ok)
        print("C6 (primary):", "CONFIRMED" if ok else "NOT CONFIRMED")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
