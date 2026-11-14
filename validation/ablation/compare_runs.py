"""Compare two run_ablation.py runs.jsonl files run by run, keyed by (arm, seed, k).

Reports per (k, arm): runs in both, runs whose every field except wall_s is
identical, learnt flips (hit > 0.9), strict-success flips and hit-rate
changes.

    python validation/ablation/compare_runs.py OLD/runs.jsonl NEW/runs.jsonl
"""
import json
import sys


def load(p):
    with open(p, encoding="utf-8") as fh:
        return {(r["arm"], r["seed"], r["k"]): r
                for r in (json.loads(x) for x in fh if x.strip())}


def strip(r):
    return {k: v for k, v in r.items() if k != "wall_s"}


def main(argv=None):
    a_path, b_path = (argv or sys.argv[1:])[:2]
    a, b = load(a_path), load(b_path)
    keys = sorted(set(a) & set(b), key=lambda x: (x[2], x[0], x[1]))
    groups = {}
    for key in keys:
        x, y = a[key], b[key]
        g = groups.setdefault((key[2], key[0]), dict(n=0, identical=0, learnt_flip=[],
                                                    success_flip=[], hit_changed=0))
        g["n"] += 1
        g["identical"] += json.dumps(strip(x), sort_keys=True) == json.dumps(strip(y), sort_keys=True)
        if (x["hit_rate"] > 0.9) != (y["hit_rate"] > 0.9):
            g["learnt_flip"].append(key[1])
        if bool(x["success"]) != bool(y["success"]):
            g["success_flip"].append(key[1])
        g["hit_changed"] += x["hit_rate"] != y["hit_rate"]
    tot = dict(n=0, identical=0, learnt_flip=0, success_flip=0, hit_changed=0)
    for (k, arm), g in groups.items():
        print(f"k={k} {arm:15s} n {g['n']:3d} identical {g['identical']:3d} "
              f"learnt flips {g['learnt_flip']} success flips {g['success_flip']} "
              f"hit changed {g['hit_changed']}")
        for q in tot:
            tot[q] += len(g[q]) if isinstance(g[q], list) else g[q]
    print("total", tot, "only in one file:", len(set(a) ^ set(b)))
    return groups, tot


if __name__ == "__main__":
    main()
