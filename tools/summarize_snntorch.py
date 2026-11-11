"""Render the measured comparison without changing its pass/fail criteria."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder = Path(__file__).resolve().parents[1]/"results"
    report = json.loads((folder/"snntorch_validation.json").read_text(encoding="utf-8"))
    rows = report["runs"]
    equal = sum(r["app_test_accuracy"] == r["snntorch_test_accuracy"] for r in rows)
    passed = sum(r["passed"] for r in rows)
    lines = ["NeuroVAT / snnTorch controlled comparison", "",
             f"Versions: PyTorch {report['torch']}; snnTorch {report['snntorch']}",
             f"Runs: {len(rows)}; epochs/run: {report['epochs']}; train/validation/test: 16/6/8 synthetic samples",
             f"Identical final test accuracy: {equal}/{len(rows)}",
             f"Passes: {passed}/{len(rows)}; without measurable drift: "
             f"{sum(r.get('drift_free', r['passed']) for r in rows)}/{len(rows)}", "",
             "The same validation-selected checkpoint epoch is used for both sides.",
             "Gradient checks at identical states isolate algorithmic errors from accumulated drift.",
             "Limits at identical states: gradient 1e-10; weights after one step 1e-9 (relative).",
             "Limits on the free-running trajectories: loss, gradient, membrane 1e-6; weights 1e-6",
             "(relative); zero spike/prediction mismatches. 'Without drift' = the former limits",
             "(loss 1e-9; gradient 1e-8; membrane 1e-8; weight 1e-11 S) also hold.", "",
             report["scope"], "",
             "Poor accuracy remains visible: matching another implementation does not mean a configuration learns well.", "",
             "rule | device | layers | seed | app test | snnTorch test | pass"]
    for row in rows:
        lines.append(f"{row['rule']} | {row['device']} | {row['layers']} | {row['seed']} | "
                     f"{row['app_test_accuracy']:.1%} | {row['snntorch_test_accuracy']:.1%} | {row['passed']}")
    lines += ["", "Maximum measured errors across runs:"]
    for key in rows[0]["errors"]:
        lines.append(f"{key}: {max(r['errors'][key] for r in rows):.12g}")
    (folder/"snntorch_validation.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    with (folder/"snntorch_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rule", "device", "layers", "seed", "app_test_accuracy", "snntorch_test_accuracy", "strict_pass", *rows[0]["errors"]])
        for r in rows:
            writer.writerow([r["rule"], r["device"], str(r["layers"]), r["seed"], r["app_test_accuracy"], r["snntorch_test_accuracy"], r["passed"], *r["errors"].values()])
    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(rows))
    ax.barh(y-.16, [100*r["app_test_accuracy"] for r in rows], height=.30, label="NeuroVAT", color="#526fbb")
    ax.barh(y+.16, [100*r["snntorch_test_accuracy"] for r in rows], height=.30, label="snnTorch matched reference", color="#35a69a")
    ax.set_yticks(y, [f"{r['rule']} / {r['device']} / {'-'.join(map(str,r['layers']))} / seed {r['seed']}" for r in rows], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 105)
    ax.set_xlabel("Held-out test accuracy (%) — eight synthetic samples per run")
    ax.set_title(f"Same final accuracy in {equal}/{len(rows)} runs\nNumerical checks: {passed}/{len(rows)} passed", loc="left")
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.15)
    fig.tight_layout()
    fig.savefig(folder/"snntorch_validation.png", dpi=160)
    plt.close(fig)
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
