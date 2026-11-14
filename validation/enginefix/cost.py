"""Wall time per stream-second of the ablation arms on the C6 config
(1000 afferents, k = 10, 8 presentations = 4 s of stream, seed 1).

    python validation/enginefix/cost.py v3_time,v3_static,ideal [--out f.json]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "validation" / "ablation")]

import run_ablation as ra  # noqa: E402
from neurosim_curve import fit_v3  # noqa: E402


def main(argv):
    arms = argv[0].split(",") if argv else ["v3_time"]
    fit = fit_v3(ra.v3_params())
    out = {}
    for arm in arms:
        t0 = time.perf_counter()
        r = ra.one_job((arm, 1, 10, 8, 1000, 1.0, fit))
        w = time.perf_counter() - t0
        out[arm] = {"wall_s": round(w, 2), "stream_s": r["seconds"],
                    "s_per_stream_s": round(w / r["seconds"], 3),
                    "post_spikes": r["post_spikes"], "writes": r["writes"]}
        print(arm, out[arm], flush=True)
    if "--out" in argv:
        Path(argv[argv.index("--out") + 1]).write_text(
            json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
