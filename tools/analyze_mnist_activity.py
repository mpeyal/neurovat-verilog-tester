"""Measure whether matched output spikes carry useful class information."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.mnist_ecfet_v2 import OUT, make_trainer, app_result, raster


def main():
    with np.load(OUT/"prepared_data.npz") as archive:
        data = {s: (archive[f"{s}_x"], archive[f"{s}_y"]) for s in ("train", "validation", "test")}
    tr = make_trainer(data)
    tr.restore_state(json.loads((OUT/"selected_app_state.json").read_text()))
    with np.load(OUT/"evaluation_spikes.npz") as archive:
        bank = archive["test"]
    counts, membranes, correct, correct_count, tied = [], [], 0, 0, 0
    for vec, target, spikes in zip(*data["test"], bank):
        result = app_result(tr, vec, spikes)
        events = raster(result)
        count = events.sum(axis=0)
        counts.append(count)
        membranes.append(float(np.max(result["v_trace"])))
        correct += result["winner"] == target
        correct_count += int(count.argmax()) == target
        tied += np.count_nonzero(count == count.max()) > 1
    counts = np.asarray(counts)
    result = {"test_images": 500, "output_spike_bins": 300000,
              "output_spikes": int(counts.sum()), "occupancy": float(counts.sum()/300000),
              "mean_output_rate_hz": float(counts.mean()/.060),
              "neuron_presentations_firing_every_step": int(np.count_nonzero(counts == 60)),
              "neuron_presentations": 5000,
              "images_with_all_neurons_firing_every_step": int(np.count_nonzero((counts == 60).all(axis=1))),
              "spike_count_argmax_accuracy": float(correct_count/500),
              "spike_count_argmax_tied_images": int(tied),
              "mean_membrane_readout_accuracy": float(correct/500),
              "max_post_reset_membrane": max(membranes),
              "interpretation": "High output occupancy is saturation at the one-spike-per-step ceiling. Matched software behavior does not establish informative spike timing. Classification uses mean pre-reset membrane, not spike count. Count argmax ties select the lowest neuron index."}
    (OUT/"activity_analysis.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT/"test_output_spike_counts.npz", counts=counts, targets=data["test"][1])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
