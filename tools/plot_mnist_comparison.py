"""Scientific figures from the measured MNIST comparison, without rerunning it."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1]/"results"/"mnist_ecfet_v2"
BLUE, ORANGE = "#1765a0", "#d56819"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220,
                     "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(OUT/f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT/f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def paired_raster(ax, example, title):
    for name, color, offset, marker in (("app", BLUE, -.14, "|"),
                                        ("snntorch", ORANGE, .14, ".")):
        time, neuron = np.nonzero(np.asarray(example[f"{name}_spikes"]))
        ax.scatter(time, neuron+offset, c=color, marker=marker,
                   s=24 if name == "app" else 10, linewidths=.7)
    ax.set(xlim=(-1, 60), ylim=(-.6, 9.6), yticks=range(10),
           xlabel="Time (ms)", ylabel="Output neuron", title=title)
    ax.grid(axis="y", alpha=.12)


def main():
    d = json.loads((OUT/"comparison.json").read_text())
    history = d["history"]
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, layout="constrained")
    for ax, split, title in zip(axes, ("train", "validation", "test"),
                                ("Training set · n = 1,000", "Validation · n = 200", "Test set · n = 500")):
        for name, color, style, marker in (("app", BLUE, "-", "o"), ("snntorch", ORANGE, "--", "x")):
            ax.plot(epochs, [100*h[split][f"{name}_accuracy"] for h in history],
                    color=color, linestyle=style, marker=marker,
                    markersize=6, linewidth=1.8, label="App" if name == "app" else "snnTorch")
        ax.axvline(d["best"]["app"][2], color=".6", lw=1, ls=":", label="Selected epoch")
        ax.set(title=title, xlabel="Epoch (0 = initial)", xticks=epochs, ylim=(0, 100))
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Classification accuracy (%)")
    axes[0].legend(fontsize=9, loc="lower right")
    save(fig, "accuracy_comparison")

    fig, ax = plt.subplots(figsize=(8.5, 3), layout="constrained")
    for j, name, color, style in ((0, "App", BLUE, "-"), (1, "snnTorch", ORANGE, "--")):
        ax.plot(epochs[1:], [h["mean_train_loss"][j] for h in history[1:]],
                color=color, linestyle=style, marker="o" if j == 0 else "x", label=name)
    ax.set(xlabel="Epoch", ylabel="Mean training cross entropy", xticks=epochs[1:])
    ax.legend(); ax.grid(alpha=.18)
    save(fig, "loss_comparison")

    examples = d["final"]["test"]["examples"]
    handles = [Line2D([], [], color=BLUE, marker="|", ls="", markersize=9, label="App"),
               Line2D([], [], color=ORANGE, marker=".", ls="", markersize=6, label="snnTorch")]
    fig, axes = plt.subplots(5, 2, figsize=(10, 12.5), layout="constrained")
    for ax, (target, example) in zip(axes.flat, examples.items()):
        title = f"Digit {target}   |   predicted {example['app_prediction']} / {example['snntorch_prediction']}"
        paired_raster(ax, example, title)
        thumb = ax.inset_axes([1.01, .24, .19, .48])
        thumb.imshow(example["image"], cmap="gray", vmin=0, vmax=1)
        thumb.set_axis_off()
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    save(fig, "spike_raster_all_digits")

    example = examples["0"]
    fig = plt.figure(figsize=(10, 7.2), layout="constrained")
    gs = fig.add_gridspec(3, 4, height_ratios=(1, 1.3, 1))
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(example["image"], cmap="gray")
    ax.set_title("Digit 0 · 14 × 14"); ax.set_axis_off()
    ax = fig.add_subplot(gs[0, 1:])
    time, pixel = np.nonzero(example["input_spikes"])
    ax.scatter(time, pixel, marker="|", s=8, color=".25", linewidths=.5)
    ax.set(xlabel="Time (ms)", ylabel="Input pixel", title="Identical Poisson input spikes", xlim=(-1, 60), ylim=(-1, 196))
    ax = fig.add_subplot(gs[1, :]); paired_raster(ax, example, "App and snnTorch output spikes")
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, .5))
    ax = fig.add_subplot(gs[2, :])
    neuron = int(example["app_prediction"])
    for name, color, style in (("app", BLUE, "-"), ("snntorch", ORANGE, "--")):
        ax.plot(np.arange(60), np.asarray(example[f"{name}_membrane"])[:, neuron],
                color=color, ls=style, label="App" if name == "app" else "snnTorch")
    ax.set(xlabel="Time (ms)", ylabel="Membrane (a.u.)", title=f"Post-reset membrane · output neuron {neuron}")
    ax.legend(loc="lower right"); ax.grid(alpha=.15)
    save(fig, "spike_raster_detail")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), layout="constrained")
    result = d["final"]["test"]
    for ax, name in zip(axes, ("app", "snntorch")):
        matrix = np.zeros((10, 10), dtype=int)
        for target, pred in zip(result["targets"], result[f"{name}_predictions"]):
            if pred >= 0: matrix[target, pred] += 1
        im = ax.imshow(matrix, vmin=0, vmax=50, cmap="Blues")
        for i in range(10):
            for j in range(10):
                if matrix[i, j]: ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9,
                                        color="white" if matrix[i, j] > 25 else "black")
        ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit",
               title=f"{'App' if name == 'app' else 'snnTorch'} · {100*result[f'{name}_accuracy']:.1f}%")
    fig.colorbar(im, ax=axes, shrink=.8, label="Test images")
    save(fig, "test_confusion")
    activity = json.loads((OUT/"activity_analysis.json").read_text())
    with np.load(OUT/"test_output_spike_counts.npz") as archive:
        counts = archive["counts"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    axes[0].hist(counts.ravel(), bins=np.arange(-.5, 61.5), color=BLUE)
    axes[0].set(xlabel="Spikes per neuron in 60 ms", ylabel="Neuron presentations",
                title="Most output neurons reach the ceiling", xlim=(0, 61))
    values = [activity["mean_membrane_readout_accuracy"]*100, activity["spike_count_argmax_accuracy"]*100]
    bars = axes[1].bar(["Mean membrane\nactual readout", "Spike count\ndiagnostic only"], values, color=[BLUE, ORANGE], width=.55)
    axes[1].bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=4)
    axes[1].set(ylabel="Test accuracy (%)", ylim=(0, 100), title="Recognition depends on the readout")
    save(fig, "activity_readout")
    print("Created accuracy, loss, paired raster and confusion figures")


if __name__ == "__main__":
    main()
