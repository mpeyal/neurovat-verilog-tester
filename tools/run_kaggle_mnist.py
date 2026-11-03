"""Run the notebook architecture through the native app's convolutional engine."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vatester.conv_snn import ConvConfig, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train", type=int, default=60000)
    parser.add_argument("--test", type=int, default=10000)
    parser.add_argument("--output", default="results/kaggle_mnist/full")
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root/args.output
    output.mkdir(parents=True, exist_ok=True)
    def progress(event):
        (output/"progress.json").write_text(json.dumps(event, indent=2))
        if event["kind"] == "done":
            result = event["result"]
            print("DONE", result["train"]["accuracy"], result["test"]["accuracy"], flush=True)
        else:
            print(json.dumps(event), flush=True)
    train(root, output, ConvConfig(epochs=args.epochs, train_limit=args.train, test_limit=args.test), progress,
          reference=args.reference)


if __name__ == "__main__":
    main()
