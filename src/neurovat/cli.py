"""Command line: ``neurovat run CONFIG`` and ``neurovat devices``."""
import argparse
import json

from . import __version__
from .devices import available_devices
from .experiments import load_config, run_experiment, set_path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="neurovat", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run an experiment / sweep config")
    run.add_argument("config", help=".toml or .json experiment config")
    run.add_argument("--out", help="output directory (default: from config)")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="override a config value, e.g. train.epochs=1")
    sub.add_parser("devices", help="list the registered device models")
    args = ap.parse_args(argv)

    if args.cmd == "devices":
        print("\n".join(available_devices()))
        return 0
    cfg = load_config(args.config)
    for item in args.set:
        key, _, text = item.partition("=")
        try:
            value = json.loads(text)
        except ValueError:
            value = text
        set_path(cfg, key, value)
    result = run_experiment(cfg, out_dir=args.out)
    for row in result["summary"]:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
