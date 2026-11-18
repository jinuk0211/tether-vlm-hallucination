import argparse
import sys

from tether.experiments.common import join, queue, tether_settings, write_queues
from tether.paths import RUNS, SETS

MODELS = ("qwen", "gemma")
TAGS = ("base", "revis", "rudder", "grains", "tether")
JUDGES = ("base", "large", "xl")
DATA = SETS / "throne.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["queue", "join", "judges"])
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--chunk", type=int, default=100)
    args = p.parse_args()

    if args.command == "queue":
        todo = [
            (m, t, RUNS / "throne_parts" / m, tether_settings(m) if t == "tether" else None)
            for m in MODELS
            for t in TAGS
        ]
        write_queues(queue(todo, DATA, args.chunk), "throne", args.gpus)
    elif args.command == "join":
        ok = all([join(DATA, RUNS / "throne_parts" / m, RUNS / "throne" / m, TAGS) for m in MODELS])
        sys.exit(0 if ok else 1)
    else:
        lines = [
            f"tether.eval.throne judge {RUNS / 'throne' / m} {t} {j}"
            for j in JUDGES
            for m in MODELS
            for t in TAGS
        ]
        write_queues(lines, "throne-judges", args.gpus)


if __name__ == "__main__":
    main()
