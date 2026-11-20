import argparse
import sys

from tether.experiments.common import TETHER, join, queue, write_queues
from tether.paths import RUNS, SETS

TABLE = {
    "qwen": ["base", "revis", "rudder", "grains", "vti-authors", "m3id", "tether"],
    "qwen3": ["base", "revis", "rudder", "grains", "vti-authors", "m3id", "tether"],
    "gemma": ["base", "revis", "rudder", "grains", "vti-authors", "m3id", "tether"],
}
ABLATION = {"qwen": TABLE["qwen"], "gemma": TABLE["gemma"]}
REMOVE = {"tether-nosteer": "alpha", "tether-nopen": "pen", "tether-nobelief": "lam"}


def setting(model, tag):
    s = dict(TETHER[model])
    if tag in REMOVE:
        s[REMOVE[tag]] = 0
    return s


def units(table):
    if table == "ablation":
        return [(m, t) for m in ABLATION for t in REMOVE]
    if table == "cases":
        return [("qwen", "vti-released")]
    return [(m, t) for m, tags in TABLE.items() for t in tags]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["queue", "join"])
    p.add_argument("--table", choices=["test", "dev", "ablation", "cases"], default="test")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--chunk", type=int, default=50)
    args = p.parse_args()

    name = "chair_dev100" if args.table == "dev" else "chair_test500"
    data = SETS / f"{name}.jsonl"
    todo = [
        (m, t, RUNS / f"{name}_parts" / m, setting(m, t) if t.startswith("tether") else None)
        for m, t in units(args.table)
    ]
    if args.command == "queue":
        write_queues(queue(todo, data, args.chunk), f"{name}-{args.table}", args.gpus)
    else:
        ok = all([join(data, parts, RUNS / name / m, [t]) for m, t, parts, _ in todo])
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
