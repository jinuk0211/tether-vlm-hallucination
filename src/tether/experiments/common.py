import json
import re
from pathlib import Path

from tether.paths import CONFIGS, RUNS
from tether.utils import read_jsonl, write_jsonl

BASELINES = ("revis", "rudder", "grains", "vti", "vti-authors")


def tether_settings(model):
    return json.loads((CONFIGS / "tether.json").read_text())[model]


def setting_args(setting):
    return " ".join(f"--{k} {v}" for k, v in setting.items())


def command(model, tag, data, output, start, end, setting=None):
    span = f"--model {model} --data {data} --output {output} --start {start} --end {end}"
    if tag in BASELINES:
        return f"tether.baselines.generate {span} --method {tag}"
    if tag in ("m3id", "vcd"):
        return f"tether.baselines.contrastive {span} --method {tag}"
    return f"tether.decode {span} --tag {tag} {setting_args(setting or {})}".rstrip()


def queue(units, data, chunk):
    n = len(read_jsonl(data))
    lines = []
    for model, tag, parts, setting in units:
        for s in range(0, n, chunk):
            lines.append(command(model, tag, data, parts, s, min(s + chunk, n), setting))
    return lines


def write_queues(lines, name, gpus):
    paths = []
    for g in range(gpus):
        path = RUNS / "queues" / f"{name}-{g}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(l + "\n" for l in lines[g::gpus]))
        paths.append(path)
    print(len(lines), "commands in", *paths)


def join(data, parts, out, tags):
    refs = read_jsonl(data)
    complete = True
    for tag in tags:
        rows = {}
        for f in Path(parts).glob(f"{tag}-*-*.jsonl"):
            if re.fullmatch(re.escape(tag) + r"-\d+-\d+\.jsonl", f.name):
                rows.update({r["id"]: r for r in read_jsonl(f)})
        missing = sum(r["id"] not in rows for r in refs)
        if missing:
            complete = False
            print(tag, "missing", missing)
            continue
        write_jsonl(Path(out) / f"{tag}-0-{len(refs)}.jsonl", [rows[r["id"]] for r in refs])
        print(tag, "joined", len(refs))
    return complete
