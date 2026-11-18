import json
import subprocess
import sys
from importlib.resources import files

from tether.paths import UPSTREAM


def fetch(name):
    url, commit = json.loads(files("tether").joinpath("resources/upstream.json").read_text())[name]
    target = UPSTREAM / name
    if not target.exists():
        target.mkdir(parents=True)
        for args in (
            ["init", "-q"],
            ["remote", "add", "origin", url],
            ["fetch", "-q", "--depth", "1", "origin", commit],
            ["checkout", "-q", "--detach", "FETCH_HEAD"],
        ):
            subprocess.run(["git", "-C", str(target), *args], check=True)
    head = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(name, head)


if __name__ == "__main__":
    for name in sys.argv[1:] or json.loads(files("tether").joinpath("resources/upstream.json").read_text()):
        fetch(name)
