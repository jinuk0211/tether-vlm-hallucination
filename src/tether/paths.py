import os
from pathlib import Path

ROOT = Path(os.environ.get("TETHER_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
COCO = DATA / "coco"
SETS = DATA / "sets"
OUT = ROOT / "outputs"
RUNS = OUT / "runs"
UPSTREAM = ROOT / "upstream"
CONFIGS = ROOT / "configs"

GRAY = OUT / "gray448.png"
VOBJ = OUT / "vobj"
DECAY = OUT / "decay"
DIRECTIONS = OUT / "directions"
JUDGES = OUT / "judges"
