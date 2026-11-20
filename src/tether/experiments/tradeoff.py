import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from tether.paths import RUNS

TITLES = {"qwen": "Qwen2.5-VL-7B", "qwen3": "Qwen3-VL-8B", "gemma": "Gemma 4 E4B"}
NAMES = {
    "base": "Base",
    "revis": "REVIS",
    "rudder": "RUDDER",
    "grains": "GrAInS",
    "vti": "VTI",
    "vti-authors": "VTI",
    "m3id": "M3ID",
    "tether": "TETHER",
}


def frontier(points):
    keep = []
    for name, x, y in points:
        if not any(x2 <= x and y2 >= y and (x2, y2) != (x, y) for _, x2, y2 in points):
            keep.append((name, x, y))
    return sorted(keep, key=lambda p: p[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=RUNS / "chair_test500")
    p.add_argument("--out", type=Path, default=Path("fig3_tradeoff.png"))
    args = p.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, model in zip(axes, TITLES):
        scores = json.loads((args.runs / model / "score.json").read_text())
        points = [(t, v["CHAIR_i"], v["recall"]) for t, v in scores.items() if t in NAMES]
        best = {n for n, _, _ in frontier(points)}
        line = frontier(points)
        ax.plot([x for _, x, _ in line], [y for _, _, y in line], "--", color="0.6", zorder=1)
        for name, x, y in points:
            color = "C3" if name == "tether" else "C0"
            ax.scatter(
                x,
                y,
                s=60,
                zorder=2,
                edgecolors=color,
                facecolors=color if name in best else "none",
                linewidths=1.5,
            )
            ax.annotate(NAMES[name], (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_title(TITLES[model])
        ax.set_xlabel("CHAIR$_i$ (%)")
        ax.set_ylabel("Recall (%)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
