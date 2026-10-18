#!/usr/bin/env python3
"""Build the transparent E02 token-augmentation paper figure."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "paper_reproduction" / "runs" / "e02_token_augmentation" / "results.json"
if not RESULTS.exists():
    RESULTS = ROOT / "experiments" / "runs" / "e02_token_augmentation" / "results.json"

variants = json.loads(RESULTS.read_text())["variants"]
keys = ["no_aug", "untyped_aug", "typed_aug"]
metrics = [
    ("Possession (pp)", lambda row: row["possession_mad_pct"]),
    ("Shots", lambda row: row["shots_mad"]["avg"]),
    ("Dribbles", lambda row: row["dribbles_mad"]["avg"]),
    ("Total Events", lambda row: row["total_events_mad"]),
]
colors = ["#a65300", "#d56548", "#258f86"]

plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "text.color": "#111111",
    "axes.labelcolor": "#111111", "axes.edgecolor": "#b8b8b8",
    "xtick.color": "#111111", "ytick.color": "#111111",
    "font.family": "DejaVu Sans", "font.size": 12,
    "font.weight": "bold", "axes.labelweight": "bold",
    "axes.titlesize": 18, "axes.titleweight": "bold",
    "grid.color": "#4b4b4f", "grid.linestyle": "--", "grid.alpha": .6,
})

fig, ax = plt.subplots(figsize=(13, 4.8))
x = np.arange(len(metrics))
width = .23
for index, key in enumerate(keys):
    values = [getter(variants[key]) for _, getter in metrics]
    bars = ax.bar(
        x + (index - 1) * width, values, width,
        label=variants[key]["paper_label"], color=colors[index], zorder=3,
    )
    for metric_index, bar in enumerate(bars):
        star = " ★" if key == "typed_aug" else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{bar.get_height():.2f}{star}", ha="center", va="bottom",
            color="#111111", fontsize=10,
        )

ax.set_xticks(x, [name for name, _ in metrics], fontsize=12)
ax.set_ylabel("MAD |real − simulated|")
ax.set_title("Different Augmentation Configurations", fontsize=20, pad=18)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.05))
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(True, axis="both", zorder=0)
ax.axhline(0, color="#aaaaaa", lw=.8)
fig.set_size_inches(8.5, 6, forward=True)
fig.tight_layout(pad=1.8)
out = HERE / "plot.png"
fig.savefig(out, dpi=220, facecolor="none", transparent=True)
plt.close(fig)
print(out)
