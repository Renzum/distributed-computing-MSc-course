#!/usr/bin/python3

with open("test_output.txt", "r") as file:
    content = file.read()

import re
import random
import matplotlib
matplotlib.use("Agg")   # must be set before importing pyplot when multiprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ── 1. Parse ──────────────────────────────────────────────────────────────────
pattern = re.compile(
    r"\{it:\s*(?P<it>\d+),\s*x:\s*(?P<x>\d+),\s*y:\s*(?P<y>\d+),"
    r"\s*dir:\s*(?P<dir>\d+),\s*val:\s*(?P<val>\d+)\}"
)

data = defaultdict(lambda: defaultdict(list))
for m in pattern.finditer(content):
    d = {k: int(v) for k, v in m.groupdict().items()}
    data[d["it"]][(d["x"], d["y"])].append((d["dir"], d["val"]))

# ── 2. Layout constants ───────────────────────────────────────────────────────
sub_offsets = {
    0: ( 0.00,  0.00),  # Center
    1: ( 0.28,  0.00),  # Right
    2: ( 0.00,  0.28),  # Up
    3: (-0.28,  0.00),  # Left
    4: ( 0.00, -0.28),  # Down
    5: ( 0.28,  0.28),  # Up Right
    6: (-0.28,  0.28),  # Up Left
    7: (-0.28, -0.28),  # Down Left
    8: ( 0.28, -0.28),  # Down Right
}

all_xy = {xy for it in data for xy in data[it]}
xs = sorted(set(x for x, y in all_xy))
ys = sorted(set(y for x, y in all_xy))
x_idx = {v: i for i, v in enumerate(xs)}
y_idx = {v: i for i, v in enumerate(ys)}
cols, rows = len(xs), len(ys)

# ── 3. Render function ────────────────────────────────────────────────────────
# All args passed explicitly — no globals, everything must be picklable
def render_frame(args):
    it, frame_data, x_idx, y_idx, cols, rows, xs, ys, sub_offsets = args

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal")
    ax.set_title(f"it = {it}", fontsize=14, fontweight="bold")
    ax.set_xticks(range(cols)); ax.set_xticklabels(xs)
    ax.set_yticks(range(rows)); ax.set_yticklabels(ys)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.3)

    for (cx, cy), nodes in frame_data.items():
        for direction, val in nodes:
            dx, dy = sub_offsets[direction]
            nx = x_idx[cx] + dx
            ny = y_idx[cy] + dy

            ax.add_patch(mpatches.Circle(
                (nx, ny), radius=0.08, color="steelblue", zorder=3
            ))
            ax.text(
                nx, ny, str(val),
                ha="center", va="center",
                fontsize=5, color="white", fontweight="bold", zorder=4
            )

    path = f"plots/plot_{it}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return it, path

# ── 4. Render all frames in parallel ─────────────────────────────────────────
iterations = sorted(data.keys())
max_workers = os.cpu_count() or 4

# Build picklable args — convert defaultdict to plain dict for each frame
args_list = [
    (it, dict(data[it]), x_idx, y_idx, cols, rows, xs, ys, sub_offsets)
    for it in iterations
]

png_paths = {}

print(f"Rendering {len(iterations)} frames with {max_workers} workers...")

if __name__ == "__main__":   # required guard for ProcessPoolExecutor on Windows/macOS
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(render_frame, args): args[0] for args in args_list}
        for i, future in enumerate(as_completed(futures)):
            it, path = future.result()
            png_paths[it] = path
            if i % 100 == 0:
                print(f"  {i}/{len(iterations)} done...")

    print("All frames rendered.")

# ── 4. Combine into MP4  ─────────────────────────────────────────────
import subprocess

subprocess.run([
    "ffmpeg", "-framerate", "1",
    "-i", "plots/plot_%d.png",
    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "plots/lattice_animation.mp4"
])