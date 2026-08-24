#!/usr/bin/python3
"""
Plot 2D lattice velocity field data (e.g. from an LBM lid-driven-cavity style
simulation) as colorized streamplots, one PNG per iteration, with small
arrows + magnitude labels hinting at the boundary (wall) velocities.

Expected input: a multi-document YAML file where the first document holds
the lattice config:

    width: 200
    height: 200
    right_wall_velocity_x: 0
    right_wall_velocity_y: -0.1
    bottom_wall_velocity_x: -0.1
    bottom_wall_velocity_y: 0
    left_wall_velocity_x: 0
    left_wall_velocity_y: 0.1
    top_wall_velocity_x: 0.1
    top_wall_velocity_y: 0

and every following document (separated by `---`) is one iteration:

    iteration: 25000
    velocities:
    - x: 0
      y: 0
      vel_x: -0.0437431
      vel_y: 0.0158957
    ...

Usage:
    python plot_velocities.py data.yaml
    python plot_velocities.py data.yaml -o ./my_plots --cmap plasma --dpi 200
"""

import argparse
import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_yaml_loader():
    """Prefer the libyaml-backed C loader (much faster) if it's available."""
    try:
        from yaml import CSafeLoader as Loader
    except ImportError:
        from yaml import SafeLoader as Loader
    return Loader


def iter_documents(path):
    """Lazily yield YAML documents one at a time instead of loading the
    whole (potentially huge, thousands-of-iterations) file into memory."""
    Loader = get_yaml_loader()
    with open(path, "r") as f:
        for doc in yaml.load_all(f, Loader=Loader):
            if doc is not None:
                yield doc


def build_grid(width, height, velocities):
    """Turn the flat list of {x, y, vel_x, vel_y} points into 2D U/V arrays."""
    n = len(velocities)
    xs = np.empty(n, dtype=np.int64)
    ys = np.empty(n, dtype=np.int64)
    us = np.empty(n, dtype=np.float64)
    vs = np.empty(n, dtype=np.float64)
    for i, point in enumerate(velocities):
        xs[i] = point["x"]
        ys[i] = point["y"]
        us[i] = point["vel_x"]
        vs[i] = point["vel_y"]

    U = np.full((height, width), np.nan)
    V = np.full((height, width), np.nan)
    mask = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    U[ys[mask], xs[mask]] = us[mask]
    V[ys[mask], xs[mask]] = vs[mask]
    return U, V


def get_wall_velocities(config):
    """Read {top,bottom,left,right}_wall_velocity_{x,y} from the config doc."""
    walls = {}
    for name in ("top", "bottom", "left", "right"):
        vx = config.get(f"{name}_wall_velocity_x", 0) or 0
        vy = config.get(f"{name}_wall_velocity_y", 0) or 0
        walls[name] = (float(vx), float(vy))
    return walls


def draw_wall_hints(ax, walls, width, height, margin_frac=0.16):
    """Draw a small arrow + magnitude label for each wall with nonzero velocity,
    fully outside the data rectangle so nothing overlaps the streamplot."""
    x0, x1 = 0, width - 1
    y0, y1 = 0, height - 1
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    margin_x = margin_frac * width
    margin_y = margin_frac * height
    arrow_len = 0.55 * min(margin_x, margin_y)

    # anchor point of each wall hint, positioned in the padding OUTSIDE the
    # data rectangle rather than on its edge
    positions = {
        "top": (cx, y1 + margin_y * 0.45),
        "bottom": (cx, y0 - margin_y * 0.45),
        "left": (x0 - margin_x * 0.45, cy),
        "right": (x1 + margin_x * 0.45, cy),
    }
    # where the magnitude label sits relative to the arrow's far end
    label_gap = {
        "top": (0, margin_y * 0.18),
        "bottom": (0, -margin_y * 0.18),
        "left": (-margin_x * 0.18, 0),
        "right": (margin_x * 0.18, 0),
    }

    # outline the actual simulation domain so it's clear what the arrows
    # outside it refer to
    ax.add_patch(plt.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, edgecolor="black", linewidth=1, zorder=4,
    ))

    for name, (vx, vy) in walls.items():
        if vx == 0 and vy == 0:
            continue  # skip stationary walls entirely

        mag = float(np.hypot(vx, vy))
        px, py = positions[name]
        dx = vx / mag * arrow_len
        dy = vy / mag * arrow_len

        ax.annotate(
            "",
            xy=(px + dx, py + dy),
            xytext=(px - dx, py - dy),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2, mutation_scale=15),
            annotation_clip=False,
            zorder=5,
        )

        ox, oy = label_gap[name]
        ax.annotate(
            f"{mag:.3g}",
            xy=(px + ox, py + oy),
            fontsize=8,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            annotation_clip=False,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.5, alpha=0.85),
        )


def plot_iteration(config, iteration_doc, output_dir, cmap="viridis", dpi=150, density=1.3):
    width = int(config["width"])
    height = int(config["height"])
    walls = get_wall_velocities(config)

    velocities = iteration_doc.get("velocities", [])
    iteration = iteration_doc.get("iteration", "unknown")

    U, V = build_grid(width, height, velocities)
    speed = np.sqrt(U ** 2 + V ** 2)

    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))

    strm = ax.streamplot(
        X, Y, U, V,
        color=speed,
        cmap=cmap,
        density=density,
        linewidth=1,
        arrowsize=1,
    )
    cbar = fig.colorbar(strm.lines, ax=ax)
    cbar.set_label("Velocity magnitude")

    margin_frac = 0.16
    draw_wall_hints(ax, walls, width, height, margin_frac=margin_frac)

    pad_x, pad_y = margin_frac * width, margin_frac * height
    ax.set_xlim(-pad_x, width - 1 + pad_x)
    ax.set_ylim(-pad_y, height - 1 + pad_y)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Velocity field \u2014 iteration {iteration}")

    fig.tight_layout()

    if isinstance(iteration, int):
        filename = f"iteration_{iteration:06d}.png"
    else:
        filename = f"iteration_{iteration}.png"
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D lattice velocity fields as colorized streamplots with wall-velocity hints."
    )
    parser.add_argument("input", help="Path to the multi-document YAML file")
    parser.add_argument(
        "-o", "--output-dir", default="./plots",
        help="Directory to write PNGs to (default: ./plots)",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap for velocity magnitude")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    parser.add_argument("--density", type=float, default=1.3, help="Streamplot line density")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    docs = iter_documents(args.input)
    try:
        config = next(docs)
    except StopIteration:
        print(f"No YAML documents found in {args.input}", file=sys.stderr)
        sys.exit(1)

    count = 0
    for doc in docs:
        path = plot_iteration(
            config, doc, args.output_dir,
            cmap=args.cmap, dpi=args.dpi, density=args.density,
        )
        count += 1
        print(f"[{count}] Saved {path}", flush=True)

    if count == 0:
        print("No iteration data found in the file.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()