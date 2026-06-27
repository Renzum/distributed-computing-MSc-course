#!/usr/bin/python3
"""
FULL DISCLOSURE: Claude AI was used to write this file
(because I am not too well versed in Python and especially matplotlib)

Reads a CSV file with columns: iteration, x, y, density
- iteration, x, y: integers
- density: float (double)

Generates one PNG per iteration showing a colour-coded 2D heatmap of the
density at each (x, y) cell.  A shared colour scale is computed across ALL
iterations so that the colour meaning stays consistent between frames.

Usage
-----
  python plot_density.py data.csv
  python plot_density.py data.csv --output-dir ./plots
  python plot_density.py data.csv --colormap plasma --output-dir ./plots
"""

import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, no GUI needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Row:
    iteration: int
    x: int
    y: int
    density: float


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def read_csv(filepath: str) -> list[Row]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        expected = {"iteration", "x", "y", "density"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV must have columns: {', '.join(sorted(expected))}. "
                f"Got: {reader.fieldnames}"
            )

        for line_num, raw in enumerate(reader, start=2):
            try:
                rows.append(Row(
                    iteration=int(raw["iteration"]),
                    x=int(raw["x"]),
                    y=int(raw["y"]),
                    density=float(raw["density"]),
                ))
            except (ValueError, KeyError) as e:
                raise ValueError(f"Parse error on line {line_num}: {e}") from e

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_iteration(
    iteration: int,
    rows: list[Row],
    max_x: int,
    max_y: int,
    vmin: float,
    vmax: float,
    colormap: str,
    output_dir: Path,
    with_labels: bool,
) -> None:
    """Render one iteration as a colour-coded density heatmap PNG."""

    grid_cols = max_x + 1
    grid_rows = max_y + 1

    # Build a 2-D array: grid[row, col] = density
    # y=0 → bottom, so we flip vertically for imshow (which has y=0 at top)
    grid = np.full((grid_rows, grid_cols), np.nan)
    for row in rows:
        grid[row.y, row.x] = row.density

    # Figure size: each cell ~0.6 inches, plus room for the colourbar
    CELL_IN = 0.6
    fig_w = grid_cols * CELL_IN + 1.8   # +1.8 for colourbar
    fig_h = grid_rows * CELL_IN + 0.9   # +0.9 for title

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(colormap)

    # imshow: origin="lower" so y=0 is at the bottom, matching CSV convention
    im = ax.imshow(
        grid,
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )

    # --- cell value labels ---
    # Choose white or black text depending on background brightness
    for row in rows:
        val = row.density
        rgba = cmap(norm(val))
        # Perceived luminance
        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        txt_color = "white" if lum < 0.45 else "black"

        if with_labels:
            val_str = (f"{val:.0f}"
                    if val == int(val) and abs(val) < 1e9
                    else f"{val:.3g}")
        else:
            val_str = ""

        ax.text(row.x, row.y, val_str,
                ha="center", va="center",
                fontsize=7, color=txt_color, fontweight="bold")

    # --- axes ---
    ax.set_xticks(range(grid_cols))
    ax.set_yticks(range(grid_rows))
    ax.set_xticklabels(range(grid_cols), fontsize=7)
    ax.set_yticklabels(range(grid_rows), fontsize=7)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.tick_params(length=0)

    # --- colourbar ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("density", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(f"Density — Iteration {iteration}",
                 fontsize=11, fontweight="bold", pad=8)

    fig.tight_layout()

    out_path = output_dir / f"density_iteration_{iteration:04d}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parallel dispatch
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> int:
    """Top-level so multiprocessing can pickle it."""
    iteration, rows, max_x, max_y, vmin, vmax, colormap, output_dir, with_labels = args
    plot_iteration(iteration, rows, max_x, max_y, vmin, vmax, colormap, output_dir, with_labels)
    return iteration


def generate_plots(rows: list[Row], output_dir: Path, colormap: str, with_labels: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    max_x = max(r.x for r in rows)
    max_y = max(r.y for r in rows)

    # Shared colour scale across all iterations for visual consistency
    vmin = min(r.density for r in rows)
    vmax = max(r.density for r in rows)

    by_iteration: dict[int, list[Row]] = defaultdict(list)
    for row in rows:
        by_iteration[row.iteration].append(row)

    iterations = sorted(by_iteration.keys())
    n_workers = min(os.cpu_count() or 1, len(iterations))

    print(f"Grid size : {max_x + 1} x {max_y + 1} cells")
    print(f"Density   : [{vmin:.4g}, {vmax:.4g}]")
    print(f"Colourmap : {colormap}")
    print(f"Iterations: {iterations}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Workers   : {n_workers}")
    print()

    work_items = [
        (it, by_iteration[it], max_x, max_y, vmin, vmax, colormap, output_dir, with_labels)
        for it in iterations
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, item): item[0] for item in work_items}
        for future in as_completed(futures):
            iteration = futures[future]
            try:
                future.result()
                print(f"  Saved: density_iteration_{iteration:04d}.png")
            except Exception as e:
                print(f"  ERROR on iteration {iteration}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot per-iteration density CSVs as colour-coded heatmap PNGs."
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory to write PNG files (default: current working directory)",
    )
    parser.add_argument(
        "--colormap", "-c",
        default="viridis",
        help="Matplotlib colormap name (default: viridis)",
    )
    parser.add_argument(
        "--labels", "-l",
        default = True,
        help = "Draw value lables at each grid cell",
        action = argparse.BooleanOptionalAction
    )
    args = parser.parse_args()

    rows = read_csv(args.csv_file)
    print(f"Loaded {len(rows)} rows from '{args.csv_file}'")

    generate_plots(rows, Path(args.output_dir), args.colormap, args.labels)
    print("\nDone.")


if __name__ == "__main__":
    main()