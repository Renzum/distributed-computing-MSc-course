#!/usr/bin/python3
"""
FULL DISCLOSURE: Claude AI was used to write this file
(because I am not too well versed in Python and especially matplotlib)

Reads a CSV file with columns: iteration, x, y, direction, value
- iteration, x, y, direction: integers
- value: float (double)

Generates one PNG per iteration showing a 2D lattice where each cell
contains a 3x3 sub-lattice with blue circles positioned by direction.

Direction -> 3x3 position mapping:
  0 -> Center       (1,1)
  1 -> Right        (1,2)
  2 -> Top          (0,1)
  3 -> Left         (1,0)
  4 -> Bottom       (2,1)
  5 -> Top Right    (0,2)
  6 -> Top Left     (0,0)
  7 -> Bottom Left  (2,0)
  8 -> Bottom Right (2,2)
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, faster than any GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Row:
    iteration: int
    x: int
    y: int
    direction: int
    value: float


# (sub_row, sub_col) within the 3x3 cell, row 0 = top
DIRECTION_TO_SUB = {
    0: (1, 1),  # Center
    1: (1, 2),  # Right
    2: (0, 1),  # Top
    3: (1, 0),  # Left
    4: (2, 1),  # Bottom
    5: (0, 2),  # Top Right
    6: (0, 0),  # Top Left
    7: (2, 0),  # Bottom Left
    8: (2, 2),  # Bottom Right
}


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

        expected = {"iteration", "x", "y", "direction", "value"}
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
                    direction=int(raw["direction"]),
                    value=float(raw["value"]),
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
    output_dir: Path,
) -> None:
    """Render one iteration as a PNG."""

    grid_cols = max_x + 1   # number of macro cells horizontally
    grid_rows = max_y + 1   # number of macro cells vertically

    # Each macro cell occupies 3 sub-cells; add 1 sub-cell padding around the border
    CELL = 3          # sub-cells per macro cell side
    PAD  = 1          # sub-cell border padding

    total_cols = grid_cols * CELL + 2 * PAD
    total_rows = grid_rows * CELL + 2 * PAD

    # Figure sizing: each sub-cell = 0.7 inches
    SUB = 0.7
    fig_w = total_cols * SUB
    fig_h = total_rows * SUB

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_cols)
    ax.set_ylim(0, total_rows)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.patch.set_facecolor("#f5f5f5")
    ax.set_facecolor("#f5f5f5")

    # --- draw macro-cell grid lines ---
    grid_color = "#cccccc"
    for gx in range(grid_cols + 1):
        lx = PAD + gx * CELL
        ax.plot([lx, lx], [PAD, PAD + grid_rows * CELL],
                color=grid_color, linewidth=1.2, zorder=1)
    for gy in range(grid_rows + 1):
        ly = PAD + gy * CELL
        ax.plot([PAD, PAD + grid_cols * CELL], [ly, ly],
                color=grid_color, linewidth=1.2, zorder=1)

    # --- place circles ---
    circle_radius = 0.38
    circle_color  = "#3a86ff"
    text_color    = "white"

    for row in rows:
        if row.direction not in DIRECTION_TO_SUB:
            print(f"  Warning: unknown direction {row.direction} on iteration "
                  f"{row.iteration}, skipping.")
            continue

        sub_r, sub_c = DIRECTION_TO_SUB[row.direction]

        # Macro cell origin in sub-cell coordinates.
        # x -> column, y -> row.  y=0 is at the BOTTOM of the plot.
        origin_col = PAD + row.x * CELL
        origin_row = PAD + row.y * CELL   # y=0 -> bottom-left macro cell

        # Sub-cell centre (sub_r=0 is top of the macro cell -> highest y)
        cx = origin_col + sub_c + 0.5
        cy = origin_row + (CELL - 1 - sub_r) + 0.5   # flip row so top=high y

        circle = Circle((cx, cy), radius=circle_radius,
                        facecolor=circle_color, edgecolor="white",
                        linewidth=0.8, zorder=3)
        ax.add_patch(circle)

        # Format value: show as int when it's a whole number
        val_str = (f"{row.value:.0f}"
                   if row.value == int(row.value) and abs(row.value) < 1e9
                   else f"{row.value:.3g}")

        ax.text(cx, cy, val_str,
                ha="center", va="center",
                fontsize=6.5, color=text_color,
                fontweight="bold", zorder=4)

    # --- x / y axis labels for macro cells ---
    label_color = "#555555"
    for gx in range(grid_cols):
        lx = PAD + gx * CELL + CELL / 2
        ax.text(lx, PAD - 0.6, str(gx),
                ha="center", va="top", fontsize=7, color=label_color)
    for gy in range(grid_rows):
        ly = PAD + gy * CELL + CELL / 2
        ax.text(PAD - 0.6, ly, str(gy),
                ha="right", va="center", fontsize=7, color=label_color)

    ax.set_title(f"Iteration {iteration}", fontsize=11, fontweight="bold",
                 pad=6, color="#222222")

    out_path = output_dir / f"iteration_{iteration:04d}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _worker(args: tuple) -> int:
    """Top-level function so multiprocessing can pickle it for child processes."""
    iteration, rows, max_x, max_y, output_dir = args
    plot_iteration(iteration, rows, max_x, max_y, output_dir)
    return iteration


def generate_plots(rows: list[Row], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    max_x = max(r.x for r in rows)
    max_y = max(r.y for r in rows)

    by_iteration: dict[int, list[Row]] = defaultdict(list)
    for row in rows:
        by_iteration[row.iteration].append(row)

    iterations = sorted(by_iteration.keys())
    n_workers = min(os.cpu_count() or 1, len(iterations))

    print(f"Grid size : {max_x + 1} x {max_y + 1} macro cells")
    print(f"Iterations: {iterations}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Workers   : {n_workers}")
    print()

    work_items = [
        (it, by_iteration[it], max_x, max_y, output_dir)
        for it in iterations
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, item): item[0] for item in work_items}
        for future in as_completed(futures):
            iteration = futures[future]
            try:
                future.result()
                print(f"  Saved: iteration_{iteration:04d}.png")
            except Exception as e:
                print(f"  ERROR on iteration {iteration}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot CSV lattice data as PNG images, one per iteration."
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory to write PNG files (default: current working directory)",
    )
    args = parser.parse_args()

    rows = read_csv(args.csv_file)
    print(f"Loaded {len(rows)} rows from '{args.csv_file}'")

    generate_plots(rows, Path(args.output_dir))
    print("\nDone.")


if __name__ == "__main__":
    main()