#!/usr/bin/python3
"""
Parse velocity-field data spread across one or more files (e.g. one file per
MPI rank, each covering a subset of the lattice) and produce a streamplot for
every simulation iteration found in the data.

Expected file format (repeated per iteration):

    --- i:0
    x,y,x_vel,y_vel
    x,y,x_vel,y_vel
    ...
    --- i:1
    ...

Each file may contain all iterations, and each iteration's data may be split
across multiple files (e.g. different lattice regions per rank). Data for a
given iteration is merged across all input files before plotting.

Usage:
    python plot_streamlines.py rank0.txt rank1.txt rank2.txt --output-dir plots

    # or with a glob (shell-expanded) / explicit list
    python plot_streamlines.py data/rank_*.txt -o plots --dpi 150
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend; still works if a display exists
import matplotlib.pyplot as plt

ITER_HEADER_RE = re.compile(r"^-{2,}\s*i\s*:\s*(-?\d+)")


def parse_file(path):
    """
    Parse a single data file.

    Returns: dict mapping iteration (int) -> list of (x, y, vx, vy) tuples.
    """
    iterations = defaultdict(list)
    current_iter = None
    line_no = 0

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line_no += 1
            line = raw_line.strip()
            if not line:
                continue

            header_match = ITER_HEADER_RE.match(line)
            if header_match:
                current_iter = int(header_match.group(1))
                continue

            if current_iter is None:
                # Data encountered before any "--- i:N" header
                print(
                    f"Warning: {path}:{line_no}: data line found before any "
                    f"iteration header, skipping: {line!r}",
                    file=sys.stderr,
                )
                continue

            parts = line.split(",")
            if len(parts) != 4:
                print(
                    f"Warning: {path}:{line_no}: expected 4 comma-separated "
                    f"values, got {len(parts)}, skipping: {line!r}",
                    file=sys.stderr,
                )
                continue

            try:
                x, y = int(parts[0]), int(parts[1])
                vx, vy = float(parts[2]), float(parts[3])
            except ValueError:
                print(
                    f"Warning: {path}:{line_no}: could not parse values, "
                    f"skipping: {line!r}",
                    file=sys.stderr,
                )
                continue

            iterations[current_iter].append((x, y, vx, vy))

    return iterations


def merge_files(paths):
    """
    Parse and merge multiple files.

    Returns: dict mapping iteration (int) -> list of (x, y, vx, vy) tuples,
    combined across all files.
    """
    merged = defaultdict(list)
    for path in paths:
        file_iterations = parse_file(path)
        if not file_iterations:
            print(f"Warning: no iteration data found in {path}", file=sys.stderr)
        for it, records in file_iterations.items():
            merged[it].extend(records)
    return merged


def build_grid(records, fill_value=np.nan):
    """
    Convert a list of (x, y, vx, vy) tuples into 2D VX, VY arrays suitable
    for matplotlib.streamplot, along with the X, Y coordinate meshgrids.

    Grid extent is inferred as [0, max(x)] x [0, max(y)] from the data.
    Any (x, y) cell not present in the data is left as `fill_value`.
    """
    xs = [r[0] for r in records]
    ys = [r[1] for r in records]
    nx = max(xs) + 1
    ny = max(ys) + 1

    vx_grid = np.full((ny, nx), fill_value, dtype=float)
    vy_grid = np.full((ny, nx), fill_value, dtype=float)

    seen = set()
    for x, y, vx, vy in records:
        if (x, y) in seen:
            print(
                f"Warning: duplicate data for (x={x}, y={y}); overwriting "
                f"with the later value.",
                file=sys.stderr,
            )
        seen.add((x, y))
        vx_grid[y, x] = vx
        vy_grid[y, x] = vy

    expected_cells = nx * ny
    missing = expected_cells - len(seen)
    if missing > 0:
        print(
            f"Warning: grid has {missing} missing lattice point(s) out of "
            f"{expected_cells} (inferred grid {nx}x{ny}). These will show as "
            f"gaps unless --fill-missing is used.",
            file=sys.stderr,
        )

    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    return X, Y, vx_grid, vy_grid


def fill_missing_values(grid):
    """
    Fill NaN cells with the nearest non-NaN neighbor's value (simple nearest
    fill). Only used if --fill-missing is passed, since streamplot cannot
    handle NaNs.
    """
    if not np.isnan(grid).any():
        return grid
    try:
        from scipy.interpolate import griddata
    except ImportError:
        print(
            "Warning: --fill-missing requires scipy, which is not installed; "
            "leaving gaps as NaN (streamplot may raise an error).",
            file=sys.stderr,
        )
        return grid

    ny, nx = grid.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    valid = ~np.isnan(grid)
    filled = griddata(
        (xx[valid], yy[valid]),
        grid[valid],
        (xx, yy),
        method="nearest",
    )
    return filled


def plot_iteration(iteration, X, Y, vx_grid, vy_grid, output_dir, dpi, density):
    speed = np.sqrt(vx_grid ** 2 + vy_grid ** 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    strm = ax.streamplot(
        X, Y, vx_grid, vy_grid,
        color=speed,
        cmap="viridis",
        density=density,
        linewidth=1,
    )
    fig.colorbar(strm.lines, ax=ax, label="speed")
    ax.set_title(f"Velocity streamplot — iteration {iteration}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    out_path = Path(output_dir) / f"streamplot_iter_{iteration:04d}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a streamplot per iteration from distributed "
        "MPI velocity-field data files."
    )
    parser.add_argument(
        "files", nargs="+", help="Paths to one or more data files."
    )
    parser.add_argument(
        "-o", "--output-dir", default="streamplots",
        help="Directory to save the generated plots (default: streamplots).",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="Output image DPI (default: 150).",
    )
    parser.add_argument(
        "--density", type=float, default=1.5,
        help="Streamplot line density (default: 1.5).",
    )
    parser.add_argument(
        "--fill-missing", action="store_true",
        help="Fill missing lattice points via nearest-neighbor interpolation "
        "(requires scipy). Without this, gaps are left as NaN, which "
        "matplotlib's streamplot may error on.",
    )
    parser.add_argument(
        "--iterations", type=int, nargs="+", default=None,
        help="Only plot these specific iteration numbers (default: all "
        "iterations found in the data).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {len(args.files)} file(s)...")
    merged = merge_files(args.files)

    if not merged:
        print("No iteration data found in any input file.", file=sys.stderr)
        sys.exit(1)

    iterations_to_plot = sorted(merged.keys())
    if args.iterations is not None:
        wanted = set(args.iterations)
        missing_requested = wanted - set(iterations_to_plot)
        if missing_requested:
            print(
                f"Warning: requested iteration(s) not found in data: "
                f"{sorted(missing_requested)}",
                file=sys.stderr,
            )
        iterations_to_plot = [i for i in iterations_to_plot if i in wanted]

    print(f"Found {len(merged)} iteration(s); plotting {len(iterations_to_plot)}.")

    for iteration in iterations_to_plot:
        records = merged[iteration]
        X, Y, vx_grid, vy_grid = build_grid(records)

        if args.fill_missing:
            vx_grid = fill_missing_values(vx_grid)
            vy_grid = fill_missing_values(vy_grid)

        try:
            out_path = plot_iteration(
                iteration, X, Y, vx_grid, vy_grid,
                output_dir, args.dpi, args.density,
            )
            print(f"  iteration {iteration}: saved {out_path}")
        except ValueError as e:
            print(
                f"  iteration {iteration}: failed to plot ({e}). This "
                f"usually means there are gaps in the grid — try "
                f"--fill-missing.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()