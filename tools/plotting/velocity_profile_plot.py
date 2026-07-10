#!/usr/bin/python3

"""
Reads a lid-driven-cavity-style YAML file (walls + per-iteration velocity
fields) into typed Python objects, and renders a velocity streamplot PNG
for each iteration.

Usage:
    python read_simulation.py path/to/file.yaml
    python read_simulation.py path/to/file.yaml -o plots --density 0.5 --stride 2
    python read_simulation.py path/to/file.yaml --parallel --workers 8

Run with -h/--help for all options.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

@dataclass
class VelocityPoint:
    x: float
    y: float
    vel_x: float
    vel_y: float


@dataclass
class Iteration:
    iteration: int
    velocities: list[VelocityPoint] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Iteration":
        return cls(
            iteration=data["iteration"],
            velocities=[VelocityPoint(**v) for v in data.get("velocities", [])],
        )


@dataclass
class Simulation:
    width: int
    height: int

    right_wall_velocity_x: float
    right_wall_velocity_y: float
    bottom_wall_velocity_x: float
    bottom_wall_velocity_y: float
    left_wall_velocity_x: float
    left_wall_velocity_y: float
    top_wall_velocity_x: float
    top_wall_velocity_y: float

    iterations: list[Iteration] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Simulation":
        iterations = [
            Iteration.from_dict(it) for it in data.get("iterations", [])
        ]

        return cls(
            width=data["width"],
            height=data["height"],
            right_wall_velocity_x=data["right_wall_velocity_x"],
            right_wall_velocity_y=data["right_wall_velocity_y"],
            bottom_wall_velocity_x=data["bottom_wall_velocity_x"],
            bottom_wall_velocity_y=data["bottom_wall_velocity_y"],
            left_wall_velocity_x=data["left_wall_velocity_x"],
            left_wall_velocity_y=data["left_wall_velocity_y"],
            top_wall_velocity_x=data["top_wall_velocity_x"],
            top_wall_velocity_y=data["top_wall_velocity_y"],
            iterations=iterations,
        )

    @classmethod
    def from_yaml_doc(cls, data: dict) -> "Simulation":
        """
        Build a Simulation from just the header document (width, height,
        wall velocities) -- no `iterations` key required. Used when the
        YAML file is split into multiple `---`-separated documents and
        iterations are streamed in one at a time.
        """
        return cls.from_dict({**data, "iterations": []})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Simulation":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


def build_grid(sim: Simulation) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the X/Y meshgrid once, so it can be reused across many
    streamplot calls instead of rebuilding it every iteration."""
    x = np.arange(sim.width)
    y = np.arange(sim.height)
    return np.meshgrid(x, y)


def add_wall_velocity_annotations(sim: Simulation, ax: plt.Axes) -> None:
    """
    For each wall with a nonzero velocity, draw an arrow just outside
    that side of the grid pointing in the direction of the wall's
    (vel_x, vel_y), with a "(vel_x, vel_y)" label beneath/beside it.
    Walls with zero velocity are skipped entirely.

    Arrows/labels are placed in axes-fraction coordinates (outside the
    [0, 1] box), NOT data coordinates -- so the plotted grid always
    stays at exactly its real size (0..width, 0..height) with no
    zooming/shifting to make room. Since they render outside the axes'
    own bounding box, make sure to save with `bbox_inches="tight"` (all
    plotting functions in this module already do) or they may get
    clipped by the figure edge.
    """
    margin = 0.14      # axes-fraction space reserved outside the plot box
    arrow_len = 0.08    # axes-fraction length of each wall-velocity arrow

    # name -> (vel_x, vel_y, anchor point just outside that wall, label offset direction)
    walls = [
        ("bottom", sim.bottom_wall_velocity_x, sim.bottom_wall_velocity_y,
         (0.5, -margin * 0.3), (0, -1)),
        ("top", sim.top_wall_velocity_x, sim.top_wall_velocity_y,
         (0.5, 1 + margin * 0.3), (0, 1)),
        ("left", sim.left_wall_velocity_x, sim.left_wall_velocity_y,
         (-margin * 0.3, 0.5), (-1, 0)),
        ("right", sim.right_wall_velocity_x, sim.right_wall_velocity_y,
         (1 + margin * 0.3, 0.5), (1, 0)),
    ]

    for name, vx, vy, (ax_x, ax_y), (off_x, off_y) in walls:
        if vx == 0 and vy == 0:
            continue

        ax.spines[name].set_color("red")
        ax.spines[name].set_linewidth(2)

        mag = (vx**2 + vy**2) ** 0.5
        dx, dy = arrow_len * vx / mag, arrow_len * vy / mag

        ax.annotate(
            "",
            xy=(ax_x + dx, ax_y + dy),
            xytext=(ax_x, ax_y),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
            annotation_clip=False,
        )

        label_x = ax_x + off_x * margin * 0.45
        label_y = ax_y + off_y * margin * 0.45
        ha = "right" if off_x < 0 else "left" if off_x > 0 else "center"
        va = "top" if off_y < 0 else "bottom" if off_y > 0 else "center"
        ax.text(
            label_x, label_y, f"({vx:g}, {vy:g})",
            transform=ax.transAxes,
            ha=ha, va=va, clip_on=False, fontsize=9, color="red",
        )


def create_empty_streamplot(
    sim: Simulation,
    iteration: int | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a blank streamplot canvas sized to the simulation's grid
    (sim.width x sim.height), with a zero velocity field everywhere.

    This is meant as a starting point/scaffold -- call ax.streamplot(...)
    again later with real u/v data to actually draw streamlines on top
    of this same axes.

    If `iteration` is given, its number is displayed as a label at the
    middle top of the figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    x = np.arange(sim.width)
    y = np.arange(sim.height)
    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)

    ax.streamplot(X, Y, U, V)

    ax.set_xlim(0, sim.width)
    ax.set_ylim(0, sim.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Velocity Streamplot")

    add_wall_velocity_annotations(sim, ax)

    if iteration is not None:
        fig.text(0.5, 0.98, f"Iteration {iteration}", ha="center", va="top")

    return fig, ax


def plot_iteration_streamplot(
    sim: Simulation,
    it: Iteration,
    ax: plt.Axes | None = None,
    grid: tuple[np.ndarray, np.ndarray] | None = None,
    density: float = 1.0,
    stride: int = 1,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the velocity field of a single Iteration as a streamplot,
    sized to the simulation's grid (sim.width x sim.height).

    For batch/loop use, pass in the same `ax` and a precomputed `grid`
    (from `build_grid(sim)`) each call to avoid re-creating a Figure and
    re-building the meshgrid every iteration -- see `plot_all_iterations`.

    `density` and `stride` are the two big performance levers if plotting
    is slow (see plot_all_iterations docstring for details):
      - `density` (default 1.0): passed to streamplot; lower values seed
        fewer streamlines, which is the most reliable way to cut runtime.
      - `stride` (default 1): subsamples the grid before plotting, e.g.
        stride=2 uses every other point. Helps most when the velocity
        field is smooth; has little effect on noisy/chaotic fields.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
        ax.clear()  # reusing an existing axes -- wipe last iteration's streamlines

    X, Y = grid if grid is not None else build_grid(sim)

    # U/V must match X/Y's shape: (height, width)
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)

    # Vectorized fill instead of a Python for-loop over points
    if it.velocities:
        xs = np.array([p.x for p in it.velocities])
        ys = np.array([p.y for p in it.velocities])
        U[ys, xs] = [p.vel_x for p in it.velocities]
        V[ys, xs] = [p.vel_y for p in it.velocities]

    if stride > 1:
        X, Y = X[::stride, ::stride], Y[::stride, ::stride]
        U, V = U[::stride, ::stride], V[::stride, ::stride]

    ax.streamplot(X, Y, U, V, density=density)

    ax.set_xlim(0, sim.width)
    ax.set_ylim(0, sim.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Velocity Streamplot")

    add_wall_velocity_annotations(sim, ax)

    ax.text(
        0.5, 1.05, f"Iteration {it.iteration}",
        transform=ax.transAxes, ha="center", va="bottom",
    )

    return fig, ax


def plot_all_iterations(
    sim: Simulation,
    iterations,
    output_dir: str | Path,
    density: float = 1.0,
    stride: int = 1,
    dpi: int = 100,
) -> None:
    """
    Render every Iteration to a PNG in `output_dir`, reusing a single
    Figure/Axes and a precomputed grid across all of them.

    `iterations` can be any iterable of Iteration (e.g. a generator that
    streams them from a multi-doc YAML file, so only one is in memory
    at a time).

    If this is slow, `density` and `stride` are your main levers -- see
    plot_iteration_streamplot's docstring. Start by lowering `density`
    (e.g. 0.5); it helps regardless of how noisy the velocity field is.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid(sim)
    fig, ax = plt.subplots(figsize=(6, 6))

    for it in iterations:
        plot_iteration_streamplot(sim, it, ax=ax, grid=grid, density=density, stride=stride)
        fig.savefig(output_dir / f"iteration_{it.iteration:05d}.png", dpi=dpi, bbox_inches="tight")

    plt.close(fig)


def _render_one_iteration(args) -> str:
    """Worker function for plot_all_iterations_parallel -- must be a plain
    module-level function (not a closure/lambda) so it can be pickled and
    sent to a separate process."""
    sim, it, output_dir, density, stride, dpi = args
    fig, ax = plt.subplots(figsize=(6, 6))
    print(f"Plotting iteration: {it.iteration}")
    plot_iteration_streamplot(sim, it, ax=ax, density=density, stride=stride)
    out_path = Path(output_dir) / f"iteration_{it.iteration:05d}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved iteration: {it.iteration}")
    plt.close(fig)
    return str(out_path)


def plot_all_iterations_parallel(
    sim: Simulation,
    iterations,
    output_dir: str | Path,
    density: float = 1.0,
    stride: int = 1,
    dpi: int = 100,
    workers: int | None = None,
    max_in_flight: int | None = None,
) -> None:
    """
    Same as plot_all_iterations, but renders iterations across multiple
    CPU cores using a process pool -- each iteration's plot is fully
    independent of the others, so this parallelizes well.

    `iterations` can be any iterable, including a generator that streams
    from a multi-doc YAML file. Iterations are submitted to the pool
    lazily as they're read -- only `max_in_flight` of them exist in
    memory at once (default: workers * 2), not the whole run. Each
    submission is fire-and-forget from the caller's perspective (we
    don't collect results), but we still bound the queue so a slow pool
    can't let memory grow unboundedly while the file is read.

    Requires this module to be imported normally (`if __name__ ==
    "__main__"` guard), since ProcessPoolExecutor re-imports it in each
    worker process.
    """
    import concurrent.futures

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = workers or (os.cpu_count() or 1)
    max_in_flight = max_in_flight or workers * 2

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        in_flight: set[concurrent.futures.Future] = set()

        for it in iterations:
            if len(in_flight) >= max_in_flight:
                done, in_flight = concurrent.futures.wait(
                    in_flight, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    fut.result()  # re-raise any worker exception now, not silently

            task = (sim, it, output_dir, density, stride, dpi)
            in_flight.add(executor.submit(_render_one_iteration, task))

        for fut in concurrent.futures.as_completed(in_flight):
            fut.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a (multi-document) simulation YAML file and "
        "render a velocity streamplot for each iteration."
    )
    parser.add_argument(
        "yaml_file",
        type=Path,
        help="Path to the simulation YAML file (--- separated documents: "
        "a header doc followed by one doc per iteration).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to write iteration_NNNNN.png files to (default: ./plots).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="streamplot density; lower values render faster (default: 1.0).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Subsample the grid by this stride before plotting (default: 1, no subsampling).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Output image DPI (default: 100).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Render iterations one at a time in a single process instead "
        "of in parallel. Streams iterations from disk (lower memory) but "
        "is slower for large iteration counts.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes to use for parallel rendering "
        "(default: CPU count). Ignored with --sequential.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.yaml_file) as f:
        docs = yaml.safe_load_all(f)
        header = next(docs)
        sim = Simulation.from_yaml_doc(header)

        print(f"Grid: {sim.width} x {sim.height}")
        print(
            "Top wall velocity: "
            f"({sim.top_wall_velocity_x}, {sim.top_wall_velocity_y})"
        )

        iterations = (Iteration.from_dict(doc) for doc in docs)

        if args.sequential:
            plot_all_iterations(
                sim, iterations, args.output_dir,
                density=args.density, stride=args.stride, dpi=args.dpi,
            )
        else:
            plot_all_iterations_parallel(
                sim, iterations, args.output_dir,
                density=args.density, stride=args.stride, dpi=args.dpi,
                workers=args.workers,
            )

    print(f"Done. Plots written to {args.output_dir}/")


if __name__ == "__main__":
    main()