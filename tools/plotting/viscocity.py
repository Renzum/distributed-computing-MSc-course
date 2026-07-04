#!/usr/bin/python3
"""
FULL DISCLOSURE: Claude AI was used to write this file
(because I am not too well versed in Python and especially matplotlib)

Reads a CSV file with columns: time_step, amplitude, omega
- time_step: integer
- amplitude: float
- omega: float (assumed constant across all rows)
- max_y: integer

Plots amplitude vs time_step as a scatter/line plot, saved as a single PNG.

Usage
-----
  python plot_amplitude.py data.csv
  python plot_amplitude.py data.csv --output plot.png
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Row:
    time_step: int
    amplitude: float
    omega: float
    max_y: int


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

        expected = {"time_step", "amplitude", "omega", "max_y"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV must have columns: {', '.join(sorted(expected))}. "
                f"Got: {reader.fieldnames}"
            )

        for line_num, raw in enumerate(reader, start=2):
            try:
                rows.append(Row(
                    time_step=int(raw["time_step"]),
                    amplitude=float(raw["amplitude"]),
                    omega=float(raw["omega"]),
                    max_y=int(raw["max_y"])
                ))
            except (ValueError, KeyError) as e:
                raise ValueError(f"Parse error on line {line_num}: {e}") from e

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(rows: list[Row], output_path: Path) -> None:
    rows_sorted = sorted(rows, key=lambda r: r.time_step)
    omega = rows_sorted[0].omega

    t = np.array([r.time_step  for r in rows_sorted])
    log_a = np.log(np.array([r.amplitude  for r in rows_sorted]))

    t_10th = [t[i] for i in range(0, len(t), 10)]
    log_a_10th = [log_a[i] for i in range(0, len(log_a), 10)]

    fig, ax = plt.subplots(figsize=(10, 4))

    t_arr = np.column_stack([t, np.ones_like(t)])

    slope, c = np.linalg.lstsq(t_arr, np.asarray(log_a, dtype=np.float64))[0];

    zeta_sqr = np.square(2 * np.pi / rows[0].max_y)
    print(-slope / zeta_sqr)

    # Plot the individual points only doing every 10th
    scatter = ax.scatter(t_10th, log_a_10th, s=18, color="#3a86ff", edgecolors="white", linewidths=0.0, zorder=3)
    scatter.set_label("Natural log of the amplitude at timestep $t$")

    lsqr = ax.axline(xy1=(0, c), slope=slope)
    lsqr.set_color(color="#FF0000")

    sign = "+"
    if c < 0:
        sign = "-"

    lsqr.set_label(f"Least Square Approximation ($y = {slope:.4}x {sign} {abs(c):.4}$)")

    ax.axhline(0, color="#FF8C00", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$log_n( a(t) )$", fontsize=12)
    ax.set_title(f"Amplitude over time  ($\\omega = {omega:g}$)", fontsize=12)

    ax.set_xlim(t[0], t[-1])

    log_a.sort()
    ax.set_ylim(log_a[0], log_a[-1])

    ax.legend()

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot amplitude vs time_step from a CSV file."
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument(
        "--output", "-o",
        default="viscocity.png",
        help="Output PNG file path (default: viscocity.png)",
    )
    args = parser.parse_args()

    rows = read_csv(args.csv_file)
    print(f"Loaded {len(rows)} rows from '{args.csv_file}'")

    plot(rows, Path(args.output))


if __name__ == "__main__":
    main()