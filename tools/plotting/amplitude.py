#!/usr/bin/python3
"""
FULL DISCLOSURE: Claude AI was used to write this file
(because I am not too well versed in Python and especially matplotlib)

Reads a CSV file with columns: time_step, amplitude, omega
- time_step: integer
- amplitude: float
- omega: float (assumed constant across all rows)
- max_y: integer (irrelevant for this script)

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
from numpy import log

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Row:
    time_step: int
    amplitude: float
    omega: float


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

    t = [r.time_step  for r in rows_sorted]
    a = [r.amplitude  for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t, a, linewidth=1.2, color="#3a86ff", zorder=2)
    ax.scatter(t, a, s=18, color="#3a86ff", edgecolors="white",
               linewidths=0.5, zorder=3)

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$a(t)$", fontsize=12)
    ax.set_title(f"Amplitude over time  ($\\omega = {omega:g}$)", fontsize=12)

    ax.set_xlim(t[0], t[-1])
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
        default="amplitude.png",
        help="Output PNG file path (default: amplitude.png)",
    )
    args = parser.parse_args()

    rows = read_csv(args.csv_file)
    print(f"Loaded {len(rows)} rows from '{args.csv_file}'")

    plot(rows, Path(args.output))


if __name__ == "__main__":
    main()