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

from typing import TypeVar
import copy

import argparse
import yaml
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DataPoint:
    time_step: int
    amplitude: float

@dataclass
class Data:
    omega: float
    max_y: int
    amplitudes: list[DataPoint]

@dataclass
class ViscocityData:
    omega: float
    viscocity: float

# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def read_yaml(filepath: str) -> list[Data]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")


    data = []
    with path.open(newline="") as f:
        reader = yaml.load(f.read(), Loader=yaml.Loader)
        
        for sim in reader:
            sim_data = Data(omega=sim["omega"], max_y=sim["max_y"], amplitudes=[])

            for amplitude_data in sim["amplitudes"]:
                try:
                    sim_data.amplitudes.append(DataPoint(
                        time_step=int(amplitude_data["time_step"]),
                        amplitude=float(amplitude_data["amplitude"]),
                    ))
                except (ValueError, KeyError) as e:
                    raise ValueError(f"{e}") from e

            data.append(sim_data)

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
T = TypeVar("T")
def take_every_ten(data_list: list[T]) -> list[T]:
    return [data_list[i] for i in range(0, len(data_list), 10)]

def least_squares_approx(time_steps: list[int], log_amplitudes: list[float]) -> tuple[ float, float ]:
    # Create a 2D vertical matrix with the time_step and coefficient of 1
    # And take the transpose
    time_step_matrix = np.vstack([np.asarray(time_steps), np.ones_like(time_steps)]).T

    # Now perform a least squares approximation on the data points
    slope,c = np.linalg.lstsq(time_step_matrix, np.asarray(log_amplitudes))[0];

    return slope, c

def plot_viscocity(data: Data, output_dir_path: Path, viscocities: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    data_sorted = take_every_ten(data.amplitudes)
    omega = data.omega
    max_y = data.max_y

    time_steps = []
    log_amplitudes = []

    for data_point in data_sorted:
        time_steps.append(data_point.time_step)
        # Take natural log of the amplitudes
        log_amplitudes.append(np.log(data_point.amplitude))

    slope, c = least_squares_approx(time_steps, log_amplitudes)
    lsqr = ax.axline(xy1=(0, c), slope=slope)
    lsqr.set_color(color="#FF0000")
    sign = "+"
    if c < 0:
        sign = "-"

    lsqr.set_label(f"Least Square Approximation ($y = {slope:.4}x {sign} {abs(c):.4}$)")

    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$log_n( a(t) )$", fontsize=12)
    ax.set_title(f"Amplitude over time  ($\\omega = {omega:g}$)", fontsize=12)

    zeta_sqr = np.square(2 * np.pi / max_y)

    viscocity = (-slope / zeta_sqr) 
    viscocities.append(ViscocityData(omega, viscocity))

    # Plot the individual points only doing every 10th
    scatter = ax.scatter(time_steps, log_amplitudes, s=18, color="#3a86ff", edgecolors="white", linewidths=0.0, zorder=3)
    scatter.set_label("Natural log of the amplitude at timestep $t$")

    ax.set_xlim(time_steps[0], time_steps[-1])
    ax.set_ylim(-100, 100)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=9)
    ax.legend()
    fig.tight_layout()

    output_dir_path.parent.mkdir(parents=True, exist_ok=True)
    joined = output_dir_path.joinpath(f"viscocity-iterations={len(data.amplitudes)}:max_y={max_y}:omega={omega}.png")

    fig.savefig(joined, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {joined.resolve()}")



def plot_amplitude(data: Data, output_dir_path: Path) -> None:
    omega = data.omega

    every_ten = take_every_ten(data.amplitudes)
    time_steps = np.array([data_point.time_step for data_point in every_ten ])
    amplitudes = np.array([data_point.amplitude for data_point in every_ten ])

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(time_steps, amplitudes, linewidth=1.2, color="#ff0000", zorder=2)
    scatter = ax.scatter(time_steps, amplitudes, s=18, color="#3a86ff", edgecolors="white",
               linewidths=0.0, zorder=3)
    scatter.set_label("Amplitudes at each time step $t$")

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$a(t)$", fontsize=12)
    ax.set_title(f"Amplitude over time  ($\\omega = {omega:g}$)", fontsize=12)

    ax.set_xlim(time_steps[0], time_steps[-1])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=9)

    ax.legend()

    fig.tight_layout()

    joined = output_dir_path.joinpath(f"amplitude-iterations={len(data.amplitudes)}:max_y={data.max_y}:omega={omega}.png")

    fig.savefig(joined, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {joined.resolve()}")

def plot_all_viscocities(viscocities: list[ViscocityData], output_dir: Path) -> None:
    sorted_viscocities = sorted(viscocities, key=lambda n: n.omega)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Draw the Chapman-Enskog
    omega_space = np.linspace(0.0001, 2, 100)
    visc_function = (1/3) * ((1/omega_space) - 1/2)

    visc_line = ax.plot(omega_space, visc_function)[0]
    visc_line.set_color("#ff0000")
    visc_line.set_label("Chapman-Enskog Viscocity Approximation")

    # Plot the viscocity and omega values
    omega_list = [data.omega for data in viscocities]
    viscocity_list = [data.viscocity for data in viscocities]

    scatter = ax.scatter(omega_list, viscocity_list)
    scatter.set_color("#0000ff")
    scatter.set_label("Measured Simulation Viscocities")
    scatter.set_zorder(10)

    ax.set_xlabel(r"$\omega$", fontsize=12)
    ax.set_ylabel(r"$\nu(\omega)$", fontsize=12)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 10)

    ax.legend()

    joined = output_dir.joinpath(f"all_viscocities.png")

    fig.savefig(joined, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {joined.resolve()}")


def plot_amp_visc(data: list[Data], output_dir: Path):
    # Create the path if doesn't exist
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    viscocities: list[ViscocityData] = []
    for data_point in data:
        plot_amplitude(data_point, output_dir)
        plot_viscocity(data_point, output_dir, viscocities)

    plot_all_viscocities(viscocities, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot amplitude vs time_step from a CSV file."
    )
    parser.add_argument("yaml_file", help="Path to the YAML file")
    parser.add_argument(
        "--output-dir", "-o",
        default="./",
        help="Output directory path (default: current working directory)",
    )
    args = parser.parse_args()

    data = read_yaml(args.yaml_file)

    plot_amp_visc(data, Path(args.output_dir))


if __name__ == "__main__":
    main()