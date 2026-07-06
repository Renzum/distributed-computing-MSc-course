#!/usr/bin/python3
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

def plot_iteration(iteration: Iteration, height: int, ax: plt.Axes):

    sorted_velocities = sorted(iteration.velocities, key=lambda vel: vel.y)

    x = np.asarray([vel.y for vel in sorted_velocities])
    y = np.asarray([vel.vel_x for vel in sorted_velocities])

    line = ax.plot(x, y, color="red")
    # Alternate line colors for visibility
    if iteration.iteration % 2 == 0:
        line[0].set_color("#ff0000")
    else:
        line[0].set_color("#0000ff")



def main():
    parser = argparse.ArgumentParser(
        description="Read a velocity profile YAML file and plot the u_x(y)"
        "over the y coordinates for each timestep."
    )

    parser.add_argument(
        "yaml_file",
        type=Path,
        help="The input velocity profile YAML file",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default="sheer_wave_decay.png",
        help="Output file name for the plot",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for the final plot",
    )

    args = parser.parse_args()

    print(f"Plotting {args.yaml_file} to {args.output_file}")

    with open(args.yaml_file, mode="r") as f:
        docs = yaml.safe_load_all(f)
        header = next(docs)

        height = header["height"]

        print(f"Lattice Height is {height}")

        fig, ax = plt.subplots(figsize=(6,6))

        ax.set_title("Sheer Wave Decay")

        ax.set_xlabel("$y/L_y$")

        ax.set_xlim(0, height)
        ax.set_ylabel("$u_x(y, t)$")

        for doc in docs:
            iteration = Iteration.from_dict(doc)
            print(f"Plotting Iteration: {iteration.iteration}")
            plot_iteration(iteration, height, ax)

        plt.savefig(args.output_file, dpi=args.dpi, bbox_inches='tight')

if __name__ == "__main__":
    main()