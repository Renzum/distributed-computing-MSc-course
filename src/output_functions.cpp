#include <fstream>
#include <iostream>
#include <string>

#include <direction_definitions.hpp>
#include <lattice_boltzmann.hpp>

#include "output_functions.hpp"

IOutput::IOutput(std::string file_name) {
    file = std::ofstream(file_name, std::ios::out);
}

IOutput::~IOutput() {
    file.close();
}

void DistributionFunctionOutput::output(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        distribution_function,
    const int &iteration) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                file << iteration << ",";
                file << x << ",";
                file << y << ",";
                file << dir << ",";
                file << distribution_function(x, y, dir) << std::endl;
            }
        }
    }
}

void DensityFunctionOutput::output(
    const LatticeBoltzmann::DensityFunction::HostMirror density_function,
    const int &iteration) {
    const int grid_width = density_function.extent_int(0);
    const int grid_height = density_function.extent_int(1);

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {
            file << iteration << ",";
            file << x << ",";
            file << y << ",";
            file << density_function(x, y) << std::endl;
        }
    }
}

LocalAverageVelocityFunctionOutput::LocalAverageVelocityFunctionOutput(
    std::string file_name, const int lattice_width, const int lattice_height,
    const LatticeBoltzmann::Walls &walls) {

    file = std::ofstream(file_name, std::ios::out | std::ios::binary);

    file << "width: " << lattice_width << "\n";
    file << "height: " << lattice_width << "\n";

    file << "right_wall_velocity_x: " << walls.right.vel_x << "\n";
    file << "right_wall_velocity_y: " << walls.right.vel_y << "\n";

    file << "bottom_wall_velocity_x: " << walls.bottom.vel_x << "\n";
    file << "bottom_wall_velocity_y: " << walls.bottom.vel_y << "\n";

    file << "left_wall_velocity_x: " << walls.left.vel_x << "\n";
    file << "left_wall_velocity_y: " << walls.left.vel_y << "\n";

    file << "top_wall_velocity_x: " << walls.top.vel_x << "\n";
    file << "top_wall_velocity_y: " << walls.top.vel_y << std::endl;
}

void LocalAverageVelocityFunctionOutput::add_timestep(
    const LatticeBoltzmann::VelocityProfile::HostMirror
        local_average_velocity_function,
    const int &iteration) {
    const int grid_width = local_average_velocity_function.extent_int(0);
    const int grid_height = local_average_velocity_function.extent_int(1);

    file << "---" << "\n";
    file << "iteration: " << iteration << "\n";
    file << "velocities: " << "\n";
    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {
            file << "- x: " << x << "\n";
            file << "  y: " << y << "\n";
            file << "  vel_x: " << local_average_velocity_function(x, y, 0)
                 << "\n";
            file << "  vel_y: " << local_average_velocity_function(x, y, 1)
                 << std::endl;
        }
    }
}