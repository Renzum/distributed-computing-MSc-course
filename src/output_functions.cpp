#include <fstream>
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
    const LatticeBoltzmann::HostDistributionMirror distribution_function,
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
    const LatticeBoltzmann::HostDensityMirror density_function,
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
    const int lattice_width, const int lattice_height, std::string file_name) {

    file << "width: " << lattice_width << "\n";
    file << "height: " << lattice_width << "\n";
    file << "iterations:" << std::endl;
}

void LocalAverageVelocityFunctionOutput::add_timestep(
    const LatticeBoltzmann::HostLocalAverageVelocityMirror
        local_average_velocity_function) {
    const int grid_width = local_average_velocity_function.extent_int(0);
    const int grid_height = local_average_velocity_function.extent_int(1);

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {
            file << "  - x: " << x << "\n";
            file << "    y: " << y << "\n";
            file << "    vel_x: " << local_average_velocity_function(x, y, 0)
                 << "\n";
            file << "    vel_y: " << local_average_velocity_function(x, y, 1)
                 << std::endl;
        }
    }
}