#include <fstream>
#include <string>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>

#include "output_functions.hpp"

IOutput::IOutput(std::string file_name) {
    file = std::ofstream(file_name, std::ios::out);
}

IOutput::~IOutput() {
    file.close();
}

DistributionFunctionOutput::DistributionFunctionOutput(std::string file_name)
    : IOutput(file_name) {
    file << "iteration,x,y,direction,value" << std::endl;
}

void DistributionFunctionOutput::output(
    const Kokkos::View<double ***> &distribution_function,
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

DensityFunctionOutput::DensityFunctionOutput(std::string file_name)
    : IOutput(file_name) {

    file << "iteration,x,y,density" << std::endl;
}

void DensityFunctionOutput::output(
    const Kokkos::View<double **> &density_function, const int &iteration) {
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
    std::string file_name)
    : IOutput(file_name) {
    file << "iteration,x,y,velocity_x,velocity_y" << std::endl;
}
void LocalAverageVelocityFunctionOutput::output(
    const Kokkos::View<double ***> &local_average_velocity_function,
    const int &iteration) {
    const int grid_width = local_average_velocity_function.extent_int(0);
    const int grid_height = local_average_velocity_function.extent_int(1);

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {
            file << iteration << ",";
            file << x << ",";
            file << y << ",";
            file << local_average_velocity_function(x, y, 0) << ",";
            file << local_average_velocity_function(x, y, 1) << std::endl;
        }
    }
}