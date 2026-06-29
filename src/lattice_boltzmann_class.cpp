#include <Kokkos_Core.hpp>

#include <density_functions.hpp>
#include <direction_definitions.hpp>
#include <lattice_boltzmann_impl.hpp>

#include "lattice_boltzmann_class.hpp"

LatticeBoltzmann::LatticeBoltzmann(const int grid_width, const int grid_height,
                                   const double viscocity) {

    this->grid_width = grid_width;
    this->grid_height = grid_height;

    this->viscocity = viscocity;

    buffer_distribution_view =
        Kokkos::View<double ***>("Buffer Distribution Function", grid_width,
                                 grid_height, TOTAL_DIRECTIONS);
    distribution_function =
        Kokkos::View<double ***>("Current Distribution Function", grid_width,
                                 grid_height, TOTAL_DIRECTIONS);

    density_function = Kokkos::View<double **>("Current Density Function",
                                               grid_width, grid_height);
    local_average_velocity = Kokkos::View<double ***>(
        "Current Local Average Velocity", grid_width, grid_height,
        2); // Each point is a two dimensional vector

    initRandomDensity(distribution_function);

    calculate_density_function();
    calculate_local_average_velocity_function();
    calculate_equilibrium_distribution_function();
}

LatticeBoltzmann::LatticeBoltzmann(
    Kokkos::View<double ***> distribution_function, const double viscocity) {
    grid_width = distribution_function.extent_int(0);
    grid_height = distribution_function.extent_int(1);

    this->viscocity = viscocity;

    this->distribution_function = distribution_function;

    calculate_density_function();
    calculate_local_average_velocity_function();
    calculate_equilibrium_distribution_function();
}

void LatticeBoltzmann::lbm_step() {
    calculate_density_function();
    calculate_local_average_velocity_function();
    calculate_equilibrium_distribution_function();
    calculate_relaxed_distribution_function();
    streaming_step();
}

void LatticeBoltzmann::streaming_step() {
    LBMImpl::streaming_step(buffer_distribution_view, distribution_function);
    iteration_count++;
}

void LatticeBoltzmann::calculate_density_function() {
    LBMImpl::calculate_density(density_function, distribution_function);
}

void LatticeBoltzmann::calculate_local_average_velocity_function() {
    LBMImpl::calculate_local_average_velocity(
        local_average_velocity, distribution_function, density_function);
}

void LatticeBoltzmann::calculate_equilibrium_distribution_function() {
    // Calculates the equilibrium distribution and stores it in the buffer
    LBMImpl::calculate_equilibrium_distribution(
        buffer_distribution_view, density_function, local_average_velocity);
}

void LatticeBoltzmann::calculate_relaxed_distribution_function() {
    // The distribution function values become relaxed
    // ACHTUNG: The buffer remains the eq distribution
    LBMImpl::relax_distribution(distribution_function, buffer_distribution_view,
                                viscocity);
}

// Outputting the Views to Files
void LatticeBoltzmann::current_distribution_to_file() {
    distribution_output_file.output(distribution_function, iteration_count);
}

void LatticeBoltzmann::current_density_to_file() {
    density_output_file.output(density_function, iteration_count);
}

void LatticeBoltzmann::current_local_average_velocity_to_file() {
    local_average_velocity_output_file.output(local_average_velocity,
                                              iteration_count);
}
