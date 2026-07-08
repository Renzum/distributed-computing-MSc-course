#pragma once

#include <direction_definitions.hpp>
#include <tuple>

#include <Kokkos_Core.hpp>

namespace LatticeBoltzmann {

using DistributionFunction = Kokkos::View<double **[TOTAL_DIRECTIONS]>;
using HostDistributionMirror = DistributionFunction::HostMirror;

using DensityFunction = Kokkos::View<double **>;
using HostDensityMirror = DensityFunction::HostMirror;

using LocalAverageVelocity = Kokkos::View<double **[2]>;
using HostLocalAverageVelocityMirror = LocalAverageVelocity::HostMirror;

/**
 * Struct containing all the necessariy Kokkos::Views to represent the
 * different functions of Lattice-Boltzmann
 */
struct Functions {
    /**
     * 3D Lattice of x and y size where each cell is an array of 9 elements
     * representing all the directions for particles
     */
    DistributionFunction distribution_function;
    HostDistributionMirror host_distribution_function;

    /**
     * 3D Lattice of x and y size where each cell is an array of 9 elements
     * representing all the directions for particles.
     *
     * Used as a buffer for storing temporary results.
     */
    DistributionFunction buffer_distribution_function;

    /**
     * 2D Lattice of x and y size where each cell represents the density of
     * the distribution function
     */
    DensityFunction density_function;
    HostDensityMirror host_density_function;

    /**
     * 3D Lattice of x and y size where each cell holds an array of size 2
     * that represents a 2D velocity vector
     */
    LocalAverageVelocity local_average_velocity;
    HostLocalAverageVelocityMirror host_local_average_velocity;

    /**
     * Allocates the necessary views with the provided lattice width and
     * height
     */
    Functions(const int grid_width, const int grid_height);
};

void streaming_step_with_periodic_bounds(
    DistributionFunction buffer_distribution_view,
    DistributionFunction distribution_function, const int ghost_layers = 0);
inline void streaming_step_with_periodic_bounds(const Functions &functions,
                                                const int ghost_layers = 0) {
    streaming_step_with_periodic_bounds(functions.buffer_distribution_function,
                                        functions.distribution_function,
                                        ghost_layers);
}

void calculate_density(DensityFunction density_function,
                       DistributionFunction distribution_function);
inline void calculate_density(const Functions &functions) {
    calculate_density(functions.density_function,
                      functions.distribution_function);
}

void calculate_local_average_velocity(
    LocalAverageVelocity local_velocty_function,
    DistributionFunction distribution_function,
    DensityFunction density_function);
inline void calculate_local_average_velocity(const Functions &functions) {
    calculate_local_average_velocity(functions.local_average_velocity,
                                     functions.distribution_function,
                                     functions.density_function);
}

void calculate_equilibrium_distribution(
    DistributionFunction equilibrium_distribution,
    DensityFunction density_function,
    LocalAverageVelocity local_average_velocity_function);
inline void calculate_equilibrium_distribution(const Functions &functions) {
    calculate_equilibrium_distribution(functions.buffer_distribution_function,
                                       functions.density_function,
                                       functions.local_average_velocity);
}

void relax_distribution(DistributionFunction distribution_function,
                        DistributionFunction equilibrium_distribution_function,
                        const double omega);
inline void relax_distribution(const Functions &functions, const double omega) {
    relax_distribution(functions.distribution_function,
                       functions.buffer_distribution_function, omega);
}

/**
 * Assumes a single layer of ghost nodes exists on the edges of the lattice
 *
 * Will treat all the edges of the lattice as fixed walls except for the top
 * wall. A lid velocity modifier view needs to be provided which corresponds to
 * the correction term of the rigid wall according to each direction index
 *
 * To treat the top wall as a fixed one instead of moving, simply initialize the
 * `lid_velocity_modifiers` as all 0
 */
void streaming_step_with_bounce_back_and_lid(
    DistributionFunction buffer_distribution_view,
    DistributionFunction distribution_function,
    DensityFunction density_function, const double lid_vel_x = 0,
    const double &lid_vel_y = 0);
inline void
streaming_step_with_bounce_back_and_lid(const Functions &functions,
                                        const double lid_vel_x = 0,
                                        const double lid_vel_y = 0) {
    streaming_step_with_bounce_back_and_lid(
        functions.buffer_distribution_function, functions.distribution_function,
        functions.density_function, lid_vel_x, lid_vel_y);
}

} // namespace LatticeBoltzmann