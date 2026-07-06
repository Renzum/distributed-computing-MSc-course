#pragma once

#include <tuple>

#include <Kokkos_Core.hpp>

namespace LatticeBoltzmann {

/**
 * Struct containing all the necessariy Kokkos::Views to represent the
 * different functions of Lattice-Boltzmann
 */
struct Functions {
    /**
     * 3D Lattice of x and y size where each cell is an array of 9 elements
     * representing all the directions for particles
     */
    Kokkos::View<double ***> distribution_function;

    /**
     * 3D Lattice of x and y size where each cell is an array of 9 elements
     * representing all the directions for particles.
     *
     * Used as a buffer for storing temporary results.
     */
    Kokkos::View<double ***> buffer_distribution_function;

    /**
     * 2D Lattice of x and y size where each cell represents the density of
     * the distribution function
     */
    Kokkos::View<double **> density_function;

    /**
     * 3D Lattice of x and y size where each cell holds an array of size 2
     * that represents a 2D velocity vector
     */
    Kokkos::View<double ***> local_average_velocity;

    /**
     * Allocates the necessary views with the provided lattice width and
     * height
     */
    Functions(int grid_width, int grid_height);
};

void streaming_step_with_periodic_bounds(
    Kokkos::View<double ***> &buffer_distribution_view,
    Kokkos::View<double ***> &distribution_function,
    const int &ghost_layers = 0);
inline void streaming_step_with_periodic_bounds(Functions &functions,
                                                const int &ghost_layers = 0) {
    streaming_step_with_periodic_bounds(functions.buffer_distribution_function,
                                        functions.distribution_function,
                                        ghost_layers);
}

void calculate_density(const Kokkos::View<double **> &density_function,
                       const Kokkos::View<double ***> &distribution_function);
inline void calculate_density(const Functions &functions) {
    calculate_density(functions.density_function,
                      functions.distribution_function);
}

void calculate_local_average_velocity(
    const Kokkos::View<double ***> &local_velocty_function,
    const Kokkos::View<double ***> &distribution_function,
    const Kokkos::View<double **> &density_function);
inline void calculate_local_average_velocity(const Functions &functions) {
    calculate_local_average_velocity(functions.local_average_velocity,
                                     functions.distribution_function,
                                     functions.density_function);
}

void calculate_equilibrium_distribution(
    const Kokkos::View<double ***> &equilibrium_distribution,
    const Kokkos::View<double **> &density_function,
    const Kokkos::View<double ***> &local_average_velocity_function);
inline void calculate_equilibrium_distribution(const Functions &functions) {
    calculate_equilibrium_distribution(functions.buffer_distribution_function,
                                       functions.density_function,
                                       functions.local_average_velocity);
}

void relax_distribution(
    const Kokkos::View<double ***> &distribution_function,
    const Kokkos::View<double ***> &equilibrium_distribution_function,
    const double omega);
inline void relax_distribution(const Functions &functions, const double omega) {
    relax_distribution(functions.distribution_function,
                       functions.buffer_distribution_function, omega);
}

/**
 * Calculate a view of the corretion term for the moving lid
 *
 * If the lid is a fixed wall, set `wall_vel_x` and `wall_vel_y` to 0 in order
 * to get a View initialized with 0 in each cell.
 */
Kokkos::View<double *>
calculate_wall_velocity_modifier(const double &wall_vel_x,
                                 const double &wall_vel_y,
                                 const double &average_density);

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
    Kokkos::View<double ***> &buffer_distribution_view,
    Kokkos::View<double ***> &distribution_function,
    const Kokkos::View<double *> &lid_velocity_modifiers);
inline void streaming_step_with_bounce_back_and_lid(
    Functions &functions,
    const Kokkos::View<double *> &lid_velocity_modifiers) {
    streaming_step_with_bounce_back_and_lid(
        functions.buffer_distribution_function, functions.distribution_function,
        lid_velocity_modifiers);
}

} // namespace LatticeBoltzmann