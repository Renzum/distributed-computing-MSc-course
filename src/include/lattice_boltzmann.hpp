#pragma once

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

    /*
     * Allocates the same size Views and copies the data to them
     */
    Functions(const Kokkos::View<double ***> &distribution_function,
              const Kokkos::View<double **> &density_function,
              const Kokkos::View<double ***> &local_average_velocity);
};

void streaming_step(Kokkos::View<double ***> &buffer_distribution_view,
                    Kokkos::View<double ***> &distribution_function);
inline void streaming_step(Functions &functions) {
    streaming_step(functions.buffer_distribution_function,
                   functions.distribution_function);
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
} // namespace LatticeBoltzmann