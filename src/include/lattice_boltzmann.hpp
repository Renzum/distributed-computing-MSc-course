#pragma once

#include <direction_definitions.hpp>
#include <lattice_boltzmann_types.hpp>
#include <tuple>

#include <Kokkos_Core.hpp>

namespace LatticeBoltzmann {

void streaming_step_with_periodic_bounds(
    DistributionFunction &buffer_distribution_view,
    DistributionFunction &distribution_function, const Walls &walls);
inline void streaming_step_with_periodic_bounds(Functions &functions,
                                                const Walls &walls) {
    streaming_step_with_periodic_bounds(functions.buffer_distribution_function,
                                        functions.distribution_function, walls);
}

void calculate_density(DensityFunction &density_function,
                       const DistributionFunction &distribution_function,
                       const Walls &walls);
inline void calculate_density(Functions &functions, const Walls &walls) {
    calculate_density(functions.density_function,
                      functions.distribution_function, walls);
}

void calculate_local_average_velocity(
    LocalAverageVelocity &local_velocty_function,
    const DistributionFunction &distribution_function,
    const DensityFunction &density_function, const Walls &walls);
inline void calculate_local_average_velocity(Functions &functions,
                                             const Walls &walls) {
    calculate_local_average_velocity(functions.local_average_velocity,
                                     functions.distribution_function,
                                     functions.density_function, walls);
}

void calculate_equilibrium_distribution(
    DistributionFunction &equilibrium_distribution,
    const DensityFunction &density_function,
    const LocalAverageVelocity &local_average_velocity_function,
    const Walls &walls);
inline void calculate_equilibrium_distribution(Functions &functions,
                                               const Walls &walls) {
    calculate_equilibrium_distribution(functions.buffer_distribution_function,
                                       functions.density_function,
                                       functions.local_average_velocity, walls);
}

void relax_distribution(
    DistributionFunction &distribution_function,
    const DistributionFunction &equilibrium_distribution_function,
    const double omega, const Walls &walls);
inline void relax_distribution(Functions &functions, const double omega,
                               const Walls &walls) {
    relax_distribution(functions.distribution_function,
                       functions.buffer_distribution_function, omega, walls);
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
    DistributionFunction &buffer_distribution_view,
    DistributionFunction &distribution_function,
    const DensityFunction &density_function, const Walls &walls);
inline void streaming_step_with_bounce_back_and_lid(Functions &functions,
                                                    const Walls &walls) {
    streaming_step_with_bounce_back_and_lid(
        functions.buffer_distribution_function, functions.distribution_function,
        functions.density_function, walls);
}

} // namespace LatticeBoltzmann