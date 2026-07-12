#pragma once

#include <direction_definitions.hpp>
#include <lattice_boltzmann_types.hpp>
#include <tuple>

#include <Kokkos_Core.hpp>

namespace LatticeBoltzmann {

void streaming_step_with_periodic_bounds(
    DistributionFunction &buffer_distribution_view,
    DistributionFunction &distribution_function);

void calculate_density(DensityFunction &density_function,
                       const DistributionFunction &distribution_function);

void calculate_local_average_velocity(
    VelocityProfile &local_velocity_function,
    const DistributionFunction &distribution_function,
    const DensityFunction &density_function);

void calculate_equilibrium_distribution(
    DistributionFunction &equilibrium_distribution,
    const DensityFunction &density_function,
    const VelocityProfile &local_average_velocity_function);

void relax_distribution(
    DistributionFunction &distribution_function,
    const DistributionFunction &equilibrium_distribution_function,
    const double omega);

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

} // namespace LatticeBoltzmann