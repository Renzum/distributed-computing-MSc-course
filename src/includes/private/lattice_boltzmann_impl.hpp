#pragma once

#include <Kokkos_Core.hpp>

namespace LBMImpl {

void streaming_step(Kokkos::View<double ***> &buffer_distribution_view,
                    Kokkos::View<double ***> &distribution_function,
                    const int grid_width, const int grid_height);

void calculate_local_average_velocity(
    const Kokkos::View<double ***> &local_velocty_function,
    const Kokkos::View<double ***> &distribution_function,
    const Kokkos::View<double **> &density_function, const int grid_width,
    const int grid_height);

void calculate_equilibrium_distribution(
    const Kokkos::View<double ***> &equilibrium_distribution,
    const Kokkos::View<double **> &density_function,
    const Kokkos::View<double ***> &local_average_velocity_function,
    const int grid_width, const int grid_height);

void calculate_density(const Kokkos::View<double **> &density_function,
                       const Kokkos::View<double ***> &distribution_function,
                       const int grid_width, const int grid_height);

void relax_distribution(
    const Kokkos::View<double ***> &distribution_function,
    const Kokkos::View<double ***> &equilibrium_distribution_function,
    const double viscocity, const int grid_width, const int grid_height);

} // namespace LBMImpl