#pragma once

#include <Kokkos_Core.hpp>

void calculate_equilibrium_distribution(
    const Kokkos::View<double ***> &equilibrium_distribution,
    const Kokkos::View<double **> &density_function,
    const Kokkos::View<double ***> &local_average_velocity_function,
    int grid_width, int grid_height);