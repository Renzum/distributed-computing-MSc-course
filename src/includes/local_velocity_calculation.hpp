#pragma once

#include <Kokkos_Core.hpp>

void calculate_local_average_velocity(
    const Kokkos::View<double ***> &local_velocty_function,
    const Kokkos::View<double ***> &distribution_function,
    const Kokkos::View<double **> &density_function, int grid_width,
    int grid_height);