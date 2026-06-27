#pragma once

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>

void calculate_density(const Kokkos::View<double **> &density_function,
                       const Kokkos::View<double ***> distribution_function,
                       int grid_width, int grid_height);