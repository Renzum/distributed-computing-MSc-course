#pragma once

#include <Kokkos_Core.hpp>

void initRandomDensity(
    const Kokkos::View<double ***> destination_density_function);

void initUniformDensity(Kokkos::View<double ***> destination_density_function);