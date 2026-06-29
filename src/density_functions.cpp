#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <direction_definitions.hpp>

#include "density_functions.hpp"

void initRandomDensity(Kokkos::View<double ***> destination_density_function) {
    const auto grid_width = destination_density_function.extent_int(0);
    const auto grid_height = destination_density_function.extent_int(1);

    auto rand = Kokkos::Random_XorShift64_Pool<>(/* seed = */ 12345);

    Kokkos::parallel_for(
        "Initialize Random Density",
        Kokkos::MDRangePolicy({0, 0, 0},
                              {grid_width, grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            auto gen = rand.get_state();

            destination_density_function(x, y, dir) = gen.rand(0, 100);

            rand.free_state(gen);
        });
}

void initUniformDensity(Kokkos::View<double ***> destination_density_function) {
    const auto grid_width = destination_density_function.extent_int(0);
    const auto grid_height = destination_density_function.extent_int(1);

    Kokkos::parallel_for(
        "Initialize Uniform Density",
        Kokkos::MDRangePolicy({0, 0, 0},
                              {grid_width, grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            destination_density_function(x, y, dir) =
                velocity_fraction[dir] * 0.5;
        });
}