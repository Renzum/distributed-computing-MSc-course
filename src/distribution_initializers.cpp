#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <direction_definitions.hpp>
#include <lattice_boltzmann.hpp>

#include "distribution_initializers.hpp"

namespace LatticeBoltzmann {

namespace DistributionInitializers {

void uniform_density(const Kokkos::View<double ***> &distribution_function,
                    const double uniform_value = 1.0) {
    const int lattice_width = distribution_function.extent_int(0);
    const int lattice_height = distribution_function.extent_int(1);

    Kokkos::parallel_for(
        "Init Uniform",
        Kokkos::MDRangePolicy(
            {0, 0, 0}, {lattice_width, lattice_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            distribution_function(x, y, dir) = uniform_value;
        });
}

void uniform_density_with_higher_center(
    const Kokkos::View<double ***> &distribution_function,
    const double uniform_value = 1.0, const double higher_value = 1.0) {

    const int lattice_width = distribution_function.extent_int(0);
    const int lattice_height = distribution_function.extent_int(1);

    const double width_center_min = double(lattice_width) / 3.0;
    const double width_center_max = 2.0 * double(lattice_width) / 3.0;

    const double height_center_min = double(lattice_height) / 3.0;
    const double height_center_max = 2.0 * double(lattice_height) / 3.0;

    // Lambda for checking if the x y is in the center 1/3 of the lattice
    auto is_center = [&width_center_min, &width_center_max, &height_center_min,
                      &height_center_max](int x, int y) -> bool {
        return (width_center_min < x && x < width_center_max) &&
               (height_center_min < y && y < height_center_max);
    };

    Kokkos::parallel_for(
        "Init Uniform with Dense Center",
        Kokkos::MDRangePolicy(
            {0, 0, 0}, {lattice_width, lattice_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            if (is_center(x, y)) {
                distribution_function(x, y, dir) = higher_value;
            } else {
                distribution_function(x, y, dir) = uniform_value;
            }
        });
}

void random_density(const Kokkos::View<double ***> &distribution_function) {
    const int lattice_width = distribution_function.extent_int(0);
    const int lattice_height = distribution_function.extent_int(1);

    auto rand = Kokkos::Random_XorShift64_Pool<>(/* seed = */ 12345);

    Kokkos::parallel_for(
        "Init Random Density",
        Kokkos::MDRangePolicy(
            {0, 0, 0}, {lattice_width, lattice_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            auto gen = rand.get_state();

            distribution_function(x, y, dir) = gen.rand(0, 100);

            rand.free_state(gen);
        });
}

} // namespace DistributionInitializers

} // namespace LatticeBoltzmann
