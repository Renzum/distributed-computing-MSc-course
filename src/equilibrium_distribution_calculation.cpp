#include <cmath>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>

#include "equilibrium_distribution_calculation.hpp"

void calculate_equilibrium_distribution(
    const Kokkos::View<double ***> &equilibrium_distribution,
    const Kokkos::View<double **> &density_function,
    const Kokkos::View<double ***> &local_average_velocity_function,
    int grid_width, int grid_height) {

    constexpr double velocity_fraction[] = {
        4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    };

    Kokkos::parallel_for(
        "Equilibrium Distribution Calculation",
        Kokkos::MDRangePolicy({0, 0, 0},
                              {grid_width, grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            const double coefficient =
                velocity_fraction[dir] * density_function(x, y);

            const auto [velocity_vec_x, velocity_vec_y] = velocity_vector[dir];

            const double avg_velocity_x =
                local_average_velocity_function(x, y, 0);
            const double avg_velocity_y =
                local_average_velocity_function(x, y, 1);

            const double dot_product = velocity_vec_x * avg_velocity_x +
                                       velocity_vec_y * avg_velocity_y;

            const double avg_velocity_vec_len_sqr =
                std::pow(avg_velocity_x, 2) + std::pow(avg_velocity_y, 2);

            equilibrium_distribution(x, y, dir) =
                coefficient *
                (1 + 3 * dot_product + (9.0 / 2.0) * std::pow(dot_product, 2) -
                 (3.0 / 2.0) * avg_velocity_vec_len_sqr);
        });
}