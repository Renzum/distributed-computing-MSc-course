#include <tuple>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>

#include "lattice_boltzman_impl.hpp"

namespace LBMImpl {

namespace {

constexpr std::tuple<int, int> velocity_vector[] = {{0, 0},  {1, 0},   {0, 1},
                                                    {-1, 0}, {0, -1},  {1, 1},
                                                    {-1, 1}, {-1, -1}, {1, -1}};

constexpr double velocity_fraction[] = {
    4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
};

std::tuple<int, int> inline calculate_new_position(const int &x, const int &y,
                                                   const int &direction,
                                                   int grid_width,
                                                   int grid_height) {
    int new_x, new_y;

    switch (direction) {
    case Direction::Left:
    case Direction::UpLeft:
    case Direction::DownLeft:
        // x wraps around to prevent segfault
        new_x = (x == 0) ? (grid_width - 1) : (x - 1);
        break;
    case Direction::Right:
    case Direction::UpRight:
    case Direction::DownRight:
        // x wraps around to prevent segfault
        new_x = (x == grid_width - 1) ? 0 : (x + 1);
        break;
    default:
        new_x = x; // x doesn't change, no need to check
        break;
    }

    switch (direction) {
    case Direction::Down:
    case Direction::DownRight:
    case Direction::DownLeft:
        // y wraps around to prevent segfaul
        new_y = (y == 0) ? (grid_height - 1) : (y - 1);
        break;
    case Direction::Up:
    case Direction::UpRight:
    case Direction::UpLeft:
        // y wraps around to prevent segfaul
        new_y = (y == grid_height - 1) ? 0 : (y + 1);
        break;
    default:
        new_y = y; // y doesn't change, no need to check
        break;
    }

    return std::tuple<int, int>(new_x, new_y);
}

} // namespace

void streaming_step(Kokkos::View<double ***> &buffer_distribution_view,
                    Kokkos::View<double ***> &distribution_function,
                    const int grid_width, const int grid_height) {
    Kokkos::parallel_for(
        "Streaming Step",
        Kokkos::MDRangePolicy({0, 0, 0},
                              {grid_width, grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &current_x, const int &current_y,
                      const int &dir) {
            auto [new_x, new_y] = calculate_new_position(
                current_x, current_y, dir, grid_width, grid_height);

            // We move the values to the buffer according to their velocity
            // And then simply swap the buffer with the distribution function
            // view
            buffer_distribution_view(new_x, new_y, dir) =
                distribution_function(current_x, current_y, dir);
        });

    Kokkos::kokkos_swap(distribution_function, buffer_distribution_view);
}

void calculate_equilibrium_distribution(
    const Kokkos::View<double ***> &equilibrium_distribution,
    const Kokkos::View<double **> &density_function,
    const Kokkos::View<double ***> &local_average_velocity_function,
    const int grid_width, const int grid_height) {

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

void calculate_density(const Kokkos::View<double **> &density_function,
                       const Kokkos::View<double ***> &distribution_function,
                       const int grid_width, const int grid_height) {
    Kokkos::parallel_for(
        "Density Calculation",
        Kokkos::MDRangePolicy({0, 0}, {grid_width, grid_height}),
        KOKKOS_LAMBDA(const int &x, const int &y) {
            double local_density = 0;
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                local_density += distribution_function(x, y, dir);
            }

            density_function(x, y) = local_density;
        });
}

void calculate_local_average_velocity(
    const Kokkos::View<double ***> &local_velocty_function,
    const Kokkos::View<double ***> &distribution_function,
    const Kokkos::View<double **> &density_function, const int grid_width,
    const int grid_height) {

    Kokkos::parallel_for(
        "Average Local Velocity Calculation",
        Kokkos::MDRangePolicy({0, 0}, {grid_width, grid_height}),
        KOKKOS_LAMBDA(const int &x, const int &y) {
            const double inverse_density = 1.0 / density_function(x, y);

            double vec_x = 0;
            double vec_y = 0;

            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                auto [temp_x, temp_y] = velocity_vector[dir];
                vec_x += temp_x * distribution_function(x, y, dir);
                vec_y += temp_y * distribution_function(x, y, dir);
            }

            vec_x *= inverse_density;
            vec_y *= inverse_density;

            local_velocty_function(x, y, 0) = vec_x;
            local_velocty_function(x, y, 1) = vec_y;
        });
}

void relax_distribution(
    const Kokkos::View<double ***> &distribution_function,
    const Kokkos::View<double ***> &equilibrium_distribution_function,
    const double viscocity, const int grid_width, const int grid_height) {

    Kokkos::parallel_for(
        "Relaxation",
        Kokkos::MDRangePolicy({0, 0, 0},
                              {grid_width, grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            const double distribution_value = distribution_function(x, y, dir);
            const double eq_distribution_value =
                equilibrium_distribution_function(x, y, dir);

            distribution_function(x, y, dir) =
                distribution_value +
                viscocity * (eq_distribution_value - distribution_value);
        });
}

} // namespace LBMImpl
