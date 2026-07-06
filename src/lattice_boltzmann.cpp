#include <cmath>
#include <functional>
#include <tuple>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>

#include "lattice_boltzmann.hpp"

namespace LatticeBoltzmann {

namespace {

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

template <typename T>
std::tuple<int, int> get_grid_extents(const Kokkos::View<T> &grid) {
    return std::tuple(grid.extent_int(0), grid.extent_int(1));
}

} // namespace

Functions::Functions(int grid_width, int grid_height) {
    distribution_function = Kokkos::View<double ***>(
        "Distribution Function", grid_width, grid_height, TOTAL_DIRECTIONS);

    buffer_distribution_function =
        Kokkos::View<double ***>("Buffer Distribution Function", grid_width,
                                 grid_height, TOTAL_DIRECTIONS);

    density_function =
        Kokkos::View<double **>("Density Function", grid_width, grid_height);

    local_average_velocity = Kokkos::View<double ***>(
        "Local Average Velocity", grid_width, grid_height, 2);
};

void streaming_step_with_periodic_bounds(
    Kokkos::View<double ***> &buffer_distribution_view,
    Kokkos::View<double ***> &distribution_function, const int &ghost_layers) {
    const auto [grid_width, grid_height] =
        get_grid_extents(distribution_function);

    // By default, ghost layers is 0, so the streaming step will perform a
    // periodic boundary operation and wrap the values across the lattice
    // If we provide a positive ghost_layers integer, it will loop through the
    // inner lattice skipping the ghost layers
    Kokkos::parallel_for(
        "Streaming Step",
        Kokkos::MDRangePolicy({0 + ghost_layers, 0 + ghost_layers, 0},
                              {grid_width - ghost_layers,
                               grid_height - ghost_layers, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &current_x, const int &current_y,
                      const int &dir) {
            // If ghost_layers > 0, new position will never wrap around since
            // any possible x will be greater than 0 and less than the lattice
            // width. Same for y.
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

void calculate_density(const Kokkos::View<double **> &density_function,
                       const Kokkos::View<double ***> &distribution_function) {
    const auto [grid_width, grid_height] = get_grid_extents(density_function);

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
    const Kokkos::View<double **> &density_function) {

    const auto [grid_width, grid_height] =
        get_grid_extents(local_velocty_function);

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

void calculate_equilibrium_distribution(
    const Kokkos::View<double ***> &equilibrium_distribution,
    const Kokkos::View<double **> &density_function,
    const Kokkos::View<double ***> &local_average_velocity_function) {
    const auto [grid_width, grid_height] =
        get_grid_extents(equilibrium_distribution);

    // Calculate constants at compile time
    constexpr double C1 = 3.0;
    constexpr double C2 = 1.0;
    constexpr double C3 = 9.0 / 2.0;
    constexpr double C4 = -3.0 / 2.0;

    Kokkos::parallel_for(
        "Equilibrium Distribution Calculation",
        Kokkos::MDRangePolicy({0, 0, 0},
                              {grid_width, grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            // This is where the fun begins

            // Precompute the w_i * rho coefficient
            const double coefficient =
                velocity_fraction[dir] * density_function(x, y);

            const auto [velocity_vec_x, velocity_vec_y] = velocity_vector[dir];

            const double avg_velocity_x =
                local_average_velocity_function(x, y, 0);
            const double avg_velocity_y =
                local_average_velocity_function(x, y, 1);

            // We use FMA to reduce floating point rounding errors as much as
            // possible

            // c_ix * ux + c_iy * uy
            const double dot_product =
                std::fma(velocity_vec_x, avg_velocity_x,
                         velocity_vec_y * avg_velocity_y);

            // ux * ux + uy * uy
            const double avg_velocity_vec_len_sqr =
                std::fma(avg_velocity_x, avg_velocity_x,
                         avg_velocity_y * avg_velocity_y);

            // A1 = 3.0 * (c_i * u)  + 1.0
            const double A1 = std::fma(C1, dot_product, C2);

            // A2 = (9.0 / 2.0) * (c_i * u)(c_i * u) + A1
            const double A2 = std::fma(C3, dot_product * dot_product, A1);

            // A3 = (-3.0 / 2.0) * (|u| * |u|) + A2
            const double A3 = std::fma(C4, avg_velocity_vec_len_sqr, A2);

            // Result = w_i * rho * A3
            equilibrium_distribution(x, y, dir) = coefficient * A3;
        });
}

void relax_distribution(
    const Kokkos::View<double ***> &distribution_function,
    const Kokkos::View<double ***> &equilibrium_distribution_function,
    const double omega) {
    const auto [grid_width, grid_height] =
        get_grid_extents(distribution_function);

    Kokkos::parallel_for(
        "Relaxation",
        Kokkos::MDRangePolicy({0, 0, 0},
                              {grid_width, grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            const double distribution_value = distribution_function(x, y, dir);
            const double eq_distribution_value =
                equilibrium_distribution_function(x, y, dir);

            distribution_function(x, y, dir) =
                std::fma(omega, eq_distribution_value - distribution_value,
                         distribution_value);
        });
}

double calc() {
    return 1;
}
double zero() {
    return 0;
}

Kokkos::View<double *>
calculate_wall_velocity_modifier(const double &wall_vel_x,
                                 const double &wall_vel_y,
                                 const double &average_density) {
    Kokkos::View<double *> wall_velocity_modifier("Wall Velocity Modifiers",
                                                  TOTAL_DIRECTIONS);

    if (wall_vel_x == 0 && wall_vel_y == 0) {
        Kokkos::parallel_for(
            "Wall Velocity Modifier Zero Init",
            Kokkos::RangePolicy(0, TOTAL_DIRECTIONS),
            KOKKOS_LAMBDA(const int &dir) { wall_velocity_modifier(dir) = 0; });
        return wall_velocity_modifier;
    }

    constexpr double c1 = 2.0 / (1.0 / 3.0);
    const double c2 = c2 * average_density;

    Kokkos::parallel_for(
        "Wall Velocity Modifier Calculation",
        Kokkos::RangePolicy(0, TOTAL_DIRECTIONS),
        KOKKOS_LAMBDA(const int &dir) {
            const auto [vec_x, vec_y] = velocity_vector[dir];

            const double dot_product = vec_x * wall_vel_x + vec_y * wall_vel_y;
            wall_velocity_modifier(dir) =
                c2 * dot_product * velocity_fraction[dir];
        });

    return wall_velocity_modifier;
}

void streaming_step_with_bounce_back_and_lid(
    Kokkos::View<double ***> &buffer_distribution_view,
    Kokkos::View<double ***> &distribution_function,
    const Kokkos::View<double *> &lid_velocity_modifiers) {

    static double avg_density = 0;

    const auto [lattice_width, lattice_height] =
        get_grid_extents(distribution_function);

    streaming_step_with_periodic_bounds(buffer_distribution_view,
                                        distribution_function, 1);

    // Since we moved all the values to the ghost nodes already, we make the
    // loop range only be the inner lattice points (i.e. excluding the outer
    // ghost layers)
    Kokkos::parallel_for(
        "Bounce Back Vertical", Kokkos::RangePolicy(1, lattice_height - 1),
        KOKKOS_LAMBDA(const int &y) {
            // Left Wall
            distribution_function(1, y, Direction::Right) =
                distribution_function(0, y, Direction::Left);
            // Diagonals
            distribution_function(1, y, Direction::UpRight) =
                distribution_function(0, y - 1, Direction::DownLeft);
            distribution_function(1, y, Direction::DownRight) =
                distribution_function(0, y + 1, Direction::UpLeft);

            // Right Wall
            distribution_function(lattice_width - 2, y, Direction::Left) =
                distribution_function(lattice_width - 1, y, Direction::Right);
            // Diagonals
            distribution_function(lattice_width - 2, y, Direction::UpLeft) =
                distribution_function(lattice_width - 1, y - 1,
                                      Direction::DownRight);
            distribution_function(lattice_width - 2, y, Direction::DownLeft) =
                distribution_function(lattice_width - 1, y + 1,
                                      Direction::UpRight);
        });

    Kokkos::parallel_for(
        "Bounce Back Horizontal", Kokkos::RangePolicy(1, lattice_width - 1),
        KOKKOS_LAMBDA(const int &x) {
            // Bottom Wall
            distribution_function(x, 1, Direction::Up) =
                distribution_function(x, 0, Direction::Down);
            // Diagonals
            distribution_function(x, 1, Direction::UpRight) =
                distribution_function(x - 1, 0, Direction::DownLeft);
            distribution_function(x, 1, Direction::UpLeft) =
                distribution_function(x + 1, 0, Direction::DownRight);

            // Top Wall (Lid)
            distribution_function(x, lattice_height - 2, Direction::Down) =
                distribution_function(x, lattice_height - 1, Direction::Up) -
                lid_velocity_modifiers(Direction::Up);
            // Diagonals
            distribution_function(x, lattice_height - 2, Direction::DownLeft) =
                distribution_function(x + 1, lattice_height - 1,
                                      Direction::UpRight) -
                lid_velocity_modifiers(Direction::UpRight);
            distribution_function(x, lattice_height - 2, Direction::DownRight) =
                distribution_function(x - 1, lattice_height - 1,
                                      Direction::UpLeft) -
                lid_velocity_modifiers(Direction::UpLeft);
            ;
        });
}

} // namespace LatticeBoltzmann
