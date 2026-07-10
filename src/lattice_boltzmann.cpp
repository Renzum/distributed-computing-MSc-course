#include <iostream>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>
#include <lattice_boltzmann_types.hpp>

#include "lattice_boltzmann.hpp"

namespace LatticeBoltzmann {

namespace {

KOKKOS_INLINE_FUNCTION
void calculate_new_position(const int &old_x, const int &old_y,
                            const int &direction, const int grid_width,
                            const int grid_height, int &new_x, int &new_y) {
    switch (direction) {
    case Direction::Left:
    case Direction::UpLeft:
    case Direction::DownLeft:
        // x wraps around to prevent segfault
        new_x = (old_x == 0) ? (grid_width - 1) : (old_x - 1);
        break;
    case Direction::Right:
    case Direction::UpRight:
    case Direction::DownRight:
        // x wraps around to prevent segfault
        new_x = (old_x == grid_width - 1) ? 0 : (old_x + 1);
        break;
    default:
        new_x = old_x; // x doesn't change, no need to check
        break;
    }

    switch (direction) {
    case Direction::Down:
    case Direction::DownRight:
    case Direction::DownLeft:
        // y wraps around to prevent segfault
        new_y = (old_y == 0) ? (grid_height - 1) : (old_y - 1);
        break;
    case Direction::Up:
    case Direction::UpRight:
    case Direction::UpLeft:
        // y wraps around to prevent segfault
        new_y = (old_y == grid_height - 1) ? 0 : (old_y + 1);
        break;
    default:
        new_y = old_y; // y doesn't change, no need to check
        break;
    }
}

} // namespace

void streaming_step_with_periodic_bounds(
    DistributionFunction &buffer_distribution_view,
    DistributionFunction &distribution_function, const Walls &walls) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    // By default, ghost layers is 0, so the streaming step will perform a
    // periodic boundary operation and wrap the values across the lattice
    // If we provide positive ghost_layers integers, it will loop through the
    // inner lattice skipping the ghost layers
    Kokkos::parallel_for(
        "Streaming Step",
        Kokkos::MDRangePolicy(
            {walls.left.ghost_layers, walls.bottom.ghost_layers, 0},
            {grid_width - walls.right.ghost_layers,
             grid_height - walls.top.ghost_layers, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &current_x, const int &current_y,
                      const int &dir) {
            int new_x, new_y;
            calculate_new_position(current_x, current_y, dir, grid_width,
                                   grid_height, new_x, new_y);

            // We move the values to the buffer according to their velocity
            // And then simply swap the buffer with the distribution function
            // view
            buffer_distribution_view(new_x, new_y, dir) =
                distribution_function(current_x, current_y, dir);
        });

    Kokkos::kokkos_swap(distribution_function, buffer_distribution_view);
}

void calculate_density(DensityFunction &density_function,
                       const DistributionFunction &distribution_function,
                       const Walls &walls) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    Kokkos::parallel_for(
        "Density Calculation",
        Kokkos::MDRangePolicy(
            {walls.left.ghost_layers, walls.bottom.ghost_layers},
            {grid_width - walls.right.ghost_layers,
             grid_height - walls.top.ghost_layers}),
        KOKKOS_LAMBDA(const int &x, const int &y) {
            double local_density = 0;
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                local_density += distribution_function(x, y, dir);
            }

            density_function(x, y) = local_density;
        });
}

void calculate_local_average_velocity(
    LocalAverageVelocity &local_velocty_function,
    const DistributionFunction &distribution_function,
    const DensityFunction &density_function, const Walls &walls) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    Kokkos::parallel_for(
        "Average Local Velocity Calculation",
        Kokkos::MDRangePolicy(
            {walls.left.ghost_layers, walls.bottom.ghost_layers},
            {grid_width - walls.right.ghost_layers,
             grid_height - walls.top.ghost_layers}),
        KOKKOS_LAMBDA(const int &x, const int &y) {
            int velocity_vector_x, velocity_vector_y;

            const double inverse_density = 1.0 / density_function(x, y);

            double vec_x = 0;
            double vec_y = 0;

            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {

                get_velocity_vector(static_cast<Direction>(dir),
                                    velocity_vector_x, velocity_vector_y);

                vec_x += velocity_vector_x * distribution_function(x, y, dir);
                vec_y += velocity_vector_y * distribution_function(x, y, dir);
            }

            vec_x *= inverse_density;
            vec_y *= inverse_density;

            local_velocty_function(x, y, 0) = vec_x;
            local_velocty_function(x, y, 1) = vec_y;
        });
}

void calculate_equilibrium_distribution(
    DistributionFunction &equilibrium_distribution,
    const DensityFunction &density_function,
    const LocalAverageVelocity &local_average_velocity_function,
    const Walls &walls) {
    const int grid_width = equilibrium_distribution.extent_int(0);
    const int grid_height = equilibrium_distribution.extent_int(1);

    // Calculate constants at compile time
    constexpr double C1 = 3.0;
    constexpr double C2 = 1.0;
    constexpr double C3 = 9.0 / 2.0;
    constexpr double C4 = -3.0 / 2.0;

    Kokkos::parallel_for(
        "Equilibrium Distribution Calculation",
        Kokkos::MDRangePolicy(
            {walls.left.ghost_layers, walls.bottom.ghost_layers, 0},
            {grid_width - walls.right.ghost_layers,
             grid_height - walls.top.ghost_layers, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            // This is where the fun begins

            double velocity_fraction;
            get_velocity_fraction(static_cast<Direction>(dir),
                                  velocity_fraction);

            // Precompute the w_i * rho coefficient
            const double coefficient =
                velocity_fraction * density_function(x, y);

            int velocity_vec_x, velocity_vec_y;
            get_velocity_vector(static_cast<Direction>(dir), velocity_vec_x,
                                velocity_vec_y);

            const double avg_velocity_x =
                local_average_velocity_function(x, y, 0);
            const double avg_velocity_y =
                local_average_velocity_function(x, y, 1);

            // We use FMA to reduce floating point rounding errors as much
            // as possible

            // c_ix * ux + c_iy * uy
            const double dot_product =
                Kokkos::fma(velocity_vec_x, avg_velocity_x,
                            velocity_vec_y * avg_velocity_y);

            // ux * ux + uy * uy
            const double avg_velocity_vec_len_sqr =
                Kokkos::fma(avg_velocity_x, avg_velocity_x,
                            avg_velocity_y * avg_velocity_y);

            // A1 = 3.0 * (c_i * u)  + 1.0
            const double A1 = Kokkos::fma(C1, dot_product, C2);

            // A2 = (9.0 / 2.0) * (c_i * u)(c_i * u) + A1
            const double A2 = Kokkos::fma(C3, dot_product * dot_product, A1);

            // A3 = (-3.0 / 2.0) * (|u| * |u|) + A2
            const double A3 = Kokkos::fma(C4, avg_velocity_vec_len_sqr, A2);

            // Result = w_i * rho * A3
            equilibrium_distribution(x, y, dir) = coefficient * A3;
        });
}

void relax_distribution(
    DistributionFunction &distribution_function,
    const DistributionFunction &equilibrium_distribution_function,
    const double omega, const Walls &walls) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    Kokkos::parallel_for(
        "Relaxation",
        Kokkos::MDRangePolicy(
            {walls.left.ghost_layers, walls.bottom.ghost_layers, 0},
            {grid_width - walls.right.ghost_layers,
             grid_height - walls.top.ghost_layers, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            const double distribution_value = distribution_function(x, y, dir);
            const double eq_distribution_value =
                equilibrium_distribution_function(x, y, dir);

            distribution_function(x, y, dir) =
                Kokkos::fma(omega, eq_distribution_value - distribution_value,
                            distribution_value);
        });
}

void streaming_step_with_bounce_back_and_lid(
    DistributionFunction &buffer_distribution_view,
    DistributionFunction &distribution_function,
    const DensityFunction &density_function, const Walls &walls) {

    streaming_step_with_periodic_bounds(buffer_distribution_view,
                                        distribution_function, walls);

    const int lattice_width = distribution_function.extent_int(0);
    const int lattice_height = distribution_function.extent_int(1);

    auto correction =
        KOKKOS_LAMBDA(const Direction &direction, const double &density,
                      const double &wall_vel_x, const double &wall_vel_y)
            ->double {
        if (wall_vel_x == 0 && wall_vel_y == 0) {
            return 0;
        }

        double weight;
        int vel_vec_x, vel_vec_y;

        get_velocity_fraction(direction, weight);
        get_velocity_vector(direction, vel_vec_x, vel_vec_y);

        const double dot_product =
            Kokkos::fma(wall_vel_x, vel_vec_x, wall_vel_y * vel_vec_y);

        return 6.0 * weight * density * dot_product;
    };

    auto bounce_back =
        KOKKOS_LAMBDA(const int &x, const int &y, const Direction &direction,
                      const double &local_density, const double &wall_vel_x,
                      const double &wall_vel_y) {
        const Direction opposite_direction = get_opposite_direction(direction);

        const double previous_value =
            buffer_distribution_view(x, y, opposite_direction);
        distribution_function(x, y, direction) =
            previous_value - correction(opposite_direction, local_density,
                                        wall_vel_x, wall_vel_y);
    };

    // If either left or right wall are of type Bounce Back, perform the bounce
    // back compilation on them
    if (walls.left.wall_type == WallType::BounceBack ||
        walls.right.wall_type == WallType::BounceBack) {

        Kokkos::parallel_for(
            "Bounce Back Vertical Walls",
            Kokkos::RangePolicy(walls.bottom.ghost_layers,
                                lattice_height - walls.top.ghost_layers),
            KOKKOS_LAMBDA(const int &y) {
                int wall_x;
                double local_density;
                double wall_vel_x, wall_vel_y;

                // Left Wall
                if (walls.left.wall_type == WallType::BounceBack) {
                    // We only need to compute bounce back on the non-ghost
                    // layers
                    wall_x = walls.left.ghost_layers;

                    local_density = density_function(wall_x, y);

                    wall_vel_x = walls.left.vel_x;
                    wall_vel_y = walls.left.vel_y;

                    bounce_back(walls.left.ghost_layers, y, Direction::UpRight,
                                local_density, walls.left.vel_x, wall_vel_y);
                    bounce_back(walls.left.ghost_layers, y, Direction::Right,
                                local_density, walls.left.vel_x, wall_vel_y);
                    bounce_back(walls.left.ghost_layers, y,
                                Direction::DownRight, local_density,
                                walls.left.vel_x, wall_vel_y);
                }

                // Right Wall
                if (walls.right.wall_type == WallType::BounceBack) {
                    // We only need to compute bounce back on the non-ghost
                    // layers
                    wall_x = lattice_width - walls.right.ghost_layers - 1;
                    local_density = density_function(wall_x, y);
                    wall_vel_x = walls.right.vel_x;
                    wall_vel_y = walls.right.vel_y;

                    bounce_back(walls.left.ghost_layers, y, Direction::UpLeft,
                                local_density, walls.left.vel_x, wall_vel_y);
                    bounce_back(walls.left.ghost_layers, y, Direction::Left,
                                local_density, walls.left.vel_x, wall_vel_y);
                    bounce_back(walls.left.ghost_layers, y, Direction::DownLeft,
                                local_density, walls.left.vel_x, wall_vel_y);
                }
            });
    }

    // If either left or right wall are of type Bounce Back, perform the bounce
    // back compilation on them
    if (walls.bottom.wall_type == WallType::BounceBack ||
        walls.top.wall_type == WallType::BounceBack) {
        Kokkos::parallel_for(
            "Bounce Back Horizontal Walls",
            Kokkos::RangePolicy(walls.left.ghost_layers,
                                lattice_width - walls.right.ghost_layers),
            KOKKOS_LAMBDA(const int &x) {
                int wall_y;
                double local_density;
                double wall_vel_x, wall_vel_y;

                // Bottom Wall
                if (walls.bottom.wall_type == WallType::BounceBack) {
                    // We only need to compute bounce back on the non-ghost
                    // layers
                    wall_y = walls.bottom.ghost_layers; // Bottom Wall Y

                    local_density = density_function(x, wall_y);
                    wall_vel_x = walls.bottom.vel_x;
                    wall_vel_y = walls.bottom.vel_y;

                    bounce_back(x, wall_y, Direction::UpRight, local_density,
                                wall_vel_x, wall_vel_y);
                    bounce_back(x, wall_y, Direction::Up, local_density,
                                wall_vel_x, wall_vel_y);
                    bounce_back(x, wall_y, Direction::UpLeft, local_density,
                                wall_vel_x, wall_vel_y);
                }

                // === Top Wall (AKA Lid) ===

                if (walls.top.wall_type == WallType::BounceBack) {

                    // We only need to compute bounce back on the non-ghost
                    // layers
                    wall_y = lattice_height - walls.top.ghost_layers - 1;

                    local_density = density_function(x, wall_y);

                    wall_vel_x = walls.bottom.vel_x;
                    wall_vel_y = walls.bottom.vel_y;
                    bounce_back(x, wall_y, Direction::DownRight, local_density,
                                wall_vel_x, wall_vel_y);
                    bounce_back(x, wall_y, Direction::Down, local_density,
                                wall_vel_x, wall_vel_y);
                    bounce_back(x, wall_y, Direction::DownLeft, local_density,
                                wall_vel_x, wall_vel_y);
                }
            });
    }
}

} // namespace LatticeBoltzmann
