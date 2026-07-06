#include <iostream>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>
#include <lattice_boltzmann_types.hpp>

#include "lattice_boltzmann.hpp"

namespace LatticeBoltzmann {

void streaming_step_with_periodic_bounds(
    DistributionFunction &buffer_distribution_view,
    DistributionFunction &distribution_function, const Walls &walls) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    // We only want to compute the cells which aren't a ghost layer
    const int x_start = walls.left.ghost_layers;
    const int x_end = grid_width - walls.right.ghost_layers;

    const int y_start = walls.bottom.ghost_layers;
    const int y_end = grid_height - walls.top.ghost_layers;

    auto calculate_new_position =
        KOKKOS_LAMBDA(const int &old_x, const int &old_y, const int &direction,
                      int &new_x, int &new_y) {
        switch (direction) {
        case Direction::Left:
        case Direction::UpLeft:
        case Direction::DownLeft:
            // x wraps around to prevent segfault
            new_x = (old_x == x_start) ? (x_end - 1) : (old_x - 1);
            break;
        case Direction::Right:
        case Direction::UpRight:
        case Direction::DownRight:
            // x wraps around to prevent segfault
            new_x = (old_x == x_end - 1) ? x_start : (old_x + 1);
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
            new_y = (old_y == y_start) ? (y_end - 1) : (old_y - 1);
            break;
        case Direction::Up:
        case Direction::UpRight:
        case Direction::UpLeft:
            // y wraps around to prevent segfault
            new_y = (old_y == y_end - 1) ? y_start : (old_y + 1);
            break;
        default:
            new_y = old_y; // y doesn't change, no need to check
            break;
        }
    };

    Kokkos::parallel_for(
        "Streaming Step",
        Kokkos::MDRangePolicy({x_start, y_start, 0},
                              {x_end, y_end, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &current_x, const int &current_y,
                      const int &dir) {
            int new_x, new_y;
            calculate_new_position(current_x, current_y, dir, new_x, new_y);

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

    // We only want to compute the cells which aren't a ghost layer
    const int x_start = walls.left.ghost_layers;
    const int x_end = grid_width - walls.right.ghost_layers;

    const int y_start = walls.bottom.ghost_layers;
    const int y_end = grid_height - walls.top.ghost_layers;

    Kokkos::parallel_for(
        "Density Calculation",
        Kokkos::MDRangePolicy({x_start, y_start}, {x_end, y_end}),
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

    // We only want to compute the cells which aren't a ghost layer
    int x_start = walls.left.ghost_layers;
    int x_end = grid_width - walls.right.ghost_layers;

    int y_start = walls.bottom.ghost_layers;
    int y_end = grid_height - walls.top.ghost_layers;

    Kokkos::parallel_for(
        "Average Local Velocity Calculation",
        Kokkos::MDRangePolicy({x_start, y_start}, {x_end, y_end}),
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

    // We only want to compute the cells which aren't a ghost layer
    const int x_start = walls.left.ghost_layers;
    const int x_end = grid_width - walls.right.ghost_layers;

    const int y_start = walls.bottom.ghost_layers;
    const int y_end = grid_height - walls.top.ghost_layers;

    Kokkos::parallel_for(
        "Equilibrium Distribution Calculation",
        Kokkos::MDRangePolicy({x_start, y_start, 0},
                              {x_end, y_end, TOTAL_DIRECTIONS}),
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

    // We only want to compute the cells which aren't a ghost layer
    const int x_start = walls.left.ghost_layers;
    const int x_end = grid_width - walls.right.ghost_layers;

    const int y_start = walls.bottom.ghost_layers;
    const int y_end = grid_height - walls.top.ghost_layers;

    Kokkos::parallel_for(
        "Relaxation",
        Kokkos::MDRangePolicy({x_start, y_start, 0},
                              {x_end, y_end, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            const double distribution_value = distribution_function(x, y, dir);
            const double eq_distribution_value =
                equilibrium_distribution_function(x, y, dir);

            const double difference =
                eq_distribution_value - distribution_value;
            distribution_function(x, y, dir) =
                Kokkos::fma(omega, difference, distribution_value);
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

    /**
     *
     * Lambda which calculates the correction term for the moving wall bounce
     * back.
     *
     * If the wall is fixed, i.e. not moving, set both velocity values to `0.0`.
     *
     * @param direction which direction the bounce back must result in (i.e the
     * direction that a value must be after the bounce back)
     *
     * @param density the local density of the lattice cell at the x and y
     * @param wall_vel_x the velocity of the moving wall in the x axis (positive
     * means right)
     * @param wall_vel_y the velocity of the moving wall in the y axis (positive
     * means up)
     *
     */
    auto correction =
        KOKKOS_LAMBDA(const Direction &direction, const double &density,
                      const double &wall_vel_x, const double &wall_vel_y)
            ->double {

        // If the both x and y components of the wall velocity are 0, the
        // correction component is also 0
        if (wall_vel_x == 0 && wall_vel_y == 0) {
            return 0;
        }

        double weight;
        get_velocity_fraction(direction, weight);

        int vel_vec_x, vel_vec_y;
        get_velocity_vector(direction, vel_vec_x, vel_vec_y);

        // Compute the dot product between the wall velocity an the direction
        // velocity vectors
        const double dot_product =
            Kokkos::fma(wall_vel_x, vel_vec_x, wall_vel_y * vel_vec_y);

        // Since c^2_s is equal to 1/3, we can simplify it to
        // 2 / (1/3) = 2 * 3 = 6
        return 6.0 * weight * density * dot_product;
    };

    /**
     *
     * Lambda which performs a bounce back on a given cell given a resulting
     * direction.
     *
     * If the wall is fixed, i.e. not moving, set both velocity values to `0.0`.
     *
     * @param x the X position of the cell in the lattice
     * @param y the Y position of the cell in the lattice
     * @param direction the direction in which the bounce back must result to
     * @param local_density the local density of the cell in the lattice
     * @param wall_vel_x the velocity of the moving wall in the x axis (positive
     * means right)
     * @param wall_vel_y the velocity of the moving wall in the y axis (positive
     * means up).
     *
     */
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

        // We only want to compute the cells which aren't a ghost layer
        const int y_start = walls.bottom.ghost_layers;
        const int y_end = lattice_height - walls.top.ghost_layers;

        Kokkos::parallel_for(
            "Bounce Back Vertical Walls", Kokkos::RangePolicy(y_start, y_end),
            KOKKOS_LAMBDA(const int &y) {
                // Initialize variables so they can be reused
                int wall_x;
                double local_density;
                double wall_vel_x, wall_vel_y;

                // Left Wall
                // (Skip if type isn't set to bounce back)
                if (walls.left.wall_type == WallType::BounceBack) {

                    // The X index of the left-most layer is the
                    // the number of ghost layers on the left
                    wall_x = walls.left.ghost_layers;

                    local_density = density_function(wall_x, y);

                    wall_vel_x = walls.left.vel_x;
                    wall_vel_y = walls.left.vel_y;

                    // Perform bounce back on the resulting directions
                    // (i.e the expected directions AFTER the bounce back)
                    bounce_back(wall_x, y, Direction::UpRight, local_density,
                                wall_vel_x, wall_vel_y);
                    bounce_back(wall_x, y, Direction::Right, local_density,
                                wall_vel_x, wall_vel_y);
                    bounce_back(wall_x, y, Direction::DownRight, local_density,
                                wall_vel_x, wall_vel_y);
                }

                // Right Wall
                // (Skip if type isn't set to bounce back)
                if (walls.right.wall_type == WallType::BounceBack) {

                    // The X index of the right-most layer is the
                    // width of the lattice
                    // minus number of ghost layers on the right
                    // minus 1
                    wall_x = lattice_width - 1 - walls.right.ghost_layers;

                    local_density = density_function(wall_x, y);

                    wall_vel_x = walls.right.vel_x;
                    wall_vel_y = walls.right.vel_y;

                    // Perform bounce back on the resulting directions
                    // (i.e the expected directions AFTER the bounce back)
                    bounce_back(wall_x, y, Direction::UpLeft, local_density,
                                wall_vel_x, wall_vel_y);
                    bounce_back(wall_x, y, Direction::Left, local_density,
                                wall_vel_x, wall_vel_y);
                    bounce_back(wall_x, y, Direction::DownLeft, local_density,
                                wall_vel_x, wall_vel_y);
                }
            });
    }

    // If either left or right wall are of type Bounce Back, perform the bounce
    // back compilation on them
    if (walls.bottom.wall_type == WallType::BounceBack ||
        walls.top.wall_type == WallType::BounceBack) {

        // We only want to compute the cells which aren't a ghost layer
        const int x_start = walls.left.ghost_layers;
        const int x_end = lattice_width - walls.right.ghost_layers;

        Kokkos::parallel_for(
            "Bounce Back Horizontal Walls", Kokkos::RangePolicy(x_start, x_end),
            KOKKOS_LAMBDA(const int &x) {
                // Initialize variables so they can be reused
                int wall_y;
                double local_density;
                double wall_vel_x, wall_vel_y;

                // Bottom Wall
                // (Skip if type isn't set to bounce back)
                if (walls.bottom.wall_type == WallType::BounceBack) {

                    // The Y index of the bottom-most layer is the
                    // the number of ghost layers on the bottom
                    wall_y = walls.bottom.ghost_layers;

                    local_density = density_function(x, wall_y);

                    // Store the velocity of the bottom wall
                    wall_vel_x = walls.bottom.vel_x;
                    wall_vel_y = walls.bottom.vel_y;

                    // Perform bounce back on the resulting directions
                    // (i.e the expected directions AFTER the bounce back)
                    bounce_back(x, wall_y, Direction::UpRight, local_density,
                                wall_vel_x, wall_vel_y);
                    bounce_back(x, wall_y, Direction::Up, local_density,
                                wall_vel_x, wall_vel_y);
                    bounce_back(x, wall_y, Direction::UpLeft, local_density,
                                wall_vel_x, wall_vel_y);
                }

                // Top Wall (AKA Lid)
                // (Skip if type isn't set to bounce back)
                if (walls.top.wall_type == WallType::BounceBack) {

                    // The Y index of the top-most layer is the
                    // height of the lattice
                    // minus number of ghost layers on the top
                    // minus 1
                    wall_y = lattice_height - 1 - walls.top.ghost_layers;

                    local_density = density_function(x, wall_y);

                    // Store the velocity of the top wall
                    wall_vel_x = walls.top.vel_x;
                    wall_vel_y = walls.top.vel_y;

                    // Perform bounce back on the resulting directions
                    // (i.e the expected directions AFTER the bounce back)
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
