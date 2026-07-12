#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

#include <milestone05_helpers.hpp>

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK) {
    // A 20x20 lattice with fixed bounce-back boundary walls
    const int lattice_width = 20, lattice_height = 20;
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall(0, 0),
        LatticeBoltzmann::Wall(0, 0),
        LatticeBoltzmann::Wall(0, 0),
        LatticeBoltzmann::Wall(0, 0),
    };

    // Initialize all the necessary Views
    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", lattice_width, lattice_height);
    LatticeBoltzmann::DistributionFunction buffer_distribution_function(
        "Buffer Distribution Function", lattice_width, lattice_height);
    LatticeBoltzmann::DensityFunction density_function(
        "Density Function", lattice_width, lattice_height);
    LatticeBoltzmann::VelocityProfile velocity_profile(
        "Velocity Profile", lattice_width, lattice_height);

    // Initialize the  distribution with random values
    LatticeBoltzmann::DistributionInitializers::random_density(
        distribution_function);

    // Omega doesn't matter much here, but lower is better so the system
    // reaches an equilibrium slower
    constexpr double omega = 0.5;

    // For accessing the current and previous distributions in Host Space
    auto current_distribution =
        Kokkos::create_mirror_view(distribution_function);
    // We always create a view here to store the previous distribution
    auto previous_distribution = Kokkos::create_mirror(distribution_function);

    // Perform the Lattice Boltzmann step with boundary check 20 times and
    // test bounce back correctness each time
    for (int i = 0; i < 1; i++) {
        // Perform Lattice-Boltzmann normally until streaming and bounce-back
        LatticeBoltzmann::calculate_density(density_function,
                                            distribution_function);
        LatticeBoltzmann::calculate_local_average_velocity(
            velocity_profile, distribution_function, density_function);
        LatticeBoltzmann::calculate_equilibrium_distribution(
            buffer_distribution_function, density_function, velocity_profile);
        LatticeBoltzmann::relax_distribution(
            distribution_function, buffer_distribution_function, omega);

        // Copy the current (relaxed) distribution to host space
        Kokkos::deep_copy(previous_distribution, distribution_function);

        // Perform streaming and bounce-back
        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(
            buffer_distribution_function, distribution_function,
            density_function, walls);

        // Copy the current (post-streaming and bounce-back) distribution to
        // host space
        Kokkos::deep_copy(current_distribution, distribution_function);

        // Iterate through the height of the lattice and check the bounce-back
        // correctness on Horizontal Walls
        for (int y = 0; y < lattice_height; y++) {

            // Left Wall
            check_bounce_back_up_left(current_distribution,
                                      previous_distribution, 0, y);

            check_bounce_back_left(current_distribution, previous_distribution,
                                   0, y);

            check_bounce_back_down_left(current_distribution,
                                        previous_distribution, 0, y);

            // Right Wall
            check_bounce_back_up_right(current_distribution,
                                       previous_distribution, lattice_width - 1,
                                       y);

            check_bounce_back_right(current_distribution, previous_distribution,
                                    lattice_width - 1, y);

            check_bounce_back_down_right(current_distribution,
                                         previous_distribution,
                                         lattice_width - 1, y);

            // Ensure Center Values Don't Move
            ASSERT_EQ(current_distribution(0, y, Direction::Center),
                      previous_distribution(0, y, Direction::Center));

            ASSERT_EQ(
                current_distribution(lattice_width - 1, y, Direction::Center),
                previous_distribution(lattice_width - 1, y, Direction::Center));
        };

        // Iterate through the width of the lattice and check bounce-back
        // correctness on the Vertical Walls
        for (int x = 0; x < lattice_width; x++) {
            // Bottom Wall

            check_bounce_back_down_left(current_distribution,
                                        previous_distribution, x, 0);
            check_bounce_back_down(current_distribution, previous_distribution,
                                   x, 0);
            check_bounce_back_down_right(current_distribution,
                                         previous_distribution, x, 0);

            // Top Wall
            check_bounce_back_up_left(current_distribution,
                                      previous_distribution, x,
                                      lattice_height - 1);
            check_bounce_back_up(current_distribution, previous_distribution, x,
                                 lattice_height - 1);
            check_bounce_back_up_right(current_distribution,
                                       previous_distribution, x,
                                       lattice_height - 1);

            // // Ensure Center Values Don't Move
            ASSERT_EQ(current_distribution(x, 0, Direction::Center),
                      previous_distribution(x, 0, Direction::Center));
            ASSERT_EQ(
                current_distribution(x, lattice_height - 1, Direction::Center),
                previous_distribution(x, lattice_height - 1,
                                      Direction::Center));
        };
    }
}

// Generalized Testing Function for bounce-back checking when walls are moving
void moving_wall_test(const int lattice_width, const int lattice_height,
                      const LatticeBoltzmann::Walls &walls) {
    // Omega doesn't matter much here, but lower is better so the system
    // reaches an equilibrium slower
    constexpr double omega = 0.5;

    // Initialize all the necessary Views
    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", lattice_width, lattice_height);

    LatticeBoltzmann::DistributionFunction buffer_distribution(
        "Buffer Distribution", lattice_width, lattice_height);

    LatticeBoltzmann::DensityFunction density_function(
        "Density Function", lattice_width, lattice_height);

    LatticeBoltzmann::VelocityProfile velocity_profile(
        "Velocity Profile", lattice_width, lattice_height);

    // Initialize the Distribution as random
    LatticeBoltzmann::DistributionInitializers::random_density(
        distribution_function);

    // Create views for accessing on host space
    auto current_distribution =
        Kokkos::create_mirror_view(distribution_function);

    auto current_density = Kokkos::create_mirror_view(density_function);

    // Create View always to preserve the previous distribution
    auto previous_distribution = Kokkos::create_mirror(distribution_function);

    // Perform the Lattice Boltzmann step with boundary check 20 times and test
    // bounce back correctness each time
    for (int i = 0; i < 10; i++) {
        // Perform Lattice Boltzmann normally until streaming and bounce-back
        LatticeBoltzmann::calculate_density(density_function,
                                            distribution_function);
        LatticeBoltzmann::calculate_local_average_velocity(
            velocity_profile, distribution_function, density_function);
        LatticeBoltzmann::calculate_equilibrium_distribution(
            buffer_distribution, density_function, velocity_profile);
        LatticeBoltzmann::relax_distribution(distribution_function,
                                             buffer_distribution, omega);

        // Save the current (relaxed) distribution
        Kokkos::deep_copy(previous_distribution, distribution_function);

        // Perform streaming and bounce-back
        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(
            buffer_distribution, distribution_function, density_function,
            walls);

        // Copy views from device to host
        Kokkos::deep_copy(current_distribution, distribution_function);
        Kokkos::deep_copy(current_density, density_function);

        double density, correction;

        double wall_vel_x, wall_vel_y;

        // Go through the whole width of the lattice, and check that the cells
        // at the top wall and bottom wall are bouncing back with the proper
        // correction
        const int top_layer_y = lattice_height - 1; // y index of the top layer
        for (int x = 0; x < lattice_width; x++) {

            // If the bottom wall is moving, check the correct bounce-back
            if (walls.bottom.vel_x > 0 || walls.bottom.vel_y > 0) {
                density = current_density(x, 0);

                wall_vel_x = walls.bottom.vel_x;
                wall_vel_y = walls.bottom.vel_y;

                correction =
                    correction_down_left(x, 0, density, wall_vel_x, wall_vel_y);

                check_bounce_back_down_left(current_distribution,
                                            previous_distribution, x, 0,
                                            correction);

                correction =
                    correction_down(x, 0, density, wall_vel_x, wall_vel_y);

                check_bounce_back_down(current_distribution,
                                       previous_distribution, x, 0, correction);

                correction = correction_down_right(x, 0, density, wall_vel_x,
                                                   wall_vel_y);

                check_bounce_back_down_right(current_distribution,
                                             previous_distribution, x, 0,
                                             correction);
            }

            // If the top wall is moving, check the correct bounce-back
            if (walls.bottom.vel_x > 0 || walls.bottom.vel_y > 0) {
                if (walls.top.vel_x > 0 || walls.top.vel_y > 0) {
                    density = current_density(x, top_layer_y);

                    wall_vel_x = walls.top.vel_x;
                    wall_vel_y = walls.top.vel_y;

                    correction = correction_up_right(x, top_layer_y, density,
                                                     wall_vel_x, wall_vel_y);

                    check_bounce_back_up_right(current_distribution,
                                               previous_distribution, x,
                                               top_layer_y, correction);

                    correction = correction_up(x, top_layer_y, density,
                                               wall_vel_x, wall_vel_y);

                    check_bounce_back_up(current_distribution,
                                         previous_distribution, x, top_layer_y,
                                         correction);

                    correction = correction_up_left(x, top_layer_y, density,
                                                    wall_vel_x, wall_vel_y);

                    check_bounce_back_up_left(current_distribution,
                                              previous_distribution, x,
                                              top_layer_y, correction);
                }
            }

            // Go through the whole width of the lattice, and check that the
            // cells at the top wall and bottom wall are bouncing back with the
            // proper correction
            const int right_layer_x =
                lattice_width - 1; // x index of the right layer
            for (int y = 0; y < lattice_width; y++) {
                // If the left wall is moving, check the correct bounce-back
                if (walls.left.vel_x > 0 || walls.left.vel_y > 0) {
                    density = current_density(0, y);

                    wall_vel_x = walls.left.vel_x;
                    wall_vel_y = walls.left.vel_y;

                    // The horizontal walls (top and bottom) take priority and
                    // will overwrite the bounce-back of the vertical walls
                    // (left and right). So the corners are a special case.

                    // If y = 0 (i.e we're at the bottom corner), use the bottom
                    // wall velocity instead Even if the bottom wall isn't
                    // moving, it's bounce-back will over-write the horizontal
                    // walls
                    if (y == 0) {
                        correction = correction_down_left(0, y, density,
                                                          walls.bottom.vel_x,
                                                          walls.bottom.vel_y);
                    } else {
                        correction = correction_down_left(
                            0, y, density, wall_vel_x, wall_vel_y);
                    }

                    check_bounce_back_down_left(current_distribution,
                                                previous_distribution, 0, y,
                                                correction);

                    correction =
                        correction_left(0, y, density, wall_vel_x, wall_vel_y);

                    check_bounce_back_left(current_distribution,
                                           previous_distribution, 0, y,
                                           correction);

                    correction = correction_up_left(0, y, density, wall_vel_x,
                                                    wall_vel_y);

                    // If y = lattice_height - 1 (i.e we're at the top corner),
                    // use the top wall velocity instead Even if the bottom wall
                    // isn't moving, it's bounce-back will over-write the
                    // horizontal walls
                    if (y == lattice_height - 1) {
                        correction = correction_up_left(
                            0, y, density, walls.top.vel_x, walls.top.vel_y);
                    } else {
                        correction = correction_up_left(0, y, density,
                                                        wall_vel_x, wall_vel_y);
                    }
                    check_bounce_back_up_left(current_distribution,
                                              previous_distribution, 0, y,
                                              correction);
                }

                // If the left wall is moving, check the correct bounce-back
                if (walls.right.vel_x > 0 || walls.right.vel_y > 0) {
                    density = current_density(right_layer_x, y);

                    wall_vel_x = walls.right.vel_x;
                    wall_vel_y = walls.right.vel_y;

                    // The horizontal walls (top and bottom) take priority and
                    // will overwrite the bounce-back of the vertical walls
                    // (left and right). So the corners are a special case.

                    // If y = 0 (i.e we're at the bottom corner), use the bottom
                    // wall velocity instead Even if the bottom wall isn't
                    // moving, it's bounce-back will over-write the horizontal
                    // walls
                    if (y == 0) {
                        correction = correction_down_right(
                            right_layer_x, y, density, walls.bottom.vel_x,
                            walls.bottom.vel_y);
                    } else {
                        correction = correction_down_right(
                            right_layer_x, y, density, wall_vel_x, wall_vel_y);
                    }

                    check_bounce_back_down_right(current_distribution,
                                                 previous_distribution,
                                                 right_layer_x, y, correction);

                    correction = correction_right(right_layer_x, y, density,
                                                  wall_vel_x, wall_vel_y);

                    check_bounce_back_right(current_distribution,
                                            previous_distribution,
                                            right_layer_x, y, correction);

                    // If y = lattice_height - 1 (i.e we're at the top corner),
                    // use the top wall velocity instead Even if the bottom wall
                    // isn't moving, it's bounce-back will over-write the
                    // horizontal walls
                    if (y == lattice_height - 1) {
                        correction = correction_up_right(
                            right_layer_x, y, density, walls.top.vel_x,
                            walls.top.vel_y);
                    } else {
                        correction = correction_up_right(
                            right_layer_x, y, density, wall_vel_x, wall_vel_y);
                    }

                    check_bounce_back_up_right(current_distribution,
                                               previous_distribution,
                                               right_layer_x, y, correction);
                }
            };
        }
    }
}

TEST(MILESTONE05, MOVING_WALL_TOP) {
    const int lattice_width = 20, lattice_height = 20;

    const LatticeBoltzmann::Walls walls{
        // All Walls as fixed walls with bounce back
        LatticeBoltzmann::Wall(0, 0),
        LatticeBoltzmann::Wall(0, 0),
        LatticeBoltzmann::Wall(0, 0),
        // Top Wall with 0.1 velocity to the right
        LatticeBoltzmann::Wall(0.1, 0),
    };
    moving_wall_test(lattice_width, lattice_height, walls);
}

TEST(MILESTONE05, MOVING_WALL_BOTTOM) {
    const int lattice_width = 20, lattice_height = 20;

    // All Walls as fixed walls with bounce back
    // Bottom Wall with 0.1 velocity to the right
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall(0, 0),
        LatticeBoltzmann::Wall(0.1, 0),
        LatticeBoltzmann::Wall(0, 0),
        LatticeBoltzmann::Wall(0, 0),
    };
    moving_wall_test(lattice_width, lattice_height, walls);
}

TEST(MILESTONE05, MOVING_WALL_TOP_AND_RIGHT) {
    const int lattice_width = 20, lattice_height = 20;

    // All Walls as fixed walls with bounce back
    // Bottom Wall with 0.1 velocity to the right
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall(0, 0.08),
        LatticeBoltzmann::Wall(0, 0),
        LatticeBoltzmann::Wall(0, 0),
        LatticeBoltzmann::Wall(0.1, 0),
    };

    moving_wall_test(lattice_width, lattice_height, walls);
}

TEST(MILESTONE05, MOVING_WALL_ALL) {
    const int lattice_width = 20, lattice_height = 20;

    // All Walls as fixed walls with bounce back
    // Bottom Wall with 0.1 velocity to the right
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall(0, 0.08),
        LatticeBoltzmann::Wall(-0.05, 0),
        LatticeBoltzmann::Wall(0, 0.12),
        LatticeBoltzmann::Wall(0.1, 0),
    };

    moving_wall_test(lattice_width, lattice_height, walls);
}
