#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

void fixed_walls(const int lattice_width, const int lattice_height,
                 const LatticeBoltzmann::Walls &walls) {
    auto lbm_functions =
        LatticeBoltzmann::Functions(lattice_width, lattice_height);

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    // Omega doesn't matter much here, but lower is better so the system reaches
    // an equilibrium slower
    constexpr double omega = 0.5;

    // For storing the distribution before the streaming step
    LatticeBoltzmann::HostDistributionMirror previous_distribution =
        Kokkos::create_mirror(lbm_functions.distribution_function);

    // Perform the Lattice Boltzmann step with boundary check 20 times and test
    // bounce back correctness each time
    for (int i = 0; i < 10; i++) {
        LatticeBoltzmann::calculate_density(lbm_functions, walls);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions,
                                                           walls);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions,
                                                             walls);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega, walls);

        Kokkos::deep_copy(previous_distribution,
                          lbm_functions.distribution_function);

        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(lbm_functions,
                                                                  walls);

        Kokkos::deep_copy(lbm_functions.host_distribution_function,
                          lbm_functions.distribution_function);

        // Horizontal Walls
        for (int y = walls.bottom.ghost_layers;
             y < lattice_height - walls.top.ghost_layers; y++) {
            // Left Wall
            ASSERT_EQ(lbm_functions.host_distribution_function(
                          walls.left.ghost_layers, y, Direction::UpRight),
                      previous_distribution(walls.left.ghost_layers, y,
                                            Direction::DownLeft))
                << "Iteration = " << i << " X = " << walls.left.ghost_layers
                << " Y = " << y;

            ASSERT_EQ(lbm_functions.host_distribution_function(
                          walls.left.ghost_layers, y, Direction::Right),
                      previous_distribution(walls.left.ghost_layers, y,
                                            Direction::Left));

            ASSERT_EQ(lbm_functions.host_distribution_function(
                          walls.left.ghost_layers, y, Direction::DownRight),
                      previous_distribution(walls.left.ghost_layers, y,
                                            Direction::UpLeft));

            // Right Wall
            ASSERT_EQ(lbm_functions.host_distribution_function(
                          lattice_width - walls.right.ghost_layers - 1, y,
                          Direction::UpLeft),
                      previous_distribution(lattice_width -
                                                walls.right.ghost_layers - 1,
                                            y, Direction::DownRight));

            ASSERT_EQ(lbm_functions.host_distribution_function(
                          lattice_width - walls.right.ghost_layers - 1, y,
                          Direction::Left),
                      previous_distribution(lattice_width -
                                                walls.right.ghost_layers - 1,
                                            y, Direction::Right));

            ASSERT_EQ(lbm_functions.host_distribution_function(
                          lattice_width - walls.right.ghost_layers - 1, y,
                          Direction::DownLeft),
                      previous_distribution(lattice_width -
                                                walls.right.ghost_layers - 1,
                                            y, Direction::UpRight));

            // Ensure Center Values Don't Move
            ASSERT_EQ(lbm_functions.host_distribution_function(
                          walls.left.ghost_layers, y, Direction::Center),
                      previous_distribution(walls.left.ghost_layers, y,
                                            Direction::Center));

            ASSERT_EQ(lbm_functions.host_distribution_function(
                          lattice_width - walls.right.ghost_layers - 1, y,
                          Direction::Center),
                      previous_distribution(lattice_width -
                                                walls.right.ghost_layers - 1,
                                            y, Direction::Center));
        };

        // Vertical Walls
        for (int x = walls.left.ghost_layers;
             x < lattice_width - walls.right.ghost_layers; x++) {
            // Bottom Wall
            ASSERT_EQ(lbm_functions.host_distribution_function(
                          x, walls.bottom.ghost_layers, Direction::UpLeft),
                      previous_distribution(x, walls.bottom.ghost_layers,
                                            Direction::DownRight));

            ASSERT_EQ(lbm_functions.host_distribution_function(
                          x, walls.bottom.ghost_layers, Direction::Up),
                      previous_distribution(x, walls.bottom.ghost_layers,
                                            Direction::Down));

            ASSERT_EQ(lbm_functions.host_distribution_function(
                          x, walls.bottom.ghost_layers, Direction::UpRight),
                      previous_distribution(x, walls.bottom.ghost_layers,
                                            Direction::DownLeft));

            // Top Wall
            ASSERT_EQ(lbm_functions.host_distribution_function(
                          x, lattice_height - walls.top.ghost_layers - 1,
                          Direction::DownRight),
                      previous_distribution(
                          x, lattice_height - walls.top.ghost_layers - 1,
                          Direction::UpLeft));

            ASSERT_EQ(lbm_functions.host_distribution_function(
                          x, lattice_height - walls.top.ghost_layers - 1,
                          Direction::Down),
                      previous_distribution(
                          x, lattice_height - walls.top.ghost_layers - 1,
                          Direction::Up));

            ASSERT_EQ(lbm_functions.host_distribution_function(
                          x, lattice_height - walls.top.ghost_layers - 1,
                          Direction::DownLeft),
                      previous_distribution(
                          x, lattice_height - walls.top.ghost_layers - 1,
                          Direction::UpRight));

            // Ensure Center Values Don't Move
            ASSERT_EQ(lbm_functions.host_distribution_function(
                          x, walls.bottom.ghost_layers, Direction::Center),
                      previous_distribution(x, walls.bottom.ghost_layers,
                                            Direction::Center));
            ASSERT_EQ(lbm_functions.host_distribution_function(
                          x, lattice_height - walls.top.ghost_layers - 1,
                          Direction::Center),
                      previous_distribution(
                          x, lattice_height - walls.top.ghost_layers - 1,
                          Direction::Center));
        };
    }
}

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK_NO_GHOST_LAYER) {
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
    };

    fixed_walls(20, 20, walls);
}

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK_LEFT_GHOST_LAYER) {
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack, 1),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
    };

    fixed_walls(20, 20, walls);
}

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK_RIGHT_GHOST_LAYER) {
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack, 1),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
    };
    fixed_walls(20, 20, walls);
}

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK_TOP_GHOST_LAYER) {
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack, 1),
    };
    fixed_walls(20, 20, walls);
}

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK_BOTTOM_GHOST_LAYER) {
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack, 1),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
    };
    fixed_walls(20, 20, walls);
}

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK_ALL_GHOST_LAYER) {
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack, 1),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack, 1),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack, 1),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack, 1),
    };
    fixed_walls(20, 20, walls);
}

double correction_left(const int &x, const int &y, const double &density,
                       const double &wall_vel_x, const double &wall_vel_y) {
    // Left
    constexpr int vec_x = -1, vec_y = 0;                             // Vector
    constexpr double weight = 1.0 / 9.0;                             // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_left(
    const LatticeBoltzmann::HostDistributionMirror &distribution_function,
    const LatticeBoltzmann::HostDistributionMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_left) {

    const double current_value = distribution_function(x, y, Direction::Right);
    const double previous_value =
        previous_distribution_function(x, y, Direction::Left);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_left);
}

double correction_up_left(const int &x, const int &y, const double &density,
                          const double &wall_vel_x, const double &wall_vel_y) {
    // Up Left
    constexpr int vec_x = -1, vec_y = 1;                             // Vector
    constexpr double weight = 1.0 / 36.0;                            // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_up_left(
    const LatticeBoltzmann::HostDistributionMirror &distribution_function,
    const LatticeBoltzmann::HostDistributionMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_up_left) {

    const double current_value =
        distribution_function(x, y, Direction::DownRight);
    const double previous_value =
        previous_distribution_function(x, y, Direction::UpLeft);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_up_left);
}

double correction_up(const int &x, const int &y, const double &density,
                     const double &wall_vel_x, const double &wall_vel_y) {
    // Up
    constexpr int vec_x = 0, vec_y = 1;                              // Vector
    constexpr double weight = 1.0 / 9.0;                             // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_up(
    const LatticeBoltzmann::HostDistributionMirror &distribution_function,
    const LatticeBoltzmann::HostDistributionMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_up) {

    const double current_value = distribution_function(x, y, Direction::Down);
    const double previous_value =
        previous_distribution_function(x, y, Direction::Up);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_up);
}

double correction_up_right(const int &x, const int &y, const double &density,
                           const double &wall_vel_x, const double &wall_vel_y) {
    // Up Right
    constexpr int vec_x = 1, vec_y = 1;                              // Vector
    constexpr double w5 = 1.0 / 36.0;                                // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * w5 * density * dot_prod;
};

void check_bounce_back_up_right(
    const LatticeBoltzmann::HostDistributionMirror &distribution_function,
    const LatticeBoltzmann::HostDistributionMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_up_right) {

    const double current_value =
        distribution_function(x, y, Direction::DownLeft);
    const double previous_value =
        previous_distribution_function(x, y, Direction::UpRight);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_up_right);
}

double correction_right(const int &x, const int &y, const double &density,
                        const double &wall_vel_x, const double &wall_vel_y) {
    // Right
    constexpr int vec_x = 1, vec_y = 0;                              // Vector
    constexpr double weight = 1.0 / 9.0;                             // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_right(
    const LatticeBoltzmann::HostDistributionMirror &distribution_function,
    const LatticeBoltzmann::HostDistributionMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_right) {

    const double current_value = distribution_function(x, y, Direction::Left);
    const double previous_value =
        previous_distribution_function(x, y, Direction::Right);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_right);
}

double correction_down_right(const int &x, const int &y, const double &density,
                             const double &wall_vel_x,
                             const double &wall_vel_y) {
    // Down Right
    constexpr int vec_x = 1, vec_y = -1;                             // Vector
    constexpr double weight = 1.0 / 36.0;                            // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_down_right(
    const LatticeBoltzmann::HostDistributionMirror &distribution_function,
    const LatticeBoltzmann::HostDistributionMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_down_right) {

    const double current_value = distribution_function(x, y, Direction::UpLeft);
    const double previous_value =
        previous_distribution_function(x, y, Direction::DownRight);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_down_right);
}

double correction_down(const int &x, const int &y, const double &density,
                       const double &wall_vel_x, const double &wall_vel_y) {
    // Down
    constexpr int vec_x = 0, vec_y = -1;                             // Vector
    constexpr double weight = 1.0 / 9.0;                             // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_down(
    const LatticeBoltzmann::HostDistributionMirror &distribution_function,
    const LatticeBoltzmann::HostDistributionMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_down) {

    const double current_value = distribution_function(x, y, Direction::Up);
    const double previous_value =
        previous_distribution_function(x, y, Direction::Down);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_down);
}

double correction_down_left(const int &x, const int &y, const double &density,
                            const double &wall_vel_x,
                            const double &wall_vel_y) {
    // Down Left
    constexpr int vec_x = -1, vec_y = -1;                            // Vector
    constexpr double weight = 1.0 / 36.0;                            // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_down_left(
    const LatticeBoltzmann::HostDistributionMirror &distribution_function,
    const LatticeBoltzmann::HostDistributionMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_down_left) {

    const double current_value =
        distribution_function(x, y, Direction::UpRight);
    const double previous_value =
        previous_distribution_function(x, y, Direction::DownLeft);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_down_left);
}

TEST(MILESTONE05, MOVING_WALL_TOP) {
    const int lattice_width = 20, lattice_height = 20;

    constexpr double top_wall_vel_x = 0.1, top_wall_vel_y = 0;

    // No Ghost Layers
    const LatticeBoltzmann::Walls walls{
        // All Walls as fixed walls with bounce back
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        LatticeBoltzmann::Wall(LatticeBoltzmann::WallType::BounceBack),
        // Top Wall with 0.1 velocity to the right
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::WallType::BounceBack,
            0,
            top_wall_vel_x,
            top_wall_vel_y,
        },
    };

    auto lbm_functions =
        LatticeBoltzmann::Functions(lattice_width, lattice_height);

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    // Omega doesn't matter much here, but lower is better so the system reaches
    // an equilibrium slower
    constexpr double omega = 0.5;

    // For storing the distribution before the streaming step
    LatticeBoltzmann::HostDistributionMirror previous_distribution =
        Kokkos::create_mirror(lbm_functions.distribution_function);

    // Perform the Lattice Boltzmann step with boundary check 20 times and test
    // bounce back correctness each time
    for (int i = 0; i < 10; i++) {
        LatticeBoltzmann::calculate_density(lbm_functions, walls);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions,
                                                           walls);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions,
                                                             walls);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega, walls);

        Kokkos::deep_copy(previous_distribution,
                          lbm_functions.distribution_function);

        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(lbm_functions,
                                                                  walls);

        Kokkos::deep_copy(lbm_functions.host_distribution_function,
                          lbm_functions.distribution_function);
        Kokkos::deep_copy(lbm_functions.host_density_function,
                          lbm_functions.density_function);

        const int top_layer_y = lattice_height - 1;
        for (int x = 0; x < lattice_width; x++) {
            double current_value, previous_value, correction;

            double density = lbm_functions.density_function(x, top_layer_y);

            // Check Up Right
            correction = correction_up_right(x, top_layer_y, density,
                                             top_wall_vel_x, top_wall_vel_y);
            check_bounce_back_up_right(lbm_functions.host_distribution_function,
                                       previous_distribution, x, top_layer_y,
                                       correction);

            // Check Up
            correction = correction_up(x, top_layer_y, density, top_wall_vel_x,
                                       top_wall_vel_y);
            check_bounce_back_up(lbm_functions.host_distribution_function,
                                 previous_distribution, x, top_layer_y,
                                 correction);

            // Check Up Left
            correction = correction_up_left(x, top_layer_y, density,
                                            top_wall_vel_x, top_wall_vel_y);
            check_bounce_back_up_left(lbm_functions.host_distribution_function,
                                      previous_distribution, x, top_layer_y,
                                      correction);
        }
    };
}

TEST(MILESTONE05, MOVING_WALL_LEFT) {
    // Allocate a 20 x 20 lattice
    // With all walls set as bounce back
    // No ghost layers
    // Left wall being a moving wall with velocity (0, -0.08)
    const int lattice_width = 20, lattice_height = 20;
    constexpr double left_wall_vel_x = 0, left_wall_vel_y = -0.08;

    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall{LatticeBoltzmann::WallType::BounceBack}, // Right
        LatticeBoltzmann::Wall{LatticeBoltzmann::WallType::BounceBack}, // Down
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::WallType::BounceBack,
            0,
            left_wall_vel_x,
            left_wall_vel_y,
        },                                                              // Left
        LatticeBoltzmann::Wall{LatticeBoltzmann::WallType::BounceBack}, // Top
    };
    auto lbm_functions =
        LatticeBoltzmann::Functions(lattice_width, lattice_height);

    // Initialize the distribution with a random density
    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    // Omega doesn't matter much here, but lower is better so the system reaches
    // an equilibrium slower
    constexpr double omega = 0.5;

    // For storing the distribution before the streaming step
    LatticeBoltzmann::HostDistributionMirror previous_distribution =
        Kokkos::create_mirror(lbm_functions.distribution_function);

    // Perform the Lattice Boltzmann step with boundary check 20 times and test
    // bounce back correctness each time
    for (int i = 0; i < 10; i++) {
        // Perform a Lattice-Boltzmann step until the streaming and bounce back
        LatticeBoltzmann::calculate_density(lbm_functions, walls);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions,
                                                           walls);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions,
                                                             walls);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega, walls);

        // Store the current relaxation before streaming step
        Kokkos::deep_copy(previous_distribution,
                          lbm_functions.distribution_function);

        // Perform the streaming and bounce back
        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(lbm_functions,
                                                                  walls);

        // Copy to host views (only relevant if test is run on GPU)
        Kokkos::deep_copy(lbm_functions.host_distribution_function,
                          lbm_functions.distribution_function);
        Kokkos::deep_copy(lbm_functions.host_density_function,
                          lbm_functions.density_function);

        // Left-most wall has x index of 0
        const int left_layer_x = 0;

        double current_value, previous_value, local_density, correction;

        // The horizontal walls (top and bottom) will always over-write the
        // results of the vertical wall bounce-back (due to implementation)
        // So the corners are a special case and we will check if the y value is
        // either 0 (bottom) or lattice_height - 1 (top) and use velocity 0 for
        // the walls

        for (int y = 1; y < lattice_height - 1; y++) {
            local_density = lbm_functions.density_function(left_layer_x, y);

            // Check top corner case
            if (y == lattice_height - 1) {
                correction =
                    correction_up_left(left_layer_x, y, local_density, 0, 0);
            } else {
                correction =
                    correction_up_left(left_layer_x, y, local_density,
                                       left_wall_vel_x, left_wall_vel_y);
            }
            check_bounce_back_up_left(lbm_functions.host_distribution_function,
                                      previous_distribution, left_layer_x, y,
                                      correction);

            correction = correction_left(left_layer_x, y, local_density,
                                         left_wall_vel_x, left_wall_vel_y);
            check_bounce_back_left(lbm_functions.host_distribution_function,
                                   previous_distribution, left_layer_x, y,
                                   correction);

            // Check bottom corner case
            if (y == 0) {
                correction =
                    correction_down_left(left_layer_x, y, local_density, 0, 0);
            } else {
                correction =
                    correction_down_left(left_layer_x, y, local_density,
                                         left_wall_vel_x, left_wall_vel_y);
            }
            check_bounce_back_down_left(
                lbm_functions.host_distribution_function, previous_distribution,
                left_layer_x, y, correction);
        }
    };
}

TEST(MILESTONE05, MOVING_WALL_BOTTOM_AND_RIGHT) {
    // Allocate a 20 x 20 lattice
    // With left and top walls set as fixed bounce back with no ghost layers
    // Right wall set as a moving wall with velocity (0, 0.02) and 1 ghost layer
    // Bottom wall set as a moving wall with velocity (-0.1, 0) and no ghost
    // layer

    // We also test a non-square dimension of the lattice
    const int lattice_width = 20, lattice_height = 25;
    constexpr double right_wall_vel_x = 0, right_wall_vel_y = 0.02;
    constexpr double bottom_wall_vel_x = -0.1, bottom_wall_vel_y = 0;

    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::WallType::BounceBack,
            1,
            right_wall_vel_x,
            right_wall_vel_y,
        }, // Right
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::WallType::BounceBack,
            0,
            bottom_wall_vel_x,
            bottom_wall_vel_y,
        }, // Bottom
        LatticeBoltzmann::Wall{LatticeBoltzmann::WallType::BounceBack}, // Left
        LatticeBoltzmann::Wall{LatticeBoltzmann::WallType::BounceBack}, // Top
    };
    auto lbm_functions =
        LatticeBoltzmann::Functions(lattice_width, lattice_height);

    // Initialize the distribution with a random density
    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    // Omega doesn't matter much here, but lower is better so the system reaches
    // an equilibrium slower
    constexpr double omega = 0.5;

    // For storing the distribution before the streaming step
    LatticeBoltzmann::HostDistributionMirror previous_distribution =
        Kokkos::create_mirror(lbm_functions.distribution_function);

    // Perform the Lattice Boltzmann step with boundary check 20 times and test
    // bounce back correctness each time
    for (int i = 0; i < 10; i++) {
        // Perform a Lattice-Boltzmann step until the streaming and bounce back
        LatticeBoltzmann::calculate_density(lbm_functions, walls);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions,
                                                           walls);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions,
                                                             walls);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega, walls);

        // Store the current relaxation before streaming step
        Kokkos::deep_copy(previous_distribution,
                          lbm_functions.distribution_function);

        // Perform the streaming and bounce back
        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(lbm_functions,
                                                                  walls);

        // Copy to host views (only relevant if test is run on GPU)
        Kokkos::deep_copy(lbm_functions.host_distribution_function,
                          lbm_functions.distribution_function);
        Kokkos::deep_copy(lbm_functions.host_density_function,
                          lbm_functions.density_function);

        // Left-most wall has x index of 0
        const int bottom_layer_y = 0, right_layer_x = lattice_width - 2;

        double current_value, previous_value, local_density, correction;

        // The horizontal walls (top and bottom) will always over-write the
        // results of the vertical wall bounce-back (due to implementation)
        // So the corners are a special case.

        // Loop horizontally over the bottom lattice cells and check bounce-back
        // NOTE: We do `lattice_width - 1` due to the ghost layer on the right
        for (int x = 0; x < lattice_width - 1; x++) {

            // NOTE: The bottom wall will always over-write the right wall, so
            // we have no special case when looping through the bottom cells

            local_density =
                lbm_functions.host_density_function(x, bottom_layer_y);

            correction =
                correction_down_left(x, bottom_layer_y, local_density,
                                     bottom_wall_vel_x, bottom_wall_vel_y);
            check_bounce_back_down_left(
                lbm_functions.host_distribution_function, previous_distribution,
                x, bottom_layer_y, correction);

            correction = correction_down(x, bottom_layer_y, local_density,
                                         bottom_wall_vel_x, bottom_wall_vel_y);
            check_bounce_back_down(lbm_functions.host_distribution_function,
                                   previous_distribution, x, bottom_layer_y,
                                   correction);

            correction =
                correction_down_right(x, bottom_layer_y, local_density,
                                      bottom_wall_vel_x, bottom_wall_vel_y);
            check_bounce_back_down_right(
                lbm_functions.host_distribution_function, previous_distribution,
                x, bottom_layer_y, correction);
        }

        // Loop vertically through the right-most lattice cells and check their
        // bounce-back values
        for (int y = 0; y < lattice_height; y++) {
            local_density =
                lbm_functions.host_density_function(right_layer_x, y);

            // The top wall with 0 velocity over-writes the top right corner
            // lattice cell's up_right bounce back, so we check the corner case
            // (pun intended) with fixed wall velocity of 0
            if (y == lattice_height - 1) {
                correction =
                    correction_up_right(right_layer_x, y, local_density, 0, 0);
            } else {
                correction =
                    correction_up_right(right_layer_x, y, local_density,
                                        right_wall_vel_x, right_wall_vel_y);
            }
            check_bounce_back_up_right(lbm_functions.host_distribution_function,
                                       previous_distribution, right_layer_x, y,
                                       correction);

            correction = correction_right(right_layer_x, y, local_density,
                                          right_wall_vel_x, right_wall_vel_y);
            check_bounce_back_right(lbm_functions.host_distribution_function,
                                    previous_distribution, right_layer_x, y,
                                    correction);

            // In the bottom right corner, the bottom wall velocity takes
            // priority. For redundancy we just check using the bottom wall
            // velocity instead of the right wall.
            if (y == 0) {
                correction =
                    correction_down_right(right_layer_x, y, local_density,
                                          bottom_wall_vel_x, bottom_wall_vel_y);
            } else {
                correction =
                    correction_down_right(right_layer_x, y, local_density,
                                          right_wall_vel_x, right_wall_vel_y);
            }
            check_bounce_back_down_right(
                lbm_functions.host_distribution_function, previous_distribution,
                right_layer_x, y, correction);
        }
    };
}