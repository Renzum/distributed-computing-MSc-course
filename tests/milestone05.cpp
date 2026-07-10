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

double correction_up_left(const int &x, const int &y, const double &density,
                          const double &lid_vel_x, const double &lid_vel_y) {
    // Up Left
    constexpr int vec_x = -1, vec_y = 1;                           // Vector
    constexpr double weight = 1.0 / 36.0;                          // Weight
    const double dot_prod = lid_vel_x * vec_x + lid_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};
double correction_up(const int &x, const int &y, const double &density,
                     const double &lid_vel_x, const double &lid_vel_y) {
    // Up
    constexpr int vec_x = 0, vec_y = 1;                            // Vector
    constexpr double weight = 1.0 / 9.0;                           // Weight
    const double dot_prod = lid_vel_x * vec_x + lid_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};
double correction_up_right(const int &x, const int &y, const double &density,
                           const double &lid_vel_x, const double &lid_vel_y) {
    // Up Right
    constexpr int vec_x = 1, vec_y = 1;                            // Vector
    constexpr double w5 = 1.0 / 36.0;                              // Weight
    const double dot_prod = lid_vel_x * vec_x + lid_vel_y * vec_y; // c * u

    return 6 * w5 * density * dot_prod;
};

double correction_right(const int &x, const int &y, const double &density,
                        const double &lid_vel_x, const double &lid_vel_y) {
    // Right
    constexpr int vec_x = 1, vec_y = 0;                            // Vector
    constexpr double weight = 1.0 / 9.0;                           // Weight
    const double dot_prod = lid_vel_x * vec_x + lid_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

double correction_down_right(const int &x, const int &y, const double &density,
                             const double &lid_vel_x, const double &lid_vel_y) {
    // Down Right
    constexpr int vec_x = 1, vec_y = -1;                           // Vector
    constexpr double weight = 1.0 / 36.0;                          // Weight
    const double dot_prod = lid_vel_x * vec_x + lid_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

double correction_down(const int &x, const int &y, const double &density,
                       const double &lid_vel_x, const double &lid_vel_y) {
    // Down
    constexpr int vec_x = 0, vec_y = -1;                           // Vector
    constexpr double weight = 1.0 / 9.0;                           // Weight
    const double dot_prod = lid_vel_x * vec_x + lid_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

double correction_down_left(const int &x, const int &y, const double &density,
                            const double &lid_vel_x, const double &lid_vel_y) {
    // Down Left
    constexpr int vec_x = -1, vec_y = -1;                          // Vector
    constexpr double weight = 1.0 / 36.0;                          // Weight
    const double dot_prod = lid_vel_x * vec_x + lid_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

double correction_left(const int &x, const int &y, const double &density,
                       const double &lid_vel_x, const double &lid_vel_y) {
    // Left
    constexpr int vec_x = -1, vec_y = 0;                           // Vector
    constexpr double weight = 1.0 / 9.0;                           // Weight
    const double dot_prod = lid_vel_x * vec_x + lid_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

TEST(MILESTONE05, MOVING_WALL_TOP) {
    const int lattice_width = 20, lattice_height = 20;

    constexpr double t_wall_vel_x = 0.1, t_wall_vel_y = 0;

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
            t_wall_vel_x,
            t_wall_vel_y,
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

            // Check Down Left
            current_value = lbm_functions.distribution_function(
                x, top_layer_y, Direction::DownLeft);
            previous_value =
                previous_distribution(x, top_layer_y, Direction::UpRight);
            correction = correction_up_right(x, top_layer_y, density,
                                             t_wall_vel_x, t_wall_vel_y);
            ASSERT_DOUBLE_EQ(current_value, previous_value - correction);

            // Check Down
            current_value = lbm_functions.distribution_function(
                x, top_layer_y, Direction::Down);
            previous_value =
                previous_distribution(x, top_layer_y, Direction::Up);
            correction = correction_up(x, top_layer_y, density, t_wall_vel_x,
                                       t_wall_vel_y);
            ASSERT_DOUBLE_EQ(current_value, previous_value - correction);

            // Check Down Right
            current_value = lbm_functions.distribution_function(
                x, top_layer_y, Direction::DownRight);
            previous_value =
                previous_distribution(x, top_layer_y, Direction::UpLeft);
            correction = correction_up_left(x, top_layer_y, density,
                                            t_wall_vel_x, t_wall_vel_y);
            ASSERT_DOUBLE_EQ(current_value, previous_value - correction);
        }
    };
}

TEST(MILESTONE05, MOVING_WALL_LEFT) {
    const int lattice_width = 20, lattice_height = 20;

    constexpr double l_wall_vel_x = 0, l_wall_vel_y = -0.08;

    // No Ghost Layers
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall{LatticeBoltzmann::WallType::BounceBack},
        LatticeBoltzmann::Wall{LatticeBoltzmann::WallType::BounceBack},
        LatticeBoltzmann::Wall{LatticeBoltzmann::WallType::BounceBack, 0,
                               l_wall_vel_x, l_wall_vel_y},
        LatticeBoltzmann::Wall{LatticeBoltzmann::WallType::BounceBack},
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

        const int left_layer_x = 0;

        double current_value, previous_value, correction;
        double density = lbm_functions.density_function(left_layer_x, 0);

        // The horizontal walls (top and bottom) will always over-write the
        // results of the vertical wall bounce-back.
        // So the corners are a special case

        // Bottom Left Corner
#pragma region Bottom Left Corner Check
        // Check Up Right
        current_value = lbm_functions.distribution_function(left_layer_x, 0,
                                                            Direction::UpRight);
        previous_value =
            previous_distribution(left_layer_x, 0, Direction::DownLeft);
        correction = correction_down_left(left_layer_x, 0, density, 0, 0);
        ASSERT_DOUBLE_EQ(current_value, previous_value - correction)
            << " X = " << left_layer_x << " Y = " << 0
            << " Previous Value = " << previous_value
            << " Correction = " << correction
            << " Current Value = " << current_value;

        // Check Down
        current_value = lbm_functions.distribution_function(left_layer_x, 0,
                                                            Direction::Right);
        previous_value =
            previous_distribution(left_layer_x, 0, Direction::Left);
        correction = correction_left(left_layer_x, 0, density, 0, 0);
        ASSERT_DOUBLE_EQ(current_value, previous_value - correction)
            << " X = " << left_layer_x << " Y = " << lattice_height - 1
            << " Previous Value = " << previous_value
            << " Correction = " << correction
            << " Current Value = " << current_value;

        // Check Down Right
        // (The bottom wall doesn't over-write this one, so we check with the
        // correct velocity)
        current_value = lbm_functions.distribution_function(
            left_layer_x, 0, Direction::DownRight);
        previous_value =
            previous_distribution(left_layer_x, 0, Direction::UpLeft);
        correction = correction_up_left(left_layer_x, 0, density, 0, -0.08);
        ASSERT_DOUBLE_EQ(current_value, previous_value - correction)
            << " X = " << left_layer_x << " Y = " << lattice_height - 1
            << " Previous Value = " << previous_value
            << " Correction = " << correction
            << " Current Value = " << current_value;
#pragma endregion

        // Bottom Left Corner
#pragma region Bottom Left Corner Check
        density =
            lbm_functions.density_function(left_layer_x, lattice_height - 1);

        // Check Up Right
        // (The top wall doesn't over-write this one, so we check with the
        // correct velocity)
        current_value = lbm_functions.distribution_function(
            left_layer_x, lattice_height - 1, Direction::UpRight);
        previous_value = previous_distribution(left_layer_x, lattice_height - 1,
                                               Direction::DownLeft);
        correction = correction_down_left(left_layer_x, lattice_height - 1,
                                          density, l_wall_vel_x, l_wall_vel_y);
        ASSERT_DOUBLE_EQ(current_value, previous_value - correction)
            << " X = " << left_layer_x << " Y = " << lattice_height - 1
            << " Previous Value = " << previous_value
            << " Correction = " << correction
            << " Current Value = " << current_value;

        // Check Down
        current_value = lbm_functions.distribution_function(
            left_layer_x, lattice_height - 1, Direction::Right);
        previous_value = previous_distribution(left_layer_x, lattice_height - 1,
                                               Direction::Left);
        correction =
            correction_left(left_layer_x, lattice_height - 1, density, 0, 0);
        ASSERT_DOUBLE_EQ(current_value, previous_value - correction)
            << " X = " << left_layer_x << " Y = " << lattice_height - 1
            << " Previous Value = " << previous_value
            << " Correction = " << correction
            << " Current Value = " << current_value;

        // Check Down Right
        current_value = lbm_functions.distribution_function(
            left_layer_x, lattice_height - 1, Direction::DownRight);
        previous_value = previous_distribution(left_layer_x, lattice_height - 1,
                                               Direction::UpLeft);
        correction =
            correction_up_left(left_layer_x, lattice_height - 1, density, 0, 0);
        ASSERT_DOUBLE_EQ(current_value, previous_value - correction)
            << " X = " << left_layer_x << " Y = " << 0
            << " Previous Value = " << previous_value
            << " Correction = " << correction
            << " Current Value = " << current_value;
#pragma endregion

        for (int y = 1; y < lattice_height - 1; y++) {
            double current_value, previous_value, correction;

            density = lbm_functions.density_function(left_layer_x, y);
            // Check Up Right
            current_value = lbm_functions.distribution_function(
                left_layer_x, y, Direction::UpRight);
            previous_value =
                previous_distribution(left_layer_x, y, Direction::DownLeft);
            correction = correction_down_left(left_layer_x, y, density,
                                              l_wall_vel_x, l_wall_vel_y);
            ASSERT_DOUBLE_EQ(current_value, previous_value - correction)
                << " X = " << left_layer_x << " Y = " << y
                << " Previous Value = " << previous_value
                << " Correction = " << correction
                << " Current Value = " << current_value;

            // Check Down
            current_value = lbm_functions.distribution_function(
                left_layer_x, y, Direction::Right);
            previous_value =
                previous_distribution(left_layer_x, y, Direction::Left);
            correction = correction_left(left_layer_x, y, density, l_wall_vel_x,
                                         l_wall_vel_y);
            ASSERT_DOUBLE_EQ(current_value, previous_value - correction);

            // Check Down Right
            current_value = lbm_functions.distribution_function(
                left_layer_x, y, Direction::DownRight);
            previous_value =
                previous_distribution(left_layer_x, y, Direction::UpLeft);
            correction = correction_up_left(left_layer_x, y, density,
                                            l_wall_vel_x, l_wall_vel_y);
            ASSERT_DOUBLE_EQ(current_value, previous_value - correction);
        }
    };
}