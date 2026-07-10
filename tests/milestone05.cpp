#include <gtest/gtest.h>

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

    DistributionFunctionOutput output{"ms5test.csv"};

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
