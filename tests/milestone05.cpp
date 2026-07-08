#include <gtest/gtest.h>

#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

void fixed_walls(const int lattice_width, const int lattice_height,
                 const LatticeBoltzmann::GhostLayers &ghost_layers = {}) {
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
        LatticeBoltzmann::calculate_density(lbm_functions, ghost_layers);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions,
                                                           ghost_layers);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions,
                                                             ghost_layers);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega,
                                             ghost_layers);

        Kokkos::deep_copy(previous_distribution,
                          lbm_functions.distribution_function);
        // output.output(previous_distribution, i);

        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(
            lbm_functions, 0, 0, ghost_layers);

        Kokkos::deep_copy(lbm_functions.host_distribution_function,
                          lbm_functions.distribution_function);
        // output.output(lbm_functions.host_distribution_function, ++i);

        // Horizontal Walls
        for (int y = ghost_layers.bottom; y < lattice_height - ghost_layers.top;
             y++) {
            // Left Wall
            ASSERT_EQ(lbm_functions.host_distribution_function(
                          ghost_layers.left, y, Direction::UpRight),
                      previous_distribution(ghost_layers.left, y,
                                            Direction::DownLeft))
                << "Iteration = " << i << " X = " << ghost_layers.left
                << " Y = " << y;

            ASSERT_EQ(
                lbm_functions.host_distribution_function(ghost_layers.left, y,
                                                         Direction::Right),
                previous_distribution(ghost_layers.left, y, Direction::Left));

            ASSERT_EQ(
                lbm_functions.host_distribution_function(ghost_layers.left, y,
                                                         Direction::DownRight),
                previous_distribution(ghost_layers.left, y, Direction::UpLeft));

            // Right Wall
            ASSERT_EQ(
                lbm_functions.host_distribution_function(
                    lattice_width - ghost_layers.right - 1, y,
                    Direction::UpLeft),
                previous_distribution(lattice_width - ghost_layers.right - 1, y,
                                      Direction::DownRight));

            ASSERT_EQ(
                lbm_functions.host_distribution_function(
                    lattice_width - ghost_layers.right - 1, y, Direction::Left),
                previous_distribution(lattice_width - ghost_layers.right - 1, y,
                                      Direction::Right));

            ASSERT_EQ(
                lbm_functions.host_distribution_function(
                    lattice_width - ghost_layers.right - 1, y,
                    Direction::DownLeft),
                previous_distribution(lattice_width - ghost_layers.right - 1, y,
                                      Direction::UpRight));

            // Ensure Center Values Don't Move
            ASSERT_EQ(
                lbm_functions.host_distribution_function(ghost_layers.left, y,
                                                         Direction::Center),
                previous_distribution(ghost_layers.left, y, Direction::Center));

            ASSERT_EQ(
                lbm_functions.host_distribution_function(
                    lattice_width - ghost_layers.right - 1, y,
                    Direction::Center),
                previous_distribution(lattice_width - ghost_layers.right - 1, y,
                                      Direction::Center));
        };

        // Vertical Walls
        for (int x = ghost_layers.left; x < lattice_width - ghost_layers.right;
             x++) {
            // Bottom Wall
            ASSERT_EQ(lbm_functions.host_distribution_function(
                          x, ghost_layers.bottom, Direction::UpLeft),
                      previous_distribution(x, ghost_layers.bottom,
                                            Direction::DownRight));

            ASSERT_EQ(
                lbm_functions.host_distribution_function(x, ghost_layers.bottom,
                                                         Direction::Up),
                previous_distribution(x, ghost_layers.bottom, Direction::Down));

            ASSERT_EQ(lbm_functions.host_distribution_function(
                          x, ghost_layers.bottom, Direction::UpRight),
                      previous_distribution(x, ghost_layers.bottom,
                                            Direction::DownLeft));

            // Top Wall
            ASSERT_EQ(
                lbm_functions.host_distribution_function(
                    x, lattice_height - ghost_layers.top - 1,
                    Direction::DownRight),
                previous_distribution(x, lattice_height - ghost_layers.top - 1,
                                      Direction::UpLeft));

            ASSERT_EQ(
                lbm_functions.host_distribution_function(
                    x, lattice_height - ghost_layers.top - 1, Direction::Down),
                previous_distribution(x, lattice_height - ghost_layers.top - 1,
                                      Direction::Up));

            ASSERT_EQ(
                lbm_functions.host_distribution_function(
                    x, lattice_height - ghost_layers.top - 1,
                    Direction::DownLeft),
                previous_distribution(x, lattice_height - ghost_layers.top - 1,
                                      Direction::UpRight));

            // Ensure Center Values Don't Move
            ASSERT_EQ(lbm_functions.host_distribution_function(
                          x, ghost_layers.bottom, Direction::Center),
                      previous_distribution(x, ghost_layers.bottom,
                                            Direction::Center));
            ASSERT_EQ(
                lbm_functions.host_distribution_function(
                    x, lattice_height - ghost_layers.top - 1,
                    Direction::Center),
                previous_distribution(x, lattice_height - ghost_layers.top - 1,
                                      Direction::Center));
        };
    }
}

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK_NO_GHOST_LAYER) {
    fixed_walls(20, 20);
}

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK_LEFT_GHOST_LAYER) {
    fixed_walls(20, 20, {0, 0, 1, 0});
}

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK_RIGHT_GHOST_LAYER) {
    fixed_walls(20, 20, {1, 0, 0, 0});
}

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK_TOP_GHOST_LAYER) {
    fixed_walls(20, 20, {0, 0, 0, 1});
}

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK_BOTTOM_GHOST_LAYER) {
    fixed_walls(20, 20, {0, 1, 0, 0});
}

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK_ALL_GHOST_LAYER) {
    fixed_walls(20, 20, {1, 1, 1, 1});
}
