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
    fixed_walls(20, 20); //{1, 1, 1, 1});
}

// TEST(MILESTONE05, MOVING_LID) {
//     auto lbm_functions = LatticeBoltzmann::Functions(20, 20);
//     LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

//     // Omega doesn't matter much here, but lower is better so the system
//     // reaches an equilibrium slower
//     constexpr double omega = 0.5;

//     // For storing the distribution before the streaming step
//     Kokkos::View<double ***> previous_distribution(
//         "Previous Distribution View",
//         lbm_functions.distribution_function.extent_int(0),
//         lbm_functions.distribution_function.extent_int(1),
//         TOTAL_DIRECTIONS);

//     const double lid_vel_x = 0.1;
//     const double lid_vel_y = 0.0;

//     double velocity_fraction;
//     int velocity_vector_x, velocity_vector_y;

//     for (int i = 0; i < 20; i++) {
//         LatticeBoltzmann::calculate_density(lbm_functions);
//         LatticeBoltzmann::calculate_local_average_velocity(lbm_functions);
//         LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions);
//         LatticeBoltzmann::relax_distribution(lbm_functions, omega);

//         Kokkos::deep_copy(previous_distribution,
//                           lbm_functions.distribution_function);

//         LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(
//             lbm_functions, lid_vel_x, lid_vel_y);

//         const double tolerance = std::numeric_limits<double>::epsilon();

//         auto correction_calculation =
//             [&lbm_functions, &lid_vel_x, &lid_vel_y, &velocity_vector_x,
//              &velocity_vector_y, &velocity_fraction](
//                 const int &x, const int &y, const Direction &dir) ->
//                 double {
//             get_velocity_vector(static_cast<Direction>(dir),
//             velocity_vector_x,
//                                 velocity_vector_y);
//             get_velocity_fraction(static_cast<Direction>(dir),
//                                   velocity_fraction);

//             return 2.0 * velocity_fraction *
//                    lbm_functions.density_function(x, y) *
//                    (lid_vel_x * velocity_vector_x +
//                     lid_vel_y * velocity_vector_y) /
//                    (1.0 / 3.0);
//         };

//         // Top Left Corner
//         ASSERT_NEAR(lbm_functions.distribution_function(1, 18,
//         Direction::Down),
//                     previous_distribution(1, 18, Direction::Up) -
//                         correction_calculation(1, 18, Direction::Up),
//                     tolerance);
//         ASSERT_NEAR(lbm_functions.distribution_function(1, 18, DownLeft),
//                     previous_distribution(1, 18, Direction::UpRight) -
//                         correction_calculation(1, 18,
//                         Direction::UpRight),
//                     tolerance);
//         ASSERT_NEAR(
//             lbm_functions.distribution_function(1, 18,
//             Direction::DownRight), previous_distribution(1, 18,
//             Direction::UpLeft) -
//                 correction_calculation(1, 18, Direction::UpLeft),
//             tolerance);

//         // Top Middle Node
//         ASSERT_NEAR(
//             lbm_functions.distribution_function(10, 18, Direction::Down),
//             previous_distribution(10, 18, Direction::Up) -
//                 correction_calculation(10, 18, Direction::Up),
//             tolerance);
//         ASSERT_NEAR(lbm_functions.distribution_function(10, 18,
//         DownLeft),
//                     previous_distribution(10, 18, Direction::UpRight) -
//                         correction_calculation(10, 18,
//                         Direction::UpRight),
//                     tolerance);
//         ASSERT_NEAR(
//             lbm_functions.distribution_function(10, 18,
//             Direction::DownRight), previous_distribution(10, 18,
//             Direction::UpLeft) -
//                 correction_calculation(10, 18, Direction::UpLeft),
//             tolerance);

//         // Top Left Corner
//         ASSERT_NEAR(
//             lbm_functions.distribution_function(18, 18, Direction::Down),
//             previous_distribution(18, 18, Direction::Up) -
//                 correction_calculation(18, 18, Direction::Up),
//             tolerance);
//         ASSERT_NEAR(lbm_functions.distribution_function(18, 18,
//         DownLeft),
//                     previous_distribution(18, 18, Direction::UpRight) -
//                         correction_calculation(18, 18,
//                         Direction::UpRight),
//                     tolerance);
//         ASSERT_NEAR(
//             lbm_functions.distribution_function(18, 18,
//             Direction::DownRight), previous_distribution(18, 18,
//             Direction::UpLeft) -
//                 correction_calculation(18, 18, Direction::UpLeft),
//             tolerance);
//     }
// }