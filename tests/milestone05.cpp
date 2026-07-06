#include <gtest/gtest.h>

#include <limits>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

TEST(MILESTONE05, FIXED_WALL_BOUNCE_BACK) {
    auto lbm_functions = LatticeBoltzmann::Functions(20, 20);
    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    // Omega doesn't matter much here, but lower is better so the system reaches
    // an equilibrium slower
    constexpr double omega = 0.5;

    // For storing the distribution before the streaming step
    Kokkos::View<double ***> previous_distribution(
        "Previous Distribution View",
        lbm_functions.distribution_function.extent_int(0),
        lbm_functions.distribution_function.extent_int(1), TOTAL_DIRECTIONS);

    // Since the lid is fixed in this test, this will be a one dimensional
    // view of size `TOTAL_DIRECTIONS` all initialized to 0
    const auto fixed_lid_velocity_correction =
        LatticeBoltzmann::calculate_wall_velocity_modifier(0, 0, 0);

    // Perform the Lattice Boltzmann step with boundary check 20 times and test
    // bounce back correctness each time
    for (int i = 0; i < 20; i++) {
        LatticeBoltzmann::calculate_density(lbm_functions);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega);

        Kokkos::deep_copy(previous_distribution,
                          lbm_functions.distribution_function);

        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(
            lbm_functions, fixed_lid_velocity_correction);

        // Bottom Left Corner
        ASSERT_EQ(lbm_functions.distribution_function(1, 1, Direction::UpLeft),
                  previous_distribution(1, 1, Direction::DownRight));
        ASSERT_EQ(lbm_functions.distribution_function(1, 1, Direction::Up),
                  previous_distribution(1, 1, Direction::Down));
        ASSERT_EQ(lbm_functions.distribution_function(1, 1, Direction::UpRight),
                  previous_distribution(1, 1, Direction::DownLeft));
        ASSERT_EQ(lbm_functions.distribution_function(1, 1, Direction::Right),
                  previous_distribution(1, 1, Direction::Left));
        ASSERT_EQ(
            lbm_functions.distribution_function(1, 1, Direction::DownRight),
            previous_distribution(1, 1, Direction::UpLeft));

        // Left Edge Node
        ASSERT_EQ(lbm_functions.distribution_function(1, 5, Direction::Right),
                  previous_distribution(1, 5, Direction::Left));
        ASSERT_EQ(lbm_functions.distribution_function(1, 5, Direction::UpRight),
                  previous_distribution(1, 5, Direction::DownLeft));
        ASSERT_EQ(
            lbm_functions.distribution_function(1, 5, Direction::DownRight),
            previous_distribution(1, 5, Direction::UpLeft));

        // Upper Left Corner
        ASSERT_EQ(lbm_functions.distribution_function(1, 18, Direction::Right),
                  previous_distribution(1, 18, Direction::Left));
        ASSERT_EQ(
            lbm_functions.distribution_function(1, 18, Direction::DownRight),
            previous_distribution(1, 18, Direction::UpLeft));
        ASSERT_EQ(lbm_functions.distribution_function(1, 18, Direction::Down),
                  previous_distribution(1, 18, Direction::Up));
        ASSERT_EQ(
            lbm_functions.distribution_function(1, 18, Direction::DownLeft),
            previous_distribution(1, 18, Direction::UpRight));

        // Top Edge Node
        ASSERT_EQ(
            lbm_functions.distribution_function(5, 18, Direction::DownRight),
            previous_distribution(5, 18, Direction::UpLeft));
        ASSERT_EQ(lbm_functions.distribution_function(5, 18, Direction::Down),
                  previous_distribution(5, 18, Direction::Up));
        ASSERT_EQ(
            lbm_functions.distribution_function(5, 18, Direction::DownLeft),
            previous_distribution(5, 18, Direction::UpRight));

        // Upper Right Corner
        ASSERT_EQ(
            lbm_functions.distribution_function(18, 18, Direction::UpLeft),
            previous_distribution(18, 18, Direction::DownRight));
        ASSERT_EQ(lbm_functions.distribution_function(18, 18, Direction::Left),
                  previous_distribution(18, 18, Direction::Right));
        ASSERT_EQ(
            lbm_functions.distribution_function(18, 18, Direction::DownLeft),
            previous_distribution(18, 18, Direction::UpRight));
        ASSERT_EQ(lbm_functions.distribution_function(18, 18, Direction::Down),
                  previous_distribution(18, 18, Direction::Up));
        ASSERT_EQ(
            lbm_functions.distribution_function(18, 18, Direction::DownRight),
            previous_distribution(18, 18, Direction::UpLeft));

        // Right Edge Node
        ASSERT_EQ(lbm_functions.distribution_function(18, 7, Direction::UpLeft),
                  previous_distribution(18, 7, Direction::DownRight));
        ASSERT_EQ(lbm_functions.distribution_function(18, 7, Direction::Left),
                  previous_distribution(18, 7, Direction::Right));
        ASSERT_EQ(
            lbm_functions.distribution_function(18, 7, Direction::DownLeft),
            previous_distribution(18, 7, Direction::UpRight));

        // Lower Right Corner
        ASSERT_EQ(lbm_functions.distribution_function(18, 1, Direction::Up),
                  previous_distribution(18, 1, Direction::Down));
        ASSERT_EQ(lbm_functions.distribution_function(18, 1, Direction::UpLeft),
                  previous_distribution(18, 1, Direction::DownRight));
        ASSERT_EQ(lbm_functions.distribution_function(18, 1, Direction::Left),
                  previous_distribution(18, 1, Direction::Right));
        ASSERT_EQ(
            lbm_functions.distribution_function(18, 1, Direction::DownLeft),
            previous_distribution(18, 1, Direction::UpRight));

        // Bottom Edge Node
        ASSERT_EQ(lbm_functions.distribution_function(4, 1, Direction::UpRight),
                  previous_distribution(4, 1, Direction::DownLeft));
        ASSERT_EQ(lbm_functions.distribution_function(4, 1, Direction::Up),
                  previous_distribution(4, 1, Direction::Down));
        ASSERT_EQ(lbm_functions.distribution_function(4, 1, Direction::UpLeft),
                  previous_distribution(4, 1, Direction::DownRight));
    }
}