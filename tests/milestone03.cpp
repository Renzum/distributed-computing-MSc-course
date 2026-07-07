#include <gtest/gtest.h>

#include <limits>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

TEST(MILESTONE03, DENSITY_CALCULATION) {
    const int grid_width = 6;
    const int grid_height = 6;

    auto lbm_functions = LatticeBoltzmann::Functions(grid_width, grid_height);

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    LatticeBoltzmann::calculate_density(lbm_functions);

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {

            double expected_density = 0;
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                expected_density +=
                    lbm_functions.distribution_function(x, y, dir);
            }

            ASSERT_DOUBLE_EQ(lbm_functions.density_function(x, y),
                             expected_density)
                << "Density of each cell is the sum of all values.";
        }
    }
}

TEST(MILESTONE03, MASS_CONSERVATION) {
    const int grid_width = 10;
    const int grid_height = 10;

    const double omega = 0.5;

    auto lbm_functions = LatticeBoltzmann::Functions(grid_width, grid_height);

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    auto calculate_total_mass = [&lbm_functions, &grid_width,
                                 &grid_height]() -> double {
        double total_mass = 0;

        for (int x = 0; x < grid_width; x++) {
            for (int y = 0; y < grid_height; y++) {
                for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                    total_mass +=
                        lbm_functions.distribution_function(x, y, dir);
                }
            }
        }

        return total_mass;
    };

    double expected_total_mass = calculate_total_mass();

    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        LatticeBoltzmann::calculate_density(lbm_functions);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega);
        LatticeBoltzmann::streaming_step_with_periodic_bounds(lbm_functions);

        EXPECT_NEAR(calculate_total_mass(), expected_total_mass,
                    std::numeric_limits<double>::epsilon() *
                        expected_total_mass * 100)
            << "Total mass must be equal at every iteration.";
    }
}

TEST(MILESTONE03, MOMENTUM_CONSERVATION) {
    const int grid_width = 10;
    const int grid_height = 10;

    const double omega = 0.5;

    auto lbm_functions = LatticeBoltzmann::Functions(grid_width, grid_height);

    int velocity_vector_x, velocity_vector_y;

    auto check_local_momentum = [&lbm_functions, &velocity_vector_x,
                                 &velocity_vector_y]() {
        for (int x = 0; x < grid_width; x++) {
            for (int y = 0; y < grid_height; y++) {
                const long double expected_x =
                    lbm_functions.density_function(x, y) *
                    lbm_functions.local_average_velocity(x, y, 0);
                const long double expected_y =
                    lbm_functions.density_function(x, y) *
                    lbm_functions.local_average_velocity(x, y, 1);

                long double sum_x = 0;
                long double sum_y = 0;
                for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                    get_velocity_vector(static_cast<Direction>(dir),
                                        velocity_vector_x, velocity_vector_y);

                    sum_x += lbm_functions.distribution_function(x, y, dir) *
                             velocity_vector_x;
                    sum_y += lbm_functions.distribution_function(x, y, dir) *
                             velocity_vector_y;
                }

                EXPECT_NEAR(expected_x, sum_x, 1e-10 * std::abs(expected_x))
                    << "X dimension momentum must be conserved.";
                EXPECT_NEAR(expected_y, sum_y, 1e-10 * std::abs(expected_y))
                    << "Y dimension momentum must be conserved.";
            }
        }
    };

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        LatticeBoltzmann::calculate_density(
            lbm_functions.density_function,
            lbm_functions.distribution_function);
        LatticeBoltzmann::calculate_local_average_velocity(
            lbm_functions.local_average_velocity,
            lbm_functions.distribution_function,
            lbm_functions.density_function);
        LatticeBoltzmann::calculate_equilibrium_distribution(
            lbm_functions.buffer_distribution_function,
            lbm_functions.density_function,
            lbm_functions.local_average_velocity);

        check_local_momentum();
        LatticeBoltzmann::relax_distribution(
            lbm_functions.distribution_function,
            lbm_functions.buffer_distribution_function, omega);
        check_local_momentum();

        LatticeBoltzmann::streaming_step_with_periodic_bounds(
            lbm_functions.buffer_distribution_function,
            lbm_functions.distribution_function);
    }
}

TEST(MILESTONE03, FIXED_POINT) {
    const int grid_width = 10;
    const int grid_height = 10;

    const double omega = 0.5;

    auto lbm_functions = LatticeBoltzmann::Functions{grid_width, grid_height};

    // Populate the density with a uniform value of 1.0
    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_width; y++) {
            lbm_functions.density_function(x, y) = 1.0;
        }
    }

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_width; y++) {
            lbm_functions.local_average_velocity(x, y, 0) = 0.1;
            lbm_functions.local_average_velocity(x, y, 1) = 0.1;
        }
    }

    LatticeBoltzmann::calculate_equilibrium_distribution(
        lbm_functions.buffer_distribution_function,
        lbm_functions.density_function, lbm_functions.local_average_velocity);

    // We copy the equilibrium distribution to the distribution function
    // This gives us f = f_eq
    Kokkos::deep_copy(lbm_functions.distribution_function,
                      lbm_functions.buffer_distribution_function);

    auto previous_distribution_function =
        Kokkos::View<double ***>("Previous Distribution Function", grid_width,
                                 grid_height, TOTAL_DIRECTIONS);

    const int iterations = 10;
    for (int i = 0; i < iterations; i++) {
        Kokkos::deep_copy(previous_distribution_function,
                          lbm_functions.distribution_function);

        LatticeBoltzmann::calculate_density(lbm_functions);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions);

        LatticeBoltzmann::relax_distribution(lbm_functions, omega);
        LatticeBoltzmann::streaming_step_with_periodic_bounds(lbm_functions);

        for (int direction = 0; direction < TOTAL_DIRECTIONS; direction++) {
            for (int x = 0; x < grid_width; x++) {
                for (int y = 0; y < grid_height; y++) {
                    const double prev_value =
                        previous_distribution_function(x, y, direction);
                    const double current_value =
                        lbm_functions.distribution_function(x, y, direction);

                    // Assert that the previous distribution value is the same
                    // as the new one after the streaming step
                    ASSERT_NEAR(prev_value, current_value,
                                std::numeric_limits<double>::epsilon() *
                                    prev_value);
                }
            }
        }
    }
}

TEST(MILESTONE03, BUMP_TO_UNIFORM) {
    const int grid_width = 20;
    const int grid_height = 20;

    auto lbm_functions = LatticeBoltzmann::Functions{grid_width, grid_height};

    // Fill the distribution with a uniform value 1.0 unless the location is in
    // the middle 1/3 of the lattice, in which case we fill the cells with 1.1
    // to achieve a slightly higher density in the center
    LatticeBoltzmann::DistributionInitializers::
        uniform_density_with_higher_center(lbm_functions, 1.0, 1.1);

    const double omega = 0.5;

    // Perform 1000 Lattice Boltzmann steps to make sure the density has enough
    // time to reach the uniform distribution
    const int iteration_limit = 1000;
    for (int i = 0; i < iteration_limit; i++) {
        LatticeBoltzmann::calculate_density(lbm_functions);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega);
        LatticeBoltzmann::streaming_step_with_periodic_bounds(lbm_functions);
    }

    // Capture the x = 0 y = 0 density value as a reference point
    // and compare all of the others with it using an epsilon based tolerance
    // which is scaled with the reference point value
    const double reference_point = lbm_functions.density_function(0, 0);
    const double tolerance =
        std::numeric_limits<double>::epsilon() * std::abs(reference_point);

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {
            ASSERT_NEAR(reference_point, lbm_functions.density_function(x, y),
                        tolerance);
        }
    }
}