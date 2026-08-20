#include <gtest/gtest.h>

#include <limits>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

TEST(MILESTONE03, DENSITY_CALCULATION) {
    const int grid_width = 20;
    const int grid_height = 20;

    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", grid_width, grid_height);

    LatticeBoltzmann::DensityFunction density_function("Density Function",
                                                       grid_width, grid_height);

    LatticeBoltzmann::DistributionInitializers::random_density(
        distribution_function);

    LatticeBoltzmann::calculate_density(density_function,
                                        distribution_function);

    auto host_distribution_mirror =
        Kokkos::create_mirror_view(distribution_function);
    auto host_density_mirror = Kokkos::create_mirror_view(density_function);

    Kokkos::deep_copy(host_distribution_mirror, distribution_function);
    Kokkos::deep_copy(host_density_mirror, density_function);

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {

            double expected_density = 0;
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                expected_density += host_distribution_mirror(x, y, dir);
            }

            ASSERT_DOUBLE_EQ(host_density_mirror(x, y), expected_density)
                << "Density of each cell is the sum of all values.";
        }
    }
}

TEST(MILESTONE03, MASS_CONSERVATION) {
    const int grid_width = 10;
    const int grid_height = 10;

    const double omega = 0.5;

    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", grid_width, grid_height);

    LatticeBoltzmann::DistributionFunction buffer_distribution_function(
        "Buffer Distribution Function", grid_width, grid_height);

    LatticeBoltzmann::DensityFunction density_function("Density Function",
                                                       grid_width, grid_height);
    LatticeBoltzmann::VelocityProfile velocity_profile("Velocity Profile",
                                                       grid_width, grid_height);

    auto distribution_mirror =
        Kokkos::create_mirror_view(distribution_function);

    LatticeBoltzmann::DistributionInitializers::random_density(
        distribution_function);

    auto host_distribution_mirror =
        Kokkos::create_mirror_view(distribution_function);

    Kokkos::deep_copy(host_distribution_mirror, distribution_function);

    auto calculate_total_mass = [&host_distribution_mirror, grid_width,
                                 grid_height]() -> double {
        double total_mass = 0;

        for (int x = 0; x < grid_width; x++) {
            for (int y = 0; y < grid_height; y++) {
                for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                    total_mass += host_distribution_mirror(x, y, dir);
                }
            }
        }

        return total_mass;
    };

    double expected_total_mass = calculate_total_mass();

    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        LatticeBoltzmann::calculate_density(density_function,
                                            distribution_function);

        LatticeBoltzmann::calculate_local_average_velocity(
            velocity_profile, distribution_function, density_function);
        LatticeBoltzmann::calculate_equilibrium_distribution(
            buffer_distribution_function, density_function, velocity_profile);
        LatticeBoltzmann::relax_distribution(
            distribution_function, buffer_distribution_function, omega);
        LatticeBoltzmann::streaming_step_with_periodic_bounds(
            buffer_distribution_function, distribution_function);

        Kokkos::deep_copy(host_distribution_mirror, distribution_function);

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

    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", grid_width, grid_height);
    auto host_distribution_mirror =
        Kokkos::create_mirror_view(distribution_function);
    LatticeBoltzmann::DistributionFunction buffer_distribution_function(
        "Buffer Distribution Function", grid_width, grid_height);
    LatticeBoltzmann::DensityFunction density_function("Density Function",
                                                       grid_width, grid_height);
    auto host_density_function_mirror =
        Kokkos::create_mirror_view(density_function);
    LatticeBoltzmann::VelocityProfile velocity_profile("Velocity Profile",
                                                       grid_width, grid_height);
    auto host_velocity_profile_mirror =
        Kokkos::create_mirror_view(velocity_profile);

    int velocity_vector_x, velocity_vector_y;

    auto check_local_momentum = [&host_distribution_mirror,
                                 &host_density_function_mirror,
                                 &host_velocity_profile_mirror,
                                 &velocity_vector_x, &velocity_vector_y]() {
        for (int x = 0; x < grid_width; x++) {
            for (int y = 0; y < grid_height; y++) {
                const long double expected_x =
                    host_density_function_mirror(x, y) *
                    host_velocity_profile_mirror(x, y, 0);
                const long double expected_y =
                    host_density_function_mirror(x, y) *
                    host_velocity_profile_mirror(x, y, 1);

                long double sum_x = 0;
                long double sum_y = 0;
                for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                    get_velocity_vector(static_cast<Direction>(dir),
                                        velocity_vector_x, velocity_vector_y);

                    sum_x +=
                        host_distribution_mirror(x, y, dir) * velocity_vector_x;
                    sum_y +=
                        host_distribution_mirror(x, y, dir) * velocity_vector_y;
                }

                EXPECT_NEAR(expected_x, sum_x, 1e-10 * std::abs(expected_x))
                    << "X dimension momentum must be conserved.";
                EXPECT_NEAR(expected_y, sum_y, 1e-10 * std::abs(expected_y))
                    << "Y dimension momentum must be conserved.";
            }
        }
    };

    LatticeBoltzmann::DistributionInitializers::random_density(
        distribution_function);

    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic},
        LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic},
        LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic},
        LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic},
    };

    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        LatticeBoltzmann::calculate_density(density_function,
                                            distribution_function);
        LatticeBoltzmann::calculate_local_average_velocity(
            velocity_profile, distribution_function, density_function);

        LatticeBoltzmann::calculate_equilibrium_distribution(
            buffer_distribution_function, density_function, velocity_profile);

        Kokkos::deep_copy(host_distribution_mirror, distribution_function);
        Kokkos::deep_copy(host_density_function_mirror, density_function);
        Kokkos::deep_copy(host_velocity_profile_mirror, velocity_profile);

        check_local_momentum();

        LatticeBoltzmann::relax_distribution(
            distribution_function, buffer_distribution_function, omega);

        Kokkos::deep_copy(host_density_function_mirror, density_function);
        Kokkos::deep_copy(host_velocity_profile_mirror, velocity_profile);

        check_local_momentum();

        LatticeBoltzmann::streaming_step_with_periodic_bounds(
            buffer_distribution_function, distribution_function);
    }
}

TEST(MILESTONE03, FIXED_POINT) {
    const int grid_width = 10;
    const int grid_height = 10;

    const double omega = 0.5;

    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", grid_width, grid_height);
    auto host_distribution_mirror =
        Kokkos::create_mirror_view(distribution_function);
    LatticeBoltzmann::DistributionFunction buffer_distribution_function(
        "Buffer Distribution Function", grid_width, grid_height);
    LatticeBoltzmann::DensityFunction density_function("Density Function",
                                                       grid_width, grid_height);
    auto host_density_function_mirror =
        Kokkos::create_mirror_view(density_function);
    LatticeBoltzmann::VelocityProfile velocity_profile("Velocity Profile",
                                                       grid_width, grid_height);
    auto host_velocity_profile_mirror =
        Kokkos::create_mirror_view(velocity_profile);

    // Populate the density with a uniform value of 1.0
    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_width; y++) {
            host_density_function_mirror(x, y) = 1.0;
        }
    }

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_width; y++) {
            host_velocity_profile_mirror(x, y, 0) = 0.1;
            host_velocity_profile_mirror(x, y, 1) = 0.1;
        }
    }

    Kokkos::deep_copy(density_function, host_density_function_mirror);
    Kokkos::deep_copy(velocity_profile, host_velocity_profile_mirror);

    LatticeBoltzmann::calculate_equilibrium_distribution(
        buffer_distribution_function, density_function, velocity_profile);

    // We copy the equilibrium distribution to the distribution function
    // This gives us f = f_eq
    Kokkos::deep_copy(distribution_function, buffer_distribution_function);

    auto previous_distribution_function =
        Kokkos::create_mirror_view(distribution_function);

    const int iterations = 10;
    for (int i = 0; i < iterations; i++) {
        Kokkos::deep_copy(previous_distribution_function,
                          distribution_function);

        LatticeBoltzmann::calculate_density(density_function,
                                            distribution_function);
        LatticeBoltzmann::calculate_local_average_velocity(
            velocity_profile, distribution_function, density_function);
        LatticeBoltzmann::calculate_equilibrium_distribution(
            buffer_distribution_function, density_function, velocity_profile);

        LatticeBoltzmann::relax_distribution(
            distribution_function, buffer_distribution_function, omega);
        LatticeBoltzmann::streaming_step_with_periodic_bounds(
            buffer_distribution_function, distribution_function);

        Kokkos::deep_copy(host_distribution_mirror, distribution_function);

        for (int direction = 0; direction < TOTAL_DIRECTIONS; direction++) {
            for (int x = 0; x < grid_width; x++) {
                for (int y = 0; y < grid_height; y++) {
                    const double prev_value =
                        previous_distribution_function(x, y, direction);
                    const double current_value =
                        host_distribution_mirror(x, y, direction);

                    // Assert that the previous distribution value is the
                    // same as the new one after the streaming step
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

    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", grid_width, grid_height);
    auto host_distribution_mirror =
        Kokkos::create_mirror_view(distribution_function);
    LatticeBoltzmann::DistributionFunction buffer_distribution_function(
        "Buffer Distribution Function", grid_width, grid_height);
    LatticeBoltzmann::DensityFunction density_function("Density Function",
                                                       grid_width, grid_height);
    auto host_density_function_mirror =
        Kokkos::create_mirror_view(density_function);
    LatticeBoltzmann::VelocityProfile velocity_profile("Velocity Profile",
                                                       grid_width, grid_height);
    auto host_velocity_profile_mirror =
        Kokkos::create_mirror_view(velocity_profile);

    // Fill the distribution with a uniform value 1.0 unless the location is
    // in the middle 1/3 of the lattice, in which case we fill the cells
    // with 1.1 to achieve a slightly higher density in the center
    LatticeBoltzmann::DistributionInitializers::
        uniform_density_with_higher_center(distribution_function, 1.0, 1.1);

    const double omega = 0.5;

    // Perform 1000 Lattice Boltzmann steps to make sure the density has
    // enough time to reach the uniform distribution
    const int iteration_limit = 1000;
    for (int i = 0; i < iteration_limit; i++) {
        LatticeBoltzmann::calculate_density(density_function,
                                            distribution_function);
        LatticeBoltzmann::calculate_local_average_velocity(
            velocity_profile, distribution_function, density_function);
        LatticeBoltzmann::calculate_equilibrium_distribution(
            buffer_distribution_function, density_function, velocity_profile);
        LatticeBoltzmann::relax_distribution(
            distribution_function, buffer_distribution_function, omega);
        LatticeBoltzmann::streaming_step_with_periodic_bounds(
            buffer_distribution_function, distribution_function);
    }

    // Capture the x = 0 y = 0 density value as a reference point
    // and compare all of the others with it using an epsilon based
    // tolerance which is scaled with the reference point value
    Kokkos::deep_copy(host_density_function_mirror, density_function);

    const double reference_point = host_density_function_mirror(0, 0);
    const double tolerance =
        std::numeric_limits<double>::epsilon() * std::abs(reference_point);

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {
            ASSERT_NEAR(reference_point, host_density_function_mirror(x, y),
                        tolerance);
        }
    }
}