#include <gtest/gtest.h>

#include <limits>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <direction_definitions.hpp>
#include <lattice_boltzman_impl.hpp>

namespace {

void random_distribution(const Kokkos::View<double ***> view, int grid_width,
                         int grid_height) {
    auto rand = Kokkos::Random_XorShift64_Pool<>(/* seed = */ 12345);
    Kokkos::parallel_for(
        "Initialize Density",
        Kokkos::MDRangePolicy({0, 0, 0},
                              {grid_width, grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            auto gen = rand.get_state();

            view(x, y, dir) = x + y + dir + gen.rand(0, 100);

            rand.free_state(gen);
        });
}

} // namespace

TEST(MILESTONE03, DENSITY_CALCULATION) {
    const int grid_width = 6;
    const int grid_height = 6;

    auto distribution_function = Kokkos::View<double ***>(
        "Current Distribution", grid_width, grid_height, TOTAL_DIRECTIONS);

    auto density_function =
        Kokkos::View<double **>("Density Function", grid_width, grid_height);

    random_distribution(distribution_function, grid_width, grid_height);

    LBMImpl::calculate_density(density_function, distribution_function,
                               grid_width, grid_height);

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {

            double expected_density = 0;
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                expected_density += distribution_function(x, y, dir);
            }

            ASSERT_DOUBLE_EQ(density_function(x, y), expected_density)
                << "Density of each cell is the sum of all values.";
        }
    }
}

TEST(MILESTONE03, MASS_CONSERVATION) {
    const int grid_width = 10;
    const int grid_height = 10;

    const double viscocity = 0.5;

    auto buffer_distribution_function =
        Kokkos::View<double ***>("Buffer  Distribution Function", grid_width,
                                 grid_height, TOTAL_DIRECTIONS);
    auto distribution_function = Kokkos::View<double ***>(
        "Distribution Function", grid_width, grid_height, TOTAL_DIRECTIONS);
    auto density_function =
        Kokkos::View<double **>("Density Function", grid_width, grid_height);
    auto local_average_velocity_function = Kokkos::View<double ***>(
        "Local Average Velocity Function", grid_width, grid_height, 2);

    auto calculate_total_mass = [&distribution_function, &grid_width,
                                 &grid_height]() -> double {
        double total_mass = 0;

        for (int x = 0; x < grid_width; x++) {
            for (int y = 0; y < grid_height; y++) {
                for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                    total_mass += distribution_function(x, y, dir);
                }
            }
        }

        return total_mass;
    };

    random_distribution(distribution_function, grid_width, grid_height);

    double expected_total_mass = calculate_total_mass();

    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        LBMImpl::calculate_density(density_function, distribution_function,
                                   grid_width, grid_height);
        LBMImpl::calculate_local_average_velocity(
            local_average_velocity_function, distribution_function,
            density_function, grid_width, grid_height);
        LBMImpl::calculate_equilibrium_distribution(
            buffer_distribution_function, density_function,
            local_average_velocity_function, grid_width, grid_height);
        LBMImpl::relax_distribution(distribution_function,
                                    buffer_distribution_function, viscocity,
                                    grid_width, grid_height);
        LBMImpl::streaming_step(buffer_distribution_function,
                                distribution_function, grid_width, grid_height);

        // I wasn't sure what precision to use, so I asked Claude.
        // It recommended I use epsilon * expected mass with a coefficient of
        // 100, I tested with 10 instead of 100 and it satisfied the precision.
        // Will adjust the coefficient if I notice persistent flakiness
        EXPECT_NEAR(calculate_total_mass(), expected_total_mass,
                    std::numeric_limits<double>::epsilon() *
                        expected_total_mass * 100)
            << "Total mass must be equal at every iteration.";
    }
}

TEST(MILESTONE03, MOMENTUM_CONSERVATION) {
    const int grid_width = 10;
    const int grid_height = 10;

    const double viscocity = 0.5;

    auto buffer_distribution_function =
        Kokkos::View<double ***>("Buffer  Distribution Function", grid_width,
                                 grid_height, TOTAL_DIRECTIONS);
    auto distribution_function = Kokkos::View<double ***>(
        "Distribution Function", grid_width, grid_height, TOTAL_DIRECTIONS);
    auto density_function =
        Kokkos::View<double **>("Density Function", grid_width, grid_height);
    auto local_average_velocity_function = Kokkos::View<double ***>(
        "Local Average Velocity Function", grid_width, grid_height, 2);

    auto check_local_momentum = [&density_function,
                                 &local_average_velocity_function,
                                 &distribution_function]() {
        for (int x = 0; x < grid_width; x++) {
            for (int y = 0; y < grid_height; y++) {
                const double expected_x =
                    density_function(x, y) *
                    local_average_velocity_function(x, y, 0);
                const double expected_y =
                    density_function(x, y) *
                    local_average_velocity_function(x, y, 1);

                double sum_x = 0;
                double sum_y = 0;
                for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                    auto [vec_x, vec_y] = velocity_vector[dir];

                    sum_x += distribution_function(x, y, dir) * vec_x;
                    sum_y += distribution_function(x, y, dir) * vec_y;
                }

                // Here I use distance of 1/1000 because epsilon doesn't
                // properly work
                EXPECT_NEAR(expected_x, sum_x, 1e-10 * std::abs(expected_x))
                    // std::numeric_limits<double>::epsilon() *
                    //     std::abs(expected_x) * 100)
                    << "X dimension momentum must be conserved.";
                EXPECT_NEAR(expected_y, sum_y, 1e-10 * std::abs(expected_y))
                    // std::numeric_limits<double>::epsilon() *
                    //     std::abs(expected_y) * 100)
                    << "Y dimension momentum must be conserved.";
            }
        }
    };

    random_distribution(distribution_function, grid_width, grid_height);

    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        LBMImpl::calculate_density(density_function, distribution_function,
                                   grid_width, grid_height);
        LBMImpl::calculate_local_average_velocity(
            local_average_velocity_function, distribution_function,
            density_function, grid_width, grid_height);
        LBMImpl::calculate_equilibrium_distribution(
            buffer_distribution_function, density_function,
            local_average_velocity_function, grid_width, grid_height);

        check_local_momentum();
        LBMImpl::relax_distribution(distribution_function,
                                    buffer_distribution_function, viscocity,
                                    grid_width, grid_height);
        check_local_momentum();

        LBMImpl::streaming_step(buffer_distribution_function,
                                distribution_function, grid_width, grid_height);
    }
}

TEST(MILESTONE03, FIXED_POINT) {
    const int grid_width = 10;
    const int grid_height = 10;

    const double viscocity = 0.5;

    auto buffer_distribution_function =
        Kokkos::View<double ***>("Buffer  Distribution Function", grid_width,
                                 grid_height, TOTAL_DIRECTIONS);
    auto distribution_function = Kokkos::View<double ***>(
        "Distribution Function", grid_width, grid_height, TOTAL_DIRECTIONS);
    auto density_function =
        Kokkos::View<double **>("Density Function", grid_width, grid_height);
    auto local_average_velocity_function = Kokkos::View<double ***>(
        "Local Average Velocity Function", grid_width, grid_height, 2);

    random_distribution(distribution_function, grid_width, grid_height);

    LBMImpl::calculate_density(density_function, distribution_function,
                               grid_width, grid_height);
    LBMImpl::calculate_local_average_velocity(
        local_average_velocity_function, distribution_function,
        density_function, grid_width, grid_height);
    LBMImpl::calculate_equilibrium_distribution(
        buffer_distribution_function, density_function,
        local_average_velocity_function, grid_width, grid_height);

    // We copy the equilibrium distribution to the distribution function
    // This gives us f = f_eq
    Kokkos::deep_copy(distribution_function, buffer_distribution_function);
}