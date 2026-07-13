#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>

void streaming_test(
    LatticeBoltzmann::DistributionFunction &distribution_function) {
}

TEST(MILESTONE02, STREAMING_STEP) {
    const int grid_width = 20;
    const int grid_height = 20;

    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", grid_width, grid_height);
    LatticeBoltzmann::DistributionFunction buffer_distribution(
        "Buffer", grid_width, grid_height);

    LatticeBoltzmann::DistributionInitializers::random_density(
        distribution_function);

    LatticeBoltzmann::HostDistributionFunction host_mirror(
        "Host Distribution", grid_width, grid_height);

    LatticeBoltzmann::HostDistributionFunction host_buffer(
        "Host Buffer", grid_width, grid_height);

    LatticeBoltzmann::streaming_step_with_periodic_bounds(
        buffer_distribution, distribution_function);

    Kokkos::deep_copy(host_mirror, distribution_function);
    Kokkos::deep_copy(host_buffer, buffer_distribution);

    auto right = [&grid_width](int x) -> int {
        return (x == grid_width - 1) ? 0 : x + 1;
    };

    auto left = [&grid_width](int x) -> int {
        return (x == 0) ? grid_width - 1 : x - 1;
    };

    auto up = [&grid_height](int y) -> int {
        return (y == grid_height - 1) ? 0 : y + 1;
    };

    auto down = [&grid_height](int y) -> int {
        return (y == 0) ? grid_height - 1 : y - 1;
    };

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {

            // TODO: Fill the rest of the unit test
            ASSERT_EQ(host_buffer(x, y, Direction::Center),
                      host_mirror(x, y, Direction::Center));

            ASSERT_EQ(host_buffer(x, y, Direction::Right),
                      host_mirror(right(x), y, Direction::Right));

            ASSERT_EQ(host_buffer(x, y, Direction::Down),
                      host_mirror(x, down(y), Direction::Down));

            ASSERT_EQ(host_buffer(x, y, Direction::Left),
                      host_mirror(left(x), y, Direction::Left));

            ASSERT_EQ(host_buffer(x, y, Direction::Up),
                      host_mirror(x, up(y), Direction::Up));

            ASSERT_EQ(host_buffer(x, y, Direction::UpRight),
                      host_mirror(right(x), up(y), Direction::UpRight));

            ASSERT_EQ(host_buffer(x, y, Direction::UpLeft),
                      host_mirror(left(x), up(y), Direction::UpLeft));

            ASSERT_EQ(host_buffer(x, y, Direction::DownLeft),
                      host_mirror(left(x), down(y), Direction::DownLeft));

            ASSERT_EQ(host_buffer(x, y, Direction::DownRight),
                      host_mirror(right(x), down(y), Direction::DownRight));
        }
    }
}
