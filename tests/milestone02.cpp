#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>

TEST(MILESTONE02, STREAMING_STEP) {
    const int grid_width = 3;
    const int grid_height = 3;

    auto lbm_functions = LatticeBoltzmann::Functions(grid_width, grid_height);

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    auto host_buffer =
        Kokkos::create_mirror_view(lbm_functions.distribution_function);

    Kokkos::deep_copy(host_buffer, lbm_functions.buffer_distribution_function);

    LatticeBoltzmann::streaming_step_with_periodic_bounds(lbm_functions);

    Kokkos::deep_copy(lbm_functions.host_distribution_function,
                      lbm_functions.distribution_function);

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

            ASSERT_EQ(host_buffer(x, y, Direction::Center),
                      lbm_functions.host_distribution_function(
                          x, y, Direction::Center));

            ASSERT_EQ(host_buffer(x, y, Direction::Right),
                      lbm_functions.host_distribution_function(
                          right(x), y, Direction::Right));

            ASSERT_EQ(host_buffer(x, y, Direction::Down),
                      lbm_functions.host_distribution_function(
                          x, down(y), Direction::Down));
        }
    }

    // ASSERT_EQ(host_buffer(1, 1, Direction::Up),
    //           lbm_functions.host_distribution_function(1, 2, Direction::Up))
    //     << "Values with `UP` velocity need to moved to the cell above.";

    // ASSERT_EQ(host_buffer(1, 1, Direction::Down),
    //           lbm_functions.host_distribution_function(1, 0,
    //           Direction::Down))
    //     << "Values with `DOWN` velocity need to moved to the cell below.";

    // ASSERT_EQ(host_buffer(1, 1, Direction::Left),
    //           lbm_functions.host_distribution_function(0, 1,
    //           Direction::Left))
    //     << "Values with `LEFT` velocity need to moved to the cell to the
    //     left.";

    // ASSERT_EQ(host_buffer(1, 1, Direction::Right),
    //           lbm_functions.host_distribution_function(2, 1,
    //           Direction::Right))
    //     << "Values with `RIGHT` velocity need to moved to the cell to the "
    //        "right.";

    // ASSERT_EQ(host_buffer(1, 1, Direction::UpLeft),
    //           lbm_functions.host_distribution_function(0, 2,
    //           Direction::UpLeft))
    //     << "Values with `UPLEFT` velocity need to moved to the cell to the up
    //     "
    //        "right.";

    // ASSERT_EQ(
    //     host_buffer(1, 1, Direction::UpRight),
    //     lbm_functions.host_distribution_function(2, 2, Direction::UpRight))
    //     << "Values with `UPRIGHT` velocity need to moved to the cell to the
    //     up "
    //        "right.";

    // ASSERT_EQ(
    //     host_buffer(1, 1, Direction::DownLeft),
    //     lbm_functions.host_distribution_function(0, 0, Direction::DownLeft))
    //     << "Values with `DOWNLEFT` velocity need to moved to the cell to the
    //     "
    //        "bottom left.";

    // ASSERT_EQ(
    //     host_buffer(1, 1, Direction::DownRight),
    //     lbm_functions.host_distribution_function(2, 0, Direction::DownRight))
    //     << "Values with `DOWNLEFT` velocity need to moved to the cell to the
    //     "
    //        "bottom right.";

    // ASSERT_EQ(host_buffer(0, 1, Direction::Left),
    //           lbm_functions.host_distribution_function(2, 1,
    //           Direction::Left))
    //     << "Values wrap around borders correctly.";

    // ASSERT_EQ(
    //     host_buffer(0, 0, Direction::DownLeft),
    //     lbm_functions.host_distribution_function(2, 2, Direction::DownLeft))
    //     << "Values wrap around borders correctly.";
}
