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

    LatticeBoltzmann::streaming_step(lbm_functions);

    ASSERT_EQ(
        lbm_functions.buffer_distribution_function(0, 0, Direction::Center),
        lbm_functions.distribution_function(0, 0, Direction::Center))
        << "Values with `CENTER` velocity remain in their cells.";

    ASSERT_EQ(lbm_functions.buffer_distribution_function(1, 1, Direction::Up),
              lbm_functions.distribution_function(1, 2, Direction::Up))
        << "Values with `UP` velocity need to moved to the cell above.";

    ASSERT_EQ(lbm_functions.buffer_distribution_function(1, 1, Direction::Down),
              lbm_functions.distribution_function(1, 0, Direction::Down))
        << "Values with `DOWN` velocity need to moved to the cell below.";

    ASSERT_EQ(lbm_functions.buffer_distribution_function(1, 1, Direction::Left),
              lbm_functions.distribution_function(0, 1, Direction::Left))
        << "Values with `LEFT` velocity need to moved to the cell to the left.";

    ASSERT_EQ(
        lbm_functions.buffer_distribution_function(1, 1, Direction::Right),
        lbm_functions.distribution_function(2, 1, Direction::Right))
        << "Values with `RIGHT` velocity need to moved to the cell to the "
           "right.";

    ASSERT_EQ(
        lbm_functions.buffer_distribution_function(1, 1, Direction::UpLeft),
        lbm_functions.distribution_function(0, 2, Direction::UpLeft))
        << "Values with `UPLEFT` velocity need to moved to the cell to the up "
           "right.";

    ASSERT_EQ(
        lbm_functions.buffer_distribution_function(1, 1, Direction::UpRight),
        lbm_functions.distribution_function(2, 2, Direction::UpRight))
        << "Values with `UPRIGHT` velocity need to moved to the cell to the up "
           "right.";

    ASSERT_EQ(
        lbm_functions.buffer_distribution_function(1, 1, Direction::DownLeft),
        lbm_functions.distribution_function(0, 0, Direction::DownLeft))
        << "Values with `DOWNLEFT` velocity need to moved to the cell to the "
           "bottom left.";

    ASSERT_EQ(
        lbm_functions.buffer_distribution_function(1, 1, Direction::DownRight),
        lbm_functions.distribution_function(2, 0, Direction::DownRight))
        << "Values with `DOWNLEFT` velocity need to moved to the cell to the "
           "bottom right.";

    ASSERT_EQ(lbm_functions.buffer_distribution_function(0, 1, Direction::Left),
              lbm_functions.distribution_function(2, 1, Direction::Left))
        << "Values wrap around borders correctly.";

    ASSERT_EQ(
        lbm_functions.buffer_distribution_function(0, 0, Direction::DownLeft),
        lbm_functions.distribution_function(2, 2, Direction::DownLeft))
        << "Values wrap around borders correctly.";
}
