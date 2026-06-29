#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <direction_definitions.hpp>
#include <lattice_boltzmann_impl.hpp>

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

TEST(MILESTONE02, STREAMING_STEP) {
    const int grid_width = 3;
    const int grid_height = 3;

    auto buffer_distribution_view = Kokkos::View<double ***>(
        "Previous Distribution", grid_width, grid_height, TOTAL_DIRECTIONS);

    auto distribution_function = Kokkos::View<double ***>(
        "Current Distribution", grid_width, grid_height, TOTAL_DIRECTIONS);

    random_distribution(distribution_function, grid_width, grid_height);

    LBMImpl::streaming_step(buffer_distribution_view, distribution_function,
                            grid_width, grid_height);

    ASSERT_EQ(buffer_distribution_view(0, 0, Direction::Center),
              distribution_function(0, 0, Direction::Center))
        << "Values with `CENTER` velocity remain in their cells.";

    ASSERT_EQ(buffer_distribution_view(1, 1, Direction::Up),
              distribution_function(1, 2, Direction::Up))
        << "Values with `UP` velocity get moved to the cell above.";

    ASSERT_EQ(buffer_distribution_view(1, 1, Direction::Down),
              distribution_function(1, 0, Direction::Down))
        << "Values with `DOWN` velocity get moved to the cell below.";

    ASSERT_EQ(buffer_distribution_view(1, 1, Direction::Left),
              distribution_function(0, 1, Direction::Left))
        << "Values with `LEFT` velocity get moved to the cell to the left.";

    ASSERT_EQ(buffer_distribution_view(1, 1, Direction::Right),
              distribution_function(2, 1, Direction::Right))
        << "Values with `RIGHT` velocity get moved to the cell to the right.";

    ASSERT_EQ(buffer_distribution_view(1, 1, Direction::UpLeft),
              distribution_function(0, 2, Direction::UpLeft))
        << "Values with `UPLEFT` velocity get moved to the cell to the up "
           "right.";

    ASSERT_EQ(buffer_distribution_view(1, 1, Direction::UpRight),
              distribution_function(2, 2, Direction::UpRight))
        << "Values with `UPRIGHT` velocity get moved to the cell to the up "
           "right.";

    ASSERT_EQ(buffer_distribution_view(1, 1, Direction::DownLeft),
              distribution_function(0, 0, Direction::DownLeft))
        << "Values with `DOWNLEFT` velocity get moved to the cell to the "
           "bottom left.";

    ASSERT_EQ(buffer_distribution_view(1, 1, Direction::DownRight),
              distribution_function(2, 0, Direction::DownRight))
        << "Values with `DOWNLEFT` velocity get moved to the cell to the "
           "bottom right.";

    ASSERT_EQ(buffer_distribution_view(0, 1, Direction::Left),
              distribution_function(2, 1, Direction::Left))
        << "Values wrap around borders correctly.";

    ASSERT_EQ(buffer_distribution_view(0, 0, Direction::DownLeft),
              distribution_function(2, 2, Direction::DownLeft))
        << "Values wrap around borders correctly.";
}
