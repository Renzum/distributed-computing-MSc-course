#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <density_calculation.hpp>
#include <direction_definitions.hpp>
#include <streaming_step.hpp>

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

TEST(MILESTONE03, STREAMING_STEP) {
    const int grid_width = 3;
    const int grid_height = 3;

    auto previous_distribution = Kokkos::View<double ***>(
        "Previous Distribution", grid_width, grid_height, TOTAL_DIRECTIONS);

    auto current_distribution = Kokkos::View<double ***>(
        "Current Distribution", grid_width, grid_height, TOTAL_DIRECTIONS);

    random_distribution(current_distribution, grid_width, grid_height);

    streaming_step(previous_distribution, current_distribution, grid_width,
                   grid_height);

    ASSERT_EQ(previous_distribution(0, 0, Direction::Center),
              current_distribution(0, 0, Direction::Center))
        << "Values with `CENTER` velocity remain in their cells.";

    ASSERT_EQ(previous_distribution(1, 1, Direction::Up),
              current_distribution(1, 2, Direction::Up))
        << "Values with `UP` velocity get moved to the cell above.";

    ASSERT_EQ(previous_distribution(1, 1, Direction::Down),
              current_distribution(1, 0, Direction::Down))
        << "Values with `DOWN` velocity get moved to the cell below.";

    ASSERT_EQ(previous_distribution(1, 1, Direction::Left),
              current_distribution(0, 1, Direction::Left))
        << "Values with `LEFT` velocity get moved to the cell to the left.";

    ASSERT_EQ(previous_distribution(1, 1, Direction::Right),
              current_distribution(2, 1, Direction::Right))
        << "Values with `RIGHT` velocity get moved to the cell to the right.";

    ASSERT_EQ(previous_distribution(1, 1, Direction::UpLeft),
              current_distribution(0, 2, Direction::UpLeft))
        << "Values with `UPLEFT` velocity get moved to the cell to the up "
           "right.";

    ASSERT_EQ(previous_distribution(1, 1, Direction::UpRight),
              current_distribution(2, 2, Direction::UpRight))
        << "Values with `UPRIGHT` velocity get moved to the cell to the up "
           "right.";

    ASSERT_EQ(previous_distribution(1, 1, Direction::DownLeft),
              current_distribution(0, 0, Direction::DownLeft))
        << "Values with `DOWNLEFT` velocity get moved to the cell to the "
           "bottom left.";

    ASSERT_EQ(previous_distribution(1, 1, Direction::DownRight),
              current_distribution(2, 0, Direction::DownRight))
        << "Values with `DOWNLEFT` velocity get moved to the cell to the "
           "bottom right.";

    ASSERT_EQ(previous_distribution(0, 1, Direction::Left),
              current_distribution(2, 1, Direction::Left))
        << "Values wrap around borders correctly.";

    ASSERT_EQ(previous_distribution(0, 0, Direction::DownLeft),
              current_distribution(2, 2, Direction::DownLeft))
        << "Values wrap around borders correctly.";
}

TEST(MILESTONE03, DENSITY_CALCULATION) {
    const int grid_width = 3;
    const int grid_height = 3;

    auto current_distribution = Kokkos::View<double ***>(
        "Current Distribution", grid_width, grid_height, TOTAL_DIRECTIONS);

    auto density_function =
        Kokkos::View<double **>("Density Function", grid_width, grid_height);

    random_distribution(current_distribution, grid_width, grid_height);

    calculate_density(density_function, current_distribution, grid_width,
                      grid_height);

    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {

            double expected_density = 0;
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                expected_density += current_distribution(x, y, dir);
            }

            ASSERT_EQ(density_function(x, y), expected_density)
                << "Density of each cell is the sum of all values.";
        }
    }
}