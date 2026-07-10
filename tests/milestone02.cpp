#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>

void streaming_test(LatticeBoltzmann::Functions &lbm_functions,
                    const LatticeBoltzmann::Walls &walls) {

    const int grid_width = lbm_functions.distribution_function.extent_int(0);
    const int grid_height = lbm_functions.distribution_function.extent_int(1);

    auto host_buffer =
        Kokkos::create_mirror_view(lbm_functions.distribution_function);

    Kokkos::deep_copy(host_buffer, lbm_functions.buffer_distribution_function);

    LatticeBoltzmann::streaming_step_with_periodic_bounds(lbm_functions, walls);

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

    for (int x = walls.left.ghost_layers;
         x < grid_width - walls.right.ghost_layers; x++) {
        for (int y = walls.bottom.ghost_layers;
             y < grid_height - walls.top.ghost_layers; y++) {

            // TODO: Fill the rest of the unit test
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
}

TEST(MILESTONE02, STREAMING_STEP_NO_GHOST_LAYERS) {
    using Wall = LatticeBoltzmann::Wall;
    auto walls = LatticeBoltzmann::Walls(Wall{}, Wall{}, Wall{}, Wall{});
    auto lbm_functions = LatticeBoltzmann::Functions(20, 20);

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    streaming_test(lbm_functions, walls);
}

TEST(MILESTONE02, STREAMING_STEP_ALL_GHOST_LAYERS) {
    using Wall = LatticeBoltzmann::Wall;
    // Initialize all walls with 1 ghost layer and (by default) wall type set to
    // Streaming
    auto walls = LatticeBoltzmann::Walls(Wall(1), Wall(1), Wall(1), Wall(1));

    auto lbm_functions = LatticeBoltzmann::Functions(20, 20);

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    streaming_test(lbm_functions, walls);
}

TEST(MILESTONE02, STREAMING_STEP_LEFT_GHOST_LAYER) {
    using Wall = LatticeBoltzmann::Wall;
    // Initialize all walls as default (no ghost layers and type set to
    // streaming) except left wall. Left wall has 1 ghost layer and type set to
    // streaming
    auto walls = LatticeBoltzmann::Walls(Wall{}, Wall{}, Wall(1), Wall{});
    auto lbm_functions = LatticeBoltzmann::Functions(20, 20);

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    streaming_test(lbm_functions, walls);
}
