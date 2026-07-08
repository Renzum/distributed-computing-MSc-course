#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>

void streaming_test(LatticeBoltzmann::Functions &lbm_functions,
                    const LatticeBoltzmann::GhostLayers &ghost_layers = {}) {

    const int grid_width = lbm_functions.distribution_function.extent_int(0);
    const int grid_height = lbm_functions.distribution_function.extent_int(1);

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

    for (int x = ghost_layers.left; x < grid_width - ghost_layers.right; x++) {
        for (int y = ghost_layers.bottom; y < grid_height - ghost_layers.top;
             y++) {

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
    auto ghost_layers = LatticeBoltzmann::GhostLayers{};
    auto lbm_functions = LatticeBoltzmann::Functions(20, 20);

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    streaming_test(lbm_functions);
}

TEST(MILESTONE02, STREAMING_STEP_ALL_GHOST_LAYERS) {
    auto ghost_layers = LatticeBoltzmann::GhostLayers{1, 1, 1, 1};
    auto lbm_functions = LatticeBoltzmann::Functions(
        20 + ghost_layers.left + ghost_layers.right,
        20 + ghost_layers.bottom + ghost_layers.top);

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    streaming_test(lbm_functions);
}

TEST(MILESTONE02, STREAMING_STEP_LEFT_GHOST_LAYER) {
    auto ghost_layers = LatticeBoltzmann::GhostLayers{0, 0, 1, 0};
    auto lbm_functions = LatticeBoltzmann::Functions(
        20 + ghost_layers.left + ghost_layers.right,
        20 + ghost_layers.bottom + ghost_layers.top);

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    streaming_test(lbm_functions);
}
