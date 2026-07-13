#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <distribution_initializers.hpp>
#include <lattice_boltzmann_types.hpp>

TEST(KOKKOS_BEHAVIOR_TEST, CONTIGIOUS_MEMORY) {
    LatticeBoltzmann::DistributionFunction distribution_function(
        "Dist Function", 20, 20);

    LatticeBoltzmann::DistributionInitializers::random_density(
        distribution_function);

    LatticeBoltzmann::DistributionFunction::HostMirror host_mirror =
        Kokkos::create_mirror_view(distribution_function);

    ASSERT_TRUE(host_mirror.span_is_contiguous());

    Kokkos::deep_copy(host_mirror, distribution_function);

    for (int x = 0; x < 20; x++) {
        for (int y = 0; y < 20; y++) {
            ASSERT_EQ(host_mirror(x, y, Direction::Center),
                      distribution_function(x, y, Direction::Center));
        }
    }
}