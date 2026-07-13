#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <distribution_initializers.hpp>
#include <lattice_boltzmann_types.hpp>

TEST(KOKKOS_BEHAVIOR_TEST, CONTIGIOUS_MEMORY) {
    LatticeBoltzmann::DistributionFunction distribution_function(
        "Dist Function", 20, 20);

    LatticeBoltzmann::DistributionInitializers::random_density(
        distribution_function);

    auto host_mirror = Kokkos::create_mirror_view(distribution_function);

    ASSERT_TRUE(host_mirror.span_is_contiguous());
}