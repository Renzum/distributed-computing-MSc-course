#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <distribution_initializers.hpp>
#include <lattice_boltzmann_types.hpp>

TEST(KOKKOS_BEHAVIOR_TEST, CONTIGIOUS_MEMORY) {
    LatticeBoltzmann::DistributionFunction distribution_function(
        "Dist Function", 20, 20);

    LatticeBoltzmann::DistributionInitializers::random_density(
        distribution_function);

    Kokkos::View<double *[9], Kokkos::HostSpace> mpi_buffer("MPI Buffer", 20);
    ASSERT_TRUE(mpi_buffer.span_is_contiguous());

    auto distribution_subview =
        Kokkos::subview(distribution_function, 0, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(mpi_buffer, distribution_subview);

    for (int y = 0; y < 20; y++) {
        for (int dir = 0; dir < 9; dir++) {
            mpi_buffer(y, dir) = y;
        }
    }
}