#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

#define GRID_WIDTH 6
#define GRID_HEIGHT 6

#define GENERATIONS 10

void streaming() {
    DistributionFunctionOutput distribution_output{
        "milestone02_random_density_streaming_only-distribution.csv"};

    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", GRID_WIDTH, GRID_HEIGHT);
    LatticeBoltzmann::DistributionFunction buffer_distribution_function(
        "Buffer Distribution Function", GRID_WIDTH, GRID_HEIGHT);

    LatticeBoltzmann::DistributionFunction::HostMirror
        host_distribution_mirror =
            Kokkos::create_mirror_view(distribution_function);

    LatticeBoltzmann::DistributionInitializers::random_density(
        distribution_function);

    for (int i = 0; i < GENERATIONS; i++) {
        Kokkos::deep_copy(host_distribution_mirror, distribution_function);

        distribution_output.output(host_distribution_mirror, i);
        LatticeBoltzmann::streaming_step_with_periodic_bounds(
            buffer_distribution_function, distribution_function);
    }
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    streaming();

    Kokkos::finalize();
}
