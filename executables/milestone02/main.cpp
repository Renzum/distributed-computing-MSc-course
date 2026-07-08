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

    auto lbm_functions = LatticeBoltzmann::Functions(GRID_WIDTH, GRID_HEIGHT);

    LatticeBoltzmann::DistributionInitializers::random_density(
        lbm_functions.distribution_function);

    for (int i = 0; i < GENERATIONS; i++) {
        Kokkos::deep_copy(lbm_functions.host_distribution_function,
                          lbm_functions.distribution_function);

        distribution_output.output(lbm_functions.host_distribution_function, i);
        LatticeBoltzmann::streaming_step_with_periodic_bounds(lbm_functions);
    }
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    streaming();

    Kokkos::finalize();
}
