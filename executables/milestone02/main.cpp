#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

#define GRID_WIDTH 20
#define GRID_HEIGHT 20

#define GENERATIONS 40

void streaming() {
    DistributionFunctionOutput distribution_output{
        "milestone02_random_density_streaming_only-distribution.csv"};

    auto lbm_functions = LatticeBoltzmann::Functions(GRID_WIDTH, GRID_HEIGHT);

    for (int i = 0; i < GENERATIONS; i++) {
        distribution_output.output(lbm_functions.distribution_function, i);
        LatticeBoltzmann::streaming_step(lbm_functions);
    }
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    streaming();

    Kokkos::finalize();
}
