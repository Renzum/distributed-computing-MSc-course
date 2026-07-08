#include <Kokkos_Core.hpp>

#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

#include "main.hpp"

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    ms5();

    Kokkos::finalize();
}

void ms5() {
    const int lattice_width = 20, lattice_height = 20;

    const LatticeBoltzmann::GhostLayers ghost_layers{};

    auto lbm_functions =
        LatticeBoltzmann::Functions(lattice_width, lattice_height);

    auto previous_distribution =
        Kokkos::create_mirror_view(lbm_functions.distribution_function);

    Kokkos::deep_copy(lbm_functions.buffer_distribution_function,
                      lbm_functions.distribution_function);

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    auto distribution_output = DistributionFunctionOutput("ms5.csv");

    const int iterations = 10;
    for (int i = 0; i < iterations; i++) {

        // LatticeBoltzmann::calculate_density(lbm_functions, ghost_layers);
        // LatticeBoltzmann::calculate_local_average_velocity(lbm_functions,
        //                                                    ghost_layers);
        // LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions,
        //                                                      ghost_layers);
        // LatticeBoltzmann::relax_distribution(lbm_functions, 0.5,
        // ghost_layers);

        Kokkos::deep_copy(previous_distribution,
                          lbm_functions.distribution_function);

        distribution_output.output(previous_distribution, i);

        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(
            lbm_functions, 0, 0, ghost_layers);

        // Kokkos::deep_copy(lbm_functions.host_distribution_function,
        //                   lbm_functions.distribution_function);

        // distribution_output.output(lbm_functions.host_distribution_function,
        //                            i + 1);
    }
}