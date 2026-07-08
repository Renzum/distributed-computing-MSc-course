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
    const int lattice_width = 128, lattice_height = 128;
    const double lid_vel_x = 0.1, lid_vel_y = 0;

    auto local_vel_output = LocalAverageVelocityFunctionOutput(
        "ms5_local_average_velocity.yaml", lattice_width, lattice_height);

    auto lbm_functions =
        LatticeBoltzmann::Functions(lattice_width, lattice_height);

    LatticeBoltzmann::DistributionInitializers::uniform_density(lbm_functions);

    const double omega = 0.5;
    const int iterations = 600;
    for (int i = 0; i < iterations; i++) {
        LatticeBoltzmann::calculate_density(lbm_functions);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions);

        Kokkos::deep_copy(lbm_functions.host_local_average_velocity,
                          lbm_functions.local_average_velocity);

        local_vel_output.add_timestep(
            lbm_functions.host_local_average_velocity);

        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega);

        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(
            lbm_functions, lid_vel_x, lid_vel_y);
    }
}