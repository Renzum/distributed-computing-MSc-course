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
    const int lattice_width = 100, lattice_height = 100;

    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::WallType::BounceBack,
        }, // Right
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::WallType::BounceBack,
            0,
            -0.1,
            0,
        }, // Bottom
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::WallType::BounceBack,
        }, // Left
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::WallType::BounceBack,
            0,
            0.1,
            0,
        }, // Top
    };

    auto lbm_functions =
        LatticeBoltzmann::Functions(lattice_width, lattice_height);

    LatticeBoltzmann::DistributionInitializers::uniform_at_rest(lbm_functions);

    auto velocity_profile_output = LocalAverageVelocityFunctionOutput(
        "ms5_velocity_profile.yaml", lattice_width, lattice_height, walls);

    const int iterations = 500;
    for (int i = 0; i < iterations; i++) {

        Kokkos::deep_copy(lbm_functions.host_local_average_velocity,
                          lbm_functions.local_average_velocity);

        velocity_profile_output.add_timestep(
            lbm_functions.host_local_average_velocity, i);

        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions,
                                                             walls);
        LatticeBoltzmann::relax_distribution(lbm_functions, 0.5, walls);

        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(lbm_functions,
                                                                  walls);

        LatticeBoltzmann::calculate_density(lbm_functions, walls);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions,
                                                           walls);
    }
}