#include <iostream>

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
    const int lattice_width = 200, lattice_height = 200;

    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::WallType::BounceBack,
        }, // Right
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::WallType::BounceBack,
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

    Kokkos::Timer timer;
    const int iterations = 1000;
    for (int i = 0; i < iterations; i++) {
        if (i % 100 == 0) {
            std::cout << "Heartbeat: Iteration " << i << std::endl;
        }

        Kokkos::deep_copy(lbm_functions.host_local_average_velocity,
                          lbm_functions.local_average_velocity);

        velocity_profile_output.add_timestep(
            lbm_functions.host_local_average_velocity, i);

        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions,
                                                             walls);
        LatticeBoltzmann::relax_distribution(lbm_functions, 1.7, walls);

        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(lbm_functions,
                                                                  walls);

        LatticeBoltzmann::calculate_density(lbm_functions, walls);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions,
                                                           walls);
    }
    Kokkos::fence();
    double runtime = timer.seconds();

    double actual_width =
        lattice_width - walls.left.ghost_layers - walls.right.ghost_layers;
    double actual_height =
        lattice_height - walls.bottom.ghost_layers - walls.top.ghost_layers;
    double mlups =
        (actual_width * actual_height * iterations) / (runtime * 1e6);

    std::cout << "MLUPS = " << mlups << std::endl;
}