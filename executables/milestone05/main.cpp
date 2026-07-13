#include <iostream>

#include <Kokkos_Core.hpp>

#include <fmt/format.h>

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
    const int lattice_size = 100;

    // Sliding Lid wall configuration
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::BoundaryType::BounceBack,
        }, // Right
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::BoundaryType::BounceBack,
        }, // Bottom
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::BoundaryType::BounceBack,
        }, // Left
        LatticeBoltzmann::Wall{
            LatticeBoltzmann::BoundaryType::BounceBack,
            0,
            0.1,
            0,
        }, // Top
    };

    constexpr double omega = 1.7;

    auto lbm_functions =
        LatticeBoltzmann::Functions(lattice_size, lattice_size);

    LatticeBoltzmann::HostLocalAverageVelocityMirror previous_velocity_profile =
        Kokkos::create_mirror_view(lbm_functions.local_average_velocity);

    LatticeBoltzmann::DistributionInitializers::uniform_at_rest(lbm_functions);

    auto velocity_profile_output = LocalAverageVelocityFunctionOutput(
        "ms5_velocity_profile.yaml", lattice_size, lattice_size, walls);

    Kokkos::Timer timer;

    const int iterations = 200;
    for (int i = 0; i < iterations; i++) {
        if (i % 20 == 0) {
            std::cout << "Heartbeat: Iteration " << i << std::endl;
            Kokkos::deep_copy(lbm_functions.host_local_average_velocity,
                              lbm_functions.local_average_velocity);

            velocity_profile_output.add_timestep(
                lbm_functions.host_local_average_velocity, i);
        }

        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions,
                                                             walls);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega, walls);

        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(lbm_functions,
                                                                  walls);

        LatticeBoltzmann::calculate_density(lbm_functions, walls);

        Kokkos::deep_copy(previous_velocity_profile,
                          lbm_functions.local_average_velocity);

        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions,
                                                           walls);
    }

    Kokkos::fence();
    double runtime = timer.seconds();

    // Actual width excludes the ghost nodes since none of the steps iterate
    // over them
    int actual_width =
        lattice_size - walls.left.ghost_layers - walls.right.ghost_layers;
    int actual_height =
        lattice_size - walls.bottom.ghost_layers - walls.top.ghost_layers;

    // MLUPS = (N_x + N_y + time_steps) / runtime * 1000000
    double mlups =
        (actual_width * actual_height * iterations) / (runtime * 1000000);

    const double viscosity = (1.0 / 3.0) * ((1.0 / omega) - 0.5);
    std::cout << "Visc:" << viscosity << std::endl;

    std::cout << fmt::format(
                     "Lattice Dimensions (excluding ghost layers): {:d}x{:d}\n",
                     actual_width, actual_height)
              << "Total Iterations Completed: " << iterations << "\n"
              << fmt::format("Total Simulation Runtime: {:f} seconds\n",
                             runtime)
              << fmt::format("MLUPS = (Width x Height x Total Iteration) / "
                             "(Runtime Seconds * 1000000) = {:f}\n",
                             mlups)
              << fmt::format("Reynolds Number = {:f}",
                             walls.top.vel_x * actual_width / viscosity)
              << std::endl;

    auto final_velocity_profile_output = LocalAverageVelocityFunctionOutput(
        "ms5_final_50000_iter.yaml", lattice_size, lattice_size, walls);

    Kokkos::deep_copy(lbm_functions.host_local_average_velocity,
                      lbm_functions.local_average_velocity);

    final_velocity_profile_output.add_timestep(
        lbm_functions.host_local_average_velocity, iterations);
}