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
        LatticeBoltzmann::Wall{0, 0},   // Right
        LatticeBoltzmann::Wall{0, 0},   // Bottom
        LatticeBoltzmann::Wall{0, 0},   // Left
        LatticeBoltzmann::Wall{0.1, 0}, // Top
    };

    constexpr double omega = 1.7;

    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", lattice_size, lattice_size);
    LatticeBoltzmann::DistributionFunction buffer_distribution_function(
        "Buffer Distribution Function", lattice_size, lattice_size);

    LatticeBoltzmann::DensityFunction density_function(
        "Density Function", lattice_size, lattice_size);

    LatticeBoltzmann::VelocityProfile velocity_profile(
        "Velocity Profile", lattice_size, lattice_size);

    LatticeBoltzmann::VelocityProfile::HostMirror host_velocity_profile =
        Kokkos::create_mirror_view(velocity_profile);

    Kokkos::deep_copy(host_velocity_profile, velocity_profile);

    LatticeBoltzmann::DistributionInitializers::uniform_at_rest(
        density_function, velocity_profile);

    auto velocity_profile_output = LocalAverageVelocityFunctionOutput(
        "ms5_velocity_profile.yaml", lattice_size, lattice_size, walls);

    Kokkos::Timer timer;

    const int iterations = 200;
    for (int i = 0; i < iterations; i++) {
        if (i % 20 == 0) {
            std::cout << "Heartbeat: Iteration " << i << std::endl;

            velocity_profile_output.add_timestep(host_velocity_profile, i);
        }

        LatticeBoltzmann::calculate_equilibrium_distribution(
            buffer_distribution_function, density_function, velocity_profile);
        LatticeBoltzmann::relax_distribution(
            distribution_function, buffer_distribution_function, omega);

        LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(
            buffer_distribution_function, distribution_function,
            density_function, walls);

        LatticeBoltzmann::calculate_density(density_function,
                                            distribution_function);

        LatticeBoltzmann::calculate_local_average_velocity(
            velocity_profile, distribution_function, density_function);

        Kokkos::deep_copy(host_velocity_profile, velocity_profile);
    }

    Kokkos::fence();
    double runtime = timer.seconds();

    // MLUPS = (N_x + N_y + time_steps) / runtime * 1000000
    double mlups =
        (lattice_size * lattice_size * iterations) / (runtime * 1000000);

    const double viscosity = (1.0 / 3.0) * ((1.0 / omega) - 0.5);
    std::cout << "Visc:" << viscosity << std::endl;

    std::cout << fmt::format(
                     "Lattice Dimensions (excluding ghost layers): {:d}x{:d}\n",
                     lattice_size, lattice_size)
              << "Total Iterations Completed: " << iterations << "\n"
              << fmt::format("Total Simulation Runtime: {:f} seconds\n",
                             runtime)
              << fmt::format("MLUPS = (Width x Height x Total Iteration) / "
                             "(Runtime Seconds * 1000000) = {:f}\n",
                             mlups)
              << fmt::format("Reynolds Number = {:f}",
                             walls.top.vel_x * lattice_size / viscosity)
              << std::endl;

    auto final_velocity_profile_output = LocalAverageVelocityFunctionOutput(
        "ms5_final_50000_iter.yaml", lattice_size, lattice_size, walls);

    final_velocity_profile_output.add_timestep(velocity_profile, iterations);
}