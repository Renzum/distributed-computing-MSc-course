#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>

#include <fmt/format.h>

#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

#include "main.hpp"

struct Arguments {
    int lattice_size = -1;
    int iterations = 1000;
    double omega = 0.8;
    std::string output_tag;
    double right_wall_y = 0.0;
    double bottom_wall_x = 0.0;
    double left_wall_y = 0.0;
    double top_wall_x = 0.0;
    bool only_final = false;
};

Arguments get_args(int argc, char *argv[]) {
    Arguments args{};

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "-s", 2) == 0) {
            args.lattice_size = std::stoi(argv[++i]);
        }

        if (strncmp(argv[i], "-i", 2) == 0) {
            args.iterations = std::stoi(argv[++i]);
        }

        if (strncmp(argv[i], "-o", 2) == 0) {
            args.omega = std::stod(argv[++i]);
        }

        if (strncmp(argv[i], "--tag", 5) == 0) {
            args.output_tag = std::string(argv[++i]);
        }

        if (strncmp(argv[i], "-r", 2) == 0) {
            args.right_wall_y = std::stod(argv[++i]);
        }

        if (strncmp(argv[i], "-b", 2) == 0) {
            args.bottom_wall_x = std::stod(argv[++i]);
        }

        if (strncmp(argv[i], "-l", 2) == 0) {
            args.left_wall_y = std::stod(argv[++i]);
        }

        if (strncmp(argv[i], "-t", 2) == 0) {
            args.top_wall_x = std::stod(argv[++i]);
        }

        if (strncmp(argv[i], "--only-final", 12) == 0) {
            args.only_final = true;
        }
    }

    if (args.lattice_size == -1) {
        std::cerr << "Please set width and height using the -s argument."
                  << std::endl;
        exit(1);
    }

    return args;
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    auto args = get_args(argc, argv);
    ms5(args);

    Kokkos::finalize();
}

void ms5(Arguments args) {
    // Sliding Lid wall configuration
    const LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall(LatticeBoltzmann::BoundaryType::BounceBack, 0,
                               args.right_wall_y), // Right
        LatticeBoltzmann::Wall(LatticeBoltzmann::BoundaryType::BounceBack,
                               args.bottom_wall_x, 0), // Bottom
        LatticeBoltzmann::Wall(LatticeBoltzmann::BoundaryType::BounceBack, 0,
                               args.left_wall_y), // Left
        LatticeBoltzmann::Wall(LatticeBoltzmann::BoundaryType::BounceBack,
                               args.top_wall_x, 0) // Top
    };

    const int lattice_size = args.lattice_size;

    const double omega = args.omega;

    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", lattice_size, lattice_size);
    LatticeBoltzmann::DistributionFunction buffer_distribution_function(
        "Buffer Distribution Function", lattice_size, lattice_size);

    LatticeBoltzmann::DensityFunction density_function(
        "Density Function", lattice_size, lattice_size);

    LatticeBoltzmann::VelocityProfile velocity_profile(
        "Velocity Profile", lattice_size, lattice_size);

    auto host_velocity_profile = Kokkos::create_mirror_view(velocity_profile);

    Kokkos::deep_copy(host_velocity_profile, velocity_profile);

    LatticeBoltzmann::DistributionInitializers::uniform_at_rest(
        density_function, velocity_profile);

    const int iterations = args.iterations;

    auto velocity_profile_output = LocalAverageVelocityFunctionOutput(
        fmt::format("ms5_velocity_profile-{:d}x{:d}-i{:d}-o{:f}-{:s}.yaml",
                    lattice_size, lattice_size, iterations, omega,
                    args.output_tag),
        lattice_size, lattice_size, walls);

    Kokkos::Timer timer;

    for (int i = 0; i < iterations; i++) {
        if (!args.only_final && i % 20 == 0) {
            std::cout << "Heartbeat: Iteration " << i << std::endl;

            Kokkos::deep_copy(host_velocity_profile, velocity_profile);
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
    }

    Kokkos::fence();
    double runtime = timer.seconds();

    // MLUPS = (N_x + N_y + time_steps) / runtime * 1000000
    double mlups =
        (static_cast<double>(lattice_size) * lattice_size * iterations) / (runtime * 1000000);

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

    Kokkos::deep_copy(host_velocity_profile, velocity_profile);

    auto final_velocity_profile_output = LocalAverageVelocityFunctionOutput(

        fmt::format(
            "ms5_final_velocity_profile-{:d}x{:d}-i{:d}-o{:f}-{:s}.yaml",
            lattice_size, lattice_size, iterations, omega, args.output_tag),
        lattice_size, lattice_size, walls);

    final_velocity_profile_output.add_timestep(host_velocity_profile,
                                               iterations);
}
