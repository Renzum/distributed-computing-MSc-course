#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <fmt/format.h>
#include <mpi.h>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <mpi_layer.hpp>
#include <output_functions.hpp>

#include "main.hpp"

Arguments get_cmd_args(int argc, char *argv[]) {

    Arguments args{};

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--width", 7) == 0) {
            args.domain_width = std::stoi(argv[++i]);
        }

        if (strncmp(argv[i], "--height", 8) == 0) {
            args.domain_height = std::stoi(argv[++i]);
        }

        if (strncmp(argv[i], "--iterations", 12) == 0) {
            args.iterations = std::stoi(argv[++i]);
        }

        if (strncmp(argv[i], "--print-final", 13) == 0) {
            args.print_final = true;
        }
    }

    if (args.domain_width == 0 || args.domain_height == 0 ||
        args.iterations == 0) {
        std::cerr << "Please provide the --width --height and --iterations for "
                     "the simulation."
                  << std::endl;

        std::exit(1);
    }

    return args;
}

int main(int argc, char *argv[]) {
    auto args = get_cmd_args(argc, argv);

    const int domain_width = args.domain_width;
    const int domain_height = args.domain_height;
    const int iterations = args.iterations;

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    {
        MPILayer mpi_layer(domain_width, domain_height);

        {
            std::cout << fmt::format("{:d}: Using GPU with Device ID {:d}",
                                     mpi_layer.mpi_data->rank,
                                     Kokkos::device_id())
                      << std::endl;

            std::cout << fmt::format(
                             "I'm node with rank {:d}, coordinates [{:d}, "
                             "{:d}] and dimensions {:d}x{:d}",
                             mpi_layer.mpi_data->rank,
                             mpi_layer.mpi_data->coords[0],
                             mpi_layer.mpi_data->coords[1],
                             mpi_layer.local_lattice_height,
                             mpi_layer.local_lattice_width)
                      << std::endl;

            const double omega = 1.2;

            const double lid_vel_x = 0.1, lid_vel_y = 0.0;

            LatticeBoltzmann::Walls walls{
                // If there is no right neighbor, the right wall is a
                // bounce-back
                (mpi_layer.ghost_layers.right != 0)
                    ? LatticeBoltzmann::Wall{ 
                          LatticeBoltzmann::BoundaryType::Periodic, }
                    : LatticeBoltzmann::Wall{ 
                          LatticeBoltzmann::BoundaryType::BounceBack, 0, 0, },
                // If there is no down neighbor, the bottom wall is a
                // bounce-back
                (mpi_layer.ghost_layers.down != 0) ? LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic,}
                                             : LatticeBoltzmann::Wall{ LatticeBoltzmann::BoundaryType::BounceBack, 0, 0 ,},
                // If there is no left neighbor, the left wall is a
                // bounce-back (otherwise periodic)
                (mpi_layer.ghost_layers.left != 0) ? LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic}
                                             : LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::BounceBack, 0, 0 ,},
                // If there is no up neighbor, the top wall is a
                // moving lid (otherwise periodic)
                (mpi_layer.ghost_layers.up != 0)
                    ? LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic}
                    : LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::BounceBack, lid_vel_x, lid_vel_y ,},
            };

            LatticeBoltzmann::DistributionFunction distribution_function(
                "Distribution Function", mpi_layer.local_lattice_width,
                mpi_layer.local_lattice_height);

            LatticeBoltzmann::DistributionInitializers::random_density(
                distribution_function);
            LatticeBoltzmann::DistributionFunction buffer_distribution(
                "Buffer Distribution Function", mpi_layer.local_lattice_width,
                mpi_layer.local_lattice_height);
            LatticeBoltzmann::DensityFunction density_function(
                "Distribution Function", mpi_layer.local_lattice_width,
                mpi_layer.local_lattice_height);
            LatticeBoltzmann::VelocityProfile velocity_profile(
                "Distribution Function", mpi_layer.local_lattice_width,
                mpi_layer.local_lattice_height);

            std::cout << "starting iterations" << std::endl;

            auto timer = Kokkos::Timer();

            for (int i = 0; i < iterations; i++) {
                LatticeBoltzmann::calculate_density(density_function,
                                                    distribution_function);
                LatticeBoltzmann::calculate_local_average_velocity(
                    velocity_profile, distribution_function, density_function);
                LatticeBoltzmann::calculate_equilibrium_distribution(
                    buffer_distribution, density_function, velocity_profile);
                LatticeBoltzmann::relax_distribution(
                    distribution_function, buffer_distribution, omega);

                Kokkos::fence();

                mpi_layer.communicate(distribution_function);

                LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(
                    buffer_distribution, distribution_function,
                    density_function, walls);
            }

            Kokkos::fence();

            auto elapsed = timer.seconds();
            auto mlups = (static_cast<double>(domain_width) * domain_height *
                          iterations) /
                         (elapsed * 1000000);

            std::cout << fmt::format("{:d}: {:d}x{:d} - {:d} iterations took "
                                     "{:f} seconds = {:f} MLUPS",
                                     mpi_layer.mpi_data->rank, domain_width,
                                     domain_height, iterations, elapsed, mlups)
                      << std::endl;

            if (args.print_final) {
                auto velocity_subview = Kokkos::subview(
                    velocity_profile,
                    std::make_pair(mpi_layer.ghost_layers.left,
                                   mpi_layer.local_lattice_width -
                                       mpi_layer.ghost_layers.right),
                    std::make_pair(mpi_layer.ghost_layers.down,
                                   mpi_layer.local_lattice_height -
                                       mpi_layer.ghost_layers.up),
                    Kokkos::ALL);

                auto actual_velocity_host_mirror =
                    Kokkos::create_mirror_view(velocity_subview);

                std::cout << fmt::format("Subview dimensions [{:d}, {:d}]",
                                         velocity_subview.extent_int(0),
                                         velocity_subview.extent_int(1))
                          << std::endl;

                Kokkos::deep_copy(actual_velocity_host_mirror,
                                  velocity_subview);

                // Print
            }
        }
    }

    Kokkos::finalize();
    MPI_Finalize();
}
