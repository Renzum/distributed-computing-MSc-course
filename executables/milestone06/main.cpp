#include <iostream>
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

#define GLOBAL_DOMAIN_SIZE 100

void get_domain_dimensions(int argc, char *argv[], int &domain_width,
                           int &domain_height) {
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--width", 7) == 0) {
            domain_width = atoi(argv[++i]);
        }

        if (strncmp(argv[i], "--height", 8) == 0) {
            domain_height = atoi(argv[++i]);
        }
    }
}

int main(int argc, char *argv[]) {
    int domain_width = 0, domain_height = 0;
    get_domain_dimensions(argc, argv, domain_width, domain_height);

    if (domain_width == 0 || domain_height == 0) {
        std::cout << "Please provide the --width and --height of the domain "
                     "via command line arguments."
                  << std::endl;
        std::exit(1);
    }

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    {
        MPILayer mpi_communicator(domain_width, domain_height);

        std::cout << fmt::format(
                         "I'm node with rank {:d} and dimensions {:d}x{:d}",
                         mpi_communicator.mpi_data->rank,
                         mpi_communicator.lattice_width,
                         mpi_communicator.lattice_height)
                  << std::endl;

        const double omega = 1.2;
        const int iterations = 10;

        const double lid_vel_x = 0.1, lid_vel_y = 0.0;

        LatticeBoltzmann::Walls walls{
                // If there is no right neighbor, the right wall is a
                // bounce-back
                (mpi_communicator.ghost_layers.right != 0)
                    ? LatticeBoltzmann::Wall{ 
                          LatticeBoltzmann::BoundaryType::Periodic, }
                    : LatticeBoltzmann::Wall{ 
                          LatticeBoltzmann::BoundaryType::BounceBack, 0, 0, },
                // If there is no down neighbor, the bottom wall is a
                // bounce-back
                (mpi_communicator.ghost_layers.down != 0) ? LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic,}
                                             : LatticeBoltzmann::Wall{ LatticeBoltzmann::BoundaryType::BounceBack, 0, 0 ,},
                // If there is no left neighbor, the left wall is a
                // bounce-back (otherwise periodic)
                (mpi_communicator.ghost_layers.left != 0) ? LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic}
                                             : LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::BounceBack, 0, 0 ,},
                // If there is no up neighbor, the top wall is a
                // moving lid (otherwise periodic)
                (mpi_communicator.ghost_layers.up != 0)
                    ? LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic}
                    : LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::BounceBack, lid_vel_x, lid_vel_y ,},
            };

        LatticeBoltzmann::DistributionFunction distribution_function(
            "Distribution Function", mpi_communicator.lattice_width,
            mpi_communicator.lattice_height);

        LatticeBoltzmann::DistributionInitializers::random_density(
            distribution_function);
        LatticeBoltzmann::DistributionFunction buffer_distribution(
            "Buffer Distribution Function", mpi_communicator.lattice_width,
            mpi_communicator.lattice_height);
        LatticeBoltzmann::DensityFunction density_function(
            "Distribution Function", mpi_communicator.lattice_width,
            mpi_communicator.lattice_height);
        LatticeBoltzmann::VelocityProfile velocity_profile(
            "Distribution Function", mpi_communicator.lattice_width,
            mpi_communicator.lattice_height);

        std::cout << "starting iterations" << std::endl;

        auto timer = Kokkos::Timer();

        for (int i = 0; i < iterations; i++) {
            LatticeBoltzmann::calculate_density(density_function,
                                                distribution_function);
            LatticeBoltzmann::calculate_local_average_velocity(
                velocity_profile, distribution_function, density_function);
            LatticeBoltzmann::calculate_equilibrium_distribution(
                buffer_distribution, density_function, velocity_profile);
            LatticeBoltzmann::relax_distribution(distribution_function,
                                                 buffer_distribution, omega);

            Kokkos::fence();

            mpi_communicator.communicate(distribution_function);

            LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(
                buffer_distribution, distribution_function, density_function,
                walls);
        }

        Kokkos::fence();

        auto elapsed = timer.seconds();
        auto mlups =
            (static_cast<double>(domain_width) * domain_height * iterations) /
            (elapsed * 1000000);

        std::cout << fmt::format("{:d}: {:d}x{:d} - {:d} iterations took "
                                 "{:f} seconds = {:f} MLUPS",
                                 mpi_communicator.mpi_data->rank, domain_width,
                                 domain_height, iterations, elapsed, mlups)
                  << std::endl;
    }

    Kokkos::finalize();
    MPI_Finalize();
}
