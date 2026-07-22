#include <iostream>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <fmt/format.h>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <mpi_node.hpp>
#include <output_functions.hpp>

#define GLOBAL_DOMAIN_SIZE 32000

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    {
        Node node(GLOBAL_DOMAIN_SIZE);

        std::cout << fmt::format(
                         "I'm node with rank {:d} and dimensions {:d}x{:d}",
                         node.rank, node.lattice_width, node.lattice_height)
                  << std::endl;

        const double omega = 1.2;
        const int iterations = 200;

        const double lid_vel_x = 0.1, lid_vel_y = 0.0;

        LatticeBoltzmann::Walls walls{
                // If there is no right neighbor, the right wall is a
                // bounce-back
                (node.right != MPI_PROC_NULL)
                    ? LatticeBoltzmann::Wall{ 
                          LatticeBoltzmann::BoundaryType::Periodic, }
                    : LatticeBoltzmann::Wall{ 
                          LatticeBoltzmann::BoundaryType::BounceBack, 0, 0, },
                // If there is no down neighbor, the bottom wall is a
                // bounce-back
                (node.down != MPI_PROC_NULL) ? LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic,}
                                             : LatticeBoltzmann::Wall{ LatticeBoltzmann::BoundaryType::BounceBack, 0, 0 ,},
                // If there is no left neighbor, the left wall is a
                // bounce-back (otherwise periodic)
                (node.left != MPI_PROC_NULL) ? LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic}
                                             : LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::BounceBack, 0, 0 ,},
                // If there is no up neighbor, the top wall is a
                // moving lid (otherwise periodic)
                (node.up != MPI_PROC_NULL)
                    ? LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic}
                    : LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::BounceBack, lid_vel_x, lid_vel_y ,},
            };

        LatticeBoltzmann::DistributionFunction distribution_function(
            "Distribution Function", node.lattice_width, node.lattice_height);

        LatticeBoltzmann::DistributionInitializers::random_density(
            distribution_function);
        LatticeBoltzmann::DistributionFunction buffer_distribution(
            "Buffer Distribution Function", node.lattice_width,
            node.lattice_height);
        LatticeBoltzmann::DensityFunction density_function(
            "Distribution Function", node.lattice_width, node.lattice_height);
        LatticeBoltzmann::VelocityProfile velocity_profile(
            "Distribution Function", node.lattice_width, node.lattice_height);

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

            node.communicate_all(distribution_function);

            LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(
                buffer_distribution, distribution_function, density_function,
                walls);
        }

        Kokkos::fence();

        auto elapsed = timer.seconds();
        auto actual_width = node.lattice_width - node.left - node.right;
        auto actual_height = node.lattice_height - node.down - node.up;
        auto mlups = (static_cast<double>(GLOBAL_DOMAIN_SIZE) *
                      GLOBAL_DOMAIN_SIZE * iterations) /
                     (elapsed * 1000000);

        std::cout << fmt::format("{:d}: {:d}x{:d} - {:d} iterations took "
                                 "{:f} seconds = {:f} MLUPS",
                                 node.rank, actual_width, actual_height,
                                 iterations, elapsed, mlups)
                  << std::endl;

        // // Make a view which excludes the ghost layers of the velocity
        // profile
        // // view
        // // TODO: Optimize. Only distribution function view needs ghost
        // layers. auto effective_velocity_profile = Kokkos::subview(
        //     velocity_profile,
        //     std::make_pair(node.left, node.lattice_width - node.right),
        //     std::make_pair(node.down, node.lattice_height - node.up),
        //     Kokkos::ALL);
        // auto velocity_profile_mirror =
        //     Kokkos::create_mirror_view(effective_velocity_profile);

        // Kokkos::View<double *[2], Kokkos::HostSpace> receive_buffer;

        // Kokkos::deep_copy(velocity_profile_mirror,
        // effective_velocity_profile);

        // if (node.rank == 0) {
        //     receive = LatticeBoltzmann::VelocityProfile("Receive Buffer")
        //         MPI_Gather(velocity_profile_mirror.data(), );
        // }
    }

    Kokkos::finalize();
    MPI_Finalize();
}
