#include <filesystem>
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

class DistributedVelocityProfileOutput {
    std::ofstream output_file;

    // Since we're writing each MPI node's velocity into a separate file, we
    // need to keep track of the x and y offsets of the local lattice in
    // relation to the global lattice according to the domain decomposition
    int x_offset, y_offset;
    int lattice_width, lattice_height;
    MPILayer::GhostLayers ghost_layers;

  public:
    DistributedVelocityProfileOutput(std::filesystem::path &output_directory,
                                     int domain_width, int domain_height,
                                     MPILayer &mpi_layer) {

        ghost_layers = mpi_layer.ghost_layers;

        lattice_width = mpi_layer.lattice_width;
        lattice_height = mpi_layer.lattice_height;

        auto file_path = output_directory.concat(fmt::format(
            "/velocity_profile_{:d}.data", mpi_layer.mpi_data->rank));

        output_file = std::ofstream(file_path, std::ios::out | std::ios::trunc);

        x_offset = mpi_layer.mpi_data->coords[1] *
                   static_cast<int>(domain_width / mpi_layer.mpi_data->dims[1]);
        if (mpi_layer.mpi_data->coords[1] > 0) {
            x_offset += (domain_width % mpi_layer.mpi_data->dims[1]);
        }

        int inv_y_coord =
            mpi_layer.mpi_data->dims[0] - mpi_layer.mpi_data->coords[0] - 1;
        y_offset = inv_y_coord * static_cast<int>(domain_height /
                                                  mpi_layer.mpi_data->dims[0]);
    }

    void write_velocity_profile(
        const LatticeBoltzmann::VelocityProfile::HostMirror &velocity_profile,
        int iteration) {
        output_file << "--- i:" << iteration << "\n";

        double x_vel = 0, y_vel = 0;

        for (int x = ghost_layers.left; x < lattice_width - ghost_layers.right;
             x++) {
            for (int y = ghost_layers.down;
                 y < lattice_height - ghost_layers.up; y++) {
                x_vel = velocity_profile(x, y, 0);
                y_vel = velocity_profile(x, y, 1);

                output_file << fmt::format(
                    "{:d},{:d},{:f},{:f}\n", x + x_offset - ghost_layers.left,
                    y + y_offset - ghost_layers.down, x_vel, y_vel);
            }
        }

        output_file << std::flush;
    }
};

void get_cmd_args(int argc, char *argv[], int &domain_width, int &domain_height,
                  int &iterations, std::filesystem::path &output_path) {
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--width", 7) == 0) {
            domain_width = atoi(argv[++i]);
        }

        if (strncmp(argv[i], "--height", 8) == 0) {
            domain_height = atoi(argv[++i]);
        }

        if (strncmp(argv[i], "--iterations", 12) == 0) {
            iterations = atoi(argv[++i]);
        }

        if (strncmp(argv[i], "--output-path", 13) == 0) {
            output_path = std::filesystem::canonical(argv[++i]);
        }
    }
}

int main(int argc, char *argv[]) {
    int domain_width = 0, domain_height = 0, iterations = 0;

    std::filesystem::path output_directory = std::filesystem::canonical("./");

    get_cmd_args(argc, argv, domain_width, domain_height, iterations,
                 output_directory);

    if (domain_width == 0 || domain_height == 0 || iterations == 0) {
        std::cout << "Please provide the --width and --height of the domain "
                     "and the number of --iterations "
                     "via command line arguments."
                  << std::endl;
        std::exit(1);
    }

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    {
        MPILayer mpi_layer(domain_width, domain_height);
        LatticeBoltzmann::VelocityProfile final_velocity_profile;
        DistributedVelocityProfileOutput velocity_profile_output(
            output_directory, domain_width, domain_height, mpi_layer);

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
                             mpi_layer.lattice_height, mpi_layer.lattice_width)
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
                "Distribution Function", mpi_layer.lattice_width,
                mpi_layer.lattice_height);

            LatticeBoltzmann::DistributionInitializers::random_density(
                distribution_function);
            LatticeBoltzmann::DistributionFunction buffer_distribution(
                "Buffer Distribution Function", mpi_layer.lattice_width,
                mpi_layer.lattice_height);
            LatticeBoltzmann::DensityFunction density_function(
                "Distribution Function", mpi_layer.lattice_width,
                mpi_layer.lattice_height);
            LatticeBoltzmann::VelocityProfile velocity_profile(
                "Distribution Function", mpi_layer.lattice_width,
                mpi_layer.lattice_height);

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

            final_velocity_profile = std::move(velocity_profile);
        }

        LatticeBoltzmann::VelocityProfile::HostMirror final_velocity_mirror =
            Kokkos::create_mirror_view(final_velocity_profile);

        Kokkos::deep_copy(final_velocity_mirror, final_velocity_profile);

        velocity_profile_output.write_velocity_profile(final_velocity_mirror,
                                                       0);
    }

    Kokkos::finalize();
    MPI_Finalize();
}
