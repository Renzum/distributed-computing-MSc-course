#include <fstream>
#include <string>

#include <Kokkos_Core.hpp>
#include <fmt/format.h>

#include <lattice_boltzmann_types.hpp>
#include <mpi_layer.hpp>

class DistributedVelocityOutput {
  private:
    std::ofstream file;

    int global_lattice_offset_x, global_lattice_offset_y;
    int actual_lattice_x_start, actual_lattice_x_end;
    int actual_lattice_y_start, actual_lattice_y_end;

  public:
    DistributedVelocityOutput(std::string file_name, MPILayer mpi_layer) {
        const std::string formatted_file_name =
            fmt::format("{:d}_{:s}", mpi_layer.mpi_data->rank, file_name);
        file = std::ofstream(formatted_file_name,
                             std::ios::out | std::ios::binary);

        global_lattice_offset_x = mpi_layer.global_lattice_offset_x;
        global_lattice_offset_y = mpi_layer.global_lattice_offset_y;

        std::cout << fmt::format(
                         "{:d}: Global lattice offset [x, y] is [{:d}, {:d}]",
                         mpi_layer.mpi_data->rank, global_lattice_offset_x,
                         global_lattice_offset_y)
                  << std::endl;

        actual_lattice_x_start = mpi_layer.ghost_layers.left;
        actual_lattice_x_end =
            mpi_layer.local_lattice_width - mpi_layer.ghost_layers.right;

        actual_lattice_y_start = mpi_layer.ghost_layers.down;
        actual_lattice_y_end =
            mpi_layer.local_lattice_height - mpi_layer.ghost_layers.up;

        std::cout << fmt::format(
                         "{:d}: The velocity profile subview "
                         "dimensions [x, y] are [{:d}:{:d}, {:d}:{:d}]",
                         mpi_layer.mpi_data->rank, actual_lattice_x_start,
                         actual_lattice_x_end - 1, actual_lattice_y_start,
                         actual_lattice_y_end - 1)
                  << std::endl;
    }

    void
    print_velocity_profile(LatticeBoltzmann::VelocityProfile velocity_profile,
                           int iteration) {
        auto velocity_subview = Kokkos::subview(
            velocity_profile,
            std::make_pair(actual_lattice_x_start, actual_lattice_x_end),
            std::make_pair(actual_lattice_y_start, actual_lattice_y_end),
            Kokkos::ALL);

        auto actual_velocity_host_mirror =
            Kokkos::create_mirror_view(velocity_subview);

        std::cout << fmt::format("Subview dimensions [{:d}, {:d}]",
                                 actual_velocity_host_mirror.extent_int(0),
                                 actual_velocity_host_mirror.extent_int(1))
                  << std::endl;

        Kokkos::deep_copy(actual_velocity_host_mirror, velocity_subview);

        file << "--- i:" << iteration;

        int width = actual_velocity_host_mirror.extent_int(0);
        int height = actual_velocity_host_mirror.extent_int(1);

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                file << fmt::format("\n{:d},{:d},{:f},{:f}",
                                    x + global_lattice_offset_x,
                                    y + global_lattice_offset_y,
                                    actual_velocity_host_mirror(x, y, 0),
                                    actual_velocity_host_mirror(x, y, 1));
            }
        }

        file << std::flush;
    }

    ~DistributedVelocityOutput() {
        file.close();
    }
};