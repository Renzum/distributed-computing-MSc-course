#pragma once

#include <mpi.h>

#include <lattice_boltzmann_types.hpp>

// class IMPIComm {
//   public:
//     int lattice_width;
//     int lattice_height;

//     IMPIComm(int width_width_ghost_layers, int height_with_ghost_layers)
//         : lattice_width(lattice_width),
//           lattice_height(lattice_height) {};

//     virtual void communicate_left(
//         LatticeBoltzmann::DistributionFunction &distribution_function);
//     virtual void communicate_right(
//         LatticeBoltzmann::DistributionFunction &distribution_function);
//     virtual void communicate_down(
//         LatticeBoltzmann::DistributionFunction &distribution_function);
//     virtual void communicate_up(
//         LatticeBoltzmann::DistributionFunction &distribution_function);
//     virtual void communicate_all(
//         LatticeBoltzmann::DistributionFunction &distribution_function);
// };

namespace GPUAware {

class Node {
  private:
  public:
    struct GhostLayers {
        int left = 0, right = 0, down = 0, up = 0;
    };

    GhostLayers ghost_layers{};

    MPI_Comm cart{};

    int dims[2] = {0, 0};
    int periods[2] = {0, 0};

    int coords[2];

    int up, down, left, right;

    // Buffers for moving slices from GPU to CPU and vice versa
    using GPUBuffer = Kokkos::View<double *[TOTAL_DIRECTIONS]>;
    GPUBuffer left_right_send_gpu_buffer;
    GPUBuffer left_right_recv_gpu_buffer;
    int left_right_buffer_len;

    GPUBuffer up_down_send_gpu_buffer;
    GPUBuffer up_down_recv_gpu_buffer;
    int up_down_buffer_len;

    int lattice_width, lattice_height;

    int rank, mpi_domain_size;

    Node(int global_domain_size);

    void communicate_left(
        LatticeBoltzmann::DistributionFunction &distribution_function);
    void communicate_right(
        LatticeBoltzmann::DistributionFunction &distribution_function);
    void communicate_down(
        LatticeBoltzmann::DistributionFunction &distribution_function);
    void communicate_up(
        LatticeBoltzmann::DistributionFunction &distribution_function);
    void communicate_all(
        LatticeBoltzmann::DistributionFunction &distribution_function);
};

} // namespace GPUAware