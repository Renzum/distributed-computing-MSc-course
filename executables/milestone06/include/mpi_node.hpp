#pragma once

#include <mpi.h>

#include <lattice_boltzmann_types.hpp>

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
    GPUBuffer left_right_gpu_buffer;
    GPUBuffer up_down_gpu_buffer;

    // CPU Space Buffers for contagious data sending via MPI
    using SendRecvBuffer = GPUBuffer::HostMirror;
    SendRecvBuffer left_right_send_buffer;
    SendRecvBuffer left_right_recv_buffer;
    int left_right_buffer_len;

    SendRecvBuffer up_down_send_buffer;
    SendRecvBuffer up_down_recv_buffer;
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