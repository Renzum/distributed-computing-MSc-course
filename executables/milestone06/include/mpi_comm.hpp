#pragma once

#include <mpi.h>

#include <lattice_boltzmann_types.hpp>
#include <mpi_impl.hpp>

class MPICommunicator {
  public:
    IMPIImplementation mpi_implementation;

    struct GhostLayers {
        int left = 0, right = 0, down = 0, up = 0;
    };

    GhostLayers ghost_layers{};

    int mpi_domain_size;
    int mpi_rank;
    MPI_Comm cart{};

    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2];

    int up, down, left, right;

    int lattice_width, lattice_height;

    // Buffers for moving slices from GPU to CPU and vice versa
    using GPUBuffer = Kokkos::View<double *[TOTAL_DIRECTIONS]>;
    GPUBuffer left_right_send_gpu_buffer;
    GPUBuffer left_right_recv_gpu_buffer;
    int left_right_buffer_len;

    GPUBuffer up_down_send_gpu_buffer;
    GPUBuffer up_down_recv_gpu_buffer;
    int up_down_buffer_len;

    MPICommunicator(int domain_width, int domain_height);

    void communicate_over_mpi(
        LatticeBoltzmann::DistributionFunction &distribution_function);
};
