#pragma once

#include <mpi_data.hpp>
#include <mpi_impl.hpp>

class MPINotGPUAware : public MPIImplementation {
    int lattice_width, lattice_height;
    std::shared_ptr<MPIData> mpi_data;

    // Buffers for moving slices from GPU to CPU and vice versa
    MPIImplementation::GPUBuffer left_right_gpu_buffer;
    int left_right_buffer_len;

    MPIImplementation::GPUBuffer down_up_gpu_buffer;
    int down_up_buffer_len;

    // Host Space buffers for sending without GPU Aware MPI
    using GPUBufferMirror = GPUBuffer::HostMirror;
    GPUBufferMirror left_right_send_buffer, left_right_recv_buffer;
    GPUBufferMirror down_up_send_buffer, down_up_recv_buffer;

  public:
    MPINotGPUAware(const int lattice_width, const int lattice_height,
                   std::shared_ptr<MPIData> mpi_data);

    void
    communicate_left(const LatticeBoltzmann::DistributionFunction &) override;
    void communicate_right(const LatticeBoltzmann::DistributionFunction &

                           ) override;
    void
    communicate_down(const LatticeBoltzmann::DistributionFunction &) override;
    void
    communicate_up(const LatticeBoltzmann::DistributionFunction &) override;
};