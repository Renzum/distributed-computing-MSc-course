#pragma once

#include <memory>

#include <lattice_boltzmann_types.hpp>
#include <mpi_data.hpp>
#include <mpi_impl.hpp>

class MPIGPUAware : public MPIImplementation {
    int lattice_width, lattice_height;
    std::shared_ptr<MPIData> mpi_data;

    // Buffers for moving slices from GPU to CPU and vice versa
    MPIImplementation::GPUBuffer left_right_gpu_send_buffer;
    MPIImplementation::GPUBuffer left_right_gpu_recv_buffer;
    int left_right_buffer_len;

    MPIImplementation::GPUBuffer down_up_gpu_send_buffer;
    MPIImplementation::GPUBuffer down_up_gpu_recv_buffer;
    int down_up_buffer_len;

  public:
    MPIGPUAware(const int lattice_width, const int lattice_height,
                std::shared_ptr<MPIData> mpi_data);

    void communicate_left(const LatticeBoltzmann::DistributionFunction
                              &distribution_function) override;
    void communicate_right(const LatticeBoltzmann::DistributionFunction
                               &distribution_function) override;
    void communicate_down(const LatticeBoltzmann::DistributionFunction
                              &distribution_function) override;
    void communicate_up(const LatticeBoltzmann::DistributionFunction
                            &distribution_function) override;
};